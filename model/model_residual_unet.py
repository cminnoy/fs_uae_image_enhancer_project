# model_residual_unet.py
import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from residual_feature_block import ResidualFeatureBlock
from activations import get_activation
from loss_vgg import PerceptualLoss, charbonnier_loss

# Residual U-Net with Multi-Path Residual Feature Blocks
#
# This model is designed specifically for image enhancement of the FS-UAE Amiga emulator.
# FS-UAE uses a single frame buffer with 32-bit RGBA format, where each color channel is in sRGB space.
# The model is trained on linear space images, so input images are converted from sRGB to linear space.
# An sRGB to linear conversion and normalisation is done after the model is trained by the ONNX export script.
# 
# The FS-UAE framebuffer has a fixed size of 752x576 pixels.
# Amiga lores mode uses 4 raw pixels for every displayed pixel, in a 2x2 grid, resulting in a 376x288 effective resolution.
# Amiga hires mode uses 2 raw pixels for every displayed pixel, in a 1x2 grid, resulting in a 752x288 effective resolution.
# Amiga lores interlace mode uses 2 raw pixels for every displayed pixel, in a 2x1 grid, resulting in a 376x576 effective resolution.
# Amiga hires interlace mode uses 1 raw pixel for every displayed pixel, in a 1x1 grid, resulting in a 752x576 effective resolution.
# 
# Amiga superresolution modes (e.g., Super HiRes) are not supported by FS-UAE and thus not considered here.
#
# The model uses a Residual U-Net architecture with multi-path residual feature blocks.
# The input is processed through a head module that combines pixel unshuffling, pooling, and
# a new convolutional path to extract rich features before entering the U-Net.
# The U-Net consists of an encoder-decoder structure with skip connections and a bottleneck.
# The output is refined and combined with a global skip connection from the input.
# The model is trained using a perceptual loss that combines pixel-wise loss,
# VGG-based perceptual loss, and high-frequency loss.
# 
# Amiga games are mostly handdrawn pixel art with sharp edges and limited color palettes.
# Often artists used dithering to simulate more colors and gradients, but not always in a consistent way.
# The model is trained on a dataset of high resolution images in full colour with there corresponding lores counterparts.
# 
# Dithering patterns applied to lores images in the training dataset include:
# - Floyd-Steinberg dithering
# - Bayer ordered dithering '2x2', '4x4', '8x8'
# - Atkinson dithering
# - Sierra dithering '2', '3'
# - Stucki dithering
# - Burkes dithering
# - Checkerboard dithering 
# - No dithering
#
# Colour modes in the training dataset include:
# - 16 color palette
# - 24 color palette 
# - 32 color palette
# - 64 color paletted extra half-brite
# - 128 color palette
# - 256 color palette
# - 512 color palette
# - HAM6 for Amiga lores and lores interlace
# - SHAM (Split HAM6) for Amiga lores and lores interlace
# - DynamicHires (16 color palette per scanline) for Amiga hires and hires interlace

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class kPathResidualFeatureBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 k_paths=3, with_skip_connection=True,
                 use_concatenation=True, use_attention=True):
        super().__init__()

        self.k_paths = k_paths
        self.with_skip_connection = with_skip_connection
        self.use_concatenation = use_concatenation
        self.use_attention = use_attention

        # Build multi-path feature extraction blocks
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=False)
            )
            for _ in range(k_paths)
        ])

        # Combine features from paths
        combined_channels = out_channels * k_paths if use_concatenation else out_channels

        self.merge_conv = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=False)
        )

        # Optional attention
        if use_attention:
            self.attention = SqueezeExcite(out_channels)
    
        # Optional skip projection (if dimensions differ)
        self.skip_proj = None
        if with_skip_connection and in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Parallel paths
        path_outputs = [path(x) for path in self.paths]

        # Aggregate
        if self.use_concatenation:
            combined = torch.cat(path_outputs, dim=1)
        else:
            combined = sum(path_outputs) / self.k_paths

        # Merge & attention
        fused = self.merge_conv(combined)
        if self.use_attention:
            fused = self.attention(fused)

        # Residual
        if self.with_skip_connection:
            skip = self.skip_proj(x) if self.skip_proj else x
            fused = fused + skip
            fused = F.elu(fused, inplace=False)

        return fused

class HeadProcessing(nn.Module):
    def __init__(self, input_channels, base_channels, k_paths):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.min_pool = nn.MaxPool2d(2, stride=2)
        
        # New convolution layer
        self.conv_2x2_path = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels, # Adjust if needed
            kernel_size=2,
            stride=2
        )
        
        # We've added a new path, so we need to adjust the `in_channels` for `conv_expand`
        # `input_channels * 4` (unshuffle) + `input_channels` (conv_2x2_path)
        # However, the input to conv_expand is just the unshuffled tensor, so its in_channels remains the same.
        # But we need to adjust the `out_channels` to make space for the new path in the final concatenation
        self.conv_expand = nn.Conv2d(
            in_channels=input_channels * 4,
            out_channels=base_channels - 3 * input_channels - input_channels, # Adjusted output channels
            kernel_size=1,
            padding=0,
            stride=1
        )
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True
            ),
            nn.ReLU(inplace=False),
            kPathResidualFeatureBlock(
                in_channels=base_channels,
                mid_channels=int(base_channels / 1.5),
                out_channels=base_channels,
                k_paths=k_paths,
                with_skip_connection=False,
                use_concatenation=True,
                use_attention=False
            )
        )

    def forward(self, x):
        unshuffled = self.pixel_unshuffle(x)   # shape: (B, C*4, H/2, W/2)
        unshuffled_expanded = self.conv_expand(unshuffled)
        
        avg_pool = self.avg_pool(x)            # shape: (B, C,   H/2, W/2)
        max_pool = self.max_pool(x)            # shape: (B, C,   H/2, W/2)
        min_pool = self.min_pool(-x)           # shape: (B, C,   H/2, W/2)
        contrast = max_pool - min_pool         # shape: (B, C,   H/2, W/2)
        
        conv_2x2 = self.conv_2x2_path(x)       # New path, shape: (B, C, H/2, W/2)
        
        # Concatenate all tensors
        concat = torch.cat([unshuffled_expanded, avg_pool, max_pool, contrast, conv_2x2], dim=1)
        
        return self.conv_stack(concat)

class ResidualUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3,
                 base_channels=48, unet_depth=3, blocks_per_level=2,
                 k_paths=3, internal_block_channels_ratio=1.25, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.unet_depth = unet_depth
        self.output_channels = output_channels

        # --- Criterion ---
        self.perceptual_criterion = PerceptualLoss(
                                        pixel_loss_weight=0.985,
                                        vgg_weight=0.010,
                                        pixel_loss_type='charbonnier',
                                        high_frequency_weight=0.005,
                                        high_frequency_type='laplacian',
                                        lambda_lum=0.0,
                                        input_is_linear=True
                                    )
    
        # --- Global Skip Connection ---
        self.global_skip = (
            nn.Identity()
                if input_channels == output_channels
                else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )
        self.skip_alpha = nn.Parameter(torch.tensor(1.0))
    
        # --- Head ---
        self.head = HeadProcessing(input_channels, base_channels, k_paths)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        in_ch = base_channels
        for d in range(unet_depth):
            out_ch = base_channels * (2 ** d)
            
            # Account for PixelUnshuffle expansion (x4) for levels d > 0
            block_in_ch = in_ch * 4 if d > 0 else in_ch

            blocks = [ResidualFeatureBlock(block_in_ch if i == 0 else out_ch,
                                           int(out_ch * internal_block_channels_ratio),
                                           out_ch, 3)
                      for i in range(blocks_per_level)]
            self.encoder_blocks.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.downs = nn.ModuleList([nn.PixelUnshuffle(2) for _ in range(unet_depth - 1)])

        # --- Bottleneck ---
        bottleneck_ch = base_channels * (2 ** (unet_depth - 1))
        self.bottleneck = nn.Sequential(
            kPathResidualFeatureBlock(bottleneck_ch, bottleneck_ch // 2, bottleneck_ch,
                                      k_paths=k_paths, with_skip_connection=True,
                                      use_concatenation=True, use_attention=True),
            ResidualFeatureBlock(bottleneck_ch, bottleneck_ch // 2, bottleneck_ch, 3),
            SqueezeExcite(bottleneck_ch)
        )

        # --- Decoder ---
        self.ups = nn.ModuleList([nn.PixelShuffle(2) for _ in range(unet_depth - 1)])
        self.decoder_blocks = nn.ModuleList()
        
        prev_out_ch = bottleneck_ch

        for d in reversed(range(unet_depth)):
            if d == 0:
                # Final Stage:
                # Logic Fix: This stage has NO Upsampling (PixelShuffle) because d=0.
                # So we simply take prev_out_ch directly.
                upsampled_ch = prev_out_ch # No division by 4
                skip_ch = base_channels
                in_ch = upsampled_ch + skip_ch
                
                self.decoder_blocks.append(
                    nn.Conv2d(in_ch, output_channels, 1)
                )
            else:
                out_ch = base_channels * (2 ** d)
                
                # Intermediate Stage (Upsampling happens):
                upsampled_ch = prev_out_ch // 4
                skip_ch = base_channels * (2 ** (d - 1))
                in_ch = upsampled_ch + skip_ch
                
                blocks = [ResidualFeatureBlock(in_ch if i == 0 else out_ch,
                                               int(out_ch * internal_block_channels_ratio),
                                               out_ch, 3)
                          for i in range(blocks_per_level)]
                self.decoder_blocks.append(nn.Sequential(*blocks))
                
                prev_out_ch = out_ch

        # --- Refinement ---
        self.refine = nn.Sequential(
                    # Expand channels x4 to prepare for PixelShuffle
                    nn.Conv2d(output_channels, output_channels * 4, 3, padding=1),
                    # Upsample: (B, 12, H/2, W/2) -> (B, 3, H, W)
                    nn.PixelShuffle(2)
                )

    def forward(self, x):
        x_in = x

        x_head = self.head(x)

        if self.verbose:
            print(f"[Head] {x_head.shape}")

        encoder_features = []
        
        # Level 0
        x = self.encoder_blocks[0](x_head)
        encoder_features.append(x)

        # Levels 1 to Depth-1
        for d in range(1, self.unet_depth):
            x = self.downs[d - 1](x)
            x = self.encoder_blocks[d](x)
            encoder_features.append(x)

        x = self.bottleneck(x)

        # Decoder path
        for i, block in enumerate(self.decoder_blocks):
            d_val = self.unet_depth - 1 - i
            
            if i < len(self.ups):
                x_up = self.ups[i](x)
            else:
                x_up = x 
            
            skip = x_head if d_val == 0 else encoder_features[d_val - 1]
            
            if x_up.shape[2:] != skip.shape[2:]:
                diffY = skip.size(2) - x_up.size(2)
                diffX = skip.size(3) - x_up.size(3)
                x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
            x = torch.cat([x_up, skip], dim=1)
            x = block(x)

        x = F.relu(x)
        x = self.refine(x)
        x = x + self.skip_alpha * self.global_skip(x_in)
        x = F.elu(x, inplace=False)
        return x
    
    # === Criterion ===
    def criterion(self, output, target):
       # return nn.L1Loss()(output, target)
        return self.perceptual_criterion(output, target)

    # === Benchmark ===
    def benchmark(self, input_tensor, warmup_iters=20, test_duration=20.0):
        self.eval()
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = self(input_tensor)

            start_time = time.time()
            iterations = 0
            while time.time() - start_time < test_duration:
                _ = self(input_tensor)
                iterations += 1

        elapsed = time.time() - start_time
        fps = iterations / elapsed
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_size_mb = param_count * 2 / (1024 ** 2)
        return {
            'fps': fps,
            'params': param_count,
            'size_mb': param_size_mb,
            'output_shape': self(input_tensor).shape
        }

def get_model(name: str = 'lightweight', verbose: bool = False):
    """
    Returns a selected model configuration.
    Supported: 'lightweight', 'heavyweight', 'advanced_light', 'advanced_heavy'
    """

    if name == 'light':
        return ResidualUNet(
            unet_depth=1,
            blocks_per_level=2,
            base_channels=36,
            k_paths=5,
            internal_block_channels_ratio=1.50,
            verbose=verbose
        )
    elif name == 'heavy':
        return ResidualUNet(
            unet_depth=3,
            blocks_per_level=3,
            base_channels=72,
            internal_block_channels_ratio=1.50,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test performance')
    parser.add_argument('--model_type', type=str, required=True, choices=['light', 'heavy'], help='Type of model: light, heavy')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmarking')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile for debugging.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose printing for debugging.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the model state_dict.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_type, verbose=args.verbose).to(device).half().eval()

    if args.save_model:
        print(f"Saving model state_dict to {args.save_model}")
        torch.save(model.state_dict(), args.save_model)
        print("Model saved successfully.")

    print("Attempting to compile model...")
    if not args.no_compile: # Conditional compilation
        try:
            model = torch.compile(model, mode="default", fullgraph=True)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Falling back to eager mode.")
            model = model.to(device)
    else:
        print("torch.compile disabled for debugging.")
        model = model.to(device) # Ensure model is on device even if not compiled

    dummy_input = torch.rand((args.batch_size, 3, 576, 736), dtype=torch.float16).to(device)
    results = model.benchmark(dummy_input)

    print("\n--- Results ---")
    print(f"Model output shape: {results['output_shape']}")
    print(f"Model size (trainable parameters): {results['params']}")
    print(f"Model size (MB, assuming float16): {results['size_mb']:.2f} MB")
    print(f"Average FPS: {results['fps']:.2f}")
    print("---------------")