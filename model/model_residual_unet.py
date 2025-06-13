# model_residual_unet.py
import torch
import torch.nn as nn
import time
from residual_feature_block import ResidualFeatureBlock
from activations import get_activation
from loss_vgg import PerceptualLoss, charbonnier_loss # Assuming these are still used for training setup
import argparse

class ResidualUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 output_channels=3,
                 base_channels=36,
                 unet_depth=2, # Represents the number of downsampling/upsampling stages
                 blocks_per_level=2,
                 activation1='identity',
                 activation1_params=None,
                 activation2='relu',
                 activation2_params=None,
                 activation3='relu',
                 activation3_params=None,
                 activation4='identity',
                 activation4_params=None,
                 internal_block_channels_ratio=1.0, # Controls mid_channels in ResidualFeatureBlock
                 verbose=False):
        super().__init__()
        self.unet_depth = unet_depth
        self.verbose = verbose
        self.internal_block_channels_ratio = internal_block_channels_ratio
        self.output_channels = output_channels # Store for final layer calculation

        self.perceptual_criterion = PerceptualLoss(
                                        pixel_loss_weight=0.990,
                                        vgg_weight=0.007,
                                        pixel_loss_type='charbonnier',
                                        high_frequency_weight=0.003,
                                        high_frequency_type='laplacian',
                                        lambda_lum=0.0,
                                        input_is_linear=True
                                    )

        act_config = {
            'act1': activation1,
            'act1_params': activation1_params,
            'act2': activation2,
            'act2_params': activation2_params,
            'act3': activation3,
            'act3_params': activation3_params,
            'act4': activation4,
            'act4_params': activation4_params,
        }

        # --- Encoder Path ---
        # Encoder downsampling layers (PixelUnshuffle) and block sequences
        self.encoder_downs = nn.ModuleList()
        self.encoder_block_sequences = nn.ModuleList()

        current_channels = input_channels # Start with input_channels
        for d in range(unet_depth):
            # Each encoder stage starts with a PixelUnshuffle
            self.encoder_downs.append(nn.PixelUnshuffle(2))
            
            # Channels after downsampling for the current stage's blocks
            # Input to the first block of this level will be current_channels * 4
            in_ch_for_blocks = current_channels * 4

            # Output channels for blocks at this encoder level
            # These are also the channels of the skip connection for the next decoder level
            out_ch_for_blocks = base_channels * (2 ** d)

            # Calculate mid_channels for the ResidualFeatureBlocks at this level
            mid_ch = max(1, int(out_ch_for_blocks * self.internal_block_channels_ratio))

            level_blocks = []
            for i in range(blocks_per_level):
                # The first block takes `in_ch_for_blocks`. Subsequent blocks take `out_ch_for_blocks`.
                current_block_in_ch = in_ch_for_blocks if i == 0 else out_ch_for_blocks
                level_blocks.append(
                    ResidualFeatureBlock(current_block_in_ch, mid_ch, out_ch_for_blocks, 3, acts=act_config)
                )
            self.encoder_block_sequences.append(nn.Sequential(*level_blocks))
            
            # Update current_channels for the next encoder stage's input
            current_channels = out_ch_for_blocks

        # --- Bottleneck ---
        # Bottleneck input/output channels are the current_channels after the last encoder stage
        bottleneck_ch = current_channels
        self.bottleneck = nn.Sequential(*[
            ResidualFeatureBlock(
                bottleneck_ch,
                max(1, int(bottleneck_ch * self.internal_block_channels_ratio)),
                bottleneck_ch,
                3, acts=act_config
            ) for _ in range(blocks_per_level)
        ])

        # --- Decoder Path ---
        # Decoder upsampling layers (PixelShuffle) and block sequences
        self.decoder_ups = nn.ModuleList()
        self.decoder_block_sequences = nn.ModuleList()

        # Start with channels from the bottleneck for the first decoder stage
        current_channels = bottleneck_ch 

        for d in reversed(range(unet_depth)):
            # Each decoder stage starts with a PixelShuffle
            self.decoder_ups.append(nn.PixelShuffle(2))

            # Channels after upsampling for the current stage's blocks
            upsampled_ch = current_channels // 4

            # Corrected skip connection channels:
            # For d=0, skip is the original input_channels.
            # For d > 0, skip is the output of encoder_block_sequences[d-1],
            # which has base_channels * (2**(d-1)) channels.
            if d == 0:
                skip_ch = input_channels
            else:
                skip_ch = base_channels * (2**(d-1))
            
            # Input channels for the first block of this decoder level: concatenated upsampled features and skip
            in_ch_for_blocks = upsampled_ch + skip_ch

            level_blocks = []
            if d == 0: # For the final decoding stage, use a single Conv2d to map to output_channels
                level_blocks.append(
                    nn.Conv2d(in_channels=in_ch_for_blocks, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=True)
                )
                # Update current_channels for the next (non-existent) stage, or for direct output
                current_channels = self.output_channels
            else:
                # Output channels for blocks at this decoder level (for stages other than the final one)
                out_ch_for_blocks = base_channels * (2**d)
                # Calculate mid_channels for the ResidualFeatureBlocks at this level
                mid_ch = max(1, int(out_ch_for_blocks * self.internal_block_channels_ratio))

                for i in range(blocks_per_level):
                    # The first block takes `in_ch_for_blocks`. Subsequent blocks take `out_ch_for_blocks`.
                    current_block_in_ch = in_ch_for_blocks if i == 0 else out_ch_for_blocks
                    level_blocks.append(
                        ResidualFeatureBlock(current_block_in_ch, mid_ch, out_ch_for_blocks, 3, acts=act_config)
                    )
                # Update current_channels for the next decoder stage's input
                current_channels = out_ch_for_blocks

            self.decoder_block_sequences.append(nn.Sequential(*level_blocks))
            
        # --- Final Layer ---
        # The final ReLU is applied in the forward pass.

    def forward(self, x):
        skips = [] # To store feature maps for skip connections

        # --- Encoder Path ---
        # The 'skips' are saved BEFORE the downsampling at each stage.
        # So, skips[0] will be the raw input (highest resolution).
        # skips[d] will be the output of encoder_block_sequences[d-1]
        
        current_x = x # Start with the raw input tensor
        for d in range(self.unet_depth):
            skips.append(current_x) # Save for skip connection at this resolution

            current_x = self.encoder_downs[d](current_x) # Apply PixelUnshuffle
            current_x = self.encoder_block_sequences[d](current_x) # Apply ResidualFeatureBlocks

        # --- Bottleneck ---
        current_x = self.bottleneck(current_x)

        # --- Decoder Path ---
        # Iterate in reverse order of encoder (from deepest to highest resolution)
        for i, dec_up in enumerate(self.decoder_ups):
            # Pop the corresponding skip connection (skips are in increasing order of depth)
            # The decoder_ups are appended in reversed order, so decoder_ups[0] is for deepest level.
            # We need to get skip for the level corresponding to this upsample operation.
            # (self.unet_depth - 1 - i) gives the 'd' for the current decoder level being processed.
            skip_idx = self.unet_depth - 1 - i
            skip = skips[skip_idx] # Retrieve the skip from the correct encoder level

            current_x = dec_up(current_x) # Apply PixelShuffle

            # Pad if needed (PixelShuffle should ideally match skip dimensions if resolutions are powers of 2)
            if current_x.shape[2:] != skip.shape[2:]:
                diffY = skip.size(2) - current_x.size(2)
                diffX = skip.size(3) - current_x.size(3)
                current_x = nn.functional.pad(current_x, [diffX // 2, diffX - diffX // 2,
                                                          diffY // 2, diffY - diffY // 2])
            
            current_x = torch.cat([current_x, skip], dim=1) # Concatenate with skip connection
            current_x = self.decoder_block_sequences[i](current_x) # Apply decoder blocks
        
        # --- Final Layer ---
        # The output of the last decoder block is now at the correct channel count.
        # Apply final ReLU as specified.
        current_x = nn.functional.relu(current_x)
        return current_x

    # Criterion used by the training loop (Decided on perceptual loss)
    def criterion(self, output, target):
        return charbonnier_loss(output, target)

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
        param_size_mb = param_count * 2 / (1024 ** 2)  # Assuming fp16 = 2 bytes

        return {
            'fps': fps,
            'params': param_count,
            'size_mb': param_size_mb,
            'output_shape': self(input_tensor).shape
        }

def get_model(name:str='lightweight'):
    if name == 'lightweight':
        return ResidualUNet(unet_depth=2, blocks_per_level=2, base_channels=24, internal_block_channels_ratio=1.50)
    elif name == 'heavyweight':
        return ResidualUNet(unet_depth=4, blocks_per_level=2, base_channels=48, internal_block_channels_ratio=1.50)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test performance')
    parser.add_argument('--model_type', type=str, required=True, choices=['lightweight', 'heavyweight'], help='Type of model: lightweight, heavyweight')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_type).to(device).half().eval()

    print("Attempting to compile model...")
    try:
        model = torch.compile(model, mode="default", fullgraph=True)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Falling back to eager mode.")
        model = model.to(device)

    # Note: Dummy input resolution 576x752 might require adjustments if it's not perfectly divisible
    # by (2**unet_depth) which is 4 for lightweight, 16 for heavyweight, at each downsample step.
    dummy_input = torch.rand((1, 3, 576, 752), dtype=torch.float16).to(device)
    results = model.benchmark(dummy_input)

    print("\n--- Results ---")
    print(f"Model output shape: {results['output_shape']}")
    print(f"Model size (trainable parameters): {results['params']}")
    print(f"Model size (MB, assuming float16): {results['size_mb']:.2f} MB")
    print(f"Average FPS: {results['fps']:.2f}")
    print("---------------")