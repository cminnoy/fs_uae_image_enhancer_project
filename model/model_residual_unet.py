# model_residual_unet.py
import torch
import torch.nn as nn
import time
from residual_feature_block import ResidualFeatureBlock
from activations import get_activation
from loss_vgg import PerceptualLoss, charbonnier_loss
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
        self.encoder_downs = nn.ModuleList() # Stores PixelUnshuffle(2) for d=1 to unet_depth-1
        self.encoder_block_sequences = nn.ModuleList()

        # Step 1: Initial PixelUnshuffle(2) and Conv2d
        self.first_pixel_unshuffle_and_conv = nn.Sequential(
            nn.PixelUnshuffle(2), # Input: input_channels -> input_channels * 4 (e.g., 3 -> 12)
            nn.Conv2d(input_channels * 4, base_channels, kernel_size=1, stride=1, padding=0, bias=True) # 12 -> base_channels
        )
        
        # Step 2: First set of ResidualFeatureBlocks (for d=0 encoder level)
        # Input to this block sequence is base_channels (from above Conv2d)
        in_ch_d0_blocks = base_channels
        out_ch_d0_blocks = base_channels # Output channels for d=0 level blocks
        mid_ch_d0 = max(1, int(out_ch_d0_blocks * self.internal_block_channels_ratio))
        level_blocks_d0 = []
        for i in range(blocks_per_level):
            current_block_in_ch = in_ch_d0_blocks if i == 0 else out_ch_d0_blocks
            level_blocks_d0.append(
                ResidualFeatureBlock(current_block_in_ch, mid_ch_d0, out_ch_d0_blocks, 3, acts=act_config)
            )
        self.encoder_block_sequences.append(nn.Sequential(*level_blocks_d0)) # Index 0 for d=0

        current_channels = out_ch_d0_blocks # Channels for the input to the next encoder stage (d=1)

        # --- Subsequent Encoder Stages (d=1 to unet_depth-1) ---
        for d in range(1, unet_depth): # Loop starts from d=1
            self.encoder_downs.append(nn.PixelUnshuffle(2)) # Store PixelUnshuffle for this stage
            
            in_ch_for_blocks = current_channels * 4 # Channels after downsampling by self.encoder_downs[d-1]
            out_ch_for_blocks = base_channels * (2 ** d) # Why this size? Why not just use in_ch_for_blocks?  
            mid_ch = max(1, int(out_ch_for_blocks * self.internal_block_channels_ratio))

            level_blocks = []
            for i in range(blocks_per_level):
                current_block_in_ch = in_ch_for_blocks if i == 0 else out_ch_for_blocks
                level_blocks.append(
                    ResidualFeatureBlock(current_block_in_ch, mid_ch, out_ch_for_blocks, 3, acts=act_config)
                )
            self.encoder_block_sequences.append(nn.Sequential(*level_blocks)) # Index d for this stage
            
            current_channels = out_ch_for_blocks # Update for next iteration

        # --- Bottleneck ---
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
        self.decoder_ups = nn.ModuleList()
        self.decoder_block_sequences = nn.ModuleList()

        current_channels = bottleneck_ch 

        for d in reversed(range(unet_depth)):
            self.decoder_ups.append(nn.PixelShuffle(2))

            upsampled_ch = current_channels // 4
            
            # Determine skip connection channels for this decoder stage
            if d == 0:
                skip_ch = input_channels # For the highest resolution stage, skip is the original input
            else:
                # For other stages, skip comes from the output of encoder_block_sequences[d-1]
                # which has `base_channels * (2**(d-1))` channels from our new encoder structure
                skip_ch = base_channels * (2**(d-1))
            
            in_ch_for_blocks = upsampled_ch + skip_ch

            level_blocks = []
            if d == 0: # For the final decoding stage, use a single Conv2d to map to output_channels
                level_blocks.append(
                    nn.Conv2d(in_channels=in_ch_for_blocks, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=True)
                )
                current_channels = self.output_channels
            else:
                out_ch_for_blocks = base_channels * (2**d)
                mid_ch = max(1, int(out_ch_for_blocks * self.internal_block_channels_ratio))

                for i in range(blocks_per_level):
                    current_block_in_ch = in_ch_for_blocks if i == 0 else out_ch_for_blocks
                    level_blocks.append(
                        ResidualFeatureBlock(current_block_in_ch, mid_ch, out_ch_for_blocks, 3, acts=act_config)
                    )
                current_channels = out_ch_for_blocks

            self.decoder_block_sequences.append(nn.Sequential(*level_blocks))
            
    def forward(self, x):
        original_input_x = x # (1, 3, 576, 736)
        encoder_features = [] # Store features for skip connections (from encoder blocks)

        if self.verbose:
            print(f"--- Forward Pass Start ---")
            print(f"Initial input x shape: {x.shape}")

        # Initial stage encoder
        x = self.first_pixel_unshuffle_and_conv(x) # (1, 24, 288, 368)
        if self.verbose:
            print(f"After first_pixel_unshuffle_and_conv: {x.shape}")
        x = self.encoder_block_sequences[0](x) # (1, 24, 288, 368)
        if self.verbose:
            print(f"After encoder_block_sequences[0]: {x.shape}")
        encoder_features.append(x) # encoder_features[0] = output of encoder_block_sequences[0]

        # Subsequent encoder stages
        for d in range(1, self.unet_depth):
            if self.verbose:
                print(f"\n--- Encoder Stage d={d} ---")
                print(f"Before encoder_downs[{d-1}]: {x.shape}")
            x = self.encoder_downs[d-1](x) # Downsample
            if self.verbose:
                print(f"After encoder_downs[{d-1}]: {x.shape}")
            x = self.encoder_block_sequences[d](x) # Apply blocks
            if self.verbose:
                print(f"After encoder_block_sequences[{d}]: {x.shape}")
            encoder_features.append(x) # encoder_features[d] = output of encoder_block_sequences[d]

        if self.verbose:
            print(f"\n--- Bottleneck ---")
            print(f"Before bottleneck: {x.shape}")
        x = self.bottleneck(x)
        if self.verbose:
            print(f"After bottleneck: {x.shape}")

        # Decoder Path
        # Iterate over decoder modules, using a separate `d_val` to track the logical depth
        for i in range(self.unet_depth): # i goes from 0 to unet_depth-1
            d_val = self.unet_depth - 1 - i # This computes the encoder depth corresponding to this decoder stage

            dec_up = self.decoder_ups[i] # Current upsample module (indexed 0 for deepest, 1 for shallowest)
            dec_block_seq = self.decoder_block_sequences[i] # Current decoder block sequence (indexed 0 for deepest, 1 for shallowest)

            if self.verbose:
                print(f"\n--- Decoder Stage i={i} (d_val={d_val}) ---")
                print(f"Input to current decoder stage (x, from prev stage): {x.shape}")

            current_x_upsampled = dec_up(x)
            if self.verbose:
                print(f"After decoder_ups[{i}]: {current_x_upsampled.shape}")
            
            skip = None
            if d_val == 0: # This is the highest resolution decoder stage (i=unet_depth-1)
                skip = original_input_x # (1, 3, 576, 736)
            else: # For other decoder stages (d_val > 0)
                # The skip comes from encoder_features at index d_val-1
                skip_idx = d_val - 1
                skip = encoder_features[skip_idx]
            
            if self.verbose:
                print(f"Skip for stage d_val={d_val} (encoder_features[{skip_idx}] or original_input_x) shape: {skip.shape}")
                
                # Dynamically determine the expected input channels
                expected_in_channels = 0
                if d_val == 0: # Final stage, Conv2d
                    expected_in_channels = dec_block_seq[0].in_channels
                else: # ResidualFeatureBlock
                    expected_in_channels = dec_block_seq[0].conv1.in_channels
                print(f"Expected input channels for decoder_block_sequences[{i}]: {expected_in_channels}")
            
            # Check if spatial dimensions match before concatenation, and pad if necessary
            if current_x_upsampled.shape[2:] != skip.shape[2:]:
                if self.verbose:
                    print(f"Padding needed: upsampled {current_x_upsampled.shape[2:]} vs skip {skip.shape[2:]}")
                diffY = skip.size(2) - current_x_upsampled.size(2)
                diffX = skip.size(3) - current_x_upsampled.size(3)
                current_x_upsampled = nn.functional.pad(current_x_upsampled, [diffX // 2, diffX - diffX // 2,
                                                                              diffY // 2, diffY - diffY // 2])
                if self.verbose:
                    print(f"After padding current_x_upsampled: {current_x_upsampled.shape}")
            
            x = torch.cat([current_x_upsampled, skip], dim=1) # Concatenate features
            if self.verbose:
                print(f"After concatenation, input to decoder_block_sequences[{i}]: {x.shape}")
            
            x = dec_block_seq(x) # Apply decoder blocks
            if self.verbose:
                print(f"After decoder_block_sequences[{i}]: {x.shape}")
        
        x = nn.functional.relu(x) # Apply final ReLU
        if self.verbose:
            print(f"Final output shape: {x.shape}")
            print(f"--- Forward Pass End ---")
        return x

    # Criterion used by the training loop
    def criterion(self, output, target):
        return self.perceptual_criterion(output, target)

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

def get_model(name:str='lightweight', verbose:bool=False):
    if name == 'lightweight':
        return ResidualUNet(unet_depth=3, blocks_per_level=1, base_channels=36, internal_block_channels_ratio=1.50, verbose=verbose)
    elif name == 'heavyweight':
        return ResidualUNet(unet_depth=4, blocks_per_level=4, base_channels=72, internal_block_channels_ratio=1.50, verbose=verbose)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test performance')
    parser.add_argument('--model_type', type=str, required=True, choices=['lightweight', 'heavyweight'], help='Type of model: lightweight, heavyweight')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmarking')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile for debugging.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose printing for debugging.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_type, verbose=args.verbose).to(device).half().eval()

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
