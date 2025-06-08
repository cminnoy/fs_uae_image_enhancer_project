import torch
import torch.nn as nn
from loss_vgg import PerceptualLoss
import argparse, time, sys
import activations

# Import quantization module and function
try:
    from torch.ao.quantization import fuse_modules
    FUSION_UTILITY = fuse_modules
except ImportError:
    print("Warning: torch.ao.quantization module not found. Skipping PyTorch-side layer fusion.")
    FUSION_UTILITY = None
except AttributeError:
    print("Warning: torch.ao.quantization found, but fuse_modules is not in it. Skipping PyTorch-side layer fusion.")
    FUSION_UTILITY = None

# Conv2D 6 layer model
class Model(nn.Module):
    def __init__(self, layer1_out_channels = 36,
                       layer1_kernel_size = 3,
                       layer1_act1 = 'identity',
                       layer1_act2 = 'relu',
                       layer2_out_channels = 36,
                       layer2_kernel_size = 3,
                       layer2_act1 = 'mish',
                       layer2_act2 = 'biased_relu',
                       layer2_act3 = 'tanh',
                       layer2_act4 = 'relu6',
                       layer3_out_channels = 36,
                       layer3_kernel_size = 3,
                       layer3_act1 = 'identity',
                       layer3_act2 = 'identity',
                       layer4_out_channels = 36,
                       layer4_kernel_size = 3,
                       layer4_act1 = 'telu',
                       layer4_act2 = 'leaky_relu',
                       layer4_act3 = 'tanh',
                       layer4_act4 = 'identity',
                       layer5_out_channels = 36,
                       layer5_kernel_size = 3,
                       layer5_act1 = 'identity',
                       layer5_act2 = 'identity',
                       layer6_out_channels = 36,
                       layer6_kernel_size = 3,
                       layer6_act1 = 'mish',
                       layer6_act2 = 'prelu',
                       layer7_kernel_size = 3,
                       layer7_act1 = 'sinlu',
                       layer7_act2 = 'prelu',
                       verbose=False
                ):
        """
        Initializes the Model with a sequence of Conv2d layers followed by a combination of activation functions.

        Args:
            initial_out_channels (int): Number of output channels for the first convolution.
            mid_out_channels (int): Number of output channels for the third and fourth convolution.
            kernel_size (int): Size of the convolutional kernel (must be odd, e.g., 3 or 5).
        """
        super(Model, self).__init__()

        for ks in[layer1_kernel_size, layer2_kernel_size, layer3_kernel_size, layer4_kernel_size, layer5_kernel_size, layer6_kernel_size, layer7_kernel_size]: 
            if ks % 2 == 0:
                raise ValueError("kernel_size must be odd for symmetric padding")

        self.verbose = verbose
        if verbose:
            print(f"Model initialized with:\n"
                  f"Layer 1: {layer1_out_channels} channels, kernel size {layer1_kernel_size}, activations {layer1_act1}, {layer1_act2}\n"
                  f"Layer 2: {layer2_out_channels} channels, kernel size {layer2_kernel_size}, activations {layer2_act1}, {layer2_act2}, {layer2_act3}, {layer2_act4}\n"
                  f"Layer 3: {layer3_out_channels} channels, kernel size {layer3_kernel_size}, activations {layer3_act1}, {layer3_act2}\n"
                  f"Layer 4: {layer4_out_channels} channels, kernel size {layer4_kernel_size}, activations {layer4_act1}, {layer4_act2}, {layer4_act3}, {layer4_act4}\n"
                  f"Layer 5: {layer5_out_channels} channels, kernel size {layer5_kernel_size}, activations {layer5_act1}, {layer5_act2}\n"
                  f"Layer 6: {layer6_out_channels} channels, kernel size {layer6_kernel_size}, activations {layer6_act1}, {layer6_act2}\n"
                  f"Layer 7: kernel size {layer7_kernel_size}, activations {layer7_act1}, {layer7_act2}")

        # Define padding based on kernel size
        conv1_padding = (layer1_kernel_size - 1) // 2
        conv2_padding = (layer2_kernel_size - 1) // 2
        conv3_padding = (layer3_kernel_size - 1) // 2
        conv4_padding = (layer4_kernel_size - 1) // 2
        conv5_padding = (layer5_kernel_size - 1) // 2
        conv6_padding = (layer6_kernel_size - 1) // 2
        conv7_padding = (layer7_kernel_size - 1) // 2

        # Layer 0 
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3 * (2**2), out_channels=layer1_out_channels, kernel_size=layer1_kernel_size, stride=1, padding=conv1_padding, bias=True)
        self.l1_act1 = activations.get_activation(layer1_act1)
        self.l1_act2 = activations.get_activation(layer1_act2)

        # Skip connection projection for Layer 2 addition
        self.skip1_proj_conv = None
        if layer1_out_channels != layer2_out_channels:
            self.skip1_proj_conv = nn.Conv2d(in_channels=layer1_out_channels, out_channels=layer2_out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=layer1_out_channels, out_channels=layer2_out_channels, kernel_size=layer2_kernel_size, stride=1, padding=conv2_padding, bias=True)
        self.l2_act1 = activations.get_activation(layer2_act1)
        self.l2_act2 = activations.get_activation(layer2_act2)
        self.l2_act3 = activations.get_activation(layer2_act3)
        self.l2_act4 = activations.get_activation(layer2_act4)

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=layer2_out_channels, out_channels=layer3_out_channels, kernel_size=layer3_kernel_size, stride=1, padding=conv3_padding, bias=True)
        self.l3_act1 = activations.get_activation(layer3_act1)
        self.l3_act2 = activations.get_activation(layer3_act2)

        # Skip connection projection for Layer 4 addition
        self.skip2_proj_conv = None
        if layer3_out_channels != layer4_out_channels:
            self.skip2_proj_conv = nn.Conv2d(in_channels=layer3_out_channels, out_channels=layer4_out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=layer3_out_channels, out_channels=layer4_out_channels, kernel_size=layer4_kernel_size, stride=1, padding=conv4_padding, bias=True)
        self.l4_act1 = activations.get_activation(layer4_act1)
        self.l4_act2 = activations.get_activation(layer4_act2)
        self.l4_act3 = activations.get_activation(layer4_act3)
        self.l4_act4 = activations.get_activation(layer4_act4)

        # Layer 5
        self.conv5 = nn.Conv2d(in_channels=layer4_out_channels, out_channels=layer5_out_channels, kernel_size=layer5_kernel_size, stride=1, padding=conv5_padding, bias=True)
        self.l5_act1 = activations.get_activation(layer5_act1)
        self.l5_act2 = activations.get_activation(layer5_act2)

        # Layer 6 - Concatenate primary features with later features
        self.conv6 = nn.Conv2d(in_channels=layer1_out_channels + layer5_out_channels, out_channels=layer6_out_channels, kernel_size=layer6_kernel_size, stride=1, padding=conv6_padding, bias=True)
        self.l6_act1 = activations.get_activation(layer6_act1)
        self.l6_act2 = activations.get_activation(layer6_act2)

        # Layer 7
        self.conv7 = nn.Conv2d(in_channels=layer6_out_channels, out_channels=3 * (2**2), kernel_size=layer7_kernel_size, stride=1, padding=conv7_padding, bias=True)
        self.l7_act1 = activations.get_activation(layer7_act1)
        self.l7_act2 = activations.get_activation(layer7_act2)

        # Layer 8
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.act8 = nn.ReLU()

        # Instantiate the PerceptualLoss module
        self.perceptual_criterion = PerceptualLoss(
                                        pixel_loss_weight=0.990,
                                        vgg_weight=0.007,
                                        pixel_loss_type='charbonnier',
                                        high_frequency_weight=0.003,
                                        high_frequency_type='laplacian',
                                        lambda_lum=0.0,
                                        input_is_linear=True
                                    )        

    # Fuse_layers uses the imported fuse_modules
    def fuse_layers(self):
        """
        Fuses consecutive standard Conv2d, BatchNorm2d, and ReLU layers or BatchNorm2d and ReLU if fuse_modules utility is found.
        Modifies the model in-place. Requires model.eval().
        Uses torch.ao.quantization.fuse_modules.
        """
        if FUSION_UTILITY is None:
            print("Skipping PyTorch-side layer fusion: No suitable fusion utility found (torch.ao.quantization.fuse_modules).")
            print("Hopeful that ONNX Runtime will perform some fusion.")
            return # Exit if no utility

        if self.vebose:
            print("Attempting to fuse standard Conv2d/BatchNorm2d/ReLU sequences using torch.ao.quantization.fuse_modules...")
        self.eval() # Ensure eval mode

        try:
            # Define the modules to fuse by name.
            # fuse_modules primarily understands standard Conv/BN/ReLU sequences.
            # We can fuse the standard Conv+BN[+ReLU] sequences and the BN+ReLU sequences.
            # Note: This is an attempt at fusion. Given the arbitrary activations, it's expected to fail gracefully.
            fused_modules_list = [
                ['conv1', 'l1_act1'], 
                ['conv2', 'l2_act1'],
                ['conv3', 'l3_act1'],
                ['conv4', 'l4_act1'],
                ['conv5', 'l5_act1'],
                ['conv6', 'l6_act1'],
                ['conv7', 'l7_act1']
            ]
            # Use the assigned fusion utility (fuse_modules)
            FUSION_UTILITY(self, fused_modules_list, inplace=True)
            # Check for remaining activations is simplified for dynamic activation names
            if self.verbose:
                print("Fusion utility ran. Check model structure to verify effectiveness.")

        except Exception as e:
            if self.verbose:
                print(f"\nAn unexpected error occurred during layer fusion using {FUSION_UTILITY.__name__}: {e}")
                print("Skipping PyTorch-side layer fusion.")
                print("Hopeful that ONNX Runtime will perform fusion during inference.")

    # Forward pass
    def forward(self, x):
        """
        Forward pass of the model. Handles both fused/unfused and FP32/FP16 paths.
        Input: uint8 RGBA. Output: scaled FP16 RGBA when model is in FP16 mode.
        """
        identity = x

        # Apply downsampling by taking every even pixel; this reduces H and W by a factor of 2; x will become (Batch, Channels, H/2, W/2)
        x = self.pixel_unshuffle(x)

        # Layer 1
        x = self.conv1(x)
        if hasattr(self, 'l1_act1'): x = self.l1_act1(x)
        if hasattr(self, 'l1_act2'): x = self.l1_act2(x)
        long_skip = x
    
        # Layer 2
        short_skip_l2 = x # This is the skip connection from layer 1
        x = self.conv2(x)
        if hasattr(self, 'l2_act1'): x = self.l2_act1(x)
        if hasattr(self, 'l2_act2'): x = self.l2_act2(x)
        
        # Apply projection if channel dimensions differ for Layer 2 skip
        if self.skip1_proj_conv:
            short_skip_l2 = self.skip1_proj_conv(short_skip_l2)
        x = short_skip_l2 + x # Add the skip connection

        if hasattr(self, 'l2_act3'): x = self.l2_act3(x)
        if hasattr(self, 'l2_act4'): x = self.l2_act4(x)

        # Layer 3
        x = self.conv3(x)
        if hasattr(self, 'l3_act1'): x = self.l3_act1(x)
        if hasattr(self, 'l3_act2'): x = self.l3_act2(x)

        # Layer 4
        short_skip_l4 = x # This is the skip connection from layer 3
        x = self.conv4(x)
        if hasattr(self, 'l4_act1'): x = self.l4_act1(x)
        if hasattr(self, 'l4_act2'): x = self.l4_act2(x)

        # Apply projection if channel dimensions differ for Layer 4 skip
        if self.skip2_proj_conv:
            short_skip_l4 = self.skip2_proj_conv(short_skip_l4)
        x = short_skip_l4 + x # Add the skip connection

        if hasattr(self, 'l4_act3'): x = self.l4_act3(x)
        if hasattr(self, 'l4_act4'): x = self.l4_act4(x)

        # Layer 5
        x = self.conv5(x)
        if hasattr(self, 'l5_act1'): x = self.l5_act1(x)
        if hasattr(self, 'l5_act2'): x = self.l5_act2(x)      

        # Layer 6 - Concatenate primary features with later features
        x = torch.cat([long_skip, x], dim=1)
        x = self.conv6(x)
        if hasattr(self, 'l6_act1'): x = self.l6_act1(x)
        if hasattr(self, 'l6_act2'): x = self.l6_act2(x)

        # Layer 7
        x = self.conv7(x)
        if hasattr(self, 'l7_act1'): x = self.l7_act1(x)
        if hasattr(self, 'l7_act2'): x = self.l7_act2(x)

        # Layer 8 - This will bring it back to original resolution; upscale_factor=2
        x = self.pixel_shuffle(x)
        x = identity + x
        x = self.act8(x) # Make sure negative values can't escape

        return x

    # Criterion used by the training loop (Decided on perceptual loss)
    def criterion(self, output, target):
        return self.perceptual_criterion(output, target)

def get_model(name:str='lightweight'):
    if name == 'lightweight':
        return Model(layer1_out_channels=36, layer2_out_channels=36, layer3_out_channels=72, layer4_out_channels=72, layer5_out_channels=36, layer6_out_channels=36, layer7_out_channels=36)
    elif name == 'heavyweight':
        return Model(layer1_out_channels=36, layer2_out_channels=36, layer3_out_channels=108, layer4_out_channels=108, layer5_out_channels=36, layer6_out_channels=36, layer7_out_channels=36)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test performance')
    parser.add_argument('--model_type', type=str, required=True, choices=['lightweight', 'heavyweight'], help='Type of model: lightweight, heavyweight')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("==============================")
    print(" Edge model float16")
    print("==============================")
    model = get_model(args.model_type).to(device)
    model.eval()
    model.fuse_layers()
    model.half()

    print("Attempting to compile model...")
    try:
        model = torch.compile(model, mode="default", fullgraph=True)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Falling back to eager mode.")
        model = model.to(device)

    x = torch.rand((1, 3, 576, 752), dtype=torch.float16).to(device)

    # Warm-up
    print("Starting warm-up...")
    with torch.no_grad(): # No gradients needed for warm-up and inference
        for _ in range(20):
            _ = model(x) # Call the model's forward pass
    print("Warm-up finished.")

    # Measure FPS over 20 seconds
    print("Measuring FPS...")
    start_time = time.time()
    num_iterations = 0

    with torch.no_grad(): # No gradients needed for inference
        while time.time() - start_time < 20:
            _ = model(x) # Call the model's forward pass
            num_iterations += 1

    elapsed_time = time.time() - start_time
    fps = num_iterations / elapsed_time

    # Get model output shape and size
    # Run one more forward pass to get the output shape after warm-up
    with torch.no_grad():
         output_shape = model(x).shape

    # Calculate model size (only trainable parameters)
    model_size_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_size_params * torch.finfo(torch.float16).bits / 8 / 1e6

    print("\n--- Results ---")
    print("Model output shape:", output_shape)
    print(f"Model size (trainable parameters): {model_size_params}")
    print(f"Model size (MB, assuming float16): {model_size_mb:.2f} MB")
    print(f"Average FPS: {fps:.2f}")
    print("---------------")
    sys.exit(0)