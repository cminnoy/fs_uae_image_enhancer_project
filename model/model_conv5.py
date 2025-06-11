import torch
import torch.nn as nn
from loss_vgg import PerceptualLoss
from loss_ssim import SSIMLoss
import sys, time
import argparse

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

# Conv2D 5 layer model
class Model(nn.Module):

    def __init__(self, initial_out_channels=32, mid_out_channels=64, final_out_channels=3, kernel_size=3):
        """
        Initializes the BasicModel with a sequence of Conv2d layers (no bias) followed by BatchNorm2d and ReLU6.
        Preserves spatial dimensions. The output FP32 tensor will be scaled to the [0.0, 255.0] range.

        Args:
            initial_out_channels (int): Number of output channels for the first convolution and BatchNorm. Used also in the second and fourth layer.
            mid_out_channels (int): Number of output channels for the third convolution and BatchNorm.
            final_out_channels (int): Number of output channels for the final convolution and BatchNorm (must match target channels).
            kernel_size (int): Size of the convolutional kernel (must be odd, e.g., 3 or 5).
        """
        super(Model, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")

        padding = (kernel_size - 1) // 2

        # Define the sequence of Conv2d (no bias) and BatchNorm2d, and Activation layers
        # Layer 1: Conv -> BatchNorm -> Activation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=initial_out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_out_channels)
        self.act1 = nn.ReLU(inplace=True)

        # Layer 2: Conv -> BatchNorm -> Activation
        self.conv2 = nn.Conv2d(in_channels=initial_out_channels, out_channels=initial_out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(initial_out_channels)
        self.act2 = nn.ReLU(inplace=True)

        # Layer 3: Conv -> BatchNorm -> Activation
        self.conv3 = nn.Conv2d(in_channels=initial_out_channels, out_channels=mid_out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_out_channels)
        self.act3 = nn.ReLU(inplace=True)

        # Layer 4: Conv -> BatchNorm -> Activation
        self.conv4 = nn.Conv2d(in_channels=mid_out_channels, out_channels=mid_out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_out_channels)
        self.act4 = nn.ReLU(inplace=True)

        # Layer 5 (Output layer): Conv -> BatchNorm (No Activation)
        self.conv5 = nn.Conv2d(in_channels=mid_out_channels, out_channels=final_out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn5 = nn.BatchNorm2d(final_out_channels)
        self.act5 = nn.Sigmoid()

        # Instantiate the PerceptualLoss module
       # self.perceptual_criterion = PerceptualLoss(pixel_loss_weight=0.9, vgg_weight=0.1, pixel_loss_type='charbonnier', charbonnier_epsilon=1e-6, input_is_linear=False)
        self.perceptual_criterion = SSIMLoss(pixel_loss_type='L1', convert_to_linear_for_ssim=False, ssim_weight=0.01, pixel_loss_weight=0.99)

    # Fuse_layers uses the imported fuse_modules
    def fuse_layers(self):
        """
        Fuses consecutive standard Conv2d, BatchNorm2d, and ReLU layers or BatchNorm2d and ReLU if fuse_modules utility is found.
        Modifies the model in-place. Requires model.eval().
        Uses torch.ao.quantization.fuse_modules.
        """
        if FUSION_UTILITY is None:
            print("Skipping PyTorch-side layer fusion: No suitable fusion utility found (torch.ao.quantization.fuse_modules).")
            print("Hopeful that ONNX Runtime will perform Conv+BN[+ReLU] fusion.")
            return # Exit if no utility

        print("Attempting to fuse standard Conv2d/BatchNorm2d/ReLU sequences using torch.ao.quantization.fuse_modules...")
        self.eval() # Ensure eval mode

        try:
            # Define the modules to fuse by name.
            # fuse_modules primarily understands standard Conv/BN/ReLU sequences.
            # We can fuse the standard Conv+BN[+ReLU] sequences and the BN+ReLU sequences.
            fused_modules_list = [
                ['conv1', 'bn1', 'act1'],
                ['conv2', 'bn2', 'act2'],
                ['conv3', 'bn3', 'act3'],
                ['conv4', 'bn4', 'act4'],
                ['conv5', 'bn5', 'act5']
            ]
            # Use the assigned fusion utility (fuse_modules)
            FUSION_UTILITY(self, fused_modules_list, inplace=True)

            # Check if fusion actually removed the BN and ReLU layers that were expected to fuse
            remaining_bn = hasattr(self, 'bn1') or hasattr(self, 'bn2') or hasattr(self, 'bn3') or hasattr(self, 'bn4') or hasattr(self, 'bn5')
            remaining_act = hasattr(self, 'act1') or hasattr(self, 'act2') or hasattr(self, 'act3') or hasattr(self, 'act4') or hasattr(self, 'act5')

            if not remaining_bn and not remaining_act:
                 print("Standard Conv/BN/ReLU sequences fused successfully.")
            else:
                 print("Fusion utility ran, but some BatchNorm or Activation layers still exist. Fusion might not have been fully effective.")

        except Exception as e:
            print(f"\nAn unexpected error occurred during layer fusion using {FUSION_UTILITY.__name__}: {e}")
            print("Skipping PyTorch-side layer fusion.")
            print("Hopeful that ONNX Runtime will perform Conv+BN[+ReLU] fusion during inference.")

    # Forward pass
    def forward(self, x):
        """
        Forward pass of the model. Handles both fused/unfused and FP32/FP16 paths.
        Input: uint8 RGBA. Output: scaled FP16 RGBA when model is in FP16 mode.
        """        
        # Pass through the sequence of layers.
        # Check if BN/Activation layers still exist after fusion.

        # Layer 1
        x = self.conv1(x)
        if hasattr(self, 'bn1'): x = self.bn1(x)
        if hasattr(self, 'act1'): x = self.act1(x)
    
        # Layer 2
        skip = x
        x = self.conv2(x)
        if hasattr(self, 'bn2'): x = self.bn2(x)
        x = skip + x
        if hasattr(self, 'act2'): x = self.act2(x)

        # Layer 3
        x = self.conv3(x)
        if hasattr(self, 'bn3'): x = self.bn3(x)
        if hasattr(self, 'act3'): x = self.act3(x)

        # Layer 4
        skip = x
        x = self.conv4(x)
        if hasattr(self, 'bn4'): x = self.bn4(x)
        x = skip + x
        if hasattr(self, 'act4'): x = self.act4(x)

        # Layer 5
        x = self.conv5(x)
        if hasattr(self, 'bn5'): x = self.bn5(x)
        if hasattr(self, 'act5'): x = self.act5(x)

        return x

    # Criterion used by the training loop
    def criterion(self, output, target):
        return self.perceptual_criterion(output, target)

def get_model(name:str='lightweight'):
    if name == 'lightweight':
        return Model(initial_out_channels=32, mid_out_channels=64)
    elif name == 'heavyweight':
        return Model(initial_out_channels=64, mid_out_channels=128)
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

    x = torch.randint(0, 256, (1, 3, 576, 752), dtype=torch.uint8).to(device)

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


