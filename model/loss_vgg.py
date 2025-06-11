import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import kornia
from gamma import linear_to_srgb_approx

# Charbonnier loss
def charbonnier_loss(output, target, epsilon=1e-6):
    """
    Calculates the Charbonnier loss between the output and target tensors.

    Args:
        output (torch.Tensor): The model's output tensor.
        target (torch.Tensor): The target tensor.
        epsilon (float): Small constant to prevent division by zero or infinite gradient.

    Returns:
        torch.Tensor: The calculated Charbonnier loss (mean reduction).
    """
    # Ensure tensors have the same shape
    if output.shape != target.shape:
        raise ValueError(f"Output and target tensors must have the same shape, but got {output.shape} and {target.shape}")

    # Calculate the squared difference
    squared_diff = (output - target)**2

    # Calculate the Charbonnier loss per element
    loss_per_element = torch.sqrt(squared_diff + epsilon**2)

    # Return the mean loss over all elements
    return torch.mean(loss_per_element)
    
# Define the Perceptual Loss class combining VGG with L1 or Charbonnier
class PerceptualLoss(nn.Module):
    def __init__(self, vgg_layer_weights=None, pixel_loss_weight=1.0, vgg_weight=0.006,
                 pixel_loss_type='l1', charbonnier_epsilon=1e-6,
                 high_frequency_weight=0.0, high_frequency_type='laplacian',
                 lambda_lum = 0.0,
                 input_is_linear:bool = False):
        """
        Initializes the PerceptualLoss module using a pre-trained VGG network
        for feature extraction and combining it with pixel-wise loss (L1 or Charbonnier).

        Args:
            vgg_layer_weights (dict, optional): A dictionary mapping VGG layer names
                to their respective weights for the perceptual loss.
                If None, uses a default set of layers and weights.
            pixel_loss_weight (float): Weight for the pixel-wise loss component (L1 or Charbonnier).
            vgg_weight (float): Weight for the total VGG perceptual loss component.
            pixel_loss_type (str): Type of pixel-wise loss to use. Must be 'l1' or 'charbonnier'.
            charbonnier_epsilon (float):  Small constant for Charbonnier loss to prevent instability.
            high_frequency_weight (float): Weight for the high-frequency loss component.
            high_frequency_type (str): Type of high-frequency filter. Must be 'laplacian'.
        """
        super(PerceptualLoss, self).__init__()

        # Load a pre-trained VGG16 model
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # Set VGG to evaluation mode
        self.vgg.eval()

        # Define VGG layer names and indices (as before)
        self.layer_names = [
            'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'
        ]
        self.layer_indices = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_3': 26
        }

        # Map numerical indices back to descriptive names for easier lookup
        self._vgg_numerical_to_descriptive_map = {
            str(idx): name for name, idx in self.layer_indices.items()
        }

        # Define default VGG layer weights
        if vgg_layer_weights is None:
            self.vgg_layer_weights = {
                'relu1_2': 1.0/2.6,
                'relu2_2': 1.0/4.8,
                'relu3_3': 1.0/3.7,
                'relu4_3': 1.0/5.6
            }
        else:
            self.vgg_layer_weights = vgg_layer_weights

        self.pixel_loss_weight = pixel_loss_weight
        self.vgg_weight = vgg_weight
        self.pixel_loss_type = pixel_loss_type.lower()
        self.charbonnier_epsilon = charbonnier_epsilon
        self.high_frequency_weight = high_frequency_weight
        self.high_frequency_type = high_frequency_type.lower()
        self.lambda_lum = lambda_lum
        self.input_is_linear = input_is_linear

        # Validate pixel loss type
        if self.pixel_loss_type not in ['l1', 'charbonnier']:
            raise ValueError(f"Invalid pixel_loss_type: {pixel_loss_type}. Must be 'l1' or 'charbonnier'")

        # Validate high-frequency loss type
        if self.high_frequency_type not in ['laplacian']:
            raise ValueError(f"Invalid high_frequency_type: {high_frequency_type}. Must be 'laplacian'")

        # Define image normalization for VGG input
        self.normalize = transforms.Normalize(mean=[0.48235, 0.45882, 0.40784],
                                              std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])

        # Define Laplacian kernel for high-frequency loss
        # This kernel is 3x3, applied depthwise (across channels)
        # It detects edges by approximating the second derivative.
        self.laplacian_kernel = torch.tensor([
            [ 0,  1,  0],
            [ 1, -4,  1],
            [ 0,  1,  0]
        ], dtype=torch.float32).reshape(1, 1, 3, 3) # Reshape for conv2d: (out_channels, in_channels, kH, kW)

    def extract_vgg_features(self, x):
        features = x
        outputs_dict = {}
        # Iterate over named children of the VGG features attribute
        for name, layer in self.vgg.features.named_children():
            features = layer(features)
            numerical_index = int(name)
            if numerical_index in self.layer_indices.values():
                descriptive_name = self._vgg_numerical_to_descriptive_map.get(name, None)
                if descriptive_name:
                    outputs_dict[descriptive_name] = features

        return outputs_dict

    def calculate_pixel_loss(self, output, target):
        """
        Calculates the pixel-wise loss (L1 or Charbonnier) based on the chosen type.
        """
        if self.pixel_loss_type == 'l1':
            return F.l1_loss(output, target)
        elif self.pixel_loss_type == 'charbonnier':
            return charbonnier_loss(output, target, self.charbonnier_epsilon)
        else:
            raise ValueError("Invalid pixel_loss_type (this should not happen)")

    def calculate_high_frequency_loss(self, output, target):
        """
        Calculates the high-frequency loss using a specified filter (e.g., Laplacian).
        Applies the filter channel-wise to both output and target, then computes L1 loss.
        """
        if self.high_frequency_type == 'laplacian':
            # Ensure kernel is on the same device as input
            kernel = self.laplacian_kernel.to(output.device)

            # Apply Laplacian filter to each channel
            # Use F.conv2d for filtering. groups=output.shape[1] applies it channel-wise.
            high_freq_output = F.conv2d(output, kernel.repeat(output.shape[1], 1, 1, 1),
                                        padding='same', groups=output.shape[1])
            high_freq_target = F.conv2d(target, kernel.repeat(target.shape[1], 1, 1, 1),
                                        padding='same', groups=target.shape[1])

            # Use L1 loss for the difference in high-frequency components
            return F.l1_loss(high_freq_output, high_freq_target)
        else:
            raise ValueError("Invalid high_frequency_type (this should not happen)")

    def forward(self, output, target):
        if self.input_is_linear:
            output_for_vgg = self.normalize(linear_to_srgb_approx(output).clamp(0.0, 1.0)) # Convert to sRGB
            target_for_vgg = self.normalize(linear_to_srgb_approx(target)) # Convert to sRGB
        else:
            output_for_vgg = self.normalize(output.clamp(0.0, 1.0)) # Assumed sRGB
            target_for_vgg = self.normalize(target) # Assumed sRGB

        # Calculate pixel-wise loss (L1 or Charbonnier)
        pixel_loss = self.calculate_pixel_loss(output, target)

        if self.vgg_weight > 0:
            vgg_output_features = self.extract_vgg_features(output_for_vgg)
            vgg_target_features = self.extract_vgg_features(target_for_vgg)
            vgg_loss = sum(
                self.vgg_layer_weights.get(layer, 0.0) * F.l1_loss(vgg_output_features[layer], vgg_target_features[layer])
                for layer in self.layer_names if layer in vgg_output_features and layer in vgg_target_features
            )
        else:
            vgg_loss = 0.0

        if self.lambda_lum > 0:
            output_ycbcr = kornia.color.rgb_to_ycbcr(output_for_vgg)
            target_ycbcr = kornia.color.rgb_to_ycbcr(target_for_vgg)
            luminance_loss = F.l1_loss(output_ycbcr[:, 0:1, :, :], target_ycbcr[:, 0:1, :, :])
        else:
            luminance_loss = 0.0

        if self.high_frequency_weight > 0:
            high_frequency_loss = self.calculate_high_frequency_loss(output, target)
        else:
            high_frequency_loss = 0.0
         
        # Combine all losses
        total_loss = self.pixel_loss_weight * pixel_loss + \
                     self.vgg_weight * vgg_loss + \
                     self.high_frequency_weight * high_frequency_loss + \
                     self.lambda_lum * luminance_loss
       # print(f"total loss: {total_loss}, pixel weight: {self.pixel_loss_weight} pixel loss: {pixel_loss}, vgg weight {self.vgg_weight}, vgg_loss {vgg_loss}")
        return total_loss
