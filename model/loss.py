import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# Define the Perceptual Loss class combining VGG with L1 or Charbonnier
class PerceptualLoss(nn.Module):
    def __init__(self, vgg_layer_weights=None, pixel_loss_weight=1.0, vgg_weight=0.006,
                 pixel_loss_type='l1', charbonnier_epsilon=1e-6):
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
        """
        super(PerceptualLoss, self).__init__()

        # Load a pre-trained VGG16 model
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # Set VGG to evaluation mode
        self.vgg.eval()

        # Define VGG layer names and indices (as before)
        self.layer_names = [
            'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'
        ]
        self.layer_indices = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
            'relu5_3': 30
        }

        # Define default VGG layer weights
        if vgg_layer_weights is None:
            self.vgg_layer_weights = {
                'relu1_2': 0.1,
                'relu2_2': 0.2,
                'relu3_3': 0.3,
                'relu4_3': 0.4,
                'relu5_3': 0.5
            }
        else:
            self.vgg_layer_weights = vgg_layer_weights

        self.pixel_loss_weight = pixel_loss_weight
        self.vgg_weight = vgg_weight
        self.pixel_loss_type = pixel_loss_type.lower() # Enforce lowercase
        self.charbonnier_epsilon = charbonnier_epsilon

        # Validate pixel loss type
        if self.pixel_loss_type not in ['l1', 'charbonnier']:
            raise ValueError(f"Invalid pixel_loss_type: {pixel_loss_type}. Must be 'l1' or 'charbonnier'")

        # Define image normalization for VGG input
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def extract_vgg_features(self, x):
        features = {}
        x = self.normalize(x)

        for name, module in self.vgg._modules.items():
            x = module(x)
            for layer_name, index in self.layer_indices.items():
                 if int(name) == index and layer_name in self.layer_names:
                    features[layer_name] = x
                    break
        return features

    def calculate_pixel_loss(self, output, target):
        """
        Calculates the pixel-wise loss (L1 or Charbonnier) based on the chosen type.
        """
        if self.pixel_loss_type == 'l1':
            return F.l1_loss(output, target)
        elif self.pixel_loss_type == 'charbonnier':
            return self.charbonnier_loss(output, target, self.charbonnier_epsilon)
        else:
            raise ValueError("Invalid pixel_loss_type (this should not happen)")

    def charbonnier_loss(self, output, target, epsilon=1e-6):
        """
        Calculates the Charbonnier loss.
        """
        squared_diff = (output - target)**2
        return torch.mean(torch.sqrt(squared_diff + epsilon**2))

    def forward(self, output, target):
        output_rgb = output[:, :3, :, :].float().div(255.0)
        target_rgb = target[:, :3, :, :].float().div(255.0)

        # Calculate pixel-wise loss (L1 or Charbonnier)
        pixel_loss = self.calculate_pixel_loss(output_rgb, target_rgb)

        # Calculate VGG perceptual loss
        vgg_output_features = self.extract_vgg_features(output_rgb)
        vgg_target_features = self.extract_vgg_features(target_rgb)

        vgg_loss = 0.0
        for layer_name in self.layer_names:
            if layer_name in vgg_output_features and layer_name in vgg_target_features:
                weight = self.vgg_layer_weights.get(layer_name, 0.0)
                vgg_loss += weight * F.l1_loss(vgg_output_features[layer_name], vgg_target_features[layer_name])

        # Combine pixel-wise and VGG losses
        total_loss = self.pixel_loss_weight * pixel_loss + self.vgg_weight * vgg_loss

        return total_loss
