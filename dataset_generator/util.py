from PIL import Image, ImageOps # Pillow for image manipulation
import warnings # For generating warnings
import numpy as np # For image processing (e.g., interlacing)

# Optional PyTorch import
try:
    import torch
    from torchvision.transforms import ToPILImage
except ImportError:
    torch = None
    ToPILImage = None

# --- Assume necessary imports from quantize module ---
# Make sure your quantize.py module defines:
# - DIFFUSION_MAPS: A dictionary mapping dither method names (strings) to their implementation.
# - reduce_color_depth_and_dither(img_pil, color_space, target_palette_size, dithering_method):
#   A function that takes a PIL image and applies the specified color space reduction,
#   optional palette quantization, and optional dithering.
#   color_space: String, e.g., 'RGB888', 'RGB565', 'RGB444', 'RGB332'.
#   target_palette_size: Integer for palette size, or None for no palette reduction.
#   dithering_method: String, 'None' or a key from DIFFUSION_MAPS.
try:
    from quantize import DIFFUSION_MAPS, reduce_color_depth_and_dither
    # Add 'None' as a valid dithering method string if your quantize doesn't handle it explicitly
    _all_dither_methods = ['None', 'checkerboard'] + list(DIFFUSION_MAPS.keys())
except ImportError:
    warnings.warn("Could not import quantization module (quantize.py). Quantization and dithering styles will not work.")
    DIFFUSION_MAPS = {}
    _all_dither_methods = ['None']
    # Provide a dummy implementation if quantize is missing
    def reduce_color_depth_and_dither(img_pil, color_space, target_palette_size, dithering_method):
        warnings.warn(f"Quantization function (reduce_color_depth_and_dither) not available for {color_space}/{target_palette_size}/{dithering_method}. Returning original image.")
        return img_pil.copy() # Return original as fallback


# --- Define Global Constants (Ensure consistent with generator args) ---
# These constants define the valid string names for resolution styles and dithering methods
# that will be used in filenames and argparse choices.
SUPPORTED_DITHER_METHODS = _all_dither_methods
SUPPORTED_RESOLUTION_STYLES = ['lores', 'hires', 'lores_laced', 'hires_laced'] # Names matching filename encoding


# --- Utility Functions ---

def is_pure_black_pil(img_pil: Image.Image) -> bool:
    """
    Checks if an RGB Pillow image is pure black (all pixels are exactly (0, 0, 0)).
    This uses PIL's getextrema() for a precise check.
    """
    # Ensure the image is in RGB mode for consistent check
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    # getextrema() returns (min, max) for each channel.
    # For RGB, it returns ((min_R, max_R), (min_G, max_G), (min_B, max_B)).
    stats = img_pil.getextrema()

    # An image is pure black if and only if the maximum value for *each* RGB channel is 0.
    # If max is 0, min must also be 0.
    is_black = all(max_val == 0 for min_val, max_val in stats)

    return is_black

def get_crop_and_pad(image_pil: Image.Image, crop_x: int, crop_y: int, crop_w: int, crop_h: int) -> Image.Image:
    """
    Extracts a crop of size (crop_w, crop_h) from a PIL image starting at (crop_x, crop_y).
    Pads with black pixels if the crop region extends outside the image boundaries.
    Coordinates (crop_x, crop_y) can be negative.
    """
    img_w, img_h = image_pil.size

    # Calculate necessary padding amounts for each side. max(0, -coord) handles negative starts.
    padding_left = max(0, -crop_x)
    padding_top = max(0, -crop_y)
    # Calculate padding needed on right/bottom if the end of the crop extends past the image edge
    padding_right = max(0, (crop_x + crop_w) - img_w)
    padding_bottom = max(0, (crop_y + crop_h) - img_h)

    # Apply padding if any is needed using ImageOps.expand
    if padding_left > 0 or padding_top > 0 or padding_right > 0 or padding_bottom > 0:
        try:
            # ImageOps.expand adds a border (padding) to the image.
            # The border tuple is (left, top, right, bottom).
            # Fill color is black (0, 0, 0) for RGB.
            padded_image_pil = ImageOps.expand(image_pil,
                                               border=(padding_left, padding_top, padding_right, padding_bottom),
                                               fill=(0, 0, 0))
        except Exception as e:
             warnings.warn(f"Error during ImageOps.expand with padding ({padding_left}, {padding_top}, {padding_right}, {padding_bottom}) on image size {image_pil.size}: {e}. Returning black image of crop size.")
             # Return a black image of the target crop size as a fallback if padding fails
             return Image.new('RGB', (crop_w, crop_h), (0, 0, 0))
    else:
        # No padding required, work directly with the original image
        padded_image_pil = image_pil

    # Calculate the coordinates of the crop box relative to the *padded* image.
    # The original (crop_x, crop_y) point is now shifted by the padding amounts.
    padded_crop_x1 = crop_x + padding_left
    padded_crop_y1 = crop_y + padding_top
    padded_crop_x2 = padded_crop_x1 + crop_w
    padded_crop_y2 = padded_crop_y1 + crop_h

    # Perform the crop operation on the padded image
    try:
        crop_pil = padded_image_pil.crop((padded_crop_x1, padded_crop_y1, padded_crop_x2, padded_crop_y2))
    except Exception as e:
        warnings.warn(f"Error during PIL crop operation with box ({padded_crop_x1}, {padded_crop_y1}, {padded_crop_x2}, {padded_crop_y2}) on padded image size {padded_image_pil.size}: {e}. Returning black image of crop size.")
        # If cropping fails, return a black image of the expected crop size
        return Image.new('RGB', (crop_w, crop_h), (0, 0, 0))

    # Final check to ensure the output image has the correct dimensions
    if crop_pil.size != (crop_w, crop_h):
         warnings.warn(f"Cropping resulted in incorrect size. Expected {(crop_w, crop_h)}, got {crop_pil.size}. Returning black image of crop size.")
         # If the size is wrong, return a black image as a fallback
         return Image.new('RGB', (crop_w, crop_h), (0, 0, 0))

    return crop_pil

def apply_rotation(image_pil: Image.Image, angle_degrees: int, supersample_factor: int = 2, pil_filter=Image.Resampling.BICUBIC) -> Image.Image:
    """
    Rotates a PIL image by a specified angle in degrees.
    Uses the LANCZOS resampling filter for quality.
    Applies Anti-Aliasing using the super_sample factor.
    The canvas is expanded to include the entire rotated image without cropping.
    Returns a new PIL Image object.
    """
    if supersample_factor < 1:
        raise ValueError("supersample_factor must be an integer greater than or equal to 1.")

    # Normalize the angle to be within [0, 360) degrees
    normalized_angle = angle_degrees % 360
    if normalized_angle == 0:
        # No rotation needed, return a copy to ensure the original image isn't modified elsewhere.
        return image_pil.copy()

    try:
        original_width, original_height = image_pil.size
        if supersample_factor > 1:
            supersampled_width = original_width * supersample_factor
            supersampled_height = original_height * supersample_factor
            supersampled_image = image_pil.resize((supersampled_width, supersampled_height), Image.Resampling.BICUBIC)
            rotated_supersampled_image = supersampled_image.rotate(angle_degrees, resample=Image.Resampling.NEAREST)
            final_image = rotated_supersampled_image.resize((original_width, original_height), Image.Resampling.BICUBIC)
        else:
            final_image = image_pil.rotate(angle_degrees, resample=pil_filter)
        return final_image
    except Exception as e:
         warnings.warn(f"Error during PIL rotate operation by {angle_degrees} degrees: {e}. Returning original image copy.")
         # Return a copy of the original image as a fallback if rotation fails
         return image_pil.copy()

def apply_downscaling(image_pil: Image.Image, percentage: int) -> Image.Image:
    """
    Downscales a PIL image by a specified percentage (e.g., 50 for 50% size).
    Uses the LANCZOS resampling filter for quality.
    Percentage must be an integer > 0 and < 100.
    Returns a new PIL Image object.
    If percentage is invalid (<= 0 or >= 100), returns a copy of the original image.
    """
    if percentage <= 0 or percentage >= 100:
        # Invalid percentage for downscaling. 0% implies 100% scale, 100% is not downscaling.
        # Return a copy of the original image as if no downscaling was requested.
        warnings.warn(f"Invalid downscale percentage {percentage}%. Must be > 0 and < 100. Returning original image copy.")
        return image_pil.copy()

    # Calculate the target dimensions based on the percentage
    original_w, original_h = image_pil.size
    target_w = int(original_w * (percentage / 100.0))
    target_h = int(original_h * (percentage / 100.0))

    # Ensure the target size is at least 1x1 pixel (should be true if percentage > 0)
    target_w = max(1, target_w)
    target_h = max(1, target_h)

    # If the calculated target size is the same as the original, no scaling is needed.
    if target_w == original_w and target_h == original_h:
         return image_pil.copy()

    # Resize the image to the target dimensions using LANCZOS filter for quality downsampling.
    try:
        downscaled_pil = image_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return downscaled_pil
    except Exception as e:
        warnings.warn(f"Error during PIL resize (downscaling by {percentage}%) from {image_pil.size} to ({target_w}, {target_h}): {e}. Returning original image copy.")
        # Return a copy of the original image as a fallback if downscaling fails
        return image_pil.copy()

def apply_resolution_style(image_pil: Image.Image, style: str) -> Image.Image:
    """
    Applies resolution style effects (like pixel simulation or interlacing) to a PIL image.
    The input image is expected to be the size of the target crop (W x H).
    The output image will also be the same size (W x H).
    Returns a new PIL Image object.
    """
    if style not in SUPPORTED_RESOLUTION_STYLES:
        warnings.warn(f"Unknown resolution style '{style}'. Returning original image copy.")
        return image_pil.copy()

    output_pil = image_pil.copy() # Start with a copy of the input quantized image

    try:
        w, h = output_pil.size # Size of the input image (should be the crop size)

        # --- Implement Resolution Style Visual Effects ---
        # These effects simulate different display resolutions/modes *within* the fixed crop size.
        # They might involve downscaling followed by nearest-neighbor upscaling to simulate blocky pixels,
        # or direct pixel manipulation for interlacing.

        if style == 'lores':
            # Simulate 2x2 source pixels mapping to 1 visual pixel (blocky 2x2 pixels).
            # Downscale by 2x2 using BOX (simple average) and then upscale by 2x2 using Nearest Neighbor.
            lores_sim_pil = output_pil.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
            output_pil = lores_sim_pil.resize((w, h), Image.Resampling.NEAREST)

        elif style == 'lores_laced':
            # Simulate 2x1 source pixels mapping to 1 visual pixel width (blocky 2x1 pixels) + interlacing.
            # Downscale width by 2 using BOX, then upscale width by 2 using Nearest Neighbor.
            lores_sim_pil = output_pil.resize((w // 2, h), Image.Resampling.LANCZOS)
            output_pil = lores_sim_pil.resize((w, h), Image.Resampling.NEAREST)

        elif style == 'hires':
             # Simulate 1x2 source pixels mapping to 1 visual pixel height (blocky 1x2 pixels) + interlacing.
             # Downscale height by 2 using BOX, then upscale height by 2 using Nearest Neighbor.
             hires_sim_pil = output_pil.resize((w, h // 2), Image.Resampling.LANCZOS)
             output_pil = hires_sim_pil.resize((w, h), Image.Resampling.NEAREST)

        elif style == 'hires_laced':
             pass

        # Ensure the output size remains the same as the input size (the crop size)
        # This should be handled by the logic above, but this is a safety check.
        if output_pil.size != (w, h):
             warnings.warn(f"Resolution style '{style}' transformation resulted in unexpected size. Expected {(w, h)}, got {output_pil.size}. Resizing to expected size using Nearest Neighbor.")
             # Resize back to expected size using Nearest Neighbor if size changed unexpectedly
             output_pil = output_pil.resize((w, h), Image.Resampling.NEAREST)

    except Exception as e:
         warnings.warn(f"Error applying resolution style '{style}': {e}. Returning original image copy.")
         return image_pil.copy() # Return original as fallback

    return output_pil

def pre_apply_resolution_style(image_pil: Image.Image, style: str) -> Image.Image:
    """
    Applies resolution style effects (like pixel simulation or interlacing) to a PIL image.
    The input image is expected to be the size of the target crop (W x H).
    The output image may have a different resolution.
    Returns a new PIL Image object.
    """
    if style not in SUPPORTED_RESOLUTION_STYLES:
        warnings.warn(f"Unknown resolution style '{style}'. Returning original image copy.")
        return image_pil.copy()

    output_pil = image_pil.copy() # Start with a copy of the input

    w, h = output_pil.size # Size of the input image (should be the crop size)

    # --- Implement Resolution Style Visual Effects ---
    # These effects simulate different display resolutions/modes.
    if style == 'lores':
        # Simulate 2x2 source pixels mapping to 1 visual pixel (blocky 2x2 pixels).
        # Downscale by 2x2
        output_pil = output_pil.resize((w // 2, h // 2), Image.Resampling.BICUBIC)
    elif style == 'lores_laced':
        # Simulate 2x1 source pixels mapping to 1 visual pixel width (blocky 2x1 pixels) + interlacing.
        # Downscale width by 2
        output_pil = output_pil.resize((w // 2, h), Image.Resampling.BICUBIC)
    elif style == 'hires':
        # Simulate 1x2 source pixels mapping to 1 visual pixel height (blocky 1x2 pixels) + interlacing.
        # Downscale height by 2
        output_pil = output_pil.resize((w, h // 2), Image.Resampling.BICUBIC)
    elif style == 'hires_laced':
        pass

    return output_pil

def post_apply_resolution_style(image_pil: Image.Image, style: str) -> Image.Image:
    """
    Applies resolution style effects (like pixel simulation or interlacing) to a PIL image.
    The input image is expected to be first processed by pre_resolution_style.
    The output image will be back the original size (W x H).
    Returns a new PIL Image object.
    """
    if style not in SUPPORTED_RESOLUTION_STYLES:
        warnings.warn(f"Unknown resolution style '{style}'. Returning original image copy.")
        return image_pil.copy()

    output_pil = image_pil.copy() # Start with a copy of the input quantized image

    w, h = output_pil.size # Size of the input image (should be the crop size)

    # --- Implement Resolution Style Visual Effects ---
    # These effects simulate different display resolutions/modes.
    if style == 'lores':
        # Simulate 2x2 source pixels mapping to 1 visual pixel (blocky 2x2 pixels).
        # Upscale by 2x2 using Nearest Neighbor.
        output_pil = output_pil.resize((w * 2, h * 2), Image.Resampling.NEAREST)
    elif style == 'lores_laced':
        # Simulate 2x1 source pixels mapping to 1 visual pixel width (blocky 2x1 pixels) + interlacing.
        # Upscale width by 2 using Nearest Neighbor.
        output_pil = output_pil.resize((w * 2, h), Image.Resampling.NEAREST)
    elif style == 'hires':
        # Simulate 1x2 source pixels mapping to 1 visual pixel height (blocky 1x2 pixels) + interlacing.
        # Upscale height by 2 using Nearest Neighbor.
        output_pil = output_pil.resize((w, h * 2), Image.Resampling.NEAREST)
    elif style == 'hires_laced':
        pass

    return output_pil

def load_model(model_path: str):
    """
    Loads a PyTorch model from the specified path.
    Returns the model and its device.
    """
    if torch is None:
        warnings.warn("PyTorch model loading requested, but torch not available.")
        return None

    try:
        model = torch.jit.load(model_path, map_location='cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model
    except Exception as e:
        warnings.warn(f"Error loading PyTorch model from {model_path}: {e}.")
        return None
