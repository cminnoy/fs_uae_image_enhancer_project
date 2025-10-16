# In generator_overhaul.py

import os, sys
import argparse
import numpy as np
from PIL import Image
import random
import time
from datetime import timedelta 
import warnings
import concurrent.futures
import math
import re
from quantize import reduce_color_depth_and_dither, DIFFUSION_MAPS
from cache import ScanCache, DEFAULT_TRAIN_CACHE_FILE, DEFAULT_TEST_CACHE_FILE
import signal

stop_processing = False

# Import utility functions and constants
try:
    from util import (
        should_discard_by_black_ratio,
        get_crop_and_pad,
        apply_rotation,
        apply_downscaling,        
        pre_apply_resolution_style, post_apply_resolution_style,
        load_model,
        SUPPORTED_DITHER_METHODS,
        SUPPORTED_RESOLUTION_STYLES
    )
except ImportError:
    print("Please ensure util.py is in the same directory.")
    exit(1)

# --- Helper to Construct Filenames ---
# This is the reverse of parsing, must match exactly for scan/match to work
def construct_filename(params: dict, is_target: bool) -> str:
    """
    Constructs a filename based on the given parameters dictionary.
    Assumes parameters dictionary keys match the structure returned by parse_generated_filename.
    """
    # Check for mandatory parameters regardless of type
    if 'crop_x' not in params or 'crop_y' not in params or 'scale_perc' not in params or 'rot_deg' not in params:
         raise ValueError("Missing mandatory crop/pre-processing parameters for filename construction.")


    if is_target:
        # Target filename format: target_<X>_<Y>_s<scale>_r<rot>.png
        return f"target_{params['crop_x']}_{params['crop_y']}_s{params['scale_perc']}_r{params['rot_deg']}.png"
    else:
        # Styled filename format: <resolution>_<X>_<Y>_s<scale>_r<rot>_rgb<rgb>_p<pal>_d<dither>.png
        # Check for mandatory style parameters
        if 'resolution' not in params or 'rgb' not in params or 'pal' not in params or 'dither' not in params:
             raise ValueError("Missing mandatory style parameters for filename construction.")

        # Format palette size and dither method for filename string
        pal_str = str(params['pal']) if params['pal'] is not None else 'None'
        dither_str = str(params['dither']) # Assuming dither name doesn't need special encoding

        return (
            f"{params['resolution']}_{params['crop_x']}_{params['crop_y']}_s{params['scale_perc']}_r{params['rot_deg']}"
            f"_rgb{params['rgb']}_p{pal_str}_d{dither_str}.png"
        )

# --- Helper Function to Calculate Grid Coordinates ---
# Implements the 20% overlap / 80% step logic
def calculate_grid_coords(img_w: int, img_h: int, crop_w: int, crop_h: int, overlap_percentage: float = 0.20) -> list[tuple[int, int]]:
    """
    Calculates the top-left (x, y) coordinates for a grid of overlapping crops
    centered over an image. Coordinates can be negative.
    """
    if crop_w <= 0 or crop_h <= 0:
        warnings.warn(f"Invalid crop size ({crop_w}, {crop_h}). Cannot calculate grid.")
        return []

    if img_w <= 0 or img_h <= 0:
         # warnings.warn(f"Invalid image size ({img_w}, {img_h}). Cannot calculate grid.") # Too noisy for very small scaled images
         return []

    # Calculate step size based on overlap. Round to integer.
    step_x = int(crop_w * (1.0 - overlap_percentage))
    step_y = int(crop_h * (1.0 - overlap_percentage))

    # Ensure step size is at least 1 to avoid infinite loops or zero steps on large images
    step_x = max(1, step_x)
    step_y = max(1, step_y)

    # Calculate the number of steps needed to cover the image dimensions.
    # Use ceil to ensure full coverage, even partial steps.
    # Ensure at least one step even for images smaller than the step size.
    num_steps_x = max(1, math.ceil(img_w / step_x))
    num_steps_y = max(1, math.ceil(img_h / step_y))

    # Calculate the total size of the grid based on the number of steps and crop size.
    # This determines the canvas size if the grid were fully laid out.
    total_grid_w = (num_steps_x - 1) * step_x + crop_w
    total_grid_h = (num_steps_y - 1) * step_y + crop_h

    # Calculate the offset needed to center the image within this total grid size.
    # This determines the starting point (top-left of the first crop).
    start_offset_x = (total_grid_w - img_w) // 2
    start_offset_y = (total_grid_h - img_h) // 2

    coords = []
    # Generate the top-left coordinates for each cell in the grid
    for i in range(num_steps_x):
        # x = i * step_x - start_offset_x # This calculates the coordinate relative to the *image origin*
        # Correction: Grid calculation should yield coordinates relative to the transformed image origin.
        # The first crop starts at -start_offset_x relative to the image origin.
        # Each subsequent crop is step_x away.
        x = i * step_x - start_offset_x
        for j in range(num_steps_y):
            y = j * step_y - start_offset_y
            coords.append((x, y))

    return coords
   
def create_binary_image(image):
    """Create a binary (1-bit) mask image of the same size as the original image."""
    binary_image = Image.new('1', image.size, 1)  # 1 represents white
    return binary_image

def find_valid_positions_binary_image(binary_image, crop_width, crop_height):
    """Find compact valid positions for cropping based on the binary mask."""
    a = np.array(binary_image)
    valid_positions = []
    for y in range(a.shape[0] - crop_height + 1):
        x_start = None
        width = 0
        for x in range(a.shape[1] - crop_width + 1):
            if (
                a[y, x] == 1 and
                a[y, x + crop_width - 1] == 1 and
                a[y + crop_height - 1, x] == 1 and
                a[y + crop_height - 1, x + crop_width - 1] == 1
            ):
                if x_start is None:
                    x_start = x
                width += 1
            elif x_start is not None:
                break
        if x_start is not None:
            valid_positions.append((y, x_start, width))
    return valid_positions

# --- Helper to get output path ---
# This version constructs the full path including the original image subdirectory
def get_output_path(dest_dir: str, split: str, original_base_filename_without_ext: str, filename: str) -> str:
    """
    Constructs the full output path for a generated file within the new structure.
    Ensures the parent directory (original_base_filename_without_ext) exists.
    """
    # The structure is dest_dir / split / original_base_filename_without_ext / filename
    split_dir = os.path.join(dest_dir, split)
    img_subdir = os.path.join(split_dir, original_base_filename_without_ext)
    os.makedirs(img_subdir, exist_ok=True) # Ensure the subdirectory exists
    return os.path.join(img_subdir, filename)

# --- Helper Functions (outside the class for multiprocessing workers) ---
def _scan_image_params_task(image_path, crop_w, crop_h, rot_deg, ds_perc, discard_black_pil_func, image_sizes_dict, get_crop_and_pad_func, verbose):
    """
    Scans a single image file for valid crop locations with specific parameters.
    Optimized to skip unnecessary processing if the downscaled image is smaller than the crop size.
    """
    valid_coords_list = []
    total_coords_count = 0

    try:
        # Get the original image dimensions
        original_width, original_height = image_sizes_dict.get(image_path, (None, None))
        if original_width is None or original_height is None:
            warnings.warn(f"Could not get size for image {image_path}. Skipping scan task.")
            return (image_path, crop_w, crop_h, rot_deg, ds_perc, [], 0)

        # Calculate the dimensions of the downscaled image
        if ds_perc > 0 and ds_perc < 100:
            scaled_width = int(original_width * (ds_perc / 100))
            scaled_height = int(original_height * (ds_perc / 100))
        else:
            scaled_width, scaled_height = original_width, original_height

        # Skip processing if the downscaled image is smaller than the crop size
        if scaled_width < crop_w or scaled_height < crop_h:
            if verbose >= 2:
                print(f"Skipping downscale {ds_perc}% for {image_path}: Scaled size ({scaled_width}x{scaled_height}) is smaller than crop size ({crop_w}x{crop_h}).")
            return (image_path, crop_w, crop_h, rot_deg, ds_perc, [], 0)

        # Open the image and convert to RGB
        with Image.open(image_path) as img_pil_full:
            img_pil_full = img_pil_full.convert("RGB")

            # Apply rotation
            if rot_deg != 0:
                img_rotated_pil = apply_rotation(img_pil_full, rot_deg, supersample_factor=1, pil_filter=Image.Resampling.NEAREST) # No need for AA for simply detecting the valid crop areas
            else:
                img_rotated_pil = img_pil_full.copy()

            # Apply downscaling
            if ds_perc > 0 and ds_perc < 100:
                img_scaled_pil = img_rotated_pil.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            else:
                img_scaled_pil = img_rotated_pil.copy()

            if verbose >= 2:
                print(f"Scanning image: {image_path}, scaled size: {scaled_width}x{scaled_height}, rotation: {rot_deg}, scale: {ds_perc}%")

            # Calculate possible crop positions
            for crop_y in range(0, scaled_height - crop_h + 1, crop_h):
                for crop_x in range(0, scaled_width - crop_w + 1, crop_w):
                    total_coords_count += 1

                    # Get the crop
                    crop_pil = get_crop_and_pad_func(img_scaled_pil, crop_x, crop_y, crop_w, crop_h)

                    # Check if the crop is pure black
                    if not discard_black_pil_func(crop_pil):
                        valid_coords_list.append((crop_x, crop_y))
                    elif verbose >= 2:
                        print(f"Invalid crop at ({crop_x}, {crop_y}): Pure black")

    except Exception as e:
        warnings.warn(f"Error scanning image {image_path} with params (w={crop_w}, h={crop_h}, r={rot_deg}, s={ds_perc}): {e}")

    if verbose >= 1:
        print(f"Found {len(valid_coords_list)} valid crops out of {total_coords_count} total for image: {image_path} with ds {ds_perc}% and rot {rot_deg} degrees.")
    return (image_path, crop_w, crop_h, rot_deg, ds_perc, valid_coords_list, total_coords_count)

def save_single_target_worker(target_spec, crop_w, crop_h, dest_dir, split_source, image_sizes_dict, base_filenames_dict):
    """
    Worker function to load original image, apply pre-processing (rot, scale),
    extract crop, and save the target file.
    """
    global stop_processing
    if stop_processing: return (target_spec, False, "")
    # target_spec: (img_path, crop_x, crop_y, rot_deg, ds_perc)
    img_path, crop_x, crop_y, rot_deg, ds_perc = target_spec
    original_base_filename = base_filenames_dict.get(img_path)
    if not original_base_filename: return (target_spec, False, "Missing base filename in worker dict")
    img_size = image_sizes_dict.get(img_path)
    if not img_size: return (target_spec, False, "Missing image size in worker dict") # Should be in dict if img_path was processed

    try:
        # Load the original full resolution image
        # Use a context manager to ensure the file is closed
        with Image.open(img_path) as img_pil_full:
            img_pil_full = img_pil_full.convert("RGB")

            # Apply rotation (pre-processing)
            if rot_deg != 0:
                rotated_img_pil = apply_rotation(img_pil_full, rot_deg, supersample_factor=2)
            else:
                rotated_img_pil = img_pil_full.copy() # Work on a copy if no rotation

            # Apply downscaling (pre-processing)
            if ds_perc > 0 and ds_perc < 100:
                # apply_downscaling handles the percentage validation internally
                scaled_img_pil = apply_downscaling(rotated_img_pil, ds_perc)
            else: # ds_perc == 0 or 100 (100% scale) means no downscaling
                scaled_img_pil = rotated_img_pil.copy() # Work on a copy if no scaling

            # Extract crop with padding from the scaled image
            # get_crop_and_pad handles potentially negative coords and padding
            crop_pil = get_crop_and_pad(scaled_img_pil, crop_x, crop_y, crop_w, crop_h) # Use global crop_w, crop_h from generator args

            # Construct the output path for the target file
            target_params_for_filename = {
                'crop_x': crop_x, 'crop_y': crop_y,
                'scale_perc': ds_perc, 'rot_deg': rot_deg
            }
            target_filename = construct_filename(target_params_for_filename, is_target=True)
            output_path = get_output_path(dest_dir, split_source, original_base_filename, target_filename)

            # Save the target file
            # Use a high quality PNG save
            crop_pil.save(output_path, format='PNG', quality=100)

            # Return success status and the original target spec
            return (target_spec, True, "")

    except Exception as e:
        # Return failure status, original target spec, and error message
        return (target_spec, False, str(e))

# --- Helper to Parse Filenames ---
# Reads the filename format: <resolution>_<X>_<Y>_s<scale>_r<rot>_rgb<rgb>_p<pal>_d<dither>.png
# And the target format: target_<X>_<Y>_s<scale>_r<rot>.png
def parse_generated_filename(filename: str, verbose: int = 1) -> dict | None:
    """
    Parses a generated filename to extract its components.
    Handles both styled output files and target files based on the defined encoding.
    """
    name, ext = os.path.splitext(filename)
    if ext.lower() != '.png':
        return None # Only process PNGs

    # Attempt to parse as a target file first: target_<X>_<Y>_s<scale>_r<rot>.png
    # Using regex with named capture groups for clarity
    target_match = re.match(r'^target_(?P<crop_x>-?\d+)_(?P<crop_y>-?\d+)_s(?P<scale_perc>\d+)_r(?P<rot_deg>\d+)$', name)
    if target_match:
        try:
            # Extract matched groups and convert to appropriate types
            crop_x = int(target_match.group('crop_x'))
            crop_y = int(target_match.group('crop_y'))
            scale_perc = int(target_match.group('scale_perc'))
            rot_deg = int(target_match.group('rot_deg'))

            # Return a dictionary representing the parsed target file parameters
            return {
                'type': 'target', # File type
                'crop_x': crop_x,
                'crop_y': crop_y,
                'scale_perc': scale_perc, # Pre-processing scale percentage
                'rot_deg': rot_deg,     # Pre-processing rotation angle
                # Style parameters are None for target files
                'resolution': None, 'rgb': None, 'pal': None, 'dither': None,
                'full_filename': filename # Store the original filename
            }
        except ValueError:
             # This should ideally not happen if the regex groups match, but handle for safety
             warnings.warn(f"ValueError during parsing target filename: {filename}. Skipping.")
             return None


    # Attempt to parse as a styled file: <resolution>_<X>_<Y>_s<scale>_r<rot>_rgb<rgb>_p<pal>_d<dither>.png
    # Using regex with named capture groups
    style_match = re.match(r'^(?P<resolution>\w+)_(-?\d+)_(-?\d+)_s(?P<scale_perc>\d+)_r(?P<rot_deg>\d+)_rgb(?P<rgb_val>\d+)_p(?P<pal_str>\w+)_d(?P<dither_name>[\w-]+)$', name)
    if style_match:
        try:
            # Extract matched groups
            resolution = style_match.group('resolution')
            crop_x = int(style_match.group(2)) # Group 2 is the first (-?\d+) after resolution
            crop_y = int(style_match.group(3)) # Group 3 is the second (-?\d+)
            scale_perc = int(style_match.group('scale_perc'))
            rot_deg = int(style_match.group('rot_deg'))
            rgb_val = int(style_match.group('rgb_val'))
            pal_str = style_match.group('pal_str')
            dither_name = style_match.group('dither_name')

            # Convert palette string 'None' to actual None object
            pal = int(pal_str) if pal_str.lower() != 'none' else None
            if dither_name == "none": dither_name = "None"

            # Perform basic validation on parsed values against supported constants
            if resolution not in SUPPORTED_RESOLUTION_STYLES:
                 if verbose >= 2: warnings.warn(f"Unsupported resolution '{resolution}' in filename: {filename}. Skipping.")
                 return None
            if dither_name not in SUPPORTED_DITHER_METHODS:
                 if verbose >= 2: warnings.warn(f"Unsupported dither method '{dither_name}' in filename: {filename}. Skipping.")
                 return None
            
            # Add more validation here for rgb_val and pal if needed
            # Return a dictionary representing the parsed styled file parameters
            return {
                'type': 'style', # File type
                'crop_x': crop_x,
                'crop_y': crop_y,
                'scale_perc': scale_perc, # Pre-processing scale percentage
                'rot_deg': rot_deg,     # Pre-processing rotation angle
                'resolution': resolution, # Resolution style
                'rgb': f"RGB{rgb_val}", # Color format value
                'pal': pal,             # Palette size (int or None)
                'dither': dither_name,  # Dither method name
                'full_filename': filename # Store the original filename
            }
        except ValueError:
             # Handle errors during integer conversion for crop_x, crop_y, scale_perc, rot_deg, rgb_val, pal_str
             warnings.warn(f"ValueError during parsing styled filename: {filename}. Skipping.")
             return None
        except Exception as e:
             # Catch any other unexpected errors during parsing
             warnings.warn(f"Unexpected error parsing styled filename {filename}: {e}. Skipping.")
             return None


    # If the filename didn't match either expected target or styled format
    if verbose >= 2: warnings.warn(f"Filename did not match expected format: {filename}. Skipping.")
    return None

# Worker Function for Generating and Saving Styled Output
def generate_and_save_styled_worker(styled_spec, crop_w_worker, crop_h_worker, dest_dir_worker, split_source_worker, image_sizes_dict_worker, base_filenames_dict_worker, palette_algorithm, verbose_worker):
    """Worker function to generate and save a single styled output file."""
    img_path, crop_x, crop_y, rot_deg, ds_perc, cs, pal, dm, res = styled_spec
    original_base_filename = base_filenames_dict_worker.get(img_path)
    if not original_base_filename: return (styled_spec, False, "Missing base filename in worker dict")

    global stop_processing
    if stop_processing: return (styled_spec, False, "")

    # Helper function for debug prints (keep this if you want detailed type tracing)
    def print_image_info(img_obj, step_name, spec_info):
        if verbose_worker >= 3:
            obj_type = type(img_obj).__name__
            if isinstance(img_obj, Image.Image):
                info = f"Type: PIL.Image.Image, Mode: {img_obj.mode}, Size: {img_obj.size}"
            elif isinstance(img_obj, np.ndarray):
                info = f"Type: numpy.ndarray, Shape: {img_obj.shape}, Dtype: {img_obj.dtype}"
            else:
                info = f"Type: {obj_type}"
            print(f"DEBUG WORKER [{spec_info}]: After {step_name}: {info}")

    # Create a string representation of the spec for debug prints
    spec_info_str = f"{res}_{cs}_{pal}_{dm}_{crop_x}_{crop_y}_{rot_deg}_{ds_perc}% from {os.path.basename(img_path)}"

    try:
        with Image.open(img_path) as img_pil_full:
            img_pil_full = img_pil_full.convert("RGB") # Ensure RGB mode
            # print_image_info(img_pil_full, "Initial Load/Convert", spec_info_str) # Uncomment for verbosity

            # NOTE: Amiga games have a mix of images which use Anti-Aliasing, not clear yet if styled images should use AA or not.
            #       For maximum quality, we should combine the rotation and downscaling, so the downscale of the AA does the final downscale in one step.

            # Apply pre-processing (rotation, scaling, cropping) resulting in a PIL Image
            if rot_deg != 0: rotated_img_pil = apply_rotation(img_pil_full, rot_deg, supersample_factor=2) # MAYBE SET TO 1 and USE Image.Resampling.NEAREST
            else: rotated_img_pil = img_pil_full.copy()
            # print_image_info(rotated_img_pil, "Rotation", spec_info_str) # Uncomment for verbosity

            if ds_perc > 0 and ds_perc < 100: scaled_img_pil = apply_downscaling(rotated_img_pil, ds_perc)
            else: scaled_img_pil = rotated_img_pil.copy()
            # print_image_info(scaled_img_pil, "Scaling", spec_info_str) # Uncomment for verbosity

            crop_pil = get_crop_and_pad(scaled_img_pil, crop_x, crop_y, crop_w_worker, crop_h_worker)
            # print_image_info(crop_pil, "Cropping/Padding", spec_info_str) # Uncomment for verbosity

            # Apply resolution style (expects PIL, assuming returns PIL)
            # Error 'mode' was previously reported here if the input was wrong.
            try:
                processed_res_pil = pre_apply_resolution_style(crop_pil, res)
                if verbose_worker >= 3: print_image_info(processed_res_pil, "Resolution Styling", spec_info_str)
            except Exception as e:
                # This is a likely spot for the 'mode' error if the input was wrong or the function returned NumPy
                if verbose_worker >= 1: warnings.warn(f"Worker error during resolution styling for {spec_info_str}: {e}", stacklevel=2)
                return (styled_spec, False, f"Resolution styling failed: {e}")
            
            # --- Convert PIL Image to NumPy array for quantization/dithering ---
            # The reduce_color_depth_and_dither function expects NumPy.
            try:
                processed_res_np = np.array(processed_res_pil)
                if verbose_worker >= 3: print_image_info(processed_res_np, "PIL to NumPy (before quantization)", spec_info_str)
            except Exception as e:
                if verbose_worker >= 1: warnings.warn(f"Worker error converting to NumPy before quantization for {spec_info_str}: {e}", stacklevel=2)
                return (styled_spec, False, f"Failed conversion to NumPy: {e}")


            # --- Apply quantization/palette/dither using the reduce_color_depth_and_dither function ---
            # This function is designed to expect NumPy and return NumPy.
            try:
                # Arguments for reduce_color_depth_and_dither:
                # image_np, color_space, target_palette_size=None, dithering_method='None', verbose=1

                color_space_str = cs # Already a string like 'RGB888', 'RGB444'
                palette_size_param = pal # Already int or None

                # --- Correctly map the dither parameter (dm) to the string 'None' if it's None ---
                # The styled_spec might contain None object or the string 'None'.
                # The reduce_color_depth_and_dither function expects a string like 'None'.
                dithering_method_param = 'None' if (dm is None or (isinstance(dm, str) and dm.lower() == 'None')) else dm.lower()

                if verbose_worker >= 3: print(f"DEBUG WORKER [{spec_info_str}]: Calling reduce_color_depth_and_dither with color_space='{color_space_str}', palette={palette_size_param}, dither='{dithering_method_param}'.")

                # Call the function from quantize.py (ensure you have the import)
                processed_quantized_np = reduce_color_depth_and_dither(
                    image_np=processed_res_np, # Pass the NumPy array input
                    color_space=color_space_str, # Pass the color space string
                    target_palette_size=palette_size_param, # Pass the palette size (int or None)
                    dithering_method=dithering_method_param, # Pass the corrected dither method string ('None' or a DIFFUSION_MAPS key)
                    palette_algorithm=palette_algorithm, # Pass the palette algorithm
                    verbose=verbose_worker >= 2 # Pass verbosity level
                )
                if verbose_worker >= 3: print_image_info(processed_quantized_np, "After reduce_color_depth_and_dither", spec_info_str)

            except Exception as e:
                # This catches errors specifically from the reduce_color_depth_and_dither call.
                # The 'mode' error is reported as happening here.
                if verbose_worker >= 1: warnings.warn(f"Worker error during quantization/dither ({color_space_str}, {palette_size_param}, {dithering_method_param}) for {spec_info_str}: {e}", stacklevel=2)
                return (styled_spec, False, f"Quantization/dither failed: {e}")


            # --- Convert NumPy array back to PIL Image for resolution styling ---
            try:
                 processed_quantized_pil = Image.fromarray(processed_quantized_np)
                 if verbose_worker >= 3: print_image_info(processed_quantized_pil, "NumPy to PIL (before resolution style)", spec_info_str)
            except Exception as e:
                 if verbose_worker >= 1: warnings.warn(f"Worker error converting back to PIL after quantization for {spec_info_str}: {e}. Original was type {type(processed_quantized_np).__name__}.", stacklevel=2)
                 return (styled_spec, False, f"Failed conversion after quantization: {e}")

            # --- Post apply resolution style ---
            post_quantized_pil = post_apply_resolution_style(processed_quantized_pil, res)

            # The result before inference is now expected to be a PIL Image
            final_output_obj = post_quantized_pil.copy() # Make a copy to be safe
            if verbose_worker >= 3: print_image_info(final_output_obj, "Before Inference (copy)", spec_info_str)
            
            # --- Save the final output image ---
            # The object to save is final_output_obj. It should be a PIL Image at this point.
            styled_params_for_filename = {
                'crop_x': crop_x, 'crop_y': crop_y, 'scale_perc': ds_perc, 'rot_deg': rot_deg,
                'rgb': int(cs.replace('RGB', '')), 'pal': pal, 'dither': dm, 'resolution': res
            }
            styled_filename = construct_filename(styled_params_for_filename, is_target=False)
            output_path = get_output_path(dest_dir_worker, split_source_worker, original_base_filename, styled_filename)

            # Final check and conversion attempt before saving
            # The 'mode' error could occur here if final_output_obj is unexpectedly NumPy
            if not isinstance(final_output_obj, Image.Image):
                 if verbose_worker >= 1: warnings.warn(f"Final output object for {spec_info_str} is not a PIL Image (it's {type(final_output_obj).__name__}). Attempting conversion before saving.", stacklevel=2)
                 try:
                      final_output_pil_for_save = Image.fromarray(final_output_obj)
                      if verbose_worker >= 3: print_image_info(final_output_pil_for_save, "Conversion to PIL before saving", spec_info_str)
                 except Exception as e:
                      # If conversion fails here, we can't save.
                      if verbose_worker >= 1: warnings.warn(f"Failed to convert final output object to PIL before saving for {spec_info_str}: {e}. Skipping save.", stacklevel=2)
                      return (styled_spec, False, f"Failed conversion before saving: {e}")
            else:
                 # If it's already a PIL Image, just use it
                 final_output_pil_for_save = final_output_obj

            # --- The .save() method expects a PIL Image ---
            # Error 'mode' could also originate from inside save() if it receives wrong type
            try:
                final_output_pil_for_save.save(output_path, format='PNG', quality=100)
                if verbose_worker >= 3: print(f"DEBUG WORKER [{spec_info_str}]: Successfully saved {output_path}")
            except Exception as e:
                 if verbose_worker >= 1: warnings.warn(f"Worker error saving styled output {output_path} for {spec_info_str}: {e}", stacklevel=2)
                 return (styled_spec, False, f"Save failed: {e}")


            return (styled_spec, True, "") # Success result tuple

    except Exception as e:
        # This is the general exception catcher for any unhandled errors in the try block.
        # The line number in the UserWarning points to the warnings.warn call below.
        # The error message "'numpy.ndarray' object has no attribute 'mode'" comes from 'message'.
        message = str(e)
        if verbose_worker >= 1:
             warnings.warn(f"Failed to generate styled output {styled_spec[8]}_{styled_spec[1]}_{styled_spec[2]} (Spec: {styled_spec}). Full Error: {message}", stacklevel=2)
        return (styled_spec, False, message)

# --- Main Generator Class ---
class DatasetGenerator:
    def __init__(self, args):
        # Store arguments and initialize instance variables
        self.args = args
        self.train_images_dir = args.train_images
        self.test_images_dir = args.test_images
        self.dest_dir = args.destination_dir
        self.crop_w, self.crop_h = args.crop_size
        self.train_num_crops = args.train_num_crops
        self.test_num_crops = args.test_num_crops
        self.model_path = args.model_path
        self.max_workers = args.max_workers if args.max_workers is not None else os.cpu_count()
        self.verbose = args.verbose
        self.stop_requested = False

        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Initialize caches (train and test)
        self.train_cache = None
        self.test_cache = None

        # Use the provided cache file paths, if any
        train_cache_path = args.train_cache_file
        test_cache_path = args.test_cache_file

        # If no cache path is specified for a split, caching is disabled for that split.
        # We instantiate the ScanCache even if the path is None; the class handles it.
        self.train_cache = ScanCache(cache_path=train_cache_path, verbose=self.verbose)
        self.test_cache = ScanCache(cache_path=test_cache_path, verbose=self.verbose)

        # --- Internal State (populated by methods) ---
        self.train_image_paths = []
        self.test_image_paths = []
        self.image_sizes = {} # {img_path: (w, h)}
        self.base_filenames = {} # {img_path: base_filename_without_ext}

        self.possible_target_crop_operations = {'train': [], 'test': []} # List of (img_path, x, y, rot, scale)
        self.active_style_characteristics = set() # Set of (cs, ps, dm) tuples
        self.active_style_combinations = set() # Set of (res, cs, ps, dm) tuples
        self.requested_resolutions = []

        self.full_valid_target_specs = {'train': set(), 'test': set()} # Set of (img_path, x, y, rot, scale)
        self.full_valid_output_specs = {'train': set(), 'test': set()} # Set of (img_path, x, y, rot, scale, cs, ps, dm, res)

        self.existing_target_specs = {'train': set(), 'test': set()}
        self.existing_output_specs = {'train': set(), 'test': set()}
        self.invalid_files = {'train': [], 'test': []} # Files on disk that don't match valid specs

        # These track what needs generation *after* considering existing files AND shrinking
        self.final_to_generate_target_specs = {'train': set(), 'test': set()}
        self.final_to_generate_output_specs = {'train': set(), 'test': set()}

        # --- Validation and Setup ---
        self._validate_args()
        os.makedirs(self.dest_dir, exist_ok=True)
        self.model = load_model(self.model_path) if self.model_path else None

    def _handle_interrupt(self, signum, frame):
        """
        Signal handler for CTRL-C (SIGINT).
        Sets a flag to stop the process and performs cleanup.
        """        
        global stop_processing
        stop_processing = True
        if self.verbose >= 1 and not self.stop_requested:
            print("\nCTRL-C detected. Stopping generation process...")
        self.stop_requested = True

    def _load_image_paths(self):
        """
        Loads all PNG image paths from the train and test directories into instance variables.
        Also populates self.image_sizes with the dimensions of each image and self.base_filenames with base filenames.
        """
        if self.verbose >= 1:
            print("Loading image paths from train and test directories...")

        # Load train image paths and sizes
        self.train_image_paths = []
        if self.train_images_dir is not None:
            for root, _, files in os.walk(self.train_images_dir):  # Recursively traverse train directory
                for f in files:
                    if f.lower().endswith(".png"):
                        img_path = os.path.join(root, f)
                        self.train_image_paths.append(img_path)
                        base_filename = os.path.splitext(os.path.basename(img_path))[0]
                        self.base_filenames[img_path] = base_filename
                        try:
                            with Image.open(img_path) as img:
                                self.image_sizes[img_path] = img.size  # Store (width, height)
                        except Exception as e:
                            warnings.warn(f"Failed to load image size for {img_path}: {e}")

        if self.verbose >= 2:
            print(f"Loaded {len(self.train_image_paths)} train images.")
            print(f"Train image sizes: {list(self.image_sizes.items())[:5]}")  # Print first 5 for debugging
            print(f"Train base filenames: {list(self.base_filenames.items())[:5]}")  # Print first 5 for debugging

        # Load test image paths and sizes (if test directory is provided)
        if self.test_images_dir:
            self.test_image_paths = []
            for root, _, files in os.walk(self.test_images_dir):  # Recursively traverse test directory
                for f in files:
                    if f.lower().endswith(".png"):
                        img_path = os.path.join(root, f)
                        self.test_image_paths.append(img_path)
                        base_filename = os.path.splitext(os.path.basename(img_path))[0]
                        self.base_filenames[img_path] = base_filename
                        try:
                            with Image.open(img_path) as img:
                                self.image_sizes[img_path] = img.size  # Store (width, height)
                        except Exception as e:
                            warnings.warn(f"Failed to load image size for {img_path}: {e}")

            if self.verbose >= 2:
                print(f"Loaded {len(self.test_image_paths)} test images.")
                print(f"Test image sizes: {list(self.image_sizes.items())[:5]}")  # Print first 5 for debugging
                print(f"Test base filenames: {list(self.base_filenames.items())[:5]}")  # Print first 5 for debugging

    def _validate_args(self):
        # Encapsulate argument validation logic here
        if self.train_images_dir and not os.path.isdir(self.train_images_dir):
            raise FileNotFoundError(f"Training images directory not found: {self.train_images_dir}")
        if self.test_images_dir and not os.path.isdir(self.test_images_dir):
            raise FileNotFoundError(f"Test images directory not found: {self.test_images_dir}")
        if self.crop_w <= 0 or self.crop_h <= 0:
            raise ValueError(f"Invalid crop size ({self.crop_w}, {self.crop_h}).")
        if self.train_num_crops < 0:
            raise ValueError("--train_num_crops cannot be negative.")
        if self.test_num_crops < 0:
            raise ValueError("--test_num_crops cannot be negative.")

        # Validate --downscale values
        self.valid_downscales = [0]
        if self.args.downscale is not None:
            for ds_perc in self.args.downscale:
                if not isinstance(ds_perc, int) or ds_perc <= 0 or ds_perc >= 100:
                    warnings.warn(f"Invalid downscale percentage ignored: {ds_perc}. Must be an integer > 0 and < 100.")
                else:
                    self.valid_downscales.append(ds_perc)
        self.valid_downscales = sorted(list(set(self.valid_downscales)))

        # Validate --rotate values
        self.valid_rotations = [0]
        if self.args.rotate is not None:
             for rot_deg in self.args.rotate:
                 if not isinstance(rot_deg, int):
                      warnings.warn(f"Invalid rotation angle ignored: {rot_deg}. Must be an integer.")
                 else:
                      self.valid_rotations.append(rot_deg % 360)
        self.valid_rotations = sorted(list(set(self.valid_rotations)))

        # Validate --rgb, --palette, --resolution and determine active style combinations
        self._determine_active_style_combinations()

    def _determine_active_style_combinations(self):
        # Encapsulate logic to determine active style combinations from args
        # Sets self.active_style_characteristics and self.active_style_combinations

        if self.verbose >= 1: print("Determining active style combinations...")

        # Assume SUPPORTED_RGB_FORMATS and SUPPORTED_DITHER_METHODS are accessible
        # from .quantize import SUPPORTED_DITHER_METHODS, SUPPORTED_RGB_FORMATS # Example if needed

        # Define supported values based on your quantize.py
        supported_rgb_formats = [888, 555, 565, 444, 666] # As per your quantize.py code
        supported_palette_sizes = [0, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096] # 0 means all colours
        # Get supported dither methods keys + 'None' from your quantize.py DIFFUSION_MAPS
        SUPPORTED_DITHER_METHODS_KEYS = list(DIFFUSION_MAPS.keys()) + ['None', 'checkerboard', 'bayer2x2', 'bayer4x4', 'bayer8x8']


        # --- 1. Determine requested values from args, with validation and defaults ---

        requested_rgb_formats = []
        if self.args.rgb is not None: # Check if arg was provided
            for rgb_val in self.args.rgb:
                if not isinstance(rgb_val, int) or rgb_val not in supported_rgb_formats:
                    warnings.warn(f"Unsupported RGB format ignored: {rgb_val}. Supported: {supported_rgb_formats}.")
                else:
                    requested_rgb_formats.append(rgb_val)
        # Default to 888 if none specified or none were valid
        if not requested_rgb_formats:
             if self.verbose >= 2: print("Debug: No valid RGB formats specified, defaulting to 888.")
             requested_rgb_formats = [888]
        requested_rgb_formats = sorted(list(set(requested_rgb_formats))) # Remove duplicates and sort


        requested_palette_sizes = []
        if self.args.palette is not None: # Check if arg was provided
            for pal_size in self.args.palette:
                if not isinstance(pal_size, int) or pal_size not in supported_palette_sizes:
                    warnings.warn(f"Unsupported palette size ignored: {pal_size}. Supported: {supported_palette_sizes}.")
                else:
                    if pal_size == 0:
                        requested_palette_sizes.append(None)
                    else:
                        requested_palette_sizes.append(pal_size)
      
        # Determine requested dithering methods (strings, validated)
        requested_dither_methods = []
        if self.args.dither is not None: # Check if arg was provided
            # Map 'None' string or None object to the string 'None' and validate
            for d in self.args.dither:
                method_str = None # Use a temporary variable for lowercasing
                if d == "None":
                    method_str = d
                elif isinstance(d, str):
                    method_str = d.lower()
                elif d is None:
                    method_str = 'None' # Explicitly map None object to 'None' string
                else:
                    warnings.warn(f"Invalid type for dithering method: {type(d).__name__}. Skipping value: {d}")
                    continue # Skip invalid type

                # Check if the lowercased method string is supported
                if method_str in SUPPORTED_DITHER_METHODS_KEYS:
                    requested_dither_methods.append(method_str)
                else:
                    warnings.warn(f"Unsupported dithering method specified: '{d}'. Supported: {SUPPORTED_DITHER_METHODS_KEYS}. Skipping.")
                    continue # Skip this unsupported method

        # Default to 'None' if none specified or none were valid
        if not requested_dither_methods:
            if self.verbose >= 2 and self.args.dither is not None: # Only print debug if arg was provided but invalid
                 print("Debug: No valid dithering methods specified via --dither. Defaulting to 'None'.")
            elif self.verbose >= 2 and self.args.dither is None: # Print debug if arg was not provided
                 print("Debug: --dither not specified. Defaulting dithering to 'None'.")
            requested_dither_methods = ['None']

        requested_dither_methods = sorted(list(set(requested_dither_methods))) # Remove duplicates and sort

        self.requested_resolutions = self.args.resolution if self.args.resolution is not None else ['lores'] # Default to lores
        # Assuming SUPPORTED_RESOLUTION_STYLES is accessible
        # SUPPORTED_RESOLUTION_STYLES = ['lores', 'hires', 'lores_laced', 'hires_laced'] # Example definition
        if not all(res in SUPPORTED_RESOLUTION_STYLES for res in self.requested_resolutions):
            unsupported = [res for res in self.requested_resolutions if res not in SUPPORTED_RESOLUTION_STYLES]
            raise ValueError(f"Unsupported resolution styles requested: {unsupported}. Supported: {SUPPORTED_RESOLUTION_STYLES}.")
        self.requested_resolutions = sorted(list(set(self.requested_resolutions)))


        # --- 2. Determine the set of active style characteristics (color_space, target_palette_size, dithering_method) ---
        self.active_style_characteristics = set() # Set of (cs, ps, dm) tuples
        from itertools import product # Need product for combinations

        # Case A: No palette size was requested (--palette was not used) # FIXME checkerboard should work both on palette as pure RGB
        if not requested_palette_sizes:
            if self.verbose >= 2: print("Debug: No palette sizes requested (--palette not used). Generating non-paletted outputs.")
            # Combine requested RGB formats with None palette size.
            # Dithering requires a palette, so only include 'None' dithering for non-paletted outputs.
            for rgb_val in requested_rgb_formats:
                cs_name = f'RGB{rgb_val}'                
                # For non-paletted outputs, only the 'None' dither method is valid.
                if len(requested_dither_methods) == 0 or requested_dither_methods.count("None") == 1:
                    self.active_style_characteristics.add((cs_name, None, 'None'))
                elif 'checkerboard' in requested_dither_methods:
                    self.active_style_characteristics.add((cs_name, None, 'checkerboard'))

        # Case B: Palette sizes *were* requested (--palette was used)
        else:
            if self.verbose >= 2: print(f"Debug: Palette sizes requested: {requested_palette_sizes}. Combining with RGBs and requested dithers.")
            # If palette is requested, but no RGB is requested, default RGB to 888 for palette combinations
            # The user's original logic used requested_rgb_formats if args.rgb is not None, else [888]
            # Since requested_rgb_formats defaults to [888] if args.rgb is None, we can just use requested_rgb_formats
            rgb_formats_for_palette = requested_rgb_formats

            # Combine RGB formats, requested palette sizes, AND requested dithering methods
            # This is where the --dither argument controls which dither methods are included for paletted outputs
            for cs_val, pal_size, dither_method in product(rgb_formats_for_palette, requested_palette_sizes, requested_dither_methods):
                cs_name = f'RGB{cs_val}'
                if pal_size == None and dither_method == 'checkerboard':
                    self.active_style_characteristics.add((cs_name, pal_size, 'None'))
                    continue # FIXME checkerboard should also work on pure RGB
                # Add this characteristic combination. Filtering for invalid dither/palette will happen later.
                self.active_style_characteristics.add((cs_name, pal_size, dither_method))

        if not self.active_style_characteristics:
            raise ValueError("No valid style characteristics combinations were generated from arguments.")

        # --- 3. Determine the set of active style combinations including resolution style ---
        self.active_style_combinations = set() # Set of (res, cs, ps, dm) tuples

        # Combine requested resolutions with the determined style characteristics
        for res in self.requested_resolutions:
            for cs, ps, dm in self.active_style_characteristics: # dm is already the validated string

                # --- Final Filtering of invalid combinations before adding to final set ---
                # Rule: Dithering (methods other than 'None') requires a specified palette size (ps is not None)
                if dm != 'None' and dm != 'checkerboard' and ps is None:
                     if self.verbose >= 2:
                          print(f"Debug: Skipping invalid final style combination (dither method '{dm}' requires a palette): Resolution='{res}', ColorSpace='{cs}', PaletteSize={ps}, DitherMethod='{dm}'")
                     continue # Skip this combination

                # Add the valid combination to the final set
                self.active_style_combinations.add((res, cs, ps, dm)) # dm is already the correct string ('None' or a key)

        if not self.active_style_combinations:
             raise ValueError("No valid style combinations were generated after filtering.")


        # Print the total number of active style combinations generated
        if self.verbose >= 1:
            print(f"Generated {len(self.active_style_combinations)} active style combinations.")
            if self.verbose >= 2:
                 print("  Active style combinations:")
                 # Sort for consistent output in debug prints
                 sorted_combinations = sorted(list(self.active_style_combinations))
                 for combo in sorted_combinations:
                      # Format Palette Size and Dither Method for clear printing
                      ps_str = 'None' if combo[2] is None else str(combo[2])
                      # Capitalize dither methods for print, handle 'None' string
                      dm_print_str = combo[3].capitalize() if combo[3] != 'None' else 'None'
                      print(f"    - Resolution: {combo[0]}, Color Space: {combo[1]}, Palette Size: {ps_str}, Dither Method: {dm_print_str}")

    def _scan_ground_truth(self):
        """
        Scans ground truth images for valid crop positions in parallel.
        Uses caching to avoid re-scanning unchanged images.
        """
        if self.verbose >= 1:
            print("Scanning ground truth images and calculating possible target crop operations...")

        # Combine train and test image paths for processing
        all_image_paths = {
            'train': self.train_image_paths,
            'test': self.test_image_paths if self.test_images_dir else []
        }
        
        # Prepare a ThreadPoolExecutor for parallel scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for split, image_paths in all_image_paths.items():
                for img_path in image_paths:
                    # Check cache for this image
                    cache = self.train_cache if split == 'train' else self.test_cache
                    for rot_deg in self.valid_rotations:
                        for ds_perc in self.valid_downscales:
                            if self.stop_requested: return
                            cache_key = f"{img_path}_rot{rot_deg}_ds{ds_perc}"
                            cached_data = cache.get_image_cache(cache_key)
                            if self.verbose >= 2:
                                print(f"Cached data for {img_path} (rot: {rot_deg}°, scale: {ds_perc}%) : {cached_data}")

                            # If cache is valid, skip scanning
                            if cached_data and cached_data['mtime'] == os.path.getmtime(img_path):
                                if self.verbose >= 2:
                                    print(f"Using cached data for {img_path} in {split} split.")
                                for crop_x, crop_y in cached_data['valid_crops']: 
                                    self.possible_target_crop_operations[split].append( (img_path, crop_x, crop_y, rot_deg, ds_perc) )
                                continue
                
                            # Submit a task to scan the image
                            futures.append(
                                executor.submit(
                                    _scan_image_params_task,
                                    img_path, self.crop_w, self.crop_h, rot_deg, ds_perc,
                                    should_discard_by_black_ratio, self.image_sizes, get_crop_and_pad,
                                    self.args.verbose
                                )
                            )

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                if self.stop_requested: return
                try:
                    img_path, crop_w, crop_h, rot_deg, ds_perc, valid_coords, total_coords = future.result()
                    cache_key = f"{img_path}_rot{rot_deg}_ds{ds_perc}"

                    # Determine the split (train/test) based on the image path
                    split = 'train' if img_path in self.train_image_paths else 'test'

                    # Collect valid crops for the specific img_path
                    valid_crops_for_image = []
                    for crop_x, crop_y in valid_coords:
                        self.possible_target_crop_operations[split].append(
                            (img_path, crop_x, crop_y, rot_deg, ds_perc)
                        )
                        valid_crops_for_image.append((crop_x, crop_y))

                    # Update the cache with only the valid crops for this specific img_path
                    cache = self.train_cache if split == 'train' else self.test_cache

                    cache.update_image_cache(
                        cache_key,
                        {
                            'mtime': os.path.getmtime(img_path),
                            'valid_crops': valid_crops_for_image  # Store all valid crops for this image
                        }
                    )

                    if self.verbose >= 2:
                        print(f"Processed {img_path} (rot: {rot_deg}°, scale: {ds_perc}%) with {len(valid_coords)} valid crops.")

                except Exception as e:
                    warnings.warn(f"Error processing image: {e}")

        if self.verbose >= 1:
            print(f"Found {len(self.possible_target_crop_operations['train'])} possible train target crop operations (excluding black crops).")
            if self.test_images_dir:
                print(f"Found {len(self.possible_target_crop_operations['test'])} possible test target crop operations (excluding black crops).")

    def _build_full_valid_specs(self):
        # Implements Step 3 logic: Build the Full Valid Set of Output File Specifications

        if self.verbose >= 1: print("Building full set of valid output specifications...")

        # Debug: Check active style combinations before building specs
        if self.verbose >= 2:
             print(f"Debug: Entering _build_full_valid_specs")
             print(f"Debug: Active style combinations count (from _determine_active_style_combinations): {len(self.active_style_combinations)}")
             if self.verbose >= 3:
                  print(f"Debug: Active style combinations: {self.active_style_combinations}")


        for split in ['train', 'test']:
             if self.verbose >= 2: print(f"Debug: Processing split: {split}")
             if self.verbose >= 2: print(f"Debug: Possible target operations for {split} count: {len(self.possible_target_crop_operations[split])}")
             if self.verbose >= 3:
                  print(f"Debug: First few possible target operations for {split}:")
                  # Print the first 5 target operations to see their format
                  for i, spec in enumerate(list(self.possible_target_crop_operations[split])[:5]):
                       print(f"    {spec}")

             for i, (img_path, crop_x, crop_y, rot_deg, ds_perc) in enumerate(self.possible_target_crop_operations[split]):
                 if self.stop_requested: return
                 if self.verbose >= 3 and i < 5: # Print details for first few targets being processed
                      print(f"Debug: Processing target operation {i+1} in {split}: (img: {os.path.basename(img_path)}, crop: {crop_x},{crop_y}, rot: {rot_deg}, scale: {ds_perc}%)")

                 target_spec_tuple = (img_path, crop_x, crop_y, rot_deg, ds_perc)
                 self.full_valid_target_specs[split].add(target_spec_tuple)
                 if self.verbose >= 4: print(f"Debug: Added target spec. Current {split} target specs count: {len(self.full_valid_target_specs[split])}")


                 # --- Debugging Styled Output Generation ---
                 if self.verbose >= 2:
                      if i == 0: # Print this once per split per target operation loop iteration
                           print(f"Debug: Iterating styled combinations for targets in {split}...")
                           if self.verbose >= 3: print(f"Debug: Active style combinations count: {len(self.active_style_combinations)}")


                 # Add all styled output specifications for this target crop operation
                 # This loop iterates over self.active_style_combinations
                 if self.verbose >= 3 and len(self.active_style_combinations) == 0:
                      print(f"Debug: Warning: active_style_combinations is empty. Skipping styled spec generation for this target.")


                 for j, (res, cs, ps, dm) in enumerate(self.active_style_combinations):
                     # Construct the styled spec tuple
                     styled_spec_tuple = (img_path, crop_x, crop_y, rot_deg, ds_perc, cs, ps, dm, res)
                     if self.verbose >= 3: print(f"Debug: Attempting to add styled spec {j+1}: {styled_spec_tuple}")
                     self.full_valid_output_specs[split].add(styled_spec_tuple)
                     # Debug: Print the size of the set after adding each styled spec
                     if self.verbose >= 3: print(f"Debug: Current {split} styled specs count after adding {j+1}: {len(self.full_valid_output_specs[split])}")


                 if self.verbose >= 3 and len(self.active_style_combinations) > 0:
                      if i < 5: # Print for first few targets
                           print(f"Debug: Finished adding styled specs for target operation {i+1} in {split}. Final styled specs count for this target iteration: {len(self.full_valid_output_specs[split])}")



        if self.verbose >= 1:
            print(f"Full valid train target specs: {len(self.full_valid_target_specs['train'])}")
            print(f"Full valid test target specs: {len(self.full_valid_target_specs['test'])}")
            print(f"Full valid train output specs (styled): {len(self.full_valid_output_specs['train'])}") # Still expecting this to be > 0
            print(f"Full valid test output specs (styled): {len(self.full_valid_output_specs['test'])}")
        if self.verbose >= 2: print(f"Debug: Exiting _build_full_valid_specs")

    def _scan_output_directory(self):
        # Implements Step 4 logic: Scan Output Directory and Identify Existing/Invalid Files

        if self.verbose >= 1: print(f"Scanning output directory {self.dest_dir} for existing files...")

        # Need access to train_image_paths and test_image_paths to determine the split source for existing files
        # These are instance attributes

        for split in ['train', 'test']:
            split_output_dir = os.path.join(self.dest_dir, split)
            if not os.path.isdir(split_output_dir):
                if self.verbose >= 1: print(f"Output directory for split '{split}' not found: {split_output_dir}. Skipping scan.")
                continue

            current_subdir_targets = set()
            current_subdir_styles = set() 

            for root, dirs, files in os.walk(split_output_dir):
                if self.stop_requested: return

                if root == split_output_dir:
                    # If we are at the root of the split output directory, skip it
                    continue

                # The directory name is the original base filename without extension
                original_base_filename_without_ext = os.path.basename(root)

                original_img_path = None
                split_source = None # Determine the split this directory corresponds to

                # Find the original image path corresponding to this base filename
                # Check in train images first (using instance attribute)
                train_match = [p for p in self.train_image_paths if os.path.splitext(os.path.basename(p))[0] == original_base_filename_without_ext]
                if train_match:
                     original_img_path = train_match[0]
                     split_source = 'train'
                else:
                     # Check in test images if not found in train (using instance attributes)
                     if self.test_images_dir:
                          test_match = [p for p in self.test_image_paths if os.path.splitext(os.path.basename(p))[0] == original_base_filename_without_ext]
                          if test_match:
                               original_img_path = test_match[0]
                               split_source = 'test'

                # If original_img_path is still None, the original image is missing or not from the specified input dirs.
                # Files in this directory might be considered invalid or from a previous run with different inputs.
                # For now, we'll still try to parse the filenames, but if the original image is truly gone,
                # the corresponding specs won't be in the 'full_valid_output_specs'.

                # Pre-scan for target filenames ---
                found_target_filenames_in_subdir = set()
                for f_name_pre_scan in files:
                    p_params_pre_scan = parse_generated_filename(f_name_pre_scan) # Match your existing call (no self.verbose)
                    if p_params_pre_scan and p_params_pre_scan['type'] == 'target':
                        # We only need the filename itself for checking existence later.
                        # Detailed image checks (size, openability) are handled in the main loop.
                        found_target_filenames_in_subdir.add(f_name_pre_scan)

                for filename in files:
                    if self.stop_requested: return

                    parsed_params = parse_generated_filename(filename)
                    if parsed_params:                        
                        if original_img_path:
                            try:
                                with Image.open(os.path.join(root, filename)) as img:
                                    size = img.size
                                    width, height = size
                                if width != self.crop_w or height != self.crop_h:
                                    self.invalid_files[split].append(os.path.join(root, filename))
                                    if self.verbose >= 2: print(f"Found file with incorrect size in subdirectory: {os.path.join(root, filename)}")
                                    continue
                            except Exception as e:
                                self.invalid_files[split].append(os.path.join(root, filename))
                                if self.verbose >= 2: print(f"Error opening image file {os.path.join(root, filename)}: {e}")
                                continue
                            if parsed_params['type'] == 'target':
                                target_spec = (original_img_path, parsed_params['crop_x'], parsed_params['crop_y'], parsed_params['rot_deg'], parsed_params['scale_perc'])
                                if target_spec not in self.full_valid_target_specs[split_source]:
                                    self.invalid_files[split_source].append(os.path.join(root, filename))
                                else:
                                    self.existing_target_specs[split_source].add(target_spec)

                            elif parsed_params['type'] == 'style':
                                style_spec = (original_img_path, parsed_params['crop_x'], parsed_params['crop_y'], parsed_params['rot_deg'], parsed_params['scale_perc'], parsed_params['rgb'], parsed_params['pal'], parsed_params['dither'], parsed_params['resolution'])
                                if style_spec not in self.full_valid_output_specs[split_source]:
                                    self.invalid_files[split_source].append(os.path.join(root, filename))
                                else:
                                    self.existing_output_specs[split_source].add(style_spec)
                                
                                # Check for missing target for this style
                                target_params_for_filename = {
                                    'crop_x': parsed_params['crop_x'],
                                    'crop_y': parsed_params['crop_y'],
                                    'scale_perc': parsed_params['scale_perc'],
                                    'rot_deg': parsed_params['rot_deg']
                                }
                                expected_target_filename = construct_filename(target_params_for_filename, is_target=True)
                                
                                if expected_target_filename not in found_target_filenames_in_subdir:
                                    self.invalid_files[split_source].append(os.path.join(root, filename))
                                    if self.verbose >= 2:
                                        warnings.warn(f"Styled file {filename} in {root} has no corresponding target file '{expected_target_filename}'. Marked for deletion.")

                        else:
                            self.invalid_files[split].append(os.path.join(root, filename))

                    else:
                        self.invalid_files[split].append(os.path.join(root, filename))
                        if self.verbose >= 2: print(f"Found file in subdirectory with missing original image: {os.path.join(root, filename)}")

            current_subdir_targets.clear()
            current_subdir_styles.clear() 


        if self.verbose >= 1:
            print(f"Found {len(self.existing_target_specs['train'])} existing train target crops.")
            print(f"Found {len(self.existing_output_specs['train'])} existing train styled outputs.")
            print(f"Found {len(self.existing_target_specs['test'])} existing test target crops.")
            print(f"Found {len(self.existing_output_specs['test'])} existing test styled outputs.")
            print(f"Found {len(self.invalid_files['train'])} invalid files in train output.")
            print(f"Found {len(self.invalid_files['test'])} invalid files in test output.")

    def _cleanup_invalid_files(self):
        # Implements Step 5 logic: Clean up Invalid Files (Optional)

        total_invalid_files = len(self.invalid_files['train']) + len(self.invalid_files['test'])
        if total_invalid_files > 0 and self.args.keep_invalid_files != True:
            print(f"\nFound {total_invalid_files} files in the output directory that do not match the current configuration (unparseable names, or parameters outside requested ranges).")
            for split in ['train', 'test']:
                if self.invalid_files[split]:
                        print(f"  {split.capitalize()} invalid files:")
                        for file_path in self.invalid_files[split]:
                            print(f"    {file_path}")
            response = input("Do you want to delete these invalid files? (yes/no): ").lower()
            if response == 'yes':
                if self.verbose >= 1: print("Deleting invalid files...")
                deleted_count = 0
                for split in ['train', 'test']:
                    for file_path in self.invalid_files[split]:
                        try:
                                os.remove(file_path)
                                deleted_count += 1
                                if self.verbose >= 2: print(f"Deleted: {file_path}")
                        except Exception as e:
                                warnings.warn(f"Error deleting invalid file {file_path}: {e}")
                if self.verbose >= 1: print(f"Deleted {deleted_count} invalid files.")
            else:
                if self.verbose >= 1: print("Keeping invalid files.")

    def _determine_generation_and_deletion(self):
        # Determines the final list of files to generate based on quotas and identifies existing targets for deletion.
        # This replaces the previous _determine_initial_to_generate and refines the quota application logic.
        # Populates self.final_to_generate_* sets and self._targets_to_delete_for_quota.

        if self.verbose >= 1: print("Determining final generation list and identifying targets for deletion based on quotas...")

        # Step 1: Determine all valid files that are currently missing (Initial scan results)
        initial_to_generate_target_specs = {'train': set(), 'test': set()}
        initial_to_generate_output_specs = {'train': set(), 'test': set()}

        for split in ['train', 'test']:
             # Combine the full set of valid target and styled specs for this split
             full_valid_for_split = self.full_valid_target_specs[split] | self.full_valid_output_specs[split]
             # Combine the set of existing target and styled specs for this split
             existing_for_split = self.existing_target_specs[split] | self.existing_output_specs[split]
             # This is the set of all valid files that are missing from disk
             to_generate_initially_for_split = full_valid_for_split - existing_for_split

             # Separate into target and styled
             for spec in to_generate_initially_for_split:
                  if len(spec) == 5: # Target spec tuple length is 5
                       initial_to_generate_target_specs[split].add(spec)
                  elif len(spec) == 9: # Styled spec tuple length is 9
                       initial_to_generate_output_specs[split].add(spec)
                  else:
                       # This should not happen if parsing and spec creation are correct, but add a warning.
                       warnings.warn(f"Unknown spec format found during initial determination: {spec}. Skipping.")

        if self.verbose >= 1:
            print(f"Initial 'to generate' counts (all valid missing files found before quota application):")
            print(f"  Train Targets: {len(initial_to_generate_target_specs['train'])}")
            print(f"  Test Targets: {len(initial_to_generate_target_specs['test'])}")
            print(f"  Train Styled Outputs: {len(initial_to_generate_output_specs['train'])}")
            print(f"  Test Styled Outputs: {len(initial_to_generate_output_specs['test'])}")

        # --- Step 2: Apply quotas and determine the FINAL generation list and what to delete ---

        self.final_to_generate_target_specs = {'train': set(), 'test': set()}
        self.final_to_generate_output_specs = {'train': set(), 'test': set()}
        # Initialize the set of existing targets that will need to be deleted to meet quota
        self._targets_to_delete_for_quota = {'train': set(), 'test': set()}


        for split in ['train', 'test']:
             quota = self.train_num_crops if split == 'train' else self.test_num_crops
             existing_targets = self.existing_target_specs[split]
             initially_missing_targets = initial_to_generate_target_specs[split]

             num_existing = len(existing_targets)
             num_initially_missing = len(initially_missing_targets)
             current_total_potential = num_existing + num_initially_missing


             if self.verbose >= 1: print(f"\nSplit '{split}': Existing targets = {num_existing}, Initially missing targets = {num_initially_missing}, Total potential = {current_total_potential} / Quota = {quota}")

             # Set of targets that we will ultimately keep (existing or newly generated)
             targets_to_keep = set()

             if num_existing >= quota:
                  # Case 1: Enough or more existing targets than the quota. Prioritize keeping existing.
                  if self.verbose >= 1: print(f"Split '{split}': Enough existing targets ({num_existing}) to meet or exceed quota ({quota}). Selecting {quota} existing targets to keep.")
                  # Randomly select exactly 'quota' existing targets to keep
                  existing_targets_list = list(existing_targets)
                  random.shuffle(existing_targets_list)
                  targets_to_keep = set(existing_targets_list[:quota])

                  # Targets to delete are existing ones that were NOT selected to keep
                  self._targets_to_delete_for_quota[split] = existing_targets - targets_to_keep

                  # No new targets are generated in this case (because we kept enough existing ones)
                  self.final_to_generate_target_specs[split] = set()

                  # Styled outputs to generate are initially missing ones whose target is among the kept targets
                  # These are styled outputs needed for the *kept existing* targets.
                  for s_spec in initial_to_generate_output_specs[split]:
                       target_spec = s_spec[:5] # Extract the target part of the styled spec
                       if target_spec in targets_to_keep:
                            self.final_to_generate_output_specs[split].add(s_spec)

             elif num_existing < quota:
                  # Case 2: Not enough existing targets. Keep all existing and add from missing until quota is met (or all missing are used).
                  if self.verbose >= 1: print(f"Split '{split}': Not enough existing targets ({num_existing}) for quota ({quota}). Keeping all existing and selecting from missing.")
                  # Keep all existing targets
                  targets_to_keep.update(existing_targets)

                  # Calculate how many more targets are needed from the missing ones to reach the quota
                  needed_more = quota - num_existing

                  # Select up to 'needed_more' targets from the initially missing ones
                  initially_missing_list = list(initially_missing_targets)
                  random.shuffle(initially_missing_list)
                  # Select min(needed_more, number of available missing targets)
                  newly_selected_targets = set(initially_missing_list[:min(needed_more, num_initially_missing)])

                  # Add the newly selected targets to the set of targets to keep
                  targets_to_keep.update(newly_selected_targets)

                  # The targets to generate are the ones we newly selected from the missing pool
                  self.final_to_generate_target_specs[split] = newly_selected_targets.copy()

                  # No existing targets are deleted for quota reasons in this case
                  self._targets_to_delete_for_quota[split] = set()

                  # Styled outputs to generate are initially missing ones whose target is among the kept targets
                  # (This includes styled outputs needed for both kept existing targets AND kept newly selected targets)
                  for s_spec in initial_to_generate_output_specs[split]:
                       target_spec = s_spec[:5] # Extract the target part
                       if target_spec in targets_to_keep:
                            self.final_to_generate_output_specs[split].add(s_spec)

        if self.verbose >= 1:
            print(f"\nFinal 'to generate' counts (after applying quota logic):")
            print(f"  Train Targets: {len(self.final_to_generate_target_specs['train'])}")
            print(f"  Test Targets: {len(self.final_to_generate_target_specs['test'])}")
            print(f"  Train Styled Outputs: {len(self.final_to_generate_output_specs['train'])}")
            print(f"  Test Styled Outputs: {len(self.final_to_generate_output_specs['test'])}")
            print(f"  Train Targets marked for deletion (existing files): {len(self._targets_to_delete_for_quota['train'])}")
            print(f"  Test Targets marked for deletion (existing files): {len(self._targets_to_delete_for_quota['test'])}")

    def _shrink_dataset(self):
        # Handles the deletion of files based on the targets marked for removal in _determine_generation_and_deletion.

        total_to_delete = len(self._targets_to_delete_for_quota['train']) + len(self._targets_to_delete_for_quota['test'])

        # Only prompt for deletion if there are targets marked for deletion
        if total_to_delete > 0:
             print("\nDetected existing targets exceeding the requested quota(s) that will be deleted.")
             response = input("Do you want to delete these excess targets (and their styled outputs) to meet the quotas? (yes/no): ").lower()

             if response == 'yes':
                  if self.verbose >= 1: print("Executing deletion for shrinking...")
                  deleted_count = 0

                  for split in ['train', 'test']:
                       for target_spec in self._targets_to_delete_for_quota[split]:
                            img_path, crop_x, crop_y, rot_deg, ds_perc = target_spec

                            # Find the base filename for path construction
                            original_base_filename = self.base_filenames.get(img_path)
                            if not original_base_filename:
                                 warnings.warn(f"Could not find base filename for {img_path}. Cannot delete target spec: {target_spec}")
                                 continue # Skip deletion for this target spec

                            # Determine the correct split source for path construction
                            target_split_source = 'train' if img_path in self.train_image_paths else 'test'
                            # Double-check that the target spec is indeed associated with the correct split based on original image path
                            if target_split_source != split:
                                 # This should ideally not happen if existing_target_specs are stored by split correctly
                                 warnings.warn(f"Split mismatch for target spec {target_spec}. Expected {split} based on set, found {target_split_source} based on image path. Skipping deletion.")
                                 continue


                            # --- Delete the target file ---
                            target_params_for_filename = {
                                'crop_x': crop_x, 'crop_y': crop_y, 'scale_perc': ds_perc, 'rot_deg': rot_deg
                            }
                            target_filename = construct_filename(target_params_for_filename, is_target=True)
                            # Use target_split_source for the path
                            target_file_path = get_output_path(self.dest_dir, target_split_source, original_base_filename, target_filename) # get_output_path is global
                            try:
                                 if os.path.exists(target_file_path): # Check existence before trying to delete
                                      os.remove(target_file_path)
                                      deleted_count += 1
                                      if self.verbose >= 2: print(f"Deleted target: {target_file_path}")
                                 # Remove from existing set
                                 self.existing_target_specs[target_split_source].discard(target_spec)
                            except Exception as e: warnings.warn(f"Error deleting target file {target_file_path}: {e}")


                            # --- Delete corresponding styled files ---
                            # Find all *existing* styled specs for this target spec
                            # We need to look in both existing_output_specs['train'] and ['test']
                            existing_styled_for_this_target = {
                                 s_spec for s_spec in (self.existing_output_specs['train'] | self.existing_output_specs['test'])
                                 if s_spec[:5] == target_spec # Match the target part of the styled spec
                            }

                            for styled_spec in existing_styled_for_this_target:
                                 # Construct the filename and path for the styled file
                                 style_params = {
                                     'crop_x': styled_spec[1], 'crop_y': styled_spec[2], 'rot_deg': styled_spec[3], 'scale_perc': styled_spec[4],
                                     'rgb': int(styled_spec[5].replace('RGB', '')), 'pal': styled_spec[6], 'dither': styled_spec[7], 'resolution': styled_spec[8]
                                 }
                                 styled_filename = construct_filename(style_params, is_target=False)
                                 # Use the correct split source (train/test) for the styled file's path
                                 styled_split_source = 'train' if styled_spec[0] in self.train_image_paths else 'test'
                                 # Double-check split source for path construction consistency
                                 if styled_split_source != (split if styled_spec[0] in (self.train_image_paths + self.test_image_paths) else 'unknown_split'):
                                      warnings.warn(f"Split mismatch for styled spec {styled_spec}. Expected {split} based on target set, found {styled_split_source} based on image path. Skipping deletion.")
                                      continue # Skip deletion for this styled spec

                                 styled_file_path = get_output_path(self.dest_dir, styled_split_source, original_base_filename, styled_filename) # get_output_path is global

                                 try:
                                     if os.path.exists(styled_file_path):
                                          os.remove(styled_file_path)
                                          deleted_count += 1
                                          if self.verbose >= 2: print(f"Deleted styled: {styled_file_path}")
                                     # Remove from existing set
                                     self.existing_output_specs[styled_split_source].discard(styled_spec)
                                 except Exception as e: warnings.warn(f"Error deleting styled file {styled_file_path}: {e}")


                  if self.verbose >= 1: print(f"Finished shrinking. Deleted {deleted_count} files total.")

             else:
                  if self.verbose >= 1:
                       print("Shrinking needed but skipped based on user input. Quota will not be met.")

    def _run_generation_phases(self):
        # Implements Step 8 Logic: The Generation Loop
        # Uses the sets that have been potentially modified by _shrink_dataset.

        # The sets to generate in this phase are the initial missing sets,
        # modified by the shrinking process (if it occurred).
        # Let's use local variables for clarity, based on the instance attributes.
        final_to_generate_target_specs = self.final_to_generate_target_specs
        final_to_generate_output_specs = self.final_to_generate_output_specs

        targets_to_save_list = list(final_to_generate_target_specs['train']) + list(final_to_generate_target_specs['test'])
        styled_to_generate_list = list(final_to_generate_output_specs['train']) + list(final_to_generate_output_specs['test'])

        targets_to_save_count = len(targets_to_save_list)
        styled_to_generate_count = len(styled_to_generate_list)

        if targets_to_save_count == 0 and styled_to_generate_count == 0:
            if self.verbose >= 1: print("\nNo files need generation after considering existing files and shrinking.")
            # Check quotas based on remaining existing targets
            targets_remaining = {'train': len(self.existing_target_specs['train']),
                                   'test': len(self.existing_target_specs['test'])}
            if targets_remaining['train'] < self.train_num_crops or \
               (self.test_images_dir and targets_remaining['test'] < self.test_num_crops):
                 warnings.warn(f"Cannot meet requested quotas ({self.train_num_crops} train, {self.test_num_crops} test). Only {targets_remaining['train']} train and {targets_remaining['test']} test targets remain after shrinking.")
            return # Exit if nothing needs generating after shrinking

        if self.verbose >= 1: print(f"\nStarting generation.")
        if targets_to_save_count > 0: print(f"Phase 1: Saving {targets_to_save_count} target crops.")
        if styled_to_generate_count > 0: print(f"Phase 2: Generating {styled_to_generate_count} styled outputs.")

        # Need to map specs to split sources for workers
        spec_to_split = {}
        # This mapping should also use the *remaining* target specs (existing + those to be saved)
        # And the styled specs (existing + those to be generated)
        # A simpler way might be to pass train_image_paths and test_image_paths to the worker
        # or determine split based on img_path inside the worker if needed.
        # For now, let's build the map using the lists we're about to process.
        all_specs_to_process = targets_to_save_list + styled_to_generate_list
        for spec in all_specs_to_process:
             img_path = spec[0]
             if img_path in self.train_image_paths:
                  spec_to_split[spec] = 'train'
             elif img_path in self.test_image_paths:
                   spec_to_split[spec] = 'test'
             # else: Warning? Should not happen if specs come from valid img_paths

        # --- Generation Phase 1: Save Targets ---
        target_futures = []
        saved_target_count = 0

        if targets_to_save_count > 0:
            if self.verbose >= 1: print(f"\nPhase 1: Saving targets...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for target_spec in targets_to_save_list:
                    split_source = spec_to_split.get(target_spec)
                    if split_source:
                        future = executor.submit(
                            save_single_target_worker,
                            tuple(target_spec),
                            self.crop_w, self.crop_h,
                            self.dest_dir, split_source,
                            self.image_sizes, self.base_filenames
                        )
                        target_futures.append(future)

                # Process results
                completed_futures = set()
                for future in concurrent.futures.as_completed(target_futures):
                    if self.stop_requested: return
                    try:
                        target_spec, success, message = future.result()
                        if success:
                            saved_target_count += 1
                            if self.verbose >= 1:
                                print(f"Saved target ({saved_target_count}/{targets_to_save_count}): {os.path.basename(target_spec[0])} ({target_spec[1]},{target_spec[2]}) rot {target_spec[3]} scale {target_spec[4]}%")
                        else:
                            warnings.warn(f"Failed to save target {os.path.basename(target_spec[0])} ({target_spec[1]},{target_spec[2]}) rot {target_spec[3]} scale {target_spec[4]}%: {message}")
                        completed_futures.add(future)
                    except Exception as e:
                        warnings.warn(f"Error processing completed target future: {e}")
                        completed_futures.add(future)
                
                target_futures = [f for f in target_futures if f not in completed_futures]

        # --- Generation Phase 2: Generate and Save Styled Outputs ---
        styled_futures = []
        generated_styled_count = 0

        if styled_to_generate_count > 0:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:                
                # Submit tasks for generating and saving styled outputs
                for styled_spec in styled_to_generate_list:
                    split_source = spec_to_split.get(styled_spec)
                    if split_source:
                        future = executor.submit(
                            generate_and_save_styled_worker,
                            tuple(styled_spec),
                            self.crop_w, self.crop_h, self.dest_dir, split_source,
                            self.image_sizes,
                            self.base_filenames,
                            self.args.palette_algorithm,
                            self.verbose
                        )
                        styled_futures.append(future)

                # Process results
                completed_futures = set()
                start_time = time.time()
                for future_index, future in enumerate(concurrent.futures.as_completed(styled_futures), start=1):
                    if self.stop_requested: return
                    try:
                        styled_spec, success, message = future.result()
                        if success:
                            generated_styled_count += 1
                            elapsed_time = time.time() - start_time
                            avg_time_per_item = elapsed_time / future_index
                            remaining_items = styled_to_generate_count - future_index
                            eta_seconds = int(remaining_items * avg_time_per_item)
                            eta_formatted = str(timedelta(seconds=eta_seconds))
                            # Print progress for styled outputs
                            if self.verbose >= 1:
                                res, sx, sy, s_perc, r_deg, rgb, pal, dm, res_name = styled_spec[8], styled_spec[1], styled_spec[2], styled_spec[4], styled_spec[3], styled_spec[5], styled_spec[6], styled_spec[7], styled_spec[8]
                                pal_str = str(pal) if pal is not None else 'None'
                                print(f"Generated styled output ({generated_styled_count}/{styled_to_generate_count}): "
                                      f"Resolution={res_name}, Crop=({sx},{sy}), Scale={s_perc}%, Rotation={r_deg}°, "
                                      f"ColorSpace={rgb}, Palette={pal_str}, Dither={dm} | ETA: {eta_formatted}")
                        else:
                            warnings.warn(f"Failed to generate styled output {styled_spec[8]}_{styled_spec[1]}_{styled_spec[2]}...: {message}")
                        completed_futures.add(future)
                    except Exception as e:
                        warnings.warn(f"Error processing completed styled future: {e}")
                        completed_futures.add(future)
                styled_futures = [f for f in styled_futures if f not in completed_futures]

        if self.verbose >= 1: print("\nGeneration phases complete.")

    def _final_summary(self):
        """
        Performs a final scan of the output directory and prints a summary report.
        """
        # These variables are local to this method for the final scan results
        final_existing_output_specs = {'train': set(), 'test': set()}
        final_existing_target_specs = {'train': set(), 'test': set()}
        final_invalid_files = {'train': [], 'test': []}  # Should ideally be empty if cleanup ran

        if self.verbose >= 1:
            print("\nPerforming final scan of output directory for summary...")

        for split in ['train', 'test']:
            split_output_dir = os.path.join(self.dest_dir, split)
            if not os.path.isdir(split_output_dir):
                if self.verbose >= 1:
                    print(f"Output directory for split '{split}' not found during final scan: {split_output_dir}. Skipping.")
                continue

            for root, _, files in os.walk(split_output_dir):
                original_base_filename_without_ext = os.path.basename(root)

                original_img_path = None
                split_source = None

                # Find the original image path based on the directory name
                train_match = [p for p in self.train_image_paths if os.path.splitext(os.path.basename(p))[0] == original_base_filename_without_ext]
                if train_match:
                    original_img_path = train_match[0]
                    split_source = 'train'
                else:
                    if self.test_images_dir:
                        test_match = [p for p in self.test_image_paths if os.path.splitext(os.path.basename(p))[0] == original_base_filename_without_ext]
                        if test_match:
                            original_img_path = test_match[0]
                            split_source = 'test'

                for filename in files:
                    parsed_params = parse_generated_filename(filename)

                    if parsed_params:
                        if original_img_path:
                            if parsed_params['type'] == 'target':
                                target_spec = (original_img_path, parsed_params['crop_x'], parsed_params['crop_y'], parsed_params['rot_deg'], parsed_params['scale_perc'])
                                final_existing_target_specs[split_source].add(target_spec)
                            elif parsed_params['type'] == 'style':
                                cs_str = f"RGB{parsed_params['rgb']}"
                                style_spec = (original_img_path, parsed_params['crop_x'], parsed_params['crop_y'], parsed_params['rot_deg'], parsed_params['scale_perc'], cs_str, parsed_params['pal'], parsed_params['dither'], parsed_params['resolution'])
                                final_existing_output_specs[split_source].add(style_spec)
                        else:
                            final_invalid_files[split].append(os.path.join(root, filename))
                    else:
                        final_invalid_files[split].append(os.path.join(root, filename))

        # Calculate the counts for the final summary
        final_generated_targets_by_split = {
            'train': len(final_existing_target_specs['train']),
            'test': len(final_existing_target_specs['test']),
        }
        final_generated_styled_by_split = {
            'train': len(final_existing_output_specs['train']),
            'test': len(final_existing_output_specs['test']),
        }
        total_final_generated_targets = final_generated_targets_by_split['train'] + final_generated_targets_by_split['test']
        total_final_generated_styled = final_generated_styled_by_split['train'] + final_generated_styled_by_split['test']
        total_final_invalid = len(final_invalid_files['train']) + len(final_invalid_files['test'])

        # Print the summary
        print(f"--- Final Dataset Summary ---")
        print(f"Total Target Crops Generated: {total_final_generated_targets}")
        print(f"  Train Targets: {final_generated_targets_by_split['train']} / Requested: {self.train_num_crops}")
        if self.test_images_dir:
            print(f"  Test Targets: {final_generated_targets_by_split['test']} / Requested: {self.test_num_crops}")

        print(f"Total Styled Outputs Generated: {total_final_generated_styled}")
        print(f"  Train Styled: {final_generated_styled_by_split['train']} / Possible Valid: {len(self.full_valid_output_specs['train'])}")
        if self.test_images_dir:
            print(f"  Test Styled: {final_generated_styled_by_split['test']} / Possible Valid: {len(self.full_valid_output_specs['test'])}")

        print(f"Total Invalid Files Remaining in Output: {total_final_invalid}")

        if self.verbose >= 1:
            print("\nDataset generation process finished.")

    def _cleanup_empty_directories(self, base_dir):
        """
        Recursively removes empty directories starting from the deepest level
        within the base_dir, but does not remove the base_dir itself.
        """
        if self.verbose >= 1: print(f"Cleaning up empty directories in {base_dir}...")

        # Use os.walk with topdown=False to visit directories from bottom up.
        # This ensures subdirectories are empty before attempting to remove parent directories.
        for root, dirs, files in os.walk(base_dir, topdown=False):
            # If the directory is empty (contains no files and no subdirectories listed by os.walk)
            # Note: When topdown=False, 'dirs' is the list of names of subdirectories
            # in 'root' that haven't been visited yet *in this walk iteration*.
            # If a subdirectory was just deleted in a previous iteration, it won't be in 'dirs'.
            # So, checking if not dirs and not files is the correct way to see if 'root' is now empty
            # of items that os.walk still knows about.
            if not dirs and not files:
                # It's empty. Now check if we should delete it.
                # We should delete any empty directory *except* the base_dir itself.
                if root != base_dir:
                    try:
                        # Attempt to remove the directory
                        os.rmdir(root)
                        if self.verbose >= 2:
                            print(f"Debug: Removed empty directory: {root}")
                    except OSError as e:
                         # Handle cases where the directory is not empty anymore (e.g., another process created a file)
                         # or permission errors. os.rmdir will raise OSError if the directory is not empty.
                         if self.verbose >= 2:
                              print(f"Debug: Could not remove directory {root} (might not be empty anymore or permission issue): {e}")
                         # No need to re-raise, just warn.
                    except Exception as e:
                         warnings.warn(f"Unexpected error removing empty directory {root}: {e}")
                         # No need to re-raise, just warn.


        if self.verbose >= 1: print("Empty directory cleanup complete.")

    def run(self):
        # Orchestrates the steps of the generator
        self._load_image_paths()
        if self.stop_requested: return
        self._scan_ground_truth()
        if self.stop_requested: return
        self._build_full_valid_specs()
        if self.stop_requested: return
        self._scan_output_directory()
        if self.stop_requested: return
        self._cleanup_invalid_files()
        if self.stop_requested: return
        self._determine_generation_and_deletion()
        if self.stop_requested: return
        self._shrink_dataset()
        if self.stop_requested: return
        self._cleanup_empty_directories(self.args.destination_dir)
        if self.stop_requested: return
        self._run_generation_phases()  # This step generates the files remaining in the final_to_generate lists
        if self.stop_requested: return
        self._final_summary()

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset of styled image crops.')
    parser.add_argument("--train_images", type=str, help="Directory containing ground truth PNG images for train (optional).")
    parser.add_argument("--test_images", type=str, help="Directory containing ground truth PNG images for test (optional).")
    parser.add_argument("--destination_dir", type=str, required=True, help="Directory to save the generated dataset.")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[752, 576], metavar=('W', 'H'), help="Crop size as W H. Defaults to 752 576.")
    parser.add_argument("--train_num_crops", type=int, default=0, help="Number of unique target crops for train (optional, default 0).")
    parser.add_argument("--test_num_crops", type=int, default=0, help="Number of unique target crops for test (optional, default 0).")
    parser.add_argument("--model_path", type=str, help="Optional path to an inference model (TorchScript recommended).")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes. 0 means all CPU cores.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2, 3], help="Verbosity level: 0 (Quiet), 1 (Progress), 2 (Debug).")
    parser.add_argument("--rgb", type=int, nargs='*', default=None, metavar='INT', help="Generate outputs in these RGB formats (e.g., 888 565). Supported: 444, 555, 565, 666, 888.")
    parser.add_argument("--palette", type=int, nargs='*', default=None, metavar='INT', help="Generate outputs with these palette sizes. Supported: 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096. 0 means all colours.")
    parser.add_argument("--rotate", type=int, nargs='*', default=None, metavar='DEGREE', help="Rotate ground truth images by these angles in degrees before cropping (e.g., 0 90 180 270). 0 is default if none specified.")
    parser.add_argument("--downscale", type=int, nargs='*', default=None, metavar='PERCENT', help="Downscale ground truth images to these percentages of the original size before cropping (e.g., 50 75). Must be > 0 and < 100. 0%% is default if none specified.")
    parser.add_argument("--resolution", type=str, nargs='*', default=['lores'], metavar='STYLE', help=f"Generate outputs with these resolution styles. Supported: {SUPPORTED_RESOLUTION_STYLES}. Default: lores.")
    parser.add_argument("--dither", type=str, nargs='*', default=None, metavar='METHOD', help=f"Apply these dithering methods after quantization (e.g., FloydSteinberg None). Supported: {SUPPORTED_DITHER_METHODS}. None is default if none specified or for non-paletted output.")
    parser.add_argument("--train_cache_file", type=str, default=DEFAULT_TRAIN_CACHE_FILE, help="Cache file for train scan results.")
    parser.add_argument("--test_cache_file", type=str, default=DEFAULT_TEST_CACHE_FILE, help="Cache file for test scan results.")
    parser.add_argument("--keep_invalid_files", action='store_true', help="Keep invalid files in the output directory instead of deleting them.")
    parser.add_argument("--palette_algorithm", type=str, default='kmeans', choices=['median_cut', 'kmeans', 'octree'], help="Algorithm to use for palette generation. Default: kmeans.")

    args = parser.parse_args()

    # Handle max_workers = 0
    if args.max_workers == 0:
        args.max_workers = os.cpu_count() or 1 # Use at least 1 worker

    # Create an instance of the DatasetGenerator and run it
    try:
        generator = DatasetGenerator(args)
        generator.run()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
