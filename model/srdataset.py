import os
import warnings
import re
import random
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F

def parse_generated_filename(filename: str, verbose: int = 1) -> dict | None:
    """
    Parses a generated filename to extract its components based on the
    actual filename format found:
    Styled: <resolution>_<crop_x>_<crop_y>_s<scale>_r<rot>_<style_name_params>.png
    Target: target_<crop_x>_<crop_y>_s<scale>_r<rot>.png

    It does NOT include the original filename part from the directory name.
    """
    name, ext = os.path.splitext(filename)
    if ext.lower() != '.png':
        return None # Only process PNGs

    # Attempt to parse as a Target file first
    target_match = re.match(r'^target_(?P<crop_x>-?\d+)_(?P<crop_y>-?\d+)_s(?P<scale_perc>\d+)_r(?P<rot_deg>-?\d+)$', name) # Added -? to rot_deg for safety
    if target_match:
        try:
            parts = target_match.groupdict()
            crop_x = int(parts['crop_x'])
            crop_y = int(parts['crop_y'])
            scale_perc = int(parts['scale_perc'])
            rot_deg = int(parts['rot_deg'])

            # Return dictionary with parsed target components
            return {
                'type': 'target',
                'crop_x': crop_x,
                'crop_y': crop_y,
                'scale_perc': scale_perc,
                'rot_deg': rot_deg,
                'style_name': None, # No style name for targets
                # Include components needed for location key construction later
                'scale_part': f's{scale_perc}',
                'rot_part': f'r{rot_deg}',
                # Other style parameters are None
                'resolution': None, 'rgb': None, 'pal': None, 'dither': None,
                'filename': filename # Store original filename
            }
        except ValueError:
             if verbose >= 2: warnings.warn(f"ValueError during parsing target filename: {filename}. Skipping.")
             return None
        except Exception as e:
             if verbose >= 2: warnings.warn(f"Unexpected error parsing target filename {filename}: {e}. Skipping.")
             return None


    # Attempt to parse as a Styled file
    # Structure: <resolution>_<crop_x>_<crop_y>_s<scale>_r<rot>_<style_name_params>
    # Style_name_params: rgb<rgb>_p<pal>_d<dither>
    style_match = re.match(r'^(?P<resolution>\w+?)_(?P<crop_x>-?\d+)_(?P<crop_y>-?\d+)_s(?P<scale_perc>\d+)_r(?P<rot_deg>-?\d+)_(?P<style_name>.+)$', name) # Added ? to resolution, captured rest as style_name

    if style_match:
        try:
            parts = style_match.groupdict()
            resolution = parts['resolution']
            crop_x = int(parts['crop_x'])
            crop_y = int(parts['crop_y'])
            scale_perc = int(parts['scale_perc'])
            rot_deg = int(parts['rot_deg'])
            style_name = parts['style_name'] # This is the part after r<rot>_ e.g., 'rgb666_p128_datkinson'


            # Attempt to parse the style name parameters from the style_name string
            # Expected format: rgb<rgb>_p<pal>_d<dither>
            style_params_match = re.match(r'^rgb(?P<rgb_val>\d+)_p(?P<pal_str>\w+)_d(?P<dither_name>[\w-]+)$', style_name)

            if not style_params_match:
                 if verbose >= 2: warnings.warn(f"Styled filename part did not match expected style parameter format: {style_name} in file {filename}. Skipping.")
                 return None # Does not match expected styled parameter format

            try:
                style_params = style_params_match.groupdict()
                rgb_val_str = style_params['rgb_val']
                pal_str = style_params['pal_str']
                dither_name = style_params['dither_name']

                # Convert rgb_val_str to integer
                rgb_val = int(rgb_val_str)

                # Convert palette string 'None' to actual None object
                pal = int(pal_str) if pal_str.lower() != 'none' else None

                # Dither method name (lowercase for consistency)
                dither_method_lower = dither_name.lower()

                # Return a dictionary representing the parsed styled file parameters
                return {
                    'type': 'style',
                    'crop_x': crop_x,
                    'crop_y': crop_y,
                    'scale_perc': scale_perc,
                    'rot_deg': rot_deg,
                    'resolution': resolution,
                    'style_name': style_name, # Store the full style name part
                    # Specific style parameters
                    'rgb': f"RGB{rgb_val}", # Store as 'RGBxxx' string
                    'pal': pal,             # Palette size (int or None)
                    'dither': dither_method_lower,  # Dither method name (lowercase string)
                    'filename': filename, # Store original filename
                    # Include components needed for location key construction later
                    'scale_part': f's{scale_perc}',
                    'rot_part': f'r{rot_deg}',
                }

            except ValueError:
                 if verbose >= 2: warnings.warn(f"ValueError during parsing styled filename parameters: {style_name} in file {filename}. Skipping.")
                 return None
            except Exception as e:
                 if verbose >= 2: warnings.warn(f"Unexpected error parsing styled filename parameters {style_name} in file {filename}: {e}. Skipping.")
                 return None

        except ValueError:
             # Handle errors during integer conversion for crop_x, crop_y, scale_perc, rot_deg from the main match
             if verbose >= 2: warnings.warn(f"ValueError during initial parsing of filename {filename}. Skipping.")
             return None
        except Exception as e:
             if verbose >= 2: warnings.warn(f"Unexpected error during initial parsing of filename {filename}: {e}. Skipping.")
             return None

    # If the filename didn't match either expected target or styled format
    if verbose >= 2: warnings.warn(f"Filename did not match expected format: {filename}. Skipping.")
    return None


# ----------------------------
# Helper function to gather all samples from a directory
# ----------------------------
def gather_all_samples_from_directory(directory_path: str, expected_crop_size: tuple[int, int], styles_to_include: set | None = None, verbose: int = 1):
    """
    Walks a directory, identifies target and styled crop files using the
    new filename format, and gathers all available (styled_path, target_path)
    pairs grouped by unique crop location (directory name + parsed components).
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Data directory not found: {directory_path}")

    # Dictionary to group parsed file information by their unique crop location key.
    # Key: (original_directory_name, crop_x, crop_y, scale_part, rot_part)
    # Value: List of parsed file dictionaries found for this unique crop instance
    grouped_files_by_location = defaultdict(list)

    if verbose >= 1: print(f"Scanning directory {directory_path} for sample files...")
    # os.walk traverses the directory tree. Root will be like .../generator_output/train/image123_without_ext
    for root, dirs, files in os.walk(directory_path):
        # We expect subdirectories within the main directory_path
        # The name of the subdirectory is typically the original filename part that generator used
        original_directory_name = os.path.basename(root)

        # Only process files directly within subdirectories of directory_path, not in directory_path itself
        if os.path.abspath(root) == os.path.abspath(directory_path):
            if verbose >= 3: print(f"DEBUG GATHER: Skipping root directory: {root}")
            continue  # Skip the root directory itself

        for filename in files:
            # Use the standalone parser to get structured info from the filename itself
            parsed_info = parse_generated_filename(filename, verbose=verbose)

            # If parsing was successful and contains necessary components for the key
            if parsed_info and all(key in parsed_info for key in ['crop_x', 'crop_y', 'scale_part', 'rot_part']):
                full_file_path = os.path.join(root, filename)
                parsed_info['full_path'] = full_file_path

                # Construct the unique location key using directory name and parsed components
                crop_location_key = (
                    original_directory_name,
                    parsed_info['crop_x'],
                    parsed_info['crop_y'],
                    parsed_info['scale_part'],
                    parsed_info['rot_part']
                )

                # Group files by this constructed location key
                grouped_files_by_location[crop_location_key].append(parsed_info)
                if verbose >= 3: print(f"DEBUG GATHER: Grouping file {filename} under key {crop_location_key}")
            elif parsed_info:
                if verbose >= 2: warnings.warn(f"Parsed info for {filename} is missing key components for location key. Skipping.")

    if verbose >= 1: print(f"Finished scanning. Found files in {len(grouped_files_by_location)} unique crop locations.")
    if verbose >= 2 and styles_to_include is not None: print(f"Debug Gather: Styles to include filter: {styles_to_include}")

    # List to hold all available (styled_input_path, target_path) pairs across all locations
    available_samples_pool: list[tuple[str, str]] = []

    # Iterate through the grouped files to create the list of available samples
    if verbose >= 2: print("Debug Gather: Pairing styled files with target files...")
    for crop_location_key, files_info_list in grouped_files_by_location.items():
        target_file_info = None
        styled_files_info: list[dict] = []

        # Separate target from styled files within the group and apply style filtering
        for file_info in files_info_list:
            if file_info['type'] == 'target':
                target_file_info = file_info
            elif file_info['type'] == 'style':
                # Apply the styles_to_include filter here
                if styles_to_include is None or (
                    file_info.get('style_name') is not None and
                    any(substring in file_info['style_name'] for substring in styles_to_include)
                ):
                    styled_files_info.append(file_info)
                elif verbose >= 2 and file_info.get('style_name') is not None:
                    print(f"DEBUG GATHER: Skipping styled file {file_info['full_path']} with style '{file_info['style_name']}' due to style filter.")
                elif verbose >= 2:
                    print(f"DEBUG GATHER: Skipping styled file {file_info.get('full_path', 'unknown')} due to missing style name or style filter.")

        # If a target file was found AND there are included styled files at this location, create pairs
        if target_file_info and styled_files_info:
            target_path = target_file_info['full_path']

            # Optional: Validate target image size
            try:
                with Image.open(target_path) as img:
                    if img.size != expected_crop_size:
                        warnings.warn(f"Target image {target_path} has unexpected dimensions: {img.size}. Expected: {expected_crop_size}. Skipping all samples for this crop location.")
                        continue
            except FileNotFoundError:
                warnings.warn(f"Target image file not found: {target_path}. Skipping all samples for this crop location.")
                continue
            except Exception as e:
                warnings.warn(f"Could not read target image {target_path} for size validation: {e}. Skipping all samples for this crop location.")
                continue

            for styled_file_info in styled_files_info:
                styled_path = styled_file_info['full_path']
                available_samples_pool.append((styled_path, target_path))
                if verbose >= 2:
                    print(f"DEBUG GATHER: Added pair: ({os.path.basename(styled_path)}, {os.path.basename(target_path)})")

    if verbose >= 1: print(f"Sample gathering complete. Found {len(available_samples_pool)} total sample pairs matching criteria.")

    return available_samples_pool

# ----------------------------
# Dataset for Super-Resolution
# Adapted to take a pre-gathered list of samples
# ----------------------------
class SRDataset(Dataset):
    """
    Dataset for loading styled input crops and corresponding target crops
    from a pre-defined list of (styled_path, target_path) pairs.

    Dataset length is decoupled from the number of available files by num_samples.
    Applies synchronized random horizontal and vertical flips.
    Loads pre-cropped images based on the provided paths.
    """
    # Changed __init__ signature
    def __init__(self, sample_pairs_list: list[tuple[str, str]], expected_crop_size: tuple[int, int], num_samples: int):
        """
        Args:
            sample_pairs_list: A list of (styled_img_path, target_img_path) tuples.
                               This is the pool of samples to draw from.
            expected_crop_size: The expected (width, height) of the cropped images.
                                Used for potential validation within __getitem__.
            num_samples: The declared length of the dataset for one epoch.
                         __len__ will return this value. Samples are drawn randomly
                         from the available pool (sample_pairs_list).
        """
        # Removed dataset_dir, split, and styles_to_include from args
        # styles_to_include filtering is done during sample gathering now

        self.available_samples_pool: list[tuple[str, str]] = sample_pairs_list # Directly assign the list
        self.expected_crop_w, self.expected_crop_h = expected_crop_size
        self.num_samples = num_samples # The declared length of the dataset for an epoch

        # Check if the pool of available samples is empty
        if not self.available_samples_pool:
            # Warning handled before dataset creation, but defensive check.
            warnings.warn(f"SRDataset initialized with an empty sample pool.")


        # self.to_tensor = ToTensor() # No longer use ToTensor for input
        # Removed _parse_filename and _gather_available_samples methods
        # Removed self.styles_to_include

    def __len__(self):
        """
        Returns the declared length of the dataset, determined by num_samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a randomly selected sample (styled input image, target image) pair
        from the pool of available samples and applies transforms.

        Ignores the input index `idx`.
        """
        # Ensure there are samples available to draw from
        if not self.available_samples_pool:
            raise IndexError("SRDataset pool is empty. Cannot draw samples.")

        # Randomly select an index from the list of available samples.
        # The index must be within the bounds of the actual available samples pool, not the declared num_samples.
        pool_size = len(self.available_samples_pool)
        if pool_size == 0: # Double check size
             raise IndexError("SRDataset pool is empty. Cannot draw samples.")
        random_idx = random.randint(0, pool_size - 1)


        # Get the file paths for the styled input and the target image using the random index
        styled_img_path, target_img_path = self.available_samples_pool[random_idx]

        # Load the images using Pillow.
        try:
            # TODO just keep float32 BCHW format and adapt the onnx converter to add layer on front
            # Load input image as RGBA and convert to uint8 tensor (4 channels, HxWxD)
            styled_img_pil = Image.open(styled_img_path).convert('RGBA')
            # Convert PIL image to numpy array (uint8, HxWxChannels)
            styled_img_np = np.array(styled_img_pil)
            # Convert numpy array to torch tensor (uint8, HxWxChannels) and permute to CxHxW
            lr_t = torch.from_numpy(styled_img_np).permute(2, 0, 1)

            # Load target image as RGB and convert to float32 tensor, scaled to [0.0, 255.0]
            target_img_pil = Image.open(target_img_path).convert('RGBA')
            # Use ToTensor to get float32 [0, 1] CxHxW, then scale to [0.0, 255.0]
            to_tensor = ToTensor() # Can define here or in init if needed elsewhere
            hr_t = to_tensor(target_img_pil).mul(255.0)

        except Exception as e:
             # If there's an error loading the selected sample, try to get another one randomly.
             # This is a simple retry mechanism.
             warnings.warn(f"Error loading images for randomly selected sample ({styled_img_path}, {target_img_path}): {e}. Retrying.")
             # Recursively call __getitem__ to get another random sample.
             # Be cautious with deep recursion in case of many corrupted files or infinite loops.
             # A better approach might be to log and return a dummy blank image, or limit retries.
             return self.__getitem__(idx) # Pass the original idx (though it's ignored)

        # Optional: Add validation for image dimensions
        # You might want to log or handle cases where dimensions don't match
        if styled_img_pil.size != (self.expected_crop_w, self.expected_crop_h): # Use PIL size before numpy conversion
             warnings.warn(f"Styled image {styled_img_path} has unexpected dimensions: {styled_img_pil.size}. Expected: {(self.expected_crop_w, self.expected_crop_h)}. Proceeding but this may cause issues.")
        # Validate target image dimensions (already done during gathering, but double check is okay)
        if target_img_pil.size != (self.expected_crop_w, self.expected_crop_h):
             warnings.warn(f"Target image {target_img_path} has unexpected dimensions: {target_img_pil.size}. Expected: {(self.expected_crop_w, self.expected_crop_h)}. Proceeding but this may cause issues.")


        # Apply synchronized random horizontal flip
        # Note: Need to ensure transforms work on the tensor types/ranges
        # F.hflip works on float tensors. Need to handle lr_t (uint8).
        # It's easier to flip the PIL images *before* converting to tensor.
        # Or, convert uint8 to float for flipping and then back if necessary, but that adds overhead.
        # Let's flip PIL images first.

        # Synchronized random flip decision
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5

        if do_hflip:
            styled_img_pil = F.hflip(styled_img_pil)
            target_img_pil = F.hflip(target_img_pil)

        if do_vflip:
            styled_img_pil = F.vflip(styled_img_pil)
            target_img_pil = F.vflip(target_img_pil)

        # Convert flipped PIL images to tensors
        # Input (LR) - RGBA uint8
        styled_img_np = np.array(styled_img_pil.convert('RGBA')) # Convert back to RGBA after flip
        lr_t = torch.from_numpy(styled_img_np).permute(2, 0, 1) # uint8, 4 channels, CxHxW

        # Target (HR) - RGB float32 scaled to [0, 255]
        to_tensor = ToTensor() # Re-create or use self.to_tensor if in init
        hr_t = to_tensor(target_img_pil.convert('RGBA')).mul(255.0) # Convert back to RGB after flip, float32, 3 channels, CxHxW, scaled


        # Return the randomly selected styled input (LR) and target (HR) tensors
        # Model expects LR to be uint8 4-channel
        # Loss expects HR to be float, matching scaled output
        return lr_t, hr_t