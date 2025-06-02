import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import numba as nb
import time # For timing the Numba parts

# --- Numba Helper Functions ---

@nb.njit(cache=True)
def _find_closest_color_index_numba(pixel: np.ndarray, palette: np.ndarray) -> int:
    """
    Numba-accelerated function to find the index of the closest color in a palette.
    Assumes pixel (1,3) and palette (N,3) are float64.
    """
    min_dist_sq = np.inf
    closest_index = 0
    # Numba requires iterating over array indices explicitly or using range
    for i in range(palette.shape[0]):
        # Calculate squared Euclidean distance
        dist_sq = (pixel[0] - palette[i, 0])**2 + \
                  (pixel[1] - palette[i, 1])**2 + \
                  (pixel[2] - palette[i, 2])**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_index = i
    return closest_index

# Corrected Numba dithering function
@nb.njit(cache=True)
def _apply_palette_dithering_numba(image_float: np.ndarray, diff_map_list: list, palette_float: np.ndarray):
    """
    Numba-accelerated error diffusion dithering onto a specified palette.
    Modifies image_float in place.
    diff_map_list is a list of (dx, dy, weight) tuples where dx, dy are int, weight is float.
    """
    height, width, _ = image_float.shape

    for y in range(height):
        # Process rows alternating direction for better results (snake pattern)
        if y % 2 == 0:
            x_range = range(width)
        else:
            x_range = range(width - 1, -1, -1)

        for x in x_range:
            current_pixel = image_float[y, x]

            # Find the closest color in the palette
            closest_index = _find_closest_color_index_numba(current_pixel, palette_float)
            closest_color = palette_float[closest_index]

            # Calculate the error
            error = current_pixel - closest_color

            # Set the current pixel to the palette color
            image_float[y, x] = closest_color # Store the palette color (float representation)

            # Diffuse the error to neighbors
            if error[0] != 0.0 or error[1] != 0.0 or error[2] != 0.0:
                # Iterate over the list of tuples directly
                for i in range(len(diff_map_list)):
                    dx, dy, weight = diff_map_list[i] # dx, dy are integers, weight is float

                    # Adjust dx based on current row direction
                    effective_dx = dx if y % 2 == 0 else -dx
                    nx, ny = x + effective_dx, y + dy # ny, nx are integers

                    # Check if neighbor is within bounds AND in a future step
                    if 0 <= ny < height and 0 <= nx < width:
                         if ny > y or (ny == y and ((y % 2 == 0 and nx > x) or (y % 2 != 0 and nx < x))):
                            image_float[ny, nx, 0] += error[0] * weight
                            image_float[ny, nx, 1] += error[1] * weight
                            image_float[ny, nx, 2] += error[2] * weight

@nb.njit(cache=True)
def _apply_checkerboard_dithering_numba_optimized(
    image_float_input: np.ndarray,  # (H, W, 3) float64
    palette_float: np.ndarray,      # (N, 3) float64, for distance calculations
    palette_uint8: np.ndarray,      # (N, 3) uint8, for assignment
    output_image_uint8: np.ndarray  # (H, W, 3) uint8, to be filled
):
    """
    Numba-accelerated checkerboard dithering using a specified palette.
    For each pixel in the input image, it finds the two closest colors
    from the palette and alternates them in a checkerboard pattern.
    Modifies output_image_uint8 in place.
    """
    height, width, _ = image_float_input.shape
    num_palette_colors = palette_float.shape[0]

    if num_palette_colors == 0: # Should ideally be caught by caller
        for y_idx in range(height):
            for x_idx in range(width):
                output_image_uint8[y_idx, x_idx, 0] = 0
                output_image_uint8[y_idx, x_idx, 1] = 0
                output_image_uint8[y_idx, x_idx, 2] = 0
        return

    if num_palette_colors == 1: # Only one color in palette
        color_val_r = palette_uint8[0, 0]
        color_val_g = palette_uint8[0, 1]
        color_val_b = palette_uint8[0, 2]
        for y_idx in range(height):
            for x_idx in range(width):
                output_image_uint8[y_idx, x_idx, 0] = color_val_r
                output_image_uint8[y_idx, x_idx, 1] = color_val_g
                output_image_uint8[y_idx, x_idx, 2] = color_val_b
        return

    # Main logic for num_palette_colors >= 2
    for y_idx in range(height):
        for x_idx in range(width):
            current_pixel_float_r = image_float_input[y_idx, x_idx, 0]
            current_pixel_float_g = image_float_input[y_idx, x_idx, 1]
            current_pixel_float_b = image_float_input[y_idx, x_idx, 2]

            # Find 1st closest color index
            min_dist_sq1 = np.inf
            idx1 = 0
            for i in range(num_palette_colors):
                d_r1 = current_pixel_float_r - palette_float[i, 0]
                d_g1 = current_pixel_float_g - palette_float[i, 1]
                d_b1 = current_pixel_float_b - palette_float[i, 2]
                dist_sq = d_r1**2 + d_g1**2 + d_b1**2
                if dist_sq < min_dist_sq1:
                    min_dist_sq1 = dist_sq
                    idx1 = i
            
            # Find 2nd closest color index (must be different from idx1)
            min_dist_sq2 = np.inf
            idx2 = 0 # Default initialization
            if idx1 == 0: # Ensure idx2 starts as a different index
                idx2 = 1 # num_palette_colors is guaranteed >= 2 here
            # else idx2 remains 0, which is different from a non-zero idx1, so loop will find correct one.

            for i in range(num_palette_colors):
                if i == idx1:
                    continue
                d_r2 = current_pixel_float_r - palette_float[i, 0]
                d_g2 = current_pixel_float_g - palette_float[i, 1]
                d_b2 = current_pixel_float_b - palette_float[i, 2]
                dist_sq = d_r2**2 + d_g2**2 + d_b2**2
                if dist_sq < min_dist_sq2:
                    min_dist_sq2 = dist_sq
                    idx2 = i
            
            # If the closest colour is an exact match (error is zero), always choose that one
            if min_dist_sq1 == 0.0:
                chosen_idx = idx1
            else: # Otherwise alternate between closest and second closest colour
                chosen_idx = idx1 if (x_idx + y_idx) % 2 == 0 else idx2
            
            chosen_color_r = palette_uint8[chosen_idx, 0]
            chosen_color_g = palette_uint8[chosen_idx, 1]
            chosen_color_b = palette_uint8[chosen_idx, 2]
            output_image_uint8[y_idx, x_idx, 0] = chosen_color_r
            output_image_uint8[y_idx, x_idx, 1] = chosen_color_g
            output_image_uint8[y_idx, x_idx, 2] = chosen_color_b

# --- Diffusion Maps ---
DIFFUSION_MAPS = {
    "floyd-steinberg": [
                                         (1, 0, 7 / 16),
        (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16),
    ],
    "atkinson": [
                                       (1, 0, 1 / 8), (2, 0, 1 / 8),
        (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 1 / 8),
                        (0, 2, 1 / 8),
    ],
    "sierra2": [
                                                          (1, 0, 4 / 16), (2, 0, 3 / 16),
        (-2, 1, 1 / 16), (-1, 1, 2 / 16), (0, 1, 3 / 16), (1, 1, 2 / 16), (2, 1, 1 / 16),
    ],
    "stucki": [
                                                          (1, 0, 8 / 42), (2, 0, 4 / 42),
        (-2, 1, 2 / 42), (-1, 1, 4 / 42), (0, 1, 8 / 42), (1, 1, 4 / 42), (2, 1, 2 / 42),
        (-2, 2, 1 / 42), (-1, 2, 2 / 42), (0, 2, 4 / 42), (1, 2, 2 / 42), (2, 2, 1 / 42),
    ],
     "burkes": [
                                                          (1, 0, 8 / 32), (2, 0, 4 / 32),
        (-2, 1, 2 / 32), (-1, 1, 4 / 32), (0, 1, 8 / 32), (1, 1, 4 / 32), (2, 1, 2 / 32),
    ],
     "sierra3": [
                                                          (1, 0, 5 / 32), (2, 0, 3 / 32),
        (-2, 1, 2 / 32), (-1, 1, 4 / 32), (0, 1, 5 / 32), (1, 1, 4 / 32), (2, 1, 2 / 32),
                         (-1, 2, 2 / 32), (0, 2, 3 / 32), (1, 2, 2 / 32),
    ],
}


# --- Main Python Function ---

def reduce_color_depth_and_dither(
    image_np: np.ndarray,
    color_space: str,
    target_palette_size: int = None,
    dithering_method: str = 'none',
    verbose: int = 1
) -> np.ndarray:
    """
    Reduces the color depth of an RGB888 NumPy image, optionally reduces the
    palette size, and applies error diffusion or checkerboard dithering.

    Args:
        image_np: Input image as a NumPy array (height, width, 3) in uint8 format (RGB888).
        color_space: Target color space for initial grid quantization ('RGB888', 'RGB565', 'RGB444', 'RGB555' or 'RGB666').
                     If 'RGB888', no initial grid quantization is applied when generating palette.
                     This argument primarily affects palette generation if target_palette_size is set.
        target_palette_size: Optional. The number of colors in the final palette (16, 32, 64, 128, 256, 512, 1024, 2048, 4096).
                             If None, the palette is determined by the color_space grid (for 444/565/666 if dither='none')
                             or the full RGB888 space (if color_space is RGB888 and dither='none').
                             Note: Dithering (error diffusion or checkerboard) requires target_palette_size to be specified.
        dithering_method: The dithering method ('none', 'checkerboard', 'floyd-steinberg', etc.).

    Returns:
        A NumPy array representing the processed image (height, width, 3)
        in uint8 format (RGB888 representation of the reduced colors).
    """
    if image_np.ndim != 3 or image_np.shape[2] != 3 or image_np.dtype != np.uint8:
        raise ValueError("Input image must be a 3-channel (RGB) NumPy array of type uint8.")

    valid_color_spaces = ['RGB888', 'RGB565', 'RGB444', 'RGB555', 'RGB666']
    if color_space not in valid_color_spaces:
        raise ValueError(f"color_space must be one of {valid_color_spaces}.")

    valid_palette_sizes = [None, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    if target_palette_size not in valid_palette_sizes:
        raise ValueError(f"target_palette_size must be one of {valid_palette_sizes}.")

    valid_methods = ['none', 'checkerboard'] + list(DIFFUSION_MAPS.keys())
    if dithering_method not in valid_methods:
        raise ValueError(f"dithering_method must be one of {valid_methods}.")

    if verbose:
        print(f"Processing: color_space={color_space}, target_palette_size={target_palette_size}, dithering_method={dithering_method}")

    # --- Determine the Target Palette ---
    target_palette_8bit = None
    palette_float = None
    pixels_for_kmeans = None

    # If a target palette size is specified, we calculate a palette using K-Means
    if target_palette_size is not None:
        if verbose > 1:
            print(f"Calculating {target_palette_size}-color palette...")

        start_time = time.time()

        if color_space == 'RGB888':
            pixels_for_kmeans = image_np.astype(np.float64).reshape(-1, 3)
            if verbose > 1: print("Calculating palette from full RGB888 image.")
        elif color_space in ['RGB444', 'RGB666', 'RGB555', 'RGB565']:
            if verbose > 1: print(f"Calculating palette from {color_space} grid for K-Means input.")
            img_quantized_temp_np = image_np.astype(np.float64).copy()
            if color_space == 'RGB444':
                img_quantized_temp_np = (np.floor(img_quantized_temp_np / 16) * 16)
            elif color_space == 'RGB666':
                img_quantized_temp_np = (np.floor(img_quantized_temp_np / 4) * 4)
            elif color_space == 'RGB565':
                img_quantized_temp_np[:, :, 0] = (np.floor(img_quantized_temp_np[:, :, 0] / 8) * 8)
                img_quantized_temp_np[:, :, 1] = (np.floor(img_quantized_temp_np[:, :, 1] / 4) * 4)
                img_quantized_temp_np[:, :, 2] = (np.floor(img_quantized_temp_np[:, :, 2] / 8) * 8)
            elif color_space == 'RGB555':
                img_quantized_temp_np = (np.floor(img_quantized_temp_np / 8) * 8)
            pixels_for_kmeans = img_quantized_temp_np.reshape(-1, 3)
        else:
             raise ValueError(f"Invalid color_space '{color_space}' for palette calculation source.")

        unique_colors = np.unique(pixels_for_kmeans, axis=0)
        n_clusters = min(target_palette_size, len(unique_colors))

        if n_clusters == 0: # Should ideally not happen with real images
             target_palette_8bit = np.zeros((1, 3), dtype=np.uint8) # Default to black
             if verbose > 0: print("Image seems to have no discernible colors or is empty. Using a single black color palette.")
        elif n_clusters < target_palette_size:
             target_palette_8bit = unique_colors.astype(np.uint8)
             if verbose > 0:
                print(f"Warning: Requested palette size {target_palette_size} > unique colors ({len(unique_colors)}) "
                            f"from {color_space} pre-quantization. Using {n_clusters} unique colors.")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(pixels_for_kmeans)
            target_palette_8bit = kmeans.cluster_centers_.astype(np.uint8)

        end_time = time.time()
        if verbose > 1: print(f"Palette calculation took {end_time - start_time:.2f} seconds.")
        palette_float = target_palette_8bit.astype(np.float64)

    # --- Apply Dithering or direct mapping ---
    img_output_np = np.zeros_like(image_np) # Initialize output image

    if dithering_method == 'none':
        if target_palette_size is None:
            if color_space == 'RGB888':
                 if verbose > 1: print("No color reduction, palette, or dithering. Returning original image.")
                 img_output_np = image_np.copy()
            elif color_space in ['RGB444', 'RGB555', 'RGB666', 'RGB565']:
                if verbose > 1: print(f"Applying {color_space} grid quantization (no dithering).")
                temp_float_img = image_np.astype(np.float64) # Work with float then convert
                if color_space == 'RGB444':
                    temp_float_img = (np.floor(temp_float_img / 16) * 16)
                elif color_space == 'RGB666':
                    temp_float_img = (np.floor(temp_float_img / 4) * 4)
                elif color_space == 'RGB565':
                    temp_float_img[:, :, 0] = (np.floor(temp_float_img[:, :, 0] / 8) * 8)
                    temp_float_img[:, :, 1] = (np.floor(temp_float_img[:, :, 1] / 4) * 4)
                    temp_float_img[:, :, 2] = (np.floor(temp_float_img[:, :, 2] / 8) * 8)
                elif color_space == 'RGB555':
                    temp_float_img = (np.floor(temp_float_img / 8) * 8)
                img_output_np = np.clip(temp_float_img, 0, 255).astype(np.uint8)
        else: # target_palette_size is specified, dithering_method is 'none'
            if verbose > 1:
                print(f"Mapping to {target_palette_8bit.shape[0]}-color fixed palette (no dithering).")
            img_float = image_np.astype(np.float64)
            pixels_float = img_float.reshape(-1, 3)
            distances_sq = np.sum((pixels_float[:, np.newaxis, :] - palette_float)**2, axis=2)
            labels = np.argmin(distances_sq, axis=1)
            img_output_np = target_palette_8bit[labels].reshape(image_np.shape)

    elif dithering_method == 'checkerboard':
        if target_palette_size is None or target_palette_8bit is None or palette_float is None:
            raise ValueError("Checkerboard dithering requires a target_palette_size to be specified, "
                             "which defines the palette for dithering.")
        
        if verbose > 1:
            print(f"Applying checkerboard dithering with {target_palette_8bit.shape[0]}-color palette.")

        # Output array should already be initialized, ensure it's uint8 for the Numba function
        img_output_np = np.zeros_like(image_np, dtype=np.uint8) 
        image_for_dither_float = image_np.astype(np.float64)

        _apply_checkerboard_dithering_numba_optimized(
            image_for_dither_float,
            palette_float,
            target_palette_8bit,
            img_output_np # Numba function modifies this in place
        )

    elif dithering_method in DIFFUSION_MAPS:
        if target_palette_8bit is None or palette_float is None:
             raise RuntimeError(f"Error diffusion dithering ('{dithering_method}') requires a palette. "
                                "Ensure target_palette_size is specified.")
        
        diff_map_list = DIFFUSION_MAPS[dithering_method]
        img_float_dither = image_np.astype(np.float64).copy() # Numba function modifies this copy

        if verbose > 1:
            print(f"Applying {dithering_method} dithering onto the {target_palette_8bit.shape[0]}-color palette...")
        start_time = time.time()
        _apply_palette_dithering_numba(img_float_dither, diff_map_list, palette_float)
        end_time = time.time()
        if verbose > 1:
            print(f"Dithering took {end_time - start_time:.2f} seconds.")
        img_output_np = np.clip(img_float_dither, 0, 255).astype(np.uint8)
    
    # valid_methods check at the beginning should prevent reaching here with an unknown method
    # else:
    #    raise ValueError(f"Internal error or unknown dithering_method: {dithering_method}")

    return img_output_np.astype(np.uint8)


# --- Example Usage ---
if __name__ == '__main__':
    width, height = 320, 200
    test_image_np = np.zeros((height, width, 3), dtype=np.uint8)
    for y_coord in range(height): # Renamed y to y_coord
        for x_coord in range(width): # Renamed x to x_coord
            test_image_np[y_coord, x_coord, 0] = int(x_coord / width * 255)
            test_image_np[y_coord, x_coord, 1] = int(y_coord / height * 255)
            test_image_np[y_coord, x_coord, 2] = 128

    try:
       img_pil = Image.open("test_image.png").convert("RGB")
       test_image_np = np.array(img_pil)
       print("Loaded image from file.")
    except FileNotFoundError:
       print("Test image file not found ('test_image.png'), using generated gradient.")
       try:
           Image.fromarray(test_image_np).save("test_image.png")
           print("Generated and saved test_image.png")
       except Exception as e:
           print(f"Could not save generated test_image.png: {e}")

    print("Original image shape:", test_image_np.shape, test_image_np.dtype)

    # 1. RGB444, no palette reduction, no dithering
    img_rgb444_none_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', dithering_method='none', verbose=2)
    Image.fromarray(img_rgb444_none_np).save("output_rgb444_none.png")
    print("Saved output_rgb444_none.png")

    # 2. RGB444, 32-color palette, Floyd-Steinberg dithering
    img_rgb444_32_fs_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', target_palette_size=32, dithering_method='floyd-steinberg', verbose=2)
    Image.fromarray(img_rgb444_32_fs_np).save("output_rgb444_32_fs.png")
    print("Saved output_rgb444_32_fs.png")

    # 3. RGB555, 64-color palette, Checkerboard dithering (NEW TEST)
    img_rgb555_64_checker_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB555', target_palette_size=64, dithering_method='checkerboard', verbose=2)
    Image.fromarray(img_rgb555_64_checker_np).save("output_rgb555_64_checkerboard.png")
    print("Saved output_rgb555_64_checkerboard.png")

    # 4. RGB888 (full color for K-Means), 16-color palette, Checkerboard dithering
    img_rgb888_16_checker_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB888', target_palette_size=16, dithering_method='checkerboard', verbose=2)
    Image.fromarray(img_rgb888_16_checker_np).save("output_rgb888_16_checkerboard.png")
    print("Saved output_rgb888_16_checkerboard.png")

    # 5. RGB666, 256-color palette (AGA like), Atkinson dithering
    img_rgb666_256_atkinson_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB666', target_palette_size=256, dithering_method='atkinson', verbose=2)
    Image.fromarray(img_rgb666_256_atkinson_np).save("output_rgb666_256_atkinson.png")
    print("Saved output_rgb666_256_atkinson.png")

    # 6. Test case: Checkerboard with a very small palette (2 colors)
    #    Force a small palette for testing by creating an image with few colors for K-Means.
    small_palette_img_np = np.zeros((height, width, 3), dtype=np.uint8)
    small_palette_img_np[:height//2, :, :] = [30, 60, 90]  # One color
    small_palette_img_np[height//2:, :, :] = [150, 180, 210] # Another color
    # Use RGB888 so K-Means picks from these exact colors if target_palette_size=2
    try:
        Image.fromarray(small_palette_img_np).save("small_palette_source_img.png")
        print("Saved small_palette_source_img.png for checkerboard test.")
    except Exception as e:
        print(f"Could not save small_palette_source_img.png: {e}")
    
    img_2color_checker_np = reduce_color_depth_and_dither(small_palette_img_np, color_space='RGB888', target_palette_size=2, dithering_method='checkerboard', verbose=2)
    Image.fromarray(img_2color_checker_np).save("output_2color_checkerboard.png")
    print("Saved output_2color_checkerboard.png (should dither between the two main colors of input).")

    # 7. Test case: Checkerboard with target_palette_size=None (should raise error)
    try:
        print("Testing checkerboard with target_palette_size=None (expecting ValueError)...")
        reduce_color_depth_and_dither(test_image_np, color_space='RGB444', dithering_method='checkerboard', verbose=2)
    except ValueError as e:
        print(f"Caught expected error for checkerboard without palette: {e}")