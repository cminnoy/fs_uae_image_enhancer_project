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
                    # The condition checks if the neighbor is on a subsequent row (ny > y)
                    # or on the current row but *ahead* in the processing direction.
                    if 0 <= ny < height and 0 <= nx < width:
                         if ny > y or (ny == y and ((y % 2 == 0 and nx > x) or (y % 2 != 0 and nx < x))):
                            image_float[ny, nx, 0] += error[0] * weight # Use += for inplace addition
                            image_float[ny, nx, 1] += error[1] * weight
                            image_float[ny, nx, 2] += error[2] * weight

# --- Diffusion Maps ---
# Keys are strings to be used in dithering_method argument
# Values are lists of (dx, dy, weight) tuples. dx, dy MUST be integers.
DIFFUSION_MAPS = {
    "floyd-steinberg": [#  
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16),
    ],
    "atkinson": [ # From Apple Lisa, differs by only diffusing a fraction of the error and discarding the rest. 
        (1, 0, 1 / 8),
        (2, 0, 1 / 8),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ],
    "sierra2": [ # Sierra Two-Row
         (1, 0, 4 / 16),
         (2, 0, 3 / 16),
         (-2, 1, 1 / 16),
         (-1, 1, 2 / 16),
         (0, 1, 3 / 16),
         (1, 1, 2 / 16),
         (2, 1, 1 / 16),
    ],
    "stucki": [ # Similar to Floyd-Steinberg but diffuses error over a larger, 3-row area. 
        (1, 0, 8 / 42), (2, 0, 4 / 42),
        (-2, 1, 2 / 42), (-1, 1, 4 / 42), (0, 1, 8 / 42), (1, 1, 4 / 42), (2, 1, 2 / 42),
        (-2, 2, 1 / 42), (-1, 2, 2 / 42), (0, 2, 4 / 42), (1, 2, 2 / 42), (2, 2, 1 / 42),
    ],
     "burkes": [ # A faster variant of Stucki. 
        (1, 0, 8 / 32), (2, 0, 4 / 32),
        (-2, 1, 2 / 32), (-1, 1, 4 / 32), (0, 1, 8 / 32), (1, 1, 4 / 32), (2, 1, 2 / 32),
    ],
     "sierra3": [ #  Sierra Tree Row
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
    dithering_method: str = 'none', # Ensure this is expected as a string 'none'
    verbose: int = 1
) -> np.ndarray:
    """
    Reduces the color depth of an RGB888 NumPy image, optionally reduces the
    palette size, and applies error diffusion dithering.

    Args:
        image_np: Input image as a NumPy array (height, width, 3) in uint8 format (RGB888).
        color_space: Target color space for initial grid quantization ('RGB888', 'RGB565', 'RGB444', 'RGB555' or 'RGB666').
                     If 'RGB888', no initial grid quantization is applied.
        target_palette_size: Optional. The number of colors in the final palette (16, 32, 64, 128, 256, 512, 1024, 2048, 4096).
                             If None, the palette is determined by the color_space grid (for 444/565/666)
                             or the full RGB888 space (if color_space is RGB888).
                             Note: Dithering requires target_palette_size to be specified.
        dithering_method: The error diffusion method ('none', 'floyd-steinberg', etc.).

    Returns:
        A NumPy array representing the processed image (height, width, 3)
        in uint8 format (RGB888 representation of the reduced colors).
    """
    if image_np.ndim != 3 or image_np.shape[2] != 3 or image_np.dtype != np.uint8:
        raise ValueError("Input image must be a 3-channel (RGB) NumPy array of type uint8.")

    # Include RGB565 in the valid color spaces
    valid_color_spaces = ['RGB888', 'RGB565', 'RGB444', 'RGB555', 'RGB666']
    if color_space not in valid_color_spaces:
        raise ValueError(f"color_space must be one of {valid_color_spaces}.")

    valid_palette_sizes = [None, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    if target_palette_size not in valid_palette_sizes:
        raise ValueError(f"target_palette_size must be one of {valid_palette_sizes}.")

    valid_methods = ['none', 'checkerboard'] + list(DIFFUSION_MAPS.keys())
    if dithering_method not in valid_methods:
        raise ValueError(f"dithering_method must be one of {valid_methods}.")

    if verbose:
        print(f"Quantization: color_space={color_space}, target_palette_size={target_palette_size}, dithering_method={dithering_method}")

    # --- Determine the Target Palette ---
    target_palette_8bit = None
    palette_float = None
    pixels_for_kmeans = None # Prepare variable for pixels used for K-Means

    # CHECKERBOARD IS BROKEN; DON'T USE IT
    # For checkerboard we only use grid quantization
    if dithering_method == 'checkerboard':
        # Only support grid spaces for checkerboard
        if color_space not in ['RGB444', 'RGB666', 'RGB565', 'RGB555']:
            raise ValueError("Checkerboard dithering requires a grid quantized color_space (RGB444, RGB555, RGB666, RGB565)")

        img = image_np.astype(np.uint8).copy()
        # Determine grid divider
        if color_space == 'RGB444':
            divider = 16
        elif color_space == 'RGB555':
            divider = 8
        elif color_space == 'RGB666':
            divider = 4
        else:  # RGB565, uneven channels but approximate with G=6 bits
            divider = None
            # Fallback to per-channel handling

        h, w, _ = img.shape
        out = np.zeros_like(img)
        for y in range(h):
            for x in range(w):
                px = img[y, x]
                if divider:
                    low = (px // divider) * divider
                    high = np.minimum(low + divider, 255)
                else:
                    # RGB565 per-channel
                    low = np.array([ (px[0]//8)*8, (px[1]//4)*4, (px[2]//8)*8 ], dtype=np.uint8)
                    high = np.array([ min(low[0]+8,255), min(low[1]+4,255), min(low[2]+8,255) ], dtype=np.uint8)
                # Checkerboard decides low/high
                out[y, x] = high if ((x + y) % 2 == 0) else low
        return out

    # If a target palette size is specified, we calculate a palette using K-Means
    if target_palette_size is not None:
        if verbose > 1:
            print(f"Calculating {target_palette_size}-color palette...")

        start_time = time.time()

        # --- Determine the source pixels for K-Means based on color_space ---
        if color_space == 'RGB888':
            # If target color space is RGB888, calculate palette directly from the original image pixels
            pixels_for_kmeans = image_np.astype(np.float64).reshape(-1, 3)
            if verbose > 1: print("Calculating palette from full RGB888 image.")

        elif color_space in ['RGB444', 'RGB666', 'RGB555', 'RGB565']:
            # If color space is reduced, calculate palette from the reduced grid
            if verbose > 1: print(f"Calculating palette from {color_space} grid.")
            # Apply the color space grid quantization temporarily using float64 copy

            img_quantized_temp_np = image_np.astype(np.float64).copy() # Use copy to avoid modifying original
            if color_space == 'RGB444':
                # Apply RGB444 quantization (4 bits per channel)
                # Round down to the nearest multiple of 16
                img_quantized_temp_np = (np.floor(img_quantized_temp_np / 16) * 16)
            elif color_space == 'RGB666':
                # Apply RGB666 quantization (6 bits per channel)
                # Round down to the nearest multiple of 4
                img_quantized_temp_np = (np.floor(img_quantized_temp_np / 4) * 4)
            elif color_space == 'RGB565':
                # Apply RGB565 quantization (5 bits for R/B, 6 bits for G)
                # Round down R and B to the nearest multiple of 8, G to the nearest multiple of 4
                img_quantized_temp_np[:, :, 0] = (np.floor(img_quantized_temp_np[:, :, 0] / 8) * 8) # R (5 bits)
                img_quantized_temp_np[:, :, 1] = (np.floor(img_quantized_temp_np[:, :, 1] / 4) * 4) # G (6 bits)
                img_quantized_temp_np[:, :, 2] = (np.floor(img_quantized_temp_np[:, :, 2] / 8) * 8) # B (5 bits)
            elif color_space == 'RGB555':
                # Apply RGB555 quantization (5 bits per channel)
                # Round down to the nearest multiple of 8
                img_quantized_temp_np = (np.floor(img_quantized_temp_np / 8) * 8)

            pixels_for_kmeans = img_quantized_temp_np.reshape(-1, 3)

        else:
             # This case should be caught by initial validation, but defensive
             raise ValueError(f"Invalid color_space '{color_space}' for palette calculation.")


        # --- K-Means Palette Calculation ---
        # Ensure enough unique colors exist for K-Means from the chosen pixel source
        unique_colors = np.unique(pixels_for_kmeans, axis=0)
        n_clusters = min(target_palette_size, len(unique_colors))

        if n_clusters == 0:
             target_palette_8bit = np.zeros((1, 3), dtype=np.uint8)
             if verbose > 0: print("Image is a single color. Palette is 1 color.")
        elif n_clusters < target_palette_size:
             target_palette_8bit = unique_colors.astype(np.uint8)
             if verbose > 0:
                print(f"Warning: Requested palette size {target_palette_size} > unique colors ({len(unique_colors)}) "
                            f"from {color_space} grid. Using {n_clusters} unique colors.")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(pixels_for_kmeans)
            target_palette_8bit = kmeans.cluster_centers_.astype(np.uint8)

        end_time = time.time()
        if verbose > 1: print(f"Palette calculation took {end_time - start_time:.2f} seconds.")

        # Prepare float palette for Numba dithering or non-dithered mapping
        palette_float = target_palette_8bit.astype(np.float64)


    # --- 2. Apply Dithering or direct mapping ---

    if dithering_method == 'none':
        if target_palette_size is None:
            # Case: No dithering, no fixed palette -> apply color space grid quantization only or return original
            if color_space == 'RGB888':
                 # If color_space is RGB888 and no palette/dither, return original image
                 if verbose > 1: print("No color reduction, palette, or dithering requested. Returning original image.")
                 img_output_np = image_np.copy()
            elif color_space in ['RGB444', 'RGB555', 'RGB666', 'RGB565']:
                # Case: Color space grid quantization only
                if verbose > 1: print(f"Applying {color_space} grid quantization (no dithering).")
                img_output_np = image_np.astype(np.float64).copy() # Start with a copy

                if color_space == 'RGB444':
                    # Apply RGB444 quantization (4 bits per channel)
                    # Round down to the nearest multiple of 16
                    img_output_np = (np.floor(img_output_np / 16) * 16)
                elif color_space == 'RGB666':
                    # Apply RGB666 quantization (6 bits per channel)
                    # Round down to the nearest multiple of 4
                    img_output_np = (np.floor(img_output_np / 4) * 4)
                elif color_space == 'RGB565':
                    # Apply RGB565 quantization (5 bits for R/B, 6 bits for G)
                    # Round down R and B to the nearest multiple of 8, G to the nearest multiple of 4
                    img_output_np[:, :, 0] = (np.floor(img_output_np[:, :, 0] / 8) * 8) # R (5 bits)
                    img_output_np[:, :, 1] = (np.floor(img_output_np[:, :, 1] / 4) * 4) # G (6 bits)
                    img_output_np[:, :, 2] = (np.floor(img_output_np[:, :, 2] / 8) * 8) # B (5 bits)
                elif color_space == 'RGB555':
                    # Apply RGB555 quantization (5 bits per channel)
                    # Round down to the nearest multiple of 8
                    img_output_np = (np.floor(img_output_np / 8) * 8)

        else: # target_palette_size is specified, dithering_method is 'none'
            # Case: Fixed palette mapping, no dithering
            if verbose > 1:
                print(f"Mapping to {target_palette_size}-color fixed palette (no dithering).")
            # Convert original image to float for distance calculation
            img_float = image_np.astype(np.float64)

            # Map each pixel to the closest color in the target palette
            pixels_float = img_float.reshape(-1, 3)
            distances_sq = np.sum((pixels_float[:, np.newaxis, :] - palette_float)**2, axis=2)
            labels = np.argmin(distances_sq, axis=1)
            img_output_np = target_palette_8bit[labels].reshape(image_np.shape)

    else: # Dithering is requested (implies target_palette_size is specified and checked)
        if target_palette_8bit is None or palette_float is None:
             raise RuntimeError("Palette not calculated correctly for dithering.")

        if dithering_method not in DIFFUSION_MAPS:
             raise ValueError(f"Unknown dithering method: {dithering_method}")
        diff_map_list = DIFFUSION_MAPS[dithering_method]

        img_float = image_np.astype(np.float64)

        if verbose > 1:
            print(f"Applying {dithering_method} dithering onto the {target_palette_size}-color palette...")
        start_time = time.time()
        _apply_palette_dithering_numba(img_float, diff_map_list, palette_float)
        end_time = time.time()
        if verbose > 1:
            print(f"Dithering took {end_time - start_time:.2f} seconds.")

        img_output_np = np.clip(img_float, 0, 255).astype(np.uint8)

    return img_output_np

# --- Example Usage ---

if __name__ == '__main__':
    # Create a dummy test image (a simple gradient)
    width, height = 320, 200
    test_image_np = np.zeros((height, width, 3), dtype=np.uint8)

    # Simple gradient: R varies horizontally, G varies vertically, B fixed
    for y in range(height):
        for x in range(width):
            test_image_np[y, x, 0] = int(x / width * 255) # Red gradient
            test_image_np[y, x, 1] = int(y / height * 255) # Green gradient
            test_image_np[y, x, 2] = 128 # Fixed Blue

    # Or load a real image using PIL
    try:
       img_pil = Image.open("test_image.png").convert("RGB") # Replace with your image file
       test_image_np = np.array(img_pil)
       print("Loaded image from file.")
    except FileNotFoundError:
       print("Test image file not found ('test_image.png'), using generated gradient.")
       # If you don't have test_image.png, save the gradient once
       try:
           Image.fromarray(test_image_np).save("test_image.png")
           print("Generated and saved test_image.png")
       except Exception as e:
           print(f"Could not save generated test_image.png: {e}")


    print("Original image shape:", test_image_np.shape, test_image_np.dtype)

    # Example Conversions:

    # 1. RGB444, no palette reduction, no dithering
    img_rgb444_none_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', dithering_method='none')
    Image.fromarray(img_rgb444_none_np).save("output_rgb444_none.png")
    print("Saved output_rgb444_none.png")

    # 2. RGB666, no palette reduction, no dithering
    img_rgb666_none_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB666', dithering_method='none')
    Image.fromarray(img_rgb666_none_np).save("output_rgb666_none.png")
    print("Saved output_rgb666_none.png")

    # 3. RGB444, 32-color palette (OCS/ECS like), no dithering
    img_rgb444_32_none_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', target_palette_size=32, dithering_method='none')
    Image.fromarray(img_rgb444_32_none_np).save("output_rgb444_32_none.png")
    print("Saved output_rgb444_32_none.png")

    # 4. RGB444, 32-color palette, Floyd-Steinberg dithering
    # This is the one that previously failed - should work now
    img_rgb444_32_fs_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', target_palette_size=32, dithering_method='floyd-steinberg')
    Image.fromarray(img_rgb444_32_fs_np).save("output_rgb444_32_fs.png")
    print("Saved output_rgb444_32_fs.png")

    # 5. RGB666, 256-color palette (AGA like), Atkinson dithering
    img_rgb666_256_atkinson_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB666', target_palette_size=256, dithering_method='atkinson')
    Image.fromarray(img_rgb666_256_atkinson_np).save("output_rgb666_256_atkinson.png")
    print("Saved output_rgb666_256_atkinson.png")

    # 6. RGB444, 64-color palette, Sierra2 dithering
    img_rgb444_64_sierra2_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', target_palette_size=64, dithering_method='sierra2')
    Image.fromarray(img_rgb444_64_sierra2_np).save("output_rgb444_64_sierra2.png")
    print("Saved output_rgb444_64_sierra2.png")

    # Example of a different dithering method
    # 7. RGB444, 32-color palette, Stucki dithering
    img_rgb444_32_stucki_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', target_palette_size=32, dithering_method='stucki')
    Image.fromarray(img_rgb444_32_stucki_np).save("output_rgb444_32_stucki.png")
    print("Saved output_rgb444_32_stucki.png")

    # Example with 16 colors
    img_rgb444_16_fs_np = reduce_color_depth_and_dither(test_image_np, color_space='RGB444', target_palette_size=16, dithering_method='floyd-steinberg')
    Image.fromarray(img_rgb444_16_fs_np).save("output_rgb444_16_fs.png")
    print("Saved output_rgb444_16_fs.png")