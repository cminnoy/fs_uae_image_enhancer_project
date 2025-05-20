#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from PIL import Image
import onnxruntime as ort

def load_raw_rgba(raw_path, width=752, height=576):
    """
    Load a raw RGBA image from disk.

    The raw file is expected to contain exactly width * height * 4 bytes,
    in row-major order, with 4 bytes per pixel (R, G, B, A), line by line.

    Returns:
        np.ndarray of shape (1, height, width, 4), dtype=np.uint8
    """
    expected_size = width * height * 4
    try:
        data = np.fromfile(raw_path, dtype=np.uint8)
    except Exception as e:
        print(f"Error reading raw image file '{raw_path}': {e}")
        sys.exit(1)

    if data.size != expected_size:
        print(f"Error: Expected raw file of size {expected_size} bytes "
              f"({width}×{height} RGBA), but got {data.size} bytes.")
        sys.exit(1)

    # Reshape to (height, width, 4)
    img = data.reshape((height, width, 4))

    # Add batch dimension: (1, height, width, 4)
    img = np.expand_dims(img, axis=0)
    return img  # dtype=np.uint8

def save_rgba_png(output_array, output_path):
    """
    Save an (height, width, 4) uint8 RGBA array as a PNG file.
    """
    # Remove batch dimension if present
    if output_array.ndim == 4 and output_array.shape[0] == 1:
        output_array = output_array[0]

    # Expect shape (height, width, 4)
    if output_array.ndim != 3 or output_array.shape[2] != 4:
        print(f"Error: Unexpected output array shape {output_array.shape}")
        sys.exit(1)

    # Ensure array is uint8
    if output_array.dtype != np.uint8:
        output_array = output_array.astype(np.uint8)

    img = Image.fromarray(output_array, mode="RGBA")
    img.save(output_path)
    print(f"Saved output PNG to '{output_path}'")

def run_inference(model_path, raw_path):
    # Load the raw RGBA input
    input_tensor = load_raw_rgba(raw_path)  # shape: (1, 576, 752, 4)

    # Create ONNX Runtime session (use CPUExecutionProvider or adjust as needed)
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Error loading ONNX model '{model_path}': {e}")
        sys.exit(1)

    # Get the single input name
    inputs = session.get_inputs()
    if len(inputs) != 1:
        print(f"Error: Model '{model_path}' must have exactly one input, but has {len(inputs)}")
        sys.exit(1)
    input_name = inputs[0].name

    # Run inference
    try:
        outputs = session.run(None, {input_name: input_tensor})
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

    if len(outputs) != 1:
        print(f"Warning: Expected one output, but got {len(outputs)}. Using the first.")
    output_tensor = outputs[0]  # expected shape: (1, 576, 752, 4), dtype=uint8

    # Determine output PNG path: same base name as raw, but .png
    base, _ = os.path.splitext(raw_path)
    png_path = base + ".png"

    save_rgba_png(output_tensor, png_path)

def main():
    parser = argparse.ArgumentParser(
        description="Run a raw-RGBA image through an ONNX model and save as PNG."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the ONNX model file."
    )
    parser.add_argument(
        "raw_image_path",
        type=str,
        help="Path to the raw RGBA image (752×576, 4 bytes per pixel)."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(args.raw_image_path):
        print(f"Error: Raw image file '{args.raw_image_path}' does not exist.")
        sys.exit(1)

    run_inference(args.model_path, args.raw_image_path)

if __name__ == "__main__":
    main()
