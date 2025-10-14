#!/usr/bin/env python3

import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageCms, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".png", ".gif", ".tif", ".tiff", ".bmp", ".avif"}

def process_image(src: Path, input_dir: Path, output_dir: Path, max_crop_size):
    try:
        with Image.open(src) as img:
            # Convert color space to sRGB if ICC profile present
            icc_bytes = img.info.get("icc_profile")
            if icc_bytes:
                try:
                    src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                    dst_profile = ImageCms.createProfile("sRGB")
                    img = ImageCms.profileToProfile(img, src_profile, dst_profile, outputMode="RGB")
                    print(f"[ICC→sRGB] {src}")
                except Exception as e:
                    print(f"[ICC warn] {src} — could not apply ICC profile ({e}); falling back to RGB")
                    if img.mode != "RGB":
                        img = img.convert("RGB")
            else:
                if img.mode != "RGB":
                    print(f"[RGB] {src} → RGB")
                    img = img.convert("RGB")

            # Optional resize (keeps aspect ratio)
            if max_crop_size:
                w, h = img.size
                if w > max_crop_size[0] or h > max_crop_size[1]:
                    img.thumbnail(max_crop_size, Image.LANCZOS)

            # Mirror subdirectory structure, change extension to .png
            rel = src.relative_to(input_dir)
            dst = (output_dir / rel).with_suffix(".png")
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                return f"SKIP (exists) {dst}"

            img.save(dst, format="PNG")
            return f"OK {src} -> {dst}"
    except UnidentifiedImageError:
        return f"SKIP (unidentified image) {src}"
    except Exception as e:
        return f"ERROR {src}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Prepare groundtruth images (recursive, ICC→sRGB, PNG out).")
    parser.add_argument("input_dir", type=Path, help="Directory containing input images (may have subdirectories).")
    parser.add_argument("output_dir", type=Path, help="Directory to save processed PNGs (subdirs mirrored).")
    parser.add_argument("--max_crop_size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Maximum size (width height); images larger than this are downscaled proportionally.")
    parser.add_argument("--workers", type=int, default=32, help="Number of worker threads (default: 32).")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    max_crop_size = tuple(args.max_crop_size) if args.max_crop_size else None

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files recursively
    src_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not src_files:
        print("No images found in input_dir.")
        return

    print(f"Found {len(src_files)} images under {input_dir}. Processing...")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_image, p, input_dir, output_dir, max_crop_size) for p in src_files]
        for fut in as_completed(futures):
            msg = fut.result()
            if msg:
                print(msg)

if __name__ == "__main__":
    main()
