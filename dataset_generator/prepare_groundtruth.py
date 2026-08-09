#!/usr/bin/env python3

import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageCms, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".png", ".gif", ".tif", ".tiff", ".bmp", ".avif"}

class ImageProcessor:
    def __init__(self, input_dir, output_dir, max_crop_size, pad_to_max=False, prefix_dir=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_crop_size = max_crop_size
        self.pad_to_max = pad_to_max
        self.prefix_dir = prefix_dir

    def process_image(self, src: Path):
        try:
            with Image.open(src) as img:
                # 1. Color Space Management
                icc_bytes = img.info.get("icc_profile")
                if icc_bytes:
                    try:
                        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                        dst_profile = ImageCms.createProfile("sRGB")
                        img = ImageCms.profileToProfile(img, src_profile, dst_profile, outputMode="RGB")
                    except Exception:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                else:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                # 2. Resize and Pad
                if self.max_crop_size:
                    target_w, target_h = self.max_crop_size
                    img.thumbnail(self.max_crop_size, Image.LANCZOS)
                    
                    if self.pad_to_max:
                        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                        curr_w, curr_h = img.size
                        offsets = ((target_w - curr_w) // 2, (target_h - curr_h) // 2)
                        new_img.paste(img, offsets)
                        img = new_img

                # 3. Path and Filename Logic
                rel_path = src.relative_to(self.input_dir)
                parent_dir_name = rel_path.parent.name
                
                # Apply prefix if requested and if a parent subdirectory exists
                if self.prefix_dir and parent_dir_name:
                    new_name = f"{parent_dir_name}_{src.stem}.png"
                else:
                    new_name = f"{src.stem}.png"

                dst = self.output_dir / rel_path.parent / new_name
                dst.parent.mkdir(parents=True, exist_ok=True)

                if dst.exists():
                    return f"SKIP (exists) {dst.name}"

                img.save(dst, format="PNG")
                return f"OK {src.name} -> {dst.name}"
                
        except UnidentifiedImageError:
            return f"SKIP (unidentified) {src}"
        except Exception as e:
            return f"ERROR {src}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Prepare groundtruth images with optional padding and prefixing.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max_crop_size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--pad", action="store_true", help="Pad to max_crop_size with black borders.")
    parser.add_argument("--prefix", action="store_true", help="Rename files to <subdir>_<original_name>.")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    max_size = tuple(args.max_crop_size) if args.max_crop_size else None
    processor = ImageProcessor(args.input_dir, args.output_dir, max_size, args.pad, args.prefix)

    src_files = [p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    
    if not src_files:
        print("No images found.")
        return

    print(f"Processing {len(src_files)} images...")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(processor.process_image, p) for p in src_files]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                print(res)

if __name__ == "__main__":
    main()
