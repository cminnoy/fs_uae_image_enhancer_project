import os
import argparse
from PIL import Image, ImageCms
from concurrent.futures import ThreadPoolExecutor

file_formats = [".jpg", ".jpeg", ".webp", ".png", ".gif", ".tiff"] 
# target_width, target_height = 752, 576

def process_image(filename, input_dir, output_dir, max_crop_size):
    if not os.path.isfile(os.path.join(input_dir, filename)):
        return
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in file_formats:
        return
    
    try:
        img = Image.open(os.path.join(input_dir, filename))
        icc = img.info.get('icc_profile', None)
        width, height = img.size        

        if img.mode != 'RGB':
            print(f"Converting {filename} to RGB")
            img = img.convert('RGB')

        if icc:
            print(f"ERROR for file {input_dir}/{filename}; ICC profile found but not supported!")

        # Resize if max_crop_size is set and both dimensions exceed it
        if max_crop_size and (width > max_crop_size[0] or height > max_crop_size[1]):
            img.thumbnail(max_crop_size, Image.LANCZOS)
            width, height = img.size

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        # if width < target_width or height < target_height:
        #     bg = Image.new('RGB', (target_width, target_height), color=(0, 0, 0))
        #     bg.paste(img)
        #     bg.save(output_path, format='PNG')
        # else:
        img.save(output_path, format='PNG')
    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare groundtruth images.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save processed images.")
    parser.add_argument("--max_crop_size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help="Maximum crop size (width height) for resizing images.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_crop_size = tuple(args.max_crop_size) if args.max_crop_size else None

    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(lambda f: process_image(f, input_dir, output_dir, max_crop_size), files)

if __name__ == "__main__":
    main()
