import argparse
import torch
import numpy as np
import gamma
from PIL import Image
from pathlib import Path
from model_residual_unet import get_model

class AmigaEnhancer:
    def __init__(self, model_type, checkpoint_path, lores_only, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load Architecture
        self.model = get_model(model_type, lores_only).to(self.device)
        
        # Load Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model state from the full checkpoint dictionary
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Handle architecture mismatch (e.g., skip_alpha/perceptual keys)
        # We use strict=False to ignore training-only keys like perceptual_criterion
        self.model.load_state_dict(state_dict, strict=False)
        self.model.half().eval()

    def process(self, input_path, output_path):
        with Image.open(input_path).convert("RGB") as img:
            img_np = np.array(img).astype(np.float32) / 255.0
            input_srgb = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device).half()

        with torch.no_grad():
            output_srgb = self.model(input_srgb).clamp(0, 1)

        output_final = output_srgb.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        
        Image.fromarray((output_final * 255.0).astype(np.uint8)).save(output_path)
        print(f"Processed {input_path.name} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Amiga Image Enhancer Inference')
    parser.add_argument('--type', type=str, required=True, choices=['light', 'heavy'])
    parser.add_argument('--checkpoint', '-c', type=str, required=False,
                        help='Path to .pth checkpoint file. If omitted, defaults to model/<type>/best_model.pth')
    parser.add_argument('--input', type=str, required=True, help='Path to input image file')
    parser.add_argument('--output', type=str, required=True, help='Path to output image file or directory')
    parser.add_argument('--lores_only', action='store_true', help='Process only low-resolution input')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    # Determine checkpoint path: explicit override or default location
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = base_dir / args.type / "best_model.pth"

    if not ckpt_path.exists():
        print(f"Error: Weight file not found at {ckpt_path}")
        return

    # Resolve input path: require the provided path to exist (no implicit samples/ fallback)
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file not found at {in_path}")
        return

    # Resolve output path: if user provided a directory, save with input filename inside it
    out_path = Path(args.output)
    if out_path.exists() and out_path.is_dir():
        out_path = out_path / in_path.name
    else:
        # Ensure parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

    enhancer = AmigaEnhancer(args.type, ckpt_path, args.lores_only)
    enhancer.process(in_path, out_path)

if __name__ == "__main__":
    main()
