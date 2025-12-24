import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model_residual_unet import get_model

class AmigaEnhancer:
    def __init__(self, model_type, checkpoint_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load Architecture
        self.model = get_model(model_type).to(self.device)
        
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

    def _srgb_to_linear(self, x):
        """ Conversion required for FS-UAE framebuffers """
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def _linear_to_srgb(self, x):
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

    def process(self, input_path, output_path):
        with Image.open(input_path).convert("RGB") as img:
            img_np = np.array(img).astype(np.float32) / 255.0

        img_linear = self._srgb_to_linear(img_np)
        input_tensor = torch.from_numpy(img_linear).permute(2, 0, 1).unsqueeze(0).half().to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        output = output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        output = self._linear_to_srgb(np.clip(output, 0, 1))
        
        Image.fromarray((output * 255.0).astype(np.uint8)).save(output_path)
        print(f"Processed {input_path.name} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Amiga Image Enhancer Inference')
    parser.add_argument('--type', type=str, required=True, choices=['light', 'heavy'])
    parser.add_argument('--input', type=str, required=True, help='Filename in samples/ or full path')
    parser.add_argument('--output', type=str, required=True, help='Output filename')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    ckpt_path = base_dir / args.type / "best_model.pth"
    
    # Resolve input path: handle both 'sample0.png' and 'samples/sample0.png'
    in_path = Path(args.input)
    if not in_path.exists():
        in_path = base_dir / "samples" / in_path.name

    if not ckpt_path.exists():
        print(f"Error: Weight file not found at {ckpt_path}")
        return

    enhancer = AmigaEnhancer(args.type, ckpt_path)
    enhancer.process(in_path, Path(args.output))

if __name__ == "__main__":
    main()