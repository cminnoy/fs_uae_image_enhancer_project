import os
from PIL import Image

class Rgb444Generator:
     def __init__(self, width=376, height=288, output_dir='rgb444_images'):
         self.width = width
         self.height = height
         self.output_dir = output_dir
         os.makedirs(self.output_dir, exist_ok=True)

     def generate_color_image(self, r, g, b):
         """Generates a single-color RGB image."""
         image = Image.new('RGB', (self.width, self.height), (r, g, b))
         return image

     def get_hex_color(self, r, g, b):
        """Converts RGB components to a 4-digit hexadecimal string."""
        r_hex = hex(r >> 0 & 0xF)[2:].zfill(1)
        g_hex = hex(g >> 0 & 0xF)[2:].zfill(1)
        b_hex = hex(b >> 0 & 0xF)[2:].zfill(1)
        return f"{r_hex}{g_hex}{b_hex}{'0'}" # Adding '0' to make it 4 digits
 
     def generate_all_rgb444(self):
        """Generates an image for each RGB444 color."""
        for r in range(16):
            for g in range(16):
                for b in range(16):
                    r_scaled = (r * 16) + r  # Scale 4-bit to 8-bit range
                    g_scaled = (g * 16) + g
                    b_scaled = (b * 16) + b
                    image = self.generate_color_image(r_scaled, g_scaled, b_scaled)
                    hex_color = self.get_hex_color(r, g, b)
                    filename = os.path.join(self.output_dir, f'rgb444_{hex_color}.png')
                    image.save(filename)
                    print(f"Generated: {filename}")
 
if __name__ == "__main__":
    generator = Rgb444Generator()
    generator.generate_all_rgb444()
    print("Finished generating RGB444 color images.")