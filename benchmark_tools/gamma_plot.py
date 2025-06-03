import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the conversion functions
def srgb_to_linear(t):
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(t):
    return torch.where(t <= 0.0031308, t * 12.92, 1.055 * (t ** (1.0 / 2.4)) - 0.055)

def srgb_to_linear_approx(t):
    return t ** 2.2

def linear_to_srgb_approx(t):
    return t ** (1.0 / 2.2)

# Prepare data
uint8_vals_np = np.linspace(0, 255, 256)
srgb_vals_np = uint8_vals_np / 255.0

# Convert to torch tensors in FP32 and FP16
srgb_fp32 = torch.tensor(srgb_vals_np, dtype=torch.float32)
srgb_fp16 = torch.tensor(srgb_vals_np, dtype=torch.float16)

# Compute results
results = {
    "srgb_to_linear": {
        "fp32": srgb_to_linear(srgb_fp32),
        "fp16": srgb_to_linear(srgb_fp16).to(dtype=torch.float32)
    },
    "srgb_to_linear_approx": {
        "fp32": srgb_to_linear_approx(srgb_fp32),
        "fp16": srgb_to_linear_approx(srgb_fp16).to(dtype=torch.float32)
    },
    "linear_to_srgb": {
        "fp32": linear_to_srgb(srgb_fp32),
        "fp16": linear_to_srgb(srgb_fp16).to(dtype=torch.float32)
    },
    "linear_to_srgb_approx": {
        "fp32": linear_to_srgb_approx(srgb_fp32),
        "fp16": linear_to_srgb_approx(srgb_fp16).to(dtype=torch.float32)
    }
}

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# sRGB to Linear
axs[0].plot(uint8_vals_np, results["srgb_to_linear"]["fp32"], label="Exact (FP32)", color="black")
axs[0].plot(uint8_vals_np, results["srgb_to_linear_approx"]["fp32"], label="Approx (FP32)", linestyle="--", color="blue")
axs[0].plot(uint8_vals_np, results["srgb_to_linear"]["fp16"], label="Exact (FP16)", linestyle=":", color="gray")
axs[0].plot(uint8_vals_np, results["srgb_to_linear_approx"]["fp16"], label="Approx (FP16)", linestyle=":", color="cyan")
axs[0].set_title("sRGB to Linear")
axs[0].set_xlabel("sRGB Value (uint8)")
axs[0].set_ylabel("Linear Value")
axs[0].grid(True)
axs[0].legend()

# Linear to sRGB
axs[1].plot(uint8_vals_np, results["linear_to_srgb"]["fp32"], label="Exact (FP32)", color="black")
axs[1].plot(uint8_vals_np, results["linear_to_srgb_approx"]["fp32"], label="Approx (FP32)", linestyle="--", color="blue")
axs[1].plot(uint8_vals_np, results["linear_to_srgb"]["fp16"], label="Exact (FP16)", linestyle=":", color="gray")
axs[1].plot(uint8_vals_np, results["linear_to_srgb_approx"]["fp16"], label="Approx (FP16)", linestyle=":", color="cyan")
axs[1].set_title("Linear to sRGB")
axs[1].set_xlabel("Linear Value (mapped from uint8 domain)")
axs[1].set_ylabel("sRGB Value")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

