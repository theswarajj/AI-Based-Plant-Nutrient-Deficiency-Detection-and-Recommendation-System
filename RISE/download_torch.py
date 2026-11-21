"""
Standalone downloader for ImageNet-pretrained EfficientNetV2-S (PyTorch / timm).
Works with native Windows + NVIDIA GPU (unlike TensorFlow >= 2.11).

Install (CUDA 12.1 build - check https://pytorch.org/get-started/locally/
 if your driver needs a different CUDA version):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm

Run:
    python download_model_torch.py
"""
import os
import torch
import timm

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Downloading EfficientNetV2-S (ImageNet-21k -> 1k fine-tuned weights)...")
# num_classes=0 -> backbone only, no classifier head (we add our own later)
model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k",
                           pretrained=True, num_classes=0)

torch.save(model.state_dict(), os.path.join(OUT_DIR, "efficientnetv2s_imagenet.pt"))

n_params = sum(p.numel() for p in model.parameters())
print(f"Done. Parameters: {n_params:,}")
print(f"Saved in: {os.path.abspath(OUT_DIR)}")