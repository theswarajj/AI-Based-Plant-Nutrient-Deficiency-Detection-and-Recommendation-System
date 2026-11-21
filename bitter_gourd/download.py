"""
Standalone downloader for ImageNet-pretrained EfficientNetV2-S (PyTorch / timm).
Same backbone used for rice, banana, ash gourd, and tomato - reused here for
bitter gourd.

If you already have models/efficientnetv2s_imagenet.pt from an earlier crop,
skip this script and just copy it instead:

    mkdir BITTERGOURD\models
    copy tomato\models\efficientnetv2s_imagenet.pt BITTERGOURD\models\

Install (only if not already installed):
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
model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k",
                           pretrained=True, num_classes=0)

out_path = os.path.join(OUT_DIR, "efficientnetv2s_imagenet.pt")
torch.save(model.state_dict(), out_path)

n_params = sum(p.numel() for p in model.parameters())
print(f"Done. Parameters: {n_params:,}")
print(f"Saved to: {os.path.abspath(out_path)}")