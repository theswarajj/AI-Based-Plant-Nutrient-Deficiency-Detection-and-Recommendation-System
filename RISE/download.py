"""
Standalone downloader for ImageNet-pretrained EfficientNetV2-S.

Install once:
    pip install tensorflow

Run:
    python download_model.py

Output:
    models/efficientnetv2s_imagenet.keras   (backbone, no classifier head)
    models/efficientnetv2s_imagenet.weights.h5
"""
import os
from tensorflow import keras

IMG_SIZE = 384
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading EfficientNetV2-S (ImageNet weights)...")
model = keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

model.save(os.path.join(OUT_DIR, "efficientnetv2s_imagenet.keras"))
model.save_weights(os.path.join(OUT_DIR, "efficientnetv2s_imagenet.weights.h5"))

print(f"Done. Parameters: {model.count_params():,}")
print(f"Saved in: {os.path.abspath(OUT_DIR)}")