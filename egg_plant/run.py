"""
Eggplant nutrient prediction with the 5-model cross-validation ensemble.

Uses the five fold models saved by cv_ensemble.py (lr_fine = 5e-5):

    egg_plant/
      trained_model_cv_lr5e-05/fold1.pt ... fold5.pt
      trained_model_cv_lr5e-05/labels.json
      predict.py   <- this file
      test.jpg     <- default input image

Install once (if not already):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm pillow

Run:
    python predict.py                      # classify test.jpg
    python predict.py path\\to\\leaf.jpg     # classify another image
    python predict.py --check              # run on processed\\test and report how
                                           # the warnings behave (see below)
    python predict.py --check some\\folder  # same, for a folder laid out as
                                           # <folder>\\<ClassName>\\*.jpg

How it predicts:
  - The image is padded to a square on a black 384x384 canvas, exactly like
    the training images (earlier predict scripts stretched the image instead,
    which did not match training).
  - Each of the 5 models gives probabilities for the image and its horizontal
    flip. All 10 are averaged.
  - "Model agreement" is how many of the 5 models picked the top class. It is
    printed as extra information only.

Warnings never change the predicted class. They only say the result is shaky.
  - LOW CONFIDENCE: the top probability is under LOW_CONFIDENCE_THRESHOLD.
  - UNCERTAIN PAIR: the top class belongs to a pair the model often mixes up
    and the other class of the pair is within CONFUSION_MARGIN. The pairs come
    from the cross-validation confusion matrix (313 images): every pair with
    at least 15 mistakes in the two directions combined.

The two thresholds are round numbers that were never tuned. Use --check to see
how they behave. Do NOT adjust them to look good on the test images: those 56
images are your only clean check.
"""
import os
import sys
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageOps
from torchvision import transforms

# ------------------------------ config ---------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "trained_model_cv_lr5e-05")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
TEST_IMAGE = os.path.join(BASE, "test.jpg")
DEFAULT_CHECK_FOLDER = os.path.join(BASE, "processed", "test")

IMG_SIZE = 384
TTA = True
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cross-validation confusion matrix (5e-5 run, rows = true, cols = predicted):
#   ['Healthy', 'Nitrogen', 'Nitrogen_Potassium', 'Potassium']
#   [[57 11  2  7]
#    [ 5 46  6  0]
#    [10 12 42 26]
#    [ 5  1 14 69]]
# Mistakes in both directions combined:
#   Nitrogen_Potassium <-> Potassium : 26 + 14 = 40
#   Nitrogen_Potassium <-> Nitrogen  : 12 +  6 = 18
#   Healthy            <-> Nitrogen  : 11 +  5 = 16
#   Healthy <-> N_K (12) and Healthy <-> Potassium (12) fall below the cut.
CONFUSABLE_PAIRS = [
    ("Nitrogen_Potassium", "Potassium"),
    ("Nitrogen_Potassium", "Nitrogen"),
    ("Healthy", "Nitrogen"),
]
CONFUSION_MARGIN = 0.20
LOW_CONFIDENCE_THRESHOLD = 0.45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"


# ------------------------------ model ------------------------------------
class NutrientModel(nn.Module):
    def __init__(self, backbone, n_classes, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))


_MODELS = None


def get_models():
    """Load the 5 fold models once and keep them in memory."""
    global _MODELS
    if _MODELS is None:
        paths = sorted(glob.glob(os.path.join(MODEL_DIR, "fold*.pt")))
        if not paths:
            raise SystemExit(f"No fold*.pt files found in {MODEL_DIR}")
        with open(LABELS_PATH) as f:
            names = json.load(f)
        models = []
        for p in paths:
            backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k",
                                         pretrained=False, num_classes=0)
            m = NutrientModel(backbone, len(names), backbone.num_features).to(device)
            m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            m.eval()
            models.append(m)
        print(f"Loaded {len(models)} models from {MODEL_DIR}")
        _MODELS = (models, names)
    return _MODELS


# --------------------------- image preparation -----------------------------
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# BEGIN LOAD_LEAF
def load_leaf(path):
    """Same preparation as preprocessing: EXIF fix, RGB, fit inside 384x384
    keeping the aspect ratio, centred on a black square canvas."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.contain(img, (IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    canvas.paste(img, ((IMG_SIZE - img.width) // 2, (IMG_SIZE - img.height) // 2))
    return canvas
# END LOAD_LEAF


# BEGIN WARNING
def build_warning(ranked):
    """ranked: list of (class_name, probability), best first.
    Returns a warning string, or None if nothing looks shaky."""
    top_class, top_prob = ranked[0]

    if top_prob < LOW_CONFIDENCE_THRESHOLD:
        return (f"LOW CONFIDENCE ({top_prob:.1%}). The models are not sure about "
                f"this image. Retake the photo (better light, leaf filling the "
                f"frame) or get a second opinion.")

    prob_by_class = dict(ranked)
    best_warning, smallest_margin = None, None
    for a, b in CONFUSABLE_PAIRS:
        if top_class not in (a, b):
            continue
        other = b if top_class == a else a
        other_prob = prob_by_class.get(other, 0.0)
        margin = top_prob - other_prob
        if margin < CONFUSION_MARGIN and (smallest_margin is None
                                          or margin < smallest_margin):
            smallest_margin = margin
            best_warning = (
                f"UNCERTAIN: '{top_class}' and '{other}' are often confused by "
                f"this model ({top_prob:.1%} vs {other_prob:.1%}). Treat the "
                f"result as inconclusive: inspect the leaf closely or take a "
                f"follow-up photo before applying any fertilizer.")
    return best_warning
# END WARNING


# ------------------------------ prediction ---------------------------------
@torch.no_grad()
def predict_image(path):
    """Returns a dict: prediction, confidence, probabilities (ranked),
    votes (models agreeing with the top class), n_models, warning."""
    models, names = get_models()
    x = to_tensor(load_leaf(path)).unsqueeze(0).to(device)

    per_model = []
    for m in models:
        with torch.amp.autocast("cuda", enabled=use_amp):
            p = torch.softmax(m(x), dim=1)
            if TTA:
                p = (p + torch.softmax(m(torch.flip(x, dims=[3])), dim=1)) / 2
        per_model.append(p[0].float().cpu().numpy())
    per_model = np.stack(per_model)

    probs = per_model.mean(axis=0)
    ranked = sorted(zip(names, probs.tolist()), key=lambda t: t[1], reverse=True)
    top_class, top_prob = ranked[0]
    votes = int((per_model.argmax(axis=1) == names.index(top_class)).sum())
    return {
        "prediction": top_class,
        "confidence": top_prob,
        "probabilities": ranked,
        "votes": votes,
        "n_models": len(models),
        "warning": build_warning(ranked),
    }


def show(path, result):
    print(f"\nImage: {path}")
    print("Ensemble probabilities:")
    for cls, p in result["probabilities"]:
        print(f"  {cls:20s} {p:.1%}")
    print(f"\n>> Predicted class: {result['prediction']}  "
          f"(confidence: {result['confidence']:.1%})")
    print(f"   Model agreement: {result['votes']} of {result['n_models']} models "
          f"chose this class")
    if result["warning"]:
        print(f"\n[WARNING] {result['warning']}")
    else:
        print("\nNo warning.")


# ------------------------------- check mode ---------------------------------
def check_folder(folder):
    """Run on <folder>/<ClassName>/*.jpg and report how the warnings behave."""
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")
    rows = []
    for cls in sorted(os.listdir(folder)):
        d = os.path.join(folder, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                r = predict_image(os.path.join(d, fn))
                rows.append((cls, r["prediction"], r["warning"] is not None))
    if not rows:
        raise SystemExit("No images found.")

    def acc(subset):
        return sum(t == p for t, p, _ in subset) / len(subset) if subset else float("nan")

    flagged = [r for r in rows if r[2]]
    clear = [r for r in rows if not r[2]]
    print(f"\nChecked {len(rows)} images from {folder}")
    print(f"Overall accuracy:            {acc(rows):.1%}  ({len(rows)} images)")
    print(f"Warned predictions:          {len(flagged)}  "
          f"accuracy among them: {acc(flagged):.1%}")
    print(f"Predictions without warning: {len(clear)}  "
          f"accuracy among them: {acc(clear):.1%}")
    print("\nA useful warning flags the shaky predictions: accuracy among the "
          "warned images should be clearly lower than among the others. With "
          "this few images the difference is rough. Do not tune the thresholds "
          "on this set.")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--check":
        check_folder(args[1] if len(args) > 1 else DEFAULT_CHECK_FOLDER)
        return
    image_path = args[0] if args else TEST_IMAGE
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")
    show(image_path, predict_image(image_path))


if __name__ == "__main__":
    main()