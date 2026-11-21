"""
Run inference on a single leaf image using the trained ash gourd model,
and warn the user when the prediction falls into a pair of classes that
are known to be visually hard to separate (e.g. Healthy vs Potassium -
confirmed by manual inspection, see confusion matrix).

Folder layout expected:

    ashgourd/
      trained_model/ashgourd_npk_effnetv2s_best.pt
      trained_model/labels.json
      predict.py   <- this file
      test.jpg     <- the image to classify (change TEST_IMAGE below,
                       or pass a path as a command-line argument)

Install once (if not already):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm pillow

Run:
    python predict.py
    python predict.py path\to\some_other_leaf.jpg
"""
import os
import sys
import json
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

# ------------------------------ config ---------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "trained_model", "ashgourd_npk_effnetv2s_best.pt")
LABELS_PATH = os.path.join(BASE, "trained_model", "labels.json")
TEST_IMAGE = os.path.join(BASE, "test.jpg")   # default input if none given on CLI

IMG_SIZE = 384
TTA = True

# Pairs of classes that are known to be visually hard to separate.
# Add more pairs here as you find them for other crops (e.g. banana had
# similar Iron/Zinc precision-recall issues, though not confirmed by eye).
CONFUSABLE_PAIRS = [
    ("Healthy", "Potassium"),
]

# If the top prediction is in a confusable pair AND the runner-up class in
# that same pair is within this probability margin, we warn instead of
# stating the prediction as certain.
CONFUSION_MARGIN = 0.20   # e.g. top=0.55, paired-class=0.40 -> margin 0.15 -> warn

# Below this, we warn regardless of class (generic low-confidence warning).
LOW_CONFIDENCE_THRESHOLD = 0.45

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------ model ------------------------------------
class NutrientModel(nn.Module):
    def __init__(self, backbone, n_classes, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(self.dropout(feats))


def load_model():
    with open(LABELS_PATH) as f:
        class_names = json.load(f)

    backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k",
                                  pretrained=False, num_classes=0)
    feat_dim = backbone.num_features
    model = NutrientModel(backbone, len(class_names), feat_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model, class_names


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def predict_probs(model, x):
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        return torch.softmax(out, dim=1)


def classify_image(model, class_names, image_path, tta=TTA):
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)

    probs = predict_probs(model, x)
    if tta:
        probs_flip = predict_probs(model, torch.flip(x, dims=[3]))
        probs = (probs + probs_flip) / 2
    probs = probs[0].cpu().numpy()

    ranked = sorted(zip(class_names, probs), key=lambda t: t[1], reverse=True)
    return ranked   # list of (class_name, probability), sorted descending


def build_warning(ranked):
    """Returns a warning string, or None if the prediction looks reliable."""
    top_class, top_prob = ranked[0]

    # 1. Generic low-confidence check (applies to any class).
    if top_prob < LOW_CONFIDENCE_THRESHOLD:
        return (f"LOW CONFIDENCE ({top_prob:.1%}). The model isn't sure about "
                f"this image - consider retaking the photo (better lighting, "
                f"fill the frame with the leaf) or getting a second opinion.")

    # 2. Confusable-pair check.
    prob_by_class = dict(ranked)
    for a, b in CONFUSABLE_PAIRS:
        if top_class not in (a, b):
            continue
        other = b if top_class == a else a
        other_prob = prob_by_class.get(other, 0.0)
        margin = top_prob - other_prob
        if margin < CONFUSION_MARGIN:
            return (f"UNCERTAIN: '{top_class}' and '{other}' are visually similar "
                    f"and often confused by this model "
                    f"({top_prob:.1%} vs {other_prob:.1%}). "
                    f"Treat this result as inconclusive - a closer visual "
                    f"inspection or a follow-up photo of the leaf margins is "
                    f"recommended before applying any fertilizer.")
    return None


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else TEST_IMAGE
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    model, class_names = load_model()
    ranked = classify_image(model, class_names, image_path)

    print(f"\nImage: {image_path}")
    print("Prediction ranking:")
    for cls, prob in ranked:
        print(f"  {cls:22s} {prob:.1%}")

    top_class, top_prob = ranked[0]
    print(f"\n>> Predicted class: {top_class}  (confidence: {top_prob:.1%})")

    warning = build_warning(ranked)
    if warning:
        print(f"\n[WARNING] {warning}")
    else:
        print("\nNo warning - this prediction looks reliable.")


if __name__ == "__main__":
    main()