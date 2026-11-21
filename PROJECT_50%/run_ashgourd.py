"""
Run inference on a single leaf image using the trained ash gourd model,
and warn the user when the prediction falls into a pair of classes that
are known to be visually hard to separate (e.g. Healthy vs Potassium -
confirmed by manual inspection, see confusion matrix).

Folder layout expected:

    S1/
      trained_model_ashgourd/ashgourd_npk_effnetv2s_best.pt
      trained_model_ashgourd/labels.json
      INPUT/ashgourd/          <- place input images here
      OUTPUT/ashgourd/         <- results are saved here
      run_ashgourd.py          <- this file

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
import glob
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

# Visualisation & explainability utilities
from utils.gradcam import GradCAM, get_target_layer, create_combined_heatmap
from utils.gradcam import compute_integrated_gradients, save_attribution_image
from utils.visualization import create_dashboard
from utils.recommendations import get_all_recommendations
from utils.selection import select_image_interactive, list_input_images

# ------------------------------ config ---------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "trained_model_ashgourd", "ashgourd_npk_effnetv2s_best.pt")
LABELS_PATH = os.path.join(BASE, "trained_model_ashgourd", "labels.json")
INPUT_DIR = os.path.join(BASE, "INPUT", "ashgourd")
OUTPUT_DIR = os.path.join(BASE, "OUTPUT", "ashgourd")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# Set True to also generate Integrated Gradients (SHAP-like) attribution
# images. This is slower (~50 extra forward passes per class) but provides
# pixel-level explanations complementing the Grad-CAM heatmaps.
ENABLE_ATTRIBUTIONS = True

CROP_KEY = "ashgourd"          # key in recommendations.py
CROP_DISPLAY = "Ash Gourd"     # human-readable name for dashboard title

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


def process_single_image(model, class_names, gradcam_engine, image_path):
    """Classify, generate Grad-CAM heatmaps, and create analysis dashboard."""
    # 1. Classification (with TTA)
    ranked = classify_image(model, class_names, image_path)

    # 2. Grad-CAM heatmaps for top-3 classes
    img = Image.open(image_path).convert("RGB")
    input_tensor = eval_tf(img).unsqueeze(0).to(device)

    top_n = min(3, len(ranked))
    cam_maps = {}
    for cls_name, _ in ranked[:top_n]:
        cls_idx = class_names.index(cls_name)
        cam_maps[cls_name] = gradcam_engine.generate(input_tensor, cls_idx)

    combined_cam = create_combined_heatmap(cam_maps, ranked[:top_n])

    # 3. Dashboard image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dashboard_path = os.path.join(OUTPUT_DIR, f"{base_name}_dashboard.png")
    recs = get_all_recommendations(CROP_KEY)
    try:
        create_dashboard(
            original_img=img, cam_maps=cam_maps,
            combined_cam=combined_cam, ranked=ranked,
            crop_name=CROP_DISPLAY, image_name=os.path.basename(image_path),
            output_path=dashboard_path, recommendations_dict=recs,
        )
        print(f"  Dashboard -> {dashboard_path}")
    except Exception as e:
        print(f"  [WARN] Dashboard failed: {e}")

    # 4. (Optional) Integrated-Gradients / SHAP-like attribution
    if ENABLE_ATTRIBUTIONS:
        try:
            top_cls_idx = class_names.index(ranked[0][0])
            attr_map = compute_integrated_gradients(
                model, input_tensor, top_cls_idx, n_steps=50, device=device,
            )
            attr_path = os.path.join(OUTPUT_DIR, f"{base_name}_attributions.png")
            save_attribution_image(
                img, attr_map, attr_path,
                title=f"Integrated Gradients — {ranked[0][0]} ({ranked[0][1]:.1%})",
            )
            print(f"  Attributions -> {attr_path}")
        except Exception as e:
            print(f"  [WARN] Attribution failed: {e}")

    # 5. Text results
    lines = []
    lines.append(f"Image: {os.path.basename(image_path)}")
    lines.append("Prediction ranking:")
    for cls, prob in ranked:
        lines.append(f"  {cls:22s} {prob:.1%}")
    top_class, top_prob = ranked[0]
    lines.append(f"\n>> Predicted class: {top_class}  (confidence: {top_prob:.1%})")
    warning = build_warning(ranked)
    if warning:
        lines.append(f"[WARNING] {warning}")
    else:
        lines.append("No warning - this prediction looks reliable.")
    lines.append(f"Dashboard: {dashboard_path}")

    return "\n".join(lines), os.path.basename(image_path), top_class, top_prob


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--all", "-a"):
            image_paths = list_input_images(INPUT_DIR, IMAGE_EXTENSIONS)
            if not image_paths:
                raise SystemExit(f"No images found in {INPUT_DIR}")
        elif sys.argv[1] in ("--help", "-h"):
            print("Usage: python run_ashgourd.py [OPTIONS] [IMAGE_PATH ...]")
            print("\nOptions:")
            print("  --all, -a     Process all images in INPUT/ashgourd without prompting")
            print("  --help, -h    Show this help message and exit")
            print("\nIf no arguments are provided, an interactive image selection menu is shown.")
            return
        else:
            image_paths = sys.argv[1:]
    else:
        if not os.path.isdir(INPUT_DIR):
            raise SystemExit(f"Input folder not found: {INPUT_DIR}")
        all_images = list_input_images(INPUT_DIR, IMAGE_EXTENSIONS)
        if not all_images:
            raise SystemExit(f"No images found in {INPUT_DIR}")
        image_paths = select_image_interactive(
            all_images,
            species_name=CROP_DISPLAY,
            folder_display=f"INPUT/{os.path.basename(INPUT_DIR)}",
        )

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, class_names = load_model()

    # Initialise Grad-CAM on the last spatial feature layer
    target_layer = get_target_layer(model.backbone)
    gradcam_engine = GradCAM(model, target_layer)
    print(f"Grad-CAM target layer: {target_layer.__class__.__name__}")

    all_results = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"[SKIP] Image not found: {image_path}")
            continue

        result_text, img_name, top_class, top_prob = process_single_image(
            model, class_names, gradcam_engine, image_path
        )
        all_results.append(result_text)
        print(f"\n{'='*60}")
        print(result_text)

        # Save individual result
        base_name = os.path.splitext(img_name)[0]
        out_file = os.path.join(OUTPUT_DIR, f"{base_name}_result.txt")
        with open(out_file, "w") as f:
            f.write(result_text)
        print(f"Result saved to: {out_file}")

    # Cleanup hooks
    gradcam_engine.remove_hooks()

    # Save combined summary
    summary_path = os.path.join(OUTPUT_DIR, "ashgourd_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Ash Gourd Prediction Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total images processed: {len(all_results)}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"\n{'='*60}\n".join(all_results))

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(all_results)} image(s).")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()