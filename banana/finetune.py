"""
Improved fine-tuning for the banana nutrient deficiency classifier (8 classes,
Manganese excluded). Two key upgrades over the previous version, aimed
directly at the imbalance problems you saw (Iron over-predicted / low
precision, Zinc under-predicted / low recall):

  1. WEIGHTED RANDOM SAMPLING on top of (softer) class-weighted loss.
     Previously, imbalance was handled ONLY through loss weighting, capped
     at 5.0x. That cap under-corrects for Iron (58 images) and barely
     touches Zinc (276 images, near-neutral weight). A weighted sampler
     physically shows the model more images of rare classes each epoch,
     which is a stronger and more direct fix than loss weighting alone.

  2. MODEL SELECTION BY MACRO-F1, not accuracy. Accuracy is dominated by
     the big classes (Healthy=663, Sulphur=486). Macro-F1 averages
     per-class F1 equally, so the checkpoint we keep is the one that's
     genuinely best ACROSS ALL 8 classes, not just the popular ones.

Folder layout expected (same as before):

    BANANA/
      models/efficientnetv2s_imagenet.pt
      processed/train/<Class>/*.jpg
      processed/val/<Class>/*.jpg
      processed/test/<Class>/*.jpg
      finetune_torch.py   <- this file

Install once:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm scikit-learn

Run:
    python finetune_torch.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ------------------------------ config ---------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_PATH = os.path.join(BASE, "models", "efficientnetv2s_imagenet.pt")
DATA_DIR = os.path.join(BASE, "processed")
OUT_DIR = os.path.join(BASE, "trained_model")
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE_CLASSES = {"Manganese"}   # too few samples (16 train / 4 test)

IMG_SIZE = 384
BATCH = 16
SEED = 42
EPOCHS_HEAD = 10
EPOCHS_FINE = 60            # raised ceiling; patience decides the real length
UNFREEZE_LAST_BLOCKS = 4
PATIENCE = 12
NUM_WORKERS = 0
TTA = True
SAMPLER_POWER = 0.5         # sampling weight = 1/count^SAMPLER_POWER. 1.0 = full
                              # inverse-frequency (what caused the last run's
                              # over-correction); 0.5 = gentler, partial rebalancing.
                              # Loss weighting is now OFF entirely - the sampler
                              # is the ONLY imbalance correction, to avoid the
                              # double-counting that hurt the previous run.

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device,
      "-", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")


# ------------------------------ data ------------------------------------
def make_filtered_imagefolder(split, transform):
    full = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=transform)
    keep_classes = [c for c in full.classes if c not in EXCLUDE_CLASSES]
    old_to_new = {full.class_to_idx[c]: i for i, c in enumerate(keep_classes)}

    kept_samples = [(path, old_to_new[label])
                     for path, label in full.samples if label in old_to_new]

    full.samples = kept_samples
    full.targets = [lbl for _, lbl in kept_samples]
    full.classes = keep_classes
    full.class_to_idx = {c: i for i, c in enumerate(keep_classes)}
    return full


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Slightly stronger augmentation than before, since we now see rare-class
# images more often (via the sampler) and want to avoid memorizing them.
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = make_filtered_imagefolder("train", train_tf)
val_ds = make_filtered_imagefolder("val", eval_tf)
test_ds = make_filtered_imagefolder("test", eval_tf)

class_names = train_ds.classes
print("Classes (Manganese excluded):", class_names)

counts = np.bincount([label for _, label in train_ds.samples], minlength=len(class_names))
print("Train counts:", dict(zip(class_names, counts)))

# ---- weighted sampler: the ONLY imbalance correction (partial, not full) ----
# power=1.0 would fully equalize every class's effective frequency (this is
# what over-corrected last time, especially hurting Healthy and Sulphur).
# power=0.5 partially rebalances: rare classes are seen more often, but not
# so aggressively that the model forgets the common classes.
sample_weights_per_class = 1.0 / (counts ** SAMPLER_POWER)
sample_weights = np.array([sample_weights_per_class[label]
                            for _, label in train_ds.samples])
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(train_ds),
    replacement=True,
)
effective_freq = counts / (counts ** SAMPLER_POWER)
effective_freq = effective_freq / effective_freq.sum()
print("Approx. effective class share per epoch (vs raw):")
for c, raw, eff in zip(class_names, counts / counts.sum(), effective_freq):
    print(f"  {c:10s} raw={raw:.3f}  effective~={eff:.3f}")

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                           num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# No class-weighted loss this time - the sampler is the only correction,
# to avoid stacking two corrections on top of each other.
class_weight = None

# ------------------------------ model ------------------------------------
backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k",
                              pretrained=False, num_classes=0)
backbone.load_state_dict(torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=True))


class NutrientModel(nn.Module):
    def __init__(self, backbone, n_classes, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(self.dropout(feats))


feat_dim = backbone.num_features
model = NutrientModel(backbone, len(class_names), feat_dim).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


def run_epoch(loader, train, optimizer=None):
    model.train(mode=train)
    total_loss, correct, n = 0.0, 0, 0
    all_true, all_pred = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(imgs)
                loss = criterion(out, labels)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / n, correct / n, macro_f1


def evaluate(loader):
    return run_epoch(loader, train=False)


# ------------------------- stage 1: train head ----------------------------
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
print("\n--- Stage 1: training classifier head (backbone frozen) ---")
for epoch in range(EPOCHS_HEAD):
    tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, train=True, optimizer=optimizer)
    val_loss, val_acc, val_f1 = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS_HEAD}  train_loss={tr_loss:.4f} "
          f"train_acc={tr_acc:.4f} train_f1={tr_f1:.4f}  "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

# ------------------------- stage 2: fine-tune -----------------------------
for p in model.backbone.parameters():
    p.requires_grad = False
blocks = list(model.backbone.blocks)
for block in blocks[-UNFREEZE_LAST_BLOCKS:]:
    for p in block.parameters():
        p.requires_grad = True
for p in model.backbone.conv_head.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.3, patience=3, min_lr=1e-7)

best_val_f1, patience_left = 0.0, PATIENCE
print("\n--- Stage 2: fine-tuning top backbone layers ---")
for epoch in range(EPOCHS_FINE):
    tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, train=True, optimizer=optimizer)
    val_loss, val_acc, val_f1 = evaluate(val_loader)
    scheduler.step(val_f1)
    print(f"Epoch {epoch+1}/{EPOCHS_FINE}  train_loss={tr_loss:.4f} "
          f"train_acc={tr_acc:.4f} train_f1={tr_f1:.4f}  "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_left = PATIENCE
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
        print(f"  -> new best (val_f1={val_f1:.4f}), saved to trained_model/best.pt")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print("Early stopping.")
            break

# ------------------------------ evaluate ----------------------------------
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt"), weights_only=True))
model.eval()
print(f"\nLoaded best checkpoint (val macro-F1 = {best_val_f1:.4f})")


def predict_probs(imgs):
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(imgs)
        return torch.softmax(out, dim=1)


@torch.no_grad()
def evaluate_test(loader, tta=False):
    y_true, y_pred = [], []
    total_correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        probs = predict_probs(imgs)
        if tta:
            probs_flip = predict_probs(torch.flip(imgs, dims=[3]))
            probs = (probs + probs_flip) / 2
        preds = probs.argmax(1)
        total_correct += (preds == labels).sum().item()
        n += imgs.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    return total_correct / n, y_true, y_pred


test_acc, y_true, y_pred = evaluate_test(test_loader, tta=TTA)
test_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"\nTest accuracy{' (with TTA)' if TTA else ''}: {test_acc:.4f}")
print(f"Test macro-F1: {test_macro_f1:.4f}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion matrix (rows=true, cols=predicted):")
print(class_names)
print(confusion_matrix(y_true, y_pred))

# ------------------------------ save ---------------------------------------
# best.pt (saved during training, selected by val macro-F1) is already the
# best checkpoint. We additionally save a copy under the final descriptive
# name, and the label list, for use in your FastAPI backend later.
final_path = os.path.join(OUT_DIR, "banana_npk_effnetv2s_best.pt")
torch.save(model.state_dict(), final_path)
with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
    json.dump(class_names, f)
print(f"\nBest model saved to: {final_path}")
print(f"(Training checkpoint also kept at: {os.path.join(OUT_DIR, 'best.pt')})")
print(f"Labels saved to: {os.path.join(OUT_DIR, 'labels.json')}")


# --------------------------- inference helper -------------------------------
def predict(image_path, tta=TTA):
    from PIL import Image
    model.eval()
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)
    probs = predict_probs(x)
    if tta:
        probs_flip = predict_probs(torch.flip(x, dims=[3]))
        probs = (probs + probs_flip) / 2
    probs = probs[0].cpu().numpy()
    i = int(probs.argmax())
    return class_names[i], float(probs[i])