"""
5-fold cross-validation + ensemble for the eggplant nutrient classifier.

WHY: with only 56 validation and 56 test images, a single train/val/test run
gives a very noisy score. This script gives two more trustworthy numbers:

  1. OUT-OF-FOLD (OOF) score on all 313 train+val images. Every image is
     predicted by a model that never saw it, so the score uses 313 images
     instead of 56.
  2. ENSEMBLE score on your original 56-image TEST split, which is never
     used for training or for any choice. The 5 fold models are averaged.
     The same test images were used in your earlier run, so the numbers are
     directly comparable.

NO CHECKPOINT SELECTION: each fold trains for a fixed number of epochs and
keeps the final weights. Picking the best epoch on the fold's validation
images would make the OOF score optimistic (the problem seen in the single
run, where validation looked better than test).

Fold splitting is plain stratified. That is safe for eggplant because
preprocessing found 0 near-duplicate pairs. For a crop that has near-duplicate
groups, use StratifiedGroupKFold with the group ids instead.

Folder layout expected:

    egg_plant/
      models/efficientnetv2s_imagenet.pt
      processed/train/<Class>/*.jpg
      processed/val/<Class>/*.jpg
      processed/test/<Class>/*.jpg
      cv_ensemble.py   <- this file

Install once:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm scikit-learn

Run:
    python cv_ensemble.py

LR_FINE is now 1e-4. Earlier runs used 1e-5 and then 5e-5, which was better.
Each run writes to its own folder (trained_model_cv_lr<value>), so nothing is
overwritten. When the run ends, the script compares itself with the saved
5e-5 report fold by fold and prints a verdict. Judge the setting by the OOF
macro-F1 and the per-fold comparison, not by the 56-image test score.
This is the last setting worth trying: every extra setting compared on the
same 313 images makes the best score look slightly better than it is.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

# ------------------------------ config ---------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_PATH = os.path.join(BASE, "models", "efficientnetv2s_imagenet.pt")
DATA_DIR = os.path.join(BASE, "processed")

N_FOLDS = 5
IMG_SIZE = 384
BATCH = 16
SEED = 42
EPOCHS_HEAD = 10          # stage 1: head only, backbone frozen
EPOCHS_FINE = 25          # stage 2: last blocks unfrozen, cosine LR decay
UNFREEZE_LAST_BLOCKS = 4
LR_HEAD = 1e-3
LR_FINE = 1e-4            # earlier runs: 1e-5, then 5e-5 (the better of the two)
# the report this run is compared with at the end (the current best setting)
BASELINE_REPORT = os.path.join(BASE, "trained_model_cv_lr5e-05", "cv_report.json")
SAMPLER_POWER = 0.5       # partial rebalancing, the only imbalance correction
NUM_WORKERS = 0           # 0 avoids the Windows "freeze_support" error
TTA = True                # average each prediction with its horizontal flip

OUT_DIR = os.path.join(BASE, f"trained_model_cv_lr{LR_FINE:g}")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print("Using device:", device,
      "-", torch.cuda.get_device_name(0) if use_amp else "CPU")

# ------------------------------ data ------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# No hue/saturation jitter: leaf colour IS the deficiency signal.
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


class PathDataset(Dataset):
    """Dataset over an explicit list of (path, label) pairs."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ImageFolder is used here only to list files and class names
tr_folder = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
va_folder = datasets.ImageFolder(os.path.join(DATA_DIR, "val"))
te_folder = datasets.ImageFolder(os.path.join(DATA_DIR, "test"))
assert tr_folder.classes == va_folder.classes == te_folder.classes, \
    "train/val/test class folders do not match - delete processed/ and re-run preprocessing"

class_names = tr_folder.classes
n_classes = len(class_names)
print("Classes:", class_names)

# train + val are merged and re-split into folds; test stays untouched
all_samples = tr_folder.samples + va_folder.samples
all_labels = np.array([lbl for _, lbl in all_samples])
test_samples = te_folder.samples
print(f"Images for cross-validation: {len(all_samples)}   "
      f"held-out test images: {len(test_samples)}")
print("Per-class counts (CV set):",
      dict(zip(class_names, np.bincount(all_labels, minlength=n_classes).tolist())))

test_loader = DataLoader(PathDataset(test_samples, eval_tf), batch_size=BATCH,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# ------------------------------ model ------------------------------------
class NutrientModel(nn.Module):
    def __init__(self, backbone, n_classes, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.dropout(self.backbone(x)))


def make_model():
    backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k",
                                 pretrained=False, num_classes=0)
    backbone.load_state_dict(
        torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=True))
    return NutrientModel(backbone, n_classes, backbone.num_features).to(device)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total, n = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(imgs), labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total / n


@torch.no_grad()
def get_probs(model, loader):
    """Softmax probabilities (flip-TTA if enabled) and true labels, in loader order."""
    model.eval()
    all_p, all_y = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            p = torch.softmax(model(imgs), dim=1)
            if TTA:
                p_flip = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
                p = (p + p_flip) / 2
        all_p.append(p.float().cpu().numpy())
        all_y.extend(labels.numpy().tolist())
    return np.concatenate(all_p), np.array(all_y)


def train_fold(tr_samples, fold):
    # partial weighted sampler built from THIS fold's training counts
    counts = np.bincount([lbl for _, lbl in tr_samples], minlength=n_classes)
    per_class_w = 1.0 / (np.maximum(counts, 1) ** SAMPLER_POWER)
    weights = np.array([per_class_w[lbl] for _, lbl in tr_samples])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(tr_samples), replacement=True)
    loader = DataLoader(PathDataset(tr_samples, train_tf), batch_size=BATCH,
                        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)

    model = make_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # stage 1: classifier head only
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(model.head.parameters(), lr=LR_HEAD)
    for e in range(EPOCHS_HEAD):
        loss = train_epoch(model, loader, opt, criterion, scaler)
        if (e + 1) % 5 == 0 or e == EPOCHS_HEAD - 1:
            print(f"  fold {fold} stage1 epoch {e+1}/{EPOCHS_HEAD}  loss={loss:.4f}")

    # stage 2: last blocks + conv_head, cosine LR decay, fixed length
    for p in model.backbone.parameters():
        p.requires_grad = False
    for block in list(model.backbone.blocks)[-UNFREEZE_LAST_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.backbone.conv_head.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR_FINE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS_FINE, eta_min=1e-7)
    for e in range(EPOCHS_FINE):
        loss = train_epoch(model, loader, opt, criterion, scaler)
        sched.step()
        if (e + 1) % 5 == 0 or e == EPOCHS_FINE - 1:
            print(f"  fold {fold} stage2 epoch {e+1}/{EPOCHS_FINE}  loss={loss:.4f}")
    return model


# ------------------------- cross-validation loop --------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(all_samples), n_classes))
test_probs_per_fold = []
fold_rows = []

for fold, (tr_idx, va_idx) in enumerate(
        skf.split(np.zeros(len(all_labels)), all_labels), start=1):
    print(f"\n=== Fold {fold}/{N_FOLDS}: {len(tr_idx)} train / {len(va_idx)} held-out ===")
    torch.manual_seed(SEED + fold)
    tr_samples = [all_samples[i] for i in tr_idx]
    va_samples = [all_samples[i] for i in va_idx]

    model = train_fold(tr_samples, fold)

    va_loader = DataLoader(PathDataset(va_samples, eval_tf), batch_size=BATCH,
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    va_probs, va_y = get_probs(model, va_loader)
    oof_probs[va_idx] = va_probs            # loader order matches va_idx order
    va_pred = va_probs.argmax(1)
    fold_acc = accuracy_score(va_y, va_pred)
    fold_f1 = f1_score(va_y, va_pred, average="macro", zero_division=0)

    te_probs, te_y = get_probs(model, test_loader)
    test_probs_per_fold.append(te_probs)
    te_acc = accuracy_score(te_y, te_probs.argmax(1))

    print(f"Fold {fold}: held-out acc={fold_acc:.4f} macro-F1={fold_f1:.4f}   "
          f"| single-model TEST acc={te_acc:.4f}")
    fold_rows.append({"fold": fold, "heldout_acc": float(fold_acc),
                      "heldout_macro_f1": float(fold_f1),
                      "single_model_test_acc": float(te_acc)})

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"fold{fold}.pt"))
    del model
    if use_amp:
        torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
    json.dump(class_names, f)

# --------------------------- out-of-fold score ----------------------------
oof_pred = oof_probs.argmax(1)
oof_acc = accuracy_score(all_labels, oof_pred)
oof_f1 = f1_score(all_labels, oof_pred, average="macro", zero_division=0)
f1s = [r["heldout_macro_f1"] for r in fold_rows]

print("\n" + "=" * 60)
print(f"OUT-OF-FOLD RESULT on {len(all_labels)} images "
      f"(each predicted by a model that never saw it)")
print(f"Accuracy: {oof_acc:.4f}   Macro-F1: {oof_f1:.4f}")
print(f"Per-fold macro-F1: mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
print("\nClassification report (OOF):")
print(classification_report(all_labels, oof_pred, target_names=class_names,
                            zero_division=0))
print("Confusion matrix (OOF, rows=true, cols=predicted):")
print(class_names)
oof_cm = confusion_matrix(all_labels, oof_pred)
print(oof_cm)

# --------------------------- ensemble on test -----------------------------
ens_probs = np.mean(test_probs_per_fold, axis=0)
ens_pred = ens_probs.argmax(1)
ens_acc = accuracy_score(te_y, ens_pred)
ens_f1 = f1_score(te_y, ens_pred, average="macro", zero_division=0)
single_accs = [r["single_model_test_acc"] for r in fold_rows]

print("\n" + "=" * 60)
print(f"ENSEMBLE of {N_FOLDS} fold models on the held-out TEST split "
      f"({len(te_y)} images)")
print(f"Single-model test accuracy: mean={np.mean(single_accs):.4f}  "
      f"min={np.min(single_accs):.4f}  max={np.max(single_accs):.4f}")
print(f"Ensemble test accuracy: {ens_acc:.4f}   Macro-F1: {ens_f1:.4f}")
print("\nClassification report (ensemble, test):")
print(classification_report(te_y, ens_pred, target_names=class_names,
                            zero_division=0))
print("Confusion matrix (ensemble, test, rows=true, cols=predicted):")
print(class_names)
ens_cm = confusion_matrix(te_y, ens_pred)
print(ens_cm)

# ------------------------------ save report --------------------------------
report = {
    "config": {"n_folds": N_FOLDS, "epochs_head": EPOCHS_HEAD,
               "epochs_fine": EPOCHS_FINE, "lr_fine": LR_FINE,
               "unfreeze_last_blocks": UNFREEZE_LAST_BLOCKS,
               "sampler_power": SAMPLER_POWER, "tta": TTA, "seed": SEED},
    "classes": class_names,
    "folds": fold_rows,
    "oof": {"n_images": int(len(all_labels)), "accuracy": float(oof_acc),
            "macro_f1": float(oof_f1), "confusion_matrix": oof_cm.tolist()},
    "ensemble_test": {"n_images": int(len(te_y)), "accuracy": float(ens_acc),
                      "macro_f1": float(ens_f1),
                      "confusion_matrix": ens_cm.tolist()},
}
with open(os.path.join(OUT_DIR, "cv_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved 5 fold models, labels.json and cv_report.json to: {OUT_DIR}")

# ----------------- comparison with the earlier (1e-5) run --------------------
# BEGIN COMPARISON
if (os.path.exists(BASELINE_REPORT)
        and os.path.abspath(os.path.dirname(BASELINE_REPORT)) != os.path.abspath(OUT_DIR)):
    with open(BASELINE_REPORT) as f:
        base_rep = json.load(f)
    if base_rep["classes"] == class_names and len(base_rep["folds"]) == len(fold_rows):
        base_lr = base_rep["config"]["lr_fine"]
        print("\n" + "=" * 60)
        print(f"COMPARISON: lr_fine={LR_FINE:g} (this run) vs lr_fine={base_lr:g} (earlier run)")
        print("Per-fold held-out macro-F1 (the folds are identical in both runs):")
        wins = 0
        for old, new in zip(base_rep["folds"], fold_rows):
            d = new["heldout_macro_f1"] - old["heldout_macro_f1"]
            wins += int(d > 0)
            print(f"  fold {new['fold']}: {old['heldout_macro_f1']:.4f} -> "
                  f"{new['heldout_macro_f1']:.4f}  ({d:+.4f})")
        old_oof = base_rep["oof"]["macro_f1"]
        print(f"Out-of-fold macro-F1: {old_oof:.4f} -> {oof_f1:.4f}  ({oof_f1 - old_oof:+.4f})")
        print(f"Folds improved: {wins} of {len(fold_rows)}")
        if oof_f1 > old_oof and wins >= 4:
            print(f"VERDICT: lr_fine={LR_FINE:g} is better on the agreed rule "
                  f"(higher OOF macro-F1 and better in at least 4 of 5 folds). Adopt it.")
        else:
            print(f"VERDICT: not clearly better. Keep lr_fine={base_lr:g}.")
    else:
        print("\nBaseline report found but its classes or fold count differ - "
              "comparison skipped.")
else:
    print("\nNo earlier report found to compare with (or this is the same run).")
# END COMPARISON