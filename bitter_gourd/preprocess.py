"""
Preprocessing for the bitter gourd subset of the OLID I dataset -
NUTRIENT DEFICIENCY CLASSES ONLY.

Your DATASET folder contains ALL crops as sibling folders. This script:
  1. Filters for ONLY folders starting with "bitter_gourd_"
  2. From those, keeps ONLY the 6 nutrient-deficiency classes
     (healthy, K, K_Mg, N, N_K, N_Mg) and SKIPS the disease/pest classes
     entirely (DM = Downy Mildew, JAS = Jassid damage, LS = Leaf Spot) -
     they are never scanned or copied, per your request to focus purely
     on nutrient deficiency.

Folder layout expected:

    bitter_gourd/
      preprocess.py
      DATASET/
        bitter_gourd_healthy/   <- used
        bitter_gourd_K/         <- used
        bitter_gourd_K_Mg/      <- used
        bitter_gourd_N/         <- used
        bitter_gourd_N_K/       <- used
        bitter_gourd_N_Mg/      <- used
        bitter_gourd_DM/        <- SKIPPED (disease)
        bitter_gourd_JAS/       <- SKIPPED (pest)
        bitter_gourd_LS/        <- SKIPPED (disease)
        ash_gourd_.../          <- ignored (different crop)
        bottle_gourd_.../       <- ignored (different crop)
        ... (everything else)   <- ignored

Install once:
    pip install pillow numpy

Run:
    python preprocess.py

What it does (same safety pipeline used for your other crops):
  1. Finds every image under the 6 nutrient-class folders; skips corrupt files
  2. Removes exact duplicates; drops any image appearing under 2 classes
  3. Finds NEAR-duplicates (dHash) and keeps each group in ONE split
  4. Splits 70/15/15 (train/val/test) per class
  5. Fixes EXIF rotation, converts to RGB, resizes to a square (384px)
  6. Writes  processed/{train,val,test}/<Class>/*.jpg  and
             processed/split_report.json
"""
import os
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageOps

# ------------------------------ config --------------------------------
BASE = Path(__file__).resolve().parent
SRC = BASE / "DATASET"
OUT = BASE / "processed"

CROP_PREFIX = "bitter_gourd_"

# ONLY these suffixes are used - DM, JAS, LS are deliberately excluded
# (disease/pest, not nutrient deficiency) and their folders are never
# scanned at all.
CLASS_NAME_MAP = {
    "healthy": "Healthy",
    "K": "Potassium",
    "K_Mg": "Potassium_Magnesium",
    "N": "Nitrogen",
    "N_K": "Nitrogen_Potassium",
    "N_Mg": "Nitrogen_Magnesium",
}

IMG_SIZE = 384
PAD_TO_SQUARE = True
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
NEAR_DUP_THRESHOLD = 4
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)


# ----------------------------- helpers --------------------------------
def resolve_class_name(folder_name):
    """'bitter_gourd_K_Mg' -> 'Potassium_Magnesium', or None if this folder
    is not one of the 6 nutrient classes (e.g. DM/JAS/LS) and should be
    skipped entirely. Handles a possible double-underscore variant too
    (some crops in this dataset use bitter_gourd__K instead of bitter_gourd_K)."""
    suffix = folder_name[len(CROP_PREFIX):].lstrip("_")
    return CLASS_NAME_MAP.get(suffix)   # None if not in the allowed set


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def load_rgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_square(img, size, pad):
    if not pad:
        return img.resize((size, size), Image.LANCZOS)
    img = ImageOps.contain(img, (size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


# ------------------------------ main ----------------------------------
def main():
    if not SRC.exists():
        raise SystemExit(f"Dataset folder not found: {SRC}")

    all_dirs = sorted(d for d in SRC.iterdir() if d.is_dir())
    crop_dirs = [d for d in all_dirs if d.name.startswith(CROP_PREFIX)]

    # Separate into "used" (nutrient) vs "skipped" (disease/pest/unknown)
    class_dirs = []
    skipped_dirs = []
    for d in crop_dirs:
        cls = resolve_class_name(d.name)
        if cls is None:
            skipped_dirs.append(d.name)
        else:
            class_dirs.append((d, cls))

    print(f"Found {len(all_dirs)} folders total in DATASET, "
          f"{len(crop_dirs)} match '{CROP_PREFIX}*'.")
    print(f"Using {len(class_dirs)} nutrient-deficiency folders:",
          [d.name for d, _ in class_dirs])
    print(f"Skipping {len(skipped_dirs)} non-nutrient folders "
          f"(disease/pest, never scanned):", skipped_dirs)

    if len(class_dirs) != 6:
        print(f"WARNING: expected 6 nutrient classes, found {len(class_dirs)}. "
              f"Check CLASS_NAME_MAP and folder names.")

    # ---- 1. scan + verify -------------------------------------------
    records = []
    corrupt = []
    for d, cls in class_dirs:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                img = load_rgb(p)
                records.append({
                    "path": p, "cls": cls,
                    "md5": file_md5(p), "dhash": dhash(img),
                })
            except Exception as e:                   # noqa: BLE001
                corrupt.append((str(p), str(e)))
    print(f"Readable images: {len(records)}   corrupt/skipped: {len(corrupt)}")

    # ---- 2. exact duplicates + cross-class conflicts -----------------
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)

    kept, exact_dups, conflicts = [], 0, 0
    for md5, items in by_md5.items():
        classes = {i["cls"] for i in items}
        if len(classes) > 1:
            conflicts += len(items)
            continue
        kept.append(items[0])
        exact_dups += len(items) - 1
    print(f"Exact duplicates removed: {exact_dups}   "
          f"label-conflict images dropped: {conflicts}")

    # ---- 3. near-duplicate groups (per class) ------------------------
    per_class = defaultdict(list)
    for r in kept:
        per_class[r["cls"]].append(r)

    groups_by_class = {}
    near_dup_links = 0
    for cls, items in per_class.items():
        uf = UnionFind(len(items))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if hamming(items[i]["dhash"], items[j]["dhash"]) <= NEAR_DUP_THRESHOLD:
                    uf.union(i, j)
                    near_dup_links += 1
        groups = defaultdict(list)
        for idx, r in enumerate(items):
            groups[uf.find(idx)].append(r)
        groups_by_class[cls] = list(groups.values())
    print(f"Near-duplicate pairs found: {near_dup_links} "
          f"(kept together in the same split)")

    # ---- 4. group-aware stratified split -----------------------------
    assignment = {"train": [], "val": [], "test": []}
    for cls, groups in groups_by_class.items():
        total = sum(len(g) for g in groups)
        target = {s: SPLIT[s] * total for s in SPLIT}
        current = {s: 0 for s in SPLIT}
        random.shuffle(groups)
        groups.sort(key=len, reverse=True)
        for g in groups:
            s = max(SPLIT, key=lambda k: (target[k] - current[k]) / target[k])
            assignment[s].extend(g)
            current[s] += len(g)

    # ---- 5. resize + write ------------------------------------------
    for split in SPLIT:
        for cls in groups_by_class:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    counter = defaultdict(int)
    for split, items in assignment.items():
        for r in items:
            counter[(split, r["cls"])] += 1
            n = counter[(split, r["cls"])]
            img = resize_square(load_rgb(r["path"]), IMG_SIZE, PAD_TO_SQUARE)
            out_path = OUT / split / r["cls"] / f"{r['cls']}_{n:04d}.jpg"
            img.save(out_path, "JPEG", quality=95)

    # ---- 6. report ---------------------------------------------------
    summary = {s: {c: counter[(s, c)] for c in groups_by_class} for s in SPLIT}
    report = {
        "image_size": IMG_SIZE,
        "readable": len(records),
        "corrupt": corrupt,
        "exact_duplicates_removed": exact_dups,
        "label_conflicts_dropped": conflicts,
        "near_duplicate_pairs": near_dup_links,
        "excluded_non_nutrient_folders": skipped_dirs,
        "split_counts": summary,
    }
    OUT.mkdir(exist_ok=True)
    with open(OUT / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nFinal split (images per class):")
    for s in SPLIT:
        print(f"  {s:5s}", summary[s], " total =", sum(summary[s].values()))
    print(f"\nDone. Output folder: {OUT}")


if __name__ == "__main__":
    main()