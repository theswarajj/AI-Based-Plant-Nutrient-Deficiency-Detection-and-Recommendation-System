"""
Preprocessing for an OLID I crop subset - NUTRIENT CLASSES ONLY.

Preconfigured for EGGPLANT (the folder in your screenshot). Change CROP below
to "snake_gourd" to reuse the same script for snake gourd.

Folder layout expected (matches your screenshot):

    egg_plant/
      preprocess.py
      dataset/
        eggplant__healthy/   <- used   (Healthy)
        eggplant__K/         <- used   (Potassium)
        eggplant__N/         <- used   (Nitrogen)
        eggplant__N_K/       <- used   (Nitrogen_Potassium)
        eggplant__EB/        <- SKIPPED (Epilachna beetle, insect)
        eggplant__FB/        <- SKIPPED (flea beetle, insect)
        eggplant__JAS/       <- SKIPPED (jassid, insect)
        eggplant__MIT/       <- SKIPPED (mite, insect)
        eggplant__MIT_EB/    <- SKIPPED (mite + Epilachna beetle, insect)

Only the nutrient-related folders are ever opened. Insect (and, for snake
gourd, disease) folders are skipped entirely, per the nutrient-only focus.
The folder may contain other crops too; anything not starting with the crop
prefix is ignored.

Install once:
    pip install pillow numpy

Run:
    python preprocess.py

What it does (same safety pipeline as your other crops):
  1. Finds every image in the nutrient folders; skips corrupt files
  2. Removes exact duplicates; drops any image appearing under 2 classes
  3. Finds NEAR-duplicates (dHash) and keeps each group in ONE split
  4. Splits 70/15/15 (train/val/test) per class
  5. Fixes EXIF rotation, converts to RGB, resizes to a square (384px)
  6. Writes  processed/{train,val,test}/<Class>/*.jpg  and
             processed/split_report.json

Your original files are never modified.
"""
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageOps

# ------------------------------ config --------------------------------
CROP = "eggplant"       # "eggplant" or "snake_gourd"

CROPS = {
    "eggplant": {
        "prefix": "eggplant_",
        "classes": {                      # folder suffix -> class name
            "healthy": "Healthy",
            "K": "Potassium",
            "N": "Nitrogen",
            "N_K": "Nitrogen_Potassium",
        },
    },
    "snake_gourd": {
        "prefix": "snake_gourd_",
        "classes": {                      # LS (leaf spot, disease) is skipped
            "healthy": "Healthy",
            "K": "Potassium",
            "N": "Nitrogen",
            "N_K": "Nitrogen_Potassium",
        },
    },
}

BASE = Path(__file__).resolve().parent
# accept either "dataset" or "DATASET" as the folder name
SRC = next((BASE / n for n in ("dataset", "DATASET") if (BASE / n).exists()),
           BASE / "dataset")
OUT = BASE / "processed"

CROP_PREFIX = CROPS[CROP]["prefix"]
CLASS_NAME_MAP = CROPS[CROP]["classes"]

IMG_SIZE = 384
PAD_TO_SQUARE = True
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
NEAR_DUP_THRESHOLD = 4     # dHash Hamming distance (0-64); lower = stricter
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)


# ----------------------------- helpers --------------------------------
def resolve_class_name(folder_name):
    """'eggplant__N_K' -> 'Nitrogen_Potassium', or None if the folder is not
    one of the nutrient classes (insect/disease) and must be skipped.
    Leading underscores are stripped so single- and double-underscore
    folder names both work."""
    suffix = folder_name[len(CROP_PREFIX):].lstrip("_")
    return CLASS_NAME_MAP.get(suffix)


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

    class_dirs, skipped_dirs = [], []
    for d in crop_dirs:
        cls = resolve_class_name(d.name)
        if cls is None:
            skipped_dirs.append(d.name)
        else:
            class_dirs.append((d, cls))

    print(f"Crop: {CROP}   source folder: {SRC}")
    print(f"Found {len(all_dirs)} folders, {len(crop_dirs)} match '{CROP_PREFIX}*'.")
    print(f"Using {len(class_dirs)} nutrient folders:",
          [d.name for d, _ in class_dirs])
    print(f"Skipping {len(skipped_dirs)} non-nutrient folders "
          f"(never scanned):", skipped_dirs)

    if len(class_dirs) != len(CLASS_NAME_MAP):
        print(f"WARNING: expected {len(CLASS_NAME_MAP)} nutrient classes, "
              f"found {len(class_dirs)}. Check CROP, the prefix and folder names.")

    # ---- 1. scan + verify -------------------------------------------
    records, corrupt = [], []
    for d, cls in class_dirs:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                img = load_rgb(p)
                records.append({"path": p, "cls": cls,
                                "md5": file_md5(p), "dhash": dhash(img)})
            except Exception as e:                   # noqa: BLE001
                corrupt.append((str(p), str(e)))
    print(f"Readable images: {len(records)}   corrupt/skipped: {len(corrupt)}")

    # ---- 2. exact duplicates + cross-class conflicts -----------------
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)

    kept, exact_dups, conflicts = [], 0, 0
    for items in by_md5.values():
        if len({i["cls"] for i in items}) > 1:
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

    groups_by_class, near_dup_links = {}, 0
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
            img.save(OUT / split / r["cls"] / f"{r['cls']}_{n:04d}.jpg",
                     "JPEG", quality=95)

    # ---- 6. report ---------------------------------------------------
    summary = {s: {c: counter[(s, c)] for c in groups_by_class} for s in SPLIT}
    report = {
        "crop": CROP,
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