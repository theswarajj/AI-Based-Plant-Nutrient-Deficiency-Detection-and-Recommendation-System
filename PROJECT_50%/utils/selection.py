"""
Interactive selection utilities for crop nutrient deficiency models.

Maps crop species to their corresponding input folders and scripts,
and provides terminal-based interactive selection menus for both species
and individual images.
"""

import os
import sys
import glob

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPECIES_MAP = {
    "rice": {
        "key": "rice",
        "display_name": "Rice",
        "folder_name": "rice",
        "input_dir": os.path.join(BASE, "INPUT", "rice"),
        "output_dir": os.path.join(BASE, "OUTPUT", "rice"),
        "script": os.path.join(BASE, "run_rice.py"),
        "model_dir": os.path.join(BASE, "trained_model_rice"),
        "description": "EffNetV2-S model (Nitrogen, Phosphorus, Potassium)",
    },
    "banana": {
        "key": "banana",
        "display_name": "Banana",
        "folder_name": "banana",
        "input_dir": os.path.join(BASE, "INPUT", "banana"),
        "output_dir": os.path.join(BASE, "OUTPUT", "banana"),
        "script": os.path.join(BASE, "run_banana.py"),
        "model_dir": os.path.join(BASE, "trained_model_banana"),
        "description": "EffNetV2-S model (8 classes: B, Ca, Fe, Mg, K, S, Zn, Healthy)",
    },
    "ashgourd": {
        "key": "ashgourd",
        "display_name": "Ash Gourd",
        "folder_name": "ashgourd",
        "input_dir": os.path.join(BASE, "INPUT", "ashgourd"),
        "output_dir": os.path.join(BASE, "OUTPUT", "ashgourd"),
        "script": os.path.join(BASE, "run_ashgourd.py"),
        "model_dir": os.path.join(BASE, "trained_model_ashgourd"),
        "description": "EffNetV2-S model (7 classes: Healthy, N, N+Mg, N+K, K, K+Mg, Mildew)",
    },
    "bitter_gourd": {
        "key": "bitter_gourd",
        "display_name": "Bitter Gourd",
        "folder_name": "bitter_gourd",
        "input_dir": os.path.join(BASE, "INPUT", "bitter_gourd"),
        "output_dir": os.path.join(BASE, "OUTPUT", "bitter_gourd"),
        "script": os.path.join(BASE, "run_bitter_gourd.py"),
        "model_dir": os.path.join(BASE, "trained_model_bitter_gourd"),
        "description": "EffNetV2-S model (6 classes: Healthy, N, N+Mg, N+K, K, K+Mg)",
    },
}


def list_input_images(folder_path, extensions=IMAGE_EXTENSIONS):
    """Return sorted list of image file paths found in folder_path."""
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        f for f in glob.glob(os.path.join(folder_path, "*"))
        if os.path.splitext(f)[1].lower() in extensions
    )


def select_image_interactive(image_paths, species_name="Crop", folder_display=None, allow_all=True):
    """
    Prompt user in terminal to select an image from image_paths.
    Returns list of selected image path(s).
    """
    if not image_paths:
        raise SystemExit(f"No valid images found to select.")

    total = len(image_paths)
    folder_str = f" (Folder: {folder_display})" if folder_display else ""

    print(f"\n{'='*64}")
    print(f"[{species_name}] Leaf Image Selection{folder_str}")
    print(f"{'='*64}")
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        try:
            file_size_kb = os.path.getsize(img_path) / 1024
            size_str = f"{file_size_kb:.1f} KB"
        except OSError:
            size_str = "unknown size"
        print(f"  [{idx}] {filename:<35} ({size_str})")

    if allow_all and total > 1:
        print(f"  [A] Process All Images ({total} total)")
    print(f"{'-'*64}")

    default_choice = "1"
    prompt_range = f"1-{total}" if total > 1 else "1"
    hint = f"[{prompt_range}, or 'A' for all]" if (allow_all and total > 1) else f"[{prompt_range}]"

    while True:
        try:
            choice = input(f"Select image to compute {hint} (default: {default_choice}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[Non-interactive / Cancelled] Defaulting to image [1]: {os.path.basename(image_paths[0])}")
            return [image_paths[0]]

        # Pressing Enter selects default
        if choice == "":
            selected = image_paths[0]
            print(f">> Selected: {os.path.basename(selected)}\n")
            return [selected]

        # 'A' or 'all' selects all images
        if allow_all and choice.lower() in ("a", "all"):
            print(f">> Selected all {total} image(s).\n")
            return image_paths

        # Single number selection
        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= total:
                selected = image_paths[val - 1]
                print(f">> Selected: {os.path.basename(selected)}\n")
                return [selected]
            else:
                print(f"  [Error] Selection '{choice}' out of range. Please enter 1 to {total}.")
                continue

        # Comma- or space-separated numbers (e.g. "1, 2")
        parts = [p.strip() for p in choice.replace(",", " ").split() if p.strip()]
        if parts and all(p.isdigit() for p in parts):
            indices = [int(p) for p in parts]
            if all(1 <= i <= total for i in indices):
                selected = []
                for i in indices:
                    p = image_paths[i - 1]
                    if p not in selected:
                        selected.append(p)
                print(f">> Selected {len(selected)} image(s): {', '.join(os.path.basename(p) for p in selected)}\n")
                return selected
            else:
                print(f"  [Error] One or more numbers out of range (1-{total}).")
                continue

        # Exact filename match
        matched = [p for p in image_paths if os.path.basename(p).lower() == choice.lower()]
        if matched:
            selected = matched[0]
            print(f">> Selected: {os.path.basename(selected)}\n")
            return [selected]

        print(f"  [Error] Invalid selection '{choice}'. Please enter 1-{total}{' or A' if allow_all else ''}.")


def select_species_interactive():
    """
    Prompt user in terminal to select a crop species.
    Returns the species config dict from SPECIES_MAP.
    """
    species_list = list(SPECIES_MAP.values())
    total = len(species_list)

    print(f"\n{'='*64}")
    print(f"Crop Species Selection Tool")
    print(f"{'='*64}")
    for idx, sp in enumerate(species_list, 1):
        num_imgs = len(list_input_images(sp["input_dir"]))
        print(f"  [{idx}] {sp['display_name']:<14} Folder: INPUT/{sp['folder_name']:<14} ({num_imgs} images)")
    print(f"{'-'*64}")

    while True:
        try:
            choice = input(f"Select crop species [1-{total}] (default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[Defaulting to species 1]: {species_list[0]['display_name']}")
            return species_list[0]

        if choice == "":
            chosen = species_list[0]
            print(f">> Selected: {chosen['display_name']}\n")
            return chosen

        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= total:
                chosen = species_list[val - 1]
                print(f">> Selected: {chosen['display_name']}\n")
                return chosen
            else:
                print(f"  [Error] Selection '{choice}' out of range. Please enter 1 to {total}.")
                continue

        for sp in species_list:
            if choice.lower() in (sp["key"].lower(), sp["folder_name"].lower(), sp["display_name"].lower()):
                print(f">> Selected: {sp['display_name']}\n")
                return sp

        print(f"  [Error] Invalid choice '{choice}'. Please enter 1-{total} or crop name.")
