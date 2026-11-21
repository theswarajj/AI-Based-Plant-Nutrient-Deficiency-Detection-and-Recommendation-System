"""
Unified Crop Species & Leaf Image Selection Tool.

Maps crop species to their corresponding input folders and models:
  - Rice         -> INPUT/rice         -> run_rice.py
  - Banana       -> INPUT/banana       -> run_banana.py
  - Ash Gourd    -> INPUT/ashgourd     -> run_ashgourd.py
  - Bitter Gourd -> INPUT/bitter_gourd -> run_bitter_gourd.py

Allows interactive selection of species and image in the terminal,
and computes predictions, Grad-CAM heatmaps, and attribution dashboards
only on the selected image.
"""

import os
import sys
import subprocess
from utils.selection import (
    SPECIES_MAP,
    list_input_images,
    select_species_interactive,
    select_image_interactive,
    IMAGE_EXTENSIONS,
)

BASE = os.path.dirname(os.path.abspath(__file__))


def run_crop_pipeline(species_cfg, image_paths):
    """Run the chosen crop script on the selected image paths."""
    script_path = species_cfg["script"]
    if not os.path.exists(script_path):
        print(f"[Error] Script not found: {script_path}")
        return 1

    cmd = [sys.executable, script_path] + image_paths
    print(f"\n{'='*64}")
    print(f"Launching {species_cfg['display_name']} Model Inference...")
    print(f"Target images ({len(image_paths)}): {', '.join(os.path.basename(p) for p in image_paths)}")
    print(f"{'='*64}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    # CLI help
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("Usage: python select_and_run.py [CROP_NAME] [IMAGE_PATH ...]")
        print("\nAvailable crops:")
        for key, sp in SPECIES_MAP.items():
            print(f"  {key:<12} -> {sp['display_name']} (Folder: INPUT/{sp['folder_name']})")
        print("\nExamples:")
        print("  python select_and_run.py                    (Interactive species & image menu)")
        print("  python select_and_run.py rice               (Interactive image menu for Rice)")
        print("  python select_and_run.py banana path/to/img (Direct execution)")
        return

    # Check if crop is passed via CLI
    selected_species = None
    remaining_args = []
    if len(sys.argv) > 1:
        arg_crop = sys.argv[1].lower().strip()
        for key, sp in SPECIES_MAP.items():
            if arg_crop in (key, sp["folder_name"].lower(), sp["display_name"].lower()):
                selected_species = sp
                remaining_args = sys.argv[2:]
                break

    # If no valid crop from CLI, prompt interactively
    if selected_species is None:
        selected_species = select_species_interactive()

    input_dir = selected_species["input_dir"]
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Determine image paths
    if remaining_args:
        if remaining_args[0] in ("--all", "-a"):
            chosen_images = list_input_images(input_dir, IMAGE_EXTENSIONS)
        else:
            chosen_images = remaining_args
    else:
        all_images = list_input_images(input_dir, IMAGE_EXTENSIONS)
        if not all_images:
            raise SystemExit(f"No images found in {input_dir}")
        chosen_images = select_image_interactive(
            all_images,
            species_name=selected_species["display_name"],
            folder_display=f"INPUT/{selected_species['folder_name']}",
        )

    # Run the computation only on selected image(s)
    sys.exit(run_crop_pipeline(selected_species, chosen_images))


if __name__ == "__main__":
    main()
