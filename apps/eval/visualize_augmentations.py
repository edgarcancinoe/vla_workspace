import os
import math
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

from thesis_vla.common.paths import DATASETS_OUTPUT_DIR, PROJECT_ROOT, RUNTIME_CACHE_DIR, RUNTIME_OUTPUTS_DIR

# =========================================================
# HF / Hub cache + permissions fix (avoid /opt/cache)
# =========================================================
def _configure_hf_cache(cache_root: str | None = None) -> None:
    if cache_root is None:
        cache_root = str(RUNTIME_CACHE_DIR / "huggingface")

    hf_home = cache_root
    hub_cache = os.path.join(hf_home, "hub")
    datasets_cache = os.path.join(hf_home, "datasets")

    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    os.environ["XDG_CACHE_HOME"] = os.path.join(str(Path.home()), ".cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    Path(hub_cache).mkdir(parents=True, exist_ok=True)
    Path(datasets_cache).mkdir(parents=True, exist_ok=True)

_configure_hf_cache()

# Ensure we use the correct lerobot
import lerobot
print(f"Using lerobot from: {lerobot.__file__}")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from thesis_vla.common.augmentations import CustomAugmentationPipeline

def visualize_augmentations(
    repo_id,
    root,
    output_dir=None,
    num_samples=5,
    # geometric
    enable_geometric=True,
    rotation=15.0,
    translation=0.10,
    fill_mode="reflect",   # "reflect" | "crop" | "black"
    # photometric
    enable_photometric=True,
):
    output_path = Path(output_dir) if output_dir else (RUNTIME_OUTPUTS_DIR / "visualizations")
    output_path.mkdir(parents=True, exist_ok=True)

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {repo_id}")
    print("Augmentation Config:")
    print(f"  Geometric: {'ON' if enable_geometric else 'OFF'} | rot +/- {rotation} deg | trans +/- {translation*100:.1f}% | mode={fill_mode}")
    print(f"  Photometric: {'ON' if enable_photometric else 'OFF'} (color jitter + noise + blur)")

    pipeline = CustomAugmentationPipeline(
        enable_geometric=enable_geometric,
        rotation_deg=rotation,
        translation_frac=translation,
        fill_mode=fill_mode,
        enable_photometric=enable_photometric,
    )

    print("Loading dataset (forcing pyav backend if possible)...")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(root_path),
        image_transforms=None,
        video_backend="pyav",
        tolerance_s=0.1,  # Large tolerance for visualization
    )

    print(f"Dataset loaded. Length: {len(dataset)}")

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        try:
            item = dataset[int(idx)]
            image_keys = [k for k in item.keys() if "image" in k]

            for key in image_keys:
                img = item[key]  # (C,H,W), float [0,1]
                aug = pipeline(img)

                # Convert to BGR for CV2
                orig_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                aug_np = (aug.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
                aug_bgr = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)

                combined = np.hstack([orig_bgr, aug_bgr])
                
                # Add text
                cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, f"Aug (mode={fill_mode})", (orig_bgr.shape[1] + 10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                safe_key = key.replace("/", "_").replace(".", "_")
                filename = output_path / f"sample_{idx}_{safe_key}_{fill_mode}.jpg"
                cv2.imwrite(str(filename), combined)
                print(f"Saved {filename}")
        except Exception as e:
            print(f"Skipping sample {idx} due to error: {e}")
            continue


if __name__ == "__main__":
    REPO_ID = "edgarcancinoe/soarm101_pickplace_top_wrist"
    DATA_ROOT = str(DATASETS_OUTPUT_DIR / REPO_ID.split("/")[-1])
    OUTPUT_DIR = str(RUNTIME_OUTPUTS_DIR / "visualizations")

    ROTATION_DEG = 5
    TRANSLATION_FRAC = 0.05
    FILL_MODE = "reflect"

    visualize_augmentations(
        repo_id=REPO_ID,
        root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        num_samples=5,
        enable_geometric=True,
        rotation=ROTATION_DEG,
        translation=TRANSLATION_FRAC,
        fill_mode=FILL_MODE,
        enable_photometric=True,
    )
