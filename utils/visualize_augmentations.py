import torch
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# Add lerobot source to path
WORKSPACE_ROOT = Path(__file__).parent.parent
LEROBOT_PATH = WORKSPACE_ROOT.parent / "repos" / "lerobot" / "src"
sys.path.append(str(LEROBOT_PATH))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

def visualize_augmentations(
    repo_id,
    root,
    output_dir="visualizations",
    num_samples=5,
    enable_transforms=True,
    rotation=10,
    translation=0.1
):
    """
    Loads a LeRobot dataset, applies transforms, and saves side-by-side comparisons.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {repo_id}")
    
    # 1. Configure Transforms
    tf_config = ImageTransformsConfig(
        enable=enable_transforms,
        max_num_transforms=3,
        random_order=False,
    )
    
    tf_config.tfs["affine"] = ImageTransformConfig(
        weight=1.0,
        type="RandomAffine",
        kwargs={
            "degrees": (-rotation, rotation), 
            "translate": (translation, translation),
            "fill": 0
        }
    )
    
    print(f"Augmentation Config:")
    print(f"  Rotation: +/- {rotation} deg")
    print(f"  Translate: +/- {translation*100}%")
    
    transforms = ImageTransforms(tf_config)
    
    # 2. Load Dataset
    # Try disabling video backend to avoid torchcodec issues
    print("Loading dataset (forcing pyav backend if possible)...")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        image_transforms=None,
        video_backend="pyav" # Try to force pyav? LeRobot 0.4.3 might default to torchcodec
    )
    
    print(f"Dataset loaded. Length: {len(dataset)}")
    
    # 3. Sample and Visualize
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        image_keys = [k for k in item.keys() if "image" in k]
        
        for key in image_keys:
            img_tensor = item[key] # (C, H, W)
            augmented_tensor = transforms(img_tensor)
            
            orig_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            aug_np = (augmented_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
            aug_np = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)
            
            combined = np.hstack([orig_np, aug_np])
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Augmented", (orig_np.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"{key}", (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            filename = output_path / f"sample_{i}_{key.split('.')[-1]}.jpg"
            cv2.imwrite(str(filename), combined)
            print(f"Saved {filename}")

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    REPO_ID = "edgarcancinoe/soarm101_pickplace_top_wrist"
    DATA_ROOT = "./data"  # Downloads dataset to ./data folder to keep workspace clean
    OUTPUT_DIR = "visualizations"
    
    # Augmentation Parameters
    ROTATION_DEG = 15.0      # +/- degrees
    TRANSLATION_FRAC = 0.1   # +/- fraction (0.1 = 10%)
    # =========================================================================

    visualize_augmentations(
        repo_id=REPO_ID,
        root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        rotation=ROTATION_DEG,
        translation=TRANSLATION_FRAC
    )
