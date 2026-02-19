import sys
from pathlib import Path
import torch

# Add lerobot to path
lerobot_path = Path.cwd() / "lerobot_src" / "src"
sys.path.append(str(lerobot_path))

from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig

def test_factory():
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test", root="test_data"),
        policy=PreTrainedConfig(type="act")
    )
    # Enable transforms
    cfg.dataset.image_transforms.enable = True
    
    # Mock some image keys if needed, but make_dataset just creates the object
    print("Calling make_dataset...")
    try:
        # We don't need a real dataset existing, just want to see the image_transforms field
        # of the returned dataset (but make_dataset might fail later if metadata doesn't exist)
        # So let's mock LeRobotDatasetMetadata or just catch the error after transforms are created
        
        # Actually, let's just look at the code in factory.py directly via a simpler test
        import logging
        logging.basicConfig(level=logging.INFO)
        
        from lerobot.datasets.factory import make_dataset
        # make_dataset will try to create metadata, so let's just test the logic inside factory.py
        # by manually calling the part we changed if needed, or providing a valid skeleton.
        
        print("Integration Check:")
        # We'll just trace the return of make_dataset if possible, or use a smaller mock
    except Exception as e:
        print(f"Caught expected error: {e}")

    # Manual check of what ImageTransforms would be
    from lerobot.datasets.factory import make_dataset
    # I'll just check if CustomAugmentationPipeline is in the global scope of factory.py
    # or if I can import it from there.
    
    print("\nAttempting to import CustomAugmentationPipeline using factory.py's method:")
    try:
        workspace_root = Path.cwd() 
        if str(workspace_root) not in sys.path:
            sys.path.append(str(workspace_root))
        from utils.augmentations import CustomAugmentationPipeline
        print("SUCCESS: CustomAugmentationPipeline is importable from workspace root.")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_factory()
