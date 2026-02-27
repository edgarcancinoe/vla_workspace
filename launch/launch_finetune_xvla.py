import os
import sys
import subprocess
import datetime
import itertools
import json
from pathlib import Path

# ============================================================================
# X-VLA Finetuning Launch Script (Grid Search)
# ============================================================================
# This script launches sequential finetuning runs for the SmolVLA model 
# iterating over combinations of action modes and normalization maps.
# ============================================================================

# Check if we're already in the vla environment
if os.environ.get("CONDA_DEFAULT_ENV") != "vla":
    print("WARNING: 'vla' conda environment is not activated.")

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
# Use local /tmp for all cache to avoid NFS issues
USER = os.environ.get("USER", "default_user")
CACHE_ROOT = f"/tmp/vla_cache_{USER}"
os.makedirs(CACHE_ROOT, exist_ok=True)

os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HF_LEROBOT_HOME"] = f"{CACHE_ROOT}/lerobot"

# Create directories
os.makedirs(os.environ["HF_LEROBOT_HOME"], exist_ok=True)

# ============================================================================
# CONFIGURATION - Adjust these variables to your setup
# ============================================================================
HF_USER = os.environ.get("HF_USER", "edgarcancinoe")
# Dataset to use -----------------------------------
DATASET_NAME_STR = os.environ.get("DATASET_NAME_STR", "soarm101_pickplace_10d")
# --------------------------------------------------

# Base model ---------------------------------------
BASE_USER = "lerobot"
BASE_NAME = "xvla-base"
OPTIMIZER_LR = "1e-4"

# Policy Configuration Grids
ACTION_MODES = [
    "so101_ee6d", 
    "so101_joint"
]

NORMALIZATION_MAPPINGS = [
    '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
    '{"ACTION": "MEAN_STD", "STATE": "IDENTITY", "VISUAL": "IDENTITY"}',
]

EMPTY_CAMERAS = os.environ.get("EMPTY_CAMERAS", "1")
POLICY_NUM_IMAGE_VIEWS = os.environ.get("POLICY_NUM_IMAGE_VIEWS", "2")

# Model paths
BASE_POLICY_PATH = f"{BASE_USER}/{BASE_NAME}"
# --------------------------------------------------

# Image Augmentation
ENABLE_AUGMENTATION = "false"
AUGMENTATION_DEGREES = "[-2.5, 2.5]"
AUGMENTATION_TRANSLATE = "[0.025, 0.025]"

# Workspace path detection -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
# --------------------------------------------------

# Dataset Configuration ----------------------------
DATASET_NAME = os.environ.get("DATASET_NAME", DATASET_NAME_STR)
DATASET_REPO_ID = f"{HF_USER}/{DATASET_NAME}"
# --------------------------------------------------

# Training Hyperparameters
BATCH_SIZE = "8"
STEPS = "60000"
LOG_FREQ = "1000"
EVAL_FREQ = "-1"

DEVICE = 'cuda'
CUDA_DEVICE = '1'
NUM_WORKERS = '4'

SAVE_FREQ = "20000"
PUSH_HF_EVERY = "20000"

# Resume configuration
RESUME = "false"
POLICY_PUSH_TO_HUB = "true"
WANDB_ENABLE = "true"

# Policy Structure / Freezing
POLICY_DTYPE = "bfloat16"
FREEZE_VISION_ENCODER = "false"
FREEZE_LANGUAGE_ENCODER = "false"
TRAIN_POLICY_TRANSFORMER = "true"
TRAIN_SOFT_PROMPTS = "true"

# Data Overrides
RENAME_MAP = '{"observation.images.main": "observation.images.image", "observation.images.secondary": "observation.images.image2"}'

# ============================================================================
# HELPER FOR NAMING
# ============================================================================
def get_norm_suffix(mapping_str):
    """Generate a clean identifier for the parameter name based on json keys."""
    try:
        m = json.loads(mapping_str)
        am = str(m.get("ACTION", "ID")).replace("_", "").lower()[:1]
        sm = str(m.get("STATE", "ID")).replace("_", "").lower()[:1]
        return f"a-{am}_s-{sm}"
    except Exception:
        return "custom"

# ============================================================================
# DEVICE SETUP
# ============================================================================
# Set CUDA visibility if a specific device is requested
if CUDA_DEVICE:
    print(f"Restricting CUDA visibility to device: {CUDA_DEVICE}")
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

print("============================================================================")
print("SmolVLA Finetuning Launch Script (Grid Search)")
print("============================================================================")
print()

# Check if lerobot Python package is installed
try:
    import lerobot
    print("✓ lerobot package found\n")
except ImportError:
    print("ERROR: lerobot Python package not found!\n")
    print("Please install lerobot first:")
    print("  pip install lerobot\n")
    print("Or if you have a local clone, ensure you're in the correct conda environment.")
    sys.exit(1)

# Check if HF_USER is set
if HF_USER == "YOUR_HF_USERNAME":
    print("WARNING: HF_USER is not set!")
    print("Please set your Hugging Face username:")
    print("  export HF_USER=your_username")
    print("  or edit this script directly\n")
    response = input("Continue anyway? (y/n) ")
    if response.strip().lower() != 'y':
        sys.exit(1)

# ============================================================================
# RUN GRID SEARCH LOOP
# ============================================================================
print("============================================================================")
print(f"Starting {len(ACTION_MODES) * len(NORMALIZATION_MAPPINGS)} grid search iterations...")
print("============================================================================")
print()

for action_mode, norm_mapping in itertools.product(ACTION_MODES, NORMALIZATION_MAPPINGS):
    norm_suffix = get_norm_suffix(norm_mapping)
    
    # --------------------------------------------------
    # Policy name to use when saving
    # --------------------------------------------------
    # Append the action mode and norm mapping to unique-ify the runs
    POLICY_NAME = f"{BASE_NAME}_ft_{DATASET_NAME_STR}_{action_mode}_{norm_suffix}"
    if ENABLE_AUGMENTATION == "true":
        POLICY_NAME += "_aug"
        
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    JOB_NAME = os.environ.get("JOB_NAME", f"{POLICY_NAME}_{TIMESTAMP}")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(WORKSPACE_DIR / "outputs" / "train" / f"{POLICY_NAME}_{TIMESTAMP}"))
    POLICY_REPO_ID = f"{HF_USER}/{POLICY_NAME}"

    print("\n" + "="*76)
    print("LAUNCHING TRAINING RUN")
    print("="*76)
    print(f"  Dataset:        {DATASET_REPO_ID}")
    print(f"  Action Mode:    {action_mode}")
    print(f"  Norm Mapping:   {norm_mapping}")
    print(f"  Output Dir:     {OUTPUT_DIR}")
    print(f"  Policy Repo ID: {POLICY_REPO_ID}")
    print("="*76 + "\n")

    augmentation_tfs = (
        f'{{"affine": {{"type": "RandomAffine", "kwargs": {{"degrees": {AUGMENTATION_DEGREES},'
        f' "translate": {AUGMENTATION_TRANSLATE}}}}}}}'
    )

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.path={BASE_POLICY_PATH}",
        f"--policy.repo_id={POLICY_REPO_ID}",
        f"--policy.push_to_hub={POLICY_PUSH_TO_HUB}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--rename_map={RENAME_MAP}",
        f"--dataset.image_transforms.enable={ENABLE_AUGMENTATION}",
        f"--dataset.image_transforms.tfs={augmentation_tfs}",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        f"--log_freq={LOG_FREQ}",
        f"--eval_freq={EVAL_FREQ}",
        f"--save_freq={SAVE_FREQ}",
        f"--push_every={PUSH_HF_EVERY}",
        f"--output_dir={OUTPUT_DIR}",
        f"--job_name={JOB_NAME}",
        f"--policy.device={DEVICE}",
        f"--wandb.enable={WANDB_ENABLE}",
        f"--num_workers={NUM_WORKERS}",
        f"--resume={RESUME}",
        f"--policy.dtype={POLICY_DTYPE}",
        f"--policy.action_mode={action_mode}",
        f"--policy.empty_cameras={EMPTY_CAMERAS}",
        f"--policy.freeze_vision_encoder={FREEZE_VISION_ENCODER}",
        f"--policy.freeze_language_encoder={FREEZE_LANGUAGE_ENCODER}",
        f"--policy.train_policy_transformer={TRAIN_POLICY_TRANSFORMER}",
        f"--policy.train_soft_prompts={TRAIN_SOFT_PROMPTS}",
        f"--policy.num_image_views={POLICY_NUM_IMAGE_VIEWS}",
        f"--policy.normalization_mapping={norm_mapping}",
    ]

    try:
        exit_code = subprocess.call(cmd, env=os.environ.copy())
    except KeyboardInterrupt:
        print("\n\nUser interrupted with Ctrl+C. Exiting grid search.")
        sys.exit(130)
    except Exception as e:
        print(f"Error launching training: {e}")
        exit_code = 1

    if exit_code != 0:
        print(f"\nTraining failed with exit code: {exit_code}")
        print("Stopping further grid search iterations.")
        sys.exit(exit_code)
    
    print(f"\nCompleted run for {action_mode} | {norm_suffix}")

print("\n============================================================================")
print("All requested training runs completed successfully!")
print("============================================================================")
sys.exit(0)
