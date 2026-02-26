import os
import sys
import subprocess
import datetime
from pathlib import Path

# ============================================================================
# SmolVLA Finetuning Launch Script (Python Version)
# ============================================================================
# This script launches the finetuning process for the SmolVLA model using 
# lerobot code.
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

# Policy Configuration
ACTION_MODE = os.environ.get("ACTION_MODE", "so101_ee6d")    # Options: "auto", "ee6d" (or specific to your robot)
EMPTY_CAMERAS = os.environ.get("EMPTY_CAMERAS", "1")         # Number of empty camera slots (if needed)
POLICY_NUM_IMAGE_VIEWS = os.environ.get("POLICY_NUM_IMAGE_VIEWS", "2")

# Normalization mapping per feature type.
# Supported modes: IDENTITY (no-op), MIN_MAX (→ [-1,1]), MEAN_STD (zero-mean unit-var),
#                  QUANTILES (q01/q99 → [-1,1]), QUANTILE10 (q10/q90 → [-1,1])
NORMALIZATION_MAPPING = os.environ.get("NORMALIZATION_MAPPING", '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}')

# Model paths
BASE_POLICY_PATH = f"{BASE_USER}/{BASE_NAME}"
# --------------------------------------------------

# Image Augmentation
ENABLE_AUGMENTATION = os.environ.get("ENABLE_AUGMENTATION", "false")
AUGMENTATION_DEGREES = os.environ.get("AUGMENTATION_DEGREES", "[-2.5, 2.5]")
AUGMENTATION_TRANSLATE = os.environ.get("AUGMENTATION_TRANSLATE", "[0.025, 0.025]")

# Policy name to use when saving -------------------
POLICY_NAME = f"{BASE_NAME}_finetuned_{DATASET_NAME_STR}_{ACTION_MODE}"
# Append _aug suffix if augmentation is enabled
if ENABLE_AUGMENTATION == "true":
    POLICY_NAME += "_aug"
# --------------------------------------------------

# Workspace path detection -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
# --------------------------------------------------

# Dataset Configuration ----------------------------
DATASET_NAME = os.environ.get("DATASET_NAME", DATASET_NAME_STR)
DATASET_REPO_ID = f"{HF_USER}/{DATASET_NAME}"
# --------------------------------------------------

# Training Hyperparameters
BATCH_SIZE = os.environ.get("BATCH_SIZE", "8")
STEPS = os.environ.get("STEPS", "60000")
LOG_FREQ = os.environ.get("LOG_FREQ", "1000")
EVAL_FREQ = os.environ.get("EVAL_FREQ", "-1")

DEVICE = os.environ.get("DEVICE", "cuda")
CUDA_DEVICE = os.environ.get("CUDA_DEVICE", "2")
NUM_WORKERS = os.environ.get("NUM_WORKERS", "4")

SAVE_FREQ = os.environ.get("SAVE_FREQ", "20000")
PUSH_HF_EVERY = os.environ.get("PUSH_HF_EVERY", "20000")

# ============================================================================
# OUTPUT & STORAGE CONFIGURATION
# ============================================================================
# 1. IDENTIFIERS & LOGGING
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Job name for logging and Weights & Biases
JOB_NAME = os.environ.get("JOB_NAME", f"{POLICY_NAME}_{TIMESTAMP}")

# 2. LOCAL OUTPUT (Checkpoints & Logs)
# Directory where training logs and checkpoints will be saved LOCALLY
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(WORKSPACE_DIR / "outputs" / "train" / f"{POLICY_NAME}_{TIMESTAMP}"))

# Resume configuration
RESUME = os.environ.get("RESUME", "false")

# 3. HUGGING FACE HUB OUTPUT
# Set to 'true' to push model to Hugging Face Hub, 'false' to keep local only
POLICY_PUSH_TO_HUB = os.environ.get("POLICY_PUSH_TO_HUB", "true")

# HuggingFace Hub repository ID to push the trained model to
POLICY_REPO_ID = os.environ.get("POLICY_REPO_ID", f"{HF_USER}/{POLICY_NAME}")

# Weights & Biases Configuration
WANDB_ENABLE = "false"


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
print("SmolVLA Finetuning Launch Script")
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

# Display configuration
print("Configuration:")
print(f"  Dataset:        {DATASET_REPO_ID}")
print(f"  Batch Size:     {BATCH_SIZE}")
print(f"  Steps:          {STEPS}")
print(f"  Save Freq:      {SAVE_FREQ}")
print(f"  Push HF Every:  {PUSH_HF_EVERY}")
print(f"  Output Dir:     {OUTPUT_DIR}")
print(f"  Job Name:       {JOB_NAME}")
print(f"  Device:         {DEVICE}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"  W&B Enabled:    {WANDB_ENABLE}")
print(f"  Base Policy:    {BASE_POLICY_PATH}")
print(f"  Policy Repo ID: {POLICY_REPO_ID}")
print(f"  Push to Hub:    {POLICY_PUSH_TO_HUB}")
print(f"  Norm Mapping:   {NORMALIZATION_MAPPING}\n")

# Check W&B login status if enabled
if WANDB_ENABLE == "true":
    try:
        import wandb
        print("✓ wandb package found\n")
    except ImportError:
        print("WARNING: wandb package not found!")
        print("Install with: pip install wandb\n")


print("============================================================================")
print("Starting training...")
print("============================================================================")
print()

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

cmd = [
    sys.executable, "-m", "lerobot.scripts.lerobot_train",
    f"--policy.path={BASE_POLICY_PATH}",
    f"--policy.repo_id={POLICY_REPO_ID}",
    f"--policy.push_to_hub={POLICY_PUSH_TO_HUB}",
    f"--dataset.repo_id={DATASET_REPO_ID}",
    "--rename_map={\"observation.images.main\": \"observation.images.image\", \"observation.images.secondary\": \"observation.images.image2\"}",
    f"--dataset.image_transforms.enable={ENABLE_AUGMENTATION}",
    f"--dataset.image_transforms.tfs={{\"affine\": {{\"type\": \"RandomAffine\", \"kwargs\": {{\"degrees\": {AUGMENTATION_DEGREES}, \"translate\": {AUGMENTATION_TRANSLATE}}}}}}}",
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
    "--policy.dtype=bfloat16",
    f"--policy.action_mode={ACTION_MODE}",
    f"--policy.empty_cameras={EMPTY_CAMERAS}",
    "--policy.freeze_vision_encoder=false",
    "--policy.freeze_language_encoder=false",
    "--policy.train_policy_transformer=true",
    "--policy.train_soft_prompts=true",
    f"--policy.num_image_views={POLICY_NUM_IMAGE_VIEWS}",
    f"--policy.normalization_mapping={NORMALIZATION_MAPPING}",
]

try:
    # Run the command
    exit_code = subprocess.call(cmd, env=os.environ.copy())
except KeyboardInterrupt:
    exit_code = 130
except Exception as e:
    print(f"Error launching training: {e}")
    exit_code = 1

print("\n============================================================================")
if exit_code == 0:
    print("Training completed successfully!")
else:
    print(f"Training failed with exit code: {exit_code}")
print("============================================================================")

sys.exit(exit_code)
