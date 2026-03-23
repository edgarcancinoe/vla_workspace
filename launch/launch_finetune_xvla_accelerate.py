import os
import sys
import subprocess
import datetime
import itertools
import json
import importlib
from pathlib import Path
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"

# ============================================================================
# X-VLA Finetuning Launch Script (Grid Search, Accelerate / Multi-GPU)
# ============================================================================
# This launcher preserves the current grid-search workflow, but each run is
# started through `accelerate launch` for single-node multi-GPU training.
# Configure GPU selection by editing CUDA_DEVICE_INDICES below. No CLI
# passthrough is required for GPU/process selection.
# ============================================================================

# Check if we're already in the vla environment
if os.environ.get("CONDA_DEFAULT_ENV") != "vla":
    print("WARNING: 'vla' conda environment is not activated.")

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
USER = os.environ.get("USER", "default_user")
CACHE_ROOT = f"/tmp/vla_cache_{USER}"
os.makedirs(CACHE_ROOT, exist_ok=True)

os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HF_LEROBOT_HOME"] = f"{CACHE_ROOT}/lerobot"
os.makedirs(os.environ["HF_LEROBOT_HOME"], exist_ok=True)

# ============================================================================
# CONFIGURATION - Adjust these variables to your setup
# ============================================================================
HF_USER = os.environ.get("HF_USER", "edgarcancinoe")

# Dataset to use -----------------------------------
DATASET_NAME_STR = "soarm101_pickplace_10d"
# --------------------------------------------------

# Base model ---------------------------------------
BASE_USER = "lerobot"
BASE_NAME = "xvla-base"
VERSION = os.environ.get("VERSION", "v1")

# Policy Configuration Grids
ACTION_MODES = [
    "so101_ee6d",
    # "so101_joint",
]

NORMALIZATION_MAPPINGS = [
    '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
]

# XVLA finetune contract
EMPTY_CAMERAS = 1
POLICY_NUM_IMAGE_VIEWS = 3
POLICY_TOKENIZER_MAX_LENGTH = 64
POLICY_MAX_LEN_SEQ = 1024
DATASET_VIDEO_BACKEND = "pyav"

# Accelerate / distributed launch configuration
CUDA_DEVICE_INDICES = [0, 1]
NUM_WORKERS = "0"

NUM_PROCESSES = len(CUDA_DEVICE_INDICES)
MAIN_PROCESS_PORT = 45001
MIXED_PRECISION = "bf16"

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
# BATCH_SIZE is per process / per GPU.
BATCH_SIZE = "8"
STEPS = "45000"
LOG_FREQ = "1000"
EVAL_FREQ = "-1"

DEVICE = "cuda"

SAVE_FREQ = "15000"
PUSH_HF_EVERY = "15000"

# Resume configuration
RESUME = "false"
POLICY_PUSH_TO_HUB = "true"
WANDB_ENABLE = "false"

# Policy Structure / Freezing
POLICY_DTYPE = "bfloat16"
FREEZE_VISION_ENCODER = "false"
FREEZE_LANGUAGE_ENCODER = "false"
TRAIN_POLICY_TRANSFORMER = "true"
TRAIN_SOFT_PROMPTS = "true"

# Data Overrides
RENAME_MAP = '{"observation.images.main": "observation.images.image", "observation.images.secondary": "observation.images.image2"}'


def get_norm_suffix(mapping_str: str) -> str:
    """Generate a clean identifier for the parameter name based on json keys."""
    try:
        mapping = json.loads(mapping_str)
        am = str(mapping.get("ACTION", "ID")).replace("_", "").lower()[:1]
        sm = str(mapping.get("STATE", "ID")).replace("_", "").lower()[:1]
        return f"a-{am}_s-{sm}"
    except Exception:
        return "custom"


# ============================================================================
# DEVICE / DISTRIBUTED SETUP
# ============================================================================
if not CUDA_DEVICE_INDICES:
    raise ValueError("CUDA_DEVICE_INDICES must not be empty for the accelerate launcher.")

cuda_visible_devices = ",".join(str(idx) for idx in CUDA_DEVICE_INDICES)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
print("============================================================================")
print("X-VLA Finetuning Launch Script (Grid Search, Accelerate)")
print("============================================================================")
print()

try:
    importlib.import_module("lerobot")
    print("✓ lerobot package found")
except ImportError:
    print("ERROR: lerobot Python package not found!")
    print("Please install lerobot first or ensure the correct environment is active.")
    sys.exit(1)

try:
    importlib.import_module("accelerate")
    print("✓ accelerate package found")
except ImportError:
    print("ERROR: accelerate package not found!")
    print("Install it in the active environment with: python -m pip install accelerate")
    sys.exit(1)

if HF_USER == "YOUR_HF_USERNAME":
    print("WARNING: HF_USER is not set!")
    print("Please set your Hugging Face username or edit this script directly.")
    response = input("Continue anyway? (y/n) ")
    if response.strip().lower() != "y":
        sys.exit(1)

effective_batch_size = int(BATCH_SIZE) * NUM_PROCESSES

print()
print("Distributed Configuration:")
print(f"  CUDA Device Indices:  {CUDA_DEVICE_INDICES}")
print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
print(f"  Num Processes:        {NUM_PROCESSES}")
print(f"  Main Process Port:    {MAIN_PROCESS_PORT}")
print(f"  Mixed Precision:      {MIXED_PRECISION}")
print(f"  Batch Size / GPU:     {BATCH_SIZE}")
print(f"  Effective Batch Size: {BATCH_SIZE} x {NUM_PROCESSES} = {effective_batch_size}")
print()

# ============================================================================
# RUN GRID SEARCH LOOP
# ============================================================================
num_runs = len(ACTION_MODES) * len(NORMALIZATION_MAPPINGS)
print("============================================================================")
print(f"Starting {num_runs} accelerate grid search iteration(s)...")
print("============================================================================")
print()

for action_mode, norm_mapping in itertools.product(ACTION_MODES, NORMALIZATION_MAPPINGS):
    norm_suffix = get_norm_suffix(norm_mapping)

    policy_name = f"{BASE_NAME}_{DATASET_NAME_STR}_{action_mode}_{norm_suffix}"
    if ENABLE_AUGMENTATION == "true":
        policy_name += "_aug"
    policy_name += f"_{VERSION}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = os.environ.get("JOB_NAME", f"{policy_name}_{timestamp}")
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        str(WORKSPACE_DIR / "outputs" / "train" / f"{policy_name}_{timestamp}"),
    )
    policy_repo_id = f"{HF_USER}/{policy_name}"

    print("\n" + "=" * 76)
    print("LAUNCHING ACCELERATE TRAINING RUN")
    print("=" * 76)
    print(f"  Dataset:              {DATASET_REPO_ID}")
    print(f"  Action Mode:          {action_mode}")
    print(f"  Norm Mapping:         {norm_mapping}")
    print(f"  Output Dir:           {output_dir}")
    print(f"  Policy Repo ID:       {policy_repo_id}")
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"  Num Processes:        {NUM_PROCESSES}")
    print(f"  Batch / GPU:          {BATCH_SIZE}")
    print(f"  Effective Batch:      {effective_batch_size}")
    print("=" * 76 + "\n")

    augmentation_tfs = (
        f'{{"affine": {{"type": "RandomAffine", "kwargs": {{"degrees": {AUGMENTATION_DEGREES},'
        f' "translate": {AUGMENTATION_TRANSLATE}}}}}}}'
    )

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        f"--num_processes={NUM_PROCESSES}",
        "--num_machines=1",
        f"--main_process_port={MAIN_PROCESS_PORT}",
        f"--mixed_precision={MIXED_PRECISION}",
        "--dynamo_backend=no",
    ]

    if NUM_PROCESSES > 1:
        cmd.append("--multi_gpu")
        
    cmd += [
        "--module",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={BASE_POLICY_PATH}",
        f"--policy.repo_id={policy_repo_id}",
        f"--policy.push_to_hub={POLICY_PUSH_TO_HUB}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--rename_map={RENAME_MAP}",
        f"--dataset.image_transforms.enable={ENABLE_AUGMENTATION}",
        f"--dataset.image_transforms.tfs={augmentation_tfs}",
        f"--dataset.video_backend={DATASET_VIDEO_BACKEND}",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        f"--log_freq={LOG_FREQ}",
        f"--eval_freq={EVAL_FREQ}",
        f"--save_freq={SAVE_FREQ}",
        f"--push_every={PUSH_HF_EVERY}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
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
        f"--policy.tokenizer_max_length={POLICY_TOKENIZER_MAX_LENGTH}",
        f"--policy.max_len_seq={POLICY_MAX_LEN_SEQ}",
        f"--policy.normalization_mapping={norm_mapping}",
    ]

    try:
        exit_code = subprocess.call(cmd, env=os.environ.copy())
    except KeyboardInterrupt:
        print("\n\nUser interrupted with Ctrl+C. Exiting accelerate grid search.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error launching accelerate training: {exc}")
        exit_code = 1

    if exit_code != 0:
        print(f"\nTraining failed with exit code: {exit_code}")
        print("Stopping further grid search iterations.")
        sys.exit(exit_code)

    print(f"\nCompleted run for {action_mode} | {norm_suffix}")

print("\n============================================================================")
print("All requested accelerate training runs completed successfully!")
print("============================================================================")
sys.exit(0)
