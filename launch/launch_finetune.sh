#!/bin/bash

# ============================================================================
# SmolVLA Finetuning Launch Script
# ============================================================================
# This script launches the finetuning process for the SmolVLA model using 
# lerobot code.
# ============================================================================

# Activate vla conda environment
# Check if we're already in the vla environment
if [[ "$CONDA_DEFAULT_ENV" != "vla" ]]; then
    echo "Activating vla conda environment..."
    # Try different activation methods
    if [ -f "$HOME/conda/etc/profile.d/conda.sh" ]; then
        source "$HOME/conda/etc/profile.d/conda.sh"
        conda activate vla
    elif [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
        source "$CONDA_PREFIX/etc/profile.d/conda.sh"
        conda activate vla
    else
        echo "WARNING: Could not find conda.sh. Assuming environment is already activated."
    fi
fi

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
# Use local /tmp for all cache to avoid NFS issues
CACHE_ROOT="/tmp/vla_cache_${USER}"
mkdir -p "$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT"
export HF_LEROBOT_HOME="${CACHE_ROOT}/lerobot"

# Create directories
mkdir -p "$HF_LEROBOT_HOME"

# ============================================================================
# CONFIGURATION - Adjust these variables to your setup
# ============================================================================

# Workspace path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Dataset Configuration
# Replace with your Hugging Face username and dataset name
HF_USER="${HF_USER:-edgarcancinoe}"
# DATASET_NAME="${DATASET_NAME:-soarm101_pick_cubes_place_box}"
DATASET_NAME="${DATASET_NAME:-soarm101_pick_cubes_place_box_v2}"
DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

# Training Hyperparameters
# Batch size: number of samples processed in parallel before gradient update
# Reduce this if you have low GPU memory (try 32, 16, or 8)
BATCH_SIZE="${BATCH_SIZE:-64}"

# Number of training steps
STEPS="${STEPS:-20000}"

# Output Configuration
# Directory where training logs and checkpoints will be saved
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/train/${DATASET_NAME}_smolvla_${TIMESTAMP}}"

# Job name for logging and Weights & Biases
JOB_NAME="${JOB_NAME:-${DATASET_NAME}_smolvla_finetuning_${TIMESTAMP}}"

# Device Configuration
# Options: cuda (NVIDIA GPU), mps (Apple Silicon), cpu (no GPU)
DEVICE="${DEVICE:-cuda}"

# CUDA Device Selection
# Specify which GPU to use (0, 1, 2, etc.). Leave empty to use all available GPUs.
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Weights & Biases Configuration
# Set to true to enable W&B logging (requires: wandb login)
# Set to false to disable W&B
WANDB_ENABLE="${WANDB_ENABLE:-true}"

# Model Configuration
# Path to the base model
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"

# HuggingFace Hub repository to push the trained model to
# Format: your_hf_username/model_name
POLICY_REPO_ID="${POLICY_REPO_ID:-${HF_USER}/smolvla_finetuned}"

# ============================================================================
# OPTIONAL: Additional Training Arguments
# ============================================================================
# Uncomment and modify these if you need more control

# Learning rate
# LR="${LR:-1e-4}"

# Gradient accumulation steps (effective batch size = BATCH_SIZE * GRAD_ACCUM)
# GRAD_ACCUM="${GRAD_ACCUM:-1}"

# Save checkpoint every N steps
SAVE_FREQ="${SAVE_FREQ:-5000}"

POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-true}"

# Evaluation frequency
# EVAL_FREQ="${EVAL_FREQ:-500}"

# ============================================================================
# DEVICE SETUP
# ============================================================================
# Set CUDA visibility if a specific device is requested
if [ -n "$CUDA_DEVICE" ]; then
    echo "Restricting CUDA visibility to device: $CUDA_DEVICE"
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
else
    # Default to all GPUs (0,1,2,3) if not specified
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo "============================================================================"
echo "SmolVLA Finetuning Launch Script"
echo "============================================================================"
echo ""



# Check if lerobot Python package is installed
if ! python -c "import lerobot" &> /dev/null; then
    echo "ERROR: lerobot Python package not found!"
    echo ""
    echo "Please install lerobot first:"
    echo "  pip install lerobot"
    echo ""
    echo "Or if you have a local clone, ensure you're in the correct conda environment."
    exit 1
fi

echo "✓ lerobot package found"
echo ""

# Check if HF_USER is set
if [ "$HF_USER" = "YOUR_HF_USERNAME" ]; then
    echo "WARNING: HF_USER is not set!"
    echo "Please set your Hugging Face username:"
    echo "  export HF_USER=your_username"
    echo "  or edit this script directly"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display configuration
echo "Configuration:"
echo "  Dataset:        ${DATASET_REPO_ID}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  Steps:          ${STEPS}"
echo "  Save Freq:      ${SAVE_FREQ}"
echo "  Output Dir:     ${OUTPUT_DIR}"
echo "  Job Name:       ${JOB_NAME}"
echo "  Device:         ${DEVICE}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  W&B Enabled:    ${WANDB_ENABLE}"
echo "  Policy Path:    ${POLICY_PATH}"
echo "  Policy Repo ID: ${POLICY_REPO_ID}"
echo "  Push to Hub:    ${POLICY_PUSH_TO_HUB}"
echo ""

# Check W&B login status if enabled
if [ "$WANDB_ENABLE" = "true" ]; then
    if ! python -c "import wandb" &> /dev/null; then
        echo "WARNING: wandb package not found!"
        echo "Install with: pip install wandb"
        echo ""
    else
        # Check if logged in by checking if API key exists
        if ! python -c "import wandb; wandb.api.api_key" &> /dev/null; then
            echo "WARNING: Not logged in to Weights & Biases!"
            echo "Please run: python -m wandb login"
            echo ""
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            echo "✓ Logged in to W&B"
            echo ""
        fi
    fi
fi

echo "============================================================================"
echo "Starting training..."
echo "============================================================================"
echo ""

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

python -m lerobot.scripts.lerobot_train \
  --policy.path="${POLICY_PATH}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub="${POLICY_PUSH_TO_HUB}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --rename_map='{"observation.images.lateral": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}' \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device="${DEVICE}" \
  --wandb.enable="${WANDB_ENABLE}"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "============================================================================"

exit $EXIT_CODE
