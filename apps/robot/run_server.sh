#!/bin/bash
# Script to run the lerobot policy server with the thesis conda environment

# Activate conda environment
if [ -f "$HOME/conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/conda/etc/profile.d/conda.sh"
  conda activate vla
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate vla
fi

# Set HuggingFace cache directories to avoid permission issues with /opt/cache
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Run the policy server
python -m lerobot.async_inference.policy_server --config_path "${PROJECT_ROOT}/config/launch/launch_server.yaml" "$@"
