#!/bin/bash
# Script to run the lerobot policy server with the thesis conda environment

# Activate conda environment
source /home/jose/conda/etc/profile.d/conda.sh
conda activate vla

# Set HuggingFace cache directories to avoid permission issues with /opt/cache
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the policy server
python -m lerobot.async_inference.policy_server --config_path "${SCRIPT_DIR}/launch_server.yaml" "$@"

