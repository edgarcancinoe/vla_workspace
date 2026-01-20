#!/bin/bash
# Script to run the lerobot policy server with the thesis conda environment

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Set HuggingFace cache directories to avoid permission issues with /opt/cache
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"

# Run the policy server
python -m lerobot.async_inference.policy_server --config_path launch_server.yaml "$@"

