#!/bin/bash
# Script to run the lerobot policy server with the thesis conda environment

# Activate conda environment
source /home/jose/conda/etc/profile.d/conda.sh
conda activate vla

# Use GPU 7 (the free one)
export CUDA_VISIBLE_DEVICES=7

# Set HuggingFace cache directories to a location with 1.7TB free space
export HF_HOME="$HOME/huggingface_cache"
export TRANSFORMERS_CACHE="$HOME/huggingface_cache/transformers"
export HF_DATASETS_CACHE="$HOME/huggingface_cache/datasets"
export HF_HUB_CACHE="$HOME/huggingface_cache/hub"

# Run the policy server
python -m lerobot.async_inference.policy_server --config_path launch_server.yaml "$@"
