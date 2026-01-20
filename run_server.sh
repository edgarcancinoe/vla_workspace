#!/bin/bash
# Script to run the lerobot policy server with the thesis conda environment

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Run the policy server
python -m lerobot.async_inference.policy_server --config_path launch_server.yaml "$@"
