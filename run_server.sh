#!/bin/bash
# Script to run the lerobot policy server with the vla conda environment

# Activate conda environment
source ~/conda/etc/profile.d/conda.sh
conda activate vla

# Run the policy server
python -m lerobot.async_inference.policy_server --host 127.0.0.1 --port 8080 "$@"
