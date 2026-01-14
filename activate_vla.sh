#!/bin/bash
# Activate the vla conda environment and disable user site-packages
source ~/conda/etc/profile.d/conda.sh
conda activate vla
export PYTHONNOUSERSITE=1
echo "VLA environment activated"
echo "Python: $(which python)"
echo "To run server: python -m lerobot.async_inference.policy_server --host 127.0.0.1 --port 8080"
