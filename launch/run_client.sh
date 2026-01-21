#!/bin/bash
# Script to run the lerobot robot client with the thesis conda environment

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Run the robot client
python -m lerobot.async_inference.robot_client --config_path launch_client.yaml "$@"
