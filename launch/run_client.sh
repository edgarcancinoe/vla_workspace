#!/bin/bash
# Script to run the lerobot robot client with the thesis conda environment

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Run the robot client
# Extract follower port from robot_config.yaml
FOLLOWER_PORT=$(grep "port:" robot_config.yaml | grep -v "leader" | head -n 1 | awk '{print $2}')

# Run the robot client with the extracted port overriding the config
python -m lerobot.async_inference.robot_client --config_path launch_client.yaml --robot.port "$FOLLOWER_PORT" "$@"
