#!/bin/bash
# Script to run the lerobot robot client with the thesis conda environment

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Extract follower port from robot_config.yaml
FOLLOWER_PORT=$(grep "port:" "${WORKSPACE_DIR}/robot_config.yaml" | grep -v "leader" | head -n 1 | awk '{print $2}')

# Run the robot client with the extracted port overriding the config
python -m lerobot.async_inference.robot_client --config_path "${SCRIPT_DIR}/launch_client.yaml" --robot.port "$FOLLOWER_PORT" "$@"
