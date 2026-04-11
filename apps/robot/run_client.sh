#!/bin/bash
# Script to run the lerobot robot client with the thesis conda environment

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate thesis

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Extract follower port from robot_config.yaml
FOLLOWER_PORT=$(grep "port:" "${PROJECT_ROOT}/config/robot/robot_config.yaml" | grep -v "leader" | head -n 1 | awk '{print $2}')

# Run the robot client with the extracted port overriding the config
python -m lerobot.async_inference.robot_client --config_path "${PROJECT_ROOT}/config/launch/launch_client.yaml" --robot.port "$FOLLOWER_PORT" "$@"
