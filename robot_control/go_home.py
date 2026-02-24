#!/usr/bin/env python3
import yaml
import sys
import argparse
from pathlib import Path
import numpy as np
# Add project root and lerobot root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_ROOT = WORKSPACE_ROOT.parent / "repos" / "lerobot" / "src"

for path in [str(WORKSPACE_ROOT), str(LEROBOT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from robot_control.so101_control import SO101Control
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration of the arm")
    args = parser.parse_args()

    config_path = WORKSPACE_ROOT / "config" / "robot_config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    robot_cfg = config.get("robot", {})
    port = robot_cfg.get("port")
    home_pose = robot_cfg.get("home_pose")

    if not port:
        print("Error: 'port' not found in config/robot_config.yaml. Please specify the robot port.")
        sys.exit(1)
    
    if not home_pose:
        print("Error: 'home_pose' not found in config/robot_config.yaml.")
        sys.exit(1)

    # URDF Path - strictly from config
    URDF_PATH = robot_cfg.get("urdf_path")
    if not URDF_PATH:
        print("Error: 'urdf_path' not found in config/robot_config.yaml.")
        sys.exit(1)

    # Define shared calibration directory - strictly from config
    calibration_dir = robot_cfg.get("calibration_dir")
    if not calibration_dir:
        print("Error: 'calibration_dir' not found in config/robot_config.yaml.")
        sys.exit(1)
    calibration_dir = Path(calibration_dir)

    robot_id = robot_cfg.get("name", "arm_follower")
    
    print(f"Initializing Robot on {port} with calibration '{robot_id}'...")
    follower_config = SO101FollowerConfig(id=robot_id, port=port, calibration_dir=calibration_dir)
    robot = SO101Follower(follower_config)
    
    try:
        robot.connect(calibrate=args.calibrate)
        control = SO101Control(urdf_path=URDF_PATH, home_pose=home_pose)
        
        # Apply safety limits if needed
        # control.configure_safety(robot)

        print("Starting 'Go Home' sequence...")
        print(f"Target Home Pose (Kinematic Degrees): {home_pose}")
        
        control.reset_to_home(robot, duration_s=4.0)
        print("Sequence complete.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        robot.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
