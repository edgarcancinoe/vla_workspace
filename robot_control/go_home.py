#!/usr/bin/env python3
import yaml
import sys
import argparse
from pathlib import Path

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

    # URDF Path - constant for this setup
    URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

    # Define shared calibration directory
    calibration_dir = WORKSPACE_ROOT / ".cache" / "calibration"

    robot_id = robot_cfg.get("name", "arm_follower")
    wrist_roll_offset = robot_cfg.get("wrist_roll_offset", 0.0)
    
    print(f"Initializing Robot on {port} with calibration '{robot_id}'...")
    follower_config = SO101FollowerConfig(id=robot_id, port=port, calibration_dir=calibration_dir)
    robot = SO101Follower(follower_config)
    
    try:
        robot.connect(calibrate=args.calibrate)
        control = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=wrist_roll_offset, home_pose=home_pose)
        
        # Apply safety limits if needed
        # control.configure_safety(robot)

        print("Starting 'Go Home' sequence...")
        print(f"Target Home Pose (Degrees): {home_pose}")
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
