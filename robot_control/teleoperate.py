import sys
import yaml
import shutil
import argparse
import rerun as rr
import numpy as np
from pathlib import Path

# Add project root to path to find utils and robot_control
sys.path.append(str(Path(__file__).parent.parent))
from robot_control.so101_control import SO101Control
from utils import camera_calibration

# Load config
config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)


# Configuration for Rectification from robot_config.yaml
RECTIFY_TOP = config_data.get("rectification", {}).get("top", True)
RECTIFY_WRIST = config_data.get("rectification", {}).get("wrist", True)

# Wrist Roll Offset (in Degrees from config)
WRIST_ROLL_OFFSET_DEG = config_data.get("robot", {}).get("wrist_roll_offset", 0.0)

# Pre-calculate Wrist Roll Offset in Motor Units
# 1.0 degree = (100 / limit_deg) motor units
limit_deg = np.rad2deg(SO101Control.URDF_LIMITS_RAD['wrist_roll'])
WRIST_ROLL_OFFSET_MOTOR = (WRIST_ROLL_OFFSET_DEG / limit_deg) * 100.0


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

parser = argparse.ArgumentParser()
parser.add_argument("--calibrate", action="store_true", help="Force recalibration of the robot and leader arm")
args = parser.parse_args()

follower_port = config_data["robot"]["port"]
# New structure: leader: {port: ..., name: ...}
leader_dict = config_data.get("leader", {})
leader_port = leader_dict.get("port", config_data.get("leader_port"))

# Define shared calibration directory
calibration_dir = Path(__file__).parent.parent / ".cache" / "calibration"

if args.calibrate:
    print(f"Force calibration requested. Removing calibration files in {calibration_dir}...")
    if calibration_dir.exists():
        shutil.rmtree(calibration_dir)
        print("Calibration files removed.")

print(f"Follower Port: {follower_port}")
print(f"Leader Port: {leader_port}")

camera_config = {
    name: OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=30)
    for name, idx in config_data.get("cameras", {}).items()
}

# Get IDs from config
FOLLOWER_ID = config_data["robot"].get("name")
LEADER_ID = leader_dict.get("name")

robot_config = SO100FollowerConfig(
    port=follower_port,
    id=FOLLOWER_ID,
    cameras=camera_config,
    calibration_dir=calibration_dir
)

teleop_config = SO100LeaderConfig(
    port=leader_port,
    id=LEADER_ID,
    calibration_dir=calibration_dir
)

robot = SO100Follower(robot_config)
teleop_device = SO100Leader(teleop_config)

# Kinematics Control for Degree Conversion
URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
home_pose = config_data.get("robot", {}).get("home_pose")
control = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=WRIST_ROLL_OFFSET_DEG, home_pose=home_pose)

print("Connecting devices...")
# Pass calibrate flag to connection methods
robot.connect(calibrate=args.calibrate)
teleop_device.connect(calibrate=args.calibrate)

init_rerun(session_name="teleoperate_debug")

print("Connected! Teleoperating...")

step = 0
print_interval = 10  # Print positions every 30 steps (~1 second at 30 FPS)

try:
    while True:
        observation = robot.get_observation()
        
        # Debug: print observation keys on first frame
        if step == 0:
            print(f"DEBUG: Observation keys: {list(observation.keys())}")
        
        action = teleop_device.get_action()
        
        # Apply wrist roll offset (using pre-calculated Motor Unit shift)
        if "wrist_roll.pos" in action:
            new_val = action["wrist_roll.pos"] + WRIST_ROLL_OFFSET_MOTOR
            # Clamp to range [-100, 100]
            action["wrist_roll.pos"] = max(min(new_val, 100.0), -100.0)
            
        robot.send_action(action)
        
        # Rectify images based on configuration
        if RECTIFY_TOP and "top" in observation:
            observation["top"] = camera_calibration.rectify_image(
                observation["top"], "top"
            )
            
        if RECTIFY_WRIST and "wrist" in observation:
            observation["wrist"] = camera_calibration.rectify_image(
                observation["wrist"], "wrist"
            )
        
        # Print positions periodically
        if step % print_interval == 0:
            # Prepare motor values
            motor_vals = [action[f"{n}.pos"] for n in SO101Control.JOINT_NAMES]
            deg_vals = control.motor_to_deg(motor_vals)
            
            print(f"\n--- Step {step} ---")
            print("Joint Positions (Motor Units | Degrees):")
            for i, n in enumerate(SO101Control.JOINT_NAMES):
                m_val = action[f"{n}.pos"]
                d_val = deg_vals[i]
                print(f"  {n:15s}: {m_val:8.4f} units | {d_val:8.2f}°")
        
        rr.set_time_sequence("step", step)
        log_rerun_data(observation=observation, action=action)
        
        step += 1
except KeyboardInterrupt:
    print("\nStopping...")
    robot.disconnect()
    teleop_device.disconnect()