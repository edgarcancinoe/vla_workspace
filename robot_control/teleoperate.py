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


# Removed global RECTIFY_TOP/WRIST as they are now per-camera in config

# Wrist Roll Offset (in Degrees from config)
WRIST_ROLL_OFFSET_DEG = config_data.get("robot", {}).get("wrist_roll_offset", 0.0)

# Pre-calculate Wrist Roll Offset in Motor Units
# 1.0 degree = (100 / limit_deg) motor units
limit_deg = np.rad2deg(SO101Control.URDF_LIMITS_RAD['wrist_roll'])
WRIST_ROLL_OFFSET_MOTOR = (WRIST_ROLL_OFFSET_DEG / limit_deg) * 100.0


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

parser = argparse.ArgumentParser()
parser.add_argument("--calibrate", action="store_true", help="Force recalibration of the robot and leader arm")
args = parser.parse_args()

follower_port = config_data["robot"]["port"]
# New structure: leader: {port: ..., name: ...}
leader_dict = config_data.get("leader", {})
leader_port = leader_dict.get("port", config_data.get("leader_port"))

# Define shared calibration directory - strictly from config
calibration_dir = config_data.get("robot", {}).get("calibration_dir")
if not calibration_dir:
    print("Error: 'calibration_dir' not found in config/robot_config.yaml.")
    sys.exit(1)
calibration_dir = Path(calibration_dir)

if args.calibrate:
    print(f"Force calibration requested. Removing calibration files in {calibration_dir}...")
    if calibration_dir.exists():
        shutil.rmtree(calibration_dir)
        print("Calibration files removed.")

print(f"Follower Port: {follower_port}")
print(f"Leader Port: {leader_port}")

camera_config = {
    name: OpenCVCameraConfig(index_or_path=info["id"], width=640, height=480, fps=30)
    for name, info in config_data.get("cameras", {}).items()
}
# Store which cameras need rectification
RECTIFY_MAP = {
    name: info.get("rectify", False)
    for name, info in config_data.get("cameras", {}).items()
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

teleop_config = SOLeaderTeleopConfig(
    port=leader_port,
    id=LEADER_ID,
    calibration_dir=calibration_dir
)

robot = SO100Follower(robot_config)
teleop_device = SOLeader(teleop_config)

# Kinematics Control for Degree Conversion
URDF_PATH = config_data.get("robot", {}).get("urdf_path")
if not URDF_PATH:
    print("Error: 'urdf_path' not found in config/robot_config.yaml.")
    sys.exit(1)
home_pose = config_data.get("robot", {}).get("home_pose")
control = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=WRIST_ROLL_OFFSET_DEG, home_pose=home_pose)

print("Connecting devices...")
# Pass calibrate flag to connection methods
robot.connect(calibrate=args.calibrate)
teleop_device.connect(calibrate=args.calibrate)

init_rerun(session_name="teleoperate_debug")

# Log static world origin for reference
rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)), static=True)

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
        
        # Apply wrist roll offset via control helper
        motor_vals = np.array([action[f"{n}.pos"] for n in control.JOINT_NAMES])
        motor_vals = control.apply_wrist_roll_offset(motor_vals)
        for i, n in enumerate(control.JOINT_NAMES):
            action[f"{n}.pos"] = float(motor_vals[i])
            
        robot.send_action(action)
        
        # Rectify images based on per-camera configuration
        for cam_name, should_rectify in RECTIFY_MAP.items():
            if should_rectify and cam_name in observation:
                observation[cam_name] = camera_calibration.rectify_image(
                    observation[cam_name], cam_name
                )
        
        # Print positions periodically
        if step % print_interval == 0:
            # Get observed motor values for "live" pose using control helper
            obs_motor_vals = control.extract_motor_vals(observation)
            deg_vals = control.motor_to_deg(obs_motor_vals)
            
            # Calculate FK Pose using control helper
            pos, euler = control.fk_pose(obs_motor_vals)
            
            print(f"\n--- Step {step} ---")
            print("Joint Positions (Motor Units | Degrees):")
            for i, n in enumerate(control.JOINT_NAMES):
                m_val = obs_motor_vals[i]
                d_val = deg_vals[i]
                print(f"  {n:15s}: {m_val:8.4f} units | {d_val:8.2f}°")
            
            print("\nEnd-Effector Pose (Cartesian):")
            print(f"  X: {pos[0]:8.4f} m")
            print(f"  Y: {pos[1]:8.4f} m")
            print(f"  Z: {pos[2]:8.4f} m")
            print(f"  Orientation (Euler xyz): Roll={euler[0]:.2f}°, Pitch={euler[1]:.2f}°, Yaw={euler[2]:.2f}°")
        
        rr.set_time_sequence("step", step)
        
        log_rerun_data(observation=observation, action=action)

        
        step += 1
            
except KeyboardInterrupt:
    print("\nStopping...")
    robot.disconnect()
    teleop_device.disconnect()