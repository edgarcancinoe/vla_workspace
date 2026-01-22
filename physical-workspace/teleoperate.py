import yaml
import rerun as rr
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.visualization_utils import init_rerun

# Load config
config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

# ... existing code ...
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--calibrate", action="store_true", help="Force recalibration of the robot and leader arm")
args = parser.parse_args()

follower_port = config_data["robot"]["port"]
leader_port = config_data["leader_port"]

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
    "camera1": OpenCVCameraConfig(index_or_path=0, width=640, height=360, fps=30),
    "camera2": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)
}

robot_config = SO100FollowerConfig(
    port=follower_port,
    id="my_awesome_follower_arm",
    cameras=camera_config,
    calibration_dir=calibration_dir
)

teleop_config = SO100LeaderConfig(
    port=leader_port,
    id="my_awesome_leader_arm",
    calibration_dir=calibration_dir
)

robot = SO100Follower(robot_config)
teleop_device = SO100Leader(teleop_config)

print("Connecting devices...")
# Pass calibrate flag to connection methods
robot.connect(calibrate=args.calibrate)
teleop_device.connect(calibrate=args.calibrate)

init_rerun(session_name="teleoperate")

print("Connected! Teleoperating...")

step = 0
print_interval = 30  # Print positions every 30 steps (~1 second at 30 FPS)

try:
    while True:
        observation = robot.get_observation()
        action = teleop_device.get_action()
        robot.send_action(action)
        
        # Print positions periodically
        if step % print_interval == 0:
            position_dict = {key: f"{val:.4f}" for key, val in action.items() if key.endswith('.pos')}
            print("\nCurrent Positions:")
            print("performed_action = {")
            for key, val in position_dict.items():
                print(f'    "{key}": {val},')
            print("}")
        
        rr.set_time_sequence("step", step)
        for cam_name in camera_config.keys():
            key = f"observation.images.{cam_name}"
            if key in observation:
                rr.log(key, rr.Image(observation[key]))
        
        step += 1
except KeyboardInterrupt:
    print("\nStopping...")
    robot.disconnect()
    teleop_device.disconnect()