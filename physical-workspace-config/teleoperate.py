from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower

camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30)
}

robot_config = KochFollowerConfig(
    port="/dev/tty.usbmodem5AB90655421",
    id="my_red_robot_arm",
    cameras=camera_config
)

teleop_config = KochLeaderConfig(
    port="/dev/tty.usbmodem5AAF2198491",
    id="my_blue_leader_arm",
)

robot = KochFollower(robot_config)
teleop_device = KochLeader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)