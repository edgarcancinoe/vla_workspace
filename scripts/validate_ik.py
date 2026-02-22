#!/usr/bin/env python3
"""
Script to validate Inverse Kinematics (IK) for the SOARM101 real robot.
It loops through a standard sequence of Cartesian target poses, computes the IK
joint angles using Placo, and commands the real SOARM101 follower to reach them.
"""

import argparse
import time
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower.so101_follower import So101FollowerRobot
from lerobot.robots.utils import get_arm_id

# --- Configuration ---
URDF_PATH = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf" 
PORT = "/dev/tty.usbmodem585A0079511"  # Default Dynamixel port
# ---------------------

# Optional visualization imports, if needed for debugging the poses before sending
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_target_pose(x, y, z):
    """
    Create a 4x4 transformation matrix for a target position (x, y, z)
    with a straight (identity) rotation.
    """
    matrix = np.eye(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def main():
    if not URDF_PATH:
        print("Please configure URDF_PATH at the top of the script.")
        return

    print("Initializing real robot connection...")
    robot = So101FollowerRobot(port=PORT)
    robot.connect()
    print("Robot connected successfully.")

    try:
        # Read the current joint states to initialize the IK solver
        current_state = robot.read()
        current_joints_rad = current_state["present_position"]
        current_joints_deg = np.rad2deg(current_joints_rad)
        
        # Determine the number of joints based on the actual hardware connections
        joint_names = list(robot.arm_bus.motors.keys())
        print(f"Detected joint names: {joint_names}")

        print("Initializing kinematics solver...")
        kinematics = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )

        # Print current EEF pose
        initial_pose = kinematics.forward_kinematics(current_joints_deg)
        print(f"Initial EEF Pose:\n{initial_pose}")

        # The user requested a standard single position:
        # "in front of the center of the robot and 10cm elevataed on z axis. rotation should be just straight"
        # Assuming robot base is at (0,0,0) and +X is forward:
        target_x = 0.20 # 20cm forward
        target_y = 0.0  # Center
        target_z = 0.10 # 10cm elevated
        
        target_pose = create_target_pose(target_x, target_y, target_z)
        
        print("\nTarget EEF Pose:")
        print(target_pose)

        print("\nSolving Inverse Kinematics...")
        # We set orientation_weight=0.0 to only constrain the position if we just want it to reach the point,
        # but the user requested "rotation should be just straight", so we give orientation some weight.
        target_joints_deg = kinematics.inverse_kinematics(
            current_joint_pos=current_joints_deg,
            desired_ee_pose=target_pose,
            position_weight=1.0,
            orientation_weight=0.1, 
        )

        target_joints_rad = np.deg2rad(target_joints_deg)
        print(f"Target Joint Angles (rad): {target_joints_rad}")

        # Basic collision/safety check (very rough)
        # Assuming joint 0 is base pan, joint 1 is shoulder lift (if shoulder lift goes too negative it hits the table)
        # This is URDF dependent, so just a structural placeholder
        if np.any(np.isnan(target_joints_rad)):
            print("IK Solution failed (NaN returned). Aborting.")
            return

        print("\nSending command to robot in 2 seconds. PLEASE BE READY TO E-STOP...")
        time.sleep(2)

        # Because the real robot expects a 6-element action (5 joints + 1 gripper),
        # but the IK might only solve for the arm joints, let's keep the gripper's 
        # current state the same.
        action = np.copy(current_joints_rad)
        action[:len(target_joints_rad)] = target_joints_rad[:len(joint_names)]

        # Send command
        robot.write(action)
        print("Command sent! Waiting 3 seconds for it to arrive...")
        time.sleep(3)
        
        final_state = robot.read()
        final_pose = kinematics.forward_kinematics(np.rad2deg(final_state["present_position"]))
        print(f"\nFinal Reached EEF Pose:\n{final_pose}")

    finally:
        print("\nDisconnecting robot...")
        robot.disconnect()


if __name__ == "__main__":
    main()
