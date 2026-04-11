import pybullet as p
import pybullet_data
import numpy as np

class SO101PyBullet:
    """Wrapper for PyBullet visualization of the SO-101 arm."""

    def __init__(self, urdf_path, start_pos=(0, 0, 0), start_ori=(0, 0, 0, 1)):
        """
        Initializes the PyBullet GUI, loads the ground plane, and loads the robot URDF.
        """
        self.urdf_path = urdf_path
        
        # Connect to PyBullet with GUI
        self.client_id = p.connect(p.GUI)
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet GUI.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        # Load the SO-101 robot
        self.robot_id = p.loadURDF(
            str(self.urdf_path),
            start_pos,
            start_ori,
            useFixedBase=True
        )

        # Map joint names to their indices
        self.joint_indices = {}
        self.active_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            self.joint_indices[joint_name] = i
            # joint types 0 (revolute), 1 (prismatic), 2 (spherical) are movable
            if info[2] != p.JOINT_FIXED:
                self.active_joints.append(joint_name)

    def display(self, joint_rads, joint_names=None):
        """
        Sets the joint angles in PyBullet.
        Args:
            joint_rads (list or np.ndarray): Joint angles in radians.
            joint_names (list, optional): List of joint names corresponding to joint_rads.
                                          If None, expects 6 hardcoded joints.
        """
        if joint_names is None:
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
            
        for i, name in enumerate(joint_names):
            if name in self.joint_indices:
                idx = self.joint_indices[name]
                p.resetJointState(self.robot_id, idx, joint_rads[i])

    def add_sphere(self, name, xyz, color, radius=0.002):
        """
        Adds a visual sphere to the simulation.
        Args:
            name (str): Identifier (not strongly used in PyBullet, kept for compatibility).
            xyz (list or np.ndarray): [x, y, z] position.
            color (int or list): Hex color or RGB list (range 0-1).
            radius (float): Radius of the sphere.
        """
        # Convert hex color to RGB
        if isinstance(color, int):
            r = ((color >> 16) & 255) / 255.0
            g = ((color >> 8) & 255) / 255.0
            b = (color & 255) / 255.0
            color = [r, g, b, 1.0]

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=xyz
        )

    def add_line(self, name, p1, p2, color=0xffa500, thickness=2.0):
        """
        Adds a debug line between two points.
        Args:
            name (str): Identifier.
            p1 (list or np.ndarray): Start point [x, y, z].
            p2 (list or np.ndarray): End point [x, y, z].
            color (int or list): Hex color or RGB list (range 0-1).
            thickness (float): Line width.
        """
        # Convert hex color to RGB
        if isinstance(color, int):
            color = [
                ((color >> 16) & 255) / 255.0,
                ((color >> 8) & 255) / 255.0,
                (color & 255) / 255.0
            ]

        p.addUserDebugLine(
            lineFromXYZ=p1,
            lineToXYZ=p2,
            lineColorRGB=color,
            lineWidth=thickness,
            lifeTime=0  # 0 means permanent
        )

    def visualize_path(self, waypoints, color):
        """Draws the target path as connected line segments in PyBullet."""
        for i in range(len(waypoints) - 1):
            self.add_line(f"pattern/seg_{i}", waypoints[i], waypoints[i + 1], color)
        
        # Mark start point with a sphere
        self.add_sphere("pattern/start", waypoints[0], 0xffffff, radius=0.012)

    def disconnect(self):
        """Disconnects the PyBullet client."""
        p.disconnect(self.client_id)
