import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from pathlib import Path

class SO101Meshcat:
    """Wrapper for Meshcat visualization of the SO-101 arm."""

    def __init__(self, urdf_path):
        """
        Initializes the Meshcat GUI and loads the robot URDF via Pinocchio.
        """
        self.urdf_path = urdf_path
        
        # Load Pinocchio model
        package_dir = str(Path(self.urdf_path).parent)
        self.model, self.cmod, self.vmod = pin.buildModelsFromUrdf(
            self.urdf_path, package_dirs=package_dir
        )
        
        # Initialize Meshcat Visualizer
        self.viz = MeshcatVisualizer(self.model, self.cmod, self.vmod)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.viewer = self.viz.viewer
        
        print(f"Meshcat: {self.viewer.url()}")

    def display(self, joint_rads):
        """Sets the joint angles in Meshcat."""
        self.viz.display(joint_rads)

    def add_sphere(self, name, xyz, color_hex, radius=0.002):
        """Adds a visual sphere to the simulation."""
        if self.viewer is None: return
        self.viewer[name].set_object(g.Sphere(radius), g.MeshLambertMaterial(color=color_hex))
        self.viewer[name].set_transform(tf.translation_matrix(xyz))

    def add_line(self, name, p1, p2, color_hex=0xffa500, thickness=0.002):
        """Adds a debug line between two points."""
        if self.viewer is None: return
        diff = p2 - p1
        L = np.linalg.norm(diff)
        if L < 1e-6: return
        z = np.array([0., 0., 1.])
        axis = np.cross(z, diff / L)
        ang = np.arccos(np.clip(np.dot(z, diff / L), -1, 1))
        R = tf.rotation_matrix(ang, axis) if np.linalg.norm(axis) > 1e-6 else np.eye(4)
        
        self.viewer[name].set_object(
            g.Cylinder(float(L), float(thickness)),
            g.MeshLambertMaterial(color=color_hex)
        )
        self.viewer[name].set_transform(tf.translation_matrix((p1 + p2) / 2) @ R)

    def visualize_path(self, waypoints, color_hex):
        """Draws the target path as connected line segments in Meshcat."""
        for i in range(len(waypoints) - 1):
            self.add_line(f"pattern/seg_{i}", waypoints[i], waypoints[i + 1], color_hex)
        # Mark start point
        self.add_sphere("pattern/start", waypoints[0], 0xffffff, radius=0.012)
        
    def disconnect(self):
        """Meshcat doesn't need explicit disconnect, but keep for compatibility."""
        pass
