import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import copy

import yaml
from pathlib import Path

config_path = Path(__file__).parent / "config" / "robot_config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

urdf_path = config_data.get("robot", {}).get("urdf_path")
if not urdf_path:
    raise ValueError("Error: 'urdf_path' not found in config/robot_config.yaml.")

model, cmod, vmod = pin.buildModelsFromUrdf(urdf_path)
viz = MeshcatVisualizer(model, cmod, vmod)
viz.initViewer()
viz.loadViewerModel(rootNodeName="pinocchio")

# Create a second set of visual models for the ghost robot
vmod_ghost = copy.deepcopy(vmod)
for geom in vmod_ghost.geometryObjects:
    geom.meshColor = np.array([0.0, 1.0, 0.0, 0.4]) # Green and translucent

viz2 = MeshcatVisualizer(model, cmod, vmod_ghost)
viz2.initViewer(viz.viewer) # Share the same viewer
viz2.loadViewerModel(rootNodeName="expected_robot")

q = pin.neutral(model)
viz.display(q)

q2 = q.copy()
q2[0] += 0.5
viz2.display(q2)

print("Check Meshcat:", viz.viewer.url())
time.sleep(1)
