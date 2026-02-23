import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import copy

urdf_path = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

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
