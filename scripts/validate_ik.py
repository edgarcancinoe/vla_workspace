#!/usr/bin/env python3
"""
SO101 motion patterns — simulation or real robot.

Usage:
    python draw_square.py --sim  --pattern square
    python draw_square.py --sim  --pattern figure8
    python draw_square.py --sim  --pattern cross
    python draw_square.py --real --pattern square
    python draw_square.py --real --pattern figure8  --no-viz
"""

import argparse
import time
import numpy as np
from pathlib import Path
import sys

# Add the workspace root to sys.path to allow imports from robot_control
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

from lerobot.utils.robot_utils import precise_sleep
from robot_control.so101_control import SO101Control

# ── Configuration ──────────────────────────────────────────────────────────────
PORT              = None # Loaded from config
URDF_PATH         = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
JOINT_NAMES = SO101Control.JOINT_NAMES

WRIST_ROLL_OFFSET_DEG = 0.0

# Pattern scale (meters)
SCALE           = 0.05   # half-size for square/cross; amplitude for figure-8
N_WAYPOINTS     = 15     # waypoints per segment
N_FIG8          = 200    # total waypoints for figure-8 curve
DT_S            = 0.05   # 20 Hz
HOME_DURATION_S = 4.0
MAX_STEP        = 4.0


# ══════════════════════════════════════════════════════════════════════════════
# Pattern generators — all return (N, 3) arrays of XYZ waypoints
# centred on origin; caller translates to home_xyz.
# Coordinate convention: x=forward, y=left/right, z=up/down
# ══════════════════════════════════════════════════════════════════════════════

def generate_square(scale=SCALE):
    """4 corners + close, each edge split into N_WAYPOINTS steps."""
    dy, dz = scale, scale
    corners = np.array([
        [0,  -dy, -dz],   # BL
        [0,   dy, -dz],   # BR
        [0,   dy,  dz],   # TR
        [0,  -dy,  dz],   # TL
        [0,  -dy, -dz],   # BL (close)
    ])
    pts = []
    for i in range(len(corners) - 1):
        for k in range(1, N_WAYPOINTS + 1):
            t = k / N_WAYPOINTS
            pts.append((1 - t) * corners[i] + t * corners[i + 1])
    return np.array(pts)


def generate_figure_8(N=N_FIG8, scale=SCALE):
    """
    Lemniscate of Bernoulli traced via the formula from utils.py.
    Sweeps c from 0→1 to build nested loops, then returns the outermost
    (c=1, a=sqrt(2)) loop as the actual motion path.
    Y maps to robot Y, Z maps to robot Z.
    """
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    a = 1.0 * np.sqrt(2)   # c=1 → full figure-8
    y =  a * np.sin(t) / (1 + np.cos(t) ** 2)
    z =  a * (np.sin(t) * np.cos(t)) / (1 + np.cos(t) ** 2)
    # Normalise so max extent = scale
    max_extent = max(np.max(np.abs(y)), np.max(np.abs(z)), 1e-9)
    y = y / max_extent * scale
    z = z / max_extent * scale
    x = np.zeros(N)
    pts = np.stack([x, y, z], axis=1)
    # Close the loop
    pts = np.vstack([pts, pts[:1]])
    return pts


def generate_cross(scale=SCALE):
    """
    Horizontal stroke (−dy → +dy) then return to centre,
    vertical stroke (−dz → +dz) then return to centre.
    Each arm split into N_WAYPOINTS steps.
    """
    dy, dz = scale, scale
    segments = [
        # horizontal arm: left → right
        (np.array([0,  -dy, 0]), np.array([0,  dy, 0])),
        # back to centre
        (np.array([0,   dy, 0]), np.array([0,   0, 0])),
        # vertical arm: bottom → top
        (np.array([0,   0, -dz]), np.array([0,  0,  dz])),
        # back to centre
        (np.array([0,   0,  dz]), np.array([0,  0,   0])),
    ]
    pts = []
    for p_start, p_end in segments:
        for k in range(1, N_WAYPOINTS + 1):
            t = k / N_WAYPOINTS
            pts.append((1 - t) * p_start + t * p_end)
    return np.array(pts)


PATTERN_GENERATORS = {
    "square":  generate_square,
    "figure8": generate_figure_8,
    "cross":   generate_cross,
}

PATTERN_COLORS = {
    "square":  0x33ff33,
    "figure8": 0xff6600,
    "cross":   0x33ccff,
}


# ══════════════════════════════════════════════════════════════════════════════
# Kinematics
# ══════════════════════════════════════════════════════════════════════════════

kinematics = None   # set in main


# ══════════════════════════════════════════════════════════════════════════════
# Robot I/O
# ══════════════════════════════════════════════════════════════════════════════

robot = None

# Note: read_motor_real and send_motor_real are now handled by SO101Control class methods.


# ══════════════════════════════════════════════════════════════════════════════
# Meshcat helpers
# ══════════════════════════════════════════════════════════════════════════════

viz    = None
viewer = None

def meshcat_display(motor_vals):
    if viz: viz.display(kinematics.motor_to_rad(motor_vals))

def add_sphere(name, xyz, color_hex, radius=0.002):
    if not viewer: return
    viewer[name].set_object(g.Sphere(radius), g.MeshLambertMaterial(color=color_hex))
    viewer[name].set_transform(tf.translation_matrix(xyz))

def add_line(name, p1, p2, color_hex=0xffa500, thickness=0.002):
    if not viewer: return
    diff = p2 - p1
    L    = np.linalg.norm(diff)
    if L < 1e-6: return
    z    = np.array([0., 0., 1.])
    axis = np.cross(z, diff / L)
    ang  = np.arccos(np.clip(np.dot(z, diff / L), -1, 1))
    R    = tf.rotation_matrix(ang, axis) if np.linalg.norm(axis) > 1e-6 else np.eye(4)
    viewer[name].set_object(
        g.Cylinder(L, thickness),
        g.MeshLambertMaterial(color=color_hex)
    )
    viewer[name].set_transform(tf.translation_matrix((p1 + p2) / 2) @ R)

def visualize_path(waypoints, color_hex):
    """Draw the target path as connected line segments in Meshcat."""
    for i in range(len(waypoints) - 1):
        add_line(f"pattern/seg_{i}", waypoints[i], waypoints[i + 1], color_hex)
    # Mark start point
    add_sphere("pattern/start", waypoints[0], 0xffffff, radius=0.012)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global robot, viz, viewer, WRIST_ROLL_OFFSET_DEG, kinematics

    parser = argparse.ArgumentParser(description="SO101 motion patterns")
    exec_mode = parser.add_mutually_exclusive_group(required=True)
    exec_mode.add_argument("--sim",  action="store_true", help="Simulation only")
    exec_mode.add_argument("--real", action="store_true", help="Real robot")
    parser.add_argument("--pattern", choices=["square", "figure8", "cross"],
                        default="square", help="Motion pattern (default: square)")
    parser.add_argument("--scale", type=float, default=SCALE,
                        help=f"Pattern scale in metres (default: {SCALE})")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable Meshcat (real mode only)")
    args = parser.parse_args()

    # ── Meshcat ────────────────────────────────────────────────────────────────
    if not (args.real and args.no_viz):
        model, cmod, vmod = pin.buildModelsFromUrdf(
            URDF_PATH, package_dirs=str(Path(URDF_PATH).parent)
        )
        _viz = MeshcatVisualizer(model, cmod, vmod)
        _viz.initViewer(open=True)
        _viz.loadViewerModel()
        viz    = _viz
        viewer = _viz.viewer
        print(f"Meshcat: {viz.viewer.url()}")

    import yaml
    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        robot_cfg = cfg.get("robot", {})
        WRIST_ROLL_OFFSET_DEG = float(robot_cfg.get("wrist_roll_offset", 0.0))
        PORT = robot_cfg.get("port")
        ROBOT_NAME = robot_cfg.get("name", "arm_follower")
        HOME_POSE = robot_cfg.get("home_pose", {})
    else:
        WRIST_ROLL_OFFSET_DEG = 0.0
        PORT = None
        ROBOT_NAME = "arm_follower"
        HOME_POSE = {f"{n}.pos": 0.0 for n in JOINT_NAMES}

    # ── Kinematics ─────────────────────────────────────────────────────────────
    kinematics = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=WRIST_ROLL_OFFSET_DEG, home_pose=HOME_POSE)

    HOME_MOTOR = np.zeros(6)

    # ── Real robot ─────────────────────────────────────────────────────────────
    if args.real:
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

        if not PORT:
            raise ValueError("Error: 'port' not found in config/robot_config.yaml. Cannot connect to robot.")

        print(f"Robot Name: {ROBOT_NAME}")
        print(f"Wrist roll offset: {WRIST_ROLL_OFFSET_DEG}°")
        
        calibration_dir = Path(__file__).parent.parent / ".cache" / "calibration"
        robot = SO101Follower(SO101FollowerConfig(id=ROBOT_NAME, port=PORT, calibration_dir=calibration_dir))
        robot.connect()
        print("Robot connected.")

    # ── Home ───────────────────────────────────────────────────────────────────
    if args.real:
        kinematics.reset_to_home(robot, duration_s=HOME_DURATION_S, fps=1.0/DT_S)
        time.sleep(1.0)
        current = kinematics.read_motor_real(robot)
    else:
        # In simulation mode, we just start at the HOME_POSE degree targets converted to motor units
        goal_deg = np.array([float(HOME_POSE.get(f"{n}.pos", 0.0)) for n in JOINT_NAMES])
        current = kinematics.deg_to_motor(goal_deg)
        meshcat_display(current)
        time.sleep(1.0)

    home_pose = kinematics.fk(current)
    home_xyz  = home_pose[:3, 3].copy()
    ref_pose  = home_pose.copy()
    cx, cy, cz = home_xyz
    print(f"Home FK XYZ: {home_xyz.round(4)}")

    # ── Build pattern ──────────────────────────────────────────────────────────
    gen      = PATTERN_GENERATORS[args.pattern]
    # Pass scale; figure8 and square both accept it
    rel_pts  = gen(scale=args.scale) if args.pattern != "figure8" else gen(scale=args.scale)
    # Translate relative offsets to world coords centred on home
    waypoints = rel_pts + np.array([cx, cy, cz])

    print(f"\nPattern: {args.pattern.upper()}  |  {len(waypoints)} waypoints  |  scale={args.scale}m")

    # ── IK sanity check on first & last waypoint ───────────────────────────────
    for label, pt in [("start", waypoints[0]), ("end", waypoints[-1])]:
        m = kinematics.ik_motor(pt, ref_pose, HOME_MOTOR)
        if m is None:
            print(f"  WARNING: IK failed for {label} waypoint {pt.round(4)}")
        else:
            err = np.linalg.norm(pt - kinematics.fk_xyz(m))
            print(f"  IK {label}: err={err*100:.2f}cm")

    # ── Visualise target path ──────────────────────────────────────────────────
    visualize_path(waypoints, PATTERN_COLORS[args.pattern])
    add_sphere("home_marker", home_xyz, 0xffffff, radius=0.015)
    time.sleep(1.0)

    # ── Move to pattern start ──────────────────────────────────────────────────
    print(f"\nMoving to {args.pattern} start point...")
    current = kinematics.move_to_xyz(robot, waypoints[0], ref_pose, current, duration_s=3.0, viz=viz, fps=1.0/DT_S, max_step=MAX_STEP)
    time.sleep(0.5)

    # ── Execute pattern (loop) ─────────────────────────────────────────────────
    mode_str = "REAL ROBOT" if args.real else "SIMULATION"
    print(f"\n[{mode_str}] Running {args.pattern}. Ctrl+C to stop.\n")

    lap = 0
    try:
        while True:
            lap += 1
            print(f"--- Lap {lap} ---")
            current = kinematics.execute_cartesian_trajectory(robot, waypoints, ref_pose, current, viz=viz, fps=1.0/DT_S, max_step=MAX_STEP)
            actual_xyz = kinematics.fk_xyz(current)
            err = np.linalg.norm(waypoints[-1] - actual_xyz)
            print(f"  Lap {lap} done. Final EE={actual_xyz.round(4)}  err={err*100:.2f}cm")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if robot:
            print("Returning to home...")
            kinematics.reset_to_home(robot, duration_s=HOME_DURATION_S, fps=1.0/DT_S)
            time.sleep(0.5)
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()