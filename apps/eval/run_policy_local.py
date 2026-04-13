#!/usr/bin/env python3
import sys
import yaml
import numpy as np
from functools import partial
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
src_root = ROOT_DIR / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
lerobot_src = ROOT_DIR.parent / "repos" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from thesis_vla.common.paths import DATASETS_OUTPUT_DIR, ROBOT_CONFIG_PATH

from lerobot.utils.utils import log_say as lerobot_log_say
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.policies.xvla.action_contract import get_so101_slice_spec
from lerobot.policies.xvla.utils import mat_to_rotate6d
from lerobot.policies.factory import make_pre_post_processors
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig

from thesis_vla.vision import camera_calibration
from thesis_vla.inference.xvla_runtime import (
    make_xvla_runtime_processors,
    resolve_xvla_rename_map,
    sync_xvla_policy_config,
)
from thesis_vla.robot.so101_control import SO101Control

# DEBUG UTILITIES
# ============================================================================
class DBG:
    """Lightweight ANSI-color debug helpers. One call = one coloured line."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"

    # Palette
    _C = {
        "cyan":    "\033[96m",
        "green":   "\033[92m",
        "yellow":  "\033[93m",
        "red":     "\033[91m",
        "magenta": "\033[95m",
        "blue":    "\033[94m",
        "white":   "\033[97m",
        "orange":  "\033[38;5;214m",
        "teal":    "\033[38;5;86m",
        "pink":    "\033[38;5;213m",
    }

    @classmethod
    def _fmt(cls, color: str, tag: str, msg: str) -> str:
        c = cls._C.get(color, "")
        return f"{cls.BOLD}{c}[{tag}]{cls.RESET} {msg}"

    @classmethod
    def obs(cls, msg: str):    print(cls._fmt("cyan",    "OBS",     msg))
    @classmethod
    def pre(cls, msg: str):    print(cls._fmt("blue",    "PRE",     msg))
    @classmethod
    def infer(cls, msg: str):  print(cls._fmt("magenta", "INFER",   msg))
    @classmethod
    def post(cls, msg: str):   print(cls._fmt("teal",    "POST",    msg))
    @classmethod
    def act(cls, msg: str):    print(cls._fmt("green",   "ACT",     msg))
    @classmethod
    def ik(cls, msg: str):     print(cls._fmt("orange",  "IK",      msg))
    @classmethod
    def grip(cls, msg: str):   print(cls._fmt("pink",    "GRIPPER", msg))
    @classmethod
    def send(cls, msg: str):   print(cls._fmt("yellow",  "SEND",    msg))
    @classmethod
    def warn(cls, msg: str):   print(cls._fmt("red",     "WARN",    msg))
    @classmethod
    def info(cls, msg: str):   print(cls._fmt("white",   "INFO",    msg))

    @classmethod
    def tensor(cls, color: str, tag: str, name: str, t):
        """Print shape, dtype, min, max and first few values of a tensor/array."""
        import numpy as _np
        try:
            import torch as _torch
            if isinstance(t, _torch.Tensor):
                arr = t.detach().cpu().float().numpy()
            else:
                arr = _np.asarray(t, dtype=float)
            flat = arr.flatten()
            preview = " ".join(f"{v:+.4f}" for v in flat[:8])
            if len(flat) > 8:
                preview += " ..."
            print(cls._fmt(color, tag,
                f"{name}: shape={list(arr.shape)}  dtype={arr.dtype}  "
                f"min={flat.min():+.4f}  max={flat.max():+.4f}  | {preview}"
            ))
        except Exception as _e:
            print(cls._fmt(color, tag, f"{name}: <could not inspect: {_e}>"))

    @classmethod
    def divider(cls, color: str = "white", label: str = ""):
        c = cls._C.get(color, "")
        line = "─" * 64
        if label:
            print(f"{c}{cls.BOLD}┌{line}\n│  {label}\n└{line}{cls.RESET}")
        else:
            print(f"{c}{line}{cls.RESET}")

# CONFIGURATION UTIL 
# ============================================================================
def validate_configuration():
    if not URDF_PATH:
        raise ValueError("'urdf_path' not found in config/robot_config.yaml.")
    
    assert not (START_FROM_SCRATCH and RESUME_DATASET), "Cannot start from scratch and resume dataset at the same time."
    
    if not CALIBRATION_DIR:
        raise ValueError("'calibration_dir' not found in config/robot_config.yaml.")

    assert POLICY_TYPE in ["smolvla", "xvla"], f"Unknown policy type: {POLICY_TYPE}"
    print(f"Loading policy from {POLICY_PATH}")
    print(f"Policy type: {POLICY_TYPE}, Device: {DEVICE}")
    
    if START_FROM_SCRATCH or OVERWRITE_DATASET:
        import shutil
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

# CONFIGURATION
# ============================================================================

# --- Robot Setup ---
config_path = ROBOT_CONFIG_PATH
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

WRIST_ROLL_OFFSET = config_data.get("robot", {}).get("wrist_roll_offset", 0.0)
WRIST_ROLL_OFFSET = 0
URDF_PATH         = config_data.get("robot", {}).get("urdf_path")
CALIBRATION_DIR = config_data["robot"].get("calibration_dir")
FOLLOWER_PORT = config_data["robot"]["port"]
ROBOT_NAME = config_data["robot"].get("name", "arm_follower")
HOME_POSE = config_data["robot"].get("home_pose", {})
CAMERA_CONFIG_MAP = config_data.get("cameras", {})

# --- Policy ---
# POLICY_PATH = "edgarcancinoe/xvla-base_soarm101_pickplace_10d_7p5hz_resampled_so101_ee6d_a-m_s-m_v1"
POLICY_PATH = "edgarcancinoe/orange196_pickplace_multicolor_v1_7p5hz_so101_ee6d_am_sm_transformer_soft-2dce6242-step-8000"


POLICY_TYPE = "xvla" # "xvla" | "smolvla"
DEVICE      = "mps"  # "cuda" | "mps" | "cpu"

TASK_DESCRIPTION = "Pick up orange cube and place inside white box."
POLICY_PIPELINE = None

# --- Voice ---
USE_VOICE          = True

# --- Action Chunking (shared across policy types) ---
# NOTE: For XVLA, these are applied BEFORE model construction so they
#       actually affect the model's generate_actions() chunk size.
#       Set to None to use whatever the pretrained checkpoint has.
#
# IMPORTANT: XVLA outputs near-constant chunks (all 30 waypoints ≈ same
# target). n_action_steps controls how many steps are consumed before
# re-running inference. Lower = more responsive but higher compute cost.
#   - 30: one inference per second (jerky, ~5mm jumps)
#   -  5: six inferences per second (smoother, more responsive)
#   -  1: inference every step (smoothest but slowest, only for testing)
CHUNK_SIZE         = None  # None = use pretrained default (30 for this checkpoint)
N_ACTION_STEPS     = 24  # Number of control steps to execute before running the next inference chunk. Only for XVLA.

# --- SmolVLA-specific ---
MAX_ACTION_TOKENS  = None
POLICY_DELAY       = 0.2

# --- XVLA-specific ---
NUM_XVLA_OBS_STEPS         = 1
BINARY_GRIPPER_INFERENCE   = False

# --- Evaluation & Dataset ---
NUM_EPISODES               = 5
CONTROL_FPS                = 7.5
CAMERA_FPS                 = 30
EPISODE_TIME_SEC           = 600
HF_USER                    = "edgarcancinoe"
EVAL_DATASET_NAME          = "eval_" + POLICY_PATH.split("/")[-1] 
DATA_DIR                   = DATASETS_OUTPUT_DIR / EVAL_DATASET_NAME
START_FROM_SCRATCH         = True
RESUME_DATASET             = False
OVERWRITE_DATASET          = True  # Set True to delete and recreate the dataset on every run

# --- Robot & Setup
ACTIVE_XVLA_RENAME_MAP = {}
STARTING_POSITION_DURATION_S = 5
HOME_DURATION_S = STARTING_POSITION_DURATION_S
HOME_FPS = CONTROL_FPS

# --- Meshcat Visualization ---
USE_MESHCAT_VIZ                = True   # Set False to disable

# --- Rerun Visualization ---
USE_RERUN                      = True  # Set True to enable Rerun telemetry

# --- Dry Run (visualization only, no robot commands) ---
# Set True to run policy inference + Meshcat visualization WITHOUT sending any
# commands to the real robot.  Useful for inspecting predicted trajectories
# before committing to execution.  Robot is still connected for reading state.
DRY_RUN                        = False

# ─── State feature names (must match training dataset schema) ────────────────
EEF_STATE_NAMES = list(get_so101_slice_spec("so101_ee6d").names)

# Build config objects
RECTIFY_MAP = {name: info.get("rectify", False) for name, info in CAMERA_CONFIG_MAP.items()}
CAMERAS = {
    name: OpenCVCameraConfig(
        index_or_path=info["id"], 
        width=info.get("width", 640), 
        height=info.get("height", 480), 
        fps=CAMERA_FPS
    ) for name, info in CAMERA_CONFIG_MAP.items()
}

validate_configuration()

# UTIL
# ============================================================================

# Utility to log and speak, respecting the USE_VOICE toggle
log_say = partial(lerobot_log_say, play_sounds=USE_VOICE)

def init_meshcat():
    """Initialize SO101Meshcat for robot-pose + trajectory visualization.
    Returns None (gracefully) if the package is unavailable or URDF is missing."""
    try:
        from thesis_vla.sim.so101_meshcat import SO101Meshcat
        viz = SO101Meshcat(urdf_path=URDF_PATH)
        print("[x] Meshcat visualizer started.")
        return viz
    except Exception as e:
        print(f"[!] Meshcat init failed, continuing without visualization: {e}")
        return None

# > Custom get_observation():
#   1. Rectify images based on config
#   2. Inject EEF State if required
def set_custom_get_observation(robot, so101: SO101Control = None, include_eef_state: bool = False, dry_run: bool = False):
    
    base_obs_func = robot.get_observation
    
    if dry_run:
        # Initial virtual state from physical robot (one-time read)
        # or use home pose if preferred. Let's do one-time read.
        init_obs = base_obs_func()
        robot._virtual_motor = np.array([float(init_obs.get(f"{n}.pos", 0.0)) for n in so101.JOINT_NAMES])

    _obs_step = [0]  # closure counter
    def patched_get_observation():
        _obs_step[0] += 1
        if dry_run:
            if not hasattr(robot, "_virtual_motor") or robot._virtual_motor is None:
                robot._virtual_motor = np.zeros(len(so101.JOINT_NAMES))
            observation = {f"{n}.pos": float(robot._virtual_motor[i]) for i, n in enumerate(so101.JOINT_NAMES)}
            
            for cam_key, cam in robot.cameras.items():
                try:
                    observation[cam_key] = cam.async_read()
                except Exception as e:
                    raise RuntimeError(
                        f"Dry-run camera read failed for '{cam_key}'"
                    ) from e
        else:
            observation = base_obs_func()
        
        if _obs_step[0] % 30 == 1:
            # Coloured logging
            print(f"\033[94mDEBUG: Raw robot observation keys: {list(observation.keys())}\033[0m")
            if "wrist" in observation:
                print("\033[91m!!! DEBUG: 'wrist' key already in raw observation!\033[0m")
        
        # Rectify images based on config
        for cam_name, should_rectify in RECTIFY_MAP.items():
            if should_rectify and cam_name in observation:
                observation[cam_name] = camera_calibration.rectify_image(
                    observation[cam_name], cam_name
                )

        # Inject EEF State if requested
        if include_eef_state:
            motor_vals = np.array([observation.get(f"{joint}.pos", 0.0) for joint in so101.JOINT_NAMES])
            T = so101.fk(motor_vals)
            r6d = mat_to_rotate6d(T[:3, :3])
            eef = np.concatenate([T[:3, 3], r6d, [motor_vals[-1]]], dtype=np.float32)
            for i, name in enumerate(EEF_STATE_NAMES):
                observation[f"{name}.pos"] = float(eef[i])

        expected_camera_keys = {
            src.removeprefix("observation.images.")
            for src in ACTIVE_XVLA_RENAME_MAP
        }
        for cam_name in expected_camera_keys:
            if cam_name not in observation:
                raise KeyError(
                    f"Expected camera '{cam_name}' not found in observation. "
                    f"Available keys: {list(observation.keys())}"
                )

        # ── DEBUG: log observation keys & state dimensions every 30 steps ──
        if _obs_step[0] % 30 == 1:
            DBG.divider("cyan", f"OBS  step={_obs_step[0]}")
            scalar_keys = {k: v for k, v in observation.items() if not hasattr(v, 'shape')}
            tensor_keys = {k: v for k, v in observation.items() if hasattr(v, 'shape')}
            DBG.obs(f"Total obs keys: {len(observation)}  |  scalars: {len(scalar_keys)}  tensors(images): {len(tensor_keys)}")
            if scalar_keys:
                vals_str = "  ".join(f"{k.replace('.pos','')}={float(v):+.3f}" for k, v in scalar_keys.items())
                DBG.obs(f"State: [{len(scalar_keys)}D] {vals_str}")
            for k, v in tensor_keys.items():
                sh = list(v.shape) if hasattr(v, 'shape') else '?'
                DBG.obs(f"Image '{k}': shape={sh}")

        return observation
    print("==============================================================================================================")
    print(f"\033[94m[x] Robot observation patched for dynamic rectification based on config: {RECTIFY_MAP}\033[0m")
    print(f"\033[94m[x] Robot observation patched for EEF State: {EEF_STATE_NAMES}\033[0m") if include_eef_state else None
    if ACTIVE_XVLA_RENAME_MAP:
        print(f"\033[94m[x] Robot observation keeps training camera keys for rename_map: {ACTIVE_XVLA_RENAME_MAP}\033[0m")
    print("==============================================================================================================")
    robot.get_observation = patched_get_observation

# > Custom send_action():
#   1. Convert EEF State to motor space if required
#   2. Update Meshcat robot pose display (if viz provided)
def set_custom_send_action(robot, so101: SO101Control = None, viz=None, dry_run=False):
    base_action_func = robot.send_action
    gr_idx = so101.JOINT_NAMES.index("gripper")
    _step = [0]
    _last_mode = [None]

    if dry_run:
        print("[!] DRY RUN enabled — robot will NOT receive any commands.")

    def _viz_update(motor_vals):
        """Update robot pose AND draw EEF axes at the gripper frame."""
        if not viz:
            return
        viz.display(so101.motor_to_rad(motor_vals))
        T = so101.fk(motor_vals)
        viz.add_axes("eef_axes", T[:3, 3], T[:3, :3], length=0.04)

    def patched_send_action(action, **kwargs):
        _step[0] += 1

        # Reliable mode detection: make_robot_action in record_loop maps tensor indices
        # to dataset.features["action"]["names"].  With our EEF action_features override,
        # EEF policies produce keys like "x.pos", "y.pos", etc.; motor policies produce
        # "shoulder_pan.pos", "shoulder_lift.pos", etc.
        is_motor = "shoulder_pan.pos" in action   # False for EEF, True for motor
        mode = "MOTOR" if is_motor else "EEF"

        # Read current motor state once (used for IK seed, fallback, dry-run hold, and logging)
        current_motor = so101.read_motor_real(robot)
        current_motor_dict = {f"{n}.pos": float(current_motor[i]) for i, n in enumerate(so101.JOINT_NAMES)}

        # ── DEBUG: raw action dict received by send_action ──────────────────────
        log_every = (_step[0] % 30 == 1)
        if log_every:
            DBG.divider("green", f"SEND_ACTION  step={_step[0]}  mode={mode}")
            DBG.act(f"Action dict keys ({len(action)}): {list(action.keys())}")
            for k, v in action.items():
                try:
                    DBG.act(f"  {k:25s} = {float(v):+.5f}")
                except Exception:
                    sh = list(v.shape) if hasattr(v, 'shape') else type(v).__name__
                    DBG.act(f"  {k:25s} = <tensor shape={sh}>")

        # Log on mode change or every 30 steps (~1 s at 30 fps)
        if mode != _last_mode[0] or log_every:
            _last_mode[0] = mode
            if is_motor:
                vals = "  ".join(f"{n}={action.get(f'{n}.pos', 0.0):+.1f}" for n in so101.JOINT_NAMES)
                print(f"[send_action | step {_step[0]:4d}] MODE=MOTOR  | {vals}")
            else:
                cur_xyz = so101.fk_xyz(current_motor)
                tgt_xyz = np.array([action.get('x.pos', 0.0), action.get('y.pos', 0.0), action.get('z.pos', 0.0)])
                delta   = tgt_xyz - cur_xyz
                grip    = action.get('gripper.pos', 0.0)
                DBG.act(
                    f"step {_step[0]:4d} | MODE=EEF"
                    f" | cur=({cur_xyz[0]:+.3f},{cur_xyz[1]:+.3f},{cur_xyz[2]:+.3f})"
                    f"  tgt=({tgt_xyz[0]:+.3f},{tgt_xyz[1]:+.3f},{tgt_xyz[2]:+.3f})"
                    f"  Δ=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})"
                    f"  grip_raw={grip:+.4f}"
                )

        # Case: actions are in motor space
        if is_motor:
            motor_vals = np.array([float(action.get(f"{n}.pos", 0.0)) for n in so101.JOINT_NAMES])
            if log_every:
                DBG.send(f"Sending MOTOR command [{len(motor_vals)}D]: " +
                         "  ".join(f"{n}={motor_vals[i]:+.1f}°" for i, n in enumerate(so101.JOINT_NAMES)))
            if dry_run:
                _viz_update(motor_vals)
                robot._virtual_motor = motor_vals
                return action
            _viz_update(motor_vals)
            return base_action_func(action, **kwargs)

        # Case: actions are in EEF space (xyz + 6D Orientation + Gripper)
        # Keys produced by make_robot_action from EEF action_features: "x.pos", "y.pos", ...
        target_xyz  = np.array([action["x.pos"], action["y.pos"], action["z.pos"]], dtype=np.float64)
        r6d = np.array([action[f"rot6d_{i}.pos"] for i in range(6)], dtype=np.float64)
        gripper_val = float(action["gripper.pos"])

        if log_every:
            DBG.ik(f"EEF action input [{10}D]:")
            DBG.ik(f"  target_xyz = ({target_xyz[0]:+.4f},{target_xyz[1]:+.4f},{target_xyz[2]:+.4f}) m")
            DBG.ik("  rot6d      = [" + ", ".join(f"{v:+.4f}" for v in r6d) + "]")
            DBG.ik(f"  gripper_sigmoid = {gripper_val:+.4f}")
            DBG.ik(f"  IK seed (current_motor) [{len(current_motor)}D]: " +
                   "  ".join(f"{n}={current_motor[i]:+.1f}°" for i, n in enumerate(so101.JOINT_NAMES)))

        # Guard: degenerate rot6d (all-zero first column) would cause division-by-zero
        # in rot6d_to_mat's Gram-Schmidt step → NaN propagates through IK → garbage pose.
        if np.linalg.norm(r6d[:3]) < 1e-6:
            DBG.warn(f"step {_step[0]}: Degenerate rot6d (near-zero norm), holding current pose")
            _viz_update(current_motor)
            if dry_run:
                return current_motor_dict
            return base_action_func(current_motor_dict, **kwargs)

        # IK: solve for full 6D pose (position + orientation)
        target_motor = so101.ik_motor_6d(target_xyz, r6d, current_motor)

        if target_motor is None:
            DBG.warn(f"step {_step[0]}: IK failed, holding current pose")
            _viz_update(current_motor)
            if dry_run:
                return current_motor_dict
            return base_action_func(current_motor_dict, **kwargs)

        if log_every:
            DBG.ik(f"  IK solution [{len(target_motor)}D]: " +
                   "  ".join(f"{n}={target_motor[i]:+.1f}°" for i, n in enumerate(so101.JOINT_NAMES)))

        # Gripper: Use the value directly from action space postprocessor.
        # The action space already applied sigmoid and scaling, so gripper_val
        # is in the correct motor degree range [0, gripper_max].
        target_motor[gr_idx] = gripper_val

        if log_every:
            DBG.grip(f"gripper: {gripper_val:.4f}°")
        motor_dict = {f"{n}.pos": float(target_motor[i]) for i, n in enumerate(so101.JOINT_NAMES)}

        if log_every:
            DBG.send(f"Dispatching to robot [{len(motor_dict)}D motor dict]:")
            for k, v in motor_dict.items():
                DBG.send(f"  {k:30s} = {v:+.2f}°")

        if dry_run:
            _viz_update(target_motor)
            robot._virtual_motor = target_motor
            return motor_dict

        _viz_update(target_motor)
        return base_action_func(motor_dict, **kwargs)

    robot.send_action = patched_send_action

# > Custom select_action():
#   1. Inject delay when action queue is empty. Used to reduce SOARM101 shaking between actions.
def set_custom_select_action(policy, delay_s: float):
    if delay_s <= 0:
        return
    original_select_action = policy.select_action
    def patched_select_action(batch, **kwargs):
        # If queue is empty, we are about to run inference -> sleep
        # NOTE: This sleep is blocking and happens inside the record_loop. Delay counts towards the total EPISODE_TIME_SEC.
        if hasattr(policy, "action_queue") and len(policy.action_queue) == 0:
            import time
            print(f"Action queue empty. Delaying inference by {delay_s:.2f}s...")
            time.sleep(delay_s)
        return original_select_action(batch, **kwargs)
    policy.select_action = patched_select_action
    print(f"Policy patched with {delay_s}s delay before inference chunks.")

# > Trajectory visualization via Meshcat:
#   Draws the full predicted EEF path every time a new inference chunk fires.
#   Works for both xvla (EEF action space) and smolvla (motor action space).
#   Must be called BEFORE set_custom_select_action so it is the innermost wrapper.
def set_trajectory_visualization(policy, so101: SO101Control, viz, postprocessor, action_names: list[str]):
    """Patches policy.select_action to visualise the predicted chunk in Meshcat."""
    if viz is None:
        return

    original_select_action = policy.select_action

    def _ensure_dict(action) -> dict | None:
        """Convert postprocessed action (dict or tensor) → dictionary of named features."""
        if isinstance(action, dict):
            return action
        if hasattr(action, "detach"): # Tensor
            if action.ndim == 2: action = action[0]
            if len(action_names) == 0: return None
            return {name: float(action[i]) for i, name in enumerate(action_names) if i < len(action)}
        return None

    def _action_dict_to_xyz(action_dict: dict) -> np.ndarray | None:
        """Extract EEF xyz (metres) from an action dictionary."""
        # EEF action space — keys are "x.pos"/"y.pos"/"z.pos" (from EEF action_features)
        x = action_dict.get("x.pos", action_dict.get("x"))
        y = action_dict.get("y.pos", action_dict.get("y"))
        z = action_dict.get("z.pos", action_dict.get("z"))
        if x is not None and y is not None and z is not None:
            return np.array([float(x), float(y), float(z)], dtype=np.float64)
        
        # Motor action space — run FK
        try:
            motor = np.array([float(action_dict.get(f"{n}.pos", 0.0)) for n in so101.JOINT_NAMES])
            return so101.fk(motor)[:3, 3]
        except Exception:
            return None

    def _decode_tensor(t) -> dict | np.ndarray | None:
        """Apply postprocessor to a single queue tensor; try 1-D then batched form."""
        try:
            return postprocessor(t)
        except Exception:
            pass
        try:
            return postprocessor(t.unsqueeze(0))
        except Exception:
            return None

    def _policy_queue_len(p) -> int:
        """Returns remaining actions in the policy queue (handles smolvla and xvla)."""
        if hasattr(p, "action_queue"):          # smolvla: direct attribute
            return len(p.action_queue)
        if hasattr(p, "_queues") and "action" in p._queues:  # xvla: nested dict of deques
            return len(p._queues["action"])
        return 0

    def _policy_queue_list(p) -> list:
        """Returns a snapshot of the policy queue as a list."""
        if hasattr(p, "action_queue"):
            return list(p.action_queue)
        if hasattr(p, "_queues") and "action" in p._queues:
            return list(p._queues["action"])
        return []

    def patched_select_action(batch, **kwargs):
        # New inference fires when the queue is empty (or doesn't exist).
        # smolvla exposes the queue as policy.action_queue (direct attribute).
        # xvla stores it at policy._queues["action"] (a deque inside a dict).
        is_new_inference = _policy_queue_len(policy) == 0

        # ── DEBUG: log batch / observation coming into select_action ──────────
        if is_new_inference:
            DBG.divider("magenta", "POLICY INFERENCE  (queue empty → new forward pass)")
            if isinstance(batch, dict):
                DBG.infer(f"Batch keys ({len(batch)}): {list(batch.keys())}")
                for bk, bv in batch.items():
                    DBG.tensor("magenta", "INFER", f"  batch[{bk!r}]", bv)
            else:
                DBG.tensor("magenta", "INFER", "batch", batch)

        result = original_select_action(batch, **kwargs)

        # ── DEBUG: raw result tensor from policy ──────────────────────────────
        if is_new_inference:
            DBG.tensor("teal", "POST", "select_action result (unnorm/remapped)", result)
            if hasattr(policy.model, 'action_space'):
                asp = policy.model.action_space
                g_max = getattr(asp, 'gripper_max', 'N/A')
                g_idx = getattr(asp, 'gripper_idx', 'N/A')
                DBG.info(f"ActionSpace: type={type(asp).__name__}  gripper_max={g_max}  gripper_idx={g_idx}")

        if is_new_inference:
            try:
                # Full chunk = returned item + what remains in the queue (not yet consumed)
                remaining = _policy_queue_list(policy)
                chunk = [result] + remaining
                
                if POLICY_TYPE == "xvla":
                    DBG.divider("magenta", f"XVLA Inference  chunk={len(chunk)} waypoints")
                    DBG.infer("Raw model output (normalized space):")
                    if hasattr(result, 'shape'):
                        DBG.infer(f"  shape={list(result.shape)}, dtype={result.dtype}")
                        flat = result.squeeze()
                        for i, name in enumerate(action_names):
                            if i < len(flat):
                                DBG.infer(f"    [{i:2d}] {name:18s} = {float(flat[i]):+.6f}  (normalized)")
                    decoded_first = _decode_tensor(result)
                    if decoded_first is not None and hasattr(decoded_first, 'shape'):
                        DBG.post("  After postprocessor (unnormalized/real-world):")
                        flat_d = decoded_first.squeeze()
                        for i, name in enumerate(action_names):
                            if i < len(flat_d):
                                DBG.post(f"    [{i:2d}] {name:18s} = {float(flat_d[i]):+.6f}")
                    DBG.infer(f"  Chunk size: {len(chunk)} waypoints  |  action_names: {action_names}")

                print(f"[viz] New inference chunk: {len(chunk)} waypoints")
                xyz_list  = []
                rot_list  = []   # 3×3 rotation matrix per waypoint, or None
                for t in chunk:
                    decoded = _decode_tensor(t)
                    if decoded is None:
                        continue
                    
                    action_dict = _ensure_dict(decoded)
                    if action_dict is None:
                        continue

                    xyz = _action_dict_to_xyz(action_dict)
                    if xyz is None:
                        continue
                    xyz_list.append(xyz)

                    # Extract rot6d → rotation matrix when available
                    R = None
                    r6d_vals = [
                        action_dict.get(f"rot6d_{i}.pos", action_dict.get(f"rot6d_{i}"))
                        for i in range(6)
                    ]
                    if all(v is not None for v in r6d_vals):
                        r6d = np.array([float(v) for v in r6d_vals], dtype=np.float64)
                        if np.linalg.norm(r6d[:3]) > 1e-6:
                            R = so101.rot6d_to_mat(r6d)
                    rot_list.append(R)

                if xyz_list:
                    xyz_arr = np.array(xyz_list)
                    total_disp = np.linalg.norm(xyz_arr[-1] - xyz_arr[0])
                    max_step = max(np.linalg.norm(xyz_arr[i+1] - xyz_arr[i]) for i in range(len(xyz_arr)-1)) if len(xyz_arr) > 1 else 0
                    print(
                        f"[viz] Predicted trajectory: {len(xyz_list)}/{len(chunk)} valid | "
                        f"x=[{xyz_arr[:,0].min():.4f},{xyz_arr[:,0].max():.4f}]  "
                        f"y=[{xyz_arr[:,1].min():.4f},{xyz_arr[:,1].max():.4f}]  "
                        f"z=[{xyz_arr[:,2].min():.4f},{xyz_arr[:,2].max():.4f}]"
                    )
                    print(
                        f"[viz] Total displacement: {total_disp*1000:.1f}mm | "
                        f"Max single step: {max_step*1000:.2f}mm | "
                        f"Start: ({xyz_arr[0,0]:.4f},{xyz_arr[0,1]:.4f},{xyz_arr[0,2]:.4f}) "
                        f"End: ({xyz_arr[-1,0]:.4f},{xyz_arr[-1,1]:.4f},{xyz_arr[-1,2]:.4f})"
                    )
                    # Replace the previous trajectory entirely
                    viz.viewer["infer_traj"].delete()
                    for i, (xyz, R) in enumerate(zip(xyz_list, rot_list)):
                        viz.add_sphere(f"infer_traj/pt_{i}", xyz, 0x00ffcc, radius=0.004)
                        if R is not None:
                            viz.add_axes(f"infer_traj/axes_{i}", xyz, R, length=0.015)
                    for i in range(len(xyz_list) - 1):
                        viz.add_line(
                            f"infer_traj/seg_{i}",
                            xyz_list[i], xyz_list[i + 1],
                            color_hex=0x00ffcc, thickness=0.002,
                        )
            except Exception as e:
                print(f"[viz] Trajectory visualization error: {e}")

        return result

    policy.select_action = patched_select_action
    print("[x] Policy patched for predicted trajectory visualization in Meshcat.")

def get_policy_processors(policy, dataset, pipeline_key: str | None = None):
    """
    Utility to choose pre- and post-processors by a key.
    - 'xvla_default': Use our custom XVLA replication.
    - None (or any other): Use LeRobot's default factory.

    Returns: (preprocessor, postprocessor)
    """
    DBG.divider("blue", f"get_policy_processors  pipeline_key={pipeline_key!r}")
    stats = getattr(getattr(dataset, 'meta', None), 'stats', None)
    if stats:
        DBG.pre(f"Dataset stats keys: {list(stats.keys())}")
        for sk, sv in stats.items():
            if isinstance(sv, dict):
                for stat_name, stat_val in sv.items():
                    DBG.tensor("blue", "PRE", f"  stats[{sk!r}][{stat_name!r}]", stat_val)
    else:
        DBG.pre("Dataset stats: None (new dataset — no stats yet)")

    if pipeline_key == "xvla_default":
        print(f"[x] Using XVLA runtime pipeline: {pipeline_key}")
        preprocessor, postprocessor = make_xvla_runtime_processors(
            policy=policy,
            pretrained_path=POLICY_PATH,
            device=DEVICE,
            rename_map=ACTIVE_XVLA_RENAME_MAP,
        )
        return preprocessor, postprocessor

    # Default behavior: load processors from pretrained checkpoint
    print(f"[x] Using default/factory processors (Device: {DEVICE})")
    if policy.config.type == "xvla":
        preprocessor, postprocessor = make_xvla_runtime_processors(
            policy=policy,
            pretrained_path=POLICY_PATH,
            device=DEVICE,
            rename_map=ACTIVE_XVLA_RENAME_MAP,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=POLICY_PATH,
            dataset_stats=stats,
            preprocessor_overrides={"device_processor": {"device": DEVICE}},
        )
    DBG.pre(f"Preprocessor type : {type(preprocessor).__name__}")
    DBG.post(f"Postprocessor type: {type(postprocessor).__name__}")

    return preprocessor, postprocessor

def get_policy(policy_type: str, path: str, device: str):
    """
    Utility to load and configure a policy based on its type.
    Uses LeRobot's factory to resolve policy classes elegantly.

    IMPORTANT: For XVLA, config overrides (chunk_size, action_mode, etc.)
    must be applied BEFORE from_pretrained() because XVLAModel.__init__()
    copies config values at construction time. Overriding policy.config
    after construction has NO effect on model internals like
    model.chunk_size or model.action_space.
    """
    from lerobot.policies.factory import get_policy_class
    from lerobot.configs.policies import PreTrainedConfig

    print(f"[x] Loading policy {path} ({policy_type}) on {device}")
    policy_cls = get_policy_class(policy_type)

    INCLUDE_EEF_STATE = False

    if policy_type == "xvla":
        # ── Step 1: Load config first, keep the checkpoint's XVLA contract, THEN construct ──
        config = PreTrainedConfig.from_pretrained(path)
        config.device = device  # Ensure model lands on the right device

        # Apply chunk/action-step overrides (None = keep pretrained default)
        if CHUNK_SIZE is not None:
            config.chunk_size = CHUNK_SIZE
        if N_ACTION_STEPS is not None:
            config.n_action_steps = N_ACTION_STEPS

        config.n_obs_steps = NUM_XVLA_OBS_STEPS
        config.binary_gripper_inference = BINARY_GRIPPER_INFERENCE

        # ── Step 2: Construct model with the correct config ──
        policy = policy_cls.from_pretrained(path, config=config, device=device)
        INCLUDE_EEF_STATE = (getattr(policy.config, "action_mode", None) == "so101_ee6d")

        # ── Step 3: Post-creation verification ──
        action_space_name = type(policy.model.action_space).__name__
        norm_mode = config.normalization_mapping.get("ACTION", "?")
        print("[x] XVLA loaded successfully:")
        print(f"    model.chunk_size    = {policy.model.chunk_size}")
        print(f"    config.chunk_size   = {policy.config.chunk_size}")
        print(f"    n_action_steps      = {policy.config.n_action_steps}")
        print(f"    action_mode         = {policy.config.action_mode}")
        print(f"    num_image_views     = {getattr(policy.config, 'num_image_views', 'N/A')}")
        print(f"    empty_cameras       = {getattr(policy.config, 'empty_cameras', 'N/A')}")
        print(f"    action_space        = {action_space_name} (dim={policy.model.dim_action})")
        print(f"    normalization(ACTION)= {norm_mode}")
        print(f"    num_denoising_steps = {policy.config.num_denoising_steps}")
        print(f"    binary_gripper_inf  = {getattr(policy.config, 'binary_gripper_inference', False)}")
        assert policy.model.chunk_size == policy.config.chunk_size, (
            f"MISMATCH: model.chunk_size={policy.model.chunk_size} != "
            f"config.chunk_size={policy.config.chunk_size}. "
            f"Config override was not applied before model construction!"
        )

    elif policy_type == "smolvla":
        policy = policy_cls.from_pretrained(path, device=device)
        if CHUNK_SIZE is not None and hasattr(policy.config, "chunk_size"):
            policy.config.chunk_size = CHUNK_SIZE
        if N_ACTION_STEPS is not None and hasattr(policy.config, "n_action_steps"):
            policy.config.n_action_steps = N_ACTION_STEPS
        if MAX_ACTION_TOKENS:
            policy.config.max_action_tokens = MAX_ACTION_TOKENS

    else:
        policy = policy_cls.from_pretrained(path, device=device)
        if CHUNK_SIZE is not None and hasattr(policy.config, "chunk_size"):
            policy.config.chunk_size = CHUNK_SIZE
        if N_ACTION_STEPS is not None and hasattr(policy.config, "n_action_steps"):
            policy.config.n_action_steps = N_ACTION_STEPS

    policy.reset()
    return policy, INCLUDE_EEF_STATE

def get_dataset(features: dict, robot_type: str):
    """
    Utility to create or resume a LeRobotDataset for evaluation.
    """
    repo_id = f"{HF_USER}/{EVAL_DATASET_NAME}"

    if RESUME_DATASET:
        print(f"[x] Resuming dataset: {repo_id} at {DATA_DIR}")
        dataset = LeRobotDataset(repo_id=repo_id, root=DATA_DIR)
        episode_idx = dataset.num_episodes
    else:
        print(f"[x] Creating new evaluation dataset: {repo_id}")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=CONTROL_FPS,
            features=features,
            robot_type=robot_type,
            use_videos=True,
            image_writer_threads=0,
            root=DATA_DIR,
        )
        episode_idx = 0
    return dataset, episode_idx
    
def main():
    # Robot Object
    so101        = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=WRIST_ROLL_OFFSET, home_pose=HOME_POSE)
    robot_config = SO100FollowerConfig(id=ROBOT_NAME, cameras=CAMERAS, port=FOLLOWER_PORT, calibration_dir=Path(CALIBRATION_DIR))
    robot        = SO100Follower(robot_config)
    
    # Policy Object
    policy, INCLUDE_EEF_STATE = get_policy(POLICY_TYPE, POLICY_PATH, DEVICE)
    action_slice_spec = None
    if POLICY_TYPE == "xvla":
        action_slice_spec = get_so101_slice_spec(getattr(policy.config, "action_mode", None))
        if action_slice_spec is None:
            raise ValueError(
                f"Unsupported XVLA action_mode in checkpoint config: {getattr(policy.config, 'action_mode', None)!r}"
            )

    # Configure the dataset features.
    # XVLA uses the checkpoint action_mode as the source of truth for both action
    # and observation.state schemas so record_loop emits robot actions with the
    # correct key names and the preprocessor sees the expected proprio format.
    if action_slice_spec is not None:
        mode_feature_spec = {
            "dtype": "float32",
            "shape": (action_slice_spec.real_dim,),
            "names": list(action_slice_spec.feature_names(suffix=".pos")),
        }
        action_features = {"action": mode_feature_spec}
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        obs_features["observation.state"] = mode_feature_spec
    else:
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    
    dataset_features = {**action_features, **obs_features}

    # ── DEBUG: print full feature map ──────────────────────────────────────────
    DBG.divider("white", "DATASET FEATURE MAP")
    for k, v in action_features.items():
        shape = v.get('shape', '?') if isinstance(v, dict) else '?'
        names = v.get('names', []) if isinstance(v, dict) else []
        DBG.info(f"  [action ] {k}: shape={shape}  names={names}")
    for k, v in obs_features.items():
        shape = v.get('shape', '?') if isinstance(v, dict) else '?'
        names = v.get('names', []) if isinstance(v, dict) else []
        DBG.info(f"  [obs    ] {k}: shape={shape}  names={names}")
    DBG.info(f"Total dataset_features: {list(dataset_features.keys())}")

    # Dataset Object
    dataset, episode_idx = get_dataset(dataset_features, robot.name)
    if POLICY_TYPE == "xvla":
        global ACTIVE_XVLA_RENAME_MAP
        ACTIVE_XVLA_RENAME_MAP = resolve_xvla_rename_map(dataset.meta.camera_keys)
        if not ACTIVE_XVLA_RENAME_MAP:
            raise ValueError(
                "Unable to resolve XVLA rename_map from dataset camera keys: "
                f"{dataset.meta.camera_keys}"
            )
        sync_xvla_policy_config(policy, dataset.meta, ACTIVE_XVLA_RENAME_MAP)
        DBG.pre(f"XVLA rename_map: {ACTIVE_XVLA_RENAME_MAP}")
        DBG.pre(f"XVLA input features: {list(policy.config.input_features.keys())}")

    # Build preprocessor and postprocessor for policy inference
    preprocessor, postprocessor = get_policy_processors(policy=policy, dataset=dataset, pipeline_key=POLICY_PIPELINE)

    # Meshcat visualization (optional — gracefully disabled if unavailable)
    viz = init_meshcat() if USE_MESHCAT_VIZ else None

    # Start inference and episode recording
    # ============================================================================

    # Initialize the keyboard listener and (optionally) rerun visualization
    _, events = init_keyboard_listener()
    if USE_RERUN:
        import uuid
        init_rerun(session_name=f"inference_evaluation_{uuid.uuid4().hex[:8]}")

    # Create robot processors (required by record_loop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Connect the robot
    robot.connect()

    # Patch robot for observation and action
    set_custom_get_observation(robot, so101=so101, include_eef_state=INCLUDE_EEF_STATE, dry_run=DRY_RUN)
    set_custom_send_action(robot, so101=so101, viz=viz, dry_run=DRY_RUN)
    # Trajectory viz is patched BEFORE the delay wrapper so it sits closest to
    # the original select_action — it draws the chunk immediately after inference.
    # Extract action names from the "action" feature entry (hw_to_dataset_features returns a
    # single "action" key, not individual "action.joint.pos" flat keys)
    action_names = dataset_features.get("action", {}).get("names", [])
    set_trajectory_visualization(policy, so101=so101, viz=viz, postprocessor=postprocessor, action_names=action_names)
    set_custom_select_action(policy, POLICY_DELAY)

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    log_say("Starting inference evaluation loop...")
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        # Move to home pose before episode
        log_say("Moving to home pose...")
        so101.reset_to_home(robot, duration_s=HOME_DURATION_S, fps=HOME_FPS, viz=viz)
        
        log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

        # Run the policy inference loop
        record_loop(
            robot=robot,
            events=events,
            fps=CONTROL_FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        # Handle re-recording
        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save the episode
        if len(dataset.episode_buffer) > 0:
            dataset.save_episode(parallel_encoding=False)
            episode_idx += 1
        else:
            print("No frames recorded in episode buffer. Skipping save.")

    # Clean up
    log_say("Stop recording")
    robot.disconnect()
    dataset.finalize()
    
    # Push to hub
    log_say(f"Pushing evaluation dataset to hub: {HF_USER}/{EVAL_DATASET_NAME}")
    dataset.push_to_hub()
    
    log_say("Inference evaluation complete!")

if __name__ == "__main__":
    main()
