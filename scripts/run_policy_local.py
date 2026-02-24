#!/usr/bin/env python3
import sys
import yaml
import numpy as np
from functools import partial
from pathlib import Path
from lerobot.utils.utils import log_say as lerobot_log_say
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.policies.xvla.utils import mat_to_rotate6d
from lerobot.policies.factory import make_pre_post_processors
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig

sys.path.append(str(Path(__file__).parent.parent))
from utils import camera_calibration
from robot_control.so101_control import SO101Control

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
config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

WRIST_ROLL_OFFSET = config_data.get("robot", {}).get("wrist_roll_offset", 0.0)
URDF_PATH         = config_data.get("robot", {}).get("urdf_path")
CALIBRATION_DIR = config_data["robot"].get("calibration_dir")
FOLLOWER_PORT = config_data["robot"]["port"]
ROBOT_NAME = config_data["robot"].get("name", "arm_follower")
HOME_POSE = config_data["robot"].get("home_pose", {})
CAMERA_CONFIG_MAP = config_data.get("cameras", {})
RECTIFY_MAP = {name: info.get("rectify", False) for name, info in CAMERA_CONFIG_MAP.items()}
CAMERAS = {name: OpenCVCameraConfig(index_or_path=info["id"], width=640, height=480, fps=FPS) for name, info in CAMERA_CONFIG_MAP.items()}

# --- Policy ---
POLICY_PATH = "edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_orange_050e_fw_open"
# POLICY_PATH = "edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_orange_240e_fw_closed"
# POLICY_PATH = "edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_6d"
# POLICY_PATH = "edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_6d_240e_fw_closed"

POLICY_TYPE = "xvla" # "xvla" | "smolvla"
DEVICE      = "mps"  # "cuda" | "mps" | "cpu"

TASK_DESCRIPTION = "Pick up orange cube and place inside white box."
# Pipeline selection: None (default) | 'xvla_default'
POLICY_PIPELINE = None 
# --- Voice ---
USE_VOICE                  = True

# --- SmolVLA
CHUNK_SIZE         = 30
N_ACTION_STEPS     = 30
MAX_ACTION_TOKENS  = None
POLICY_DELAY               = 0

# --- XVLA
NUM_IMAGE_VIEWS            = 3
NUM_EMPTY_CAMERAS          = 1
ACTION_MODE                = "auto"
NUM_XVLA_OBS_STEPS         = 1

# --- Evaluation & Dataset ---
NUM_EPISODES               = 5
FPS                        = 30
EPISODE_TIME_SEC           = 60
HF_USER                    = "edgarcancinoe"
EVAL_DATASET_NAME          = "eval_" + POLICY_PATH.split("/")[-1] 
DATA_DIR                   = Path(__file__).parent.parent / "outputs" / "datasets" / EVAL_DATASET_NAME

START_FROM_SCRATCH         = False
RESUME_DATASET             = False
OVERWRITE_DATASET          = True  # Set True to delete and recreate the dataset on every run

# --- Robot & Setup
STARTING_POSITION = { "shoulder_pan.pos": 0.0, "shoulder_lift.pos": -80.0, "elbow_flex.pos": 100.0, "wrist_flex.pos": 30.0, "wrist_roll.pos": 40.0}
CAMERA_MAPPING = {
    "xvla": {"main": "image", "secondary": "image2"},
    "default": {"main": "main", "secondary": "secondary"}
}
STARTING_POSITION_DURATION_S = 5
HOME_DURATION_S = STARTING_POSITION_DURATION_S
HOME_FPS = FPS

# ─── EEF state feature names (must match training dataset schema) ─────────────
EEF_STATE_NAMES = [
    "x", "y", "z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
]
_EEF_KEY_SET = {"x", "y", "z", "gripper"}

validate_configuration()

# UTIL
# ============================================================================

# Utility to log and speak, respecting the USE_VOICE toggle
log_say = partial(lerobot_log_say, play_sounds=USE_VOICE)

# > Custom get_observation():
#   1. Rectify images based on config
#   2. Inject EEF State if required
def set_custom_get_observation(robot, so101: SO101Control = None, include_eef_state: bool = False):
    
    base_obs_func = robot.get_observation

    def patched_get_observation():
        observation = base_obs_func()
        
        # Rectify images based on config
        for cam_name, should_rectify in RECTIFY_MAP.items():
            if should_rectify and cam_name in observation:
                observation[cam_name] = camera_calibration.rectify_image(
                    observation[cam_name], cam_name
                )
        print(f"[x] Robot observation patched for dynamic rectification based on config: {RECTIFY_MAP}")


        # Inject EEF State if requested
        if include_eef_state:
            motor_vals = np.array([obs.get(f"{joint}.pos", 0.0) for joint in so101.JOINT_NAMES])
            T = so101.fk(motor_vals)
            r6d = mat_to_rotate6d(T[:3, :3])
            eef = np.concatenate([T[:3, 3], r6d, [motor_vals[-1]]], dtype=np.float32)
            for i, name in enumerate(EEF_STATE_NAMES):
                observation[f"{name}.pos"] = float(eef[i])
            print(f"[x] Robot observation patched for EEF State: {EEF_STATE_NAMES}")

        return observation
        
    robot.get_observation = patched_get_observation

# > Custom send_action():
#   1. Convert EEF State to motor space if required
def set_custom_send_action(robot, so101: SO101Control = None):
    base_action_func = robot.send_action
    gr_idx = so101.JOINT_NAMES.index("gripper")

    def patched_send_action(action, **kwargs):
        # Case: actions are in motor space
        if "shoulder_pan.pos" in action:
            return base_action_func(action, **kwargs)
        
        # Case: actions are in eef space (xyz + 6D Orientation + Gripper)
        xyz         = np.array([action["x"], action["y"], action["z"]], dtype=np.float64)
        r6d         = np.array([action["rot6d_0"], action["rot6d_1"], action["rot6d_2"], action["rot6d_3"], action["rot6d_4"], action["rot6d_5"]], dtype=np.float64)
        gripper_val = float(action["gripper"]) 

        # IK
        current_motor = so101.read_motor_real(robot)
        target_motor = so101.ik_motor(xyz, r6d, current_motor)

        if target_motor is None:
            print(f"[ERROR] IK failed, skipping action and holding current pose")
            motor_dict = {f"{n}.pos": float(current_motor[i]) for i, n in enumerate(so101.JOINT_NAMES)}
            return base_action_func(motor_dict, **kwargs)
        
        # Gripper override
        target_motor[gr_idx] = gripper_val
        # Convert to dictionary for lerobot send_action format
        motor_dict = {f"{n}.pos": float(target_motor[i]) for i, n in enumerate(so101.JOINT_NAMES)}
        
        return base_action_func(motor_dict, **kwargs)

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

def get_policy_processors(policy, dataset, pipeline_key: str | None = None):
    """
    Utility to choose pre- and post-processors by a key.
    - 'xvla_default': Use our custom XVLA replication.
    - None (or any other): Use LeRobot's default factory.
    """
    if pipeline_key == "xvla_default":
        print(f"[x] Using custom pipeline: {pipeline_key}")
        from custom_pipelines.xvla_processor import make_custom_xvla_processors
        return make_custom_xvla_processors(
            config=policy.config,
            dataset_stats=dataset.meta.stats,
        )
    
    # Default behavior
    print(f"[x] Using default abandoned/factory processors (Device: {DEVICE})")
    return make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=POLICY_PATH,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

def get_policy(policy_type: str, path: str, device: str):
    """
    Utility to load and configure a policy based on its type.
    Uses LeRobot's factory to resolve policy classes elegantly.
    """
    from lerobot.policies.factory import get_policy_class
    
    print(f"[x] Loading policy {path} ({policy_type}) on {device}")
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(path, device=device)
    
    # Common configurations
    if hasattr(policy.config, "chunk_size"):
        policy.config.chunk_size = CHUNK_SIZE
    if hasattr(policy.config, "n_action_steps"):
        policy.config.n_action_steps = N_ACTION_STEPS
        
    # smolvla specific
    if policy_type == "smolvla":
        if MAX_ACTION_TOKENS:
            policy.config.max_action_tokens = MAX_ACTION_TOKENS
            
    # xvla specific 
    if policy_type == "xvla":
        policy.config.n_obs_steps   = NUM_XVLA_OBS_STEPS
        policy.config.empty_cameras = NUM_EMPTY_CAMERAS
        policy.config.num_image_views = NUM_IMAGE_VIEWS
        policy.config.action_mode   = ACTION_MODE
        
    policy.reset()
    return policy

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
            fps=FPS,
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
    policy = get_policy(POLICY_TYPE, POLICY_PATH, DEVICE)

    # Configure the dataset features
    action_features  = hw_to_dataset_features(robot.action_features, "action")
    obs_features     = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Dataset Object
    dataset, episode_idx = get_dataset(dataset_features, robot.name)

    # Build preprocessor and postprocessor for policy inference
    preprocessor, postprocessor = get_policy_processors(policy=policy, dataset=dataset, pipeline_key=POLICY_PIPELINE)

    # Start inference and episode recording
    # ============================================================================

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="inference_evaluation")
    
    # Create robot processors (required by record_loop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Connect the robot
    robot.connect()
    
    # Patch robot for observation and action
    set_custom_get_observation(robot)
    set_custom_send_action(robot, so101=so101)
    set_custom_select_action(policy, POLICY_DELAY)

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    log_say("Starting inference evaluation loop...")
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        # Move to home pose before episode
        log_say("Moving to home pose...")
        so101.reset_to_home(robot, duration_s=HOME_DURATION_S, fps=HOME_FPS)
        log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

        # Run the policy inference loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
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