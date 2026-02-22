import yaml
import argparse
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop


# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
# ============================================================================

# Policy configuration
# Option 1: Use HuggingFace hub model
# POLICY_PATH = "lerobot/smolvla_base"
# POLICY_PATH = "edgarcancinoe/xvla_finetuned_orange"
POLICY_PATH = "edgarcancinoe/xvla-base_finetuned_soarm101_pickplace_orange_240e_fw_closed"
# POLICY_TYPE = "smolvla"  # Options: "act", "smolvla", "xvla"
POLICY_TYPE = "xvla"
# POLICY_TYPE = "smolvla"

# Device configuration
DEVICE = "mps"  # Options: "cuda", "mps", "cpu"

# Evaluation dataset configuration
HF_USER = "edgarcancinoe"
EVAL_DATASET_NAME = "eval_" + POLICY_PATH.split("/")[-1] 
DATA_DIR = Path("/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/outputs/datasets") / EVAL_DATASET_NAME

# Episode configuration
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60  # Match training episode duration
TASK_DESCRIPTION = "Pick up orange cube and place inside white box."

# SmolVLA
DEFAULT_CHUNK_SIZE = 30
DEFAULT_N_ACTION_STEPS = 30


POLICY_DELAY = 0
DEFAULT_MAX_ACTION_TOKENS = None

# XVLA
NUM_IMAGE_VIEWS = 3
NUM_EMPTY_CAMERAS = 1
ACTION_MODE = "auto"
NUM_XVLA_OBS_STEPS = 1

# Robot Starting Position
STARTING_POSITION = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -80.0,
    "elbow_flex.pos": 100.0,
    "wrist_flex.pos": 30.0,
    "wrist_roll.pos": 40.0,
    # "gripper.pos": 0.0, # Optional: Don't force gripper if you want to keep object held or manual
}

# Dataset options
STARTING_POSITION_DURATION_S = 5
START_FROM_SCRATCH = False
RESUME_DATASET = False
OVERWRITE_DATASET = True  # Set True to delete and recreate the dataset on every run

# Define camera name mapping
CAMERA_MAPPING = {
    "xvla": {"top": "image", "wrist": "image2"},
    "default": {"top": "top", "wrist": "wrist"}
}
    
assert not (START_FROM_SCRATCH and RESUME_DATASET), "Cannot start from scratch and resume dataset at the same time."

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Load config
config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

# Configuration for Rectification from robot_config.yaml
RECTIFY_TOP = config_data.get("rectification", {}).get("top", True)
RECTIFY_WRIST = config_data.get("rectification", {}).get("wrist", True)

import sys
# Add project root to path to find utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import camera_calibration

def patch_robot_for_rectification(robot):
    original_get_observation = robot.get_observation
    
    def patched_get_observation():
        observation = original_get_observation()
        
        cam_names = CAMERA_MAPPING.get(POLICY_TYPE, CAMERA_MAPPING["default"])

        top_key = cam_names["top"]
        if RECTIFY_TOP and top_key in observation:
            observation[top_key] = camera_calibration.rectify_image(
                observation[top_key], "top"
            )
        
        wrist_key = cam_names["wrist"]
        if RECTIFY_WRIST and wrist_key in observation:
            observation[wrist_key] = camera_calibration.rectify_image(
                observation[wrist_key], "wrist"
            )
            
        return observation
        
    robot.get_observation = patched_get_observation
    print(f"Robot observation patched for rectification (Top={RECTIFY_TOP}, Wrist={RECTIFY_WRIST})")

def _motor_base_name(action_key: str) -> str:
    # "shoulder_pan.pos" -> "shoulder_pan"
    return action_key.split(".", 1)[0]

def move_to_start(robot, target_pos, duration_s: float = 3.0, fps: int = 30):
    """
    Moves robot to target_pos (e.g. STARTING_POSITION) over duration_s seconds.

    Key fixes:
    - Reads present position with motor base-names (no ".pos" mismatch).
    - Uses fixed number of interpolation steps (duration truly affects motion).
    - Uses a deadline-based sleep to reduce drift.
    """
    import time
    from lerobot.utils.robot_utils import precise_sleep

    duration_s = float(duration_s)
    fps = int(fps)
    steps = max(1, int(round(duration_s * fps)))

    print(f"Moving to Starting Position over {duration_s:.2f}s @ {fps} FPS -> {steps} steps")
    print(f"Target: {target_pos}")

    # --- read current state robustly ---
    start_values = {}
    if hasattr(robot, "follower_arm"):
        curr_read = robot.follower_arm.read("Present_Position")  # usually {motor_name: value}
        for k in target_pos.keys():
            base = _motor_base_name(k)
            if base in curr_read:
                start_values[k] = float(curr_read[base])
            elif k in curr_read:
                start_values[k] = float(curr_read[k])
            else:
                # fall back to target if unknown
                start_values[k] = float(target_pos[k])
    else:
        # no read available; fall back
        start_values = {k: float(v) for k, v in target_pos.items()}

    t0 = time.perf_counter()
    next_tick = t0

    for i in range(1, steps + 1):
        alpha = i / steps

        interpolated_action = {}
        for key, target_val in target_pos.items():
            sv = start_values.get(key, float(target_val))
            tv = float(target_val)
            interpolated_action[key] = sv + (tv - sv) * alpha

        robot.send_action(interpolated_action)

        next_tick += 1.0 / fps
        precise_sleep(max(0.0, next_tick - time.perf_counter()))

    # Final command (exact target)
    robot.send_action({k: float(v) for k, v in target_pos.items()})
    print("Reached Starting Position.")

    robot.send_action({k: float(v) for k, v in target_pos.items()})
    print("Reached Starting Position.")

def patch_policy_with_delay(policy, delay_s: float):
    """
    Wraps policy.select_action to inject a delay when the action queue is empty.
    """
    if delay_s <= 0:
        return

    original_select_action = policy.select_action

    def patched_select_action(batch, **kwargs):
        # If queue is empty, we are about to run inference -> sleep
        # NOTE: This sleep is blocking and happens inside the record_loop. 
        # Therefore, this delay counts towards the total EPISODE_TIME_SEC.
        if hasattr(policy, "action_queue") and len(policy.action_queue) == 0:
            import time
            print(f"Action queue empty. Delaying inference by {delay_s:.2f}s...")
            time.sleep(delay_s)
        
        return original_select_action(batch, **kwargs)

    policy.select_action = patched_select_action
    print(f"Policy patched with {delay_s}s delay before inference chunks.")
    
def main():
    # ... (skipping manual delete logic, it's fine)
    # Manual delete if starting from scratch
    if START_FROM_SCRATCH:
        import shutil
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

    FOLLOWER_PORT = config_data["robot"]["port"]

    DEFAULT_CALIBRATION_DIR = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/.cache/calibration"
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run policy inference and record evaluation episodes")
    parser.add_argument("--calibrate", action="store_true", help="Force calibration of the motors")
    parser.add_argument("--calibration-dir", type=str, default=DEFAULT_CALIBRATION_DIR, help="Path to calibration directory")
    parser.add_argument("--policy-path", type=str, default=POLICY_PATH, help="Path to policy checkpoint or HF model")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset if it exists")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help="SmolVLA: predicted action sequence length (policy.chunk_size)")
    parser.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS,
                    help="SmolVLA: number of steps executed per prediction (policy.n_action_steps)")
    parser.add_argument("--max-action-tokens", type=int, default=DEFAULT_MAX_ACTION_TOKENS,
                    help="SmolVLA: max_action_tokens (if supported by checkpoint)")
    parser.add_argument("--n-obs-steps", type=int, default=NUM_XVLA_OBS_STEPS,
                    help="XVLA: Number of observation steps (n_obs_steps)")
    parser.add_argument("--empty-cameras", type=int, default=NUM_EMPTY_CAMERAS,
                    help="XVLA: Number of empty camera views to add")
    parser.add_argument("--num-image-views", type=int, default=NUM_IMAGE_VIEWS,
                    help="XVLA: Total number of image views")
    parser.add_argument("--action-mode", type=str, default=ACTION_MODE, choices=["auto", "so101_bimanual", 'agibot_ee6d', 'joint', 'ee6d'],
                    help="XVLA: Action mode")
    parser.add_argument("--action-delay", type=float, default=POLICY_DELAY,
                    help="Delay (in seconds) before inference when action queue is empty")
    
    args = parser.parse_args()

    calibration_dir = Path(args.calibration_dir)
    policy_path = args.policy_path
    chunk_size = args.chunk_size
    n_action_steps = args.n_action_steps
    max_action_tokens = args.max_action_tokens

    # Handle overwrite (config var OR CLI flag)
    if (OVERWRITE_DATASET or args.overwrite) and DATA_DIR.exists():
        import shutil
        print(f"Overwriting dataset at {DATA_DIR}")
        shutil.rmtree(DATA_DIR)

    # Create robot configuration
    # Define generic camera configs
    top_cam_config = OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)
    wrist_cam_config = OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS)
    
    cam_names = CAMERA_MAPPING.get(POLICY_TYPE, CAMERA_MAPPING["default"])

    cameras_config = {
        cam_names["top"]: top_cam_config,
        cam_names["wrist"]: wrist_cam_config,
    }

    robot_config = SO100FollowerConfig(
        id="my_awesome_follower_arm",
        cameras=cameras_config,
        port=FOLLOWER_PORT,
        calibration_dir=calibration_dir,
    )

    if args.calibrate:
        print("Calibration requested. Deleting existing calibration files if they exist to force recalibration.")
        follower_calib = calibration_dir / f"{robot_config.id}.json"
        if follower_calib.exists():
            follower_calib.unlink()

    # Initialize the robot
    robot = SO100Follower(robot_config)

    # Load the trained policy
    print(f"Loading policy from {policy_path}")
    print(f"Policy type: {POLICY_TYPE}, Device: {DEVICE}")
    
    # Load policy based on type
    if POLICY_TYPE == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(policy_path, device=DEVICE)
    elif POLICY_TYPE == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(policy_path, device=DEVICE)
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = n_action_steps
        if max_action_tokens:
            policy.config.max_action_tokens = max_action_tokens
        policy.reset()
    elif POLICY_TYPE == "xvla":
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        policy = XVLAPolicy.from_pretrained(policy_path, device=DEVICE)
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = n_action_steps
        if args.n_obs_steps is not None:
             policy.config.n_obs_steps = args.n_obs_steps
        if args.empty_cameras > 0:
            policy.config.empty_cameras = args.empty_cameras
        if args.num_image_views is not None:
            policy.config.num_image_views = args.num_image_views
        if args.empty_cameras is not None:
            policy.config.empty_cameras = args.empty_cameras
        if args.action_mode is not None:
            policy.config.action_mode = args.action_mode
        
        # Override action features to match robot config (for unpadding)
        # from lerobot.configs.types import FeatureType, PolicyFeature
        # policy.config.output_features["action"] = PolicyFeature(
        #     type=FeatureType.ACTION,
        #     shape=(len(robot.action_features),),
        # )
        
        policy.reset()
    else:
        raise ValueError(f"Unknown policy type: {POLICY_TYPE}")
    
    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    if RESUME_DATASET:
        print(f"Resuming dataset from {DATA_DIR}")
        dataset = LeRobotDataset(
            repo_id=f"{HF_USER}/{EVAL_DATASET_NAME}",
            root=DATA_DIR,
        )
        episode_idx = dataset.num_episodes
    else:
        # Create the evaluation dataset
        dataset = LeRobotDataset.create(
            repo_id=f"{HF_USER}/{EVAL_DATASET_NAME}",
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=0,
            root=DATA_DIR,
        )
        episode_idx = 0

    # Build preprocessor and postprocessor for policy inference
    # The inference device is set to the configured DEVICE
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="inference_evaluation")
    
    # Create robot processors (required by record_loop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Connect the robot
    robot.connect()
    
    # Patch robot for rectification
    patch_robot_for_rectification(robot)

    # Patch policy with delay if requested
    if args.action_delay > 0:
        patch_policy_with_delay(policy, args.action_delay)

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    # Get home pose from config, fallback to default
    home_pose = config_data.get("robot", {}).get("home_pose", STARTING_POSITION)

    log_say("Starting inference evaluation loop...")

    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        # Move to start position before episode
        log_say("Moving to start position...")
        move_to_start(robot, home_pose, duration_s=STARTING_POSITION_DURATION_S, fps=FPS)
        
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
