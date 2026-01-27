import yaml
import argparse
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop


# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
# ============================================================================

# Policy configuration
# Option 1: Use local checkpoint path
POLICY_PATH = "/home/jose/vla_workspace/launch/outputs/train/smolvla_20260121_122931/checkpoints/020000/pretrained_model"
# Option 2: Use HuggingFace hub model (uncomment to use)
# POLICY_PATH = "edgarcancinoe/your_model_name"

# Evaluation dataset configuration
HF_USER = "edgarcancinoe"
EVAL_DATASET_NAME = "eval_soarm101_pick_cubes"  # Prefix with 'eval_' to distinguish from training data
DATA_DIR = Path("/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/outputs/datasets") / EVAL_DATASET_NAME

# Episode configuration
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 40  # Match training episode duration
TASK_DESCRIPTION = "Pick up all the orange cubes and place them inside the white container."

# Dataset options
START_FROM_SCRATCH = False
RESUME_DATASET = False

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
        
        # Rectify based on configuration
        if RECTIFY_TOP and "top" in observation:
            observation["top"] = camera_calibration.rectify_image(
                observation["top"], "top"
            )
            
        if RECTIFY_WRIST and "wrist" in observation:
            observation["wrist"] = camera_calibration.rectify_image(
                observation["wrist"], "wrist"
            )
            
        return observation
        
    robot.get_observation = patched_get_observation
    print(f"Robot observation patched for rectification (Top={RECTIFY_TOP}, Wrist={RECTIFY_WRIST})")

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
    args = parser.parse_args()

    calibration_dir = Path(args.calibration_dir)
    policy_path = args.policy_path

    # Create robot configuration
    robot_config = SO100FollowerConfig(
        id="my_awesome_follower_arm",
        cameras={
            "top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
            "wrist": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
        },
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
    log_say(f"Loading policy from {policy_path}")
    policy = ACTPolicy.from_pretrained(policy_path)
    
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
            image_writer_threads=4,
            root=DATA_DIR,
        )
        episode_idx = 0

    # Build preprocessor and postprocessor for policy inference
    # The inference device is automatically set to match the detected hardware
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="inference_evaluation")

    # Connect the robot
    robot.connect()
    
    # Patch robot for rectification
    patch_robot_for_rectification(robot)

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    log_say("Starting inference evaluation loop...")

    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

        # Run the policy inference loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
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
        dataset.save_episode(parallel_encoding=False)
        episode_idx += 1

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
