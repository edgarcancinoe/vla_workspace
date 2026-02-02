import yaml
import argparse
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
import sys
from lerobot.processor import make_default_processors

# Add project root to path to find utils
sys.path.append(str(Path(__file__).parent.parent))

# Load config
config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

# Configuration for Rectification from robot_config.yaml
RECTIFY_TOP = config_data.get("rectification", {}).get("top", True)
RECTIFY_WRIST = config_data.get("rectification", {}).get("wrist", True)

NUM_EPISODES = 240
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 9
TASK_DESCRIPTION = "Pick up orange cube and place inside white box."

# Point to new merged dataset

HF_USER = "edgarcancinoe"
HF_REPO_ID = "soarm101_pickplace_cubes" 

DATA_DIR = Path(f"~/Documents/Academic/EMAI Thesis/vla_workspace/outputs/datasets/{HF_REPO_ID}").expanduser()

START_FROM_SCRATCH = False
RESUME_DATASET = True

assert not (START_FROM_SCRATCH and RESUME_DATASET), "Cannot start from scratch and resume dataset at the same time."

import json
from pathlib import Path
import pandas as pd


def _read_all_parquets(pq_dir: Path) -> pd.DataFrame:
    files = sorted(pq_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {pq_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def validate_lerobot_dataset_on_disk(root: Path) -> None:
    """
    Hard-fails if the on-disk dataset is inconsistent.
    This prevents pushing a broken dataset to the Hub.
    """
    root = Path(root)
    meta = root / "meta"
    info_path = meta / "info.json"
    episodes_dir = meta / "episodes"

    # 1) Required files/dirs exist
    if not info_path.exists():
        raise RuntimeError(f"Missing required file: {info_path}")
    if not episodes_dir.exists():
        raise RuntimeError(f"Missing required dir: {episodes_dir}")

    # 2) Episodes parquet exists and has episode_index
    df_ep = _read_all_parquets(episodes_dir)
    if "episode_index" not in df_ep.columns:
        raise RuntimeError("meta/episodes parquet missing required column 'episode_index'")

    # 3) Episode indices are sane
    unique_eps = sorted(df_ep["episode_index"].dropna().unique().tolist())
    if not unique_eps:
        raise RuntimeError("No episodes found in meta/episodes (episode_index empty)")

    # must be integers (or int-like)
    try:
        unique_eps_int = [int(x) for x in unique_eps]
    except Exception:
        raise RuntimeError(f"episode_index contains non-int-like values: {unique_eps[:10]}")

    # must be contiguous from 0..N-1 (LeRobot generally assumes this)
    n = len(unique_eps_int)
    expected = list(range(n))
    if unique_eps_int != expected:
        raise RuntimeError(
            "Non-contiguous or non-zero-based episode_index.\n"
            f"Expected {expected[:10]}... but got {unique_eps_int[:10]}...\n"
            "Fix: do not manually rewrite episode_index; let LeRobot manage it."
        )

    # 4) info.json must agree with metadata
    with open(info_path, "r") as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes", None)
    if total_episodes is None:
        raise RuntimeError("info.json missing key 'total_episodes'")
    if int(total_episodes) != n:
        raise RuntimeError(
            f"info.json total_episodes={total_episodes} but meta/episodes has {n} episodes"
        )

    # 5) Sanity: rows >> episodes (otherwise someone did row-index episode hack)
    rows = len(df_ep)
    if rows != n:
        raise RuntimeError(
            f"meta/episodes should have exactly 1 row per episode. "
            f"Got rows={rows} but unique episode_index={n}."
        )
    print(f"[OK] Dataset on disk looks sane: episodes={n}, rows={rows}")


def safe_push(dataset: "LeRobotDataset", root: Path) -> None:
    """
    Validate first; only then push.
    """
    validate_lerobot_dataset_on_disk(root)
    dataset.push_to_hub()
    print("[OK] push_to_hub completed")

class RectifiedDataset:
    """Wrapper to rectify images before adding to dataset."""
    def __init__(self, dataset):
        self.dataset = dataset
        
    def add_frame(self, frame):
        # Rectify images in frame
        for key in list(frame.keys()):
            if "observation.images" in key:
                 # Extract camera name (e.g. 'observation.images.top' -> 'top')
                 cam_name = key.split(".")[-1]
                 # Rectify
                 frame[key] = camera_calibration.rectify_image(frame[key], cam_name)
        self.dataset.add_frame(frame)
        
    def __getattr__(self, name):
        return getattr(self.dataset, name)



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
    # Manual delete
    if START_FROM_SCRATCH:
        import shutil
        # The dataset is stored in {root}
        TARGET_DIR = DATA_DIR
        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)

    FOLLOWER_PORT = config_data["robot"]["port"]
    LEADER_PORT = config_data["leader_port"]

    DEFAULT_CALIBRATION_DIR = "/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/.cache/calibration"
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Force calibration of the motors")
    parser.add_argument("--calibration-dir", type=str, default=DEFAULT_CALIBRATION_DIR, help="Path to calibration directory")
    args = parser.parse_args()

    calibration_dir = Path(args.calibration_dir)

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

    teleop_config = SO100LeaderConfig(
        id="my_awesome_leader_arm",
        port=LEADER_PORT,
        calibration_dir=calibration_dir,
    )

    if args.calibrate:
        print("Calibration requested. Deleting existing calibration files if they exist to force recalibration.")
        # Calibration files are usually named <id>.json in the calibration_dir
        # But the library handles it. If strictly 'force recalibration' is needed, we might need to rely on the library logic 
        # or manually delete the files. 
        # Usually, if the file doesn't exist, it calibrates. 
        # So if we want to FORCE, we should delete them.
        follower_calib = calibration_dir / f"{robot_config.id}.json"
        leader_calib = calibration_dir / f"{teleop_config.id}.json"
        if follower_calib.exists():
            follower_calib.unlink()
        if leader_calib.exists():
            leader_calib.unlink()

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop = SO100Leader(teleop_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    if RESUME_DATASET:
        print(f"Resuming dataset from {DATA_DIR}")
        dataset = LeRobotDataset(
            repo_id=f"{HF_USER}/{HF_REPO_ID}",
            root=DATA_DIR,
        )
        episode_idx = dataset.num_episodes
    else:
        # Create the dataset
        dataset = LeRobotDataset.create(
            repo_id=f"{HF_USER}/{HF_REPO_ID}",
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
            root=DATA_DIR,
        )
        episode_idx = 0

    # WRAP DATASET TO RECTIFY IMAGES
    # dataset = RectifiedDataset(dataset) # Removed to avoid double rectification

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")


    # Connect the robot and teleoperator
    robot.connect()
    
    # Patch robot for rectification (Handles both Rerun and Dataset)
    patch_robot_for_rectification(robot)
    
    teleop.connect()
    
    # --- Patch Teleop for Wrist Roll Offset ---
    WRIST_ROLL_OFFSET = config_data.get("robot", {}).get("wrist_roll_offset", 0.0)
    print(f"Applying Wrist Roll Offset: {WRIST_ROLL_OFFSET}")

    original_get_action = teleop.get_action
    
    def patched_get_action():
        action = original_get_action()
        if "wrist_roll.pos" in action:
            new_val = action["wrist_roll.pos"] + WRIST_ROLL_OFFSET
            # Clamp to range [-100, 100] (assuming standard LeRobot range)
            action["wrist_roll.pos"] = max(min(new_val, 100.0), -100.0)
        return action

    teleop.get_action = patched_get_action
    # ----------------------------------------

    # Create the required processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task="RESET: Move robot to start position",
                display_data=True,
            )
            log_say("Reset complete")

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode(parallel_encoding=False)
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    robot.disconnect()
    teleop.disconnect()
    dataset.finalize()
    safe_push(dataset, DATA_DIR)

if __name__ == "__main__":
    main()