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
from lerobot.processor import make_default_processors


NUM_EPISODES = 80
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 12
TASK_DESCRIPTION = "Pick up orange cube and place inside box."

DATA_DIR = Path("/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/outputs/datasets/SO101-1")

HF_USER = "edgarcancinoe"
HF_REPO_ID = "soarm101_pickup_orange" 

START_FROM_SCRATCH = False
RESUME_DATASET = True

assert not (START_FROM_SCRATCH and RESUME_DATASET), "Cannot start from scratch and resume dataset at the same time."

def main():
    # Manual delete
    if START_FROM_SCRATCH:
        import shutil
        # The dataset is stored in {root}
        TARGET_DIR = DATA_DIR
        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)

    # Load HW Config
    config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

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
            "lateral": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
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

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    # Connect the robot and teleoperator
    robot.connect()
    teleop.connect()

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
    dataset.push_to_hub()

if __name__ == "__main__":
    main()