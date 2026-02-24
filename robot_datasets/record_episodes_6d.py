#!/usr/bin/env python3
"""
Records robot teleoperation episodes and converts joint positions to
6D end-effector representations (xyz + rot6d + gripper) before pushing
the dataset to the Hugging Face hub.

Joint data is recorded first using the standard LeRobot pipeline, then
transformed in-place via forward kinematics after finalization. The
resulting dataset schema matches the output of joints_to_6d_eef.py:

  observation.state        -> 10D EEF  (xyz + rot6d + gripper)
  action                   -> 10D EEF
  observation.joint_positions -> 6D original motor positions
  action_joints            -> 6D original motor actions
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.xvla.utils import mat_to_rotate6d
from lerobot.processor import make_default_processors
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

sys.path.append(str(Path(__file__).parent.parent))

from robot_control.so101_control import SO101Control
from utils import camera_calibration

# ----------------------------- Configuration -----------------------------
config_path = Path(__file__).parent.parent / "config" / "robot_config.yaml"
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

# Store camera config and rectification flags
CAMERA_CONFIG_MAP = config_data.get("cameras", {})
RECTIFY_MAP = {name: info.get("rectify", False) for name, info in CAMERA_CONFIG_MAP.items()}

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 9
TASK_DESCRIPTION = "Pick up orange cube and place inside white box."

HF_USER = "edgarcancinoe"
HF_REPO_ID = "soarm101_pickplace_10d"

URDF_PATH = config_data.get("robot", {}).get("urdf_path")
if not URDF_PATH:
    raise ValueError("Error: 'urdf_path' not found in config/robot_config.yaml.")
EEF_FEATURE_NAMES = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]

DATA_DIR = Path(__file__).parent.parent / "outputs" / "datasets" / HF_REPO_ID

START_FROM_SCRATCH = True
RESUME_DATASET = False

assert not (START_FROM_SCRATCH and RESUME_DATASET), (
    "Cannot start from scratch and resume dataset at the same time."
)
# -------------------------------------------------------------------------


# ─── Dataset integrity helpers (same as record_episodes.py) ───────────────

def _read_all_parquets(pq_dir: Path) -> pd.DataFrame:
    files = sorted(pq_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {pq_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def validate_lerobot_dataset_on_disk(root: Path) -> None:
    """Hard-fails if the on-disk dataset is inconsistent."""
    root = Path(root)
    meta = root / "meta"
    info_path = meta / "info.json"
    episodes_dir = meta / "episodes"

    if not info_path.exists():
        raise RuntimeError(f"Missing required file: {info_path}")
    if not episodes_dir.exists():
        raise RuntimeError(f"Missing required dir: {episodes_dir}")

    df_ep = _read_all_parquets(episodes_dir)
    if "episode_index" not in df_ep.columns:
        raise RuntimeError("meta/episodes parquet missing required column 'episode_index'")

    unique_eps = sorted(df_ep["episode_index"].dropna().unique().tolist())
    if not unique_eps:
        raise RuntimeError("No episodes found in meta/episodes (episode_index empty)")

    try:
        unique_eps_int = [int(x) for x in unique_eps]
    except Exception:
        raise RuntimeError(f"episode_index contains non-int-like values: {unique_eps[:10]}")

    n = len(unique_eps_int)
    expected = list(range(n))
    if unique_eps_int != expected:
        raise RuntimeError(
            "Non-contiguous or non-zero-based episode_index.\n"
            f"Expected {expected[:10]}... but got {unique_eps_int[:10]}...\n"
            "Fix: do not manually rewrite episode_index; let LeRobot manage it."
        )

    with open(info_path, "r") as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes")
    if total_episodes is None:
        raise RuntimeError("info.json missing key 'total_episodes'")
    if int(total_episodes) != n:
        raise RuntimeError(
            f"info.json total_episodes={total_episodes} but meta/episodes has {n} episodes"
        )

    rows = len(df_ep)
    if rows != n:
        raise RuntimeError(
            f"meta/episodes should have exactly 1 row per episode. "
            f"Got rows={rows} but unique episode_index={n}."
        )
    print(f"[OK] Dataset on disk looks sane: episodes={n}, rows={rows}")


# ─── Robot helpers ────────────────────────────────────────────────────────

def patch_robot_for_rectification(robot) -> None:
    original_get_observation = robot.get_observation

    def patched_get_observation():
        observation = original_get_observation()
        
        # Rectify based on per-camera configuration
        for cam_name, should_rectify in RECTIFY_MAP.items():
            if should_rectify and cam_name in observation:
                observation[cam_name] = camera_calibration.rectify_image(
                    observation[cam_name], cam_name
                )
        return observation
        
    robot.get_observation = patched_get_observation
    print(f"Robot observation patched for dynamic rectification based on config.")


# ─── EEF transformation (adapted from joints_to_6d_eef.py) ───────────────

def _compute_stats(data_list: list) -> dict:
    arr = np.array(data_list, dtype=np.float32)
    ddof = 1 if len(arr) > 1 else 0
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0, ddof=ddof).tolist(),
        "count": [len(arr)],
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q10": np.percentile(arr, 10, axis=0).tolist(),
        "q50": np.percentile(arr, 50, axis=0).tolist(),
        "q90": np.percentile(arr, 90, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist(),
    }


def transform_dataset_to_eef(
    root: Path,
    so101: SO101Control,
) -> None:
    """
    Rewrites every parquet under root/data in-place, replacing:
      observation.state  (6D joints)  ->  10D EEF  (xyz + rot6d + gripper)
      action             (6D joints)  ->  10D EEF

    and adding the original joints as:
      observation.joint_positions (6D)
      action_joints               (6D)

    Also updates meta/info.json, meta/features.json, and meta/stats.json.
    """
    root = Path(root)
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    global_eef_states: list = []
    global_eef_actions: list = []

    print(f"\nTransforming {len(parquet_files)} parquet file(s) to 10D EEF representation...")

    for parquet_path in parquet_files:
        print(f"  Processing: {parquet_path.relative_to(root)}")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if "observation.state" not in df.columns:
            print(f"  Warning: no 'observation.state' column in {parquet_path.name}, skipping.")
            continue

        total_frames = len(df)
        eef_states: list = []
        eef_actions: list = []
        has_action = "action" in df.columns

        for i in tqdm(range(total_frames), desc=parquet_path.name, leave=False):
            # SOURCE SELECTION:
            # If the dataset was already transformed, the original 6D joints are in 
            # 'observation.joint_positions'. Otherwise, they are in 'observation.state'.
            if "observation.joint_positions" in df.columns:
                state_joints = df.iloc[i]["observation.joint_positions"]
            else:
                state_joints = df.iloc[i]["observation.state"]
            
            # If joints are 6D, compute FK. If already 10D, keep as is (idempotency).
            if len(state_joints) == 6:
                state_deg = so101.motor_to_urdf_deg(state_joints)
                T = so101.kinematics.forward_kinematics(state_deg)
                eef_states.append(
                    np.concatenate([T[:3, 3], mat_to_rotate6d(T[:3, :3]), [state_joints[-1]]])
                )
                
            elif len(state_joints) == 10:
                eef_states.append(state_joints)
            else:
                print(f" Warning: frame {i} has unexpected state shape {len(state_joints)}, skipping.")
                eef_states.append(state_joints)

            if has_action:
                if "action_joints" in df.columns:
                    action_joints = df.iloc[i]["action_joints"]
                else:
                    action_joints = df.iloc[i]["action"]

                if len(action_joints) == 6:
                    action_deg = so101.motor_to_urdf_deg(action_joints)
                    T_act = so101.kinematics.forward_kinematics(action_deg)
                    eef_actions.append(
                        np.concatenate([T_act[:3, 3], mat_to_rotate6d(T_act[:3, :3]), [action_joints[-1]]])
                    )
                elif len(action_joints) == 10:
                    eef_actions.append(action_joints)
                else:
                    eef_actions.append(action_joints)

        # Build replacement table ----------------------------------------
        data_out: dict = {}
        schema_fields: list = []

        # 10D EEF observation state
        obs_arr = np.array(eef_states, dtype=np.float32).flatten()
        data_out["observation.state"] = pa.FixedSizeListArray.from_arrays(obs_arr, 10)
        schema_fields.append(pa.field("observation.state", pa.list_(pa.float32(), 10)))

        # 10D EEF action
        if has_action:
            act_arr = np.array(eef_actions, dtype=np.float32).flatten()
            data_out["action"] = pa.FixedSizeListArray.from_arrays(act_arr, 10)
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), 10)))

        # Scalar columns (timestamps, indices, etc.)
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in df.columns:
                data_out[col] = pa.array(df[col])
                schema_fields.append(pa.field(col, table.schema.field(col).type))

        pq.write_table(pa.Table.from_pydict(data_out, schema=pa.schema(schema_fields)), parquet_path)

        global_eef_states.extend(eef_states)
        if has_action:
            global_eef_actions.extend(eef_actions)

    # Update metadata --------------------------------------------------------
    meta_dir = root / "meta"

    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info_dict = json.load(f)
        if "features" in info_dict:
            # Only update if observation.state is still 6D
            if (
                "observation.state" in info_dict["features"] 
                and info_dict["features"]["observation.state"]["shape"] == [6]
            ):
                info_dict["features"]["observation.state"]["shape"] = [10]
                info_dict["features"]["observation.state"]["names"] = EEF_FEATURE_NAMES
            if (
                "action" in info_dict["features"] 
                and info_dict["features"]["action"]["shape"] == [6]
            ):
                info_dict["features"]["action"]["shape"] = [10]
                info_dict["features"]["action"]["names"] = EEF_FEATURE_NAMES
            for k in [
                "observation.robot_state.eef.mat",
                "observation.robot_state.eef.pos",
                "observation.robot_state.eef.rot6d",
            ]:
                info_dict["features"].pop(k, None)
        with open(info_path, "w") as f:
            json.dump(info_dict, f, indent=4)
        print("Updated meta/info.json.")

    features_path = meta_dir / "features.json"
    if features_path.exists():
        with open(features_path) as f:
            feat_dict = json.load(f)
        if "observation.state" in feat_dict:
            if feat_dict["observation.state"]["shape"] == [6]:
                feat_dict["observation.state"]["shape"] = [10]
                feat_dict["observation.state"]["names"] = EEF_FEATURE_NAMES
        if "action" in feat_dict:
            if feat_dict["action"]["shape"] == [6]:
                feat_dict["action"]["shape"] = [10]
                feat_dict["action"]["names"] = EEF_FEATURE_NAMES
        for k in [
            "observation.robot_state.eef.mat",
            "observation.robot_state.eef.pos",
            "observation.robot_state.eef.rot6d",
        ]:
            feat_dict.pop(k, None)
        with open(features_path, "w") as f:
            json.dump(feat_dict, f, indent=4)
        print("Updated meta/features.json.")

    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats_dict = json.load(f)
        if "observation.state" in stats_dict and global_eef_states:
            stats_dict["observation.state"] = _compute_stats(global_eef_states)
        if "action" in stats_dict and global_eef_actions:
            stats_dict["action"] = _compute_stats(global_eef_actions)
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=4)
        print("Updated meta/stats.json.")

    print("EEF transformation complete.\n")


def revert_dataset_to_6d(root: Path) -> None:
    """
    Reverts a 10D EEF dataset back to 6D joints by swapping columns
    and updating metadata. This allows resuming a previously converted dataset.
    """
    root = Path(root)
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        return

    with open(info_path, "r") as f:
        info_dict = json.load(f)

    # Only revert if currently 10D
    if info_dict.get("features", {}).get("observation.state", {}).get("shape") != [10]:
        return

    print(f"\n[RESUME] Detected 10D EEF dataset at {root.name}. Reverting to 6D joints for recording session...")

    # 1. Update Parquet files
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    for pq_path in parquet_files:
        table = pq.read_table(pq_path)
        df = table.to_pandas()
        
        if "observation.joint_positions" not in df.columns:
            continue
            
        data_out = {}
        schema_fields = []
        
        # Restore observation.state from joint_positions
        obs_arr = np.array(df["observation.joint_positions"].tolist(), dtype=np.float32).flatten()
        data_out["observation.state"] = pa.FixedSizeListArray.from_arrays(obs_arr, 6)
        schema_fields.append(pa.field("observation.state", pa.list_(pa.float32(), 6)))
        
        # Restore action from action_joints if it exists
        if "action_joints" in df.columns:
            act_arr = np.array(df["action_joints"].tolist(), dtype=np.float32).flatten()
            data_out["action"] = pa.FixedSizeListArray.from_arrays(act_arr, 6)
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), 6)))
        elif "action" in df.columns:
            data_out["action"] = pa.array(df["action"])
            schema_fields.append(pa.field("action", table.schema.field("action").type))

        # Copy other columns, skipping the EEF columns we moved back
        for col in df.columns:
            if col not in ["observation.state", "action", "observation.joint_positions", "action_joints"]:
                data_out[col] = pa.array(df[col])
                schema_fields.append(pa.field(col, table.schema.field(col).type))
        
        pq.write_table(pa.Table.from_pydict(data_out, schema=pa.schema(schema_fields)), pq_path)

    # 2. Update info.json
    if "observation.joint_positions" in info_dict["features"]:
        info_dict["features"]["observation.state"] = info_dict["features"].pop("observation.joint_positions")
    if "action_joints" in info_dict["features"]:
        info_dict["features"]["action"] = info_dict["features"].pop("action_joints")
    with open(info_path, "w") as f:
        json.dump(info_dict, f, indent=4)

    # 3. Update features.json
    features_path = meta_dir / "features.json"
    if features_path.exists():
        with open(features_path, "r") as f:
            feat_dict = json.load(f)
        if "observation.joint_positions" in feat_dict:
            feat_dict["observation.state"] = feat_dict.pop("observation.joint_positions")
        if "action_joints" in feat_dict:
            feat_dict["action"] = feat_dict.pop("action_joints")
        with open(features_path, "w") as f:
            json.dump(feat_dict, f, indent=4)

    # 4. Update stats.json
    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats_dict = json.load(f)
        if "observation.joint_positions" in stats_dict:
            stats_dict["observation.state"] = stats_dict.pop("observation.joint_positions")
        if "action_joints" in stats_dict:
            stats_dict["action"] = stats_dict.pop("action_joints")
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=4)

    print("Reversed to 6D joints successfully.\n")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record episodes and convert to 6D EEF representation."
    )
    parser.add_argument("--calibrate", action="store_true", help="Force motor recalibration")
    CALIBRATION_DIR = config_data["robot"].get("calibration_dir")
    if not CALIBRATION_DIR:
        print("Error: 'calibration_dir' not found in config/robot_config.yaml.")
        sys.exit(1)
    parser.add_argument("--calibration-dir", type=str, default=CALIBRATION_DIR, help="Path to calibration directory")
    parser.add_argument("--push", action="store_true", help="Push dataset to Hugging Face hub")
    args = parser.parse_args()

    if START_FROM_SCRATCH:
        import shutil
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

    calibration_dir = Path(args.calibration_dir)
    FOLLOWER_PORT = config_data["robot"]["port"]
    LEADER_PORT = config_data["leader"]["port"]

    # --- Kinematics setup ---------------------------------------------------
    wrist_offset = float(config_data.get("robot", {}).get("wrist_roll_offset", 0.0))
    home_pose = config_data.get("robot", {}).get("home_pose")
    so101 = SO101Control(urdf_path=URDF_PATH, wrist_roll_offset=wrist_offset, home_pose=home_pose)

    # --- Robot / teleop setup -----------------------------------------------
    robot_config = SO100FollowerConfig(
        id=config_data.get("robot", {}).get("name", "my_awesome_follower_arm"),
        cameras={
            name: OpenCVCameraConfig(index_or_path=info["id"], width=640, height=480, fps=FPS)
            for name, info in CAMERA_CONFIG_MAP.items()
        },
        port=FOLLOWER_PORT,
        calibration_dir=calibration_dir,
    )
    teleop_config = SO100LeaderConfig(
        id=config_data.get("leader", {}).get("name", "my_awesome_leader_arm"),
        port=LEADER_PORT,
        calibration_dir=calibration_dir,
    )

    if args.calibrate:
        for calib_file in [
            calibration_dir / f"{robot_config.id}.json",
            calibration_dir / f"{teleop_config.id}.json",
        ]:
            if calib_file.exists():
                calib_file.unlink()

    robot = SO100Follower(robot_config)
    teleop = SO100Leader(teleop_config)

    # --- Dataset ------------------------------------------------------------
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    if RESUME_DATASET:
        print(f"Resuming dataset from {DATA_DIR}")
        revert_dataset_to_6d(DATA_DIR)
        dataset = LeRobotDataset(repo_id=f"{HF_USER}/{HF_REPO_ID}", root=DATA_DIR)
        episode_idx = dataset.num_episodes
    else:
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

    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    robot.connect()
    patch_robot_for_rectification(robot)
    teleop.connect()

    # --- Wrist roll offset patch --------------------------------------------
    print(f"Applying Wrist Roll Offset: {wrist_offset}° via so101 control helper")
    original_get_action = teleop.get_action

    def patched_get_action():
        action = original_get_action()
        # Apply wrist roll offset via control helper
        motor_vals = np.array([action[f"{n}.pos"] for n in so101.JOINT_NAMES])
        motor_vals = so101.apply_wrist_roll_offset(motor_vals)
        for i, n in enumerate(so101.JOINT_NAMES):
            action[f"{n}.pos"] = float(motor_vals[i])
        return action

    teleop.get_action = patched_get_action
    # -----------------------------------------------------------------------

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    # --- Recording loop -----------------------------------------------------
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

        if not events["stop_recording"] and (
            episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
        ):
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

    log_say("Stop recording")
    robot.disconnect()
    teleop.disconnect()
    dataset.finalize()

    # --- Post-process: convert joints → 6D EEF ------------------------------
    log_say("Converting joint positions to 6D EEF representation...")
    transform_dataset_to_eef(DATA_DIR, so101)

    # --- Push to hub --------------------------------------------------------
    if args.push:
        validate_lerobot_dataset_on_disk(DATA_DIR)
        repo_id = f"{HF_USER}/{HF_REPO_ID}"
        print(f"Pushing {repo_id} to the Hugging Face hub...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=DATA_DIR, repo_id=repo_id, repo_type="dataset")
        info_path = DATA_DIR / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info_dict = json.load(f)
            try:
                version = info_dict.get("codebase_version", "v3.0")
                api.create_tag(repo_id=repo_id, tag=version, repo_type="dataset")
                print(f"Tagged dataset with version {version}.")
            except Exception as e:
                print(f"Warning: Could not tag dataset version: {e}")
        print(f"Successfully pushed {repo_id} to the hub!")


if __name__ == "__main__":
    main()
