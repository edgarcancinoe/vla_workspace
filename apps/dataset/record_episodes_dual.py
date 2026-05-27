#!/usr/bin/env python3
"""
Record dual-arm SO episodes using bimanual LeRobot drivers and store both:
- primary dual EEF representation (20D): observation.state, action
- raw dual joint representation (12D): observation.joint_positions, action_joints
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import yaml
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
lerobot_src = ROOT_DIR.parent / "repos" / "lerobot" / "src"
if lerobot_src.exists():
    sys.path.insert(0, str(lerobot_src))

from thesis_vla.common.paths import DATASETS_OUTPUT_DIR
from thesis_vla.robot.so101_control import SO101Control
from thesis_vla.vision import camera_calibration

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.xvla.utils import mat_to_rotate6d
from lerobot.processor import make_default_processors
from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.teleoperators.bi_so_leader.bi_so_leader import BiSOLeader
from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


DEFAULT_DUAL_CONFIG = ROOT_DIR / "config" / "robot" / "robot_config_dual.yaml"

JOINT_NAMES_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_NAMES_12D = [*[f"left_{name}.pos" for name in JOINT_NAMES_ARM], *[f"right_{name}.pos" for name in JOINT_NAMES_ARM]]
JOINT_NAMES_12D = [*[f"left_{name}" for name in JOINT_NAMES_ARM], *[f"right_{name}" for name in JOINT_NAMES_ARM]]
EEF_NAMES_20D = ["left_x", "left_y", "left_z", "left_rot6d_0", "left_rot6d_1", "left_rot6d_2", "left_rot6d_3", "left_rot6d_4", "left_rot6d_5", "left_gripper", "right_x", "right_y", "right_z", "right_rot6d_0", "right_rot6d_1", "right_rot6d_2", "right_rot6d_3", "right_rot6d_4", "right_rot6d_5", "right_gripper"]

# ----------------------------- Configuration -----------------------------
# Keep dataset recording knobs here (same style as single-arm recorder).
HF_USER = "edgarcancinoe"
HF_REPO_ID = "soarm101_dual_pickup_cloth"
TASK_DESCRIPTION = "Dual-arm teleoperation task: Pick up square cloth and hold extended in the air."
FPS = 30
NUM_EPISODES = 2
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
VIDEO_CODEC = "h264"
STREAMING_ENCODING = True
BATCH_ENCODING_SIZE = 1
IMAGE_WRITER_THREADS = 4
DISPLAY_DATA = True
CALIBRATE = False
SKIP_RECORDING = False
PUSH_TO_HUB = False
RESUME_DATASET = True
# -------------------------------------------------------------------------


def _required(section: str, data: dict, keys: list[str]) -> None:
    for key in keys:
        if data.get(key) in (None, ""):
            raise ValueError(f"Missing required config field '{section}.{key}'.")


def _load_dual_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    for rk in ("robot_a", "robot_b"):
        if rk not in cfg:
            raise ValueError(f"Missing required section '{rk}'.")
        _required(rk, cfg[rk], ["name", "port", "urdf_path", "calibration_dir", "home_pose"])

    for lk in ("leader_a", "leader_b"):
        if lk not in cfg:
            raise ValueError(f"Missing required section '{lk}'.")
        _required(lk, cfg[lk], ["name", "port"])

    _required("config", {"robot_id": cfg.get("robot_id"), "leader_id": cfg.get("leader_id")}, ["robot_id", "leader_id"])

    return cfg


def _camera_map(robot_cfg: dict) -> tuple[dict[str, OpenCVCameraConfig], dict[str, bool]]:
    cameras_cfg = robot_cfg.get("cameras", {})
    cameras = {}
    rectify = {}
    for name, info in cameras_cfg.items():
        if "id" not in info:
            raise ValueError(f"Camera '{name}' is missing required field 'id'.")
        cameras[name] = OpenCVCameraConfig(
            index_or_path=info["id"],
            width=info.get("width", 640),
            height=info.get("height", 480),
            fps=info.get("fps", 30),
        )
        rectify[name] = bool(info.get("rectify", False))
    return cameras, rectify


def _compute_stats(vectors: list[np.ndarray]) -> dict:
    arr = np.array(vectors, dtype=np.float32)
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


def _to_fixed_list(values: list[np.ndarray], dim: int) -> pa.FixedSizeListArray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != dim:
        raise ValueError(f"Expected vectors with dim={dim}, got {arr.shape[1]}.")
    return pa.FixedSizeListArray.from_arrays(arr.flatten(), dim)


def _joint6_to_eef10(joint6: np.ndarray, control: SO101Control) -> np.ndarray:
    state_deg = control.motor_to_urdf_deg(joint6)
    T = control.kinematics.forward_kinematics(state_deg)
    return np.concatenate([T[:3, 3], mat_to_rotate6d(T[:3, :3]), [joint6[-1]]]).astype(np.float32)


def _joint12_to_dual_eef20(joint12: np.ndarray, left_ctrl: SO101Control, right_ctrl: SO101Control) -> np.ndarray:
    if len(joint12) != 12:
        raise ValueError(f"Expected 12D joint vector, got {len(joint12)}")
    left_10 = _joint6_to_eef10(np.array(joint12[:6], dtype=np.float32), left_ctrl)
    right_10 = _joint6_to_eef10(np.array(joint12[6:12], dtype=np.float32), right_ctrl)
    return np.concatenate([left_10, right_10]).astype(np.float32)


def _update_feature_specs(meta_dir: Path, to_eef: bool) -> None:
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        feats = info.get("features", {})

        if to_eef:
            if "observation.state" in feats:
                feats["observation.state"] = {"dtype": "float32", "shape": [20], "names": EEF_NAMES_20D}
            if "action" in feats:
                feats["action"] = {"dtype": "float32", "shape": [20], "names": EEF_NAMES_20D}
            feats["observation.joint_positions"] = {
                "dtype": "float32",
                "shape": [12],
                "names": JOINT_NAMES_12D,
            }
            feats["action_joints"] = {
                "dtype": "float32",
                "shape": [12],
                "names": JOINT_NAMES_12D,
            }
        else:
            if "observation.state" in feats:
                feats["observation.state"] = {"dtype": "float32", "shape": [12], "names": MOTOR_NAMES_12D}
            if "action" in feats:
                feats["action"] = {"dtype": "float32", "shape": [12], "names": MOTOR_NAMES_12D}
            feats.pop("observation.joint_positions", None)
            feats.pop("action_joints", None)

        info["features"] = feats
        with info_path.open("w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=4)

    features_path = meta_dir / "features.json"
    if features_path.exists():
        with features_path.open("r", encoding="utf-8") as handle:
            feats = json.load(handle)

        if to_eef:
            if "observation.state" in feats:
                feats["observation.state"] = {"dtype": "float32", "shape": [20], "names": EEF_NAMES_20D}
            if "action" in feats:
                feats["action"] = {"dtype": "float32", "shape": [20], "names": EEF_NAMES_20D}
            feats["observation.joint_positions"] = {
                "dtype": "float32",
                "shape": [12],
                "names": JOINT_NAMES_12D,
            }
            feats["action_joints"] = {
                "dtype": "float32",
                "shape": [12],
                "names": JOINT_NAMES_12D,
            }
        else:
            if "observation.state" in feats:
                feats["observation.state"] = {"dtype": "float32", "shape": [12], "names": MOTOR_NAMES_12D}
            if "action" in feats:
                feats["action"] = {"dtype": "float32", "shape": [12], "names": MOTOR_NAMES_12D}
            feats.pop("observation.joint_positions", None)
            feats.pop("action_joints", None)

        with features_path.open("w", encoding="utf-8") as handle:
            json.dump(feats, handle, indent=4)


def transform_dataset_to_dual_eef(root: Path, left_ctrl: SO101Control, right_ctrl: SO101Control) -> None:
    root = Path(root)
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        print("No parquet files found for transform. Skipping.")
        return

    global_eef_states = []
    global_eef_actions = []
    global_joint_states = []
    global_joint_actions = []

    print(f"Transforming {len(parquet_files)} parquet file(s) to dual EEF+joint format...")

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if "observation.state" not in df.columns:
            continue

        eef_states = []
        eef_actions = []
        joint_states = []
        joint_actions = []
        has_action = "action" in df.columns

        for i in range(len(df)):
            state_raw = df.iloc[i]["observation.joint_positions"] if "observation.joint_positions" in df.columns else df.iloc[i]["observation.state"]
            state_vec = np.array(state_raw, dtype=np.float32)

            if len(state_vec) == 12:
                joint_states.append(state_vec)
                eef_states.append(_joint12_to_dual_eef20(state_vec, left_ctrl, right_ctrl))
            elif len(state_vec) == 20:
                eef_states.append(state_vec)
            else:
                raise ValueError(f"Unexpected state dim in {parquet_path}: {len(state_vec)}")

            if has_action:
                action_raw = df.iloc[i]["action_joints"] if "action_joints" in df.columns else df.iloc[i]["action"]
                action_vec = np.array(action_raw, dtype=np.float32)

                if len(action_vec) == 12:
                    joint_actions.append(action_vec)
                    eef_actions.append(_joint12_to_dual_eef20(action_vec, left_ctrl, right_ctrl))
                elif len(action_vec) == 20:
                    eef_actions.append(action_vec)
                else:
                    raise ValueError(f"Unexpected action dim in {parquet_path}: {len(action_vec)}")

        out = {}
        schema_fields = []

        out["observation.state"] = _to_fixed_list(eef_states, 20)
        schema_fields.append(pa.field("observation.state", pa.list_(pa.float32(), 20)))

        if has_action:
            out["action"] = _to_fixed_list(eef_actions, 20)
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), 20)))

        if joint_states:
            out["observation.joint_positions"] = _to_fixed_list(joint_states, 12)
            schema_fields.append(pa.field("observation.joint_positions", pa.list_(pa.float32(), 12)))

        if has_action and joint_actions:
            out["action_joints"] = _to_fixed_list(joint_actions, 12)
            schema_fields.append(pa.field("action_joints", pa.list_(pa.float32(), 12)))

        replaced = {"observation.state", "action", "observation.joint_positions", "action_joints"}
        for col in df.columns:
            if col in replaced:
                continue
            out[col] = pa.array(df[col], type=table.schema.field(col).type)
            schema_fields.append(pa.field(col, table.schema.field(col).type))

        pq.write_table(pa.Table.from_pydict(out, schema=pa.schema(schema_fields)), parquet_path)

        global_eef_states.extend(eef_states)
        global_joint_states.extend(joint_states)
        global_eef_actions.extend(eef_actions)
        global_joint_actions.extend(joint_actions)

    _update_feature_specs(root / "meta", to_eef=True)

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if global_eef_states:
            stats["observation.state"] = _compute_stats(global_eef_states)
        if global_eef_actions:
            stats["action"] = _compute_stats(global_eef_actions)
        if global_joint_states:
            stats["observation.joint_positions"] = _compute_stats(global_joint_states)
        if global_joint_actions:
            stats["action_joints"] = _compute_stats(global_joint_actions)
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=4)

    print("Dual EEF transform complete.")


def revert_dataset_to_dual_joints(root: Path) -> None:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return

    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    current_shape = info.get("features", {}).get("observation.state", {}).get("shape")
    if current_shape != [20]:
        return

    print(f"[RESUME] Reverting transformed dual dataset '{root.name}' back to 12D joints for recording session...")

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if "observation.joint_positions" not in df.columns:
            continue

        out = {}
        schema_fields = []

        state12 = np.array(df["observation.joint_positions"].tolist(), dtype=np.float32)
        out["observation.state"] = pa.FixedSizeListArray.from_arrays(state12.flatten(), 12)
        schema_fields.append(pa.field("observation.state", pa.list_(pa.float32(), 12)))

        if "action_joints" in df.columns:
            action12 = np.array(df["action_joints"].tolist(), dtype=np.float32)
            out["action"] = pa.FixedSizeListArray.from_arrays(action12.flatten(), 12)
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), 12)))
        elif "action" in df.columns:
            out["action"] = pa.array(df["action"])
            schema_fields.append(pa.field("action", table.schema.field("action").type))

        skip_cols = {"observation.state", "action", "observation.joint_positions", "action_joints"}
        for col in df.columns:
            if col in skip_cols:
                continue
            out[col] = pa.array(df[col], type=table.schema.field(col).type)
            schema_fields.append(pa.field(col, table.schema.field(col).type))

        pq.write_table(pa.Table.from_pydict(out, schema=pa.schema(schema_fields)), parquet_path)

    _update_feature_specs(root / "meta", to_eef=False)

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if "observation.joint_positions" in stats:
            stats["observation.state"] = stats.pop("observation.joint_positions")
        if "action_joints" in stats:
            stats["action"] = stats.pop("action_joints")
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=4)

    print("Reverted dataset to 12D dual joints.")


def has_local_lerobot_dataset(root: Path) -> bool:
    root = Path(root)
    return (root / "meta" / "info.json").exists() and (root / "meta" / "episodes").exists()


def record_loop_teleop(
    robot: BiSOFollower,
    teleop: BiSOLeader,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    control_time_s: int,
    dataset: LeRobotDataset | None = None,
    single_task: str | None = None,
    display_data: bool = False,
) -> None:
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    start_episode_t = time.perf_counter()
    timestamp = 0.0

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = (
            build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            if dataset is not None
            else None
        )

        act = teleop.get_action()
        act_processed = teleop_action_processor((act, obs))
        action_values = act_processed
        robot_action_to_send = robot_action_processor((act_processed, obs))
        robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


def patch_robot_for_rectification(robot: BiSOFollower, left_rectify: dict[str, bool], right_rectify: dict[str, bool]) -> None:
    original_get_observation = robot.get_observation

    def patched_get_observation():
        observation = original_get_observation()
        for cam_name, should_rectify in left_rectify.items():
            key = f"left_{cam_name}"
            if should_rectify and key in observation:
                observation[key] = camera_calibration.rectify_image(observation[key], cam_name)
        for cam_name, should_rectify in right_rectify.items():
            key = f"right_{cam_name}"
            if should_rectify and key in observation:
                observation[key] = camera_calibration.rectify_image(observation[key], cam_name)
        return observation

    robot.get_observation = patched_get_observation


def patch_dual_wrist_offset(teleop: BiSOLeader, left_ctrl: SO101Control, right_ctrl: SO101Control) -> None:
    original_get_action = teleop.get_action

    def patched_get_action():
        action = original_get_action()

        left_vals = np.array([action[f"left_{n}.pos"] for n in left_ctrl.JOINT_NAMES], dtype=np.float32)
        left_vals = left_ctrl.apply_wrist_roll_offset(left_vals)
        for i, n in enumerate(left_ctrl.JOINT_NAMES):
            action[f"left_{n}.pos"] = float(left_vals[i])

        right_vals = np.array([action[f"right_{n}.pos"] for n in right_ctrl.JOINT_NAMES], dtype=np.float32)
        right_vals = right_ctrl.apply_wrist_roll_offset(right_vals)
        for i, n in enumerate(right_ctrl.JOINT_NAMES):
            action[f"right_{n}.pos"] = float(right_vals[i])

        return action

    teleop.get_action = patched_get_action


def _bootstrap_bimanual_calibration_files(
    calibration_dir: Path,
    follower_id: str,
    leader_id: str,
    left_robot_name: str,
    right_robot_name: str,
    left_leader_name: str,
    right_leader_name: str,
) -> None:
    """
    Reuse existing single-arm calibration files (e.g. robot_1.json, robot_7.json)
    by copying them to bimanual expected names (e.g. dual_follower_left.json).
    """
    mapping = [
        (calibration_dir / f"{left_robot_name}.json", calibration_dir / f"{follower_id}_left.json"),
        (calibration_dir / f"{right_robot_name}.json", calibration_dir / f"{follower_id}_right.json"),
        (calibration_dir / f"{left_leader_name}.json", calibration_dir / f"{leader_id}_left.json"),
        (calibration_dir / f"{right_leader_name}.json", calibration_dir / f"{leader_id}_right.json"),
    ]

    for src, dst in mapping:
        if dst.exists():
            continue
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Bootstrapped calibration: {src.name} -> {dst.name}")


def _resolve_dataset_settings() -> dict:
    hf_user = HF_USER
    hf_repo_name = HF_REPO_ID
    repo_id = hf_repo_name
    if "/" in repo_id:
        full_repo_id = repo_id
        local_name = repo_id.split("/")[-1]
    else:
        full_repo_id = f"{hf_user}/{repo_id}"
        local_name = repo_id

    return {
        "repo_id": full_repo_id,
        "local_name": local_name,
        "task": TASK_DESCRIPTION,
        "fps": FPS,
        "num_episodes": NUM_EPISODES,
        "episode_time_s": EPISODE_TIME_SEC,
        "reset_time_s": RESET_TIME_SEC,
        "resume": RESUME_DATASET,
        "video_codec": VIDEO_CODEC,
        "calibrate": CALIBRATE,
        "skip_recording": SKIP_RECORDING,
        "push": PUSH_TO_HUB,
        "display_data": DISPLAY_DATA,
        "image_writer_threads": IMAGE_WRITER_THREADS,
        "streaming_encoding": STREAMING_ENCODING,
        "batch_encoding_size": BATCH_ENCODING_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record dual-arm dataset and store both dual joints + dual EEF.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DUAL_CONFIG, help="Path to dual robot config.")
    args = parser.parse_args()

    try:
        cfg = _load_dual_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading dual config: {exc}")
        sys.exit(1)

    settings = _resolve_dataset_settings()
    data_dir = DATASETS_OUTPUT_DIR / settings["local_name"]

    left_cfg = cfg["robot_a"]
    right_cfg = cfg["robot_b"]
    left_leader_cfg = cfg["leader_a"]
    right_leader_cfg = cfg["leader_b"]

    left_cameras, left_rectify = _camera_map(left_cfg)
    right_cameras, right_rectify = _camera_map(right_cfg)

    calibration_left = Path(left_cfg["calibration_dir"])
    calibration_right = Path(right_cfg["calibration_dir"])
    calibration_dir = calibration_left
    if calibration_left != calibration_right:
        print(
            "Warning: BiSOFollower/BiSOLeader use one shared calibration_dir. "
            f"Using robot_a calibration dir: {calibration_left}"
        )

    left_ctrl = SO101Control(
        urdf_path=left_cfg["urdf_path"],
        wrist_roll_offset=float(left_cfg.get("wrist_roll_offset", 0.0)),
        home_pose=left_cfg.get("home_pose"),
    )
    right_ctrl = SO101Control(
        urdf_path=right_cfg["urdf_path"],
        wrist_roll_offset=float(right_cfg.get("wrist_roll_offset", 0.0)),
        home_pose=right_cfg.get("home_pose"),
    )

    robot_config = BiSOFollowerConfig(
        id=cfg["robot_id"],
        calibration_dir=calibration_dir,
        left_arm_config=SOFollowerConfig(port=left_cfg["port"], cameras=left_cameras),
        right_arm_config=SOFollowerConfig(port=right_cfg["port"], cameras=right_cameras),
    )
    teleop_config = BiSOLeaderConfig(
        id=cfg["leader_id"],
        calibration_dir=calibration_dir,
        left_arm_config=SOLeaderConfig(port=left_leader_cfg["port"]),
        right_arm_config=SOLeaderConfig(port=right_leader_cfg["port"]),
    )

    # Reuse existing single-arm calibration files when available.
    _bootstrap_bimanual_calibration_files(
        calibration_dir=calibration_dir,
        follower_id=robot_config.id,
        leader_id=teleop_config.id,
        left_robot_name=left_cfg["name"],
        right_robot_name=right_cfg["name"],
        left_leader_name=left_leader_cfg["name"],
        right_leader_name=right_leader_cfg["name"],
    )

    if settings["calibrate"]:
        follower_id = robot_config.id or "dual_follower"
        leader_id = teleop_config.id or "dual_leader"
        for arm in ("left", "right"):
            for dev in (follower_id, leader_id):
                p = calibration_dir / f"{dev}_{arm}.json"
                if p.exists():
                    p.unlink()
                    print(f"Removed calibration file: {p}")

    robot = BiSOFollower(robot_config)
    teleop = BiSOLeader(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    should_resume = settings["resume"] and has_local_lerobot_dataset(data_dir)
    if should_resume:
        print(f"Resuming dataset from {data_dir}")
        revert_dataset_to_dual_joints(data_dir)
        dataset = LeRobotDataset(
            repo_id=settings["repo_id"],
            root=data_dir,
            vcodec=settings["video_codec"],
            streaming_encoding=settings["streaming_encoding"],
            batch_encoding_size=settings["batch_encoding_size"],
        )
        dataset.start_image_writer(num_threads=settings["image_writer_threads"])
        episode_idx = dataset.meta.total_episodes
    else:
        if settings["resume"] and data_dir.exists() and not has_local_lerobot_dataset(data_dir):
            archived = data_dir.parent / f"{data_dir.name}_incomplete_backup"
            if archived.exists():
                shutil.rmtree(archived)
            print(f"Archiving non-resumable dataset dir '{data_dir}' -> '{archived}'")
            shutil.move(str(data_dir), str(archived))

        dataset = LeRobotDataset.create(
            repo_id=settings["repo_id"],
            fps=settings["fps"],
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            vcodec=settings["video_codec"],
            streaming_encoding=settings["streaming_encoding"],
            batch_encoding_size=settings["batch_encoding_size"],
            image_writer_threads=settings["image_writer_threads"],
            root=data_dir,
        )
        episode_idx = 0

    if settings["skip_recording"]:
        print("Skipping recording as requested.")
    else:
        if settings["display_data"]:
            init_rerun(session_name="recording_dual")
        listener, events = init_keyboard_listener()
        robot_connected = False
        teleop_connected = False
        try:
            robot.connect(calibrate=settings["calibrate"])
            robot_connected = True
            teleop.connect(calibrate=settings["calibrate"])
            teleop_connected = True
            patch_robot_for_rectification(robot, left_rectify, right_rectify)
            patch_dual_wrist_offset(teleop, left_ctrl, right_ctrl)

            teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

            while episode_idx < settings["num_episodes"] and not events["stop_recording"]:
                print("\n" + "█" * 60)
                print(f"█  EPISODE {episode_idx + 1} / {settings['num_episodes']}")
                print("█" * 60 + "\n")
                log_say(f"Recording episode {episode_idx + 1} of {settings['num_episodes']}")

                record_loop_teleop(
                    robot=robot,
                    events=events,
                    fps=settings["fps"],
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=settings["episode_time_s"],
                    single_task=settings["task"],
                    display_data=settings["display_data"],
                )

                if not events["stop_recording"] and (
                    episode_idx < settings["num_episodes"] - 1 or events["rerecord_episode"]
                ):
                    log_say("Reset the environment")
                    record_loop_teleop(
                        robot=robot,
                        events=events,
                        fps=settings["fps"],
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=settings["reset_time_s"],
                        single_task="RESET: Move robot to start position",
                        display_data=settings["display_data"],
                    )
                    log_say("Reset complete")

                if events["rerecord_episode"]:
                    log_say("Re-recording episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode(parallel_encoding=True)
                episode_idx += 1
        finally:
            log_say("Stop recording")
            if teleop_connected:
                teleop.disconnect()
            if robot_connected:
                robot.disconnect()
            if listener is not None:
                listener.stop()

    dataset.finalize()
    transform_dataset_to_dual_eef(Path(dataset.root), left_ctrl, right_ctrl)

    if settings["push"]:
        print(f"Pushing {settings['repo_id']} to Hugging Face Hub...")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=settings["repo_id"], repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=Path(dataset.root),
            repo_id=settings["repo_id"],
            repo_type="dataset",
            ignore_patterns=[".cache/**", "**/.cache/**", ".DS_Store", "**/.DS_Store"],
            delete_patterns=["data/**", "meta/**", "videos/**", "images/**", "tmp*", "tmp*/**"],
            commit_message="Sync dual-arm dataset",
        )
        print("Push complete.")


if __name__ == "__main__":
    main()
