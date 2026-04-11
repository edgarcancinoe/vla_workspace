#!/usr/bin/env python3
"""
Records robot teleoperation episodes and converts joint positions to
10D end-effector representations (xyz + rot6d + gripper).

The script records both EEF and joint data in separate columns:
  observation.state           -> 10D EEF (xyz + rot6d + gripper)
  action                      -> 10D EEF
  observation.joint_positions -> 6D original motor positions
  action_joints               -> 6D original motor actions

This format allows training on either EEF or joint space by selecting 
the appropriate columns.
"""

import argparse
import atexit
import json
import os
import select
import shutil
import sys
import termios
import threading
import tty
import uuid
import numpy as np
import pandas as pd
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import yaml
from tqdm import tqdm
from thesis_vla.common.paths import DATASETS_OUTPUT_DIR, ROBOT_CONFIG_PATH

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, load_episodes, update_chunk_file_indices
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import get_video_duration_in_s
from lerobot.policies.xvla.utils import mat_to_rotate6d
from lerobot.processor import make_default_processors
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader.config_so_leader import SO100LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO100Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say

from thesis_vla.robot.so101_control import SO101Control
from thesis_vla.vision import camera_calibration

# ----------------------------- Configuration -----------------------------
config_path = ROBOT_CONFIG_PATH
with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

# Store camera config and rectification flags
CAMERA_CONFIG_MAP = config_data.get("cameras", {})
RECTIFY_MAP = {name: info.get("rectify", False) for name, info in CAMERA_CONFIG_MAP.items()}

NUM_EPISODES = 96

FPS = 30
EPISODE_TIME_SEC = 45
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick up blue cube and place inside white box."
VIDEO_CODEC = "h264"
STREAMING_ENCODING = True
BATCH_VIDEO_ENCODING = True
ENCODER_QUEUE_MAXSIZE = 120
ENCODER_THREADS = 8

HF_USER = "edgarcancinoe"
HF_REPO_ID = "soarm101_pickplace_multicolor_v1"

START_FROM_SCRATCH = False
RESUME_DATASET = True

URDF_PATH = config_data.get("robot", {}).get("urdf_path")
if not URDF_PATH:
    raise ValueError("Error: 'urdf_path' not found in config/robot_config.yaml.")
EEF_FEATURE_NAMES = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper"]
MAX_GRIPPER_DEG = 28.0

DATA_DIR = DATASETS_OUTPUT_DIR / HF_REPO_ID

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

    meta_lengths: dict[int, int] = {}
    for _, row in df_ep.iterrows():
        meta_lengths[int(row["episode_index"])] = int(row["length"])

    data_lengths: dict[int, int] = {}
    data_root = root / "data"
    if not data_root.exists():
        raise RuntimeError(f"Missing data directory: {data_root}")
    data_parquets = sorted(data_root.rglob("*.parquet"))
    if not data_parquets:
        raise RuntimeError(f"No data parquet files found under {data_root}")

    total_data_rows = 0
    for parquet_path in data_parquets:
        df_data = pq.read_table(parquet_path, columns=["episode_index"]).to_pandas()
        total_data_rows += len(df_data)
        counts = df_data["episode_index"].value_counts().to_dict()
        for ep_idx, count in counts.items():
            ep_idx_int = int(ep_idx)
            data_lengths[ep_idx_int] = data_lengths.get(ep_idx_int, 0) + int(count)

    length_mismatches = []
    for ep_idx in expected:
        data_len = data_lengths.get(ep_idx)
        meta_len = meta_lengths.get(ep_idx)
        if data_len != meta_len:
            length_mismatches.append((ep_idx, data_len, meta_len))

    if length_mismatches:
        raise RuntimeError(
            "Data parquet rows do not match meta/episodes lengths: "
            f"{length_mismatches}. Remove or repair those episodes before pushing."
        )

    total_frames = info.get("total_frames")
    if total_frames is None:
        raise RuntimeError("info.json missing key 'total_frames'")
    if int(total_frames) != total_data_rows:
        raise RuntimeError(
            f"info.json total_frames={total_frames} but data parquets contain {total_data_rows} rows"
        )

    video_keys = [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]
    missing_video_metadata = []
    missing_video_files = []
    for _, row in df_ep.iterrows():
        ep_idx = int(row["episode_index"])
        for video_key in video_keys:
            chunk_col = f"videos/{video_key}/chunk_index"
            file_col = f"videos/{video_key}/file_index"
            if chunk_col not in row.index or file_col not in row.index or _is_missing(row[chunk_col]) or _is_missing(row[file_col]):
                missing_video_metadata.append((ep_idx, video_key))
                continue

            video_path = (
                root
                / "videos"
                / video_key
                / f"chunk-{int(row[chunk_col]):03d}"
                / f"file-{int(row[file_col]):03d}.mp4"
            )
            if not video_path.exists():
                missing_video_files.append((ep_idx, video_key, video_path))

    if missing_video_metadata:
        raise RuntimeError(
            "Missing video metadata for episodes/cameras: "
            f"{missing_video_metadata}. Remove or re-record those episodes before pushing."
        )
    if missing_video_files:
        raise RuntimeError(
            "Missing referenced video files: "
            f"{[(ep, key, str(path)) for ep, key, path in missing_video_files]}"
        )
    print(f"[OK] Dataset on disk looks sane: episodes={n}, rows={rows}")


def has_local_lerobot_dataset(root: Path) -> bool:
    """Return True only when root has enough LeRobot metadata to resume."""
    root = Path(root)
    return (root / "meta" / "info.json").exists() and (root / "meta" / "episodes").exists()


class TerminalKeyboardListener:
    """Fallback keyboard listener for terminals where pynput cannot see arrow keys."""

    def __init__(self, events: dict):
        self.events = events
        self._fd: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_termios = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            print("Terminal keyboard fallback disabled: stdin is not a TTY.", flush=True)
            return False

        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self.stop)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            self._old_termios = None

    def _read_char(self, timeout_s: float = 0.05) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([self._fd], [], [], timeout_s)
        if not readable:
            return None
        return os.read(self._fd, 1).decode(errors="ignore")

    def _read_escape_sequence(self) -> str:
        sequence = "\x1b"
        for _ in range(8):
            char = self._read_char(timeout_s=0.15)
            if char is None:
                break
            sequence += char
            if char.isalpha() or char == "~":
                break
        return sequence

    def _handle_key(self, key: str) -> None:
        if key in ("\x1b[C", "\x1bOC") or key.endswith("C"):
            print("Right arrow key pressed. Exiting loop...", flush=True)
            self.events["exit_early"] = True
        elif key in ("\x1b[D", "\x1bOD") or key.endswith("D"):
            print("Left arrow key pressed. Re-recording episode...", flush=True)
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif key in ("\x1b", "q", "Q"):
            print("Stop key pressed. Stopping...", flush=True)
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
        elif key in ("r", "R"):
            print("r pressed. Re-recording episode...", flush=True)
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif key in ("\n", "\r", " "):
            print("Skip key pressed. Exiting loop...", flush=True)
            self.events["exit_early"] = True

    def _run(self) -> None:
        while not self._stop.is_set():
            char = self._read_char()
            if char is None:
                continue

            if char == "\x1b":
                self._handle_key(self._read_escape_sequence())
            else:
                self._handle_key(char)


def init_recording_keyboard_controls() -> tuple[object | None, dict, TerminalKeyboardListener | None]:
    listener, events = init_keyboard_listener()
    terminal_listener = TerminalKeyboardListener(events)
    if terminal_listener.start():
        print(
            "Keyboard controls enabled: right/space/enter=next, left/r=re-record, esc/q=stop. "
            "Keep this terminal focused if arrow keys are not captured globally."
        )
    else:
        terminal_listener = None
    return listener, events, terminal_listener


def init_fresh_rerun(session_name: str = "recording") -> None:
    """Start a fresh Rerun recording so old streams/timelines are not reused."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    run_id = uuid.uuid4().hex
    rr.init(f"{session_name}_{run_id[:8]}", recording_id=run_id)
    rr.reset_time()
    rr.spawn(memory_limit=memory_limit, hide_welcome_screen=True)
    print(f"Started fresh Rerun recording: {run_id}", flush=True)


def encode_pending_videos(dataset: LeRobotDataset) -> None:
    """Encode any batched videos that were intentionally deferred during recording."""
    dataset.meta._close_writer()
    dataset.meta.episodes = load_episodes(dataset.root)

    missing_episodes = get_episodes_missing_video_metadata(dataset)
    if missing_episodes:
        # log_say(f"Encoding missing videos for episodes {missing_episodes}")
        encode_missing_video_metadata(dataset, missing_episodes)

    pending = getattr(dataset, "episodes_since_last_encoding", 0)
    if pending > 0:
        end_episode = dataset.meta.total_episodes
        start_episode = end_episode - pending
        log_say(f"Encoding videos for episodes {start_episode} to {end_episode - 1}")
        encode_missing_video_metadata(dataset, list(range(start_episode, end_episode)))
        dataset.episodes_since_last_encoding = 0


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def get_episodes_missing_video_metadata(dataset: LeRobotDataset) -> list[int]:
    missing = []
    video_keys = list(dataset.meta.video_keys)
    if not video_keys or dataset.meta.episodes is None:
        return missing

    for ep in dataset.meta.episodes:
        ep_idx = int(ep["episode_index"])
        for video_key in video_keys:
            key = f"videos/{video_key}/chunk_index"
            if key not in ep or _is_missing(ep[key]):
                missing.append(ep_idx)
                break
    return missing


def _episode_image_dir(dataset: LeRobotDataset, video_key: str, ep_idx: int) -> Path:
    return dataset.root / "images" / video_key / f"episode-{ep_idx:06d}"


def _episode_has_video_images(dataset: LeRobotDataset, video_key: str, ep_idx: int) -> bool:
    image_dir = _episode_image_dir(dataset, video_key, ep_idx)
    return image_dir.exists() and any(image_dir.glob("*.png"))


def _missing_image_keys(dataset: LeRobotDataset, ep_idx: int) -> list[str]:
    return [
        video_key
        for video_key in dataset.meta.video_keys
        if not _episode_has_video_images(dataset, video_key, ep_idx)
    ]


def get_complete_video_episodes(root: Path) -> list[int] | None:
    info_path = root / "meta" / "info.json"
    episodes_dir = root / "meta" / "episodes"
    if not info_path.exists() or not episodes_dir.exists():
        return None

    with open(info_path) as f:
        info = json.load(f)

    video_keys = [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]
    if not video_keys:
        return None

    complete = []
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            if all(
                f"videos/{video_key}/chunk_index" in row.index
                and not _is_missing(row[f"videos/{video_key}/chunk_index"])
                for video_key in video_keys
            ):
                complete.append(int(row["episode_index"]))
    return complete


def _write_episode_video_metadata(dataset: LeRobotDataset, ep_idx: int, metadata: dict) -> None:
    episode_df_path = None
    for parquet_path in sorted((dataset.root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index"])
        if ep_idx in set(df["episode_index"].astype(int).tolist()):
            episode_df_path = parquet_path
            break

    if episode_df_path is None:
        raise RuntimeError(f"Could not find meta/episodes row for episode_index={ep_idx}")

    episode_df = pd.read_parquet(episode_df_path)
    row_mask = episode_df["episode_index"] == ep_idx
    for key, value in metadata.items():
        if key == "episode_index":
            continue
        if key not in episode_df.columns:
            episode_df[key] = pd.NA
        episode_df.loc[row_mask, key] = value
    episode_df.to_parquet(episode_df_path)
    dataset.meta.episodes = load_episodes(dataset.root)


def _next_video_file_indices(dataset: LeRobotDataset, video_key: str) -> tuple[int, int]:
    last_chunk_idx = 0
    last_file_idx = -1
    key_chunk = f"videos/{video_key}/chunk_index"
    key_file = f"videos/{video_key}/file_index"

    if dataset.meta.episodes is not None:
        for ep in dataset.meta.episodes:
            if key_chunk not in ep or key_file not in ep:
                continue
            chunk_idx = ep[key_chunk]
            file_idx = ep[key_file]
            if _is_missing(chunk_idx) or _is_missing(file_idx):
                continue
            chunk_idx = int(chunk_idx)
            file_idx = int(file_idx)
            if (chunk_idx, file_idx) > (last_chunk_idx, last_file_idx):
                last_chunk_idx, last_file_idx = chunk_idx, file_idx

    video_root = dataset.root / "videos" / video_key
    if video_root.exists():
        for path in video_root.glob("chunk-*/file-*.mp4"):
            try:
                chunk_idx = int(path.parent.name.removeprefix("chunk-"))
                file_idx = int(path.stem.removeprefix("file-"))
            except ValueError:
                continue
            if (chunk_idx, file_idx) > (last_chunk_idx, last_file_idx):
                last_chunk_idx, last_file_idx = chunk_idx, file_idx

    if last_file_idx < 0:
        return 0, 0
    return update_chunk_file_indices(last_chunk_idx, last_file_idx, dataset.meta.chunks_size)


def _save_episode_video_as_new_file(dataset: LeRobotDataset, video_key: str, ep_idx: int) -> dict:
    ep_path = dataset._encode_temporary_episode_video(video_key, ep_idx)
    ep_duration_in_s = get_video_duration_in_s(ep_path)
    chunk_idx, file_idx = _next_video_file_indices(dataset, video_key)
    new_path = dataset.root / dataset.meta.video_path.format(
        video_key=video_key,
        chunk_index=chunk_idx,
        file_index=file_idx,
    )
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ep_path), str(new_path))
    shutil.rmtree(str(ep_path.parent))
    return {
        "episode_index": ep_idx,
        f"videos/{video_key}/chunk_index": chunk_idx,
        f"videos/{video_key}/file_index": file_idx,
        f"videos/{video_key}/from_timestamp": 0.0,
        f"videos/{video_key}/to_timestamp": ep_duration_in_s,
    }


def encode_missing_video_metadata(dataset: LeRobotDataset, episode_indices: list[int]) -> None:
    for ep_idx in sorted(set(episode_indices)):
        missing_image_keys = _missing_image_keys(dataset, ep_idx)
        if missing_image_keys:
            print(
                f"Skipping video repair for episode {ep_idx}: missing image frames for "
                f"{missing_image_keys}."
            )
            continue

        metadata = {"episode_index": ep_idx}
        for video_key in dataset.meta.video_keys:
            metadata.update(_save_episode_video_as_new_file(dataset, video_key, ep_idx))
        _write_episode_video_metadata(dataset, ep_idx, metadata)


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


# ─── EEF transformation (converts 6D joints → 10D EEF) ───────────────

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


def transform_dataset_to_eef(root: Path, so101: SO101Control) -> None:
    """
    Rewrites every parquet under root/data in-place, adding:
      observation.state  (10D EEF: xyz + rot6d + gripper)
      action             (10D EEF)
      observation.joint_positions (6D original joints)
      action_joints               (6D original joints)

    Also updates meta/info.json, meta/features.json, and meta/stats.json.
    """
    root = Path(root)
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    global_eef_states: list = []
    global_eef_actions: list = []
    global_joint_states: list = []
    global_joint_actions: list = []

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
        joint_states_6d: list = []
        joint_actions_6d: list = []
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
                joint_states_6d.append(np.array(state_joints, dtype=np.float32))
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
                    joint_actions_6d.append(np.array(action_joints, dtype=np.float32))
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

        # 6D original joint positions
        if joint_states_6d:
            jp_arr = np.array(joint_states_6d, dtype=np.float32).flatten()
            data_out["observation.joint_positions"] = pa.FixedSizeListArray.from_arrays(jp_arr, 6)
            schema_fields.append(pa.field("observation.joint_positions", pa.list_(pa.float32(), 6)))

        # 6D original joint actions
        if has_action and joint_actions_6d:
            ja_arr = np.array(joint_actions_6d, dtype=np.float32).flatten()
            data_out["action_joints"] = pa.FixedSizeListArray.from_arrays(ja_arr, 6)
            schema_fields.append(pa.field("action_joints", pa.list_(pa.float32(), 6)))

        # Scalar columns (timestamps, indices, etc.)
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in df.columns:
                data_out[col] = pa.array(df[col])
                schema_fields.append(pa.field(col, table.schema.field(col).type))

        pq.write_table(pa.Table.from_pydict(data_out, schema=pa.schema(schema_fields)), parquet_path)

        global_eef_states.extend(eef_states)
        global_joint_states.extend(joint_states_6d)
        if has_action:
            global_eef_actions.extend(eef_actions)
            global_joint_actions.extend(joint_actions_6d)

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
            # Add 6D joint feature declarations
            joint_names_6d = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            info_dict["features"]["observation.joint_positions"] = {
                "dtype": "float32", "shape": [6], "names": joint_names_6d
            }
            info_dict["features"]["action_joints"] = {
                "dtype": "float32", "shape": [6], "names": joint_names_6d
            }
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
        # Add 6D joint feature declarations
        joint_names_6d = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        feat_dict["observation.joint_positions"] = {
            "dtype": "float32", "shape": [6], "names": joint_names_6d
        }
        feat_dict["action_joints"] = {
            "dtype": "float32", "shape": [6], "names": joint_names_6d
        }
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
        if global_joint_states:
            stats_dict["observation.joint_positions"] = _compute_stats(global_joint_states)
        if global_joint_actions:
            stats_dict["action_joints"] = _compute_stats(global_joint_actions)
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

    # 2. Update info.json — restore original LeRobot feature names (.pos suffix)
    MOTOR_NAMES_6D = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    if "observation.joint_positions" in info_dict["features"]:
        info_dict["features"]["observation.state"] = info_dict["features"].pop("observation.joint_positions")
        info_dict["features"]["observation.state"]["names"] = MOTOR_NAMES_6D
        info_dict["features"]["observation.state"]["shape"] = [6]
    if "action_joints" in info_dict["features"]:
        info_dict["features"]["action"] = info_dict["features"].pop("action_joints")
        info_dict["features"]["action"]["names"] = MOTOR_NAMES_6D
        info_dict["features"]["action"]["shape"] = [6]
    with open(info_path, "w") as f:
        json.dump(info_dict, f, indent=4)

    # 3. Update features.json
    features_path = meta_dir / "features.json"
    if features_path.exists():
        with open(features_path, "r") as f:
            feat_dict = json.load(f)
        if "observation.joint_positions" in feat_dict:
            feat_dict["observation.state"] = feat_dict.pop("observation.joint_positions")
            feat_dict["observation.state"]["names"] = MOTOR_NAMES_6D
            feat_dict["observation.state"]["shape"] = [6]
        if "action_joints" in feat_dict:
            feat_dict["action"] = feat_dict.pop("action_joints")
            feat_dict["action"]["names"] = MOTOR_NAMES_6D
            feat_dict["action"]["shape"] = [6]
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
    parser.add_argument(
        "--skip-recording",
        action="store_true",
        help="Do not record new episodes; only repair videos, transform metadata/data, and optionally push.",
    )
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

    should_resume = RESUME_DATASET and has_local_lerobot_dataset(DATA_DIR)
    if RESUME_DATASET and not should_resume:
        print(
            f"Resume requested, but no complete local dataset exists at {DATA_DIR}. "
            "Creating a new dataset instead."
        )

    if should_resume:
        print(f"Resuming dataset from {DATA_DIR}")
        revert_dataset_to_6d(DATA_DIR)
        complete_video_episodes = get_complete_video_episodes(DATA_DIR)
        if complete_video_episodes is not None:
            print(
                "Loading only episodes with complete video metadata for resume validation: "
                f"{len(complete_video_episodes)} / existing episodes."
            )
        dataset = LeRobotDataset(
            repo_id=f"{HF_USER}/{HF_REPO_ID}",
            root=DATA_DIR,
            episodes=complete_video_episodes,
            vcodec=VIDEO_CODEC,
            streaming_encoding=STREAMING_ENCODING,
            batch_encoding_size=NUM_EPISODES + 1 if BATCH_VIDEO_ENCODING else 1,
            encoder_queue_maxsize=ENCODER_QUEUE_MAXSIZE,
            encoder_threads=ENCODER_THREADS,
        )
        dataset.start_image_writer(num_threads=4)
        episode_idx = dataset.meta.total_episodes
    else:
        dataset = LeRobotDataset.create(
            repo_id=f"{HF_USER}/{HF_REPO_ID}",
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            vcodec=VIDEO_CODEC,
            streaming_encoding=STREAMING_ENCODING,
            batch_encoding_size=NUM_EPISODES + 1 if BATCH_VIDEO_ENCODING else 1,
            encoder_queue_maxsize=ENCODER_QUEUE_MAXSIZE,
            encoder_threads=ENCODER_THREADS,
            image_writer_threads=4,
            root=DATA_DIR,
        )
        episode_idx = 0

    if args.skip_recording:
        print(
            f"Skipping recording. Dataset currently has {dataset.meta.total_episodes} episodes; "
            f"NUM_EPISODES is {NUM_EPISODES}."
        )
    else:
        init_fresh_rerun(session_name="recording")
        _, events, terminal_keyboard_listener = init_recording_keyboard_controls()

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
            
            # Enforce maximum gripper position (consistency for fully open state)
            if "gripper" in so101.JOINT_NAMES:
                gripper_idx = so101.JOINT_NAMES.index("gripper")
                # Convert degrees to motor units (0-100 range)
                max_gripper_motor = so101.deg_to_motor(np.full(len(so101.JOINT_NAMES), MAX_GRIPPER_DEG))[gripper_idx]
                motor_vals[gripper_idx] = np.clip(motor_vals[gripper_idx], 0.0, max_gripper_motor)

            for i, n in enumerate(so101.JOINT_NAMES):
                action[f"{n}.pos"] = float(motor_vals[i])
            return action

        teleop.get_action = patched_get_action
        # -----------------------------------------------------------------------

        teleop_action_processor, robot_action_processor, robot_observation_processor = (make_default_processors())

        # --- Recording loop -----------------------------------------------------
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            print("\n" + "█" * 60)
            print(f"█  EPISODE {episode_idx + 1} / {NUM_EPISODES}")
            print("█" * 60 + "\n")
            
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

            dataset.save_episode(parallel_encoding=True)
            episode_idx += 1

        log_say("Stop recording")
        if terminal_keyboard_listener is not None:
            terminal_keyboard_listener.stop()
        robot.disconnect()
        teleop.disconnect()
    encode_pending_videos(dataset)
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
        api.upload_folder(
            folder_path=DATA_DIR,
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=[
                ".cache/**",
                "**/.cache/**",
                ".DS_Store",
                "**/.DS_Store",
            ],
            delete_patterns=[
                "data/**",
                "meta/**",
                "videos/**",
                "images/**",
                "tmp*",
                "tmp*/**",
            ],
            commit_message="Sync dataset from clean local source",
        )
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
