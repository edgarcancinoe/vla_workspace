from __future__ import annotations

import json
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_episodes, update_chunk_file_indices
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.policies.xvla.utils import mat_to_rotate6d

from thesis_vla.vision import camera_calibration


class DBG:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    _C = {"cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m", "magenta": "\033[95m", "blue": "\033[94m", "white": "\033[97m", "orange": "\033[38;5;214m", "teal": "\033[38;5;86m", "pink": "\033[38;5;213m"}

    @classmethod
    def _fmt(cls, color: str, tag: str, msg: str) -> str:
        return f"{cls.BOLD}{cls._C.get(color, '')}[{tag}]{cls.RESET} {msg}"

    @classmethod
    def obs(cls, msg: str): print(cls._fmt("cyan", "OBS", msg))

    @classmethod
    def act(cls, msg: str): print(cls._fmt("green", "ACT", msg))

    @classmethod
    def ik(cls, msg: str): print(cls._fmt("orange", "IK", msg))

    @classmethod
    def grip(cls, msg: str): print(cls._fmt("pink", "GRIPPER", msg))

    @classmethod
    def send(cls, msg: str): print(cls._fmt("yellow", "SEND", msg))

    @classmethod
    def warn(cls, msg: str): print(cls._fmt("red", "WARN", msg))

    @classmethod
    def divider(cls, color: str = "white", label: str = ""):
        line = "─" * 64
        if label:
            print(f"{cls._C.get(color, '')}{cls.BOLD}┌{line}\n│  {label}\n└{line}{cls.RESET}")
        else:
            print(f"{cls._C.get(color, '')}{line}{cls.RESET}")


@dataclass
class ExecutionSmoothingConfig:
    enable_ema_smoothing: bool = False
    ema_alpha_motor: float = 0.35
    ema_alpha_gripper: float = 0.25
    ema_reset_on_episode_start: bool = True
    enable_interpolation: bool = False
    interp_substeps: int = 3
    interp_min_delta_motor: float = 0.0
    interp_apply_to_gripper: bool = False
    interp_keep_control_fps_budget: bool = True
    interp_min_sleep_s: float = 0.0005


@dataclass
class DatasetPushConfig:
    max_retries: int = 3
    retry_backoff_s: float = 3.0
    ignore_patterns: list[str] = field(default_factory=lambda: [".cache/**", "**/.cache/**", ".DS_Store", "**/.DS_Store"])
    delete_patterns: list[str] = field(default_factory=lambda: ["data/**", "meta/**", "videos/**", "images/**", "tmp*", "tmp*/**"])


def init_meshcat(urdf_path: str | os.PathLike[str]):
    try:
        from thesis_vla.sim.so101_meshcat import SO101Meshcat

        viz = SO101Meshcat(urdf_path=urdf_path)
        print("[x] Meshcat visualizer started.")
        return viz
    except Exception as e:
        print(f"[!] Meshcat init failed, continuing without visualization: {e}")
        return None


class TerminalKeyboardListener:
    def __init__(self, events: dict):
        self.events = events
        self._fd: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_termios = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            self._fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self.stop()
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None
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
            self._handle_key(self._read_escape_sequence()) if char == "\x1b" else self._handle_key(char)


def init_eval_keyboard_controls() -> tuple[object | None, dict, TerminalKeyboardListener | None]:
    listener, events = init_keyboard_listener()
    terminal_listener = TerminalKeyboardListener(events)
    if terminal_listener.start():
        print("Keyboard controls enabled: right/space/enter=next, left/r=re-record, esc/q=stop. Keep this terminal focused if arrow keys are not captured globally.")
    else:
        terminal_listener = None
    return listener, events, terminal_listener


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
    return [video_key for video_key in dataset.meta.video_keys if not _episode_has_video_images(dataset, video_key, ep_idx)]


def _write_episode_video_metadata(dataset: LeRobotDataset, ep_idx: int, metadata: dict) -> None:
    for parquet_path in sorted((dataset.root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        mask = df["episode_index"] == ep_idx
        if not mask.any():
            continue
        for key, value in metadata.items():
            if key != "episode_index":
                df.loc[mask, key] = value
        df.to_parquet(parquet_path)
    dataset.meta.episodes = load_episodes(dataset.root)


def _next_video_file_indices(dataset: LeRobotDataset, video_key: str) -> tuple[int, int]:
    last_chunk_idx = -1
    last_file_idx = -1
    if dataset.meta.episodes is not None:
        for ep in dataset.meta.episodes:
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            if chunk_key in ep and file_key in ep and not _is_missing(ep[chunk_key]) and not _is_missing(ep[file_key]):
                ck, fk = int(ep[chunk_key]), int(ep[file_key])
                if (ck, fk) > (last_chunk_idx, last_file_idx):
                    last_chunk_idx, last_file_idx = ck, fk
    video_root = dataset.root / "videos" / video_key
    if video_root.exists():
        for mp4 in video_root.rglob("file-*.mp4"):
            chunk_part, file_part = mp4.parent.name, mp4.stem
            if chunk_part.startswith("chunk-") and file_part.startswith("file-"):
                ck, fk = int(chunk_part.split("-")[1]), int(file_part.split("-")[1])
                if (ck, fk) > (last_chunk_idx, last_file_idx):
                    last_chunk_idx, last_file_idx = ck, fk
    return (0, 0) if last_chunk_idx < 0 else update_chunk_file_indices(last_chunk_idx, last_file_idx, dataset.meta.chunks_size)


def _save_episode_video_as_new_file(dataset: LeRobotDataset, video_key: str, ep_idx: int) -> dict:
    ep_path = dataset._encode_temporary_episode_video(video_key, ep_idx)
    chunk_idx, file_idx = _next_video_file_indices(dataset, video_key)
    new_path = dataset.root / dataset.meta.video_path.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(ep_path, new_path)
    try:
        os.rmdir(ep_path.parent)
    except OSError:
        pass
    ep_len = 0.0
    for ep in dataset.meta.episodes or []:
        if int(ep.get("episode_index", -1)) == int(ep_idx):
            ep_len = float(ep.get("length", 0))
            break
    duration = ep_len / float(max(dataset.fps, 1))
    return {f"videos/{video_key}/chunk_index": chunk_idx, f"videos/{video_key}/file_index": file_idx, f"videos/{video_key}/from_timestamp": 0.0, f"videos/{video_key}/to_timestamp": duration}


def encode_missing_video_metadata(dataset: LeRobotDataset, episode_indices: list[int]) -> None:
    for ep_idx in episode_indices:
        missing_image_keys = _missing_image_keys(dataset, ep_idx)
        if missing_image_keys:
            raise RuntimeError(f"Cannot repair episode {ep_idx}: missing image frames for {missing_image_keys}.")
        metadata = {"episode_index": ep_idx}
        for video_key in dataset.meta.video_keys:
            metadata.update(_save_episode_video_as_new_file(dataset, video_key, ep_idx))
        _write_episode_video_metadata(dataset, ep_idx, metadata)


def recover_missing_eval_videos(dataset: LeRobotDataset) -> None:
    dataset.meta._close_writer()
    dataset.meta.episodes = load_episodes(dataset.root)
    missing_episodes = get_episodes_missing_video_metadata(dataset)
    if missing_episodes:
        print(f"[repair] Missing video metadata detected for episodes: {missing_episodes}")
        encode_missing_video_metadata(dataset, missing_episodes)
        dataset.meta.episodes = load_episodes(dataset.root)


def validate_lerobot_dataset_on_disk(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    episodes_dir = root / "meta" / "episodes"
    if not info_path.exists():
        raise RuntimeError(f"Missing metadata file: {info_path}")
    if not episodes_dir.exists():
        raise RuntimeError(f"Missing episodes directory: {episodes_dir}")
    with open(info_path) as f:
        info = json.load(f)
    episode_files = sorted(episodes_dir.rglob("*.parquet"))
    if not episode_files:
        raise RuntimeError(f"No episode parquet files found in {episodes_dir}")
    episode_df = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    if episode_df.empty:
        raise RuntimeError("Episode parquet exists but contains no rows.")
    n, rows = int(info.get("total_episodes", 0)), len(episode_df)
    if n > 0 and rows != n:
        raise RuntimeError(f"Episode count mismatch: info.total_episodes={n}, rows={rows}")
    for _, row in episode_df.iterrows():
        data_path = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
        if not data_path.exists():
            raise RuntimeError(f"Missing referenced data parquet: {data_path}")
    video_features = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    for video_key in video_features:
        chunk_col, file_col = f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index"
        if chunk_col not in episode_df.columns or file_col not in episode_df.columns:
            raise RuntimeError(f"Missing video metadata columns for key '{video_key}'")
        if episode_df[chunk_col].isna().any() or episode_df[file_col].isna().any():
            raise RuntimeError(f"Missing video metadata values for key '{video_key}'")
        for _, row in episode_df.iterrows():
            video_path = root / "videos" / video_key / f"chunk-{int(row[chunk_col]):03d}" / f"file-{int(row[file_col]):03d}.mp4"
            if not video_path.exists():
                raise RuntimeError(f"Missing referenced video file: {video_path}")
    print(f"[OK] Dataset on disk looks sane: episodes={n}, rows={rows}")


def push_eval_dataset_to_hub(data_dir: Path, repo_id: str, push_config: DatasetPushConfig | None = None) -> None:
    push_config = push_config or DatasetPushConfig()
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    last_error = None
    for attempt in range(1, push_config.max_retries + 1):
        try:
            print(f"[push] Upload attempt {attempt}/{push_config.max_retries} -> {repo_id}")
            api.upload_folder(folder_path=data_dir, repo_id=repo_id, repo_type="dataset", ignore_patterns=push_config.ignore_patterns, delete_patterns=push_config.delete_patterns, commit_message="Sync evaluation dataset from local run")
            print(f"[push] Successfully pushed {repo_id}")
            return
        except Exception as e:
            last_error = e
            print(f"[push] Attempt {attempt} failed: {e}")
            if attempt < push_config.max_retries:
                sleep_s = push_config.retry_backoff_s * attempt
                print(f"[push] Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
    raise RuntimeError(f"Failed to push dataset after {push_config.max_retries} attempts: {last_error}") from last_error


def set_custom_get_observation(
    robot,
    *,
    so101,
    rectify_map: dict[str, bool],
    active_xvla_rename_map: dict[str, str],
    eef_state_names: list[str],
    include_eef_state: bool = False,
    dry_run: bool = False,
):
    base_obs_func = robot.get_observation
    if dry_run:
        init_obs = base_obs_func()
        robot._virtual_motor = np.array([float(init_obs.get(f"{n}.pos", 0.0)) for n in so101.JOINT_NAMES])

    _obs_step = [0]

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
                    raise RuntimeError(f"Dry-run camera read failed for '{cam_key}'") from e
        else:
            observation = base_obs_func()
        if _obs_step[0] % 30 == 1:
            print(f"\033[94mDEBUG: Raw robot observation keys: {list(observation.keys())}\033[0m")
            if "wrist" in observation:
                print("\033[91m!!! DEBUG: 'wrist' key already in raw observation!\033[0m")
        for cam_name, should_rectify in rectify_map.items():
            if should_rectify and cam_name in observation:
                observation[cam_name] = camera_calibration.rectify_image(observation[cam_name], cam_name)
        if include_eef_state:
            motor_vals = np.array([observation.get(f"{joint}.pos", 0.0) for joint in so101.JOINT_NAMES])
            T = so101.fk(motor_vals)
            r6d = mat_to_rotate6d(T[:3, :3])
            eef = np.concatenate([T[:3, 3], r6d, [motor_vals[-1]]], dtype=np.float32)
            for i, name in enumerate(eef_state_names):
                observation[f"{name}.pos"] = float(eef[i])
        expected_camera_keys = {src.removeprefix("observation.images.") for src in active_xvla_rename_map}
        for cam_name in expected_camera_keys:
            if cam_name not in observation:
                raise KeyError(f"Expected camera '{cam_name}' not found in observation. Available keys: {list(observation.keys())}")
        if _obs_step[0] % 30 == 1:
            DBG.divider("cyan", f"OBS  step={_obs_step[0]}")
            scalar_keys = {k: v for k, v in observation.items() if not hasattr(v, "shape")}
            tensor_keys = {k: v for k, v in observation.items() if hasattr(v, "shape")}
            DBG.obs(f"Total obs keys: {len(observation)}  |  scalars: {len(scalar_keys)}  tensors(images): {len(tensor_keys)}")
            if scalar_keys:
                vals_str = "  ".join(f"{k.replace('.pos', '')}={float(v):+.3f}" for k, v in scalar_keys.items())
                DBG.obs(f"State: [{len(scalar_keys)}D] {vals_str}")
            for k, v in tensor_keys.items():
                DBG.obs(f"Image '{k}': shape={list(v.shape) if hasattr(v, 'shape') else '?'}")
        return observation

    print("==============================================================================================================")
    print(f"\033[94m[x] Robot observation patched for dynamic rectification based on config: {rectify_map}\033[0m")
    print(f"\033[94m[x] Robot observation patched for EEF State: {eef_state_names}\033[0m") if include_eef_state else None
    if active_xvla_rename_map:
        print(f"\033[94m[x] Robot observation keeps training camera keys for rename_map: {active_xvla_rename_map}\033[0m")
    print("==============================================================================================================")
    robot.get_observation = patched_get_observation


def set_custom_send_action(robot, *, so101, control_fps: float, smoothing: ExecutionSmoothingConfig | None = None, viz=None, dry_run: bool = False):
    smoothing = smoothing or ExecutionSmoothingConfig()
    base_action_func = robot.send_action
    gr_idx = so101.JOINT_NAMES.index("gripper")
    _step, _last_mode, _last_cmd_motor, _ema_motor_state = [0], [None], [None], [None]
    if dry_run:
        print("[!] DRY RUN enabled — robot will NOT receive any commands.")
    print("[x] Runtime smoothing config | "
        f"EMA(enabled={smoothing.enable_ema_smoothing}, alpha_motor={smoothing.ema_alpha_motor}, alpha_gripper={smoothing.ema_alpha_gripper}, reset_on_episode={smoothing.ema_reset_on_episode_start}) | "
        f"Interp(enabled={smoothing.enable_interpolation}, substeps={smoothing.interp_substeps}, min_delta={smoothing.interp_min_delta_motor}, apply_gripper={smoothing.interp_apply_to_gripper}, keep_fps_budget={smoothing.interp_keep_control_fps_budget})")

    def _viz_update(motor_vals):
        if not viz:
            return
        viz.display(so101.motor_to_rad(motor_vals))
        T = so101.fk(motor_vals)
        viz.add_axes("eef_axes", T[:3, 3], T[:3, :3], length=0.04)

    def _reset_smoothing_state(seed_motor=None):
        if seed_motor is None:
            _last_cmd_motor[0], _ema_motor_state[0] = None, None
            return
        seed = np.asarray(seed_motor, dtype=np.float64).copy()
        _last_cmd_motor[0], _ema_motor_state[0] = seed.copy(), seed.copy()

    def _apply_ema(target_motor):
        target = np.asarray(target_motor, dtype=np.float64)
        if not smoothing.enable_ema_smoothing:
            return target.copy()
        if _ema_motor_state[0] is None:
            _ema_motor_state[0] = target.copy()
            return target.copy()
        prev, smoothed = _ema_motor_state[0], _ema_motor_state[0].copy()
        for idx in range(len(smoothed)):
            alpha = smoothing.ema_alpha_gripper if idx == gr_idx else smoothing.ema_alpha_motor
            smoothed[idx] = alpha * target[idx] + (1.0 - alpha) * prev[idx]
        _ema_motor_state[0] = smoothed.copy()
        return smoothed

    def _build_interpolation_waypoints(start_motor, end_motor):
        start, end = np.asarray(start_motor, dtype=np.float64), np.asarray(end_motor, dtype=np.float64)
        if not smoothing.enable_interpolation:
            return [end]
        n_substeps = max(1, int(smoothing.interp_substeps))
        if n_substeps <= 1 or np.max(np.abs(end - start)) < float(smoothing.interp_min_delta_motor):
            return [end]
        waypoints = []
        for i in range(1, n_substeps + 1):
            alpha = i / n_substeps
            wp = start + alpha * (end - start)
            if not smoothing.interp_apply_to_gripper and i < n_substeps:
                wp[gr_idx] = start[gr_idx]
            waypoints.append(wp)
        return waypoints

    def _dispatch_motor_command(motor_vals, **kwargs):
        motor_dict = {f"{n}.pos": float(motor_vals[i]) for i, n in enumerate(so101.JOINT_NAMES)}
        if dry_run:
            _viz_update(motor_vals)
            robot._virtual_motor = np.asarray(motor_vals, dtype=np.float64).copy()
            return motor_dict
        _viz_update(motor_vals)
        return base_action_func(motor_dict, **kwargs)

    robot._xvla_reset_exec_smoothing_state = _reset_smoothing_state

    def patched_send_action(action, **kwargs):
        _step[0] += 1
        is_motor = "shoulder_pan.pos" in action
        mode = "MOTOR" if is_motor else "EEF"
        current_motor = so101.read_motor_real(robot)
        current_motor_dict = {f"{n}.pos": float(current_motor[i]) for i, n in enumerate(so101.JOINT_NAMES)}
        log_every = _step[0] % 30 == 1
        if log_every:
            DBG.divider("green", f"SEND_ACTION  step={_step[0]}  mode={mode}")
            DBG.act(f"Action dict keys ({len(action)}): {list(action.keys())}")
            for k, v in action.items():
                try:
                    DBG.act(f"  {k:25s} = {float(v):+.5f}")
                except Exception:
                    DBG.act(f"  {k:25s} = <tensor shape={list(v.shape) if hasattr(v, 'shape') else type(v).__name__}>")
        if mode != _last_mode[0] or log_every:
            _last_mode[0] = mode
            if is_motor:
                print(f"[send_action | step {_step[0]:4d}] MODE=MOTOR  | " + "  ".join(f"{n}={action.get(f'{n}.pos', 0.0):+.1f}" for n in so101.JOINT_NAMES))
            else:
                cur_xyz, tgt_xyz = so101.fk_xyz(current_motor), np.array([action.get("x.pos", 0.0), action.get("y.pos", 0.0), action.get("z.pos", 0.0)])
                delta, grip = tgt_xyz - cur_xyz, action.get("gripper.pos", 0.0)
                DBG.act(f"step {_step[0]:4d} | MODE=EEF | cur=({cur_xyz[0]:+.3f},{cur_xyz[1]:+.3f},{cur_xyz[2]:+.3f})  tgt=({tgt_xyz[0]:+.3f},{tgt_xyz[1]:+.3f},{tgt_xyz[2]:+.3f})  Δ=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})  grip_raw={grip:+.4f}")
        if is_motor:
            target_motor = np.array([float(action.get(f"{n}.pos", 0.0)) for n in so101.JOINT_NAMES], dtype=np.float64)
            if log_every:
                DBG.send("Sending MOTOR target [" + str(len(target_motor)) + "D]: " + "  ".join(f"{n}={target_motor[i]:+.1f}°" for i, n in enumerate(so101.JOINT_NAMES)))
            source_motor, smoothed_motor = _last_cmd_motor[0] if _last_cmd_motor[0] is not None else current_motor, _apply_ema(target_motor)
            waypoints, sent_action = _build_interpolation_waypoints(source_motor, smoothed_motor), None
            if smoothing.interp_keep_control_fps_budget and len(waypoints) > 1:
                step_budget_s, next_tick_t = (1.0 / max(float(control_fps), 1e-6)) / len(waypoints), time.perf_counter()
                for i, wp in enumerate(waypoints):
                    sent_action = _dispatch_motor_command(wp, **kwargs)
                    if i < len(waypoints) - 1:
                        next_tick_t += step_budget_s
                        sleep_s = next_tick_t - time.perf_counter()
                        if sleep_s > smoothing.interp_min_sleep_s:
                            time.sleep(sleep_s)
            else:
                for wp in waypoints:
                    sent_action = _dispatch_motor_command(wp, **kwargs)
            _last_cmd_motor[0] = np.asarray(waypoints[-1], dtype=np.float64).copy()
            return sent_action
        target_xyz = np.array([action["x.pos"], action["y.pos"], action["z.pos"]], dtype=np.float64)
        r6d = np.array([action[f"rot6d_{i}.pos"] for i in range(6)], dtype=np.float64)
        gripper_val = float(action["gripper.pos"])
        if log_every:
            DBG.ik("EEF action input [10D]:")
            DBG.ik(f"  target_xyz = ({target_xyz[0]:+.4f},{target_xyz[1]:+.4f},{target_xyz[2]:+.4f}) m")
            DBG.ik("  rot6d      = [" + ", ".join(f"{v:+.4f}" for v in r6d) + "]")
            DBG.ik(f"  gripper_sigmoid = {gripper_val:+.4f}")
            DBG.ik("  IK seed (current_motor) [" + str(len(current_motor)) + "D]: " + "  ".join(f"{n}={current_motor[i]:+.1f}°" for i, n in enumerate(so101.JOINT_NAMES)))
        if np.linalg.norm(r6d[:3]) < 1e-6:
            DBG.warn(f"step {_step[0]}: Degenerate rot6d (near-zero norm), holding current pose")
            _viz_update(current_motor)
            return current_motor_dict if dry_run else base_action_func(current_motor_dict, **kwargs)
        target_motor = so101.ik_motor_6d(target_xyz, r6d, current_motor)
        if target_motor is None:
            DBG.warn(f"step {_step[0]}: IK failed, holding current pose")
            _viz_update(current_motor)
            return current_motor_dict if dry_run else base_action_func(current_motor_dict, **kwargs)
        if log_every:
            DBG.ik("  IK solution [" + str(len(target_motor)) + "D]: " + "  ".join(f"{n}={target_motor[i]:+.1f}°" for i, n in enumerate(so101.JOINT_NAMES)))
        target_motor[gr_idx] = gripper_val
        if log_every:
            DBG.grip(f"gripper: {gripper_val:.4f}°")
        source_motor, smoothed_motor = _last_cmd_motor[0] if _last_cmd_motor[0] is not None else current_motor, _apply_ema(target_motor)
        waypoints = _build_interpolation_waypoints(source_motor, smoothed_motor)
        if log_every:
            DBG.send(f"Dispatching MOTOR trajectory ({len(waypoints)} waypoint(s)):")
            final_motor = waypoints[-1]
            for i, n in enumerate(so101.JOINT_NAMES):
                DBG.send(f"  {n+'.pos':30s} = {float(final_motor[i]):+.2f}°")
        sent_action = None
        if smoothing.interp_keep_control_fps_budget and len(waypoints) > 1:
            step_budget_s, next_tick_t = (1.0 / max(float(control_fps), 1e-6)) / len(waypoints), time.perf_counter()
            for i, wp in enumerate(waypoints):
                sent_action = _dispatch_motor_command(wp, **kwargs)
                if i < len(waypoints) - 1:
                    next_tick_t += step_budget_s
                    sleep_s = next_tick_t - time.perf_counter()
                    if sleep_s > smoothing.interp_min_sleep_s:
                        time.sleep(sleep_s)
        else:
            for wp in waypoints:
                sent_action = _dispatch_motor_command(wp, **kwargs)
        _last_cmd_motor[0] = np.asarray(waypoints[-1], dtype=np.float64).copy()
        return sent_action

    robot.send_action = patched_send_action
