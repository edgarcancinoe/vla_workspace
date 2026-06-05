#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT_DIR / "src"
LEROBOT_SRC = ROOT_DIR.parent / "repos" / "lerobot" / "src"
for path in [SRC_ROOT, LEROBOT_SRC]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def parse_episode_list(tokens: list[str] | None) -> set[int] | None:
    if not tokens:
        return None
    indices = set()
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if part:
                indices.add(int(part))
    return indices


def default_dst_root(src_root: Path | None, dst_repo_id: str) -> Path:
    name = dst_repo_id.split("/")[-1]
    return (src_root.parent / name if src_root is not None else Path.cwd() / name).resolve()


def read_episode_arrays(dataset: LeRobotDataset, episode_index: int, keys: list[str]) -> dict[str, np.ndarray]:
    columns = [key for key in keys if key in dataset.features]
    if not columns:
        return {}
    ep_meta = dataset.meta.episodes[episode_index]
    src_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
    frame_from = int(ep_meta["dataset_from_index"])
    frame_to = int(ep_meta["dataset_to_index"])
    df = pd.read_parquet(src_path, columns=["index", *columns])
    df = df[(df["index"] >= frame_from) & (df["index"] < frame_to)].sort_values("index")
    return {key: np.stack(df[key].to_list()) for key in columns if len(df) > 0}


def movement_score(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    flat = arr.reshape(arr.shape[0], -1).astype(np.float32)
    if len(flat) < 2:
        return np.zeros((0,), dtype=np.float32)
    return np.max(np.abs(np.diff(flat, axis=0)), axis=1)


def infer_keep_length(
    episode_length: int,
    arrays: dict[str, np.ndarray],
    threshold: float,
    window: int,
    min_keep: int,
    buffer_frames: int,
) -> tuple[int, str]:
    scores = []
    if "action" in arrays:
        scores.append(movement_score(arrays["action"]))
    if "observation.state" in arrays:
        scores.append(movement_score(arrays["observation.state"]))
    if not scores:
        return episode_length, "no_action_or_state"
    combined = np.maximum.reduce(scores)
    if len(combined) == 0:
        return max(1, min_keep), "single_frame_episode"
    active = np.flatnonzero(combined > threshold)
    if len(active) == 0:
        return min(episode_length, max(1, min_keep)), "no_motion_detected"
    last_active_transition = int(active[-1])
    keep_length = last_active_transition + 2 + buffer_frames
    if window > 1 and len(combined) >= window and np.all(combined[-window:] <= threshold):
        reason = f"stagnant_tail_window={window}"
    else:
        reason = "last_active_transition"
    keep_length = max(min_keep, keep_length)
    keep_length = min(episode_length, keep_length)
    return keep_length, reason


def frame_to_writable(frame: dict) -> dict:
    out = {}
    for key, value in frame.items():
        if key in {"index", "episode_index", "frame_index"}:
            continue
        if isinstance(value, torch.Tensor):
            out[key] = value.cpu().numpy()
        else:
            out[key] = value
    return out


def copy_episode(dataset: LeRobotDataset, new_dataset: LeRobotDataset, episode_index: int, keep_length: int) -> None:
    ep_meta = dataset.meta.episodes[episode_index]
    frame_from = int(ep_meta["dataset_from_index"])
    frame_to = frame_from + keep_length
    for idx in range(frame_from, frame_to):
        new_dataset.add_frame(frame_to_writable(dataset[idx]))
    new_dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim no-motion tails from LeRobot episodes by rewriting the dataset.")
    parser.add_argument("--src-root", type=Path, default=None, help="Local source dataset root.")
    parser.add_argument("--src-repo-id", type=str, default=None, help="Source HF repo id.")
    parser.add_argument("--dst-root", type=Path, default=None, help="Output folder for rewritten dataset.")
    parser.add_argument("--dst-repo-id", type=str, default=None, help="Destination repo id metadata.")
    parser.add_argument("--revision", type=str, default=None, help="Optional source dataset revision.")
    parser.add_argument("--episodes", nargs="+", default=None, help="Only trim these episode indices. Others are copied unchanged.")
    parser.add_argument("--drop-last", type=int, default=None, help="Drop a fixed number of frames from the end of targeted episodes.")
    parser.add_argument("--trim-seconds", type=float, default=None, help="Drop a fixed number of seconds from the end of targeted episodes.")
    parser.add_argument("--movement-threshold", type=float, default=None, help="Auto-trim by keeping frames up to the last action/state change above this threshold.")
    parser.add_argument("--stagnation-window", type=int, default=3, help="How many final transitions must be quiet to label the tail as stagnant.")
    parser.add_argument("--buffer-frames", type=int, default=0, help="Extra frames to keep after the last detected movement.")
    parser.add_argument("--min-keep-frames", type=int, default=8, help="Never keep fewer than this many frames in any episode.")
    parser.add_argument("--force-cache-sync", action="store_true", help="Refresh source files from HF/cache first.")
    parser.add_argument("--push", action="store_true", help="Push rewritten dataset to Hugging Face after writing it locally.")
    parser.add_argument("--private", action="store_true", help="Create destination HF dataset as private when used with --push.")
    parser.add_argument("--dry-run", action="store_true", help="Only compute trim decisions and print the report.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.src_repo_id is None and args.src_root is None:
        raise ValueError("Provide either --src-root or --src-repo-id.")
    modes = [args.drop_last is not None, args.trim_seconds is not None, args.movement_threshold is not None]
    if sum(modes) != 1:
        raise ValueError("Choose exactly one trim mode: --drop-last, --trim-seconds, or --movement-threshold.")

    src_root = args.src_root.resolve() if args.src_root else None
    src_repo_id = args.src_repo_id or src_root.name
    target_episodes = parse_episode_list(args.episodes)
    dataset = LeRobotDataset(repo_id=src_repo_id, root=src_root, revision=args.revision, force_cache_sync=args.force_cache_sync, video_backend="pyav")
    if target_episodes is not None:
        invalid = sorted(ep for ep in target_episodes if ep < 0 or ep >= dataset.meta.total_episodes)
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}")

    dst_repo_id = args.dst_repo_id or f"{src_repo_id}_trimmed"
    dst_root = args.dst_root.resolve() if args.dst_root else default_dst_root(src_root, dst_repo_id)
    if dst_root.exists() and not args.dry_run:
        raise FileExistsError(f"Destination already exists: {dst_root}")

    fixed_drop = args.drop_last if args.drop_last is not None else int(round((args.trim_seconds or 0.0) * dataset.fps))
    report = []
    keep_lengths = {}
    for ep_idx in range(dataset.meta.total_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        episode_length = int(ep_meta["length"])
        should_trim = target_episodes is None or ep_idx in target_episodes
        if not should_trim:
            keep_length, reason = episode_length, "copied_unchanged"
        elif args.movement_threshold is not None:
            arrays = read_episode_arrays(dataset, ep_idx, ["action", "observation.state"])
            keep_length, reason = infer_keep_length(
                episode_length,
                arrays,
                threshold=args.movement_threshold,
                window=max(1, args.stagnation_window),
                min_keep=max(1, args.min_keep_frames),
                buffer_frames=max(0, args.buffer_frames),
            )
        else:
            keep_length = max(max(1, args.min_keep_frames), episode_length - max(0, fixed_drop))
            keep_length = min(episode_length, keep_length)
            reason = f"fixed_drop={max(0, fixed_drop)}"
        keep_lengths[ep_idx] = keep_length
        report.append(
            {
                "episode_index": ep_idx,
                "old_length": episode_length,
                "new_length": keep_length,
                "trimmed_frames": episode_length - keep_length,
                "trimmed_seconds": round((episode_length - keep_length) / dataset.fps, 4),
                "reason": reason,
            }
        )

    trimmed_episodes = sum(1 for row in report if row["trimmed_frames"] > 0)
    total_trimmed_frames = sum(row["trimmed_frames"] for row in report)
    logging.info("Episodes: %d | trimmed episodes: %d | trimmed frames: %d", len(report), trimmed_episodes, total_trimmed_frames)
    for row in report[:20]:
        logging.info("ep=%d old=%d new=%d trimmed=%d reason=%s", row["episode_index"], row["old_length"], row["new_length"], row["trimmed_frames"], row["reason"])
    if len(report) > 20:
        logging.info("... %d more episodes omitted from console report", len(report) - 20)

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    new_dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=dst_root,
        use_videos=len(dataset.meta.video_keys) > 0,
    )
    for ep_idx in range(dataset.meta.total_episodes):
        copy_episode(dataset, new_dataset, ep_idx, keep_lengths[ep_idx])
    new_dataset.finalize()
    report_path = dst_root / "trim_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logging.info("Wrote trimmed dataset to %s", dst_root)
    logging.info("Wrote trim report to %s", report_path)

    if args.push:
        new_dataset.push_to_hub(private=args.private)
        logging.info("Pushed dataset to %s", dst_repo_id)


if __name__ == "__main__":
    main()
