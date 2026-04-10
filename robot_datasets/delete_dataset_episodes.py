#!/usr/bin/env python3
"""
delete_dataset_episodes.py

Create a new LeRobot dataset with selected episodes removed, then optionally push it to Hugging Face.

Examples:
    python robot_datasets/delete_dataset_episodes.py \
        --src-root /path/to/local_dataset \
        --episodes 137 140

    python robot_datasets/delete_dataset_episodes.py \
        --src-root /path/to/local_dataset \
        --src-repo-id edgarcancinoe/soarm101_pickplace_10d \
        --dst-root /path/to/local_dataset_without_bad_eps \
        --dst-repo-id edgarcancinoe/soarm101_pickplace_10d_clean \
        --episodes 137,140 \
        --push

Requirements:
    pip install huggingface_hub
"""

import argparse
import json
import logging
import sys
from pathlib import Path


DEFAULT_VCODEC = "h264_videotoolbox" if sys.platform == "darwin" else "libsvtav1"


def _import_lerobot_symbols():
    try:
        from lerobot.datasets.dataset_tools import (
            _copy_and_reindex_data,
            _copy_and_reindex_episodes_metadata,
            _copy_and_reindex_videos,
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        return (
            LeRobotDataset,
            LeRobotDatasetMetadata,
            _copy_and_reindex_data,
            _copy_and_reindex_episodes_metadata,
            _copy_and_reindex_videos,
        )
    except ModuleNotFoundError:
        repo_src = Path(__file__).resolve().parents[2] / "repos" / "lerobot" / "src"
        if not repo_src.exists():
            raise
        repo_src_str = str(repo_src)
        if repo_src_str not in sys.path:
            sys.path.insert(0, repo_src_str)
        from lerobot.datasets.dataset_tools import (
            _copy_and_reindex_data,
            _copy_and_reindex_episodes_metadata,
            _copy_and_reindex_videos,
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        return (
            LeRobotDataset,
            LeRobotDatasetMetadata,
            _copy_and_reindex_data,
            _copy_and_reindex_episodes_metadata,
            _copy_and_reindex_videos,
        )


def _parse_episode_indices(tokens: list[str]) -> list[int]:
    episode_indices: list[int] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            episode_indices.append(int(part))
    return sorted(set(episode_indices))


def _default_dst_root(src_root: Path | None, dst_repo_id: str) -> Path:
    dst_name = dst_repo_id.split("/")[-1]
    if src_root is not None:
        return src_root.parent / dst_name
    return Path.cwd() / dst_name


def _read_local_total_episodes(src_root: Path | None) -> int | None:
    if src_root is None:
        return None
    info_path = src_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        info = json.load(f)
    total_episodes = info.get("total_episodes")
    return int(total_episodes) if total_episodes is not None else None


def main():
    parser = argparse.ArgumentParser(
        description="Create a new LeRobot dataset with selected episodes deleted."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help="Local root of the source dataset. If omitted, LeRobot will use its cache/download path.",
    )
    parser.add_argument(
        "--src-repo-id",
        type=str,
        default=None,
        help="Source repo id. Recommended when the source dataset came from Hugging Face.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=None,
        help="Output directory for the rewritten dataset. Defaults to a sibling folder named after --dst-repo-id.",
    )
    parser.add_argument(
        "--dst-repo-id",
        type=str,
        default=None,
        help="Repo id metadata for the rewritten dataset. Defaults to <src-repo-id>_clean or <src-dir>_clean.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        help="Episode indices to delete. Accepts space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional source dataset revision/tag/commit.",
    )
    parser.add_argument(
        "--force-cache-sync",
        action="store_true",
        help="Force LeRobot to refresh the source dataset from the repo/cache first.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default=DEFAULT_VCODEC,
        help=f"Codec to use when mixed video files must be rewritten (default: {DEFAULT_VCODEC}).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the rewritten dataset to --dst-repo-id after local rewrite completes.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push destination repo as private.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    (
        LeRobotDataset,
        LeRobotDatasetMetadata,
        _copy_and_reindex_data,
        _copy_and_reindex_episodes_metadata,
        _copy_and_reindex_videos,
    ) = _import_lerobot_symbols()

    src_root = args.src_root.resolve() if args.src_root else None
    src_repo_id = args.src_repo_id
    if src_repo_id is None:
        if src_root is None:
            raise ValueError("Provide either --src-root or --src-repo-id.")
        src_repo_id = src_root.name

    dst_repo_id = args.dst_repo_id or f"{src_repo_id}_clean"
    dst_root = args.dst_root.resolve() if args.dst_root else _default_dst_root(src_root, dst_repo_id).resolve()

    episode_indices = _parse_episode_indices(args.episodes)
    if not episode_indices:
        raise ValueError("No episode indices were provided.")

    local_total_episodes = _read_local_total_episodes(src_root)
    if local_total_episodes is not None:
        valid_indices = set(range(local_total_episodes))
        invalid = sorted(set(episode_indices) - valid_indices)
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}")
        episodes_to_keep = [idx for idx in range(local_total_episodes) if idx not in set(episode_indices)]
        if not episodes_to_keep:
            raise ValueError("Cannot delete all episodes from the dataset.")
    else:
        episodes_to_keep = None

    logging.info("Loading source dataset")
    dataset = LeRobotDataset(
        repo_id=src_repo_id,
        root=src_root,
        episodes=episodes_to_keep,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        video_backend="pyav",
    )

    if episodes_to_keep is None:
        valid_indices = set(range(dataset.meta.total_episodes))
        invalid = sorted(set(episode_indices) - valid_indices)
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}")
        episodes_to_keep = [idx for idx in range(dataset.meta.total_episodes) if idx not in set(episode_indices)]
    if not episodes_to_keep:
        raise ValueError("Cannot delete all episodes from the dataset.")

    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    logging.info("Deleting %d episode(s): %s", len(episode_indices), episode_indices)
    logging.info("Destination root: %s", dst_root)
    logging.info("Destination repo id: %s", dst_repo_id)

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=dst_repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=dst_root,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}

    video_metadata = None
    if dataset.meta.video_keys:
        video_metadata = _copy_and_reindex_videos(
            dataset,
            new_meta,
            episode_mapping,
            vcodec=args.vcodec,
        )

    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)
    _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)

    logging.info("Rewritten dataset has %d episode(s)", len(episodes_to_keep))

    new_dataset = LeRobotDataset(
        repo_id=dst_repo_id,
        root=dst_root,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
        video_backend="pyav",
    )

    if args.push:
        logging.info("Pushing rewritten dataset to %s", dst_repo_id)
        new_dataset.push_to_hub(private=args.private)
        logging.info("Push complete: https://huggingface.co/datasets/%s", dst_repo_id)
    else:
        logging.info("Skipping push. Rewritten dataset is local at %s", dst_root)


if __name__ == "__main__":
    main()
