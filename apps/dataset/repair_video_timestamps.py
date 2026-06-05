#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import av
import pandas as pd


def _video_frame_count(video_path: Path) -> int:
    with av.open(str(video_path)) as container:
        if not container.streams.video: raise ValueError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        if stream.frames and int(stream.frames) > 0: return int(stream.frames)
        frame_count = 0
        for packet in container.demux(stream):
            for _frame in packet.decode(): frame_count += 1
        return frame_count


def _load_episode_tables(root: Path) -> tuple[dict[Path, pd.DataFrame], dict[int, tuple[Path, int]]]:
    tables: dict[Path, pd.DataFrame] = {}
    row_locs: dict[int, tuple[Path, int]] = {}
    for parquet_path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        tables[parquet_path] = df
        for row_idx, ep_idx in enumerate(df["episode_index"].astype(int).tolist()):
            row_locs[int(ep_idx)] = (parquet_path, int(row_idx))
    if not row_locs: raise FileNotFoundError(f"No episode metadata found under {root / 'meta' / 'episodes'}")
    return tables, row_locs


def _video_keys(root: Path) -> list[str]:
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    return [key for key, spec in info.get("features", {}).items() if spec.get("dtype") == "video"]


def _fps(root: Path) -> float:
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    return float(info["fps"])


def _episode_groups(tables: dict[Path, pd.DataFrame], video_key: str) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_col = f"videos/{video_key}/from_timestamp"
    for df in tables.values():
        for _, row in df.iterrows():
            if chunk_col not in row.index or file_col not in row.index: continue
            if pd.isna(row[chunk_col]) or pd.isna(row[file_col]): continue
            groups[(int(row[chunk_col]), int(row[file_col]))].append(int(row["episode_index"]))
    return dict(groups)


def _sort_episode_ids(tables: dict[Path, pd.DataFrame], episode_ids: list[int], video_key: str) -> list[int]:
    from_col = f"videos/{video_key}/from_timestamp"
    by_ep: dict[int, tuple[float, int]] = {}
    for df in tables.values():
        for _, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            if ep_idx in episode_ids: by_ep[ep_idx] = (float(row.get(from_col, 0.0) or 0.0), ep_idx)
    return [ep_idx for ep_idx, _meta in sorted(by_ep.items(), key=lambda item: item[1])]


def repair_video_timestamps(root: Path, validate_loader: bool) -> None:
    root = root.resolve()
    fps = _fps(root)
    video_keys = _video_keys(root)
    if not video_keys: raise ValueError(f"No video keys found in {root / 'meta' / 'info.json'}")
    tables, row_locs = _load_episode_tables(root)
    changed = 0

    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"
        for (chunk_idx, file_idx), episode_ids in sorted(_episode_groups(tables, video_key).items()):
            ordered_episode_ids = _sort_episode_ids(tables, episode_ids, video_key)
            cursor = 0
            for ep_idx in ordered_episode_ids:
                table_path, row_idx = row_locs[ep_idx]
                df = tables[table_path]
                length = int(df.at[row_idx, "length"])
                expected_from = cursor / fps
                expected_to = (cursor + length) / fps
                old_from = float(df.at[row_idx, from_col])
                old_to = float(df.at[row_idx, to_col])
                if abs(old_from - expected_from) > 1e-6 or abs(old_to - expected_to) > 1e-6:
                    changed += 1
                    df.at[row_idx, from_col] = expected_from
                    df.at[row_idx, to_col] = expected_to
                cursor += length
            video_path = root / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            actual_frames = _video_frame_count(video_path)
            if actual_frames != cursor:
                raise RuntimeError(f"{video_path} frame count mismatch: decoded={actual_frames}, expected={cursor}")

    for parquet_path, df in tables.items():
        df.to_parquet(parquet_path, index=False)

    print(f"Repaired {changed} episode video timestamp row(s) under {root}")

    if not validate_loader: return
    repo_src = Path(__file__).resolve().parents[2] / "repos" / "lerobot" / "src"
    alt_repo_src = Path(__file__).resolve().parents[3] / "repos" / "lerobot" / "src"
    for candidate in (repo_src, alt_repo_src):
        if candidate.exists() and str(candidate) not in sys.path: sys.path.insert(0, str(candidate))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(root.name, root=root, revision="v3.0", video_backend="pyav", tolerance_s=1e-4)
    for idx in range(len(ds)): _ = ds[idx]
    print(f"Validated LeRobot loader across {len(ds)} frame(s) with tolerance_s=1e-4")


def main():
    p = argparse.ArgumentParser(description="Repair LeRobot video episode timestamps from exact frame counts and episode lengths.")
    p.add_argument("--root", type=Path, required=True, help="Local dataset root to repair.")
    p.add_argument("--validate-loader", action="store_true", help="Load every frame through LeRobot after repairing.")
    args = p.parse_args()
    repair_video_timestamps(args.root, args.validate_loader)


if __name__ == "__main__":
    main()
