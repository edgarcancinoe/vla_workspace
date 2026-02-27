#!/usr/bin/env python3
"""
resample_dataset_fps.py

Downsample a LeRobot v3.0 dataset from a higher FPS to a lower FPS by
keeping every N-th frame (no interpolation / no frame blending).

Creates a NEW dataset folder — the source is never modified.

Usage:
    python resample_dataset_fps.py \
        --src /tmp/vla_cache_jose/lerobot/edgarcancinoe/soarm101_pickplace_10d \
        --dst /tmp/vla_cache_jose/lerobot/edgarcancinoe/soarm101_pickplace_10d_7p5hz \
        --dst-fps 7.5 \
        [--hf-repo edgarcancinoe/soarm101_pickplace_10d_7p5hz]

Requirements:
    - pyarrow, numpy, tqdm, huggingface_hub (for push)
    - ffmpeg in PATH (for video re-encoding)
"""

import argparse
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Statistics helper  (identical to fix_dataset_16d.py)
# ---------------------------------------------------------------------------

def compute_stats(arr: np.ndarray) -> dict:
    ddof = 1 if len(arr) > 1 else 0
    return {
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0, ddof=ddof).tolist(),
        "count": [len(arr)],
        "q01": np.percentile(arr, 1,  axis=0).tolist(),
        "q10": np.percentile(arr, 10, axis=0).tolist(),
        "q50": np.percentile(arr, 50, axis=0).tolist(),
        "q90": np.percentile(arr, 90, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Downsample a LeRobot dataset to a lower FPS.")
    parser.add_argument(
        "--src", type=Path,
        default=Path("/tmp/vla_cache_jose/lerobot/edgarcancinoe/soarm101_pickplace_10d"),
        help="Path to the source LeRobot dataset root.",
    )
    parser.add_argument(
        "--dst", type=Path,
        default=None,
        help="Path for the output dataset. Defaults to <src>_<dst_fps_str>hz next to src.",
    )
    parser.add_argument(
        "--dst-fps", type=float, default=7.5,
        help="Target FPS for the output dataset (default: 7.5).",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace repo ID to push to, e.g. 'edgarcancinoe/soarm101_pickplace_10d_7p5hz'. "
             "If not given, the dataset is NOT pushed.",
    )
    args = parser.parse_args()

    src: Path = args.src.resolve()
    dst_fps: float = args.dst_fps

    # Derive destination path if not given
    fps_tag = f"{dst_fps:.4g}".replace(".", "p")  # e.g. 7.5 -> "7p5"
    if args.dst is None:
        dst = src.parent / f"{src.name}_{fps_tag}hz"
    else:
        dst = args.dst.resolve()

    print(f"\n{'='*60}")
    print(f"Source  : {src}")
    print(f"Dest    : {dst}")
    print(f"Dst FPS : {dst_fps}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load source info.json
    # ------------------------------------------------------------------
    info_path = src / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    src_fps: float = float(info["fps"])
    stride_f = src_fps / dst_fps
    if abs(stride_f - round(stride_f)) > 1e-6:
        print(f"ERROR: src_fps ({src_fps}) / dst_fps ({dst_fps}) = {stride_f} is not an integer.")
        sys.exit(1)
    stride = int(round(stride_f))
    print(f"Stride  : {stride}  (keep every {stride}-th frame)\n")

    # ------------------------------------------------------------------
    # STEP 1 — Create destination directory skeleton
    # ------------------------------------------------------------------
    print("STEP 1 — Creating destination directory …")
    if dst.exists():
        print(f"  WARNING: Destination '{dst}' already exists. Removing …")
        shutil.rmtree(dst)

    (dst / "data").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)
    (dst / "meta" / "episodes").mkdir(parents=True)

    # Copy tasks.parquet unchanged
    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    print("  Done.")

    # ------------------------------------------------------------------
    # STEP 2 — Process data parquets
    # ------------------------------------------------------------------
    print("\nSTEP 2 — Subsampling data parquets …")

    data_dir = src / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found under {data_dir}"); sys.exit(1)

    # Collect per-episode data for later metadata recomputation
    # Keys: episode_index → list of rows (as dicts with numpy arrays)
    episode_data: dict[int, list[dict]] = {}  # ep_idx → list of scalar/array dicts

    # We also need a full flat list for global stats
    all_frames: list[dict] = []  # each entry has the feature arrays for one (subsampled) frame

    global_new_index = 0  # continuous frame counter across all episodes

    for parquet_path in tqdm(parquet_files, desc="  Parquets"):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Identify scalar int/float columns (non-vector columns)
        scalar_cols = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        vector_cols = [c for c in df.columns if c not in scalar_cols]

        # Group by episode to handle frame renumbering correctly
        for ep_idx in df["episode_index"].unique():
            ep_mask = df["episode_index"] == ep_idx
            ep_df = df[ep_mask].copy()
            ep_df = ep_df[ep_df["frame_index"] % stride == 0].copy()

            if len(ep_df) == 0:
                continue

            # New frame indices within episode: 0, 1, 2, …
            new_local_frame_idx = np.arange(len(ep_df), dtype=np.int64)
            new_timestamps = (new_local_frame_idx / dst_fps).astype(np.float32)
            new_global_idx = np.arange(global_new_index, global_new_index + len(ep_df), dtype=np.int64)
            global_new_index += len(ep_df)

            ep_dict = {
                "episode_index": [ep_idx] * len(new_local_frame_idx),
                "task_index": ep_df["task_index"].values.tolist(),
                "frame_index": new_local_frame_idx.tolist(),
                "timestamp": new_timestamps.tolist(),
                "index": new_global_idx.tolist(),
            }
            for vc in vector_cols:
                ep_dict[vc] = ep_df[vc].values.tolist()

            if ep_idx not in episode_data:
                episode_data[ep_idx] = []
            episode_data[ep_idx].append(ep_dict)

    # Merge chunks per episode into a single list, then write out parquets in the
    # same chunk/file structure as the source (one file per source file, same path).
    # We'll just write all data into the same shard layout (one chunk, same files).
    #
    # Flatten all episode data into a global frame list sorted by episode_index then frame_index.
    global_frames = []
    for ep_idx in sorted(episode_data.keys()):
        chunks = episode_data[ep_idx]
        # each chunk is a dict of lists; merge them
        merged: dict[str, list] = {}
        for chunk in chunks:
            for k, v in chunk.items():
                merged.setdefault(k, []).extend(v if isinstance(v, list) else [v])
        for i in range(len(merged["frame_index"])):
            row = {k: v[i] for k, v in merged.items()}
            global_frames.append(row)

    # Figure out source schema for vector columns
    src_schema_dict: dict[str, pa.DataType] = {}
    for parquet_path in parquet_files[:1]:
        t = pq.read_table(parquet_path)
        for field in t.schema:
            src_schema_dict[field.name] = field.type

    # Write data parquets — same shard structure as source
    # Determine the chunk/file layout from the source directory tree
    src_chunk_dirs = sorted((src / "data").iterdir())

    frame_cursor = 0
    total_src_frames = sum(len(pq.read_table(p)) for p in parquet_files)
    # Use the same number of files, distributing frames proportionally
    shard_paths = sorted(data_dir.rglob("*.parquet"))

    # Count frames per source shard to get fractional sizes
    shard_frame_counts = [len(pq.read_table(p)) for p in shard_paths]
    total_frames_new = len(global_frames)

    print(f"\n  Total new frames: {total_frames_new}  (from {len(global_frames)} rows)")
    print(f"  Writing data parquets …")

    # Distribute new frames across shards proportionally
    src_cumulative = 0
    new_cumulative = 0
    for i, shard_path in enumerate(tqdm(shard_paths, desc="  Shards")):
        src_count = shard_frame_counts[i]
        src_cumulative += src_count

        if i < len(shard_paths) - 1:
            new_shard_end = int(round(src_cumulative / total_src_frames * total_frames_new))
        else:
            new_shard_end = total_frames_new

        shard_frames = global_frames[new_cumulative:new_shard_end]
        new_cumulative = new_shard_end

        if not shard_frames:
            continue

        # Determine output path (mirror source structure)
        rel = shard_path.relative_to(src / "data")
        out_path = dst / "data" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build pyarrow table
        schema_fields = []
        arrays = []

        # Vector columns first (to match source ordering)
        for col in vector_cols:
            if col not in shard_frames[0]:
                continue
            dtype = src_schema_dict.get(col)
            flat = np.array([f[col] for f in shard_frames], dtype=np.float32).flatten()
            vec_len = len(shard_frames[0][col]) if isinstance(shard_frames[0][col], (list, np.ndarray)) else 1
            arr = pa.FixedSizeListArray.from_arrays(flat, vec_len)
            arrays.append(arr)
            schema_fields.append(pa.field(col, pa.list_(pa.float32(), vec_len)))

        # Scalar columns
        for col, pa_type in [
            ("timestamp",     pa.float32()),
            ("frame_index",   pa.int64()),
            ("episode_index", pa.int64()),
            ("index",         pa.int64()),
            ("task_index",    pa.int64()),
        ]:
            if col not in shard_frames[0]:
                continue
            vals = [f[col] for f in shard_frames]
            arrays.append(pa.array(vals, type=pa_type))
            schema_fields.append(pa.field(col, pa_type))

        pq.write_table(
            pa.Table.from_arrays(arrays, schema=pa.schema(schema_fields)),
            out_path,
        )

    print("  Data parquets done.")

    # ------------------------------------------------------------------
    # STEP 3 — Re-encode videos
    # ------------------------------------------------------------------
    print("\nSTEP 3 — Re-encoding videos …")

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    fps_str = str(Fraction(dst_fps).limit_denominator(100))  # e.g. "15/2" for 7.5

    for vkey in video_keys:
        src_vid_dir = src / "videos" / vkey
        dst_vid_dir = dst / "videos" / vkey
        dst_vid_dir.mkdir(parents=True, exist_ok=True)

        mp4_files = sorted(src_vid_dir.rglob("*.mp4"))
        if not mp4_files:
            print(f"  WARNING: No mp4 files found for '{vkey}', skipping.")
            continue

        for mp4_src in tqdm(mp4_files, desc=f"  {vkey}"):
            rel = mp4_src.relative_to(src_vid_dir)
            mp4_dst = dst_vid_dir / rel
            mp4_dst.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg", "-y",
                "-i", str(mp4_src),
                "-vf", f"fps={fps_str}",
                "-c:v", "libaom-av1",
                "-pix_fmt", "yuv420p",
                "-g", "2",
                "-crf", "30",
                "-cpu-used", "8",   # fastest preset for libaom-av1
                "-row-mt", "1",     # enable row multithreading
                "-an",
                str(mp4_dst),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"  ERROR running ffmpeg on {mp4_src}:")
                print(result.stderr.decode())
                sys.exit(1)

    print("  Videos done.")

    # ------------------------------------------------------------------
    # STEP 4 — Update info.json
    # ------------------------------------------------------------------
    print("\nSTEP 4 — Updating info.json …")

    new_info = json.loads(json.dumps(info))  # deep copy
    new_info["fps"] = dst_fps
    new_info["total_frames"] = total_frames_new

    for feat_key, feat_val in new_info["features"].items():
        if feat_val.get("dtype") == "video" and "info" in feat_val:
            feat_val["info"]["video.fps"] = dst_fps

    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)
    print("  Done.")

    # ------------------------------------------------------------------
    # STEP 5 — Recompute stats.json
    # ------------------------------------------------------------------
    print("\nSTEP 5 — Recomputing stats.json …")

    # Load source stats to copy image stats (pixel values don't change from subsampling)
    src_stats_path = src / "meta" / "stats.json"
    src_stats = {}
    if src_stats_path.exists():
        with open(src_stats_path) as f:
            src_stats = json.load(f)

    new_stats: dict = {}

    # Copy image stats unchanged
    for k in src_stats:
        if k.startswith("observation.images"):
            new_stats[k] = src_stats[k]

    # Recompute scalar/vector stats from global_frames
    def collect_feature(col: str) -> np.ndarray:
        vals = [f[col] for f in global_frames]
        arr = np.array(vals, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    for col in vector_cols:
        if global_frames and col in global_frames[0]:
            new_stats[col] = compute_stats(collect_feature(col))

    for col in ["timestamp", "frame_index", "index", "task_index"]:
        if global_frames and col in global_frames[0]:
            new_stats[col] = compute_stats(collect_feature(col))

    with open(dst / "meta" / "stats.json", "w") as f:
        json.dump(new_stats, f, indent=4)
    print("  Done.")

    # ------------------------------------------------------------------
    # STEP 6 — Update episode metadata parquets
    # ------------------------------------------------------------------
    print("\nSTEP 6 — Updating episode metadata parquets …")

    # Build per-episode summary from global_frames
    # ep_idx → {length, from_idx, to_idx, frames: list[dict]}
    ep_summary: dict[int, dict] = {}
    cum_idx = 0
    for ep_idx in sorted(episode_data.keys()):
        chunks = episode_data[ep_idx]
        merged_frames = []
        for chunk in chunks:
            for i in range(len(chunk["frame_index"])):
                merged_frames.append({k: v[i] for k, v in chunk.items()})
        length = len(merged_frames)
        ep_summary[ep_idx] = {
            "length": length,
            "from_idx": cum_idx,
            "to_idx": cum_idx + length - 1,
            "frames": merged_frames,
        }
        cum_idx += length

    # Find video chunk/file assignment for each episode from SOURCE episode metadata
    src_ep_pqs = sorted((src / "meta" / "episodes").rglob("*.parquet"))

    for src_ep_pq in tqdm(src_ep_pqs, desc="  Episode meta parquets"):
        orig_table = pq.read_table(src_ep_pq)
        df = orig_table.to_pandas()

        for idx in range(len(df)):
            ep_idx = int(df.iloc[idx]["episode_index"])
            if ep_idx not in ep_summary:
                continue

            ep = ep_summary[ep_idx]
            length = ep["length"]
            from_idx = ep["from_idx"]
            to_idx = ep["to_idx"]

            df.at[idx, "length"] = length
            df.at[idx, "dataset_from_index"] = from_idx
            df.at[idx, "dataset_to_index"] = to_idx

            # Update video timestamp ranges for each video key
            for vkey in video_keys:
                safe_key = vkey.replace(".", "").replace("/", ".")  # col prefix in metadata
                from_col = f"videos/{vkey}/from_timestamp"
                to_col   = f"videos/{vkey}/to_timestamp"
                if from_col in df.columns:
                    df.at[idx, from_col] = from_idx / dst_fps
                if to_col in df.columns:
                    df.at[idx, to_col] = to_idx / dst_fps

            # Recompute per-episode stats
            ep_frames = ep["frames"]
            for col in vector_cols:
                if col in ep_frames[0]:
                    arr = np.array([f[col] for f in ep_frames], dtype=np.float32)
                    s = compute_stats(arr)
                    for stat_name, stat_val in s.items():
                        col_name = f"stats/{col}/{stat_name}"
                        if col_name in df.columns:
                            df.at[idx, col_name] = stat_val

            for col in ["timestamp", "frame_index", "index", "task_index"]:
                if col in ep_frames[0]:
                    arr = np.array([f[col] for f in ep_frames], dtype=np.float32)[:, None]
                    s = compute_stats(arr)
                    for stat_name, stat_val in s.items():
                        col_name = f"stats/{col}/{stat_name}"
                        if col_name in df.columns:
                            df.at[idx, col_name] = stat_val

        # Write back with pure PyArrow (no pandas metadata)
        arrays = []
        for field in orig_table.schema:
            col_data = df[field.name].tolist()
            arrays.append(pa.array(col_data, type=field.type))
        new_ep_table = pa.Table.from_arrays(arrays, schema=orig_table.schema)

        rel = src_ep_pq.relative_to(src / "meta")
        out_ep_path = dst / "meta" / rel
        out_ep_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_ep_table, out_ep_path)

    print("  Done.")

    # ------------------------------------------------------------------
    # STEP 7 — Quick sanity check
    # ------------------------------------------------------------------
    print("\nSTEP 7 — Sanity check …")
    check_pq = sorted((dst / "data").rglob("*.parquet"))[0]
    ct = pq.read_table(check_pq).to_pandas()
    ts0 = float(ct.iloc[0]["timestamp"])
    ts1 = float(ct.iloc[1]["timestamp"])
    expected_step = 1.0 / dst_fps
    print(f"  First timestamp: {ts0:.6f}")
    print(f"  Second timestamp: {ts1:.6f}  (step = {ts1 - ts0:.6f}, expected {expected_step:.6f})")
    print(f"  Total new frames: {total_frames_new}")
    if abs((ts1 - ts0) - expected_step) < 1e-4:
        print("  Timestamp spacing OK ✓")
    else:
        print("  WARNING: timestamp spacing mismatch!")

    # ------------------------------------------------------------------
    # STEP 8 — Push to HuggingFace (optional)
    # ------------------------------------------------------------------
    if args.hf_repo:
        print(f"\nSTEP 8 — Pushing to HuggingFace: {args.hf_repo} …")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(dst),
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        print(f"  Successfully pushed to https://huggingface.co/datasets/{args.hf_repo}")
    else:
        print("\nSkipping HuggingFace push (no --hf-repo given).")

    print(f"\n{'='*60}")
    print(f"Done!  Output dataset at: {dst}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
