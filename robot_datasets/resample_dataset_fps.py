#!/usr/bin/env python3
"""
resample_dataset_fps.py

Downsample a LeRobot v3.0 dataset from a higher FPS to a lower FPS by
keeping every N-th frame (no interpolation / no frame blending).

Creates a NEW dataset folder — the source is never modified.

Usage:
    python resample_dataset_fps.py \
        --src /path/to/soarm101_pickplace_10d \
        --dst-fps 7.5 \
        --hf-repo edgarcancinoe/soarm101_pickplace_10d_7p5hz

    # dst defaults to  <src>_7p5hz  next to src if --dst is not given.

Requirements:
    pip install pyarrow numpy tqdm huggingface_hub
    ffmpeg in PATH with libsvtav1 (or libaom-av1 via --vcodec libaom-av1)
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
# Statistics helper  (same as fix_dataset_16d.py)
# ---------------------------------------------------------------------------

def compute_stats(arr: np.ndarray) -> dict:
    """arr shape: (N, D) — one row per frame, one column per feature dim."""
    ddof = 1 if len(arr) > 1 else 0
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0, ddof=ddof).tolist(),
        "count": [len(arr)],
        "q01":   np.percentile(arr, 1,  axis=0).tolist(),
        "q10":   np.percentile(arr, 10, axis=0).tolist(),
        "q50":   np.percentile(arr, 50, axis=0).tolist(),
        "q90":   np.percentile(arr, 90, axis=0).tolist(),
        "q99":   np.percentile(arr, 99, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Downsample a LeRobot v3.0 dataset to a lower FPS (stride subsampling)."
    )
    parser.add_argument(
        "--src", type=Path,
        default=Path("/tmp/vla_cache_jose/lerobot/edgarcancinoe/soarm101_pickplace_10d"),
        help="Path to the source LeRobot dataset root.",
    )
    parser.add_argument(
        "--dst", type=Path, default=None,
        help="Output dataset path. Defaults to <src>_<fps>hz beside the source.",
    )
    parser.add_argument(
        "--dst-fps", type=float, default=7.5,
        help="Target FPS (default: 7.5). Must evenly divide the source FPS.",
    )
    parser.add_argument(
        "--vcodec", type=str, default="libsvtav1",
        help="FFmpeg video encoder (default: libsvtav1). Use libaom-av1 if libsvtav1 is unavailable.",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace repo ID to push to after conversion "
             "(e.g. 'edgarcancinoe/soarm101_pickplace_10d_7p5hz'). Skipped if not given.",
    )
    args = parser.parse_args()

    src: Path = args.src.resolve()
    dst_fps: float = args.dst_fps
    fps_tag = f"{dst_fps:.4g}".replace(".", "p")   # 7.5 → "7p5"

    dst: Path = args.dst.resolve() if args.dst else src.parent / f"{src.name}_{fps_tag}hz"

    print(f"\n{'='*60}")
    print(f"Source   : {src}")
    print(f"Dest     : {dst}")
    print(f"Dst FPS  : {dst_fps}")
    print(f"Encoder  : {args.vcodec}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load source info.json
    # ------------------------------------------------------------------
    with open(src / "meta" / "info.json") as f:
        info = json.load(f)

    src_fps: float = float(info["fps"])
    stride_f = src_fps / dst_fps
    if abs(stride_f - round(stride_f)) > 1e-6:
        sys.exit(f"ERROR: src_fps ({src_fps}) / dst_fps ({dst_fps}) = {stride_f:.4f} is not an integer.")
    stride = int(round(stride_f))
    print(f"Stride   : {stride}  (keep every {stride}-th frame, i.e. frames 0, {stride}, {2*stride}, …)\n")

    # ------------------------------------------------------------------
    # STEP 1 — Create destination directory skeleton
    # ------------------------------------------------------------------
    print("STEP 1 — Creating destination directory …")
    if dst.exists():
        print(f"  WARNING: '{dst}' already exists — removing it first.")
        shutil.rmtree(dst)

    (dst / "data").mkdir(parents=True)
    (dst / "meta" / "episodes").mkdir(parents=True)

    # tasks.parquet is unchanged (task descriptions don't depend on FPS)
    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    print("  Done.\n")

    # ------------------------------------------------------------------
    # STEP 2 — Subsample data parquets
    # ------------------------------------------------------------------
    print("STEP 2 — Subsampling data parquets …")

    parquet_files = sorted((src / "data").rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"ERROR: No parquet files found under {src / 'data'}")

    # Determine column types from the first parquet (all files share the same schema)
    first_table = pq.read_table(parquet_files[0])
    scalar_cols = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    vector_cols = [f.name for f in first_table.schema if f.name not in scalar_cols]
    src_schema  = {f.name: f.type for f in first_table.schema}

    # episode_data[ep_idx] = list of per-chunk dicts, each dict = {col: list_of_values}
    episode_data: dict[int, list[dict]] = {}
    global_new_index = 0

    for parquet_path in tqdm(parquet_files, desc="  Reading parquets"):
        df = pq.read_table(parquet_path).to_pandas()

        for ep_idx in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == ep_idx]
            ep_df = ep_df[ep_df["frame_index"] % stride == 0].copy()
            if ep_df.empty:
                continue

            n = len(ep_df)
            new_local_fi  = np.arange(n, dtype=np.int64)
            new_ts        = (new_local_fi / dst_fps).astype(np.float32)
            new_global_fi = np.arange(global_new_index, global_new_index + n, dtype=np.int64)
            global_new_index += n

            chunk: dict[str, list] = {
                "episode_index": [int(ep_idx)] * n,
                "task_index":    ep_df["task_index"].values.tolist(),
                "frame_index":   new_local_fi.tolist(),
                "timestamp":     new_ts.tolist(),
                "index":         new_global_fi.tolist(),
            }
            for vc in vector_cols:
                chunk[vc] = ep_df[vc].values.tolist()

            episode_data.setdefault(ep_idx, []).append(chunk)

    # Flatten into a single ordered list of frame dicts
    global_frames: list[dict] = []
    for ep_idx in sorted(episode_data.keys()):
        for chunk in episode_data[ep_idx]:
            for i in range(len(chunk["frame_index"])):
                global_frames.append({k: v[i] for k, v in chunk.items()})

    total_frames_new = len(global_frames)
    print(f"  Source frames : {sum(len(pq.read_table(p)) for p in parquet_files)}")
    print(f"  Output frames : {total_frames_new}  (~{total_frames_new / total_frames_new * 100:.0f}% of source)")

    # Distribute output frames across the same shard files as the source,
    # preserving the proportional split.
    shard_src_counts = [len(pq.read_table(p)) for p in parquet_files]
    total_src = sum(shard_src_counts)
    cum_src = 0
    cum_new = 0

    print("  Writing output parquets …")
    for i, shard_path in enumerate(tqdm(parquet_files, desc="  Shards")):
        cum_src += shard_src_counts[i]
        shard_end = total_frames_new if i == len(parquet_files) - 1 \
            else int(round(cum_src / total_src * total_frames_new))
        shard_frames = global_frames[cum_new:shard_end]
        cum_new = shard_end

        if not shard_frames:
            continue

        out_path = dst / "data" / shard_path.relative_to(src / "data")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        arrays, schema_fields = [], []

        # Vector columns
        for col in vector_cols:
            vec_len = len(shard_frames[0][col])
            flat = np.array([f[col] for f in shard_frames], dtype=np.float32).flatten()
            arrays.append(pa.FixedSizeListArray.from_arrays(flat, vec_len))
            schema_fields.append(pa.field(col, pa.list_(pa.float32(), vec_len)))

        # Scalar columns
        for col, pa_type in [
            ("timestamp",     pa.float32()),
            ("frame_index",   pa.int64()),
            ("episode_index", pa.int64()),
            ("index",         pa.int64()),
            ("task_index",    pa.int64()),
        ]:
            arrays.append(pa.array([f[col] for f in shard_frames], type=pa_type))
            schema_fields.append(pa.field(col, pa_type))

        pq.write_table(
            pa.Table.from_arrays(arrays, schema=pa.schema(schema_fields)),
            out_path,
        )

    print("  Data parquets done.\n")

    # ------------------------------------------------------------------
    # STEP 3 — Re-encode videos at dst_fps
    # ------------------------------------------------------------------
    print("STEP 3 — Re-encoding videos …")
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    fps_frac   = str(Fraction(dst_fps).limit_denominator(1000))  # "15/2" for 7.5

    # Extra encoder flags depending on codec
    extra_flags: list[str] = []
    if args.vcodec == "libsvtav1":
        extra_flags = ["-preset", "8"]          # 0=slowest … 13=fastest for SVT-AV1
    elif args.vcodec == "libaom-av1":
        extra_flags = ["-cpu-used", "8", "-row-mt", "1"]   # fastest for libaom

    for vkey in video_keys:
        src_vid_dir = src  / "videos" / vkey
        dst_vid_dir = dst  / "videos" / vkey

        mp4_files = sorted(src_vid_dir.rglob("*.mp4"))
        if not mp4_files:
            print(f"  WARNING: no mp4 files for '{vkey}' — skipping.")
            continue

        for mp4_src in tqdm(mp4_files, desc=f"  {vkey}"):
            mp4_dst = dst_vid_dir / mp4_src.relative_to(src_vid_dir)
            mp4_dst.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg", "-y",
                "-i", str(mp4_src),
                "-vf", f"fps={fps_frac}",
                "-c:v", args.vcodec,
                "-pix_fmt", "yuv420p",
                "-g", "2",
                "-crf", "30",
                *extra_flags,
                "-an",
                str(mp4_dst),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"\n  ERROR: ffmpeg failed on {mp4_src.name}")
                print(result.stderr.decode()[-2000:])   # last 2000 chars of stderr
                sys.exit(1)

    print("  Videos done.\n")

    # ------------------------------------------------------------------
    # STEP 4 — Update meta/info.json
    # ------------------------------------------------------------------
    print("STEP 4 — Updating info.json …")
    new_info = json.loads(json.dumps(info))   # deep copy
    new_info["fps"] = dst_fps
    new_info["total_frames"] = total_frames_new
    for feat_val in new_info["features"].values():
        if feat_val.get("dtype") == "video" and "info" in feat_val:
            feat_val["info"]["video.fps"] = dst_fps

    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)
    print("  Done.\n")

    # ------------------------------------------------------------------
    # STEP 5 — Recompute meta/stats.json
    # ------------------------------------------------------------------
    print("STEP 5 — Recomputing stats.json …")
    src_stats: dict = {}
    if (src / "meta" / "stats.json").exists():
        with open(src / "meta" / "stats.json") as f:
            src_stats = json.load(f)

    new_stats: dict = {}

    # Image pixel stats are unchanged by frame subsampling — copy directly
    for k, v in src_stats.items():
        if k.startswith("observation.images"):
            new_stats[k] = v

    def collect(col: str) -> np.ndarray:
        vals = np.array([f[col] for f in global_frames], dtype=np.float32)
        return vals[:, None] if vals.ndim == 1 else vals

    for col in vector_cols:
        new_stats[col] = compute_stats(collect(col))
    for col in ("timestamp", "frame_index", "index", "task_index"):
        new_stats[col] = compute_stats(collect(col))

    with open(dst / "meta" / "stats.json", "w") as f:
        json.dump(new_stats, f, indent=4)
    print("  Done.\n")

    # ------------------------------------------------------------------
    # STEP 6 — Update meta/episodes/ parquets
    # ------------------------------------------------------------------
    print("STEP 6 — Updating episode metadata parquets …")

    # Build per-episode summary (length, global index range, frame list for stats)
    ep_summary: dict[int, dict] = {}
    cum = 0
    for ep_idx in sorted(episode_data.keys()):
        frames = [
            {k: v[i] for k, v in chunk.items()}
            for chunk in episode_data[ep_idx]
            for i in range(len(chunk["frame_index"]))
        ]
        ep_summary[ep_idx] = {
            "length":   len(frames),
            "from_idx": cum,
            "to_idx":   cum + len(frames) - 1,
            "frames":   frames,
        }
        cum += len(frames)

    for src_ep_pq in tqdm(sorted((src / "meta" / "episodes").rglob("*.parquet")),
                          desc="  Episode meta"):
        orig_table = pq.read_table(src_ep_pq)
        df = orig_table.to_pandas()

        for row_i in range(len(df)):
            ep_idx = int(df.at[row_i, "episode_index"])
            if ep_idx not in ep_summary:
                continue

            ep = ep_summary[ep_idx]

            df.at[row_i, "length"]             = ep["length"]
            df.at[row_i, "dataset_from_index"] = ep["from_idx"]
            df.at[row_i, "dataset_to_index"]   = ep["to_idx"]

            # Video timestamp ranges (position within the concatenated video file)
            for vkey in video_keys:
                fc = f"videos/{vkey}/from_timestamp"
                tc = f"videos/{vkey}/to_timestamp"
                if fc in df.columns:
                    df.at[row_i, fc] = ep["from_idx"] / dst_fps
                if tc in df.columns:
                    df.at[row_i, tc] = ep["to_idx"]   / dst_fps

            # Per-episode feature stats
            ep_frames = ep["frames"]
            for col in vector_cols:
                arr = np.array([f[col] for f in ep_frames], dtype=np.float32)
                for sname, sval in compute_stats(arr).items():
                    c = f"stats/{col}/{sname}"
                    if c in df.columns:
                        df.at[row_i, c] = sval

            for col in ("timestamp", "frame_index", "index", "task_index"):
                arr = np.array([f[col] for f in ep_frames], dtype=np.float32)[:, None]
                for sname, sval in compute_stats(arr).items():
                    c = f"stats/{col}/{sname}"
                    if c in df.columns:
                        df.at[row_i, c] = sval

        # Write back with pure PyArrow (avoids pandas metadata that breaks the viewer)
        new_ep_table = pa.Table.from_arrays(
            [pa.array(df[f.name].tolist(), type=f.type) for f in orig_table.schema],
            schema=orig_table.schema,
        )
        out_ep = dst / "meta" / src_ep_pq.relative_to(src / "meta")
        out_ep.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_ep_table, out_ep)

    print("  Done.\n")

    # ------------------------------------------------------------------
    # STEP 7 — Sanity check
    # ------------------------------------------------------------------
    print("STEP 7 — Sanity check …")
    ct = pq.read_table(sorted((dst / "data").rglob("*.parquet"))[0]).to_pandas()
    ts0, ts1 = float(ct.iloc[0]["timestamp"]), float(ct.iloc[1]["timestamp"])
    step = ts1 - ts0
    expected = 1.0 / dst_fps
    ok = abs(step - expected) < 1e-4
    print(f"  ts[0]={ts0:.6f}  ts[1]={ts1:.6f}  step={step:.6f}  expected={expected:.6f}  {'✓' if ok else '✗ MISMATCH'}")
    print(f"  Total frames: {total_frames_new}  (source had {total_src})\n")

    # ------------------------------------------------------------------
    # STEP 8 — Push to HuggingFace (optional)
    # ------------------------------------------------------------------
    if args.hf_repo:
        print(f"STEP 8 — Pushing to HuggingFace: {args.hf_repo} …")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(dst), repo_id=args.hf_repo, repo_type="dataset")
        print(f"  Pushed → https://huggingface.co/datasets/{args.hf_repo}")
    else:
        print("Skipping HuggingFace push (pass --hf-repo to enable).")

    print(f"\n{'='*60}")
    print(f"Done!  Dataset written to: {dst}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
