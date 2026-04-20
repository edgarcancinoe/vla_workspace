#!/usr/bin/env python3
"""
resample_dataset_fps.py

Downsample a LeRobot v3.0 dataset from a higher FPS to a lower FPS by
keeping every N-th frame (no interpolation / no frame blending).

Creates a NEW dataset folder. The source dataset is never modified.

Usage:
    python resample_dataset_fps.py \
        --src /path/to/soarm101_pickplace_10d \
        --dst-fps 7.5 \
        --hf-repo edgarcancinoe/soarm101_pickplace_10d_7p5hz

    # dst defaults to <src>_7p5hz next to src if --dst is not given.

huggingface-cli download edgarcancinoe/soarm101_pickplace_multicolor_v1 \
    --repo-type dataset \
    --local-dir ~/datasets/edgarcancinoe/soarm101_pickplace_multicolor_v1


python apps/dataset/resample_dataset_fps.py \
    --src ~/datasets/edgarcancinoe/soarm101_pickplace_multicolor_v1 \
    --dst-fps 7.5 \
    --hf-repo edgarcancinoe/soarm101_pickplace_multicolor_v1_7p5hz

Requirements:
    pip install pyarrow numpy tqdm huggingface_hub av
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_VCODEC = "h264_videotoolbox" if sys.platform == "darwin" else "libsvtav1"


def compute_stats(arr: np.ndarray) -> dict:
    """arr shape: (N, D) — one row per frame, one column per feature dim."""
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


@dataclass(frozen=True)
class VideoJob:
    video_key: str
    chunk_index: int
    file_index: int
    src_path: Path
    dst_path: Path
    episode_indices: list[int]
    selected_source_frame_indices: list[int]
    expected_frame_count: int


def _load_source_episode_rows(src_root: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ep_path in sorted((src_root / "meta" / "episodes").rglob("*.parquet")):
        df = pq.read_table(ep_path).to_pandas()
        for row in df.to_dict(orient="records"):
            rows[int(row["episode_index"])] = row
    if not rows:
        raise FileNotFoundError(f"No episode metadata parquets found under {src_root / 'meta' / 'episodes'}")
    return rows


def _build_video_options(vcodec: str, vbitrate: str) -> list[dict[str, str]]:
    option_sets: list[dict[str, str]] = []

    if "videotoolbox" in vcodec:
        option_sets.append({"g": "2", "b:v": vbitrate})
        option_sets.append({"g": "2"})
    elif vcodec == "libsvtav1":
        option_sets.append({"g": "2", "crf": "30", "preset": "8"})
    elif vcodec == "libaom-av1":
        option_sets.append({"g": "2", "crf": "30", "cpu-used": "8", "row-mt": "1"})
    else:
        option_sets.append({"g": "2", "crf": "30"})

    option_sets.append({})
    return option_sets


def _add_output_stream(
    out_container: av.container.OutputContainer,
    input_stream: av.video.stream.VideoStream,
    vcodec: str,
    dst_fps_fraction: Fraction,
    time_base: Fraction,
    vbitrate: str,
) -> av.video.stream.VideoStream:
    errors: list[Exception] = []
    for options in _build_video_options(vcodec, vbitrate):
        try:
            output_stream = out_container.add_stream(vcodec, rate=dst_fps_fraction, options=options)
            output_stream.width = input_stream.codec_context.width
            output_stream.height = input_stream.codec_context.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.time_base = time_base
            return output_stream
        except Exception as exc:  # nosec B110
            errors.append(exc)

    raise RuntimeError(f"Could not initialize encoder '{vcodec}'. Last error: {errors[-1]}")


def _rebuild_video_file(
    job: VideoJob,
    vcodec: str,
    vbitrate: str,
    dst_fps_fraction: Fraction,
) -> tuple[Path, int]:
    if not job.selected_source_frame_indices:
        raise ValueError(f"No selected frames for {job.src_path}")

    job.dst_path.parent.mkdir(parents=True, exist_ok=True)
    time_base = Fraction(dst_fps_fraction.denominator, dst_fps_fraction.numerator)

    with av.open(str(job.src_path)) as in_container:
        if not in_container.streams.video:
            raise ValueError(f"No video stream found in {job.src_path}")
        in_stream = in_container.streams.video[0]

        with av.open(str(job.dst_path), mode="w") as out_container:
            out_stream = _add_output_stream(
                out_container, in_stream, vcodec, dst_fps_fraction, time_base, vbitrate
            )

            next_keep_idx = 0
            src_frame_idx = 0
            dst_frame_idx = 0
            last_keep = job.selected_source_frame_indices[-1]

            for packet in in_container.demux(in_stream):
                for frame in packet.decode():
                    if next_keep_idx >= len(job.selected_source_frame_indices):
                        break

                    keep_src_idx = job.selected_source_frame_indices[next_keep_idx]
                    if src_frame_idx == keep_src_idx:
                        new_frame = frame.reformat(
                            width=out_stream.width, height=out_stream.height, format=out_stream.pix_fmt
                        )
                        new_frame.pts = dst_frame_idx
                        new_frame.time_base = time_base
                        for out_packet in out_stream.encode(new_frame):
                            out_container.mux(out_packet)
                        dst_frame_idx += 1
                        next_keep_idx += 1

                    src_frame_idx += 1
                    if src_frame_idx > last_keep and next_keep_idx >= len(job.selected_source_frame_indices):
                        break

                if next_keep_idx >= len(job.selected_source_frame_indices):
                    break

            for out_packet in out_stream.encode():
                out_container.mux(out_packet)

    if next_keep_idx != len(job.selected_source_frame_indices):
        raise RuntimeError(
            f"Video ended before all selected frames were written for {job.src_path}. "
            f"Wrote {next_keep_idx}/{len(job.selected_source_frame_indices)} frames."
        )

    if dst_frame_idx != job.expected_frame_count:
        raise RuntimeError(
            f"Unexpected output frame count for {job.dst_path}: "
            f"wrote {dst_frame_idx}, expected {job.expected_frame_count}"
        )

    return job.dst_path, dst_frame_idx


def _decode_video_timestamps(video_path: Path) -> list[float]:
    timestamps: list[float] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.time is not None:
                    timestamps.append(float(frame.time))
                elif frame.pts is not None and frame.time_base is not None:
                    timestamps.append(float(frame.pts * frame.time_base))
                else:
                    raise RuntimeError(f"Could not recover timestamp for a frame in {video_path}")
    return timestamps


def _validate_video_jobs(
    video_jobs: list[VideoJob],
    dst_fps: float,
    tolerance_s: float = 1e-4,
) -> None:
    expected_step = 1.0 / dst_fps
    for job in tqdm(video_jobs, desc="  Video metadata"):
        timestamps = _decode_video_timestamps(job.dst_path)
        if len(timestamps) != job.expected_frame_count:
            raise RuntimeError(
                f"Decoded frame count mismatch for {job.dst_path}: "
                f"{len(timestamps)} vs expected {job.expected_frame_count}"
            )
        for idx, ts in enumerate(timestamps):
            expected_ts = idx * expected_step
            if abs(ts - expected_ts) > tolerance_s:
                raise RuntimeError(
                    f"Timestamp drift in {job.dst_path}: frame {idx} "
                    f"loaded at {ts:.6f}s, expected {expected_ts:.6f}s"
                )


def _validate_episode_metadata(
    dst_root: Path,
    episode_manifest: dict[int, dict],
    video_keys: list[str],
    dst_fps: float,
    tolerance_s: float = 1e-4,
) -> None:
    episode_rows: dict[int, dict] = {}
    for ep_path in sorted((dst_root / "meta" / "episodes").rglob("*.parquet")):
        df = pq.read_table(ep_path).to_pandas()
        for row in df.to_dict(orient="records"):
            episode_rows[int(row["episode_index"])] = row

    for ep_idx, manifest in episode_manifest.items():
        row = episode_rows.get(ep_idx)
        if row is None:
            raise RuntimeError(f"Missing destination episode metadata for episode {ep_idx}")

        if int(row["length"]) != manifest["length"]:
            raise RuntimeError(
                f"Episode length mismatch for episode {ep_idx}: {row['length']} vs {manifest['length']}"
            )
        if int(row["dataset_from_index"]) != manifest["from_idx"]:
            raise RuntimeError(
                f"dataset_from_index mismatch for episode {ep_idx}: "
                f"{row['dataset_from_index']} vs {manifest['from_idx']}"
            )
        if int(row["dataset_to_index"]) != manifest["to_idx"]:
            raise RuntimeError(
                f"dataset_to_index mismatch for episode {ep_idx}: "
                f"{row['dataset_to_index']} vs {manifest['to_idx']}"
            )

        for video_key in video_keys:
            fc = f"videos/{video_key}/from_timestamp"
            tc = f"videos/{video_key}/to_timestamp"
            if fc not in row or tc not in row:
                continue

            expected_from = manifest["video_from_frame"][video_key] / dst_fps
            expected_to = manifest["video_to_frame"][video_key] / dst_fps
            if abs(float(row[fc]) - expected_from) > tolerance_s:
                raise RuntimeError(
                    f"{fc} mismatch for episode {ep_idx}: {row[fc]} vs {expected_from}"
                )
            if abs(float(row[tc]) - expected_to) > tolerance_s:
                raise RuntimeError(
                    f"{tc} mismatch for episode {ep_idx}: {row[tc]} vs {expected_to}"
                )


def _find_high_risk_validation_episodes(
    shard_episodes: dict[int, list[int]],
    episode_manifest: dict[int, dict],
    stride: int,
) -> tuple[list[int], int, int]:
    best_shard = -1
    best_drift = -1

    for shard_i, ep_list in shard_episodes.items():
        src_cum = 0
        local_cum = 0
        global_cum = 0
        shard_drift = 0

        for ep_idx in ep_list:
            manifest = episode_manifest[ep_idx]
            src_length = manifest["src_length"]
            local_keep = manifest["length"]

            shard_drift = max(shard_drift, local_cum - global_cum)

            first = src_cum if src_cum % stride == 0 else src_cum + (stride - (src_cum % stride))
            if first >= src_cum + src_length:
                global_keep = 0
            else:
                global_keep = ((src_cum + src_length - 1) - first) // stride + 1

            src_cum += src_length
            local_cum += local_keep
            global_cum += global_keep
            shard_drift = max(shard_drift, local_cum - global_cum)

        if shard_drift > best_drift:
            best_shard = shard_i
            best_drift = shard_drift

    if best_shard == -1:
        return sorted(episode_manifest.keys()), -1, 0

    return shard_episodes[best_shard], best_shard, best_drift


def _build_source_file_frame_ranges(
    source_episode_rows: dict[int, dict],
    episode_manifest: dict[int, dict],
    video_key: str,
) -> dict[int, tuple[int, int]]:
    file_to_eps: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ep_idx, manifest in episode_manifest.items():
        src_row = source_episode_rows.get(ep_idx)
        if src_row is None:
            raise RuntimeError(f"Episode {ep_idx} missing from source episode metadata.")

        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        if chunk_col not in src_row or file_col not in src_row:
            continue

        file_to_eps[(int(src_row[chunk_col]), int(src_row[file_col]))].append(ep_idx)

    ranges: dict[int, tuple[int, int]] = {}
    for (chunk_idx, file_idx), eps in file_to_eps.items():
        from_col = f"videos/{video_key}/from_timestamp"
        eps = sorted(eps, key=lambda ep: float(source_episode_rows[ep][from_col]))
        src_cursor = 0
        for ep_idx in eps:
            src_len = int(episode_manifest[ep_idx]["src_length"])
            ranges[ep_idx] = (src_cursor, src_cursor + src_len)
            src_cursor += src_len

    return ranges


def _find_source_length_mismatches(
    source_episode_rows: dict[int, dict],
    episode_manifest: dict[int, dict],
) -> list[tuple[int, int, int]]:
    mismatches: list[tuple[int, int, int]] = []
    for ep_idx, manifest in sorted(episode_manifest.items()):
        src_row = source_episode_rows.get(ep_idx)
        if src_row is None:
            continue
        meta_len = int(src_row["length"])
        data_len = int(manifest["src_length"])
        if meta_len != data_len:
            mismatches.append((ep_idx, data_len, meta_len))
    return mismatches


def _decode_video_frame_count(video_path: Path) -> int:
    frame_count = 0
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for _frame in packet.decode():
                frame_count += 1
    return frame_count


def _validate_source_video_jobs(video_jobs: list[VideoJob]) -> None:
    for job in tqdm(video_jobs, desc="  Source video checks"):
        if not job.selected_source_frame_indices:
            raise RuntimeError(f"No selected source frames were computed for {job.src_path}")
        decoded_frames = _decode_video_frame_count(job.src_path)
        required_frames = job.selected_source_frame_indices[-1] + 1
        if decoded_frames < required_frames:
            raise RuntimeError(
                f"Source video is shorter than the selected source frame indices for {job.src_path}. "
                f"Decoded {decoded_frames} frame(s), but resampling requires at least {required_frames}. "
                f"Affected episodes: {job.episode_indices}."
            )


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ModuleNotFoundError:
        script_path = Path(__file__).resolve()
        candidate_paths = [
            script_path.parents[2] / "repos" / "lerobot" / "src",  # repo-local layout
            script_path.parents[3] / "repos" / "lerobot" / "src",  # sibling layout (this workspace)
        ]
        for repo_src in candidate_paths:
            if repo_src.exists():
                repo_src_str = str(repo_src)
                if repo_src_str not in sys.path:
                    sys.path.insert(0, repo_src_str)
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                return LeRobotDataset
        raise


def _validate_with_lerobot_loader(dst_root: Path, episodes: list[int]) -> None:
    LeRobotDataset = _import_lerobot_dataset()
    ds = LeRobotDataset(
        repo_id=dst_root.name,
        root=dst_root,
        episodes=episodes,
        video_backend="pyav",
    )

    for idx in tqdm(range(len(ds)), desc="  LeRobot loader"):
        _ = ds[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Downsample a LeRobot v3.0 dataset to a lower FPS (stride subsampling)."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/tmp/vla_cache_jose/lerobot/edgarcancinoe/soarm101_pickplace_10d"),
        help="Path to the source LeRobot dataset root.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output dataset path. Defaults to <src>_<fps>hz beside the source.",
    )
    parser.add_argument(
        "--dst-fps",
        type=float,
        default=7.5,
        help="Target FPS (default: 7.5). Must evenly divide the source FPS.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default=DEFAULT_VCODEC,
        help=(
            f"Video codec for PyAV re-encoding (default: {DEFAULT_VCODEC}). "
            "Hardware encoders such as 'h264_videotoolbox' may also work if your FFmpeg build supports them."
        ),
    )
    parser.add_argument(
        "--vbitrate",
        type=str,
        default="8M",
        help="Target bitrate for hardware codecs that prefer bitrate control (default: 8M).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(10, os.cpu_count() or 8),
        help="Number of parallel video workers (default: min(4, cpu_count)).",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help=(
            "HuggingFace repo ID to push to after conversion "
            "(e.g. 'edgarcancinoe/soarm101_pickplace_10d_7p5hz'). Skipped if not given."
        ),
    )
    parser.add_argument(
        "--exclude-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Episode indices to drop from the resampled output if the source dataset has known broken episodes.",
    )
    args = parser.parse_args()

    src: Path = args.src.resolve()
    dst_fps: float = args.dst_fps
    fps_tag = f"{dst_fps:.4g}".replace(".", "p")
    dst: Path = args.dst.resolve() if args.dst else src.parent / f"{src.name}_{fps_tag}hz"
    excluded_episodes = set(int(ep) for ep in args.exclude_episodes)

    print(f"\n{'=' * 60}")
    print(f"Source   : {src}")
    print(f"Dest     : {dst}")
    print(f"Dst FPS  : {dst_fps}")
    print(f"Encoder  : {args.vcodec}")
    print(f"Workers  : {args.workers}")
    print(f"{'=' * 60}\n")

    with open(src / "meta" / "info.json") as f:
        info = json.load(f)

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    source_episode_rows = _load_source_episode_rows(src)

    src_fps = float(info["fps"])
    stride_f = src_fps / dst_fps
    if abs(stride_f - round(stride_f)) > 1e-6:
        sys.exit(f"ERROR: src_fps ({src_fps}) / dst_fps ({dst_fps}) = {stride_f:.4f} is not an integer.")
    stride = int(round(stride_f))
    dst_fps_fraction = Fraction(dst_fps).limit_denominator(1000)
    print(f"Stride   : {stride}  (keep every {stride}-th frame, i.e. frames 0, {stride}, {2 * stride}, …)\n")

    print("STEP 1 — Creating destination directory …")
    if dst.exists():
        print(f"  WARNING: '{dst}' already exists — removing it first.")
        shutil.rmtree(dst)

    (dst / "data").mkdir(parents=True)
    (dst / "meta" / "episodes").mkdir(parents=True)
    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    print("  Done.\n")

    print("STEP 2 — Subsampling data parquets and building manifest …")

    parquet_files = sorted((src / "data").rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"ERROR: No parquet files found under {src / 'data'}")

    first_table = pq.read_table(parquet_files[0])
    scalar_cols = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    vector_cols = [f.name for f in first_table.schema if f.name not in scalar_cols]
    all_cols = [f.name for f in first_table.schema]
    src_schema = {f.name: f.type for f in first_table.schema}

    episode_data: dict[int, list[dict]] = {}
    episode_manifest: dict[int, dict] = {}
    shard_episodes: dict[int, list[int]] = defaultdict(list)
    global_frames: list[dict] = []
    global_new_index = 0
    total_src = 0

    for shard_i, parquet_path in enumerate(tqdm(parquet_files, desc="  Reading parquets")):
        df = pq.read_table(parquet_path).to_pandas()
        total_src += len(df)

        for ep_idx in sorted(df["episode_index"].unique()):
            if int(ep_idx) in excluded_episodes:
                continue
            ep_df_all = df[df["episode_index"] == ep_idx].copy()
            src_length = len(ep_df_all)
            selected_local_indices = ep_df_all.loc[
                ep_df_all["frame_index"] % stride == 0, "frame_index"
            ].astype(np.int64)
            ep_df = ep_df_all[ep_df_all["frame_index"] % stride == 0].copy()
            if ep_df.empty:
                raise RuntimeError(f"Episode {ep_idx} produced no frames after stride selection.")

            if ep_idx not in episode_manifest:
                shard_episodes[shard_i].append(int(ep_idx))
                episode_manifest[int(ep_idx)] = {
                    "episode_index": int(ep_idx),
                    "data_shard_index": shard_i,
                    "data_shard_relpath": parquet_path.relative_to(src / "data"),
                    "src_length": 0,
                    "selected_src_local_indices": [],
                    "length": 0,
                    "video_files": {},
                    "video_from_frame": {},
                    "video_to_frame": {},
                    "from_idx": None,
                    "to_idx": None,
                }

            n = len(ep_df)
            new_local_fi = np.arange(n, dtype=np.int64)
            new_ts = (new_local_fi / dst_fps).astype(np.float32)
            new_global_fi = np.arange(global_new_index, global_new_index + n, dtype=np.int64)
            manifest = episode_manifest[int(ep_idx)]
            manifest["src_length"] += int(src_length)
            manifest["selected_src_local_indices"].extend(selected_local_indices.tolist())
            manifest["length"] += int(n)
            if manifest["from_idx"] is None:
                manifest["from_idx"] = int(new_global_fi[0])
            manifest["to_idx"] = int(new_global_fi[-1])
            global_new_index += n

            chunk: dict[str, list] = {
                "episode_index": [int(ep_idx)] * n,
                "task_index": ep_df["task_index"].values.tolist(),
                "frame_index": new_local_fi.tolist(),
                "timestamp": new_ts.tolist(),
                "index": new_global_fi.tolist(),
            }
            for vc in vector_cols:
                chunk[vc] = ep_df[vc].values.tolist()

            episode_data.setdefault(int(ep_idx), []).append(chunk)

    length_mismatches = _find_source_length_mismatches(source_episode_rows, episode_manifest)
    if length_mismatches:
        details = ", ".join(
            f"episode {ep_idx}: data={data_len}, meta={meta_len}"
            for ep_idx, data_len, meta_len in length_mismatches[:10]
        )
        if len(length_mismatches) > 10:
            details += f", ... (+{len(length_mismatches) - 10} more)"
        raise RuntimeError(
            "Source dataset is internally inconsistent between data parquet lengths and meta/episodes lengths. "
            f"{details}. Use a cleaned source dataset or rerun with --exclude-episodes for the broken episodes."
        )

    for ep_idx in sorted(episode_data.keys()):
        frames = [
            {k: v[i] for k, v in chunk.items()}
            for chunk in episode_data[ep_idx]
            for i in range(len(chunk["frame_index"]))
        ]
        episode_manifest[ep_idx]["frames"] = frames
        global_frames.extend(frames)

    total_frames_new = len(global_frames)
    print(f"  Source frames : {total_src}")
    print(f"  Output frames : {total_frames_new}  (~{total_frames_new / total_src * 100:.1f}% of source)")

    print("  Writing output parquets …")
    for shard_i, shard_path in enumerate(tqdm(parquet_files, desc="  Shards")):
        shard_ep_list = shard_episodes.get(shard_i, [])
        shard_frames: list[dict] = []
        for ep_idx in shard_ep_list:
            shard_frames.extend(episode_manifest[ep_idx]["frames"])

        if not shard_frames:
            continue

        out_path = dst / "data" / shard_path.relative_to(src / "data")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = [pa.array([frame[col] for frame in shard_frames], type=src_schema[col]) for col in all_cols]
        schema = pa.schema([pa.field(col, src_schema[col]) for col in all_cols])
        pq.write_table(pa.Table.from_arrays(arrays, schema=schema), out_path)

    print("  Data parquets done.\n")

    print(f"STEP 3 — Rebuilding videos with exact frame indices (workers={args.workers}, codec={args.vcodec}) …")
    video_jobs_by_key: dict[str, list[VideoJob]] = {video_key: [] for video_key in video_keys}

    for video_key in video_keys:
        source_frame_ranges = _build_source_file_frame_ranges(source_episode_rows, episode_manifest, video_key)
        file_to_eps: dict[tuple[int, int], list[int]] = defaultdict(list)
        for ep_idx, manifest in episode_manifest.items():
            src_row = source_episode_rows.get(ep_idx)
            if src_row is None:
                raise RuntimeError(f"Episode {ep_idx} missing from source episode metadata.")

            chunk_col = f"videos/{video_key}/chunk_index"
            file_col = f"videos/{video_key}/file_index"
            if chunk_col not in src_row or file_col not in src_row:
                continue

            chunk_idx = int(src_row[chunk_col])
            file_idx = int(src_row[file_col])
            manifest["video_files"][video_key] = (chunk_idx, file_idx)
            file_to_eps[(chunk_idx, file_idx)].append(ep_idx)

        for (chunk_idx, file_idx), eps in sorted(file_to_eps.items()):
            from_col = f"videos/{video_key}/from_timestamp"
            eps = sorted(eps, key=lambda ep: float(source_episode_rows[ep][from_col]))

            selected_source_frame_indices: list[int] = []
            dst_frame_offset = 0

            for ep_idx in eps:
                manifest = episode_manifest[ep_idx]
                src_from_frame, src_to_frame = source_frame_ranges[ep_idx]
                if src_to_frame - src_from_frame != manifest["src_length"]:
                    raise RuntimeError(
                        f"Source episode/video length mismatch for episode {ep_idx}, {video_key}: "
                        f"{src_to_frame - src_from_frame} vs {manifest['src_length']}"
                    )

                manifest["video_from_frame"][video_key] = dst_frame_offset
                manifest["video_to_frame"][video_key] = dst_frame_offset + manifest["length"]
                dst_frame_offset += manifest["length"]

                selected_source_frame_indices.extend(
                    src_from_frame + int(local_idx) for local_idx in manifest["selected_src_local_indices"]
                )

            src_video = src / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            dst_video = dst / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            video_jobs_by_key[video_key].append(
                VideoJob(
                    video_key=video_key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                    src_path=src_video,
                    dst_path=dst_video,
                    episode_indices=eps,
                    selected_source_frame_indices=selected_source_frame_indices,
                    expected_frame_count=len(selected_source_frame_indices),
                )
            )

    for video_key, jobs in video_jobs_by_key.items():
        if not jobs:
            print(f"  WARNING: no video jobs for '{video_key}' — skipping.")
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_rebuild_video_file, job, args.vcodec, args.vbitrate, dst_fps_fraction): job for job in jobs}
            with tqdm(total=len(futures), desc=f"  {video_key}") as pbar:
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed rebuilding {job.video_key} chunk-{job.chunk_index:03d}/file-{job.file_index:03d}"
                        ) from exc
                    pbar.update(1)

    all_video_jobs = [job for jobs in video_jobs_by_key.values() for job in jobs]
    print("  Validating source video coverage …")
    _validate_source_video_jobs(all_video_jobs)
    print("  Videos done.\n")

    print("STEP 4 — Updating info.json …")
    new_info = json.loads(json.dumps(info))
    new_info["fps"] = dst_fps
    new_info["total_frames"] = total_frames_new
    for feat_val in new_info["features"].values():
        if feat_val.get("dtype") == "video" and "info" in feat_val:
            feat_val["info"]["video.fps"] = dst_fps

    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)
    print("  Done.\n")

    print("STEP 5 — Recomputing stats.json …")
    src_stats: dict = {}
    if (src / "meta" / "stats.json").exists():
        with open(src / "meta" / "stats.json") as f:
            src_stats = json.load(f)

    new_stats: dict = {}
    for key, value in src_stats.items():
        if key.startswith("observation.images"):
            new_stats[key] = value

    def collect(col: str) -> np.ndarray:
        vals = np.array([frame[col] for frame in global_frames], dtype=np.float32)
        return vals[:, None] if vals.ndim == 1 else vals

    for col in vector_cols:
        new_stats[col] = compute_stats(collect(col))
    for col in ("timestamp", "frame_index", "index", "task_index"):
        new_stats[col] = compute_stats(collect(col))

    with open(dst / "meta" / "stats.json", "w") as f:
        json.dump(new_stats, f, indent=4)
    print("  Done.\n")

    print("STEP 6 — Updating episode metadata parquets …")
    for src_ep_pq in tqdm(sorted((src / "meta" / "episodes").rglob("*.parquet")), desc="  Episode meta"):
        orig_table = pq.read_table(src_ep_pq)
        df = orig_table.to_pandas()

        for row_i in range(len(df)):
            ep_idx = int(df.at[row_i, "episode_index"])
            if ep_idx not in episode_manifest:
                continue

            manifest = episode_manifest[ep_idx]
            df.at[row_i, "length"] = manifest["length"]
            df.at[row_i, "dataset_from_index"] = manifest["from_idx"]
            df.at[row_i, "dataset_to_index"] = manifest["to_idx"]

            for video_key in video_keys:
                fc = f"videos/{video_key}/from_timestamp"
                tc = f"videos/{video_key}/to_timestamp"
                if fc in df.columns:
                    df.at[row_i, fc] = manifest["video_from_frame"][video_key] / dst_fps
                if tc in df.columns:
                    df.at[row_i, tc] = manifest["video_to_frame"][video_key] / dst_fps

            ep_frames = manifest["frames"]
            for col in vector_cols:
                arr = np.array([frame[col] for frame in ep_frames], dtype=np.float32)
                for stat_name, stat_val in compute_stats(arr).items():
                    col_name = f"stats/{col}/{stat_name}"
                    if col_name in df.columns:
                        df.at[row_i, col_name] = stat_val

            for col in ("timestamp", "frame_index", "index", "task_index"):
                arr = np.array([frame[col] for frame in ep_frames], dtype=np.float32)[:, None]
                for stat_name, stat_val in compute_stats(arr).items():
                    col_name = f"stats/{col}/{stat_name}"
                    if col_name in df.columns:
                        df.at[row_i, col_name] = stat_val

        new_ep_table = pa.Table.from_arrays(
            [pa.array(df[field.name].tolist(), type=field.type) for field in orig_table.schema],
            schema=orig_table.schema,
        )
        out_ep = dst / "meta" / src_ep_pq.relative_to(src / "meta")
        out_ep.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_ep_table, out_ep)

    print("  Done.\n")

    print("STEP 7 — Validating rebuilt dataset …")
    _validate_video_jobs(all_video_jobs, dst_fps)
    _validate_episode_metadata(dst, episode_manifest, video_keys, dst_fps)
    validation_episodes, validation_shard, validation_drift = _find_high_risk_validation_episodes(
        shard_episodes, episode_manifest, stride
    )
    if validation_shard >= 0:
        print(
            f"  LeRobot loader check will use data shard {validation_shard} "
            f"(max drift under the old algorithm: {validation_drift} frame(s))."
        )
    _validate_with_lerobot_loader(dst, validation_episodes)
    print(f"  Validated {len(all_video_jobs)} video file(s) and {len(validation_episodes)} episode(s) through LeRobot.\n")

    if args.hf_repo:
        print(f"STEP 8 — Pushing to HuggingFace: {args.hf_repo} …")
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError

        api = HfApi()
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(dst),
            repo_id=args.hf_repo,
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
            commit_message="Sync resampled dataset from local output",
        )

        codebase_version = new_info.get("codebase_version")
        if codebase_version:
            try:
                api.create_tag(
                    repo_id=args.hf_repo,
                    tag=codebase_version,
                    repo_type="dataset",
                    revision="main",
                    exist_ok=True,
                )
                print(f"  Tagged repo with codebase version: {codebase_version}")
            except HfHubHTTPError as exc:
                if exc.response is not None and exc.response.status_code == 409:
                    print(f"  Tag already exists: {codebase_version} — updating it to current main")
                    api.delete_tag(
                        repo_id=args.hf_repo,
                        tag=codebase_version,
                        repo_type="dataset",
                    )
                    api.create_tag(
                        repo_id=args.hf_repo,
                        tag=codebase_version,
                        repo_type="dataset",
                        revision="main",
                    )
                    print(f"  Retagged repo with codebase version: {codebase_version}")
                else:
                    raise
        print(f"  Pushed → https://huggingface.co/datasets/{args.hf_repo}")
    else:
        print("Skipping HuggingFace push (pass --hf-repo to enable).")

    print(f"\n{'=' * 60}")
    print(f"Done! Dataset written to: {dst}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
