#!/usr/bin/env python3
"""
Fix script: Merges the existing 10D EEF + 6D joint columns on HF into a
single 16D observation.state / action, fixes the stale episode metadata
parquets, and re-pushes.

Layout (16D):
  [x, y, z, rot6d_0..rot6d_5, gripper, shoulder_pan, shoulder_lift,
   elbow_flex, wrist_flex, wrist_roll, gripper_joint]

The HF dataset already has the correct data in separate columns:
  observation.state          = 10D EEF
  observation.joint_positions = 6D joints
  action                      = 10D EEF
  action_joints               = 6D joints

This script combines them and fixes all metadata.
"""

import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = "edgarcancinoe/soarm101_pickplace_10d"
LOCAL_ROOT = Path(__file__).parent.parent / "outputs" / "datasets" / "soarm101_pickplace_10d"

FEATURE_NAMES_16D = [
    "x", "y", "z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper_joint",
]


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def compute_stats(arr: np.ndarray) -> dict:
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


# ---------------------------------------------------------------------------
# Download helpers (to get the intact 10D+6D data from HF)
# ---------------------------------------------------------------------------

HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"


def download_file(rel_path: str, dest: Path) -> Path:
    """Download a single file from HF and save locally."""
    url = f"{HF_RESOLVE}/{rel_path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {rel_path} ...")
    urllib.request.urlretrieve(url, str(dest))
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = LOCAL_ROOT
    data_dir = root / "data"
    meta_dir = root / "meta"

    # ------------------------------------------------------------------
    # Step 0: Download the 10D+6D data parquets from HF (they have all
    #         the columns we need; local copies were reverted to 6D).
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 0 — Downloading intact data from HF ...")
    print("=" * 60)

    info_path = meta_dir / "info.json"

    # Download info.json from HF (the one with 10D features declared)
    download_file("meta/info.json", info_path)
    with open(info_path) as f:
        info_dict = json.load(f)

    # Enumerate data parquet files from the local directory structure
    local_data_files = sorted(data_dir.rglob("*.parquet"))
    if not local_data_files:
        raise FileNotFoundError(f"No local data parquet files found in {data_dir}")

    # Download each data parquet from HF (overwriting the reverted local 6D copies)
    for local_path in local_data_files:
        rel = local_path.relative_to(root)
        download_file(str(rel), local_path)

    # ------------------------------------------------------------------
    # Step 1: Merge 10D EEF + 6D joints → 16D in every data parquet
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 — Merging to 16D ...")
    print("=" * 60)

    # Collect per-episode 16D data for later stats
    episode_states: dict[int, list] = {}
    episode_actions: dict[int, list] = {}
    all_states_16d: list = []
    all_actions_16d: list = []

    for parquet_path in local_data_files:
        print(f"  Processing: {parquet_path.relative_to(root)}")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        states_16d = []
        actions_16d = []
        has_action = "action" in df.columns

        for i in range(len(df)):
            row = df.iloc[i]
            ep = int(row["episode_index"])
            if ep not in episode_states:
                episode_states[ep] = []
                episode_actions[ep] = []

            eef_10d = np.array(row["observation.state"], dtype=np.float32)

            if "observation.joint_positions" in df.columns:
                joints_6d = np.array(row["observation.joint_positions"], dtype=np.float32)
            else:
                raise RuntimeError(
                    f"'observation.joint_positions' column missing in {parquet_path.name}! "
                    "Cannot merge 10D + 6D."
                )

            state_16d = np.concatenate([eef_10d, joints_6d]).astype(np.float32)
            states_16d.append(state_16d)
            episode_states[ep].append(state_16d)

            if has_action:
                eef_act_10d = np.array(row["action"], dtype=np.float32)
                if "action_joints" in df.columns:
                    act_joints_6d = np.array(row["action_joints"], dtype=np.float32)
                else:
                    raise RuntimeError(
                        f"'action_joints' column missing in {parquet_path.name}!"
                    )
                action_16d = np.concatenate([eef_act_10d, act_joints_6d]).astype(np.float32)
                actions_16d.append(action_16d)
                episode_actions[ep].append(action_16d)

        all_states_16d.extend(states_16d)
        all_actions_16d.extend(actions_16d)

        # Build new parquet with just 16D columns (drop the separate 6D ones)
        data_out: dict = {}
        schema_fields: list = []

        obs_arr = np.array(states_16d, dtype=np.float32).flatten()
        data_out["observation.state"] = pa.FixedSizeListArray.from_arrays(obs_arr, 16)
        schema_fields.append(pa.field("observation.state", pa.list_(pa.float32(), 16)))

        if has_action and actions_16d:
            act_arr = np.array(actions_16d, dtype=np.float32).flatten()
            data_out["action"] = pa.FixedSizeListArray.from_arrays(act_arr, 16)
            schema_fields.append(pa.field("action", pa.list_(pa.float32(), 16)))

        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col in df.columns:
                data_out[col] = pa.array(df[col])
                schema_fields.append(pa.field(col, table.schema.field(col).type))

        pq.write_table(
            pa.Table.from_pydict(data_out, schema=pa.schema(schema_fields)),
            parquet_path,
        )

    print(f"  Merged {len(all_states_16d)} total frames into 16D.")

    # ------------------------------------------------------------------
    # Step 2: Update info.json
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Updating info.json ...")
    print("=" * 60)

    info_dict["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [16],
        "names": FEATURE_NAMES_16D,
    }
    info_dict["features"]["action"] = {
        "dtype": "float32",
        "shape": [16],
        "names": FEATURE_NAMES_16D,
    }
    # Remove the now-merged separate columns
    info_dict["features"].pop("observation.joint_positions", None)
    info_dict["features"].pop("action_joints", None)

    with open(info_path, "w") as f:
        json.dump(info_dict, f, indent=4)
    print("  Done.")

    # ------------------------------------------------------------------
    # Step 3: Update stats.json
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Updating stats.json ...")
    print("=" * 60)

    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats_dict = json.load(f)
    else:
        stats_dict = {}

    stats_dict["observation.state"] = compute_stats(np.array(all_states_16d))
    stats_dict["action"] = compute_stats(np.array(all_actions_16d))
    stats_dict.pop("observation.joint_positions", None)
    stats_dict.pop("action_joints", None)

    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=4)
    print("  Done.")

    # ------------------------------------------------------------------
    # Step 4: Fix episode metadata parquets (the root cause of the
    #         "Episode 0 not found" error)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Fixing episode metadata parquets ...")
    print("=" * 60)

    episodes_dir = meta_dir / "episodes"
    ep_pq_files = sorted(episodes_dir.rglob("*.parquet"))

    for ep_pq_path in ep_pq_files:
        print(f"  Fixing: {ep_pq_path.relative_to(root)}")
        orig_table = pq.read_table(ep_pq_path)
        df = orig_table.to_pandas()

        for idx in range(len(df)):
            ep_idx = int(df.iloc[idx]["episode_index"])
            if ep_idx not in episode_states:
                print(f"    Warning: episode {ep_idx} not found in data — skipping")
                continue

            ep_s = np.array(episode_states[ep_idx], dtype=np.float32)
            ep_a = np.array(episode_actions[ep_idx], dtype=np.float32) if episode_actions.get(ep_idx) else None

            # Update observation.state stats (currently 6-element → 16-element)
            s_stats = compute_stats(ep_s)
            for stat_name, stat_val in s_stats.items():
                col = f"stats/observation.state/{stat_name}"
                if col in df.columns:
                    df.at[df.index[idx], col] = stat_val

            # Update action stats
            if ep_a is not None:
                a_stats = compute_stats(ep_a)
                for stat_name, stat_val in a_stats.items():
                    col = f"stats/action/{stat_name}"
                    if col in df.columns:
                        df.at[df.index[idx], col] = stat_val

        # Drop any joint_positions / action_joints stats columns
        drop_cols = [
            c for c in df.columns
            if "observation.joint_positions" in c or "action_joints" in c
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # ----------------------------------------------------------------
        # Write back using pure PyArrow — NOT df.to_parquet() which injects
        # pandas metadata that hyparquet (the browser Parquet reader used by
        # the LeRobot visualizer Space) cannot parse, causing the
        # "Episode 0 not found in metadata" error.
        # ----------------------------------------------------------------
        keep_fields = [f for f in orig_table.schema if f.name not in drop_cols]
        new_schema = pa.schema(keep_fields)

        arrays = []
        for field in new_schema:
            col_data = df[field.name].tolist()
            arrays.append(pa.array(col_data, type=field.type))

        new_table = pa.Table.from_arrays(arrays, schema=new_schema)
        pq.write_table(new_table, ep_pq_path)

    print("  Done.")

    # ------------------------------------------------------------------
    # Step 5: Clean up temp directories
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Cleanup ...")
    print("=" * 60)

    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            print(f"  Removing temp dir: {d.name}")
            shutil.rmtree(d)
    # Also remove images/ dir if it exists (leftover from video encoding)
    images_dir = root / "images"
    if images_dir.exists():
        print(f"  Removing images/ dir (leftover from video encoding)")
        shutil.rmtree(images_dir)

    print("  Done.")

    # ------------------------------------------------------------------
    # Step 6: Validate & Push
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — Validate & Push ...")
    print("=" * 60)

    # Quick sanity check
    test_table = pq.read_table(local_data_files[0])
    test_df = test_table.to_pandas()
    sample = test_df.iloc[0]["observation.state"]
    print(f"  Sample observation.state dim: {len(sample)}")
    assert len(sample) == 16, f"Expected 16D but got {len(sample)}D!"
    print(f"  Sample values: {np.round(sample, 4)}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(root), repo_id=HF_REPO_ID, repo_type="dataset")

    print(f"\n{'=' * 60}")
    print(f"  Successfully pushed {HF_REPO_ID} with 16D features!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
