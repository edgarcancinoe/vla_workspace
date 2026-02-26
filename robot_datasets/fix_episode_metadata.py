#!/usr/bin/env python3
"""
Targeted fix for "Episode 0 not found in metadata" in the LeRobot visualizer Space.

Root cause: the episode metadata parquets were written with pandas.to_parquet()
which injects pandas-specific metadata annotations into the Parquet file footer.
The Space uses hyparquet (a browser-native Parquet reader) which cannot parse
those annotations and fails to find any episodes.

Fix: rewrite all meta/episodes/*.parquet files using pure PyArrow, preserving
the original schema exactly, and push to HF.

The local data parquets are already correct 16D — this script only touches
the episode metadata.
"""

import json
import urllib.request
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
HF_REPO_ID = "edgarcancinoe/soarm101_pickplace_10d"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"
LOCAL_ROOT = Path(__file__).parent.parent / "outputs" / "datasets" / "soarm101_pickplace_10d"
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


def main():
    root = LOCAL_ROOT
    data_dir = root / "data"
    meta_dir = root / "meta"
    episodes_dir = meta_dir / "episodes"

    # ------------------------------------------------------------------
    # Step 1: Collect per-episode 16D stats from the local data parquets
    # ------------------------------------------------------------------
    print("STEP 1 — Reading 16D data and computing per-episode stats ...")
    episode_states: dict[int, list] = {}
    episode_actions: dict[int, list] = {}

    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found in {data_dir}")

    for pq_path in data_files:
        print(f"  Reading {pq_path.relative_to(root)} ...")
        table = pq.read_table(pq_path)
        df = table.to_pandas()

        obs_dim = len(df.iloc[0]["observation.state"])
        print(f"    observation.state dim = {obs_dim}")

        for i in range(len(df)):
            row = df.iloc[i]
            ep = int(row["episode_index"])
            state = np.array(row["observation.state"], dtype=np.float64)
            action = np.array(row["action"], dtype=np.float64)
            episode_states.setdefault(ep, []).append(state)
            episode_actions.setdefault(ep, []).append(action)

    print(f"  Loaded {len(episode_states)} episodes.")

    # ------------------------------------------------------------------
    # Step 2: Download the original episode metadata parquets from HF
    #         (the ones before our pandas.to_parquet broke them)
    # ------------------------------------------------------------------
    print("\nSTEP 2 — Downloading original episode metadata from HF ...")

    ep_pq_files = sorted(episodes_dir.rglob("*.parquet"))
    orig_tables: dict[Path, pa.Table] = {}

    for local_path in ep_pq_files:
        rel = str(local_path.relative_to(root))
        url = f"{HF_RESOLVE}/{rel}"
        tmp = tempfile.mktemp(suffix=".parquet")
        print(f"  Downloading {rel} ...")
        urllib.request.urlretrieve(url, tmp)
        orig_tables[local_path] = pq.read_table(tmp)

    # ------------------------------------------------------------------
    # Step 3: Rewrite each episode metadata file using pure PyArrow,
    #         updating only the stats columns for observation.state and action
    # ------------------------------------------------------------------
    print("\nSTEP 3 — Rewriting episode metadata with correct 16D stats (pure PyArrow) ...")

    for ep_pq_path, orig_table in orig_tables.items():
        print(f"  Fixing {ep_pq_path.relative_to(root)} ...")
        schema = orig_table.schema
        df = orig_table.to_pandas()

        n_rows = len(df)
        for idx in range(n_rows):
            ep_idx = int(df.iloc[idx]["episode_index"])
            if ep_idx not in episode_states:
                print(f"    Warning: episode {ep_idx} not in data, skipping stats update")
                continue

            ep_s = np.array(episode_states[ep_idx], dtype=np.float64)
            ep_a = np.array(episode_actions[ep_idx], dtype=np.float64)

            s_stats = compute_stats(ep_s)
            a_stats = compute_stats(ep_a)

            for stat_name, val in s_stats.items():
                col = f"stats/observation.state/{stat_name}"
                if col in df.columns:
                    df.at[df.index[idx], col] = val

            for stat_name, val in a_stats.items():
                col = f"stats/action/{stat_name}"
                if col in df.columns:
                    df.at[df.index[idx], col] = val

        # Build a pure PyArrow table preserving the exact original schema
        # but WITHOUT pandas metadata in the footer (hyparquet can't read it)
        # Strip pandas key from schema metadata if present
        schema_metadata = {k: v for k, v in (schema.metadata or {}).items() if k != b"pandas"}
        clean_schema = schema.with_metadata(schema_metadata)

        arrays = []
        for field in clean_schema:
            col_data = df[field.name].tolist()
            arrays.append(pa.array(col_data, type=field.type))

        new_table = pa.Table.from_arrays(arrays, schema=clean_schema)
        pq.write_table(new_table, ep_pq_path)
        print(f"    Written {len(df)} rows, schema fields: {len(schema)}")

    # ------------------------------------------------------------------
    # Step 4: Quick sanity check — read back with PyArrow and verify
    # ------------------------------------------------------------------
    print("\nSTEP 4 — Sanity check ...")
    first_ep_file = sorted(episodes_dir.rglob("*.parquet"))[0]
    check = pq.read_table(first_ep_file).to_pandas()
    print(f"  episode_index values: {check['episode_index'].tolist()}")
    print(f"  stats/observation.state/min len: {len(check.iloc[0]['stats/observation.state/min'])}")
    print(f"  stats/action/min len: {len(check.iloc[0]['stats/action/min'])}")
    # Verify no pandas metadata in schema
    raw = pq.read_metadata(first_ep_file)
    schema_meta = raw.metadata or {}
    has_pandas_meta = b"pandas" in schema_meta
    print(f"  Has pandas metadata in parquet footer: {has_pandas_meta}")
    if has_pandas_meta:
        print("  WARNING: pandas metadata still present! hyparquet may still fail.")
    else:
        print("  CLEAN: no pandas metadata — hyparquet should read this correctly.")

    # ------------------------------------------------------------------
    # Step 5: Push
    # ------------------------------------------------------------------
    print("\nSTEP 5 — Pushing to HF ...")
    from huggingface_hub import HfApi
    api = HfApi()

    # Only upload the meta/episodes folder — data and videos are already correct
    api.upload_folder(
        folder_path=str(meta_dir),
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        path_in_repo="meta",
    )
    print(f"\n✅ Done! {HF_REPO_ID}/meta pushed successfully.")


if __name__ == "__main__":
    main()
