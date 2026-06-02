import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ROOT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_DATASETS_CACHE", str((ROOT_DIR / "runtime" / "cache" / "hf_datasets").resolve()))
src_root = ROOT_DIR / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
lerobot_src = ROOT_DIR.parent / "repos" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from thesis_vla.common.paths import DATASETS_OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Fast merge multiple LeRobot datasets without frame re-encoding.")
    p.add_argument("--src", action="append", required=True, dest="src_repo_ids", help="Source dataset repo id. Pass multiple times.")
    p.add_argument("--src-root", action="append", dest="src_roots", help="Optional local source root. Pass in the same order as --src.")
    p.add_argument("--dst", required=True, dest="dst_repo_id", help="Destination dataset repo id.")
    p.add_argument("--dst-root", dest="dst_root", help="Optional local destination root.")
    p.add_argument("--push", action="store_true", help="Push merged dataset to Hugging Face after local merge.")
    return p.parse_args()


def resolve_src_roots(repo_ids, src_roots):
    if src_roots and len(src_roots) != len(repo_ids):
        raise ValueError("--src-root must be provided the same number of times as --src.")
    if src_roots:
        return [Path(p).expanduser().resolve() for p in src_roots]
    return [(DATASETS_OUTPUT_DIR / repo_id.split("/")[-1]).resolve() for repo_id in repo_ids]


def ensure_local_sources(repo_ids, roots):
    metas = []
    for repo_id, root in zip(repo_ids, roots, strict=True):
        if not root.exists():
            raise FileNotFoundError(f"Missing local dataset root for {repo_id}: {root}")
        meta = LeRobotDatasetMetadata(repo_id, root=root, revision="v3.0")
        if meta.total_episodes <= 0:
            raise ValueError(f"Dataset has no episodes: {repo_id} at {root}")
        metas.append(meta)
        print(f"{repo_id}: episodes={meta.total_episodes}, frames={meta.total_frames}, fps={meta.fps}, root={root}")
    return metas


def push_dataset(dst_repo_id, dst_root, codebase_version):
    api = HfApi()
    api.create_repo(repo_id=dst_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(dst_root), repo_id=dst_repo_id, repo_type="dataset", ignore_patterns=[".cache/**", "**/.cache/**", ".DS_Store", "**/.DS_Store"], delete_patterns=["data/**", "meta/**", "videos/**", "images/**", "tmp*", "tmp*/**"], commit_message="Upload merged dataset")
    if codebase_version:
        try:
            api.create_tag(repo_id=dst_repo_id, tag=codebase_version, repo_type="dataset", revision="main", exist_ok=True)
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 409:
                api.delete_tag(repo_id=dst_repo_id, tag=codebase_version, repo_type="dataset")
                api.create_tag(repo_id=dst_repo_id, tag=codebase_version, repo_type="dataset", revision="main")
            else:
                raise
    print(f"Pushed -> https://huggingface.co/datasets/{dst_repo_id}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src_roots = resolve_src_roots(args.src_repo_ids, args.src_roots)
    dst_root = Path(args.dst_root).expanduser().resolve() if args.dst_root else (DATASETS_OUTPUT_DIR / args.dst_repo_id.split("/")[-1]).resolve()
    print("Source datasets:")
    metas = ensure_local_sources(args.src_repo_ids, src_roots)
    if dst_root.exists():
        raise FileExistsError(f"Destination already exists. Remove it first: {dst_root}")
    print(f"Merging into {args.dst_repo_id} at {dst_root}")
    aggregate_datasets(repo_ids=args.src_repo_ids, roots=src_roots, aggr_repo_id=args.dst_repo_id, aggr_root=dst_root)
    merged_meta = LeRobotDatasetMetadata(args.dst_repo_id, root=dst_root, revision="v3.0")
    print(f"Merged dataset: episodes={merged_meta.total_episodes}, frames={merged_meta.total_frames}, fps={merged_meta.fps}, root={dst_root}")
    if args.push:
        push_dataset(args.dst_repo_id, dst_root, merged_meta.info.get("codebase_version"))


if __name__ == "__main__":
    main()
