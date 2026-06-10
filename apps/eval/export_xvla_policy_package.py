#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT_DIR / "src"
LEROBOT_SRC = ROOT_DIR.parent / "repos" / "lerobot" / "src"
for path in (SRC_ROOT, LEROBOT_SRC):
    if path.exists() and str(path) not in sys.path: sys.path.insert(0, str(path))

from huggingface_hub import HfApi, snapshot_download

from thesis_vla.common.paths import RUNTIME_CACHE_DIR
from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, resolve_xvla_rename_map, sync_xvla_policy_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _default_lerobot_home() -> Path:
    user = os.environ.get("USER", "default_user")
    cache_root = RUNTIME_CACHE_DIR / f"xvla_{user}"
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_ASSETS_CACHE", str(cache_root / "assets"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(cache_root / "lerobot"))
    for key in ("HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_LEROBOT_HOME"): Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return Path(os.environ["HF_LEROBOT_HOME"])


def _resolve_pretrained_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists() and "/" in path: candidate = Path(snapshot_download(repo_id=path, repo_type="model"))
    migrated = candidate.parent / f"{candidate.name}_migrated"
    nested = candidate / "policy"
    if nested.exists() and (nested / "config.json").exists(): return nested
    if migrated.exists() and (migrated / "config.json").exists(): return migrated
    if (candidate / "config.json").exists(): return candidate
    raise FileNotFoundError(f"Could not resolve a policy package from: {path}")


def _resolve_dataset_root(dataset_root: str | None, dataset_repo_id: str) -> Path:
    base = Path(dataset_root).expanduser() if dataset_root else _default_lerobot_home()
    return base / dataset_repo_id


def main() -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    parser = argparse.ArgumentParser(description="Export a deployable XVLA policy package with processor files rebuilt from dataset stats.")
    parser.add_argument("--pretrained_path", required=True, help="HF repo id, local policy dir, or visual-thought checkpoint root.")
    parser.add_argument("--dataset_repo_id", required=True, help="Dataset repo id used to rebuild processor stats, e.g. edgarcancinoe/cloth-corner-fold_7p5hz.")
    parser.add_argument("--dataset_revision", default=None, help="Dataset revision/tag/branch.")
    parser.add_argument("--dataset_root", default=None, help="Optional local LeRobot dataset cache root.")
    parser.add_argument("--output_dir", required=True, help="Where to save the clean deployable package.")
    parser.add_argument("--device", default="cpu", help="Device for processor config, e.g. cpu, mps, cuda.")
    parser.add_argument("--repo_id", default=None, help="Optional HF repo id to push the exported package.")
    parser.add_argument("--commit_message", default="Upload repaired XVLA policy package", help="HF commit message if --repo_id is used.")
    args = parser.parse_args()

    pretrained_path = _resolve_pretrained_path(args.pretrained_path)
    dataset_root = _resolve_dataset_root(args.dataset_root, args.dataset_repo_id)
    output_dir = Path(args.output_dir).expanduser()
    logging.info(f"Loading dataset metadata from {dataset_root}")
    dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root, revision=args.dataset_revision)
    rename_map = resolve_xvla_rename_map(getattr(dataset.meta, "camera_keys", []))
    logging.info(f"Loading XVLA policy from {pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_path))
    sync_xvla_policy_config(policy_cfg, dataset.meta, rename_map)
    policy = XVLAPolicy.from_pretrained(str(pretrained_path), config=policy_cfg, device=args.device)
    preprocessor, postprocessor = make_xvla_runtime_processors(policy=policy, pretrained_path=str(pretrained_path), device=args.device, rename_map=rename_map, dataset_stats=dataset.meta.stats, use_dataset_stats=True, load_pretrained_processors=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    logging.info(f"Saved repaired package to {output_dir}")
    if args.repo_id:
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        url = api.upload_folder(folder_path=str(output_dir), repo_id=args.repo_id, repo_type="model", commit_message=args.commit_message)
        logging.info(f"Uploaded to {url}")


if __name__ == "__main__":
    main()
