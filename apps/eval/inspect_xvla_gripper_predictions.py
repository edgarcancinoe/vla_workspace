#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from thesis_vla.common.paths import DATASETS_OUTPUT_DIR, GRIPPER_INSPECTION_DIR

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.xvla.action_contract import get_so101_slice_spec, slice_dataset_meta_in_place
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

from thesis_vla.inference.xvla_runtime import (
    make_xvla_runtime_processors,
    resolve_xvla_rename_map,
    sync_xvla_policy_config,
)

# ============================================================================
# CONFIGURATION
# Edit these values, then run:
# python '/Users/edgarcancino/Documents/Academic/EMAI Thesis/vla_workspace/scripts/inspect_xvla_gripper_predictions.py'
# ============================================================================
# POLICY_PATH = "edgarcancinoe/xvla-base_soarm101_pickplace_10d_so101_ee6d_a-m_s-m_v1"
POLICY_PATH = "edgarcancinoe/xvla-base_soarm101_pickplace_10d_so101_joint_a-m_s-m_v1"
DATASET_REPO_ID = "edgarcancinoe/soarm101_pickplace_10d"
DATASET_ROOT = DATASETS_OUTPUT_DIR / "soarm101_pickplace_10d"
EPISODES = [0]
FRAME_PERCENTAGES = [0, 20, 40, 60, 80]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = GRIPPER_INSPECTION_DIR
VIDEO_BACKEND = "pyav"


def sanitize_name(value: str) -> str:
    safe = value.replace("/", "_").replace(":", "_").replace(" ", "_")
    return "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-", "."})


def build_action_delta_timestamps(chunk_size: int, fps: int) -> dict[str, list[float]]:
    return {"action": [step / float(fps) for step in range(chunk_size)]}


def build_observation_input(item: dict) -> dict:
    observation = {}
    for key, value in item.items():
        if not key.startswith("observation."):
            continue
        observation[key] = value
    if "task" in item:
        observation["task"] = item["task"]
    return observation


def sample_episode_frames(dataset: LeRobotDataset, episode_index: int, percentages: list[int]) -> list[int]:
    ep = dataset.meta.episodes[episode_index]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    length = end - start
    if length <= 0:
        raise ValueError(f"Episode {episode_index} has invalid length: {length}")

    sampled = []
    for pct in percentages:
        clipped_pct = min(max(pct, 0), 100)
        local_idx = int(np.floor((length - 1) * (clipped_pct / 100.0)))
        sampled.append(start + local_idx)
    return list(dict.fromkeys(sampled))


def compute_chunk_metrics(pred_bin: np.ndarray, label_bin: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask.astype(bool)
    if valid.sum() == 0:
        return {
            "target_zero_count": 0.0,
            "target_one_count": 0.0,
            "pred_zero_count": 0.0,
            "pred_one_count": 0.0,
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "accuracy": float("nan"),
        }

    pred_valid = pred_bin[valid]
    label_valid = label_bin[valid]

    tp = float(np.logical_and(pred_valid == 1, label_valid == 1).sum())
    tn = float(np.logical_and(pred_valid == 0, label_valid == 0).sum())
    fp = float(np.logical_and(pred_valid == 1, label_valid == 0).sum())
    fn = float(np.logical_and(pred_valid == 0, label_valid == 1).sum())
    total = tp + tn + fp + fn

    return {
        "target_zero_count": float((label_valid == 0).sum()),
        "target_one_count": float((label_valid == 1).sum()),
        "pred_zero_count": float((pred_valid == 0).sum()),
        "pred_one_count": float((pred_valid == 1).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total > 0 else float("nan"),
    }


def load_policy_and_processors(policy_path: str, dataset: LeRobotDataset, device: str):
    config = PreTrainedConfig.from_pretrained(policy_path)
    if config.type != "xvla":
        raise ValueError(f"Expected an XVLA checkpoint, got policy type {config.type!r}")

    config.device = device
    slice_spec = get_so101_slice_spec(getattr(config, "action_mode", None))
    if slice_spec is not None:
        slice_dataset_meta_in_place(dataset.meta, slice_spec)

    rename_map = resolve_xvla_rename_map(dataset.meta.camera_keys)
    if not rename_map:
        raise ValueError(f"Unable to resolve XVLA rename_map from camera keys: {dataset.meta.camera_keys}")

    sync_xvla_policy_config(config, dataset.meta, rename_map)
    policy = XVLAPolicy.from_pretrained(policy_path, config=config, device=device)
    preprocessor, postprocessor = make_xvla_runtime_processors(
        policy=policy,
        pretrained_path=policy_path,
        device=device,
        rename_map=rename_map,
        dataset_stats=dataset.meta.stats,
        use_dataset_stats=True,
    )
    policy.eval()
    return policy, preprocessor, postprocessor


def save_episode_outputs(output_dir: Path, checkpoint_name: str, episode_index: int, rows: list[dict], plots: list[dict], threshold: float) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{checkpoint_name}_episode_{episode_index:03d}_gripper.png"
    csv_path = output_dir / f"{checkpoint_name}_episode_{episode_index:03d}_gripper.csv"

    fieldnames = ["episode_index", "frame_index", "sample_rank", "chunk_offset", "is_pad", "pred_gripper", "label_gripper", "pred_bin", "label_bin", "correct"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(plots), 1, figsize=(12, max(4, 3.6 * len(plots))), squeeze=False)
    axes = axes[:, 0]

    for ax, plot in zip(axes, plots):
        x = plot["chunk_offsets"]
        ax.plot(x, plot["label_gripper"], label="label", color="tab:blue", linewidth=2)
        ax.plot(x, plot["pred_gripper"], label="prediction", color="tab:red", linewidth=2)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")

        padded = plot["is_pad"].astype(bool)
        if padded.any():
            ax.scatter(x[padded], plot["label_gripper"][padded], color="tab:blue", marker="o", alpha=0.3, label="label_pad")
            ax.scatter(x[padded], plot["pred_gripper"][padded], color="tab:red", marker="x", alpha=0.3, label="pred_pad")
        ax2 = ax.twinx()
        ax2.step(x, plot["label_bin"], where="mid", color="tab:blue", linestyle=":", alpha=0.7)
        ax2.step(x, plot["pred_bin"], where="mid", color="tab:red", linestyle=":", alpha=0.7)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_ylabel("bin")

        metrics = plot["metrics"]
        ax.set_title(f"Episode {episode_index} | frame {plot['frame_index']} | chunk_acc={metrics['accuracy']:.3f} | tp={metrics['tp']:.0f} tn={metrics['tn']:.0f} fp={metrics['fp']:.0f} fn={metrics['fn']:.0f}")
        ax.set_xlabel("chunk offset")
        ax.set_ylabel("gripper")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path, csv_path


def main() -> None:
    episode_indices = list(EPISODES)
    frame_percentages = list(FRAME_PERCENTAGES)
    checkpoint_name = sanitize_name(POLICY_PATH.split("/")[-1])

    policy_config = PreTrainedConfig.from_pretrained(POLICY_PATH)
    if policy_config.type != "xvla":
        raise ValueError(f"Expected an XVLA checkpoint, got policy type {policy_config.type!r}")

    dataset_probe = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        video_backend=VIDEO_BACKEND,
    )
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=build_action_delta_timestamps(
            chunk_size=int(policy_config.chunk_size),
            fps=int(dataset_probe.fps),
        ),
        video_backend=VIDEO_BACKEND,
    )
    policy, preprocessor, postprocessor = load_policy_and_processors(POLICY_PATH, dataset, DEVICE)

    gripper_idx = int(policy.model.action_space.gripper_idx[0])
    threshold = float(policy.config.gripper_open_threshold)

    for episode_index in episode_indices:
        frame_indices = sample_episode_frames(dataset, episode_index, frame_percentages)
        csv_rows: list[dict] = []
        plots: list[dict] = []

        for sample_rank, frame_index in enumerate(frame_indices):
            item = dataset[frame_index]
            observation = build_observation_input(item)

            with torch.inference_mode():
                processed_obs = preprocessor(observation)
                raw_pred_chunk = policy.predict_action_chunk(processed_obs)
                pred_chunk = postprocessor(raw_pred_chunk).detach().to(dtype=torch.float32).cpu().squeeze(0)

            label_chunk = torch.as_tensor(item["action"]).detach().to(dtype=torch.float32).cpu()
            pad_mask = torch.as_tensor(item.get("action_is_pad", torch.zeros(label_chunk.shape[0], dtype=torch.bool)))
            pad_mask = pad_mask.detach().cpu().bool().numpy().reshape(-1)

            pred_gripper = pred_chunk[..., gripper_idx].numpy().reshape(-1)
            label_gripper = label_chunk[..., gripper_idx].numpy().reshape(-1)
            pred_bin = (pred_gripper > threshold).astype(np.int64)
            label_bin = (label_gripper > threshold).astype(np.int64)
            valid_mask = ~pad_mask
            metrics = compute_chunk_metrics(pred_bin, label_bin, valid_mask)

            chunk_offsets = np.arange(len(pred_gripper))
            for chunk_offset in chunk_offsets:
                csv_rows.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "sample_rank": sample_rank,
                        "chunk_offset": int(chunk_offset),
                        "is_pad": int(pad_mask[chunk_offset]),
                        "pred_gripper": float(pred_gripper[chunk_offset]),
                        "label_gripper": float(label_gripper[chunk_offset]),
                        "pred_bin": int(pred_bin[chunk_offset]),
                        "label_bin": int(label_bin[chunk_offset]),
                        "correct": int(pred_bin[chunk_offset] == label_bin[chunk_offset]),
                    }
                )

            plots.append(
                {
                    "frame_index": frame_index,
                    "chunk_offsets": chunk_offsets,
                    "pred_gripper": pred_gripper,
                    "label_gripper": label_gripper,
                    "pred_bin": pred_bin,
                    "label_bin": label_bin,
                    "is_pad": pad_mask.astype(np.int64),
                    "metrics": metrics,
                }
            )

        png_path, csv_path = save_episode_outputs(
            output_dir=OUTPUT_DIR,
            checkpoint_name=checkpoint_name,
            episode_index=episode_index,
            rows=csv_rows,
            plots=plots,
            threshold=threshold,
        )
        print(f"[x] Saved PNG: {png_path}")
        print(f"[x] Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
