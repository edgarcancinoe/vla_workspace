from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thesis_vla.inference.xvla_runtime import sync_xvla_policy_config
from thesis_vla.training.visual_thought_trainer import XVLARuntime, _as_device, _resolve_dataset_root, _resolve_teacher_image_key, _set_seed, build_dataloader, preprocess_batch
from thesis_vla.visual_thought import load_cedirnet_decoder_config
from thesis_vla.visual_thought.cedirnet_cache import CeDiRNetTargetCache
from thesis_vla.visual_thought.checkpoints import DECODER_STATE_FILENAME, load_decoder_state
from thesis_vla.visual_thought.targets import compute_teacher_loss


TRAINER_STATE_FILENAME = "trainer_state.pt"
METADATA_FILENAME = "metadata.json"
CONFIG_FILENAME = "guided_training_config.json"


@dataclass
class GuidedXVLATrainConfig:
    name: str
    xvla_init_path: str
    decoder_init_path: str
    decoder_stack_config_path: str
    decoder_task_config_path: str
    dataset_repo_id: str
    dataset_revision: str | None
    dataset_root: str | None
    output_dir: str
    device: str
    cuda_visible_devices: tuple[int, ...] = (0,)
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    steps: int = 2_500
    log_every: int = 20
    save_every: int = 500
    weight_decay: float = 0.01
    decoder_optimizer_lr: float = 1e-4
    xvla_optimizer_lr: float = 1e-5
    action_loss_weight: float = 1.0
    expert_loss_weight: float = 0.25
    teacher_image_feature_key: str = "observation.images.image"
    teacher_target_cache_root: str | None = None
    dataset_video_backend: str = "pyav"
    dataset_tolerance_s: float = 1e-4
    fusion_mode: str = "concat"
    gated_fusion: bool = False
    guidance_train_mode: str = "warmup_freeze"
    guidance_unfreeze_step: int = 1_000
    save_final_checkpoint: bool = True
    seed: int = 42
    dry_run: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "GuidedXVLATrainConfig":
        payload = json.loads(Path(path).read_text())
        if "cuda_visible_devices" in payload: payload["cuda_visible_devices"] = tuple(int(device) for device in payload["cuda_visible_devices"])
        return cls(**payload)

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_xvla_runtime(config: GuidedXVLATrainConfig) -> XVLARuntime:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    dataset = LeRobotDataset(config.dataset_repo_id, root=_resolve_dataset_root(config), revision=config.dataset_revision, video_backend=config.dataset_video_backend, tolerance_s=float(config.dataset_tolerance_s))
    from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, resolve_xvla_rename_map

    rename_map = resolve_xvla_rename_map(getattr(dataset.meta, "camera_keys", []))
    policy_cfg = PreTrainedConfig.from_pretrained(config.xvla_init_path)
    sync_xvla_policy_config(policy_cfg, dataset.meta, rename_map)
    policy = XVLAPolicy.from_pretrained(config.xvla_init_path, config=policy_cfg, device=_as_device(config))
    preprocessor, postprocessor = make_xvla_runtime_processors(policy=policy, pretrained_path=config.xvla_init_path, device=_as_device(config), rename_map=rename_map, dataset_stats=dataset.meta.stats, use_dataset_stats=True)
    teacher_image_key = _resolve_teacher_image_key(list(getattr(dataset.meta, "camera_keys", [])), rename_map, config.teacher_image_feature_key)
    return XVLARuntime(policy=policy, dataset=dataset, preprocessor=preprocessor, postprocessor=postprocessor, rename_map=rename_map, teacher_image_key=teacher_image_key)


def _load_decoder_init(model, decoder_init_path: str) -> None:
    root = Path(decoder_init_path)
    state = load_decoder_state(root / DECODER_STATE_FILENAME if root.is_dir() else root)
    model.model.guidance_decoder.load_state_dict(state, strict=True)


def _init_guided_policy_from_base(runtime: XVLARuntime, config: GuidedXVLATrainConfig, task_cfg):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from thesis_vla.policies.xvla_guided import XVLAGuidedConfig, XVLAGuidedPolicy

    base_cfg = PreTrainedConfig.from_pretrained(config.xvla_init_path)
    sync_xvla_policy_config(base_cfg, runtime.dataset.meta, runtime.rename_map)
    guided_cfg = XVLAGuidedConfig.from_xvla_config(
        base_cfg,
        guidance_decoder_stack=dataclasses.asdict(task_cfg.stack),
        guidance_decoder_head=dataclasses.asdict(task_cfg.head),
        guidance_decoder_teacher=dataclasses.asdict(task_cfg.teacher),
        guidance_fusion_mode=config.fusion_mode,
        guidance_gated=config.gated_fusion,
        guidance_train_mode=config.guidance_train_mode,
        guidance_unfreeze_step=config.guidance_unfreeze_step,
    )
    guided_cfg.device = _as_device(config)
    policy = XVLAGuidedPolicy(guided_cfg)
    state_dict = dict(runtime.policy.state_dict())
    XVLAPolicy._resize_positional_embedding_if_needed(state_dict, policy.model.transformer.pos_emb)
    incompat = policy.load_state_dict(state_dict, strict=False)
    if getattr(incompat, "missing_keys", None): print(f"[guided-init] missing keys: {incompat.missing_keys}")
    if getattr(incompat, "unexpected_keys", None): print(f"[guided-init] unexpected keys: {incompat.unexpected_keys}")
    _load_decoder_init(policy, config.decoder_init_path)
    policy.to(_as_device(config))
    return policy


def build_optimizer(config: GuidedXVLATrainConfig, policy) -> torch.optim.Optimizer:
    decoder_params, xvla_params = [], []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad: continue
        if name.startswith("model.guidance_decoder."): decoder_params.append(parameter)
        else: xvla_params.append(parameter)
    param_groups = []
    if decoder_params: param_groups.append({"params": decoder_params, "lr": config.decoder_optimizer_lr})
    if xvla_params: param_groups.append({"params": xvla_params, "lr": config.xvla_optimizer_lr})
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def compute_guided_action_loss_from_encoder(policy, processed_batch: dict[str, Any], inputs: dict[str, torch.Tensor], enc: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    targets = policy._prepare_action_targets(processed_batch)
    target_dtype = policy.model._get_target_dtype()
    t, action_noisy = policy.model._build_corrupted_action(action=targets, device=inputs["input_ids"].device, target_dtype=target_dtype)
    proprio_m, action_noisy_m = policy.model.action_space.preprocess(inputs["proprio"].to(dtype=target_dtype), action_noisy)
    guidance_tokens = policy.model.guidance_tokens(enc["vlm_features"])
    pred_action = policy.model.transformer(domain_id=inputs["domain_id"], action_with_noise=action_noisy_m, t=t, proprio=proprio_m, guidance_tokens=guidance_tokens, **enc)
    loss_dict = policy.model.action_space.compute_loss(pred_action, targets)
    action_loss = sum(loss_dict.values())
    stats = {key: float(value.detach().item()) for key, value in loss_dict.items()}
    stats["action_total"] = float(action_loss.detach().item())
    return action_loss, stats, guidance_tokens


def compute_guidance_loss(policy, target, guidance_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = policy.model.guidance_decoder.predict_from_tokens(guidance_tokens, target_map=target.tensor)
    loss = compute_teacher_loss(prediction, target)
    return loss, {"expert_total": float(loss.detach().item())}


def load_guidance_target(cache: CeDiRNetTargetCache, raw_batch: dict[str, Any], config: GuidedXVLATrainConfig):
    return cache.target_for_batch(raw_batch, device=_as_device(config))


def _save_checkpoint(config: GuidedXVLATrainConfig, runtime: XVLARuntime, optimizer: torch.optim.Optimizer, step: int, final: bool = False) -> None:
    if not final and (config.save_every <= 0 or step % config.save_every != 0): return
    checkpoint_dir = Path(config.output_dir) / ("checkpoint_final" if final else f"checkpoint_{step:07d}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime.policy.save_pretrained(checkpoint_dir)
    if runtime.preprocessor is not None: runtime.preprocessor.save_pretrained(checkpoint_dir)
    if runtime.postprocessor is not None: runtime.postprocessor.save_pretrained(checkpoint_dir)
    torch.save({"step": int(step), "optimizer": optimizer.state_dict()}, checkpoint_dir / TRAINER_STATE_FILENAME)
    (checkpoint_dir / METADATA_FILENAME).write_text(json.dumps({"name": config.name, "global_step": int(step), "xvla_init_path": config.xvla_init_path, "decoder_init_path": config.decoder_init_path, "fusion_mode": config.fusion_mode, "gated_fusion": bool(config.gated_fusion)}, indent=2, sort_keys=True))
    (checkpoint_dir / CONFIG_FILENAME).write_text(json.dumps(config.to_json_dict(), indent=2, sort_keys=True))


def train_guided_xvla(config: GuidedXVLATrainConfig) -> None:
    _set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    runtime = build_xvla_runtime(config)
    task_cfg = load_cedirnet_decoder_config(config.decoder_stack_config_path, config.decoder_task_config_path)
    cache = CeDiRNetTargetCache.resolve(dataset_repo_id=config.dataset_repo_id, dataset_revision=config.dataset_revision, dataset_root=_resolve_dataset_root(config), dataset_length=len(runtime.dataset), teacher_cfg=task_cfg.teacher, cache_root=config.teacher_target_cache_root)
    policy = _init_guided_policy_from_base(runtime, config, task_cfg)
    runtime.policy = policy
    optimizer = build_optimizer(config, policy)
    loader = build_dataloader(runtime, config)
    if config.dry_run: return
    policy.train()
    step = 0
    progress = tqdm(total=config.steps, desc=config.name)
    while step < config.steps:
        for raw_batch in loader:
            if step >= config.steps: break
            step += 1
            policy.model.set_guidance_trainability(step)
            processed_batch = preprocess_batch(runtime, raw_batch)
            inputs = policy._build_model_inputs(processed_batch)
            enc = policy.model.forward_vlm(input_ids=inputs["input_ids"], pixel_values=inputs["image_input"], image_mask=inputs["image_mask"])
            target = load_guidance_target(cache, raw_batch, config)
            action_loss, action_stats, guidance_tokens = compute_guided_action_loss_from_encoder(policy, processed_batch, inputs, enc)
            if float(config.expert_loss_weight) > 0.0:
                expert_loss, expert_stats = compute_guidance_loss(policy, target, guidance_tokens)
            else:
                expert_loss = action_loss.new_zeros(())
                expert_stats = {"expert_total": 0.0}
            total_loss = float(config.action_loss_weight) * action_loss + float(config.expert_loss_weight) * expert_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            if step % max(int(config.log_every), 1) == 0:
                progress.set_postfix({"loss": f"{float(total_loss.detach().item()):.4f}", "action": f"{action_stats['action_total']:.4f}", "expert": f"{expert_stats['expert_total']:.4f}"})
            progress.update(1)
            _save_checkpoint(config, runtime, optimizer, step, final=False)
            if step >= config.steps: break
    progress.close()
    if config.save_final_checkpoint: _save_checkpoint(config, runtime, optimizer, step, final=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train guided XVLA policy with CeDirNet late-fusion guidance.")
    parser.add_argument("--config_path", required=True, help="Path to guided XVLA training config JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_guided_xvla(GuidedXVLATrainConfig.from_json(args.config_path))


if __name__ == "__main__":
    main()
