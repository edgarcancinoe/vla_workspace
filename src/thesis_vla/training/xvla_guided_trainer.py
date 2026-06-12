from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.xvla.action_contract import get_so101_slice_spec
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from thesis_vla.policies.xvla_guided.configuration_xvla_guided import normalize_guidance_fusion_mode
from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, resolve_xvla_rename_map, sync_xvla_policy_config
from thesis_vla.training.visual_thought_trainer import JointTrainingState, XVLARuntime, _as_device, _resolve_dataset_root, _resolve_teacher_image_key, _set_seed, ensure_xvla_slice_step, optimizer_metrics, preprocess_batch, step_optimizer, trainer_state_dict, zero_optimizer_grad
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
    action_mode: str | None = None
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
    xvla_scheduler_decay_lr: float | None = None
    action_loss_weight: float = 1.0
    expert_loss_weight: float = 0.25
    teacher_image_feature_key: str = "observation.images.image"
    teacher_target_cache_root: str | None = None
    dataset_video_backend: str = "pyav"
    dataset_tolerance_s: float = 1e-4
    fusion_mode: str = "concat"
    gated_fusion: bool | None = None
    guidance_train_mode: str = "frozen"
    guidance_unfreeze_step: int = 1_000
    freeze_xvla_vlm: bool = True
    save_final_checkpoint: bool = True
    wandb_enable: bool = True
    wandb_project: str = "xvla-guided"
    wandb_run_name: str | None = None
    validation_enable: bool = True
    validation_split_ratio: float = 0.1
    validation_freq: int = 500
    validation_max_batches: int = 10
    validation_seed: int = 1337
    seed: int = 42
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.fusion_mode = normalize_guidance_fusion_mode(self.fusion_mode, self.gated_fusion)

    @classmethod
    def from_json(cls, path: str | Path) -> "GuidedXVLATrainConfig":
        payload = json.loads(Path(path).read_text())
        if "cuda_visible_devices" in payload: payload["cuda_visible_devices"] = tuple(int(device) for device in payload["cuda_visible_devices"])
        return cls(**payload)

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_xvla_runtime(config: GuidedXVLATrainConfig) -> XVLARuntime:
    
    # Recreate the same dataset-time contract as standard XVLA finetuning:
    # load the pretrained config first, then derive the delta-timestamp sampling
    dataset_root     = _resolve_dataset_root(config)
    policy_cfg       = PreTrainedConfig.from_pretrained(config.xvla_init_path)
    if config.action_mode is not None:
        if get_so101_slice_spec(config.action_mode) is None: raise ValueError(f"Unsupported guided XVLA action_mode={config.action_mode!r}.")
        policy_cfg.action_mode = str(config.action_mode)
    if hasattr(policy_cfg, "optimizer_lr"): policy_cfg.optimizer_lr = float(config.xvla_optimizer_lr)
    if config.xvla_scheduler_decay_lr is not None and hasattr(policy_cfg, "scheduler_decay_lr"): policy_cfg.scheduler_decay_lr = float(config.xvla_scheduler_decay_lr)
    ds_meta          = LeRobotDatasetMetadata(config.dataset_repo_id, root=dataset_root, revision=config.dataset_revision)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    # Feeding delta_timestamps into the dataset is what makes this trainer use the
    # same temporal supervision contract as finetune / visual-thought runs.
    dataset          = LeRobotDataset(config.dataset_repo_id, root=dataset_root, revision=config.dataset_revision, delta_timestamps=delta_timestamps, video_backend=config.dataset_video_backend, tolerance_s=float(config.dataset_tolerance_s))
    rename_map       = resolve_xvla_rename_map(getattr(dataset.meta, "camera_keys", []))
    
    # Reconcile the pretrained XVLA feature schema with the real dataset camera keys
    # and active action_mode before instantiating the runtime policy.
    sync_xvla_policy_config(policy_cfg, dataset.meta, rename_map)
    print(f"[guided-xvla] derived chunk_size={getattr(policy_cfg, 'chunk_size', None)} action_mode={getattr(policy_cfg, 'action_mode', None)} scheduler_decay_lr={getattr(policy_cfg, 'scheduler_decay_lr', None)}")
    policy_device = _as_device(config)
    if hasattr(policy_cfg, "device"): 
        policy_cfg.device = policy_device

    policy = XVLAPolicy.from_pretrained(config.xvla_init_path, config=policy_cfg, device=policy_device)
    preprocessor, postprocessor = make_xvla_runtime_processors(policy=policy, pretrained_path=config.xvla_init_path, device=policy_device, rename_map=rename_map, dataset_stats=dataset.meta.stats, use_dataset_stats=True)
    
    # Slice action/state down to the real SO101 contract before the batch reaches
    # XVLA; the policy still does its own internal time-padding and 20D model-space
    # padding later, but that happens after the dataset/preprocessor boundary.
    ensure_xvla_slice_step(preprocessor, get_so101_slice_spec(getattr(policy.config, "action_mode", None)))
    teacher_image_key = _resolve_teacher_image_key(list(getattr(dataset.meta, "camera_keys", [])), rename_map, config.teacher_image_feature_key)
    
    return XVLARuntime(policy=policy, dataset=dataset, preprocessor=preprocessor, postprocessor=postprocessor, rename_map=rename_map, teacher_image_key=teacher_image_key, policy_device=policy_device, vlm_device=policy_device, vlm_only_distill=False)


def _load_decoder_init(model, decoder_init_path: str) -> None:
    root = Path(decoder_init_path)
    state = load_decoder_state(root / DECODER_STATE_FILENAME if root.is_dir() else root)
    model.model.guidance_decoder.load_state_dict(state, strict=True)


def _build_loader_for_subset(dataset, config: GuidedXVLATrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)


def _episode_split_indices(runtime: XVLARuntime, config: GuidedXVLATrainConfig) -> tuple[list[int], list[int]] | None:
    if not config.validation_enable or float(config.validation_split_ratio) <= 0.0: return None
    hf_dataset = getattr(runtime.dataset, "hf_dataset", None)
    if hf_dataset is None: return None
    episode_column = hf_dataset["episode_index"]
    episode_indices = [int(value.item()) if torch.is_tensor(value) else int(value) for value in episode_column]
    unique_episodes = sorted(set(episode_indices))
    if len(unique_episodes) < 2: return None
    num_val = int(len(unique_episodes) * float(config.validation_split_ratio))
    if num_val <= 0 or num_val >= len(unique_episodes): return None
    generator = torch.Generator().manual_seed(int(config.validation_seed))
    permutation = torch.randperm(len(unique_episodes), generator=generator).tolist()
    val_episode_ids = {unique_episodes[index] for index in permutation[:num_val]}
    train_indices, val_indices = [], []
    for sample_index, episode_index in enumerate(episode_indices): (val_indices if episode_index in val_episode_ids else train_indices).append(sample_index)
    if not train_indices or not val_indices: return None
    return train_indices, val_indices


def build_train_val_dataloaders(runtime: XVLARuntime, config: GuidedXVLATrainConfig) -> tuple[DataLoader, DataLoader | None]:
    split = _episode_split_indices(runtime, config)
    if split is None: return _build_loader_for_subset(runtime.dataset, config, shuffle=True), None
    train_indices, val_indices = split
    return _build_loader_for_subset(Subset(runtime.dataset, train_indices), config, shuffle=True), _build_loader_for_subset(Subset(runtime.dataset, val_indices), config, shuffle=False)


def _maybe_init_wandb(config: GuidedXVLATrainConfig):
    if not config.wandb_enable: return None
    import wandb

    return wandb.init(project=config.wandb_project, name=config.wandb_run_name or config.name, config=config.to_json_dict(), dir=config.output_dir)


def _configure_explicit_stage_trainability(policy, config: GuidedXVLATrainConfig) -> None:
    if not bool(config.freeze_xvla_vlm): return
    for parameter in policy.model.vlm.parameters(): parameter.requires_grad = False
    policy.model.vlm.eval()


def _init_guided_policy_from_base(runtime: XVLARuntime, config: GuidedXVLATrainConfig, task_cfg):
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from thesis_vla.policies.xvla_guided import XVLAGuidedConfig, XVLAGuidedPolicy

    base_cfg = runtime.policy.config
    guided_cfg = XVLAGuidedConfig.from_xvla_config(
        base_cfg,
        guidance_decoder_stack=dataclasses.asdict(task_cfg.stack),
        guidance_decoder_head=dataclasses.asdict(task_cfg.head),
        guidance_decoder_teacher=dataclasses.asdict(task_cfg.teacher),
        guidance_fusion_mode=config.fusion_mode,
        guidance_gated=bool(config.gated_fusion),
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
    _configure_explicit_stage_trainability(policy, config)
    return policy


def build_optimizer(config: GuidedXVLATrainConfig, policy) -> JointTrainingState:
    decoder_params = list(policy.model.guidance_decoder.parameters())
    decoder_optimizer = torch.optim.AdamW(decoder_params, lr=config.decoder_optimizer_lr, weight_decay=config.weight_decay)
    policy_named_params = {name: parameter for name, parameter in policy.get_optim_params().items() if not name.startswith("model.guidance_decoder.")}
    policy_optimizer = policy.config.get_optimizer_preset().build(policy_named_params)
    policy_scheduler = policy.config.get_scheduler_preset().build(policy_optimizer, config.steps)
    return JointTrainingState(policy_optimizer=policy_optimizer, decoder_optimizer=decoder_optimizer, policy_scheduler=policy_scheduler)


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


@torch.no_grad()
def run_validation(config: GuidedXVLATrainConfig, runtime: XVLARuntime, policy, cache: CeDiRNetTargetCache, val_loader: DataLoader, prefix: str = "val") -> dict[str, float]:
    policy_was_training = policy.training
    policy.eval()
    total_loss, total_action, total_expert, batches = 0.0, 0.0, 0.0, 0
    component_totals: dict[str, float] = {}
    for raw_batch in val_loader:
        if batches >= max(int(config.validation_max_batches), 1): break
        processed_batch = preprocess_batch(runtime, raw_batch)
        inputs = policy._build_model_inputs(processed_batch)
        enc = policy.model.forward_vlm(input_ids=inputs["input_ids"], pixel_values=inputs["image_input"], image_mask=inputs["image_mask"])
        target = load_guidance_target(cache, raw_batch, config)
        action_loss, action_stats, guidance_tokens = compute_guided_action_loss_from_encoder(policy, processed_batch, inputs, enc)
        expert_loss, expert_stats = compute_guidance_loss(policy, target, guidance_tokens) if float(config.expert_loss_weight) > 0.0 else (action_loss.new_zeros(()), {"expert_total": 0.0})
        total = float(config.action_loss_weight) * action_loss + float(config.expert_loss_weight) * expert_loss
        for key, value in action_stats.items(): component_totals[f"{prefix}_{key}"] = component_totals.get(f"{prefix}_{key}", 0.0) + float(value)
        for key, value in expert_stats.items(): component_totals[f"{prefix}_{key}"] = component_totals.get(f"{prefix}_{key}", 0.0) + float(value)
        total_loss += float(total.detach().item())
        total_action += float(action_loss.detach().item())
        total_expert += float(expert_loss.detach().item())
        batches += 1
    if policy_was_training:
        policy.train()
        if config.freeze_xvla_vlm: policy.model.vlm.eval()
    denom = max(batches, 1)
    metrics = {f"{prefix}_loss": total_loss / denom, f"{prefix}_action_total": total_action / denom, f"{prefix}_expert_total": total_expert / denom, f"{prefix}_batches": float(batches)}
    metrics.update({key: value / denom for key, value in component_totals.items()})
    return metrics


def _save_checkpoint(config: GuidedXVLATrainConfig, runtime: XVLARuntime, optimizer: torch.optim.Optimizer | JointTrainingState, step: int, final: bool = False) -> None:
    if not final and (config.save_every <= 0 or step % config.save_every != 0): return
    checkpoint_dir = Path(config.output_dir) / ("checkpoint_final" if final else f"checkpoint_{step:07d}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime.policy.save_pretrained(checkpoint_dir)
    if runtime.preprocessor is not None: runtime.preprocessor.save_pretrained(checkpoint_dir)
    if runtime.postprocessor is not None: runtime.postprocessor.save_pretrained(checkpoint_dir)
    torch.save({"step": int(step), **trainer_state_dict(optimizer)}, checkpoint_dir / TRAINER_STATE_FILENAME)
    (checkpoint_dir / METADATA_FILENAME).write_text(json.dumps({"name": config.name, "global_step": int(step), "xvla_init_path": config.xvla_init_path, "decoder_init_path": config.decoder_init_path, "fusion_mode": config.fusion_mode, "freeze_xvla_vlm": bool(config.freeze_xvla_vlm), "action_mode": config.action_mode or getattr(runtime.policy.config, "action_mode", None), "xvla_scheduler_decay_lr": config.xvla_scheduler_decay_lr if config.xvla_scheduler_decay_lr is not None else getattr(runtime.policy.config, "scheduler_decay_lr", None)}, indent=2, sort_keys=True))
    (checkpoint_dir / CONFIG_FILENAME).write_text(json.dumps(config.to_json_dict(), indent=2, sort_keys=True))


def train_guided_xvla(config: GuidedXVLATrainConfig) -> None:
    _set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(config)
    runtime = build_xvla_runtime(config)
    task_cfg = load_cedirnet_decoder_config(config.decoder_stack_config_path, config.decoder_task_config_path)
    cache = CeDiRNetTargetCache.resolve(dataset_repo_id=config.dataset_repo_id, dataset_revision=config.dataset_revision, dataset_root=_resolve_dataset_root(config), dataset_length=len(runtime.dataset), teacher_cfg=task_cfg.teacher, cache_root=config.teacher_target_cache_root)
    policy = _init_guided_policy_from_base(runtime, config, task_cfg)
    runtime.policy = policy
    optimizer = build_optimizer(config, policy)
    loader, val_loader = build_train_val_dataloaders(runtime, config)
    if config.dry_run:
        if wandb_run is not None: wandb_run.finish()
        return
    policy.train()
    if config.freeze_xvla_vlm: policy.model.vlm.eval()
    step = 0
    accum_steps = max(int(config.gradient_accumulation_steps), 1)
    zero_optimizer_grad(optimizer)
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
            (total_loss / accum_steps).backward()
            if step % accum_steps == 0 or step == config.steps:
                step_optimizer(optimizer)
                zero_optimizer_grad(optimizer)
            if step % max(int(config.log_every), 1) == 0:
                metrics = {
                    "event": "train_step",
                    "step": int(step),
                    "loss": float(total_loss.detach().item()),
                    **action_stats,
                    **expert_stats,
                    **optimizer_metrics(optimizer),
                }
                print(json.dumps(metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in metrics.items() if key != "event"}, step=int(step))
                progress.set_postfix({"loss": f"{float(total_loss.detach().item()):.4f}", "action": f"{action_stats['action_total']:.4f}", "expert": f"{expert_stats['expert_total']:.4f}"})
            if val_loader is not None and (step % max(int(config.validation_freq), 1) == 0 or step == config.steps):
                val_metrics = {"event": "validation_step", "step": int(step), **run_validation(config, runtime, policy, cache, val_loader, prefix="val")}
                print(json.dumps(val_metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in val_metrics.items() if key != "event"}, step=int(step))
            progress.update(1)
            _save_checkpoint(config, runtime, optimizer, step, final=False)
            if step >= config.steps: break
    progress.close()
    if config.save_final_checkpoint: _save_checkpoint(config, runtime, optimizer, step, final=True)
    if wandb_run is not None: wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train guided XVLA policy with CeDirNet late-fusion guidance.")
    parser.add_argument("--config_path", required=True, help="Path to guided XVLA training config JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_guided_xvla(GuidedXVLATrainConfig.from_json(args.config_path))


if __name__ == "__main__":
    main()
