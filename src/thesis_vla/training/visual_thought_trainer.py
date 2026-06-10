from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from thesis_vla.common.paths import RUNTIME_CACHE_DIR
from thesis_vla.inference.xvla_runtime import make_xvla_runtime_processors, resolve_xvla_rename_map, sync_xvla_policy_config
from thesis_vla.visual_thought import CeDirNetDistillationModel, DinoFeatureAlignmentModel, DinoTokenSequenceModel, compute_feature_alignment_loss, load_cedirnet_decoder_config, load_dino_decoder_config
from thesis_vla.visual_thought.cedirnet_cache import CeDiRNetTargetCache
from thesis_vla.visual_thought.checkpoints import DECODER_STATE_FILENAME, POLICY_DIRNAME, load_decoder_state, load_visual_thought_checkpoint_metadata, save_visual_thought_checkpoint
from thesis_vla.visual_thought.targets import TeacherTarget, compute_teacher_loss
from thesis_vla.visual_thought.teachers import DinoV2Teacher


TrainingStage = Literal["distill_only", "joint_multitask"]
ExpertType = Literal["cedirnet", "dino"]


@dataclass
class VisualThoughtTrainConfig:
    name: str
    training_stage: TrainingStage
    expert_type: ExpertType
    xvla_init_path: str
    decoder_init_path: str | None
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
    steps: int = 1_000
    log_every: int = 20
    save_every: int = 500
    weight_decay: float = 0.01
    decoder_optimizer_lr: float = 1e-4
    xvla_optimizer_lr: float = 1e-5
    action_loss_weight: float = 1.0
    expert_loss_weight: float = 1.0
    teacher_image_feature_key: str = "observation.images.image"
    teacher_target_cache_root: str | None = None
    dataset_video_backend: str = "pyav"
    dataset_tolerance_s: float = 1e-4
    wandb_enable: bool = False
    wandb_project: str = "visual-thought"
    wandb_run_name: str | None = None
    validation_enable: bool = False
    validation_split_ratio: float = 0.1
    validation_freq: int = 500
    validation_max_batches: int = 10
    validation_seed: int = 1337
    push_to_hub: bool = False
    push_repo_id: str | None = None
    push_every: int = 0
    align_feature_until_step: int = 0
    save_final_checkpoint: bool = True
    seed: int = 42
    dry_run: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "VisualThoughtTrainConfig":
        payload = json.loads(Path(path).read_text())
        if "cuda_visible_devices" in payload: payload["cuda_visible_devices"] = tuple(int(device) for device in payload["cuda_visible_devices"])
        return cls(**payload)

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class XVLARuntime:
    policy: Any
    dataset: Any
    preprocessor: Any
    postprocessor: Any
    rename_map: dict[str, str]
    teacher_image_key: str


def _as_device(config: VisualThoughtTrainConfig) -> str:
    if config.device == "cuda" and torch.cuda.is_available(): return config.device
    return "cpu"


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def _default_lerobot_home() -> Path:
    user = os.environ.get("USER", "default_user")
    cache_root = RUNTIME_CACHE_DIR / f"xvla_{user}"
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_ASSETS_CACHE", str(cache_root / "assets"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(cache_root / "lerobot"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_ASSETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_LEROBOT_HOME"]).mkdir(parents=True, exist_ok=True)
    return Path(os.environ["HF_LEROBOT_HOME"])


def _resolve_dataset_root(config: VisualThoughtTrainConfig) -> Path | None:
    return Path(config.dataset_root) / config.dataset_repo_id if config.dataset_root else _default_lerobot_home() / config.dataset_repo_id


def _resolve_teacher_image_key(camera_keys: list[str], rename_map: dict[str, str], requested_key: str) -> str:
    reverse = {dst: src for src, dst in rename_map.items()}
    if requested_key in reverse: return reverse[requested_key]
    if requested_key in camera_keys: return requested_key
    if reverse.get("observation.images.image") is not None: return reverse["observation.images.image"]
    if "observation.images.image" in camera_keys: return "observation.images.image"
    return camera_keys[0]


def build_xvla_runtime(config: VisualThoughtTrainConfig) -> XVLARuntime:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    dataset = LeRobotDataset(config.dataset_repo_id, root=_resolve_dataset_root(config), revision=config.dataset_revision, video_backend=config.dataset_video_backend, tolerance_s=float(config.dataset_tolerance_s))
    rename_map = resolve_xvla_rename_map(getattr(dataset.meta, "camera_keys", []))
    policy_cfg = PreTrainedConfig.from_pretrained(config.xvla_init_path)
    sync_xvla_policy_config(policy_cfg, dataset.meta, rename_map)
    policy = XVLAPolicy.from_pretrained(config.xvla_init_path, config=policy_cfg, device=_as_device(config))
    policy = policy.to(dtype=torch.float32)
    preprocessor, postprocessor = make_xvla_runtime_processors(policy=policy, pretrained_path=config.xvla_init_path, device=_as_device(config), rename_map=rename_map, dataset_stats=dataset.meta.stats, use_dataset_stats=True)
    teacher_image_key = _resolve_teacher_image_key(list(getattr(dataset.meta, "camera_keys", [])), rename_map, config.teacher_image_feature_key)
    return XVLARuntime(policy=policy, dataset=dataset, preprocessor=preprocessor, postprocessor=postprocessor, rename_map=rename_map, teacher_image_key=teacher_image_key)


def build_dataloader(runtime: XVLARuntime, config: VisualThoughtTrainConfig) -> DataLoader:
    return DataLoader(runtime.dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)


def build_train_val_dataloaders(runtime: XVLARuntime, config: VisualThoughtTrainConfig) -> tuple[DataLoader, DataLoader | None]:
    if not config.validation_enable or float(config.validation_split_ratio) <= 0.0: return build_dataloader(runtime, config), None
    dataset_len = len(runtime.dataset)
    val_len = min(max(int(round(dataset_len * float(config.validation_split_ratio))), 1), max(dataset_len - 1, 1))
    if dataset_len < 2 or val_len >= dataset_len: return build_dataloader(runtime, config), None
    generator = torch.Generator().manual_seed(int(config.validation_seed))
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    val_indices, train_indices = permutation[:val_len], permutation[val_len:]
    train_dataset, val_dataset = Subset(runtime.dataset, train_indices), Subset(runtime.dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=default_collate, drop_last=False)
    return train_loader, val_loader


def preprocess_batch(runtime: XVLARuntime, raw_batch: dict[str, Any]) -> dict[str, Any]:
    return runtime.preprocessor(raw_batch)


def get_teacher_images(raw_batch: dict[str, Any], teacher_image_key: str) -> torch.Tensor:
    images = raw_batch[teacher_image_key]
    return images[:, -1] if getattr(images, "ndim", 0) == 5 else images


def set_policy_trainability(policy, training_stage: TrainingStage) -> None:
    requires_grad = training_stage == "joint_multitask"
    for parameter in policy.parameters(): parameter.requires_grad = requires_grad
    if requires_grad: policy.train()
    else: policy.eval()


def build_teacher(config: VisualThoughtTrainConfig, runtime: XVLARuntime | None = None):
    if config.expert_type == "cedirnet":
        task_cfg = load_cedirnet_decoder_config(config.decoder_stack_config_path, config.decoder_task_config_path)
        dataset_length = len(runtime.dataset) if runtime is not None else None
        if dataset_length is None: raise ValueError("CeDiRNet cached teacher resolution requires an initialized runtime dataset.")
        cache = CeDiRNetTargetCache.resolve(dataset_repo_id=config.dataset_repo_id, dataset_revision=config.dataset_revision, dataset_root=_resolve_dataset_root(config), dataset_length=dataset_length, teacher_cfg=task_cfg.teacher, cache_root=config.teacher_target_cache_root)
        return task_cfg, cache
    task_cfg = load_dino_decoder_config(config.decoder_stack_config_path, config.decoder_task_config_path)
    return task_cfg, DinoV2Teacher(task_cfg.teacher)


def load_teacher_target(config: VisualThoughtTrainConfig, teacher_source, raw_batch: dict[str, Any], teacher_image_key: str) -> TeacherTarget:
    if config.expert_type == "cedirnet": return teacher_source.target_for_batch(raw_batch, device=_as_device(config))
    return teacher_source.predict(get_teacher_images(raw_batch, teacher_image_key).to(_as_device(config)))


def build_decoder(config: VisualThoughtTrainConfig, task_cfg, target: TeacherTarget, student_vlm_dim: int):
    if config.expert_type == "cedirnet": 
        return CeDirNetDistillationModel.from_config(student_vlm_dim=student_vlm_dim, cfg=task_cfg)
    if target.kind == "token_sequence": 
        return DinoTokenSequenceModel.from_config(student_vlm_dim=student_vlm_dim, target=target, cfg=task_cfg)
    return DinoFeatureAlignmentModel.from_config(student_vlm_dim=student_vlm_dim, target=target, cfg=task_cfg)


def load_decoder_init_if_present(decoder: torch.nn.Module, decoder_init_path: str | None) -> None:
    if not decoder_init_path: return
    root = Path(decoder_init_path)
    state = load_decoder_state(root / DECODER_STATE_FILENAME if root.is_dir() else root)
    decoder.load_state_dict(state, strict=True)


def build_optimizer(config: VisualThoughtTrainConfig, policy, decoder: torch.nn.Module) -> torch.optim.Optimizer:
    if config.training_stage == "distill_only": return torch.optim.AdamW(decoder.parameters(), lr=config.decoder_optimizer_lr, weight_decay=config.weight_decay)
    decoder_params = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    xvla_params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    return torch.optim.AdamW([{"params": decoder_params, "lr": config.decoder_optimizer_lr}, {"params": xvla_params, "lr": config.xvla_optimizer_lr}], weight_decay=config.weight_decay)


def build_xvla_inputs(policy, processed_batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = policy._build_model_inputs(processed_batch)
    enc = policy.model.forward_vlm(input_ids=inputs["input_ids"], pixel_values=inputs["image_input"], image_mask=inputs["image_mask"])
    return inputs, enc


def compute_xvla_action_loss_from_encoder(policy, processed_batch: dict[str, Any], inputs: dict[str, torch.Tensor], enc: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    targets = policy._prepare_action_targets(processed_batch)
    target_dtype = policy.model._get_target_dtype()
    t, action_noisy = policy.model._build_corrupted_action(action=targets, device=inputs["input_ids"].device, target_dtype=target_dtype)
    proprio_m, action_noisy_m = policy.model.action_space.preprocess(inputs["proprio"].to(dtype=target_dtype), action_noisy)
    transformer_parameters = policy.model.transformer.parameters() if hasattr(policy.model.transformer, "parameters") else iter(())
    transformer_parameter = next(transformer_parameters, None)
    transformer_dtype = transformer_parameter.dtype if transformer_parameter is not None else action_noisy_m.dtype
    action_noisy_m = action_noisy_m.to(dtype=transformer_dtype)
    proprio_m = proprio_m.to(dtype=transformer_dtype)
    t = t.to(dtype=transformer_dtype)
    enc = {key: value.to(dtype=transformer_dtype) if torch.is_tensor(value) and value.is_floating_point() else value for key, value in enc.items()}
    pred_action = policy.model.transformer(domain_id=inputs["domain_id"], action_with_noise=action_noisy_m, t=t, proprio=proprio_m, **enc)
    loss_dict = policy.model.action_space.compute_loss(pred_action, targets)
    action_loss = sum(loss_dict.values())
    stats = {key: float(value.detach().item()) for key, value in loss_dict.items()}
    stats["action_total"] = float(action_loss.detach().item())
    return action_loss, stats


def compute_expert_loss(config: VisualThoughtTrainConfig, decoder: torch.nn.Module, task_cfg, target: TeacherTarget, vlm_features: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
    # In joint_multitask the expert loss MUST flow gradient back into the VLM so the
    # auxiliary task actually shapes the backbone features. Only detach in distill_only,
    # where the policy is frozen anyway and the decoder is the sole thing being trained.
    if config.training_stage == "distill_only": vlm_features = vlm_features.detach()
    
    if config.expert_type == "cedirnet":
        prediction  = decoder(vlm_features, target_map=target.tensor)
        loss_target = TeacherTarget(name=target.name, tensor=target.tensor.to(dtype=prediction.dtype), kind=target.kind, loss_type=target.loss_type, weight=target.weight, aux=target.aux)
        loss        = compute_teacher_loss(prediction, loss_target).float()
        return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 0.0}
    
    if target.kind == "token_sequence":
        prediction  = decoder(vlm_features)
        loss        = compute_teacher_loss(prediction, target)
        return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 0.0}
    
    query_tokens = decoder.query_tokens(vlm_features)
    if int(step) <= int(config.align_feature_until_step):
        attended_stu, teacher_aligned_exp = decoder.query_align_features(query_tokens, target)
        loss = compute_feature_alignment_loss(attended_stu, teacher_aligned_exp, task_cfg.head.align_weight)
        return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 1.0}
    
    if target.tensor.ndim < 4: 
        raise ValueError("DINO reconstruction phase requires a spatial target.tensor. Keep align_feature_until_step >= steps when using alignment-only teacher targets.")
    
    _, prediction = decoder.query_reconstruct(query_tokens, target)
    recon = compute_teacher_loss(prediction, target)
    loss = float(task_cfg.head.recon_weight) * float(task_cfg.head.recon_scale) * recon
    return loss, {"expert_total": float(loss.detach().item()), "expert_stage": 2.0}


def prepare_models_and_target(config: VisualThoughtTrainConfig, runtime: XVLARuntime, task_cfg, teacher):
    loader, val_loader = build_train_val_dataloaders(runtime, config)
    first_raw_batch = next(iter(loader))
    processed = preprocess_batch(runtime, first_raw_batch)
    inputs, enc = build_xvla_inputs(runtime.policy, processed)
    target = load_teacher_target(config, teacher, first_raw_batch, runtime.teacher_image_key)
    decoder = build_decoder(config, task_cfg, target, student_vlm_dim=int(enc["vlm_features"].shape[-1])).to(_as_device(config))
    load_decoder_init_if_present(decoder, config.decoder_init_path)
    return loader, val_loader, decoder, target


def _checkpoint_metadata(config: VisualThoughtTrainConfig, step: int) -> dict[str, Any]:
    return {"name": config.name, "training_stage": config.training_stage, "expert_type": config.expert_type, "global_step": int(step), "xvla_init_path": config.xvla_init_path, "decoder_init_path": config.decoder_init_path}


def _push_checkpoint_to_hub(checkpoint_dir: Path, repo_id: str, commit_message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(checkpoint_dir), repo_id=repo_id, repo_type="model", commit_message=commit_message)


def _save_checkpoint_if_needed(config: VisualThoughtTrainConfig, runtime: XVLARuntime, decoder: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, final: bool = False) -> None:
    if not final and (config.save_every <= 0 or step % config.save_every != 0): return
    checkpoint_name = "checkpoint_final" if final else f"checkpoint_{step:07d}"
    checkpoint_dir = Path(config.output_dir) / checkpoint_name
    save_visual_thought_checkpoint(checkpoint_dir=checkpoint_dir, policy=runtime.policy, decoder=decoder, trainer_state={"step": int(step), "optimizer": optimizer.state_dict()}, metadata=_checkpoint_metadata(config, step), config_snapshot=config.to_json_dict(), preprocessor=runtime.preprocessor, postprocessor=runtime.postprocessor)
    should_push = bool(config.push_to_hub) and bool(config.push_repo_id) and (final or (int(config.push_every) > 0 and step % int(config.push_every) == 0))
    if should_push: _push_checkpoint_to_hub(checkpoint_dir, str(config.push_repo_id), f"Upload visual-thought checkpoint {checkpoint_name}")


def _maybe_init_wandb(config: VisualThoughtTrainConfig):
    if not config.wandb_enable: return None
    import wandb

    return wandb.init(project=config.wandb_project, name=config.wandb_run_name or config.name, config=config.to_json_dict(), dir=config.output_dir)


@torch.no_grad()
def run_validation(config: VisualThoughtTrainConfig, runtime: XVLARuntime, task_cfg, teacher, decoder: torch.nn.Module, val_loader: DataLoader) -> dict[str, float]:
    decoder_was_training = decoder.training
    policy_was_training = runtime.policy.training
    decoder.eval()
    runtime.policy.eval()
    total_loss, total_action, total_expert, batches = 0.0, 0.0, 0.0, 0
    for raw_batch in val_loader:
        if batches >= max(int(config.validation_max_batches), 1): break
        processed_batch = preprocess_batch(runtime, raw_batch)
        inputs, enc = build_xvla_inputs(runtime.policy, processed_batch)
        target = load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
        expert_loss, _ = compute_expert_loss(config, decoder, task_cfg, target, enc["vlm_features"], step=0)
        total = float(config.expert_loss_weight) * expert_loss
        action_value = 0.0
        if config.training_stage == "joint_multitask":
            action_loss, _ = compute_xvla_action_loss_from_encoder(runtime.policy, processed_batch, inputs, enc)
            action_value = float(action_loss.detach().item())
            total = float(config.action_loss_weight) * action_loss + float(config.expert_loss_weight) * expert_loss
        total_loss += float(total.detach().item())
        total_action += action_value
        total_expert += float(expert_loss.detach().item())
        batches += 1
    if decoder_was_training: decoder.train()
    if policy_was_training: runtime.policy.train()
    denom = max(batches, 1)
    return {"val_loss": total_loss / denom, "val_expert_total": total_expert / denom, "val_action_total": total_action / denom, "val_batches": float(batches)}


def train_visual_thought(config: VisualThoughtTrainConfig) -> None:
    _set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(config)

    runtime = build_xvla_runtime(config)

    set_policy_trainability(runtime.policy, config.training_stage)
    task_cfg, teacher = build_teacher(config, runtime)
    loader, val_loader, decoder, _ = prepare_models_and_target(config, runtime, task_cfg, teacher)
    if config.training_stage == "joint_multitask":
        for parameter in decoder.parameters(): parameter.requires_grad = True
        runtime.policy.train()
    else:
        decoder.train()
        
    optimizer = build_optimizer(config, runtime.policy, decoder)
    
    if config.dry_run: return

    step = 0
    accum_steps = max(int(config.gradient_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=config.steps, desc=config.name)
    while step < config.steps:
        for raw_batch in loader:
            if step >= config.steps: break
            step += 1

            processed_batch = preprocess_batch(runtime, raw_batch)
            inputs, enc     = build_xvla_inputs(runtime.policy, processed_batch)
            target          = load_teacher_target(config, teacher, raw_batch, runtime.teacher_image_key)
            expert_loss, expert_stats = compute_expert_loss(config, decoder, task_cfg, target, enc["vlm_features"], step)
            total_loss      = float(config.expert_loss_weight) * expert_loss
            
            action_stats = {}
            if config.training_stage == "joint_multitask":
                action_loss, action_stats = compute_xvla_action_loss_from_encoder(runtime.policy, processed_batch, inputs, enc)
                total_loss = float(config.action_loss_weight) * action_loss + float(config.expert_loss_weight) * expert_loss
            
            (total_loss / accum_steps).backward()
            if step % accum_steps == 0 or step == config.steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if step % max(int(config.log_every), 1) == 0:
                metrics = {"event": "train_step", "step": int(step), "loss": float(total_loss.detach().item()), **expert_stats, **action_stats}
                print(json.dumps(metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in metrics.items() if key != "event"}, step=int(step))
            if val_loader is not None and step % max(int(config.validation_freq), 1) == 0:
                val_metrics = {"event": "validation_step", "step": int(step), **run_validation(config, runtime, task_cfg, teacher, decoder, val_loader)}
                print(json.dumps(val_metrics))
                if wandb_run is not None: wandb_run.log({key: value for key, value in val_metrics.items() if key != "event"}, step=int(step))
            _save_checkpoint_if_needed(config, runtime, decoder, optimizer, step, final=False)
            progress.update(1)
    progress.close()
    if config.save_final_checkpoint: 
        _save_checkpoint_if_needed(config, runtime, decoder, optimizer, step, final=True)
    if wandb_run is not None: wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    train_visual_thought(VisualThoughtTrainConfig.from_json(args.config_path))


if __name__ == "__main__":
    main()
