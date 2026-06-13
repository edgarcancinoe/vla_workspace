from __future__ import annotations

import builtins
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pretrained import T
from lerobot.policies.xvla.modeling_xvla import XVLAModel, XVLAPolicy
from lerobot.policies.xvla.soft_transformer import DomainAwareLinear, TransformerBlock, basic_init, timestep_embedding

from thesis_vla.policies.xvla_guided.configuration_xvla_guided import XVLAGuidedConfig, normalize_guidance_fusion_mode
from thesis_vla.visual_thought import DinoTokenSequenceModel
from thesis_vla.visual_thought.cedirnet_decoder import CeDirNetDistillationModel
from thesis_vla.visual_thought.config import CeDirNetDecoderConfig, CeDirNetTeacherConfig, DecoderStackConfig, DenseMapHeadConfig, DinoDecoderConfig, DinoTeacherConfig, ExpertQueryHeadConfig


def _build_cedirnet_decoder_config(config: XVLAGuidedConfig) -> CeDirNetDecoderConfig:
    return CeDirNetDecoderConfig(
        stack=DecoderStackConfig.from_dict(config.guidance_decoder_stack),
        head=DenseMapHeadConfig.from_dict(config.guidance_decoder_head),
        teacher=CeDirNetTeacherConfig.from_dict(config.guidance_decoder_teacher),
    ).validate()


def _build_dino_decoder_config(config: XVLAGuidedConfig) -> DinoDecoderConfig:
    return DinoDecoderConfig(
        stack=DecoderStackConfig.from_dict(config.guidance_decoder_stack),
        head=ExpertQueryHeadConfig.from_dict(config.guidance_decoder_head),
        teacher=DinoTeacherConfig.from_dict(config.guidance_decoder_teacher),
    ).validate()


def _build_guidance_decoder(config: XVLAGuidedConfig, projection_dim: int) -> tuple[nn.Module, int]:
    if config.guidance_expert_type == "cedirnet":
        decoder_cfg = _build_cedirnet_decoder_config(config)
        return CeDirNetDistillationModel.from_config(student_vlm_dim=int(projection_dim), cfg=decoder_cfg), int(decoder_cfg.stack.decoder_dim)
    decoder_cfg = _build_dino_decoder_config(config)
    if decoder_cfg.teacher.target_kind != "token_sequence": raise ValueError(f"Guided DINO v1 supports token_sequence only, got {decoder_cfg.teacher.target_kind!r}.")
    return DinoTokenSequenceModel(student_vlm_dim=int(projection_dim), stack_cfg=decoder_cfg.stack), int(decoder_cfg.stack.decoder_dim)


class GuidedSoftPromptedTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        multi_modal_input_size: int,
        guidance_input_size: int,
        depth: int,
        num_heads: int,
        guidance_num_heads: int,
        mlp_ratio: float,
        num_domains: int,
        dim_action: int,
        dim_propio: int,
        dim_time: int,
        len_soft_prompts: int,
        max_len_seq: int,
        use_hetero_proj: bool,
        guidance_fusion_mode: str,
        guidance_gated: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dim_time = int(dim_time)
        self.len_soft_prompts = int(len_soft_prompts)
        self.use_hetero_proj = bool(use_hetero_proj)
        self.guidance_fusion_mode = normalize_guidance_fusion_mode(guidance_fusion_mode, guidance_gated)
        self.guidance_gated = self.guidance_fusion_mode.startswith("gated_")
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        if self.use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.guidance_proj = DomainAwareLinear(guidance_input_size, hidden_size, num_domains=num_domains)
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.guidance_proj = nn.Linear(guidance_input_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)
        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains)
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)
        if self.len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, self.len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
        if self.guidance_fusion_mode == "gated_concat": self.concat_gate = nn.Linear(hidden_size * 2, 1)
        if self.guidance_fusion_mode in {"cross_attention", "gated_cross_attention"}:
            self.cross_attn_query_norm = nn.LayerNorm(hidden_size)
            self.cross_attn_guidance_norm = nn.LayerNorm(hidden_size)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=guidance_num_heads, batch_first=True, dropout=0.1)
            if self.guidance_fusion_mode == "gated_cross_attention": self.cross_attn_gamma = nn.Parameter(torch.zeros(()))
        self.apply(basic_init)

    def _project_guidance(self, guidance_tokens: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        if self.use_hetero_proj: return self.guidance_proj(guidance_tokens, domain_id)
        return self.guidance_proj(guidance_tokens)

    def _project_native_tokens(self, domain_id: torch.LongTensor, vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor, action_with_noise: torch.Tensor, proprio: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_actions = action_with_noise.shape[:2]
        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(batch_size, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(batch_size, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        action_proj = self.action_encoder(action_tokens, domain_id)
        if self.use_hetero_proj:
            return action_proj, self.vlm_proj(vlm_features, domain_id), self.aux_visual_proj(aux_visual_inputs, domain_id)
        return action_proj, self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)

    def _apply_concat_fusion(self, native_context: torch.Tensor, guidance_tokens: torch.Tensor) -> torch.Tensor:
        if self.guidance_fusion_mode == "concat": return torch.cat([native_context, guidance_tokens], dim=1)
        pooled_native = native_context.mean(dim=1)
        pooled_guidance = guidance_tokens.mean(dim=1)
        gate = torch.sigmoid(self.concat_gate(torch.cat([pooled_native, pooled_guidance], dim=-1))).view(-1, 1, 1)
        return torch.cat([native_context, gate * guidance_tokens], dim=1)

    def _apply_cross_attention_fusion(self, z_proj: torch.Tensor, guidance_tokens: torch.Tensor, guidance_available: torch.Tensor | None = None) -> torch.Tensor:
        guidance_available = torch.ones((z_proj.shape[0], 1, 1), device=z_proj.device, dtype=z_proj.dtype) if guidance_available is None else guidance_available.to(device=z_proj.device, dtype=z_proj.dtype)
        attn, _ = self.cross_attn(self.cross_attn_query_norm(z_proj), self.cross_attn_guidance_norm(guidance_tokens), self.cross_attn_guidance_norm(guidance_tokens), need_weights=False)
        if self.guidance_fusion_mode == "gated_cross_attention": return z_proj + guidance_available * torch.tanh(self.cross_attn_gamma) * attn
        return z_proj + guidance_available * attn

    def _append_soft_prompts(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        if self.len_soft_prompts <= 0: return x
        soft_prompts = self.soft_prompt_hub(domain_id).view(x.shape[0], self.len_soft_prompts, self.hidden_size)
        return torch.cat([x, soft_prompts], dim=1)

    def forward(self, *, domain_id: torch.LongTensor, vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor, guidance_tokens: torch.Tensor, guidance_available: torch.Tensor | None = None, action_with_noise: torch.Tensor, proprio: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        action_proj, z_proj, aux_proj = self._project_native_tokens(domain_id, vlm_features, aux_visual_inputs, action_with_noise, proprio, t)
        guidance_proj = self._project_guidance(guidance_tokens, domain_id)
        guidance_available = torch.ones((guidance_proj.shape[0], 1, 1), device=guidance_proj.device, dtype=guidance_proj.dtype) if guidance_available is None else guidance_available.to(device=guidance_proj.device, dtype=guidance_proj.dtype)
        token_keep_mask = None
        if self.guidance_fusion_mode in {"concat", "gated_concat"}:
            native_context = torch.cat([action_proj, z_proj, aux_proj], dim=1)
            guidance_proj = guidance_proj * guidance_available
            x = self._apply_concat_fusion(native_context, guidance_proj)
            native_keep_mask = torch.ones((native_context.shape[0], native_context.shape[1]), device=guidance_proj.device, dtype=torch.bool)
            guidance_keep_mask = guidance_available.view(-1, 1).to(dtype=torch.bool).expand(-1, guidance_proj.shape[1])
            token_keep_mask = torch.cat([native_keep_mask, guidance_keep_mask], dim=1)
        else:
            z_guided = self._apply_cross_attention_fusion(z_proj, guidance_proj, guidance_available)
            x = torch.cat([action_proj, z_guided, aux_proj], dim=1)
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]: raise ValueError(f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}.")
        x = x + self.pos_emb[:, :seq_len, :]
        x = self._append_soft_prompts(x, domain_id)
        if token_keep_mask is not None and self.len_soft_prompts > 0:
            prompt_keep_mask = torch.ones((token_keep_mask.shape[0], self.len_soft_prompts), device=token_keep_mask.device, dtype=torch.bool)
            token_keep_mask = torch.cat([token_keep_mask, prompt_keep_mask], dim=1)
        for block in self.blocks: x = block(x, token_keep_mask=token_keep_mask)
        return self.action_decoder(self.norm(x[:, : action_with_noise.shape[1]]), domain_id)


class XVLAGuidedModel(XVLAModel):
    config: XVLAGuidedConfig

    def __init__(self, config: XVLAGuidedConfig, florence_config, proprio_dim: int) -> None:
        super().__init__(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None: raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")
        self.guidance_decoder, guidance_input_size = _build_guidance_decoder(config, int(projection_dim))
        self.transformer = GuidedSoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=int(projection_dim),
            guidance_input_size=int(guidance_input_size),
            depth=config.depth,
            num_heads=config.num_heads,
            guidance_num_heads=config.resolved_guidance_num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            dim_time=config.dim_time,
            len_soft_prompts=config.len_soft_prompts,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            guidance_fusion_mode=config.guidance_fusion_mode,
            guidance_gated=config.guidance_gated,
        )
        self._apply_freezing()
        self.set_guidance_trainability(step=0)
        self._apply_dtype()

    def set_guidance_trainability(self, step: int) -> None:
        mode = str(self.config.guidance_train_mode)
        trainable = mode == "train_from_start" or (mode == "warmup_freeze" and int(step) > int(self.config.guidance_unfreeze_step))
        if mode == "frozen": trainable = False
        for parameter in self.guidance_decoder.parameters(): parameter.requires_grad = trainable
        self.guidance_decoder.train(trainable and self.training)
        if not trainable: self.guidance_decoder.eval()

    def guidance_tokens(self, vlm_features: torch.Tensor) -> torch.Tensor:
        if self.config.guidance_expert_type == "cedirnet": return self.guidance_decoder.decoder_tokens(vlm_features)
        return self.guidance_decoder(vlm_features)

    def guidance_prediction_from_tokens(self, guidance_tokens: torch.Tensor, target_map: torch.Tensor | None = None, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        if self.config.guidance_expert_type == "cedirnet": return self.guidance_decoder.predict_from_tokens(guidance_tokens, target_map=target_map, output_size=output_size)
        return guidance_tokens

    def guidance_map(self, vlm_features: torch.Tensor, target_map: torch.Tensor | None = None, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        if self.config.guidance_expert_type != "cedirnet": raise ValueError(f"guidance_map is defined for CeDirNet guidance only, got {self.config.guidance_expert_type!r}.")
        tokens = self.guidance_tokens(vlm_features)
        return self.guidance_prediction_from_tokens(tokens, target_map=target_map, output_size=output_size)

    def forward(self, input_ids: torch.LongTensor, image_input: torch.FloatTensor, image_mask: torch.Tensor, domain_id: torch.LongTensor, proprio: torch.Tensor, action: torch.Tensor, t: torch.Tensor | None = None, action_noise: torch.Tensor | None = None, guidance_available: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        guidance_tokens = self.guidance_tokens(enc["vlm_features"])
        t, action_noisy = self._build_corrupted_action(action=action, device=input_ids.device, target_dtype=target_dtype, t=t, action_noise=action_noise)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)
        pred_action = self.transformer(domain_id=domain_id, action_with_noise=action_noisy_m, proprio=proprio_m, t=t, guidance_tokens=guidance_tokens, guidance_available=guidance_available, **enc)
        return self.action_space.compute_loss(pred_action, action), pred_action

    @torch.no_grad()
    def generate_actions(self, input_ids: torch.LongTensor, image_input: torch.FloatTensor, image_mask: torch.Tensor, domain_id: torch.LongTensor, proprio: torch.Tensor, steps: int) -> torch.Tensor:
        self.eval()
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        guidance_tokens = self.guidance_tokens(enc["vlm_features"])
        batch_size = input_ids.shape[0]
        action_dim = self.dim_action
        guidance_available = torch.ones((batch_size, 1, 1), device=proprio.device, dtype=target_dtype)
        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)
        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(domain_id=domain_id, action_with_noise=x_t_m, proprio=proprio_m, t=t, guidance_tokens=guidance_tokens, guidance_available=guidance_available, **enc)
        return self.action_space.postprocess(action)


class XVLAGuidedPolicy(XVLAPolicy):
    config_class = XVLAGuidedConfig
    name = "xvla_guided"

    def __init__(self, config: XVLAGuidedConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLAGuidedModel(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        import safetensors.torch

        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=pretrained_name_or_path, force_download=force_download, resume_download=resume_download, proxies=proxies, token=token, cache_dir=cache_dir, local_files_only=local_files_only, revision=revision, **kwargs)
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import HfHubHTTPError

            try:
                model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors", revision=revision, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, token=token, local_files_only=local_files_only)
            except HfHubHTTPError as error:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from error
        state_dict = safetensors.torch.load_file(model_file)
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict: state_dict[shared_key] = state_dict[encoder_key]
        cls._resize_positional_embedding_if_needed(state_dict, instance.model.transformer.pos_emb)
        incompat = instance.load_state_dict(state_dict, strict=bool(strict))
        if getattr(incompat, "missing_keys", None): logging.info("Guided XVLA missing keys while loading: %s", incompat.missing_keys)
        if getattr(incompat, "unexpected_keys", None): logging.info("Guided XVLA unexpected keys while loading: %s", incompat.unexpected_keys)
        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        return instance
