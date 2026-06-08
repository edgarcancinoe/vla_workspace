"""Visual-thought building blocks extracted from XVLA-VisualThought."""

from thesis_vla.visual_thought.cedirnet_decoder import CeDirNetDistillationModel, DenseMapDecoderHead
from thesis_vla.visual_thought.config import DEFAULT_CEDIRNET_HEAD_CONFIG_PATH, DEFAULT_CEDIRNET_STACK_CONFIG_PATH, DEFAULT_DINO_CONFIG_PATH, DEFAULT_DINO_EXPERT_QUERY_CONFIG_PATH, DEFAULT_DINO_FEATURE_ALIGNMENT_CONFIG_PATH, DEFAULT_DINO_STACK_CONFIG_PATH, DEFAULT_DINO_TOKEN_SEQUENCE_CONFIG_PATH, CeDirNetDecoderConfig, CeDirNetTeacherConfig, DecoderStackConfig, DenseMapHeadConfig, DinoDecoderConfig, DinoFeatureAlignmentConfig, DinoTeacherConfig, ExpertQueryHeadConfig, load_cedirnet_decoder_config, load_dino_decoder_config, load_dino_feature_alignment_config
from thesis_vla.visual_thought.decoder_stack import StackedDecoderStrategy, StudentProjection
from thesis_vla.visual_thought.dino_features import DinoFeatureAlignmentModel, DinoTokenSequenceModel, ExpertFeatureQueryHead, compute_feature_alignment_loss, resolve_expert_query_metadata
from thesis_vla.visual_thought.targets import TeacherTarget, compute_teacher_loss

__all__ = [
    "CeDirNetDistillationModel",
    "CeDirNetDecoderConfig",
    "CeDirNetTeacherConfig",
    "DEFAULT_CEDIRNET_HEAD_CONFIG_PATH",
    "DEFAULT_CEDIRNET_STACK_CONFIG_PATH",
    "DEFAULT_DINO_CONFIG_PATH",
    "DEFAULT_DINO_EXPERT_QUERY_CONFIG_PATH",
    "DEFAULT_DINO_FEATURE_ALIGNMENT_CONFIG_PATH",
    "DEFAULT_DINO_STACK_CONFIG_PATH",
    "DEFAULT_DINO_TOKEN_SEQUENCE_CONFIG_PATH",
    "DecoderStackConfig",
    "DenseMapDecoderHead",
    "DenseMapHeadConfig",
    "DinoDecoderConfig",
    "DinoFeatureAlignmentConfig",
    "DinoFeatureAlignmentModel",
    "DinoTokenSequenceModel",
    "DinoTeacherConfig",
    "ExpertFeatureQueryHead",
    "ExpertQueryHeadConfig",
    "StackedDecoderStrategy",
    "StudentProjection",
    "TeacherTarget",
    "compute_feature_alignment_loss",
    "compute_teacher_loss",
    "load_cedirnet_decoder_config",
    "load_dino_decoder_config",
    "load_dino_feature_alignment_config",
    "resolve_expert_query_metadata",
]
