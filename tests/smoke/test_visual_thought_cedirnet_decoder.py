from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from thesis_vla.common.paths import PROJECT_ROOT
from thesis_vla.visual_thought import CeDirNetDistillationModel, load_cedirnet_decoder_config


def test_cedirnet_decoder_forward_smoke():
    cfg = load_cedirnet_decoder_config(PROJECT_ROOT / "config" / "visual_thought" / "cedirnet_stack.yaml", PROJECT_ROOT / "config" / "visual_thought" / "cedirnet_head.yaml")
    model = CeDirNetDistillationModel.from_config(student_vlm_dim=64, cfg=cfg)
    vlm_features = torch.randn(2, 256, 64)
    target_map = torch.randn(2, cfg.teacher.out_channels, 96, 96)
    prediction = model(vlm_features, target_map=target_map)
    assert prediction.shape == target_map.shape
