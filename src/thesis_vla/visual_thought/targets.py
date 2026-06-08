from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class TeacherTarget:
    name: str
    tensor: torch.Tensor
    kind: str
    loss_type: str
    weight: float
    aux: dict[str, torch.Tensor | int | float | str | tuple[int, ...]] = field(default_factory=dict)


def compute_teacher_loss(prediction: torch.Tensor, target: TeacherTarget) -> torch.Tensor:
    if target.loss_type == "mse": return F.mse_loss(prediction, target.tensor)
    if target.loss_type == "l1": return F.l1_loss(prediction, target.tensor)
    if target.loss_type == "smooth_l1": return F.smooth_l1_loss(prediction, target.tensor)
    return F.binary_cross_entropy_with_logits(prediction, target.tensor)
