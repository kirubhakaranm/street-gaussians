"""Optimizer creation and learning rate scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from street_gaussians.config import TrainingConfig

if TYPE_CHECKING:
    from street_gaussians.models.scene import StreetGaussianScene


def _add_param_group(
    groups: list[dict],
    params: dict,
    key: str,
    lr: float,
    prefix: str,
) -> None:
    """Append a single param group to the list.

    Args:
        groups: Accumulator list of param group dicts.
        params: Gaussian parameter dict.
        key: Parameter key (e.g. "positions").
        lr: Learning rate for this group.
        prefix: Name prefix (e.g. "bg" or "obj3").
    """
    groups.append({
        "params": [params[key]], "lr": lr, "name": f"{prefix}_{key}",
    })


def create_scene_optimizer(
    scene: StreetGaussianScene, cfg: TrainingConfig
) -> torch.optim.Adam:
    """Create a single Adam optimizer with param groups for bg + all objects.

    Args:
        scene: The scene model.
        cfg: Training config with learning rates.

    Returns:
        Configured Adam optimizer.
    """
    groups: list[dict] = []
    lr_map = {
        "positions": cfg.lr_position_init,
        "quaternions": cfg.lr_quaternion,
        "log_scales": cfg.lr_scale,
        "logit_opacities": cfg.lr_opacity,
        "sh_coeffs": cfg.lr_sh,
    }

    for key, lr in lr_map.items():
        _add_param_group(groups, scene.bg_params, key, lr, "bg")

    for tid, obj in scene.objects.items():
        for key, lr in lr_map.items():
            _add_param_group(groups, obj.params, key, lr, f"obj{tid}")

    return torch.optim.Adam(groups, eps=1e-15)


def update_position_lr(
    optimizer: torch.optim.Adam, step: int, cfg: TrainingConfig
) -> float:
    """Exponentially decay the position learning rate.

    Args:
        optimizer: The scene optimizer.
        step: Current training step.
        cfg: Training config.

    Returns:
        The updated learning rate.
    """
    ratio = cfg.lr_position_final / cfg.lr_position_init
    lr = cfg.lr_position_init * ratio ** (step / cfg.iterations)
    for g in optimizer.param_groups:
        if g["name"].endswith("_positions"):
            g["lr"] = lr
    return lr
