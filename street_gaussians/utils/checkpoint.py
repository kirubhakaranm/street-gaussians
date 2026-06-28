"""Checkpoint save/load for StreetGaussianScene."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from street_gaussians.utils.logger import get_logger

if TYPE_CHECKING:
    from street_gaussians.config import Config
    from street_gaussians.models.scene import StreetGaussianScene

log = get_logger(__name__)


def save_checkpoint(
    scene: StreetGaussianScene, step: int, cfg: Config, out_dir: str | Path
) -> Path:
    """Serialize full scene state to disk.

    Args:
        scene: The scene model containing background and object params.
        step: Current training iteration.
        cfg: Pipeline config (saved for reproducibility).
        out_dir: Output directory.

    Returns:
        Path to the saved checkpoint file.
    """
    from dataclasses import asdict

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "config": asdict(cfg),
        "bg_params": {k: v.data.cpu() for k, v in scene.bg_params.items()},
        "objects": {},
    }
    for tid, obj in scene.objects.items():
        ckpt["objects"][tid] = {
            "type": obj.obj_type,
            "params": {k: v.data.cpu() for k, v in obj.params.items()},
            "frame_poses": {
                fi: (R.cpu(), t.cpu()) for fi, (R, t) in obj.frame_poses.items()
            },
            "frame_range": obj.frame_range,
        }

    path = out_dir / f"checkpoint_{step:06d}.pt"
    torch.save(ckpt, path)
    log.info("Checkpoint saved: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_checkpoint(
    ckpt_path: str | Path, device: str = "cuda"
) -> tuple[StreetGaussianScene, int]:
    """Restore a scene from a checkpoint file.

    Args:
        ckpt_path: Path to checkpoint .pt file.
        device: Target device for tensors.

    Returns:
        Tuple of (scene, step).
    """
    from street_gaussians.models.scene import ObjectModel, StreetGaussianScene

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    bg_params = {k: nn.Parameter(v.to(device)) for k, v in ckpt["bg_params"].items()}
    objects = {}
    for tid, odata in ckpt["objects"].items():
        tid = int(tid) if isinstance(tid, str) else tid
        params = {k: nn.Parameter(v.to(device)) for k, v in odata["params"].items()}
        frame_poses = {
            fi: (R.to(device), t.to(device))
            for fi, (R, t) in odata["frame_poses"].items()
        }
        objects[tid] = ObjectModel(
            tid, odata["type"], params, frame_poses, odata["frame_range"]
        )

    scene = StreetGaussianScene(bg_params, objects)
    log.info("Loaded checkpoint from step %d", ckpt["step"])
    return scene, ckpt["step"]
