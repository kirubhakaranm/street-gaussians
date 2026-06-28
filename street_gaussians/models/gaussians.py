"""Gaussian initialization, scale computation, and PLY export."""

import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from street_gaussians.utils.logger import get_logger

log = get_logger(__name__)


def compute_initial_scale(positions: torch.Tensor) -> torch.Tensor:
    """Estimate per-Gaussian log-scale from KNN distances.

    Args:
        positions: (N, 3) Gaussian centers.

    Returns:
        (N, 3) log-scale tensor.
    """
    N = positions.shape[0]
    if N < 4:
        return torch.full((N, 3), -5.0, device=positions.device)

    batch_size = 10000
    k = min(4, N)
    all_mean_dist = []
    for i in range(0, N, batch_size):
        batch = positions[i : i + batch_size]
        dists = torch.cdist(batch, positions)
        knn = dists.topk(k, largest=False).values[:, 1:]
        all_mean_dist.append(knn.mean(dim=-1, keepdim=True) / 2.0)
    mean_dist = torch.cat(all_mean_dist, dim=0)
    return torch.log(mean_dist).clamp(min=-10).repeat(1, 3)


def init_gaussians(
    positions: np.ndarray,
    colors: np.ndarray,
    sh_degree: int = 0,
    device: str = "cuda",
) -> dict[str, nn.Parameter]:
    """Initialize Gaussian splat parameters from a point cloud.

    Args:
        positions: (N, 3) point positions.
        colors: (N, 3) RGB colors in [0, 1].
        sh_degree: Spherical harmonics degree.
        device: Target device.

    Returns:
        Dict of named nn.Parameters.
    """
    pos = torch.tensor(positions, dtype=torch.float32)
    color = torch.tensor(colors, dtype=torch.float32)
    N = pos.shape[0]
    K = (sh_degree + 1) ** 2

    sh = torch.zeros(N, K, 3)
    sh[:, 0, :] = (color - 0.5) / 0.28209479177387814

    return {
        "positions": nn.Parameter(pos.clone().to(device)),
        "quaternions": nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).repeat(N, 1).to(device)
        ),
        "log_scales": nn.Parameter(compute_initial_scale(pos).to(device)),
        "logit_opacities": nn.Parameter(
            torch.full((N, 1), math.log(0.1 / 0.9)).to(device)
        ),
        "sh_coeffs": nn.Parameter(sh.to(device)),
    }


def fill_box_with_random_points(
    center: np.ndarray,
    dims: tuple[float, float, float],
    yaw: float,
    n_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate random points filling a 3D bounding box volume.

    Args:
        center: (3,) box center in world frame.
        dims: (h, w, l) box dimensions.
        yaw: Rotation angle around Y-axis.
        n_points: Number of points to generate.
        rng: Optional random generator.

    Returns:
        (n_points, 3) array in world frame.
    """
    from street_gaussians.models.math_utils import yaw_to_rotmat_np

    if rng is None:
        rng = np.random.default_rng()
    h, w, l = dims
    local = rng.uniform(
        [-l / 2, -h / 2, -w / 2], [l / 2, h / 2, w / 2], size=(n_points, 3)
    ).astype(np.float32)
    R = yaw_to_rotmat_np(yaw)
    return (R @ local.T).T + center


def save_ply(params: dict[str, nn.Parameter], path: str | Path) -> None:
    """Export Gaussian parameters to PLY format.

    Args:
        params: Dict of Gaussian parameter tensors.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    N = params["positions"].shape[0]
    positions = params["positions"].detach().cpu().numpy()
    normals = np.zeros_like(positions)
    quats = params["quaternions"].detach().cpu().numpy()
    log_scales = params["log_scales"].detach().cpu().numpy()
    logit_ops = params["logit_opacities"].detach().cpu().numpy()
    sh = params["sh_coeffs"].detach().cpu().numpy()
    dc = sh[:, 0, :]
    rest = sh[:, 1:, :].reshape(N, -1)

    data = np.hstack([positions, normals, dc, rest, logit_ops, log_scales, quats]).astype(
        np.float32
    )
    props = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + "\n".join(f"property float {p}" for p in props)
        + "\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    log.info("Saved PLY: %s (%.1f MB)", path, path.stat().st_size / 1e6)
