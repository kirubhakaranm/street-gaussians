"""gsplat rasterization wrapper."""

import gsplat
import torch


def render_frame(
    params: dict[str, torch.Tensor],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Render a single frame using gsplat rasterization.

    Args:
        params: Combined Gaussian parameters dict.
        viewmat: (4, 4) world-to-camera view matrix.
        K: (3, 3) camera intrinsics.
        width: Image width in pixels.
        height: Image height in pixels.
        sh_degree: Active SH degree.

    Returns:
        Tuple of (rendered_image, alpha, metadata).
        rendered_image is (H, W, 3), alpha is (H, W, 1).
    """
    rendered, alpha, meta = gsplat.rasterization(
        means=params["positions"],
        quats=params["quaternions"],
        scales=torch.exp(params["log_scales"]),
        opacities=torch.sigmoid(params["logit_opacities"]).squeeze(-1),
        colors=params["sh_coeffs"],
        sh_degree=sh_degree,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        packed=False,
        absgrad=True,
    )
    return rendered[0], alpha[0], meta
