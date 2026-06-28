"""Image quality metrics — PSNR, SSIM, LPIPS."""

from __future__ import annotations

import math

import torch

from street_gaussians.training.loss import ssim_loss
from street_gaussians.utils.logger import get_logger

log = get_logger(__name__)


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR between two (H, W, 3) images.

    Args:
        pred: Predicted image.
        gt: Ground truth image.

    Returns:
        PSNR in dB.
    """
    mse = ((pred - gt) ** 2).mean()
    return 10.0 * math.log10(1.0 / (mse.item() + 1e-8))


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute SSIM between two (H, W, 3) images.

    Args:
        pred: Predicted image.
        gt: Ground truth image.

    Returns:
        SSIM value in [0, 1].
    """
    p = pred.permute(2, 0, 1).unsqueeze(0)
    g = gt.permute(2, 0, 1).unsqueeze(0)
    return 1.0 - ssim_loss(p, g).item()


@torch.no_grad()
def _get_lpips_fn(device: torch.device | str):
    """Lazy-load LPIPS model (VGG). Returns callable or None."""
    if not hasattr(_get_lpips_fn, "_cache"):
        _get_lpips_fn._cache = {}
    key = str(device)
    if key not in _get_lpips_fn._cache:
        try:
            import lpips
            _get_lpips_fn._cache[key] = lpips.LPIPS(net="vgg").to(device).eval()
        except ImportError:
            log.info("LPIPS not available — install with: pip install lpips")
            _get_lpips_fn._cache[key] = None
    return _get_lpips_fn._cache[key]


@torch.no_grad()
def evaluate(
    scene,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    gt_images: list[torch.Tensor],
    frame_indices: list[int],
    image_sizes: torch.Tensor,
    sh_degree: int = 0,
) -> dict[str, float]:
    """Evaluate a scene on a set of views.

    Args:
        scene: StreetGaussianScene instance.
        viewmats: (N, 4, 4) view matrices.
        Ks: (N, 3, 3) intrinsics.
        gt_images: List of (H, W, 3) ground truth tensors.
        frame_indices: Frame indices for each image.
        image_sizes: (N, 2) tensor of [W, H].
        sh_degree: SH degree for rendering.

    Returns:
        Dict with "psnr", "ssim", and optionally "lpips" keys.
    """
    from street_gaussians.rendering.renderer import render_frame

    device = scene.bg_params["positions"].device
    viewmats = viewmats.to(device)
    Ks = Ks.to(device)
    image_sizes = image_sizes.to(device)

    lpips_fn = _get_lpips_fn(device)
    psnr_total = ssim_total = lpips_total = 0.0
    N = len(gt_images)

    for i in range(N):
        gt = gt_images[i].to(device)
        W, H = image_sizes[i, 0].item(), image_sizes[i, 1].item()
        combined, _ = scene.compose_frame(frame_indices[i])
        rendered, _, _ = render_frame(combined, viewmats[i], Ks[i], W, H, sh_degree)

        psnr_total += compute_psnr(rendered, gt)
        ssim_total += compute_ssim(rendered, gt)
        if lpips_fn is not None:
            pred_t = rendered.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            gt_t = gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            lpips_total += lpips_fn(pred_t, gt_t).item()

    result = {"psnr": psnr_total / N, "ssim": ssim_total / N}
    if lpips_fn is not None:
        result["lpips"] = lpips_total / N
    return result
