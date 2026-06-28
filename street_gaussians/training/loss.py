"""Loss functions — L1, SSIM, and composite training loss."""

import torch
import torch.nn.functional as F

_WIN_CACHE: dict[str, torch.Tensor] = {}


def _create_gaussian_window(
    size: int = 11, sigma: float = 1.5, channels: int = 3
) -> torch.Tensor:
    """Create a Gaussian convolution window for SSIM computation.

    Args:
        size: Window size in pixels.
        sigma: Gaussian standard deviation.
        channels: Number of image channels.

    Returns:
        (channels, 1, size, size) convolution kernel.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1d /= g1d.sum()
    g2d = (g1d.unsqueeze(1) @ g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return g2d.expand(channels, 1, size, size).contiguous()


def _get_window(device: torch.device | str) -> torch.Tensor:
    """Get or create a cached Gaussian window for the given device.

    Args:
        device: Target device.

    Returns:
        Cached (channels, 1, size, size) Gaussian window tensor.
    """
    key = str(device)
    if key not in _WIN_CACHE:
        _WIN_CACHE[key] = _create_gaussian_window().to(device)
    return _WIN_CACHE[key]


def ssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute 1 - SSIM between two (1, C, H, W) images.

    Args:
        pred: Predicted image tensor.
        gt: Ground truth image tensor.

    Returns:
        Scalar loss (1 - mean SSIM).
    """
    window = _get_window(pred.device)
    pad = window.shape[-1] // 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=3)
    mu2 = F.conv2d(gt, window, padding=pad, groups=3)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sg1 = F.conv2d(pred * pred, window, padding=pad, groups=3) - mu1s
    sg2 = F.conv2d(gt * gt, window, padding=pad, groups=3) - mu2s
    sg12 = F.conv2d(pred * gt, window, padding=pad, groups=3) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sg12 + C2)) / ((mu1s + mu2s + C1) * (sg1 + sg2 + C2))
    return 1.0 - ssim_map.mean()


def compute_loss(
    pred: torch.Tensor, gt: torch.Tensor, lambda_ssim: float = 0.2
) -> torch.Tensor:
    """Combined L1 + SSIM loss.

    Args:
        pred: (H, W, 3) predicted image.
        gt: (H, W, 3) ground truth image.
        lambda_ssim: Weight for SSIM component.

    Returns:
        Scalar loss value.
    """
    l1 = (pred - gt).abs().mean()
    p = pred.permute(2, 0, 1).unsqueeze(0)
    g = gt.permute(2, 0, 1).unsqueeze(0)
    sl = ssim_loss(p, g)
    return (1 - lambda_ssim) * l1 + lambda_ssim * sl
