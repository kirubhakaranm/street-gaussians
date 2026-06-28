"""Adaptive Density Control — gradient-based densification and pruning."""

import math

import torch
import torch.nn as nn

from street_gaussians.models.math_utils import quat_to_rotmat


class ADCHelper:
    """Tracks gradient statistics and performs densify/prune/reset operations."""

    def __init__(self) -> None:
        """Initialize with empty gradient accumulators."""
        self.grad_accum: torch.Tensor | None = None
        self.count: torch.Tensor | None = None

    def update_stats(self, grads: torch.Tensor) -> None:
        """Accumulate gradient norms.

        Args:
            grads: (N,) tensor of gradient norms for this component.
        """
        if self.grad_accum is None:
            self.grad_accum = torch.zeros(grads.shape[0], device=grads.device)
            self.count = torch.zeros(grads.shape[0], device=grads.device)
        n = min(grads.shape[0], self.grad_accum.shape[0])
        self.grad_accum[:n] += grads[:n]
        self.count[:n] += 1.0

    def densify_and_prune(
        self,
        params: dict[str, nn.Parameter],
        scene_extent: float,
        grad_threshold: float,
        opacity_prune_threshold: float,
        split_thr_factor: float = 0.01,
    ) -> dict[str, nn.Parameter]:
        """Clone small high-gradient splats, split large ones, prune transparent/huge ones.

        Args:
            params: Gaussian parameters dict.
            scene_extent: Spatial extent of the scene.
            grad_threshold: Gradient norm threshold for densification.
            opacity_prune_threshold: Opacity below which Gaussians are pruned.
            split_thr_factor: Scale threshold factor for clone vs split.

        Returns:
            Updated params dict with new nn.Parameters.
        """
        avg_grad = self.grad_accum / self.count.clamp(min=1)
        scales = torch.exp(params["log_scales"])
        max_scale = scales.max(dim=-1).values
        thr = scene_extent * split_thr_factor

        high_grad = avg_grad > grad_threshold
        clone_mask = high_grad & (max_scale <= thr)
        split_mask = high_grad & (max_scale > thr)

        N_split = int(split_mask.sum().item())
        offsets = None
        if N_split > 0:
            split_scales = scales[split_mask]
            split_R = quat_to_rotmat(params["quaternions"][split_mask])
            noise = torch.randn(2 * N_split, 3, device=split_scales.device)
            scales_2 = torch.cat([split_scales, split_scales], dim=0)
            R_2 = torch.cat([split_R, split_R], dim=0)
            offsets = torch.bmm(R_2, (scales_2 * noise).unsqueeze(-1)).squeeze(-1)

        keep_mask = ~split_mask
        new_params = {}
        for k, v in params.items():
            parts = [v.data[keep_mask], v.data[clone_mask]]
            split_vals = torch.cat([v.data[split_mask], v.data[split_mask]], dim=0)
            if k == "log_scales":
                split_vals = split_vals - math.log(1.6)
            elif k == "positions" and N_split > 0 and offsets is not None:
                split_vals = split_vals + offsets
            parts.append(split_vals)
            new_params[k] = nn.Parameter(torch.cat(parts, dim=0))

        scales_n = torch.exp(new_params["log_scales"])
        opacities = torch.sigmoid(new_params["logit_opacities"]).squeeze(-1)
        keep = ~(
            (opacities < opacity_prune_threshold)
            | (scales_n.max(dim=-1).values > scene_extent * 0.5)
        )
        for k, v in new_params.items():
            new_params[k] = nn.Parameter(v.data[keep])

        return new_params

    def reset_opacity(self, params: dict[str, nn.Parameter]) -> None:
        """Reset all opacities to near-transparent.

        Args:
            params: Gaussian parameters dict to modify in-place.
        """
        with torch.no_grad():
            params["logit_opacities"].fill_(math.log(0.01 / 0.99))

    def reset_stats(self, n: int, device: str | torch.device) -> None:
        """Zero out accumulated gradient statistics.

        Args:
            n: Number of Gaussians in the component.
            device: Target device for the new tensors.
        """
        self.grad_accum = torch.zeros(n, device=device)
        self.count = torch.zeros(n, device=device)
