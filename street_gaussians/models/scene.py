"""Street Gaussian scene model — background + per-object Gaussian models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from street_gaussians.data.base import SceneData
from street_gaussians.models.gaussians import fill_box_with_random_points, init_gaussians
from street_gaussians.models.math_utils import (
    interpolate_pose,
    quat_multiply,
    quat_to_rotmat,
    rotmat_to_quat,
    yaw_to_rotmat_np,
)
from street_gaussians.preprocessing.decompose import PointCloud
from street_gaussians.training.adc import ADCHelper
from street_gaussians.utils.logger import get_logger

log = get_logger(__name__)


class ObjectModel:
    """A tracked object with local Gaussian params and per-frame rigid poses."""

    def __init__(
        self,
        track_id: int,
        obj_type: str,
        params: dict[str, nn.Parameter],
        frame_poses: dict[int, tuple[torch.Tensor, torch.Tensor]],
        frame_range: tuple[int, int],
    ):
        """Initialize an object model with local Gaussians and per-frame poses."""
        self.track_id = track_id
        self.obj_type = obj_type
        self.params = params
        self.frame_poses = frame_poses
        self.frame_range = frame_range

    def is_visible(self, frame_idx: int) -> bool:
        """Check whether this object has a pose at the given frame.

        Args:
            frame_idx: Frame index to query.

        Returns:
            True if the object is visible at this frame.
        """
        return frame_idx in self.frame_poses


class StreetGaussianScene:
    """Holds background + per-object Gaussian models with composition logic."""

    def __init__(
        self,
        bg_params: dict[str, nn.Parameter],
        objects: dict[int, ObjectModel],
    ):
        """Initialize the scene with background Gaussians and object models."""
        self.bg_params = bg_params
        self.objects = objects
        self.bg_adc = ADCHelper()
        self.obj_adcs = {tid: ADCHelper() for tid in objects}

    def get_visible_objects(self, frame_idx: int) -> dict[int, ObjectModel]:
        """Return objects that have a pose at the given frame.

        Args:
            frame_idx: Frame index to query.

        Returns:
            Dict of track_id to ObjectModel for visible objects.
        """
        return {
            tid: obj for tid, obj in self.objects.items() if obj.is_visible(frame_idx)
        }

    def compose_frame(
        self, frame_idx: int
    ) -> tuple[dict[str, torch.Tensor], list[tuple[str | int, int, int]]]:
        """Combine background + visible objects into a single params dict.

        Args:
            frame_idx: Frame index to compose.

        Returns:
            Tuple of (combined_params, segments) where segments track
            (component_id, start_idx, end_idx) for gradient routing.
        """
        all_parts: dict[str, list[torch.Tensor]] = {k: [self.bg_params[k]] for k in self.bg_params}
        n_bg = self.bg_params["positions"].shape[0]
        segments: list[tuple[str | int, int, int]] = [("bg", 0, n_bg)]
        offset = n_bg

        for tid, obj in self.get_visible_objects(frame_idx).items():
            R_obj, t_obj = obj.frame_poses[frame_idx]
            n_obj = obj.params["positions"].shape[0]

            pos_world = (obj.params["positions"] @ R_obj.T) + t_obj
            quat_obj = rotmat_to_quat(R_obj).unsqueeze(0).expand(n_obj, -1)
            quats_world = quat_multiply(quat_obj, obj.params["quaternions"])

            all_parts["positions"].append(pos_world)
            all_parts["quaternions"].append(quats_world)
            all_parts["log_scales"].append(obj.params["log_scales"])
            all_parts["logit_opacities"].append(obj.params["logit_opacities"])
            all_parts["sh_coeffs"].append(obj.params["sh_coeffs"])

            segments.append((tid, offset, offset + n_obj))
            offset += n_obj

        combined = {k: torch.cat(v, dim=0) for k, v in all_parts.items()}
        return combined, segments

    def compose_frame_interpolated(
        self, frame_a: int, frame_b: int, alpha: float
    ) -> dict[str, torch.Tensor]:
        """Compose scene at fractional time between two frames.

        Objects visible in both frames get interpolated poses. Objects visible
        in only one frame use that frame's pose.

        Args:
            frame_a: Start frame.
            frame_b: End frame.
            alpha: Interpolation factor in [0, 1].

        Returns:
            Combined params dict.
        """
        all_parts: dict[str, list[torch.Tensor]] = {k: [self.bg_params[k]] for k in self.bg_params}

        visible_a = self.get_visible_objects(frame_a)
        visible_b = self.get_visible_objects(frame_b)

        for tid in set(visible_a.keys()) | set(visible_b.keys()):
            obj = self.objects[tid]
            n_obj = obj.params["positions"].shape[0]

            if tid in visible_a and tid in visible_b:
                R0, t0 = obj.frame_poses[frame_a]
                R1, t1 = obj.frame_poses[frame_b]
                R_interp, t_interp = interpolate_pose(R0, t0, R1, t1, alpha)
            elif tid in visible_a:
                R_interp, t_interp = obj.frame_poses[frame_a]
            else:
                R_interp, t_interp = obj.frame_poses[frame_b]

            pos_world = (obj.params["positions"] @ R_interp.T) + t_interp
            quat_obj = rotmat_to_quat(R_interp).unsqueeze(0).expand(n_obj, -1)
            quats_world = quat_multiply(quat_obj, obj.params["quaternions"])

            all_parts["positions"].append(pos_world)
            all_parts["quaternions"].append(quats_world)
            all_parts["log_scales"].append(obj.params["log_scales"])
            all_parts["logit_opacities"].append(obj.params["logit_opacities"])
            all_parts["sh_coeffs"].append(obj.params["sh_coeffs"])

        return {k: torch.cat(v, dim=0) for k, v in all_parts.items()}

    def route_gradients(
        self,
        meta: dict,
        segments: list[tuple[str | int, int, int]],
    ) -> None:
        """Extract per-component gradient norms from the render metadata."""
        m2d = meta.get("means2d")
        if m2d is None:
            return
        g = getattr(m2d, "absgrad", None)
        if g is None:
            g = m2d.grad
        if g is None:
            return
        if g.dim() == 3:
            g = g[0]
        grads = g.norm(dim=-1).detach()

        for comp_id, start, end in segments:
            comp_grads = grads[start:end]
            if comp_id == "bg":
                self.bg_adc.update_stats(comp_grads)
            elif comp_id in self.obj_adcs:
                self.obj_adcs[comp_id].update_stats(comp_grads)


def build_object_models(
    scene_data: SceneData,
    obj_clouds: dict[int, PointCloud],
    sh_degree: int,
    min_points: int,
    device: str,
) -> dict[int, ObjectModel]:
    """Build ObjectModel instances from tracks and decomposed point clouds.

    Args:
        scene_data: Loaded scene data.
        obj_clouds: Per-object point clouds in world frame.
        sh_degree: SH degree for Gaussian initialization.
        min_points: Minimum points per object (random-fill if fewer).
        device: Target device.

    Returns:
        Dict of track_id -> ObjectModel.
    """
    tracks = scene_data.tracks
    poses_c2w = scene_data.poses_cam_to_world
    models: dict[int, ObjectModel] = {}
    rng = np.random.default_rng(42)

    for tid, track in tracks.items():
        if not track.annotations:
            continue

        frame_poses: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for ann in track.annotations:
            fi = ann.frame
            if fi >= len(poses_c2w):
                continue

            center_cam = np.array([ann.x, ann.y, ann.z], dtype=np.float32)
            T_c2w = np.array(poses_c2w[fi], dtype=np.float32)
            R_c2w, t_c2w = T_c2w[:3, :3], T_c2w[:3, 3]

            center_world = R_c2w @ center_cam + t_c2w
            R_obj_cam = yaw_to_rotmat_np(ann.yaw)
            R_obj_world = R_c2w @ R_obj_cam

            frame_poses[fi] = (
                torch.tensor(R_obj_world, dtype=torch.float32, device=device),
                torch.tensor(center_world, dtype=torch.float32, device=device),
            )

        if not frame_poses:
            continue

        first_frame = track.annotations[0].frame
        if first_frame not in frame_poses:
            first_frame = min(frame_poses.keys())
        R_ref, t_ref = frame_poses[first_frame]

        pts_world = obj_clouds[tid].positions
        if pts_world.shape[0] < min_points:
            ann0 = track.annotations[0]
            center_cam = np.array([ann0.x, ann0.y, ann0.z], dtype=np.float32)
            T_c2w = np.array(poses_c2w[ann0.frame], dtype=np.float32)
            center_world = T_c2w[:3, :3] @ center_cam + T_c2w[:3, 3]
            pts_world = fill_box_with_random_points(
                center_world, (ann0.h, ann0.w, ann0.l), ann0.yaw, min_points, rng
            )
            log.info("  Object %d (%s): %d random-fill points", tid, track.obj_type, min_points)

        pts_t = torch.tensor(pts_world, dtype=torch.float32, device=device)
        pts_local = (pts_t - t_ref) @ R_ref

        local_positions = pts_local.detach().cpu().numpy()
        local_colors = np.full((local_positions.shape[0], 3), 0.5, dtype=np.float32)
        params = init_gaussians(local_positions, local_colors, sh_degree=sh_degree, device=device)
        frame_range = (min(frame_poses.keys()), max(frame_poses.keys()))

        models[tid] = ObjectModel(tid, track.obj_type, params, frame_poses, frame_range)
        log.info(
            "  Object %d (%s): %d Gaussians, frames %d-%d",
            tid, track.obj_type, params["positions"].shape[0],
            frame_range[0], frame_range[1],
        )

    return models
