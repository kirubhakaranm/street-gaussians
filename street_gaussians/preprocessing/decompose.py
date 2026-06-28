"""Scene decomposition — separate LiDAR into background and per-object point clouds."""

from __future__ import annotations

import glob as glob_module
import os
from dataclasses import dataclass, field

import numpy as np

from street_gaussians.data.base import SceneData, Track
from street_gaussians.data.kitti import load_velodyne
from street_gaussians.models.math_utils import yaw_to_rotmat_np
from street_gaussians.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PointCloud:
    """Simple point cloud container."""
    positions: np.ndarray
    colors: np.ndarray


def point_in_box(
    points: np.ndarray,
    center: np.ndarray,
    dims: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Test which points fall inside a 3D bounding box.

    Args:
        points: (N, 3) in camera frame.
        center: (3,) box center.
        dims: (3,) [h, w, l].
        yaw: Rotation around Y-axis.

    Returns:
        (N,) boolean mask.
    """
    h, w, l = dims
    R_inv = yaw_to_rotmat_np(-yaw)
    local = (points - center) @ R_inv.T
    return (
        (np.abs(local[:, 0]) <= l / 2)
        & (np.abs(local[:, 1]) <= h / 2)
        & (np.abs(local[:, 2]) <= w / 2)
    )


def decompose_scene(
    scene_data: SceneData,
    num_frames: int,
) -> tuple[PointCloud, dict[int, PointCloud]]:
    """Separate LiDAR points into background and per-object clouds.

    Accumulates across all frames in a common world frame using ego poses.

    Args:
        scene_data: Loaded scene data containing tracks, calibration, etc.
        num_frames: Number of frames to process.

    Returns:
        Tuple of (background_cloud, object_clouds) where object_clouds
        is keyed by track_id.
    """
    tracks = scene_data.tracks
    Tr_velo_to_cam = scene_data.Tr_velo_to_cam
    R_rect = scene_data.R_rect
    poses_c2w = scene_data.poses_cam_to_world
    velodyne_dir = scene_data.velodyne_dir

    bg_all: list[np.ndarray] = []
    obj_all: dict[int, list[np.ndarray]] = {tid: [] for tid in tracks}

    bin_files = sorted(glob_module.glob(os.path.join(velodyne_dir, "*.bin")))
    n_files = min(len(bin_files), num_frames)

    for frame_idx in range(n_files):
        pts_cam = load_velodyne(bin_files[frame_idx], Tr_velo_to_cam, R_rect)

        valid = (pts_cam[:, 2] > 0) & (pts_cam[:, 2] < 80)
        pts_cam = pts_cam[valid]

        T_c2w = poses_c2w[frame_idx]
        R_w = T_c2w[:3, :3].astype(np.float32)
        t_w = T_c2w[:3, 3].astype(np.float32)
        pts_world = (R_w @ pts_cam.T).T + t_w

        in_any_box = np.zeros(pts_cam.shape[0], dtype=bool)

        for tid, track in tracks.items():
            for ann in track.annotations:
                if ann.frame != frame_idx:
                    continue
                center = np.array([ann.x, ann.y, ann.z], dtype=np.float32)
                dims = np.array([ann.h, ann.w, ann.l], dtype=np.float32)
                mask = point_in_box(pts_cam, center, dims, ann.yaw)
                if mask.sum() > 0:
                    obj_all[tid].append(pts_world[mask])
                    in_any_box |= mask

        bg_all.append(pts_world[~in_any_box])

    bg_pts = np.concatenate(bg_all, axis=0) if bg_all else np.zeros((0, 3), dtype=np.float32)

    obj_clouds = {}
    for tid in tracks:
        if obj_all[tid]:
            pts = np.concatenate(obj_all[tid], axis=0)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
        obj_clouds[tid] = PointCloud(positions=pts, colors=np.full_like(pts, 0.5))

    total_obj = sum(c.positions.shape[0] for c in obj_clouds.values())
    log.info(
        "Decomposition: %s bg points, %s object points across %d tracks",
        f"{bg_pts.shape[0]:,}", f"{total_obj:,}", len(obj_clouds),
    )

    return PointCloud(positions=bg_pts, colors=np.full_like(bg_pts, 0.5)), obj_clouds
