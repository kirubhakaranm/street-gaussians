"""Nine fault-tolerant demo renderers for trained Street Gaussian scenes."""

from __future__ import annotations

import math
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import gsplat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from street_gaussians.models.math_utils import interpolate_pose
from street_gaussians.models.scene import StreetGaussianScene
from street_gaussians.rendering.renderer import render_frame
from street_gaussians.utils.logger import get_logger
from street_gaussians.utils.video import frames_to_video

log = get_logger(__name__)

DEMO_REGISTRY: dict[int, str] = {
    1: "replay",
    2: "novel_viewpoint",
    3: "object_removal",
    4: "background_only",
    5: "freeze_frame",
    6: "scene_recomposition",
    7: "depth_map",
    8: "bev_occupancy",
    9: "temporal_interpolation",
}


def parse_demo_selection(demos_str: str) -> list[int]:
    """Parse a comma-separated demo selection string.

    Args:
        demos_str: "all", or comma-separated numbers/names.

    Returns:
        Sorted list of demo IDs.
    """
    if demos_str.strip().lower() == "all":
        return list(DEMO_REGISTRY.keys())

    name_to_id = {v: k for k, v in DEMO_REGISTRY.items()}
    selected = []
    for token in demos_str.split(","):
        token = token.strip()
        if token.isdigit():
            n = int(token)
            if n in DEMO_REGISTRY:
                selected.append(n)
            else:
                log.warning("Unknown demo number %d, skipping", n)
        else:
            found = name_to_id.get(token.lower())
            if found:
                selected.append(found)
            else:
                log.warning("Unknown demo name '%s', skipping", token)
    return sorted(set(selected))


class DemoContext:
    """Bundles all rendering data needed by demo functions.

    Attributes:
        images: List of (H, W, 3) GT image tensors.
        viewmats: (N, 4, 4) view matrices on device.
        Ks: (N, 3, 3) intrinsics on device.
        sizes: (N, 2) image sizes [W, H].
        frame_indices: Original frame indices.
        poses_cam_to_world: Camera-to-world transforms per frame.
        W: Image width.
        H: Image height.
        K: (3, 3) intrinsics tensor.
        sh_degree: Active SH degree.
        fps: Video output framerate.
    """

    def __init__(
        self,
        images: list[torch.Tensor],
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        sizes: torch.Tensor,
        frame_indices: list[int],
        poses_cam_to_world: list,
        W: int,
        H: int,
        K: torch.Tensor,
        sh_degree: int,
        fps: int,
    ) -> None:
        """Initialize demo context with all rendering data."""
        self.images = images
        self.viewmats = viewmats
        self.Ks = Ks
        self.sizes = sizes
        self.frame_indices = frame_indices
        self.poses_cam_to_world = poses_cam_to_world
        self.W = W
        self.H = H
        self.K = K
        self.sh_degree = sh_degree
        self.fps = fps


# --- Helper ---

def _render_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a float [0,1] image tensor to uint8 numpy array."""
    return (tensor.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)


def _gt_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a GT image tensor to uint8 numpy array."""
    return (tensor.numpy() * 255).astype(np.uint8)


_TAG_FONT: ImageFont.ImageFont | None = None


def _get_tag_font(size: int = 18) -> ImageFont.ImageFont:
    global _TAG_FONT
    if _TAG_FONT is not None:
        return _TAG_FONT
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            _TAG_FONT = ImageFont.truetype(path, size)
            return _TAG_FONT
        except OSError:
            continue
    _TAG_FONT = ImageFont.load_default()
    return _TAG_FONT


def _draw_tag(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    bg_color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 6
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(*bg_color, 180),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)


def _add_tags(
    frame: np.ndarray,
    left_tag: str | None = None,
    right_tag: str | None = None,
) -> np.ndarray:
    img = Image.fromarray(frame)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_tag_font()
    margin = 10
    half_w = frame.shape[1] // 2
    if left_tag:
        _draw_tag(draw, left_tag, margin, margin, (30, 80, 160), font)
    if right_tag:
        x = (half_w + margin) if left_tag else margin
        _draw_tag(draw, right_tag, x, margin, (30, 130, 76), font)
    result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return np.array(result)


# --- Demo implementations ---

@torch.no_grad()
def demo_replay(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    **_: object,
) -> None:
    """Render side-by-side GT vs predicted for all frames.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, fi in enumerate(ctx.frame_indices):
        gt = ctx.images[i].to(device)
        W, H = ctx.sizes[i, 0].item(), ctx.sizes[i, 1].item()
        combined, _ = scene.compose_frame(fi)
        rendered, _, _ = render_frame(
            combined, ctx.viewmats[i], ctx.Ks[i], W, H, ctx.sh_degree
        )
        side = np.concatenate(
            [_render_to_uint8(gt), _render_to_uint8(rendered)], axis=1
        )
        side = _add_tags(side, "Original", "Rendered")
        Image.fromarray(side).save(frames_dir / f"{i:04d}.png")

    if frames_to_video(frames_dir, out_dir / "replay.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_novel_viewpoint(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    **_: object,
) -> None:
    """Render with camera shifted 1m right + 0.5m up.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    viewmats = ctx.viewmats.clone().to(device)
    for i in range(viewmats.shape[0]):
        viewmats[i, 0, 3] += 1.0
        viewmats[i, 1, 3] -= 0.5

    for i, fi in enumerate(ctx.frame_indices):
        W, H = ctx.sizes[i, 0].item(), ctx.sizes[i, 1].item()
        combined, _ = scene.compose_frame(fi)
        rendered, _, _ = render_frame(
            combined, viewmats[i], ctx.Ks[i], W, H, ctx.sh_degree
        )
        rd_np = _render_to_uint8(rendered)
        rd_np = _add_tags(rd_np, right_tag="Rendered - Novel Viewpoint")
        Image.fromarray(rd_np).save(frames_dir / f"{i:04d}.png")

    if frames_to_video(frames_dir, out_dir / "novel_viewpoint.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_object_removal(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    *,
    remove_track_id: int | None = None,
    **_: object,
) -> None:
    """Render with one tracked object removed.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
        remove_track_id: Track ID to remove (auto-picks largest if None).
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if remove_track_id is None:
        remove_track_id = max(
            scene.objects.keys(),
            key=lambda t: scene.objects[t].params["positions"].shape[0],
        )
    obj = scene.objects[remove_track_id]
    log.info(
        "Removing object %d (%s, %d Gaussians)",
        remove_track_id, obj.obj_type, obj.params["positions"].shape[0],
    )

    removed_obj = scene.objects.pop(remove_track_id)
    removed_adc = scene.obj_adcs.pop(remove_track_id, None)

    for i, fi in enumerate(ctx.frame_indices):
        W, H = ctx.sizes[i, 0].item(), ctx.sizes[i, 1].item()
        combined, _ = scene.compose_frame(fi)
        rendered, _, _ = render_frame(
            combined, ctx.viewmats[i], ctx.Ks[i], W, H, ctx.sh_degree
        )
        side = np.concatenate(
            [_gt_to_uint8(ctx.images[i]), _render_to_uint8(rendered)], axis=1
        )
        side = _add_tags(side, "Original", "Rendered - Object Removal")
        Image.fromarray(side).save(frames_dir / f"{i:04d}.png")

    scene.objects[remove_track_id] = removed_obj
    if removed_adc:
        scene.obj_adcs[remove_track_id] = removed_adc

    if frames_to_video(frames_dir, out_dir / "object_removal.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_background_only(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    **_: object,
) -> None:
    """Render with all objects removed — background only.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved_objects = scene.objects
    saved_adcs = scene.obj_adcs
    scene.objects = {}
    scene.obj_adcs = {}

    for i, fi in enumerate(ctx.frame_indices):
        W, H = ctx.sizes[i, 0].item(), ctx.sizes[i, 1].item()
        combined, _ = scene.compose_frame(fi)
        rendered, _, _ = render_frame(
            combined, ctx.viewmats[i], ctx.Ks[i], W, H, ctx.sh_degree
        )
        side = np.concatenate(
            [_gt_to_uint8(ctx.images[i]), _render_to_uint8(rendered)], axis=1
        )
        side = _add_tags(side, "Original", "Rendered - Background Only")
        Image.fromarray(side).save(frames_dir / f"{i:04d}.png")

    scene.objects = saved_objects
    scene.obj_adcs = saved_adcs

    if frames_to_video(frames_dir, out_dir / "background_only.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_freeze_frame(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    **_: object,
) -> None:
    """Gentle camera orbit around the frame with most visible objects.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    best_idx, best_count = 0, 0
    n_frames = len(ctx.frame_indices)
    for idx, fi in enumerate(ctx.frame_indices):
        if idx < n_frames * 0.15 or idx > n_frames * 0.85:
            continue
        n_vis = len(scene.get_visible_objects(fi))
        if n_vis > best_count:
            best_count = n_vis
            best_idx = idx
    freeze_fi = ctx.frame_indices[best_idx]
    log.info("Freeze at frame %d (%d visible objects)", freeze_fi, best_count)

    base_vm = ctx.viewmats[best_idx].clone().to(device)
    K = ctx.Ks[best_idx].to(device)

    for i in range(120):
        angle = (i / 120) * 2 * math.pi * 0.25
        vm = base_vm.clone()
        vm[0, 3] += 1.0 * math.sin(angle)
        vm[1, 3] += -0.3 * math.cos(angle * 2)
        vm[2, 3] += 0.5 * (1 - math.cos(angle))

        combined, _ = scene.compose_frame(freeze_fi)
        rendered, _, _ = render_frame(
            combined, vm, K, ctx.W, ctx.H, ctx.sh_degree
        )
        rd_np = _render_to_uint8(rendered)
        rd_np = _add_tags(rd_np, right_tag="Rendered - Freeze Frame")
        Image.fromarray(rd_np).save(frames_dir / f"{i:04d}.png")

    if frames_to_video(frames_dir, out_dir / "freeze_frame.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_scene_recomposition(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    *,
    recomp_track_id: int | None = None,
    recomp_offset: tuple[float, float, float] = (8.0, 0.0, 0.0),
    **_: object,
) -> None:
    """Move one tracked object by an offset and render the result.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
        recomp_track_id: Track ID to move (auto-picks largest if None).
        recomp_offset: XYZ offset in meters.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if recomp_track_id is None:
        recomp_track_id = max(
            scene.objects.keys(),
            key=lambda t: scene.objects[t].params["positions"].shape[0],
        )
    obj = scene.objects[recomp_track_id]
    offset = torch.tensor(recomp_offset, dtype=torch.float32, device=device)
    log.info(
        "Moving object %d (%s) by %s",
        recomp_track_id, obj.obj_type, recomp_offset,
    )

    original_poses = {}
    for fi, (R, t) in obj.frame_poses.items():
        original_poses[fi] = (R, t.clone())
        obj.frame_poses[fi] = (R, t + offset)

    for i, fi in enumerate(ctx.frame_indices):
        W, H = ctx.sizes[i, 0].item(), ctx.sizes[i, 1].item()
        combined, _ = scene.compose_frame(fi)
        rendered, _, _ = render_frame(
            combined, ctx.viewmats[i], ctx.Ks[i], W, H, ctx.sh_degree
        )
        side = np.concatenate(
            [_gt_to_uint8(ctx.images[i]), _render_to_uint8(rendered)], axis=1
        )
        side = _add_tags(side, "Original", "Rendered - Scene Recomposition")
        Image.fromarray(side).save(frames_dir / f"{i:04d}.png")

    for fi, (R, t) in original_poses.items():
        obj.frame_poses[fi] = (R, t)

    if frames_to_video(frames_dir, out_dir / "scene_recomposition.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_depth_map(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    **_: object,
) -> None:
    """Render depth maps using alpha-weighted Gaussian z-values.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, fi in enumerate(ctx.frame_indices):
        W, H = ctx.sizes[i, 0].item(), ctx.sizes[i, 1].item()
        combined, _ = scene.compose_frame(fi)

        n_pts = combined["positions"].shape[0]
        ones = torch.ones(n_pts, 1, device=device)
        pos_h = torch.cat([combined["positions"], ones], dim=1)
        depths = (ctx.viewmats[i] @ pos_h.T).T[:, 2:3]
        depth_sh_3ch = depths.unsqueeze(1).expand(-1, 1, 3)

        rendered_depth, _, _ = gsplat.rasterization(
            means=combined["positions"],
            quats=combined["quaternions"],
            scales=torch.exp(combined["log_scales"]),
            opacities=torch.sigmoid(
                combined["logit_opacities"]
            ).squeeze(-1),
            colors=depth_sh_3ch,
            sh_degree=0,
            viewmats=ctx.viewmats[i].unsqueeze(0),
            Ks=ctx.Ks[i].unsqueeze(0),
            width=W, height=H,
            packed=False, absgrad=False,
        )
        depth_img = rendered_depth[0, :, :, 0].cpu().numpy()

        valid = depth_img > 0
        if valid.any():
            d_min = depth_img[valid].min()
            d_max = np.percentile(depth_img[valid], 95)
            depth_norm = np.clip(
                (depth_img - d_min) / (d_max - d_min + 1e-6), 0, 1
            )
        else:
            depth_norm = np.zeros_like(depth_img)

        depth_colored = (
            plt.cm.turbo(1.0 - depth_norm)[:, :, :3] * 255
        ).astype(np.uint8)
        gt_np = _gt_to_uint8(ctx.images[i])
        side = np.concatenate([gt_np, depth_colored], axis=1)
        side = _add_tags(side, "Original", "Depth Map")
        Image.fromarray(side).save(frames_dir / f"{i:04d}.png")

    if frames_to_video(frames_dir, out_dir / "depth_map.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_bev_occupancy(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    **_: object,
) -> None:
    """Render top-down BEV showing active objects per frame.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    colors_map = {
        "Car": "red", "Van": "orange", "Truck": "brown",
        "Pedestrian": "blue", "Cyclist": "green",
    }
    bg_pos = scene.bg_params["positions"].detach().cpu().numpy()
    x_min, x_max = bg_pos[:, 0].min() - 5, bg_pos[:, 0].max() + 5
    y_min, y_max = bg_pos[:, 1].min() - 5, bg_pos[:, 1].max() + 5

    ego_x = [
        ctx.poses_cam_to_world[fi][:3, 3][0] for fi in ctx.frame_indices
    ]
    ego_y = [
        ctx.poses_cam_to_world[fi][:3, 3][1] for fi in ctx.frame_indices
    ]

    n_sub = min(10000, bg_pos.shape[0])
    sub_idx = np.random.choice(bg_pos.shape[0], n_sub, replace=False)
    bg_sub = bg_pos[sub_idx]

    for i, fi in enumerate(ctx.frame_indices):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.scatter(bg_sub[:, 0], bg_sub[:, 1], c="gray", s=0.3, alpha=0.2)
        ax.plot(ego_x, ego_y, c="white", linewidth=2, alpha=0.5)
        ax.scatter(
            [ego_x[i]], [ego_y[i]], c="yellow", s=80,
            zorder=5, edgecolors="black", linewidths=1,
        )

        visible = scene.get_visible_objects(fi)
        categories_plotted: set[str] = set()
        for tid, obj in visible.items():
            R_obj, t_obj = obj.frame_poses[fi]
            pos_w = (
                (obj.params["positions"].detach() @ R_obj.T) + t_obj
            ).cpu().numpy()
            c = colors_map.get(obj.obj_type, "purple")
            ax.scatter(pos_w[:, 0], pos_w[:, 1], c=c, s=2, alpha=0.7)
            categories_plotted.add(obj.obj_type)

        handles = [
            Line2D([0], [0], color="white", linewidth=2, label="Ego path"),
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor="yellow", markeredgecolor="black",
                markersize=8, linestyle="None", label="Current pos",
            ),
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor="gray", markersize=6,
                linestyle="None", label="Background",
            ),
        ]
        for cat in sorted(categories_plotted):
            handles.append(Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=colors_map.get(cat, "purple"),
                markersize=6, linestyle="None", label=cat,
            ))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a1a")
        ax.set_title(
            f"BEV Occupancy — Frame {fi} ({len(visible)} objects)",
            fontsize=12,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(
            handles=handles, loc="upper left", fontsize=9, framealpha=0.8,
        )
        fig.tight_layout()
        fig.savefig(frames_dir / f"{i:04d}.png")
        plt.close(fig)

    if frames_to_video(frames_dir, out_dir / "bev_occupancy.mp4", ctx.fps):
        shutil.rmtree(frames_dir, ignore_errors=True)


@torch.no_grad()
def demo_temporal_interpolation(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    out_dir: Path,
    *,
    interp_factor: int = 5,
    **_: object,
) -> None:
    """Render slow-motion via pose interpolation between consecutive frames.

    Args:
        scene: Trained scene model.
        ctx: Demo context.
        out_dir: Output directory.
        interp_factor: Number of sub-frames between each real frame.
    """
    device = scene.bg_params["positions"].device
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    N = interp_factor
    start = len(ctx.frame_indices) // 4
    end = min(start + 60, len(ctx.frame_indices) - 1)

    out_idx = 0
    for i in range(start, end):
        fi_a = ctx.frame_indices[i]
        fi_b = ctx.frame_indices[i + 1]
        W, H = ctx.sizes[i, 0].item(), ctx.sizes[i, 1].item()

        for s in range(N):
            alpha = s / N
            vm_a, vm_b = ctx.viewmats[i], ctx.viewmats[i + 1]
            R_interp, t_interp = interpolate_pose(
                vm_a[:3, :3], vm_a[:3, 3],
                vm_b[:3, :3], vm_b[:3, 3], alpha,
            )
            vm_interp = torch.eye(4, device=device)
            vm_interp[:3, :3] = R_interp
            vm_interp[:3, 3] = t_interp

            combined = scene.compose_frame_interpolated(fi_a, fi_b, alpha)
            rendered, _, _ = render_frame(
                combined, vm_interp, ctx.Ks[i], W, H, ctx.sh_degree
            )
            rd_np = _render_to_uint8(rendered)
            rd_np = _add_tags(rd_np, right_tag="Rendered - Temporal Interpolation")
            Image.fromarray(rd_np).save(frames_dir / f"{out_idx:04d}.png")
            out_idx += 1

    if frames_to_video(
        frames_dir, out_dir / "temporal_interpolation.mp4", ctx.fps
    ):
        shutil.rmtree(frames_dir, ignore_errors=True)
    log.info(
        "%d interpolated frames from %d real frames (%dx slow motion)",
        out_idx, end - start, N,
    )


# --- Runner ---

DEMO_FUNCTIONS: dict[int, tuple[str, Callable]] = {
    1: ("Replay", demo_replay),
    2: ("Novel Viewpoint", demo_novel_viewpoint),
    3: ("Object Removal", demo_object_removal),
    4: ("Background Only", demo_background_only),
    5: ("Freeze Frame", demo_freeze_frame),
    6: ("Scene Recomposition", demo_scene_recomposition),
    7: ("Depth Map", demo_depth_map),
    8: ("BEV Occupancy", demo_bev_occupancy),
    9: ("Temporal Interpolation", demo_temporal_interpolation),
}


def run_demos(
    scene: StreetGaussianScene,
    ctx: DemoContext,
    selected: list[int],
    out_dir: Path,
    **kwargs: object,
) -> dict[str, dict]:
    """Run selected demos with fault tolerance.

    Args:
        scene: Trained scene model.
        ctx: Demo context with all rendering data.
        selected: List of demo IDs to run.
        out_dir: Output directory.
        **kwargs: Extra args forwarded to individual demos.

    Returns:
        Dict of demo_name -> {status, time_sec, error?}.
    """
    results: dict[str, dict] = {}
    for demo_id in selected:
        name, func = DEMO_FUNCTIONS[demo_id]
        demo_dir = out_dir / DEMO_REGISTRY[demo_id]
        demo_dir.mkdir(parents=True, exist_ok=True)

        log.info("Demo %d: %s", demo_id, name)
        t0 = time.time()
        try:
            func(scene, ctx, demo_dir, **kwargs)
            elapsed = time.time() - t0
            results[name] = {
                "status": "SUCCESS", "time_sec": round(elapsed, 1),
            }
            log.info("  Completed in %.1fs", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {
                "status": "FAILED",
                "error": str(e),
                "time_sec": round(elapsed, 1),
            }
            log.error("  FAILED after %.1fs: %s", elapsed, e)
            traceback.print_exc()

    return results
