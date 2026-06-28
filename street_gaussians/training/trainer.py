"""Training loop for Street Gaussians."""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from street_gaussians.config import Config
from street_gaussians.evaluation.metrics import compute_psnr
from street_gaussians.evaluation.profiler import PipelineTimer, get_gpu_memory_mb
from street_gaussians.models.scene import StreetGaussianScene
from street_gaussians.rendering.renderer import render_frame
from street_gaussians.training.loss import compute_loss
from street_gaussians.training.optimizer import create_scene_optimizer, update_position_lr
from street_gaussians.utils.checkpoint import save_checkpoint
from street_gaussians.utils.logger import get_logger
from street_gaussians.utils.video import frames_to_video

log = get_logger(__name__)


def _setup_tensorboard(output_dir: str | Path) -> object | None:
    """Attempt to create a TensorBoard SummaryWriter.

    Args:
        output_dir: Base output directory for log files.

    Returns:
        SummaryWriter instance, or None if tensorboard is not installed.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(Path(output_dir) / "tb_logs"))
        log.info("TensorBoard logging -> %s/tb_logs", output_dir)
        return writer
    except ImportError:
        log.info("TensorBoard not available — install tensorboard for live logging")
        return None


def _save_progression(
    scene: StreetGaussianScene,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    ref_idx: int,
    ref_fi: int,
    ref_gt: torch.Tensor,
    ref_W: int,
    ref_H: int,
    overview_vm: torch.Tensor,
    overview_K: torch.Tensor,
    sh_degree: int,
    step: int,
    prog_count: int,
    progression_dir: Path,
    overview_dir: Path,
    bev_dir: Path,
) -> None:
    """Save training progression snapshots (render, overview, BEV)."""
    combined, _ = scene.compose_frame(ref_fi)
    snap_sh = min(step // 1000, sh_degree)
    rendered, _, _ = render_frame(
        combined, viewmats[ref_idx], Ks[ref_idx], ref_W, ref_H, snap_sh
    )

    gt_np = (ref_gt.cpu().numpy() * 255).astype(np.uint8)
    rd_np = (rendered.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    n_now = scene.bg_params["positions"].shape[0] + sum(
        o.params["positions"].shape[0] for o in scene.objects.values()
    )
    label = f"Iter {step:,}  |  #G={n_now:,}"

    side = np.concatenate([gt_np, rd_np], axis=1)
    pil_img = Image.fromarray(side)
    draw = ImageDraw.Draw(pil_img)
    draw.text((rd_np.shape[1] + 10, 10), label, fill=(255, 255, 0))
    pil_img.save(progression_dir / f"{prog_count:04d}.png")

    ov_rendered, _, _ = render_frame(
        combined, overview_vm, overview_K, ref_W, ref_H, snap_sh
    )
    ov_np = (ov_rendered.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    ov_img = Image.fromarray(ov_np)
    ImageDraw.Draw(ov_img).text((10, 10), label, fill=(255, 255, 0))
    ov_img.save(overview_dir / f"{prog_count:04d}.png")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=80)
    bg_pos = scene.bg_params["positions"].detach().cpu().numpy()
    sub_idx = np.random.choice(bg_pos.shape[0], min(5000, bg_pos.shape[0]), replace=False)
    ax.scatter(bg_pos[sub_idx, 0], bg_pos[sub_idx, 1], c="gray", s=0.3, alpha=0.2)
    for tid, obj in scene.get_visible_objects(ref_fi).items():
        R_o, t_o = obj.frame_poses[ref_fi]
        pos_w = ((obj.params["positions"].detach() @ R_o.T) + t_o).cpu().numpy()
        ax.scatter(pos_w[:, 0], pos_w[:, 1], c="red", s=1, alpha=0.5)
    ax.set_title(f"BEV — Iter {step:,} | #G={n_now:,}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(bev_dir / f"{prog_count:04d}.png")
    plt.close(fig)


def _build_overview_camera(
    viewmat: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    """Create an elevated 3/4 overview camera from a reference viewmat.

    Args:
        viewmat: (4, 4) reference view matrix.
        device: Target device.

    Returns:
        (4, 4) overview view matrix with elevation and tilt.
    """
    overview_vm = viewmat.clone().to(device)
    overview_vm[1, 3] -= 15.0
    overview_vm[2, 3] -= 20.0

    tilt = torch.tensor([
        [1, 0, 0, 0],
        [0, math.cos(-0.5), -math.sin(-0.5), 0],
        [0, math.sin(-0.5), math.cos(-0.5), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)
    return tilt @ overview_vm


def train(
    scene: StreetGaussianScene,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    gt_images: list[torch.Tensor],
    frame_indices: list[int],
    image_sizes: torch.Tensor,
    cfg: Config,
    timer: PipelineTimer | None = None,
) -> tuple[list[float], list[float], list[int]]:
    """Run the training loop.

    Args:
        scene: Initialized scene model.
        viewmats: (N, 4, 4) view matrices for training frames.
        Ks: (N, 3, 3) intrinsics.
        gt_images: List of (H, W, 3) ground truth tensors.
        frame_indices: Original frame indices for each training image.
        image_sizes: (N, 2) tensor of [W, H].
        cfg: Pipeline config.
        timer: Optional profiler.

    Returns:
        Tuple of (losses, psnr_log, count_log) lists.
    """
    tcfg = cfg.training
    device = scene.bg_params["positions"].device
    N_views = len(gt_images)

    gt_dev = [img.to(device) for img in gt_images]
    viewmats = viewmats.to(device)
    Ks = Ks.to(device)
    image_sizes = image_sizes.to(device)

    max_iters = tcfg.iterations
    densify_from = max(500, max_iters // 60)
    densify_until = max(1000, max_iters // 2)

    optimizer = create_scene_optimizer(scene, tcfg)
    bg_pos = scene.bg_params["positions"].data
    scene_ext = (bg_pos - bg_pos.mean(0)).norm(dim=-1).max().item()

    writer = _setup_tensorboard(cfg.output.dir)

    prog_interval = cfg.output.progression_interval
    out_dir = Path(cfg.output.dir)
    progression_dir = out_dir / "progression_frames"
    overview_dir = out_dir / "overview_progression_frames"
    bev_dir = out_dir / "bev_progression_frames"
    for d in [progression_dir, overview_dir, bev_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ref_idx = N_views // 2
    ref_fi = frame_indices[ref_idx]
    ref_gt = gt_dev[ref_idx]
    ref_W, ref_H = image_sizes[ref_idx, 0].item(), image_sizes[ref_idx, 1].item()
    overview_vm = _build_overview_camera(viewmats[ref_idx], device)
    overview_K = Ks[ref_idx]
    prog_count = 0

    losses, psnr_log, count_log = [], [], []
    t0 = time.time()

    for step in range(1, max_iters + 1):
        idx = torch.randint(0, N_views, (1,)).item()
        gt = gt_dev[idx]
        W, H = image_sizes[idx, 0].item(), image_sizes[idx, 1].item()
        frame_idx = frame_indices[idx]
        current_sh = min(step // 1000, tcfg.sh_degree)

        if timer:
            timer.start("compose")
        combined, segments = scene.compose_frame(frame_idx)
        if timer:
            timer.stop("compose")

        if timer:
            timer.start("forward")
        rendered, _, meta = render_frame(combined, viewmats[idx], Ks[idx], W, H, current_sh)
        meta["means2d"].retain_grad()
        loss = compute_loss(rendered, gt, tcfg.lambda_ssim)
        if timer:
            timer.stop("forward")

        if timer:
            timer.start("backward")
        loss.backward()
        scene.route_gradients(meta, segments)
        if timer:
            timer.stop("backward")

        if timer:
            timer.start("optimizer")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_position_lr(optimizer, step, tcfg)
        if timer:
            timer.stop("optimizer")

        # ADC
        if densify_from <= step <= densify_until and step % tcfg.densify_interval == 0:
            if timer:
                timer.start("adc")
            scene.bg_params = scene.bg_adc.densify_and_prune(
                scene.bg_params, scene_ext, tcfg.grad_threshold, tcfg.opacity_prune_threshold
            )
            scene.bg_adc.reset_stats(scene.bg_params["positions"].shape[0], device)
            for tid, obj in scene.objects.items():
                if obj.params["positions"].shape[0] > 0:
                    adc = scene.obj_adcs[tid]
                    if adc.grad_accum is not None:
                        obj.params = adc.densify_and_prune(
                            obj.params, scene_ext, tcfg.grad_threshold, tcfg.opacity_prune_threshold
                        )
                        adc.reset_stats(obj.params["positions"].shape[0], device)
            optimizer = create_scene_optimizer(scene, tcfg)
            if timer:
                timer.stop("adc")

        if step % tcfg.opacity_reset_interval == 0 and 0 < step < max_iters:
            scene.bg_adc.reset_opacity(scene.bg_params)
            for tid, obj in scene.objects.items():
                scene.obj_adcs[tid].reset_opacity(obj.params)
            optimizer = create_scene_optimizer(scene, tcfg)

        if step % 100 == 0:
            l = loss.item()
            p = compute_psnr(rendered.detach(), gt)
            n_bg = scene.bg_params["positions"].shape[0]
            n_obj = sum(o.params["positions"].shape[0] for o in scene.objects.values())
            n = n_bg + n_obj
            losses.append(l)
            psnr_log.append(p)
            count_log.append(n)
            elapsed = time.time() - t0

            gpu_mem = get_gpu_memory_mb()
            mem_str = f"  GPU={gpu_mem[0]:.0f}MB" if gpu_mem else ""
            log.info(
                "[%5d/%d] loss=%.4f  PSNR=%.2fdB  #G=%s (bg=%s obj=%s)  t=%.0fs%s",
                step, max_iters, l, p, f"{n:,}", f"{n_bg:,}", f"{n_obj:,}", elapsed, mem_str,
            )

            if writer:
                writer.add_scalar("train/loss", l, step)
                writer.add_scalar("train/psnr", p, step)
                writer.add_scalar("train/n_gaussians", n, step)
                writer.add_scalar("train/n_bg", n_bg, step)
                writer.add_scalar("train/n_obj", n_obj, step)
                if gpu_mem:
                    writer.add_scalar("memory/gpu_allocated_mb", gpu_mem[0], step)
                    writer.add_scalar("memory/gpu_peak_mb", gpu_mem[2], step)

        if step % tcfg.checkpoint_interval == 0:
            save_checkpoint(scene, step, cfg, out_dir)

        if step % prog_interval == 0 or step == 1:
            with torch.no_grad():
                _save_progression(
                    scene, viewmats, Ks, ref_idx, ref_fi, ref_gt, ref_W, ref_H,
                    overview_vm, overview_K, tcfg.sh_degree, step, prog_count,
                    progression_dir, overview_dir, bev_dir,
                )
                prog_count += 1

    if writer:
        writer.close()

    for name, fdir, vid_name in [
        ("Render progression", progression_dir, "training_progression.mp4"),
        ("Overview progression", overview_dir, "overview_progression.mp4"),
        ("BEV progression", bev_dir, "bev_progression.mp4"),
    ]:
        frames_to_video(fdir, out_dir / vid_name, fps=5)

    return losses, psnr_log, count_log
