"""Entry point — train a Street Gaussians scene from YAML config.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --overrides configs/kitti_0001.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from street_gaussians.config import load_config
from street_gaussians.data import get_source
from street_gaussians.evaluation.metrics import evaluate
from street_gaussians.evaluation.profiler import PipelineTimer
from street_gaussians.models.gaussians import init_gaussians, save_ply
from street_gaussians.models.scene import StreetGaussianScene, build_object_models
from street_gaussians.preprocessing.decompose import decompose_scene
from street_gaussians.training.trainer import train
from street_gaussians.utils.checkpoint import save_checkpoint
from street_gaussians.utils.logger import get_logger

log = get_logger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Street Gaussians — Train")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--overrides", type=str, default=None, help="Optional override YAML")
    return p.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = PipelineTimer(device=device)

    # --- Load data ---
    log.info("Loading data (source=%s)", cfg.data.source)
    timer.start("data_loading")
    source = get_source(cfg.data)
    scene_data = source.load()
    num_frames = len(scene_data.images)
    H, W = scene_data.images[0].shape[0], scene_data.images[0].shape[1]
    timer.stop("data_loading")

    # --- Train/test split ---
    all_indices = list(range(len(scene_data.images)))
    if cfg.data.test_every > 0:
        train_indices = [i for i in all_indices if i % cfg.data.test_every != 0]
        test_indices = [i for i in all_indices if i % cfg.data.test_every == 0]
    else:
        train_indices = all_indices
        test_indices = all_indices
    log.info("Split: %d train / %d test frames", len(train_indices), len(test_indices))

    K_torch = torch.tensor(scene_data.K, dtype=torch.float32)

    def _build_batch(indices: list[int]) -> tuple:
        """Build batched tensors for a set of frame indices."""
        vms = torch.stack([torch.tensor(scene_data.viewmats[i]) for i in indices])
        ks = K_torch.unsqueeze(0).expand(len(indices), -1, -1).contiguous()
        imgs = [scene_data.images[i] for i in indices]
        sizes = torch.tensor([[W, H]] * len(indices), dtype=torch.int32)
        return vms, ks, imgs, sizes

    train_vms, train_Ks, train_imgs, train_sizes = _build_batch(train_indices)
    test_vms, test_Ks, test_imgs, test_sizes = _build_batch(test_indices)

    # --- Scene decomposition ---
    log.info("Decomposing scene")
    timer.start("decomposition")
    bg_cloud, obj_clouds = decompose_scene(scene_data, num_frames)
    timer.stop("decomposition")

    # --- Build scene model ---
    log.info("Building scene model")
    timer.start("scene_init")

    n_bg_raw = bg_cloud.positions.shape[0]
    if n_bg_raw > cfg.data.max_bg_points:
        idx = np.random.choice(n_bg_raw, cfg.data.max_bg_points, replace=False)
        bg_cloud.positions = bg_cloud.positions[idx]
        bg_cloud.colors = bg_cloud.colors[idx]
        log.info(
            "Subsampled background: %s -> %s points",
            f"{n_bg_raw:,}", f"{cfg.data.max_bg_points:,}",
        )

    bg_params = init_gaussians(
        bg_cloud.positions, bg_cloud.colors,
        sh_degree=cfg.training.sh_degree, device=device,
    )
    log.info("Background: %s Gaussians", f"{bg_params['positions'].shape[0]:,}")

    if cfg.data.max_object_points > 0:
        for tid in obj_clouds:
            n_pts = obj_clouds[tid].positions.shape[0]
            if n_pts > cfg.data.max_object_points:
                idx = np.random.choice(n_pts, cfg.data.max_object_points, replace=False)
                obj_clouds[tid].positions = obj_clouds[tid].positions[idx]
                obj_clouds[tid].colors = obj_clouds[tid].colors[idx]

    object_models = build_object_models(
        scene_data, obj_clouds, cfg.training.sh_degree, cfg.data.min_object_points, device
    )
    scene = StreetGaussianScene(bg_params, object_models)

    n_total = bg_params["positions"].shape[0] + sum(
        o.params["positions"].shape[0] for o in object_models.values()
    )
    log.info("Total: %s Gaussians (%d objects + background)", f"{n_total:,}", len(object_models))
    timer.stop("scene_init")

    # --- Train ---
    log.info(
        "Training: %d iterations, SH degree %d",
        cfg.training.iterations, cfg.training.sh_degree,
    )
    timer.start("training_total")
    t_start = time.time()
    losses, psnr_log, count_log = train(
        scene, train_vms, train_Ks, train_imgs, train_indices, train_sizes, cfg, timer=timer,
    )
    train_time = time.time() - t_start
    timer.stop("training_total")
    log.info("Training done in %.1f min", train_time / 60)

    # --- Evaluate ---
    timer.start("evaluation")
    train_metrics = evaluate(
        scene, train_vms, train_Ks, train_imgs,
        train_indices, train_sizes, cfg.training.sh_degree,
    )
    test_metrics = evaluate(
        scene, test_vms, test_Ks, test_imgs,
        test_indices, test_sizes, cfg.training.sh_degree,
    )
    timer.stop("evaluation")

    n_bg = scene.bg_params["positions"].shape[0]
    n_obj = sum(o.params["positions"].shape[0] for o in scene.objects.values())

    log.info("=" * 60)
    log.info("RESULTS — Street Gaussians Experiment A (LiDAR + GT)")
    log.info("=" * 60)
    log.info("  PSNR (train):  %.2f dB", train_metrics["psnr"])
    log.info("  PSNR (test):   %.2f dB", test_metrics["psnr"])
    log.info("  SSIM (test):   %.4f", test_metrics["ssim"])
    if "lpips" in test_metrics:
        log.info("  LPIPS (test):  %.4f", test_metrics["lpips"])
    log.info("  #Gaussians:    %s (bg=%s, obj=%s)", f"{n_bg + n_obj:,}", f"{n_bg:,}", f"{n_obj:,}")
    log.info("  #Objects:      %d", len(scene.objects))
    log.info("  Train time:    %.0fs", train_time)

    # --- Save results ---
    results = {
        "experiment": "A_lidar_gt",
        "psnr_train": train_metrics["psnr"],
        "psnr_test": test_metrics["psnr"],
        "ssim_test": test_metrics["ssim"],
        "ssim_train": train_metrics["ssim"],
        "lpips_test": test_metrics.get("lpips"),
        "lpips_train": train_metrics.get("lpips"),
        "n_gaussians_bg": n_bg,
        "n_gaussians_obj": n_obj,
        "n_objects": len(scene.objects),
        "train_time_sec": train_time,
        "iterations": cfg.training.iterations,
        "sh_degree": cfg.training.sh_degree,
        "sequence": cfg.data.sequence,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Final checkpoint + PLY
    timer.start("save_outputs")
    save_checkpoint(scene, cfg.training.iterations, cfg, out_dir)

    combined, _ = scene.compose_frame(0)
    detached = {k: nn.Parameter(v.detach()) for k, v in combined.items()}
    save_ply(detached, out_dir / "gaussians.ply")

    # Training curves
    iters = list(range(100, (len(losses) + 1) * 100, 100))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(iters, losses); axes[0].set_title("Loss"); axes[0].set_xlabel("Iter")
    axes[1].plot(iters, psnr_log); axes[1].set_title("PSNR (dB)"); axes[1].set_xlabel("Iter")
    axes[2].plot(iters, count_log); axes[2].set_title("#Gaussians"); axes[2].set_xlabel("Iter")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()
    timer.stop("save_outputs")

    # Profiling
    profile_text = timer.summary()
    log.info(profile_text)
    with open(out_dir / "profiling.txt", "w") as f:
        f.write(profile_text)


if __name__ == "__main__":
    main()
