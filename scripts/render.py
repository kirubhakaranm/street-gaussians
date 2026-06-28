"""Entry point — render demos from a trained checkpoint.

Usage:
    python scripts/render.py --config configs/default.yaml \\
        --checkpoint output/checkpoint_050000.pt --demos all
    python scripts/render.py --config configs/default.yaml \\
        --checkpoint ckpt.pt --demos replay,depth_map
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from street_gaussians.config import load_config
from street_gaussians.data import get_source
from street_gaussians.rendering.demos import (
    DEMO_REGISTRY,
    DemoContext,
    parse_demo_selection,
    run_demos,
)
from street_gaussians.utils.checkpoint import load_checkpoint
from street_gaussians.utils.logger import get_logger

log = get_logger("render")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Street Gaussians — Render Demos")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--overrides", type=str, default=None, help="Optional override YAML")
    p.add_argument(
        "--demos", type=str, default="all",
        help="Comma-separated demo IDs or names, or 'all'",
    )
    p.add_argument("--output_dir", type=str, default="demos", help="Output directory")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--remove_track_id", type=int, default=None)
    p.add_argument("--recomp_track_id", type=int, default=None)
    p.add_argument("--recomp_offset", type=float, nargs=3, default=[8.0, 0.0, 0.0])
    p.add_argument("--interp_factor", type=int, default=5)
    return p.parse_args()


def main() -> None:
    """Load checkpoint and render selected demos."""
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    log.info("Loading checkpoint: %s", args.checkpoint)
    scene, step = load_checkpoint(args.checkpoint, device=device)
    n_bg = scene.bg_params["positions"].shape[0]
    n_obj = sum(o.params["positions"].shape[0] for o in scene.objects.values())
    log.info("Step %d, %s Gaussians (bg=%s, obj=%s), %d objects",
             step, f"{n_bg + n_obj:,}", f"{n_bg:,}", f"{n_obj:,}", len(scene.objects))

    # Load data
    log.info("Loading data")
    source = get_source(cfg.data)
    scene_data = source.load()
    H, W = scene_data.images[0].shape[0], scene_data.images[0].shape[1]

    K_torch = torch.tensor(scene_data.K, dtype=torch.float32)
    all_indices = list(range(len(scene_data.images)))
    all_viewmats = torch.stack(
        [torch.tensor(scene_data.viewmats[i]) for i in all_indices]
    ).to(device)
    all_Ks = K_torch.unsqueeze(0).expand(len(all_indices), -1, -1).contiguous().to(device)
    all_sizes = torch.tensor([[W, H]] * len(all_indices), dtype=torch.int32).to(device)

    ctx = DemoContext(
        images=scene_data.images,
        viewmats=all_viewmats,
        Ks=all_Ks,
        sizes=all_sizes,
        frame_indices=all_indices,
        poses_cam_to_world=scene_data.poses_cam_to_world,
        W=W, H=H, K=K_torch,
        sh_degree=cfg.training.sh_degree,
        fps=args.fps,
    )

    selected = parse_demo_selection(args.demos)
    log.info("Selected demos: %s", [DEMO_REGISTRY[d] for d in selected])

    results = run_demos(
        scene, ctx, selected, out_dir,
        remove_track_id=args.remove_track_id,
        recomp_track_id=args.recomp_track_id,
        recomp_offset=tuple(args.recomp_offset),
        interp_factor=args.interp_factor,
    )

    # Summary
    log.info("=" * 60)
    log.info("DEMO SUMMARY")
    for name, r in results.items():
        if r["status"] == "SUCCESS":
            log.info("  %-25s  %s  (%.1fs)", name, r["status"], r["time_sec"])
        else:
            log.info("  %-25s  %s  — %s", name, r["status"], r.get("error", ""))

    with open(out_dir / "demo_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
