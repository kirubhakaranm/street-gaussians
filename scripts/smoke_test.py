"""Smoke test — validates pipeline with synthetic KITTI data. No GPU needed."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from street_gaussians.data.base import SceneData
from street_gaussians.data.kitti import (
    load_calib,
    load_images,
    load_labels,
    load_oxts,
    load_velodyne,
)
from street_gaussians.models.gaussians import init_gaussians
from street_gaussians.models.math_utils import quat_multiply
from street_gaussians.models.scene import StreetGaussianScene, build_object_models
from street_gaussians.preprocessing.decompose import decompose_scene, point_in_box
from street_gaussians.utils.logger import get_logger

log = get_logger("smoke_test")


def create_synthetic_kitti(root: Path, n_frames: int = 10) -> None:
    """Create a minimal synthetic KITTI tracking dataset.

    Args:
        root: Directory to write synthetic data into.
        n_frames: Number of frames to generate.
    """
    seq = "0001"

    calib_dir = root / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    P2 = (
        "7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 "
        "0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 "
        "0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03"
    )
    Tr = (
        "7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 "
        "1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 "
        "9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01"
    )
    Tr_imu = (
        "9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 "
        "-7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 "
        "2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"
    )
    with open(calib_dir / f"{seq}.txt", "w") as f:
        f.write(f"P0: {P2}\nP1: {P2}\nP2: {P2}\nP3: {P2}\n")
        f.write("R_rect: 1 0 0 0 1 0 0 0 1\n")
        f.write(f"Tr_velo_cam: {Tr}\nTr_imu_velo: {Tr_imu}\n")

    oxts_dir = root / "oxts" / seq
    oxts_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        vals = [49.0, 8.0 + i * 0.00001, 115.0, 0.0, 0.0, 0.01 * i] + [0.0] * 24
        with open(oxts_dir / f"{i:06d}.txt", "w") as f:
            f.write(" ".join(f"{v:.10e}" for v in vals))

    label_dir = root / "label_02"
    label_dir.mkdir(parents=True, exist_ok=True)
    with open(label_dir / f"{seq}.txt", "w") as f:
        for frame in range(n_frames):
            x = 2.0 + frame * 0.1
            f.write(
                f"{frame} 0 Car 0.0 0 0.0 "
                f"100 150 300 350 1.5 1.8 4.5 {x} 1.5 15.0 -0.1\n"
            )
            f.write(
                f"{frame} 1 Pedestrian 0.0 0 0.0 "
                "400 200 450 380 1.7 0.6 0.8 -1.0 1.7 8.0 0.0\n"
            )

    velo_dir = root / "velodyne" / seq
    velo_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        pts = rng.uniform([-20, -20, -2], [50, 20, 3], size=(5000, 3)).astype(np.float32)
        car_pts = rng.normal(
            [15.0, -2.0, -1.5], [2.0, 0.5, 0.5], size=(200, 3)
        ).astype(np.float32)
        pts = np.vstack([pts, car_pts])
        intensity = rng.uniform(0, 1, size=(pts.shape[0], 1)).astype(np.float32)
        np.hstack([pts, intensity]).tofile(str(velo_dir / f"{i:06d}.bin"))

    img_dir = root / "image_02" / seq
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = np.random.randint(50, 200, (375, 1242, 3), dtype=np.uint8)
        Image.fromarray(img).save(img_dir / f"{i:06d}.png")

    log.info("Created synthetic KITTI at %s (%d frames)", root, n_frames)


def run_tests(root: Path) -> None:
    """Run all smoke tests against synthetic data.

    Args:
        root: Path to synthetic KITTI dataset.
    """
    seq = "0001"
    passed = 0
    total = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        """Assert a test condition and log the result."""
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" — {detail}"
        log.info(msg)
        assert condition, f"{name} failed: {detail}"

    log.info("\n--- Calibration ---")
    K, Tr, T_cam_from_imu, R_rect = load_calib(root / "calib" / f"{seq}.txt")
    check("K shape", K.shape == (3, 3))
    check("Tr shape", Tr.shape == (4, 4))
    check("fx value", abs(K[0, 0] - 721.5377) < 0.01, f"fx={K[0, 0]:.1f}")

    log.info("\n--- OXTS ---")
    viewmats, poses = load_oxts(root / "oxts" / seq, T_cam_from_imu)
    check("viewmats count", len(viewmats) == 10)
    check("viewmat shape", viewmats[0].shape == (4, 4))
    diff = np.linalg.norm(viewmats[0][:3, 3] - viewmats[5][:3, 3])
    check("poses change", diff > 0, f"displacement={diff:.4f}m")

    log.info("\n--- Labels ---")
    tracks = load_labels(root / "label_02" / f"{seq}.txt")
    check("track count", len(tracks) == 2)
    check("Car track", 0 in tracks and tracks[0].obj_type == "Car")
    check("Ped track", 1 in tracks and tracks[1].obj_type == "Pedestrian")

    log.info("\n--- Velodyne ---")
    pts = load_velodyne(root / "velodyne" / seq / "000000.bin", Tr)
    check("points shape", pts.shape[1] == 3 and pts.shape[0] > 1000, f"N={pts.shape[0]}")

    log.info("\n--- Images ---")
    images = load_images(root / "image_02" / seq)
    check("image count", len(images) == 10)
    check("image shape", images[0].shape == (375, 1242, 3))

    log.info("\n--- Point-in-box ---")
    test_pts = np.array(
        [[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [0, 0.9, 0]], dtype=np.float32
    )
    dims = np.array([2, 2, 4], dtype=np.float32)
    mask = point_in_box(test_pts, np.zeros(3, dtype=np.float32), dims, 0.0)
    check("point-in-box", list(mask) == [True, True, False, True])

    log.info("\n--- Quaternions ---")
    identity = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    q = torch.tensor([[0.707, 0.707, 0, 0]], dtype=torch.float32)
    check(
        "identity multiply",
        torch.allclose(quat_multiply(identity, q), q, atol=1e-5),
    )

    log.info("\n--- Gaussian init ---")
    params = init_gaussians(
        np.random.randn(100, 3).astype(np.float32),
        np.random.rand(100, 3).astype(np.float32),
        sh_degree=3, device="cpu",
    )
    check("positions shape", params["positions"].shape == (100, 3))
    check("sh_coeffs shape", params["sh_coeffs"].shape == (100, 16, 3))

    log.info("\n--- Decomposition ---")
    scene_data = SceneData(
        images=images, K=K, viewmats=viewmats, poses_cam_to_world=poses,
        tracks=tracks, Tr_velo_to_cam=Tr, R_rect=R_rect,
        velodyne_dir=str(root / "velodyne" / seq), image_size=(1242, 375),
    )
    bg_cloud, obj_clouds = decompose_scene(scene_data, 10)
    check("bg points", bg_cloud.positions.shape[0] > 0)
    check("obj tracks", len(obj_clouds) == 2)

    log.info("\n--- Scene model ---")
    bg_params = init_gaussians(
        bg_cloud.positions, bg_cloud.colors, sh_degree=0, device="cpu",
    )
    obj_models = build_object_models(
        scene_data, obj_clouds, sh_degree=0, min_points=50, device="cpu",
    )
    scene = StreetGaussianScene(bg_params, obj_models)
    combined, segments = scene.compose_frame(0)
    N = combined["positions"].shape[0]
    check("compose N > bg", N >= bg_params["positions"].shape[0])
    check(
        "all shapes match",
        all(combined[k].shape[0] == N for k in combined),
    )

    log.info("\n%s", "=" * 40)
    log.info("  %d/%d tests passed", passed, total)
    log.info("%s", "=" * 40)


def main() -> None:
    """Run smoke tests with synthetic KITTI data."""
    log.info("=" * 50)
    log.info("Street Gaussians — Smoke Test")
    log.info("=" * 50)

    tmp = Path(tempfile.mkdtemp(prefix="kitti_test_"))
    try:
        create_synthetic_kitti(tmp, n_frames=10)
        run_tests(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
