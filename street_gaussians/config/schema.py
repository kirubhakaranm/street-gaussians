"""Typed configuration dataclasses."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data source and preprocessing settings."""

    source: str = "kitti"
    root: str = "data/kitti/tracking/training"
    sequence: str = "0001"
    test_every: int = 8
    max_bg_points: int = 200_000
    max_object_points: int = 10_000
    min_object_points: int = 50


@dataclass
class TrainingConfig:
    """Training hyperparameters and scheduling."""

    iterations: int = 50_000
    sh_degree: int = 3
    lr_position_init: float = 1.6e-4
    lr_position_final: float = 1.6e-6
    lr_scale: float = 5e-3
    lr_quaternion: float = 1e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3
    lambda_ssim: float = 0.2
    densify_interval: int = 100
    grad_threshold: float = 5e-7
    opacity_prune_threshold: float = 0.005
    opacity_reset_interval: int = 3000
    checkpoint_interval: int = 10_000


@dataclass
class OutputConfig:
    """Output directory and snapshot frequency."""

    dir: str = "output/"
    progression_interval: int = 500


@dataclass
class Config:
    """Top-level pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
