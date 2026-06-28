"""YAML config loading with merge support."""

from pathlib import Path

import yaml

from street_gaussians.config.schema import (
    Config,
    DataConfig,
    OutputConfig,
    TrainingConfig,
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(raw: dict) -> Config:
    """Convert a raw dict to a typed Config, ignoring unknown keys."""
    data_cfg = DataConfig(**raw.get("data", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))
    output_cfg = OutputConfig(**raw.get("output", {}))
    return Config(data=data_cfg, training=training_cfg, output=output_cfg)


def load_config(path: str | Path, overrides: str | Path | None = None) -> Config:
    """Load a YAML config, optionally merging an override file on top.

    Args:
        path: Path to the base YAML config.
        overrides: Optional path to an override YAML (merged on top of base).

    Returns:
        Typed Config dataclass.
    """
    with open(path) as f:
        base = yaml.safe_load(f) or {}

    if overrides is not None:
        with open(overrides) as f:
            over = yaml.safe_load(f) or {}
        base = _deep_merge(base, over)

    return _dict_to_config(base)
