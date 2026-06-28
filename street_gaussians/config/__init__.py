"""Configuration management — typed dataclasses loaded from YAML."""

from street_gaussians.config.schema import Config, DataConfig, OutputConfig, TrainingConfig
from street_gaussians.config.loader import load_config

__all__ = ["Config", "DataConfig", "OutputConfig", "TrainingConfig", "load_config"]
