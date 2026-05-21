# evrptw_gen/__init__.py
"""
EVRPTW Instance Generator Package
---------------------------------
This package provides:
- InstanceGenerator: main class for generating EVRPTW benchmark data
- EVRPTWDataset: torch-compatible dataset wrapper
- Config: configuration loader
- Submodules: policies, utils, io, configs

Example
-------
>>> from evrptw_gen import InstanceGenerator, EVRPTWDataset, Config
>>> cfg = Config("configs/config.yaml")
>>> gen = InstanceGenerator(cfg, save_path="./Instances", num_instances=10)
>>> instances = gen.generate()
"""

from .generator import InstanceGenerator
from .configs.load_config import Config
from . import policies, utils, configs

__all__ = [
    "InstanceGenerator",
    "EVRPTWDataset",
    "Config",
    "policies",
    "utils",
    "configs",
]
