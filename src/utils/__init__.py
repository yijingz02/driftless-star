"""Shared utilities for the driftless-star Snakemake workflow."""

from .config_edit import set_assignment
from .paths import RESOLVED_COMMON_CONFIG, resolve_pipeline_paths

__all__ = ["RESOLVED_COMMON_CONFIG", "resolve_pipeline_paths", "set_assignment"]
