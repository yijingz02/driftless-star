"""Shared utilities for the driftless-star Snakemake workflow."""

from .config_edit import apply_assignments
from .paths import RESOLVED_COMMON_CONFIG, resolve_pipeline_paths

__all__ = ["RESOLVED_COMMON_CONFIG", "apply_assignments", "resolve_pipeline_paths"]
