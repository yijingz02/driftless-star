"""Stage 3 (neoclassical) shell-command composition for the Snakemake workflow.

The Stage 3 ``sfincs_jax`` radial-scan script accepts many optional flags; this
module turns the user-facing ``config.yaml`` ``stage3.sfincs_jax`` block into a
single shell command string that the Snakefile's ``rule stage3_sfincs`` runs.
"""

from __future__ import annotations

# (config_key, cli_flag) — emitted as `<flag> <value>` when the config value is not None.
_OPTIONAL_FLAGS: list[tuple[str, str]] = [
    ("profiles_source",   "--profiles-source"),
    ("neopax_result",     "--neopax-result"),
    ("ntheta",            "--ntheta"),
    ("nzeta",             "--nzeta"),
    ("nxi",               "--nxi"),
    ("nx",                "--nx"),
    ("solver_tolerance",  "--solver-tolerance"),
    ("max_parallel",      "--max-parallel"),
]

# (config_key, on_flag, off_flag) — tri-state: None = script default, True/False = explicit.
_BOOL_FLAGS: list[tuple[str, str, str]] = [
    ("plot",            "--plot",            "--no-plot"),
    ("verbose_workers", "--verbose-workers", "--no-verbose-workers"),
]


def radial_scan_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
    device: str,
) -> str:
    """Compose the Stage 3 ``sfincs_jax`` radial-scan shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 3 (e.g. ``ghcr.io/.../stage-3-sfincs-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage3.sfincs_jax`` block.
    output_dir : str
        Stage 3 output directory (already ``{run_name}``-substituted).
    device : str
        ``"cpu"`` or ``"gpu"``; controls JAX backend and GPU pinning.

    Returns
    -------
    str
        A single-line shell command suitable for a Snakemake ``shell:`` block.
        Snakemake placeholders ``{input.*}`` remain literal so they are
        substituted at rule-execution time.
    """
    parts = [
        f"{docker_prefix} {image}",
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py",
        "--neopax-config {input.neopax_config}",
        "--sfincs-template {input.config_file}",
        "--wout-path {input.wout}",
        f"--output-dir {output_dir}",
        f"--backend {device}",
    ]
    for key, flag in _OPTIONAL_FLAGS:
        v = stage_cfg.get(key)
        if v is not None:
            parts.append(f"{flag} {v}")
    if device == "gpu" and stage_cfg.get("gpu_ids") is not None:
        parts.append(f"--gpu-ids {stage_cfg['gpu_ids']}")
    for key, on, off in _BOOL_FLAGS:
        v = stage_cfg.get(key)
        if v is True:
            parts.append(on)
        elif v is False:
            parts.append(off)
    return " ".join(parts)
