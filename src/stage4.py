"""Stage 4 (turbulence) shell-command composition for the Snakemake workflow.

The Stage 4 SPECTRAX-GK radial-scan script accepts many optional flags; this
module turns the user-facing ``config.yaml`` ``stage4.spectrax_gk`` block into
a single shell command string that the Snakefile's ``rule stage4_spectrax``
runs.
"""

from __future__ import annotations

# (config_key, cli_flag) — emitted as `<flag> <value>` when the config value is not None.
_OPTIONAL_FLAGS: list[tuple[str, str]] = [
    ("profiles_source",    "--profiles-source"),
    ("neopax_result",      "--neopax-result"),
    ("nx",                 "--nx"),
    ("ny",                 "--ny"),
    ("ntheta",             "--ntheta"),
    ("t_max",              "--t-final"),
    ("average_window",     "--average-window"),
    ("sample_stride",      "--sample-stride"),
    ("diagnostics_stride", "--diagnostics-stride"),
    ("max_parallel",       "--max-parallel"),
]

# (config_key, on_flag, off_flag) — tri-state: None = script default, True/False = explicit.
_BOOL_FLAGS: list[tuple[str, str, str]] = [
    ("plot",                     "--plot",                     "--no-plot"),
    ("plot_run_heat_traces",     "--plot-run-heat-traces",     "--no-plot-run-heat-traces"),
    ("verbose_workers",          "--verbose-workers",          "--no-verbose-workers"),
    ("collect_even_if_failures", "--collect-even-if-failures", "--no-collect-even-if-failures"),
]


def radial_scan_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
    device: str,
) -> str:
    """Compose the Stage 4 SPECTRAX-GK radial-scan shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-spectrax-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage4.spectrax_gk`` block.
    output_dir : str
        Stage 4 output directory (already ``{run_name}``-substituted).
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
        "python stages/stage4-turbulence/spectrax_gk_radial_scan.py",
        "--common-config {input.common_config}",
        "--spectrax-template {input.config_file}",
        "--vmec-file-override {input.wout}",
        "--boozer-file-override {input.boozer}",
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
