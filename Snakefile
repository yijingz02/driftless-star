# StellaForge MVP Snakemake workflow

from pathlib import Path

configfile: "config.yaml"

RUN_NAME = config["run_name"]

DEVICE = config.get("device", "cpu")
if DEVICE not in ("cpu", "gpu"):
    raise ValueError(
        f"config['device'] must be 'cpu' or 'gpu', got {DEVICE!r}."
    )

GPU_FLAG       = "--gpus all " if DEVICE == "gpu" else ""
STAGE1_IMG     = f"ghcr.io/rkhashmani/stellaforge:stage-1-vmec-{DEVICE}"
STAGE2_IMG     = f"ghcr.io/rkhashmani/stellaforge:stage-2-booz-jax-{DEVICE}"
STAGE3_JAX_IMG = f"ghcr.io/rkhashmani/stellaforge:stage-3-sfincs-{DEVICE}"
STAGE4_IMG     = f"ghcr.io/rkhashmani/stellaforge:stage-4-spectrax-{DEVICE}"
STAGE5_IMG     = f"ghcr.io/rkhashmani/stellaforge:stage-5-neopax-{DEVICE}"

# --user: make bind-mounted writes host-owned (Linux docker otherwise writes as root).
# -e HOME=/tmp: pixi activation needs a writable HOME after dropping root.
DOCKER_PREFIX = (
    f'docker run --rm --pull=missing {GPU_FLAG}'
    '--user "$(id -u):$(id -g)" '
    '-e HOME=/tmp '
    '-v "$PWD:/work" -w /work'
)

# Stage 3 sfincs_jax radial-scan config + derived paths.
STAGE3_CFG  = config["stage3"]["sfincs_jax"]
S3_TEMPLATE = STAGE3_CFG["sfincs_template"] or f"stages/stage3-neoclassical/input/input.{RUN_NAME}"
S3_WOUT     = STAGE3_CFG["wout_path"]       or f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc"
S3_OUTDIR   = f"stages/stage3-neoclassical/output/{STAGE3_CFG['output_subdir']}"
S3_OUTPUT   = f"{S3_OUTDIR}/sfincs_jax_flux_profiles.h5"


_STAGE3_OPTIONAL_FLAGS = [
    ("profiles_source",   "--profiles-source"),
    ("neopax_result",     "--neopax-result"),
    ("ntheta",            "--ntheta"),
    ("nzeta",             "--nzeta"),
    ("nxi",               "--nxi"),
    ("nx",                "--nx"),
    ("solver_tolerance",  "--solver-tolerance"),
    ("max_parallel",      "--max-parallel"),
]

_STAGE3_BOOL_FLAGS = [
    ("plot",            "--plot",            "--no-plot"),
    ("verbose_workers", "--verbose-workers", "--no-verbose-workers"),
]


def _stage3_radial_scan_cmd() -> str:
    """Compose the Stage 3 sfincs_jax radial-scan shell command from config."""
    s = STAGE3_CFG
    parts = [
        f"{DOCKER_PREFIX} {STAGE3_JAX_IMG}",
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py",
        "--neopax-config {input.neopax_toml}",
        "--sfincs-template {input.template}",
        "--wout-path {input.wout}",
        f"--output-dir {S3_OUTDIR}",
        f"--backend {DEVICE}",
    ]
    for key, flag in _STAGE3_OPTIONAL_FLAGS:
        v = s.get(key)
        if v is not None:
            parts.append(f"{flag} {v}")
    if DEVICE == "gpu" and s.get("gpu_ids") is not None:
        parts.append(f"--gpu-ids {s['gpu_ids']}")
    for key, on, off in _STAGE3_BOOL_FLAGS:
        v = s.get(key)
        if v is True:
            parts.append(on)
        elif v is False:
            parts.append(off)
    return " ".join(parts)

# Stage 4 spectrax-gk radial-scan config + derived paths.
STAGE4_CFG  = config["stage4"]["spectrax_gk"]
S4_TEMPLATE = STAGE4_CFG["spectrax_template"] or "stages/stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry_quickrun.toml"
S4_WOUT     = STAGE4_CFG["vmec_file_override"]   or f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc"
S4_BOOZER   = STAGE4_CFG["boozer_file_override"] or f"stages/stage2-boozer/output/boozmn_{RUN_NAME}.nc"
S4_OUTDIR   = f"stages/stage4-turbulence/output/{STAGE4_CFG['output_subdir']}"
S4_OUTPUT   = f"{S4_OUTDIR}/neopax_fluxes.h5"


_STAGE4_OPTIONAL_FLAGS = [
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

# Tri-state booleans: null = use script default, true/false = explicit override.
_STAGE4_BOOL_FLAGS = [
    ("plot",                     "--plot",                     "--no-plot"),
    ("plot_run_heat_traces",     "--plot-run-heat-traces",     "--no-plot-run-heat-traces"),
    ("verbose_workers",          "--verbose-workers",          "--no-verbose-workers"),
    ("collect_even_if_failures", "--collect-even-if-failures", "--no-collect-even-if-failures"),
]


def _stage4_radial_scan_cmd() -> str:
    """Compose the Stage 4 spectrax-gk radial-scan shell command from config."""
    s = STAGE4_CFG
    parts = [
        f"{DOCKER_PREFIX} {STAGE4_IMG}",
        "python stages/stage4-turbulence/spectrax_gk_radial_scan.py",
        "--neopax-config {input.neopax_toml}",
        "--spectrax-template {input.template}",
        "--vmec-file-override {input.wout}",
        "--boozer-file-override {input.boozer}",
        f"--output-dir {S4_OUTDIR}",
        f"--backend {DEVICE}",
    ]
    for key, flag in _STAGE4_OPTIONAL_FLAGS:
        v = s.get(key)
        if v is not None:
            parts.append(f"{flag} {v}")
    if DEVICE == "gpu" and s.get("gpu_ids") is not None:
        parts.append(f"--gpu-ids {s['gpu_ids']}")
    for key, on, off in _STAGE4_BOOL_FLAGS:
        v = s.get(key)
        if v is True:
            parts.append(on)
        elif v is False:
            parts.append(off)
    return " ".join(parts)


# Stage 5 NEOPAX transport solver.
STAGE5_CFG = config["stage5"]["neopax"]
S5_TOML    = Path(STAGE5_CFG["toml"])
S5_TOML_DIR  = str(S5_TOML.parent)
S5_TOML_NAME = S5_TOML.name
S5_OUTPUT  = "stages/stage5-transport/output/transport_solution.h5"


# Terminal artifact of the MVP forward pass.
rule all:
    input:
        S5_OUTPUT,

rule stage1_vmec:
    input:  f"stages/stage1-equilibrium/input/input.{RUN_NAME}"
    output: f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc"
    shell:
        f"{DOCKER_PREFIX} {STAGE1_IMG} "
        "vmec_jax {input} --outdir stages/stage1-equilibrium/output"

rule stage2_boozer:
    input:  f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc"
    output: f"stages/stage2-boozer/output/boozmn_{RUN_NAME}.nc"
    shell:
        f"{DOCKER_PREFIX} {STAGE2_IMG} "
        "python stages/stage2-boozer/run_boozer.py --wout {input} --output {output}"

rule stage3_sfincs:
    input:
        template    = S3_TEMPLATE,
        wout        = S3_WOUT,
        neopax_toml = STAGE3_CFG["neopax_config"],
    output:
        S3_OUTPUT,
    shell:
        _stage3_radial_scan_cmd()

rule stage4_spectrax:
    input:
        template    = S4_TEMPLATE,
        wout        = S4_WOUT,
        boozer      = S4_BOOZER,
        neopax_toml = STAGE4_CFG["neopax_config"],
    output:
        S4_OUTPUT,
    shell:
        _stage4_radial_scan_cmd()

rule stage5_neopax:
    input:
        toml    = str(S5_TOML),
        wout    = f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc",
        boozer  = f"stages/stage2-boozer/output/boozmn_{RUN_NAME}.nc",
        neo_h5  = S3_OUTPUT,
        turb_h5 = S4_OUTPUT,
    output:
        S5_OUTPUT,
    shell:
        f"{DOCKER_PREFIX} {STAGE5_IMG} "
        f'sh -c "cd {S5_TOML_DIR} && neopax {S5_TOML_NAME}"'

rule clean:
    shell:
        """
        rm -rf stages/stage1-equilibrium/output stages/stage2-boozer/output \
               stages/stage3-neoclassical/output stages/stage4-turbulence/output \
               stages/stage5-transport/output
        """
