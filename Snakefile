# StellaForge MVP Snakemake workflow

configfile: "config.yaml"

RUN_NAME = config["run_name"]
# Substitute {run_name} into each directory value; literal paths pass through unchanged.
DIRS = {k: v.format(run_name=RUN_NAME) for k, v in config["directories"].items()}
FILES = config["filenames"]


# Substitute {run_name} into the filename for `key`; literal names pass through unchanged.
def filename(key):
    return FILES[key].format(run_name=RUN_NAME)

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

S1_INPUT  = f"{DIRS['stage1_input']}/{filename('s1_input')}"
S1_OUTPUT = f"{DIRS['stage1_output']}/{filename('s1_output')}"
S2_OUTPUT = f"{DIRS['stage2_output']}/{filename('s2_output')}"

# Stage 3 sfincs_jax radial-scan config + derived paths.
STAGE3_CFG  = config["stage3"]["sfincs_jax"]
S3_CONFIG   = f"{DIRS['stage3_input']}/{filename('s3_config')}"
S3_OUTPUT   = f"{DIRS['stage3_output']}/{filename('s3_output')}"


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
    parts = [
        f"{DOCKER_PREFIX} {STAGE3_JAX_IMG}",
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py",
        "--neopax-config {input.neopax_config}",
        "--sfincs-template {input.config_file}",
        "--wout-path {input.wout}",
        f"--output-dir {DIRS['stage3_output']}",
        f"--backend {DEVICE}",
    ]
    for key, flag in _STAGE3_OPTIONAL_FLAGS:
        v = STAGE3_CFG.get(key)
        if v is not None:
            parts.append(f"{flag} {v}")
    if DEVICE == "gpu" and STAGE3_CFG.get("gpu_ids") is not None:
        parts.append(f"--gpu-ids {STAGE3_CFG['gpu_ids']}")
    for key, on, off in _STAGE3_BOOL_FLAGS:
        v = STAGE3_CFG.get(key)
        if v is True:
            parts.append(on)
        elif v is False:
            parts.append(off)
    return " ".join(parts)

# Stage 4 spectrax-gk radial-scan config + derived paths.
STAGE4_CFG  = config["stage4"]["spectrax_gk"]
S4_CONFIG   = f"{DIRS['stage4_input']}/{filename('s4_config')}"
S4_OUTPUT   = f"{DIRS['stage4_output']}/{filename('s4_output')}"


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
    parts = [
        f"{DOCKER_PREFIX} {STAGE4_IMG}",
        "python stages/stage4-turbulence/spectrax_gk_radial_scan.py",
        "--neopax-config {input.neopax_config}",
        "--spectrax-template {input.config_file}",
        "--vmec-file-override {input.wout}",
        "--boozer-file-override {input.boozer}",
        f"--output-dir {DIRS['stage4_output']}",
        f"--backend {DEVICE}",
    ]
    for key, flag in _STAGE4_OPTIONAL_FLAGS:
        v = STAGE4_CFG.get(key)
        if v is not None:
            parts.append(f"{flag} {v}")
    if DEVICE == "gpu" and STAGE4_CFG.get("gpu_ids") is not None:
        parts.append(f"--gpu-ids {STAGE4_CFG['gpu_ids']}")
    for key, on, off in _STAGE4_BOOL_FLAGS:
        v = STAGE4_CFG.get(key)
        if v is True:
            parts.append(on)
        elif v is False:
            parts.append(off)
    return " ".join(parts)


# Stage 5 NEOPAX transport solver.
S5_CONFIG  = f"{DIRS['stage5_input']}/{filename('s5_config')}"
S5_OUTPUT  = f"{DIRS['stage5_output']}/{filename('s5_output')}"


# Terminal artifact of the MVP forward pass.
rule all:
    input:
        S5_OUTPUT,

rule stage1_vmec:
    input:  S1_INPUT
    output: S1_OUTPUT
    shell:
        f"{DOCKER_PREFIX} {STAGE1_IMG} "
        f"vmec_jax {{input}} --outdir {DIRS['stage1_output']}"

rule stage2_boozer:
    input:  S1_OUTPUT
    output: S2_OUTPUT
    shell:
        f"{DOCKER_PREFIX} {STAGE2_IMG} "
        "python stages/stage2-boozer/run_boozer.py --wout {input} --output {output}"

rule stage3_sfincs:
    input:
        config_file = S3_CONFIG,
        wout        = S1_OUTPUT,
        neopax_config = S5_CONFIG,
    output:
        S3_OUTPUT,
    shell:
        _stage3_radial_scan_cmd()

rule stage4_spectrax:
    input:
        config_file = S4_CONFIG,
        wout        = S1_OUTPUT,
        boozer      = S2_OUTPUT,
        neopax_config = S5_CONFIG,
    output:
        S4_OUTPUT,
    shell:
        _stage4_radial_scan_cmd()

rule stage5_neopax:
    input:
        config_file = S5_CONFIG,
        wout    = S1_OUTPUT,
        boozer  = S2_OUTPUT,
        neo_h5  = S3_OUTPUT,
        turb_h5 = S4_OUTPUT,
    output:
        S5_OUTPUT,
    shell:
        f"{DOCKER_PREFIX} {STAGE5_IMG} "
        f"sh -c \"cd {DIRS['stage5_input']} && neopax {filename('s5_config')}\""

rule clean:
    shell:
        f"""
        rm -rf {DIRS['stage1_output']} {DIRS['stage2_output']} \
               {DIRS['stage3_output']} {DIRS['stage4_output']} \
               {DIRS['stage5_output']}
        """
