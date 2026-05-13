# StellaForge MVP Snakemake workflow

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


def _stage3_radial_scan_cmd() -> str:
    """Compose the Stage 3 sfincs_jax radial-scan shell command from config."""
    s = STAGE3_CFG
    parts = [
        f"{DOCKER_PREFIX} {STAGE3_JAX_IMG}",
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py",
        "--neopax-config {input.neopax_toml}",
        "--sfincs-template {input.template}",
        "--wout-path {input.wout}",
        f"--profiles-source {s['profiles_source']}",
        f"--analytical-n-radii {s['analytical_n_radii']}",
        f"--time-index {s['time_index']}",
        f"--output-dir {S3_OUTDIR}",
        f"--backend {DEVICE}",
        f"--gpu-ids {s['gpu_ids']}",
        f"--max-parallel {s['max_parallel']}",
        f"--cores-per-run {s['cores_per_run']}",
        f"--worker-sharding {s['worker_sharding']}",
        f"--ntheta {s['ntheta']}",
        f"--nzeta {s['nzeta']}",
        f"--nxi {s['nxi']}",
        f"--nl {s['nl']}",
        f"--nx {s['nx']}",
        f"--solver-tolerance {s['solver_tolerance']}",
        f"--benchmark-repeats {s['benchmark_repeats']}",
        f"--benchmark-warmup {s['benchmark_warmup']}",
    ]
    if s.get("neopax_result"):
        parts.append(f"--neopax-result {s['neopax_result']}")
    if s.get("rho_indices"):
        parts.append(f"--rho-indices {s['rho_indices']}")
    if s.get("rho_min") is not None:
        parts.append(f"--rho-min {s['rho_min']}")
    if s.get("rho_max") is not None:
        parts.append(f"--rho-max {s['rho_max']}")
    if s.get("num_radii") is not None:
        parts.append(f"--num-radii {s['num_radii']}")
    if s.get("dense_fp_max") is not None:
        parts.append(f"--dense-fp-max {s['dense_fp_max']}")
    if s.get("include_phi1") is True:
        parts.append("--include-phi1")
    elif s.get("include_phi1") is False:
        parts.append("--no-include-phi1")
    parts.append("--plot" if s.get("plot") else "--no-plot")
    parts.append("--verbose-workers" if s.get("verbose_workers") else "--no-verbose-workers")
    return " ".join(parts)

# Terminal artifacts of the MVP forward pass. When Stage 5 (NEOPAX) lands,
# this list collapses to the single (or multiple) final Stage 5 output(s); Stages 2-4 outputs
# become transitive intermediates and drop out of `rule all`.
rule all:
    input:
        f"stages/stage2-boozer/output/boozmn_{RUN_NAME}.nc",
        S3_OUTPUT,
        "stages/stage4-turbulence/output/hsx_run_quickrun.summary.json",
        "stages/stage4-turbulence/output/hsx_run_quickrun.diagnostics.csv",

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

# eik_cache is geometry derived from wout; delete it before each rerun so
# spectrax-gk regenerates from the current wout rather than reusing stale cache.
rule stage4_spectrax:
    input:
        toml = "stages/stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry_quickrun.toml",
        wout = f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc",
    output:
        summary     = "stages/stage4-turbulence/output/hsx_run_quickrun.summary.json",
        diagnostics = "stages/stage4-turbulence/output/hsx_run_quickrun.diagnostics.csv",
        eik_cache   = f"stages/stage4-turbulence/output/wout_{RUN_NAME}.eik.nc",
    shell:
        "rm -f {output.eik_cache} && "
        f"{DOCKER_PREFIX} {STAGE4_IMG} "
        "spectrax-gk run --config {input.toml} "
        "--out stages/stage4-turbulence/output/hsx_run_quickrun"

rule clean:
    shell:
        """
        rm -rf stages/stage1-equilibrium/output stages/stage2-boozer/output \
               stages/stage3-neoclassical/output stages/stage4-turbulence/output
        """
