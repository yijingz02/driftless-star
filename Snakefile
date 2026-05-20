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

# Stage 4 spectrax-gk radial-scan config + derived paths.
STAGE4_CFG  = config["stage4"]["spectrax_gk"]
S4_TEMPLATE = STAGE4_CFG["spectrax_template"] or "stages/stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry_quickrun.toml"
S4_WOUT     = STAGE4_CFG["vmec_file_override"]   or f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc"
S4_BOOZER   = STAGE4_CFG["boozer_file_override"] or f"stages/stage2-boozer/output/boozmn_{RUN_NAME}.nc"
S4_OUTDIR   = f"stages/stage4-turbulence/output/{STAGE4_CFG['output_subdir']}"
S4_OUTPUT   = f"{S4_OUTDIR}/neopax_fluxes.h5"


# Optional value-bearing args: only passed when the config value is non-null,
# so the script's own defaults govern anything the user leaves alone.
_STAGE4_OPTIONAL_FLAGS = {
    "profiles_source":         "--profiles-source",
    "neopax_result":           "--neopax-result",
    "analytical_n_radii":      "--analytical-n-radii",
    "time_index":              "--time-index",
    "electron_model":          "--electron-model",
    "reference_ion":           "--reference-ion",
    "rho_indices":             "--rho-indices",
    "rho_min":                 "--rho-min",
    "rho_max":                 "--rho-max",
    "num_radii":               "--num-radii",
    "density_floor":           "--density-floor",
    "temperature_floor":       "--temperature-floor",
    "gradient_coordinate":     "--gradient-coordinate",
    "gradient_scale":          "--gradient-scale",
    "tprim_scale":             "--tprim-scale",
    "fprim_scale":             "--fprim-scale",
    "tau_e_override":          "--tau-e-override",
    "nu_ion":                  "--nu-ion",
    "nu_electron":             "--nu-electron",
    "rho_star_physical":       "--rho-star-physical",
    "nx":                      "--nx",
    "ny":                      "--ny",
    "nz":                      "--nz",
    "lx":                      "--lx",
    "ly":                      "--ly",
    "boundary":                "--boundary",
    "y0":                      "--y0",
    "ntheta":                  "--ntheta",
    "nperiod":                 "--nperiod",
    "t_max":                   "--t-max",
    "dt":                      "--dt",
    "method":                  "--method",
    "sample_stride":           "--sample-stride",
    "diagnostics_stride":      "--diagnostics-stride",
    "chunk_steps":             "--chunk-steps",
    "cfl":                     "--cfl",
    "state_sharding":          "--state-sharding",
    "ky":                      "--ky",
    "nl":                      "--nl",
    "nm":                      "--nm",
    "init_field":              "--init-field",
    "init_amp":                "--init-amp",
    "alpha":                   "--alpha",
    "npol":                    "--npol",
    "beta":                    "--beta",
    "nu_hermite":              "--nu-hermite",
    "nu_laguerre":             "--nu-laguerre",
    "nu_hyper":                "--nu-hyper",
    "p_hyper":                 "--p-hyper",
    "hypercollisions_const":   "--hypercollisions-const",
    "hypercollisions_kz":      "--hypercollisions-kz",
    "d_hyper":                 "--d-hyper",
    "damp_ends_amp":           "--damp-ends-amp",
    "damp_ends_widthfrac":     "--damp-ends-widthfrac",
    "hyperdiffusion":          "--hyperdiffusion",
    "normalization_contract":  "--normalization-contract",
    "diagnostic_norm":         "--diagnostic-norm",
    "average_window":          "--average-window",
    "gpu_ids":                 "--gpu-ids",
    "max_parallel":            "--max-parallel",
    "threads_per_run":         "--threads-per-run",
    "poll_interval":           "--poll-interval",
}

# Tri-state booleans: null = use script default, true/false = explicit override.
_STAGE4_BOOL_FLAGS = [
    ("use_diffrax",              "--use-diffrax",              "--no-use-diffrax"),
    ("fixed_dt",                 "--fixed-dt",                 "--no-fixed-dt"),
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
    for key, flag in _STAGE4_OPTIONAL_FLAGS.items():
        v = s.get(key)
        if v is not None:
            parts.append(f"{flag} {v}")
    for key, on, off in _STAGE4_BOOL_FLAGS:
        v = s.get(key)
        if v is True:
            parts.append(on)
        elif v is False:
            parts.append(off)
    return " ".join(parts)


# Stage 5 NEOPAX transport solver.
STAGE5_CFG = config["stage5"]["neopax"]
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
        toml    = STAGE5_CFG["toml"],
        wout    = f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc",
        boozer  = f"stages/stage2-boozer/output/boozmn_{RUN_NAME}.nc",
        neo_h5  = S3_OUTPUT,
        turb_h5 = S4_OUTPUT,
    output:
        S5_OUTPUT,
    shell:
        f"{DOCKER_PREFIX} {STAGE5_IMG} "
        'sh -c "cd stages/stage5-transport/input && neopax Solve_Transport_equations_noHe_radau_HSX_quickrun.toml"'

rule clean:
    shell:
        """
        rm -rf stages/stage1-equilibrium/output stages/stage2-boozer/output \
               stages/stage3-neoclassical/output stages/stage4-turbulence/output \
               stages/stage5-transport/output
        """
