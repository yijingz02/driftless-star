# StellaForge MVP Snakemake workflow

configfile: "config.yaml"

RUN_NAME       = config["run_name"]
STAGE3_BACKEND = config["stage3_backend"]
if STAGE3_BACKEND not in ("sfincs_jax", "sfincs_fortran"):
    raise ValueError(
        f"config['stage3_backend'] must be 'sfincs_jax' or 'sfincs_fortran', "
        f"got {STAGE3_BACKEND!r}."
    )

DEVICE = config.get("device", "cpu")
if DEVICE not in ("cpu", "gpu"):
    raise ValueError(
        f"config['device'] must be 'cpu' or 'gpu', got {DEVICE!r}."
    )

GPU_FLAG           = "--gpus all " if DEVICE == "gpu" else ""
STAGE1_IMG         = f"ghcr.io/rkhashmani/stellaforge:stage-1-vmec-{DEVICE}"
STAGE2_IMG         = f"ghcr.io/rkhashmani/stellaforge:stage-2-booz-jax-{DEVICE}"
STAGE3_JAX_IMG     = f"ghcr.io/rkhashmani/stellaforge:stage-3-sfincs-{DEVICE}"
STAGE3_FORTRAN_IMG = "ghcr.io/rkhashmani/stellaforge:stage-3-sfincs-fortran-cpu"  # no -gpu build published
STAGE4_IMG         = f"ghcr.io/rkhashmani/stellaforge:stage-4-spectrax-{DEVICE}"

# --user: make bind-mounted writes host-owned (Linux docker otherwise writes as root).
# -e HOME=/tmp: pixi activation needs a writable HOME after dropping root.
DOCKER_PREFIX = (
    f'docker run --rm --pull=missing {GPU_FLAG}'
    '--user "$(id -u):$(id -g)" '
    '-e HOME=/tmp '
    '-v "$PWD:/work" -w /work'
)

# Terminal artifacts of the MVP forward pass. When Stage 5 (NEOPAX) lands,
# this list collapses to the single (or multiple) final Stage 5 output(s); Stages 2-4 outputs
# become transitive intermediates and drop out of `rule all`.
rule all:
    input:
        f"stages/stage2-boozer/output/boozmn_{RUN_NAME}.nc",
        "stages/stage3-neoclassical/output/sfincsOutput.h5",
        "stages/stage4-turbulence/output/hsx_run.summary.json",
        "stages/stage4-turbulence/output/hsx_run.diagnostics.csv",

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
        namelist = f"stages/stage3-neoclassical/input/input.{RUN_NAME}",
        wout     = f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc",
    output:
        "stages/stage3-neoclassical/output/sfincsOutput.h5",
    run:
        if STAGE3_BACKEND == "sfincs_jax":
            shell(
                f"{DOCKER_PREFIX} {STAGE3_JAX_IMG} "
                "sfincs_jax {input.namelist} "
                "--out stages/stage3-neoclassical/output/sfincsOutput.h5 "
                "--wout-path {input.wout}"
            )
        else:  # sfincs_fortran
            shell(
                f"{DOCKER_PREFIX} {STAGE3_FORTRAN_IMG} "
                'sh -c "mkdir -p stages/stage3-neoclassical/output && '
                "cp {input.namelist} stages/stage3-neoclassical/output/input.namelist && "
                'cd stages/stage3-neoclassical/output && sfincs"'
            )

# eik_cache is geometry derived from wout; delete it before each rerun so
# spectrax-gk regenerates from the current wout rather than reusing stale cache.
rule stage4_spectrax:
    input:
        toml = "stages/stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry.toml",
        wout = f"stages/stage1-equilibrium/output/wout_{RUN_NAME}.nc",
    output:
        summary     = "stages/stage4-turbulence/output/hsx_run.summary.json",
        diagnostics = "stages/stage4-turbulence/output/hsx_run.diagnostics.csv",
        eik_cache   = f"stages/stage4-turbulence/output/wout_{RUN_NAME}.eik.nc",
    shell:
        "rm -f {output.eik_cache} && "
        f"{DOCKER_PREFIX} {STAGE4_IMG} "
        "spectrax-gk run --config {input.toml} "
        "--out stages/stage4-turbulence/output/hsx_run"

rule clean:
    shell:
        """
        rm -rf stages/stage1-equilibrium/output stages/stage2-boozer/output \
               stages/stage3-neoclassical/output stages/stage4-turbulence/output
        """
