# MVP Pipeline: I/O Reference

```
 INDATA                   wout_*.nc                boozmn_*.nc
   |                          |                         |
   v                          v                         v
┌────────┐  NetCDF  ┌──────────────┐  NetCDF  ┌──────────────────────┐
│Stage 1 │ -------> │   Stage 2    │ -------> │      Stage 3         │
│vmec_jax│ wout_*.nc│booz_xform_jax│boozmn_*.nc│        SFINCS        │
└────────┘          └──────────────┘          └──────────┬───────────┘
                          |                              |
                          |                              v
                          |                      sfincsOutput.h5
                          |                          (HDF5)
                          |                              |
                          |    ┌────────────┐            |
                          +--->│  Stage 4   │            |
                     wout_*.nc │ SPECTRAX-GK│            |
                               └─────┬──────┘            |
                                     |                   |
                                flux (CSV)               |
                                     |                   |
                                     v                   v
                     ┌────────────────────────────────────────┐
                     │                 Stage 5                │
                     │          Transport / Profiles          │
                     │   NEOPAX (wout + boozmn + turb flux)   │
                     └─────────────────┬──────────────────────┘
                                       |
                                       v
                                  profiles.h5
                             n(r), T(r), E_r(r), P_fus, Q
```

## Stage Test Data

Tracked reduced-accuracy quickrun inputs and the run scripts that consume them under `stages/`:

```
stages/
├── stage1-equilibrium/     input/  (input.HSX_vacuum_ns201_quickrun)
├── stage2-boozer/          run_boozer.py
├── stage3-neoclassical/    input/  (input.HSX_vacuum_ns201_quickrun)
├── stage4-turbulence/      input/  (runtime_hsx_nonlinear_vmec_geometry_quickrun.toml)
├── stage5-transport/       run_NEOPAX.py
└── stage5-post-processing/ fit_vmec_pressure_from_transport_h5.py, stage5_post_processing.py
```

Each stage's `input/` directory ships with a `_quickrun` smoke-test variant so a fresh clone is immediately runnable. Each stage's `output/` directory is gitignored: Snakemake regenerates it in place via `pixi run -e pipeline snakemake --cores 4`, and cross-stage configs read upstream outputs from there. Run stages in forward-chain order (`stage-1-vmec` first) when invoking individual pixi tasks; Snakemake handles the dependency order automatically.

---

**Note:** All paths are relative to the repository root.

**Note:** The per-stage Pixi environments live in `stages/pixi.toml` (a separate workspace from the root `pipeline` env). The stage-level commands below include `--manifest-path stages/pixi.toml` so they run from the repo root unchanged; alternatively, `cd stages` first and drop the flag.

## Stage 1 -- Equilibrium

**Code:** vmec_jax

| Direction                     | Format                                    | Location                                                                        |
| ----------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| **In**                        | Fortran-style Text                        | `stages/stage1-equilibrium/input/input.HSX_vacuum_ns201_quickrun`              |
| **Out**                       | NetCDF `wout_*.nc` (similar to hdf5 file) | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`  |
| **Additional Out** (optional) | Text (terminal output)                    | `stages/stage1-equilibrium/output/optional_terminal_output.vmec` |

> [!NOTE]
> `HSX_vacuum_ns201_quickrun` is an example name. This can be changed. As can the entirety of the name `optional_terminal_output.vmec`.

### How to Install

From the repo root

```
pixi install --manifest-path stages/pixi.toml --environment stage-1-vmec
```

### How to Run

```
pixi run --manifest-path stages/pixi.toml stage-1-vmec
```

---

## Stage 2 -- Boozer Transform

**Code:** booz_xform_jax

| Direction | Format               | Location                                                                          |
| --------- | -------------------- | --------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc`   | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`    |
| **Out**   | NetCDF `boozmn_*.nc` | `stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc` |

> [!NOTE]
> Stage 2's JAX driver takes explicit `--wout` and `--output` paths. Populate `stage1-equilibrium/output/` by running `pixi run --manifest-path stages/pixi.toml stage-1-vmec` first.

### How to Install

```
pixi install --manifest-path stages/pixi.toml --environment stage-2-booz-jax
```

### How to Run

```
pixi run --manifest-path stages/pixi.toml -e stage-2-booz-jax stage-2-booz
```

which is morally similar to

```python
import booz_xform_jax as bx
b=bx.Booz_xform()
b.read_wout("stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc")
b.run()
b.write_boozmn("stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc")
```

or directly from the command line:

```bash
python stages/stage2-boozer/run_boozer.py \
  --wout stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc \
  --output stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc
```

---

## Stage 3 -- Neoclassical

**Code:** sfincs_jax

| Direction | Format                       | Location                                                                       |
| --------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`           | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc` |
| **In**    | Fortran-style Text `input.*` | `stages/stage3-neoclassical/input/input.HSX_vacuum_ns201_quickrun`          |
| **Out**   | HDF5 `sfincsOutput.h5`       | `stages/stage3-neoclassical/output/sfincsOutput_quickrun.h5`           |

#### How to Install

```
pixi install --manifest-path stages/pixi.toml --environment stage-3-sfincs
```

#### How to Run

```
pixi run --manifest-path stages/pixi.toml stage-3-sfincs
```

> [!NOTE]
> The pixi `stage-3-sfincs` task and the Snakemake `stage3_sfincs` rule both pass the wout path to `sfincs_jax` via `--wout-path`, overriding the namelist `equilibriumFile` field. Populate `stage1-equilibrium/output/` by running `pixi run --manifest-path stages/pixi.toml stage-1-vmec` first. The `sfincs_fortran` backend has no CLI override and still reads `equilibriumFile` from the namelist.


**Code:** SFINCS (Fortran)

| Direction | Format                       | Location                                                                       |
| --------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`           | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc` |
| **In**    | Fortran-style Text `input.*` | `stages/stage3-neoclassical/input/input.HSX_vacuum_ns201_quickrun`          |
| **Out**   | HDF5 `sfincsOutput.h5`       | `stages/stage3-neoclassical/output/sfincsOutput_quickrun.h5`           |

#### How to Install

```
pixi install --manifest-path stages/pixi.toml --environment stage-3-sfincs-fortran
```

#### How to Run

```
pixi run --manifest-path stages/pixi.toml stage-3-sfincs-fortran
```

> [!NOTE]
> `SFINCS` (Fortran) is an alternative to `sfincs_jax`. It reads the same namelist and writes to the same `sfincsOutput.h5` path, so running both against one output directory will overwrite the prior result.

> [!NOTE]
> The task copies `stage3-neoclassical/input/input.HSX_vacuum_ns201_quickrun` to `stage3-neoclassical/output/input.namelist` before invoking the binary, because SFINCS (Fortran) reads `input.namelist` from its working directory.


---

## Stage 4 -- Turbulence

**Code:** SPECTRAX-GK

| Direction | Format             | Location                                                                           |
| --------- | ------------------ | ---------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc` | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`     |
| **In**    | TOML config        | `stages/stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry_quickrun.toml` |
| **Out**   | Summary `JSON`     | `stages/stage4-turbulence/output/hsx_run_quickrun.summary.json`           |
| **Out**   | `csv`              | `stages/stage4-turbulence/output/hsx_run_quickrun.diagnostics.csv`        |

> [!NOTE]
> The TOML's `vmec_file` points into `stage1-equilibrium/output/`. Populate this directory by running `pixi run --manifest-path stages/pixi.toml stage-1-vmec` first.

### How to Install

```
pixi install --manifest-path stages/pixi.toml --environment stage-4-spectrax
```

### How to Run

```
pixi run --manifest-path stages/pixi.toml stage-4-spectrax
```

which executes something morally equivalent to

```
spectrax-gk run --config runtime_hsx_nonlinear_vmec_geometry.toml --out output/hsx_run
```

---

## Stage 5 -- Transport Solver

**Code:** NEOPAX

| Direction | Format               | Location                                                                       |
| --------- | -------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`   | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`           |
| **In**    | NetCDF `boozmn_*.nc` | `stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc`              |
| **Out**   | HDF5 `profiles_*.h5` | `stages/stage5-transport/output/NEOPAX_output_quickrun.h5`                     |
> [!NOTE]
> The inputs come from Stage 1 and Stage 2.

### How to Install

```
pixi install --manifest-path stages/pixi.toml --environment stage-5-neopax
```

### How to Run

Stage 5 is orchestrated by Snakemake (`rule stage5_neopax`), which runs `neopax` on a config assembled from the upstream Stage 1/2/3/4 outputs. See the [Workflow Engine -- Snakemake](#workflow-engine----snakemake) section to run the forward pass through Stage 5, and [Closing the Loop](#closing-the-loop) for feeding the transport solution back to Stage 1.

> [!NOTE]
> `NEOPAX`, being the final stage, has additional complexities. Ideally the script using `NEOPAX` runs a loop over `sfincs_jax` fluxes to optimize for ambipolarity, which is the most computationally expensive step in the pipeline.

---

## Workflow Engine -- Snakemake

**Code:** [Snakemake](https://snakemake.readthedocs.io/) 

Automates the MVP forward pass end-to-end: `Stage 1 -> {Stage 2, Stage 3, Stage 4} -> Stage 5`. Stages 2 and 3 fan out in parallel off the Stage 1 wout; Stage 4 also needs Stage 2's boozmn, so it follows Stage 2; and Stage 5 consumes all upstream outputs (wout, boozmn, neoclassical + turbulent fluxes). Each stage runs inside its pre-built GHCR container image (`ghcr.io/driftless-star/driftless-star:stage-N-<code>-{cpu,gpu}`) via `docker run`, so no local Pixi install is required beyond the `pipeline` env itself.

| Direction | Format              | Location                                                                                            |
| --------- | ------------------- | --------------------------------------------------------------------------------------------------- |
| **In**    | YAML config         | `config.yaml` (Currently: `run_name`, `stage3_backend`, `device`)                                   |
| **In**    | Workflow definition | `Snakefile`                                                                                         |
| **In**    | Per-stage inputs    | `stages/stage{1,3,4}-*/input/`                                                                         |
| **Out**   | Stage 2 NetCDF      | `stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc`                                           |
| **Out**   | Stage 3 HDF5        | `stages/stage3-neoclassical/output/sfincsOutput_quickrun.h5`                                                    |
| **Out**   | Stage 4 JSON + CSV  | `stages/stage4-turbulence/output/hsx_run_quickrun.{summary.json,diagnostics.csv}`                      |
| **Out**   | Stage 4 cache       | `stages/stage4-turbulence/output/wout_HSX_vacuum_ns201_quickrun.eik.nc` (geometry, regenerated every rerun) |
| **Out**   | Stage 5 HDF5        | `stages/stage5-transport/output/{run_name}/transport_solution.h5` (transport solution; default `rule all` target) |

> [!NOTE]
> `rule all` targets the Stage 5 transport solution (`transport_solution.h5`); all upstream artifacts (wout, boozmn, neoclassical + turbulent fluxes) are produced transitively because downstream rules declare them as `input:`. The loop-closing post-processing step is *not* part of `rule all` -- a plain `snakemake` stays a pure forward pass; see [Closing the Loop](#closing-the-loop).

> [!NOTE]
> Docker must be running on the host. On macOS / Windows that is Docker Desktop; on Linux, the docker engine or a rootless equivalent (podman aliased to `docker`). Windows users should invoke from WSL2 or Git Bash so bash expansions like `$PWD` resolve correctly inside the Snakefile's shell directives. HPC clusters that disallow Docker are a planned follow-up (Apptainer via `--sdm apptainer`).

### How to Install

From the repo root

```
pixi install --environment pipeline
pixi run -e pipeline dot -c
```

> [!NOTE]
> `dot -c` is a one-time step required because Pixi deliberately skips package post-link scripts by default ([security rationale](https://pixi.sh/latest/reference/pixi_configuration/#run-post-link-scripts)). Graphviz's own post-link script normally calls `dot -c` to register its renderer plugins ([graphviz(1)](https://manpages.debian.org/bookworm/graphviz/dot.1.en.html), `"-c  configure plugins"`).

> [!NOTE]
> `pipeline` pulls Snakemake from the **bioconda** channel (declared on the feature in the root `pixi.toml`, not the workspace). Every stage env in `stages/pixi.toml` lives on conda-forge only.

### How to Run

From the repo root, with Docker running:

```
pixi run -e pipeline snakemake -n                         # dry-run: shows the plan without executing
pixi run -e pipeline snakemake --cores 4                  # full forward pass through Stage 5 (Stages 2 and 3 run in parallel after Stage 1; Stage 4 follows Stage 2)
pixi run -e pipeline snakemake clean --cores 1            # wipe every stage's output/ dir
```

Per-invocation overrides via `--config`:

```
pixi run -e pipeline snakemake --cores 4 --config device=gpu              # use -gpu images + --gpus all
pixi run -e pipeline snakemake --cores 4 --config stage3_backend=sfincs_fortran   # swap Stage 3 backend
```

Whole-file alternate config via `--configfile`:

```
pixi run -e pipeline snakemake --cores 4 --configfile config_bigtest.yaml
```

The alternate file is *layered on top of* `config.yaml`, not a replacement; list only the keys you want to change. Any key you omit inherits from `config.yaml`. You can also combine it with `--config` for per-invocation overrides on top of both files.

> [!NOTE]
> Precedence (highest wins): `--config key=value` on the CLI → `--configfile other.yaml` → `configfile: "config.yaml"` (in the Snakefile) → in-code defaults via `config.get(...)`. `--config` is how `device` works on machines with different hardware without committing host-specific defaults; `--configfile` is how named scenarios (e.g. `config_production.yaml`, `config_smalltest.yaml`) live alongside the base config without editing it.

> [!NOTE]
> `device=gpu` requires an NVIDIA host with `nvidia-container-toolkit` configured on the docker daemon.

### Visualizing the file-flow graph

`--filegraph` renders input/output *files* as nodes and rules as edges, so you see the data flow (wout → boozmn, sfincsOutput.h5, hsx_run.*) rather than the abstract job DAG.

SVG:

```
pixi run -e pipeline bash -c 'snakemake --filegraph | dot -Tsvg > ./driftless-star_filegraph.svg'
```

PDF:

```
pixi run -e pipeline bash -c 'snakemake --filegraph | dot -Tpdf > ./driftless-star_filegraph.pdf'
```

PNG:

```
pixi run -e pipeline bash -c 'snakemake --filegraph | dot -Tpng -Gdpi=150 > ./driftless-star_filegraph.png'
```


> [!NOTE]
> For the inverse view (*rules* as nodes, showing how they depend on each other irrespective of which files they share), swap `--filegraph` for `--rulegraph`. For the per-job DAG (every rule instance, one node each, most useful when wildcards produce many parallel jobs), swap for `--dag`.

---

## Closing the Loop

The pipeline can iterate toward a transport-consistent pressure profile by feeding Stage 5's transport solution back into the Stage 1 equilibrium input and re-running the forward pass. Because this feedback overwrites a Stage 1 input, it cannot be expressed as an acyclic Snakemake DAG; instead an external driver (`src/ouroboros.py`, exposed as the `ouroboros` pixi task) sequences the iterations, each as an independent Snakemake run.

### How to Run

From the repo root, with Docker running:

```
pixi run -e pipeline ouroboros --max-iters 3 --cores 4
```

| Flag          | Default       | Meaning                                              |
| ------------- | ------------- | ---------------------------------------------------- |
| `--config`    | `config.yaml` | Pipeline config file.                                |
| `--max-iters` | `3`           | Number of iterations (independent forward passes).   |
| `--cores`     | `4`           | Cores passed to `snakemake --cores`.                 |

> [!NOTE]
> Each iteration is a full forward pass, so Docker must be running and all stage images must be available (the loop exercises `stage-1-vmec` through `stage-5-neopax`; the post-processing step reuses the Stage 5 image). The driver runs on the orchestration `pipeline` env, like Snakemake itself.

### What it does

Each iteration runs as an independent Snakemake pass with `run_name={base}_iter_N`, targeting the **convergence signal file** rather than the default `rule all`. Targeting the signal pulls in `rule stage5_post_processing`, so the chain built is Stage 1 → … → Stage 5 → post-processing. Inside the Stage 5 container, post-processing:

1. **Fits** a VMEC pressure profile from this pass's `transport_solution.h5` and writes it into the *next* iteration's Stage 1 input (`fit_vmec_pressure_from_transport_h5.py write-input`). This is an *undeclared* output, which keeps the Snakemake DAG acyclic.
2. **Checks convergence** (`stage5_post_processing.py`), writing `converge_status.json`.

The driver chains iterations by the fit: iteration N's fitted pressure seeds iteration N+1. The base Stage 1 input and the Stage 3/4 templates are never mutated; only `_iter_N` derivatives are.

> [!NOTE]
> A plain `snakemake` (e.g. `pixi run -e pipeline snakemake --cores 4`) builds `rule all` = the Stage 5 transport solution and **stops at Stage 5**, so it never runs the fit or mutates a Stage 1 input. The loop is opt-in: it is reached only by targeting the signal file, which `ouroboros` does for you.

### Why per-iteration `run_name`

Each iteration writes into its own `{base}_iter_N` output tree. This is load-bearing: Stages 3 and 4 cache their per-radius runs keyed by output directory, so a single shared `run_name` would reuse the first pass's cache and feed Stage 5 **stale** neoclassical/turbulent fluxes on every subsequent pass. Distinct `{base}_iter_N` trees force Stages 3/4 to recompute each iteration.

### Output layout

Per-iteration stage outputs land in run-namespaced trees, and the driver snapshots the input + transport solution for each pass:

```
stages/stage{N}-*/output/{base}_iter_N/                                  # per-iteration stage outputs
stages/stage5-post-processing/output/{base}_iter_N/converge_status.json  # convergence signal
stages/loop-output/{base}/iter_N/
├── input.{base}_iter_N        # the Stage 1 input that fed this pass (snapshot, pre-overwrite)
└── transport_solution.h5      # the Stage 5 transport solution for this pass
```

`{base}` is the configured `run_name` (e.g. `HSX_vacuum_ns201_quickrun`). The `stages/loop-output/` snapshots are gitignored.

### Convergence

Convergence is currently a **stub**: `converged()` in `stage5_post_processing.py` always returns `False`, so `converge_status.json` is always `{"converged": false}` and the loop runs the full `--max-iters`. The signal file is the seam for a future "Stage 5 output unchanged" criterion.

> [!NOTE]
> To render the file-flow graph *including* this post-processing step, target the signal file; see the [README](../README.md#visualize-the-pipeline-graph). Omitting the target graphs the plain forward pass (stops at Stage 5).

---
