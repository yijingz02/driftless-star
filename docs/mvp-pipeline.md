# MVP Pipeline: I/O Reference

```
 INDATA                   wout_*.nc                boozmn_*.nc
   |                          |                         |
   v                          v                         v
┌────────┐  NetCDF  ┌──────────────┐  NetCDF  ┌──────────────────────┐
│Stage 1 │ -------> │   Stage 2    │ -------> │      Stage 3         │
│vmec_jax│ wout_*.nc│booz_xform_jax│boozmn_*.nc│ SFINCS       monkes │
└────────┘          └──────────────┘          └───┬──────────────┬───┘
                          |                       |              |
                          |                       v              v
                          |              sfincsOutput.h5      D_ij.h5
                          |                  (HDF5)           (HDF5)
                          |                       |              |
                          |    ┌────────────┐     |              |
                          +--->│  Stage 4   │     |              |
                     wout_*.nc │ SPECTRAX-GK│     |              |
                               └─────┬──────┘     |              |
                                     |             |              |
                                flux (CSV)         |              |
                                     |             |              |
                                     v             v              v
                     ┌───────────────────────────────────────────────┐
                     │                  Stage 5                     │
                     │            Transport / Profiles              │
                     │  NEOPAX (wout + boozmn + D_ij + turb flux)   │
                     └──────────────────┬──────────────────────────┘
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
├── stage3-neoclassical/    input/  (input.HSX_vacuum_ns201_quickrun) + run_monkes.py
├── stage4-turbulence/      input/  (runtime_hsx_nonlinear_vmec_geometry_quickrun.toml)
└── stage5-transport/       run_NEOPAX.py
```

Each stage's `input/` directory ships with a `_quickrun` smoke-test variant so a fresh clone is immediately runnable. Each stage's `output/` directory is gitignored: Snakemake regenerates it in place via `pixi run -e pipeline snakemake --cores 4`, and cross-stage configs read upstream outputs from there. Run stages in forward-chain order (`stage-1-vmec` first) when invoking individual pixi tasks; Snakemake handles the dependency order automatically.

---

**Note:** All paths are relative to the repository root.

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
pixi install --environment stage-1-vmec
```

### How to Run

```
pixi run stage-1-vmec
```

---

## Stage 2 -- Boozer Transform

**Code:** booz_xform_jax

| Direction | Format               | Location                                                                          |
| --------- | -------------------- | --------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc`   | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`    |
| **Out**   | NetCDF `boozmn_*.nc` | `stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc` |

> [!NOTE]
> Stage 2's JAX driver takes explicit `--wout` and `--output` paths. Populate `stage1-equilibrium/output/` by running `pixi run stage-1-vmec` first.

### How to Install

```
pixi install --environment stage-2-booz-jax
```

### How to Run

```
pixi run -e stage-2-booz-jax stage-2-booz
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

## Stage 3 -- Neoclassical (two parallel sub-stages)

**Code:** sfincs_jax

| Direction | Format                       | Location                                                                       |
| --------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`           | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc` |
| **In**    | Fortran-style Text `input.*` | `stages/stage3-neoclassical/input/input.HSX_vacuum_ns201_quickrun`          |
| **Out**   | HDF5 `sfincsOutput.h5`       | `stages/stage3-neoclassical/output/sfincsOutput_quickrun.h5`           |

#### How to Install

```
pixi install --environment stage-3-sfincs
```

#### How to Run

```
pixi run stage-3-sfincs
```

> [!NOTE]
> The pixi `stage-3-sfincs` task and the Snakemake `stage3_sfincs` rule both pass the wout path to `sfincs_jax` via `--wout-path`, overriding the namelist `equilibriumFile` field. Populate `stage1-equilibrium/output/` by running `pixi run stage-1-vmec` first. The `sfincs_fortran` backend has no CLI override and still reads `equilibriumFile` from the namelist.


**Code:** SFINCS (Fortran)

| Direction | Format                       | Location                                                                       |
| --------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`           | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc` |
| **In**    | Fortran-style Text `input.*` | `stages/stage3-neoclassical/input/input.HSX_vacuum_ns201_quickrun`          |
| **Out**   | HDF5 `sfincsOutput.h5`       | `stages/stage3-neoclassical/output/sfincsOutput_quickrun.h5`           |

#### How to Install

```
pixi install --environment stage-3-sfincs-fortran
```

#### How to Run

```
pixi run stage-3-sfincs-fortran
```

> [!NOTE]
> `SFINCS` (Fortran) is an alternative to `sfincs_jax`. It reads the same namelist and writes to the same `sfincsOutput.h5` path, so running both against one output directory will overwrite the prior result.

> [!NOTE]
> The task copies `stage3-neoclassical/input/input.HSX_vacuum_ns201_quickrun` to `stage3-neoclassical/output/input.namelist` before invoking the binary, because SFINCS (Fortran) reads `input.namelist` from its working directory.


**code:** Monkes

| Direction | Format               | Location                                                                                         |
| --------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`   | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`                   |
| **In**    | NetCDF `boozmn_*.nc` | `stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc`                |
| **Out**   | HDF5 `D_ij.h5`       | `stages/stage3-neoclassical/output/Monoenergetic_database_VMEC_s_coordinate_HSX.h5` |

> [!NOTE]
> The inputs come from both Stage 1 and Stage 2.

#### How to Install
```bash
git clone https://github.com/eduardolneto/monkes.git
cd monkes
pip install .
```

#### How to Run

Monkes is a little more involved. See `stages/stage3-neoclassical/run_monkes.py`

We basically call it inside a python loop to use Monkes to generate a database across different radial positions, electric fields, and collisionality, but it uses the same 2 input files for the entire loop.

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
> The TOML's `vmec_file` points into `stage1-equilibrium/output/`. Populate this directory by running `pixi run stage-1-vmec` first.

### How to Install

```
pixi install --environment stage-4-spectrax
```

### How to Run

```
pixi run stage-4-spectrax
```

which executes something morally equivalent to

```
spectrax-gk run --config runtime_hsx_nonlinear_vmec_geometry.toml --out output/hsx_run
```

---

## Stage 5 -- Transport Solver

**Code:** NEOPAX

| Direction                                 | Format               | Location                                                                                         |
| ----------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **In**                                    | NetCDF `wout_*.nc`   | `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`                   |
| **In**                                    | NetCDF `boozmn_*.nc` | `stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc`                |
| **In** (Only needed if `monkes` is used.) | HDF5 `D_ij.h5`       | `stages/stage3-neoclassical/output/Monoenergetic_database_VMEC_s_coordinate_HSX.h5` |
| **Out**                                   | HDF5 `profiles_*.h5` | `stages/stage5-transport/output/NEOPAX_output_quickrun.h5`                                   |
> [!NOTE]
> The inputs come from Stage 1, Stage 2, and Stage 3.

### How to Install

```
pixi install --environment stage-5-neopax
```

### How to Run

This is run inside a script. See `stages/stage5-transport/run_NEOPAX.py` as a reference.

> [!NOTE]
> `NEOPAX`, being the final stage, has additional complexities and variabilities in the workflow.
>
> For example, if `monkes` is used, then the script using `NEOPAX` won't constantly loop since `monkes` provides a database and does the loop before reaching Stage 5.
>
> If `sfincs_jax` is used, then ideally, the script using NEOPAX runs a loop to optimize over different fluxes. While ideal, this is most computationally expensive.

---

## Workflow Engine -- Snakemake

**Code:** [Snakemake](https://snakemake.readthedocs.io/) 

Automates the MVP forward pass end-to-end: `Stage 1 -> {Stage 2, Stage 3, Stage 4}` with the three downstream stages fanning out in parallel off the Stage 1 wout. Each stage runs inside its pre-built GHCR container image (`ghcr.io/rkhashmani/stellaforge:stage-N-<code>-{cpu,gpu}`) via `docker run`, so no local Pixi install is required beyond the `pipeline` env itself.

> [!NOTE]
> Stage 5 (NEOPAX) is not yet orchestrated.

| Direction | Format              | Location                                                                                            |
| --------- | ------------------- | --------------------------------------------------------------------------------------------------- |
| **In**    | YAML config         | `config.yaml` (Currently: `run_name`, `stage3_backend`, `device`)                                   |
| **In**    | Workflow definition | `Snakefile`                                                                                         |
| **In**    | Per-stage inputs    | `stages/stage{1,3,4}-*/input/`                                                                         |
| **Out**   | Stage 2 NetCDF      | `stages/stage2-boozer/output/boozmn_HSX_vacuum_ns201_quickrun.nc`                                           |
| **Out**   | Stage 3 HDF5        | `stages/stage3-neoclassical/output/sfincsOutput_quickrun.h5`                                                    |
| **Out**   | Stage 4 JSON + CSV  | `stages/stage4-turbulence/output/hsx_run_quickrun.{summary.json,diagnostics.csv}`                      |
| **Out**   | Stage 4 cache       | `stages/stage4-turbulence/output/wout_HSX_vacuum_ns201_quickrun.eik.nc` (geometry, regenerated every rerun) |

> [!NOTE]
> `rule all` lists only the terminal artifacts above. Upstream intermediates (Stage 1's wout) are produced transitively because downstream rules declare them as `input:`.

> [!NOTE]
> Docker must be running on the host. On macOS / Windows that is Docker Desktop; on Linux, the docker engine or a rootless equivalent (podman aliased to `docker`). Windows users should invoke from WSL2 or Git Bash so bash expansions like `$PWD` resolve correctly inside the Snakefile's shell directives. HPC clusters that disallow Docker are a planned follow-up (Apptainer via `--sdm apptainer`).

### How to Install

From the repo root

```
pixi install --environment pipeline
pixi run -e pipeline dot -c
```

> [!NOTE]
> `dot -c` is a one-time step required because Pixi deliberately skips package post-link scripts by default ([security rationale](https://pixi.sh/latest/reference/pixi_configuration/#run-post-link-scripts)). Graphviz's own post-link script normally calls `dot -c` to register its renderer plugins ([graphviz(1)](https://manpages.debian.org/bookworm/graphviz/dot.1.en.html) — `"-c  configure plugins"`).

> [!NOTE]
> `pipeline` pulls Snakemake from the **bioconda** channel (declared on the feature, not the workspace), not conda-forge where every other stage env lives.

### How to Run

From the repo root, with Docker running:

```
pixi run -e pipeline snakemake -n                         # dry-run: shows the plan without executing
pixi run -e pipeline snakemake --cores 4                  # full pipeline, stages 2/3/4 in parallel
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

The alternate file is *layered on top of* `config.yaml`, not a replacement — list only the keys you want to change. Any key you omit inherits from `config.yaml`. You can also combine it with `--config` for per-invocation overrides on top of both files.

> [!NOTE]
> Precedence (highest wins): `--config key=value` on the CLI → `--configfile other.yaml` → `configfile: "config.yaml"` (in the Snakefile) → in-code defaults via `config.get(...)`. `--config` is how `device` works on machines with different hardware without committing host-specific defaults; `--configfile` is how named scenarios (e.g. `config_production.yaml`, `config_smalltest.yaml`) live alongside the base config without editing it.

> [!NOTE]
> `device=gpu` requires an NVIDIA host with `nvidia-container-toolkit` configured on the docker daemon.

### Visualizing the file-flow graph

`--filegraph` renders input/output *files* as nodes and rules as edges, so you see the data flow (wout → boozmn, sfincsOutput.h5, hsx_run.*) rather than the abstract job DAG.

SVG:

```
pixi run -e pipeline bash -c 'snakemake --filegraph | dot -Tsvg > ./stellaforge_filegraph.svg'
```

PDF:

```
pixi run -e pipeline bash -c 'snakemake --filegraph | dot -Tpdf > ./stellaforge_filegraph.pdf'
```

PNG:

```
pixi run -e pipeline bash -c 'snakemake --filegraph | dot -Tpng -Gdpi=150 > ./stellaforge_filegraph.png'
```


> [!NOTE]
> For the inverse view — *rules* as nodes, showing how they depend on each other irrespective of which files they share — swap `--filegraph` for `--rulegraph`. For the per-job DAG (every rule instance, one node each — most useful when wildcards produce many parallel jobs), swap for `--dag`.

---
