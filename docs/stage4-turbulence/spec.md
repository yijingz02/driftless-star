# Stage 4: Turbulence

## Overview

Stage 4 solves the gyrokinetic equations to compute turbulent transport. The primary outputs -- heat and particle fluxes -- are both optimization objectives (to minimize) AND direct transport inputs for Stage 5.

**JAX-first priority:** `SPECTRAX-GK` is the primary code (JAX-native, differentiable). `GX` and `GENE` are traditional alternatives added later.

**Position in pipeline:** Receives geometry from Stage 1/2. Runs in parallel with Stage 3 (Neoclassical). Outputs feed Stage 5 (Transport).

**Important coordination point:** The coupling between `SPECTRAX-GK` output and `NEOPAX` (Stage 5) is less mature than the `GX`-`Trinity3D` coupling. `NEOPAX` has turbulence-coupling utilities but the public examples focus on the neoclassical reduced model. The Stage 4 and 5 owners must coordinate on this interface.

Reference: `stellarator_workflow.tex`, Section 4.7; `stellarator_io_reference.tex`, Sections 3.9-3.10.

---

## Codes

### SPECTRAX-GK (Primary JAX)
- **Repository:** https://github.com/uwplasma/SPECTRAX-GK
- **Language:** Python/JAX
- **Role:** JAX-native gyrokinetic solver for differentiable turbulence calculations

### GX (Alternative)
- **Repository:** https://bitbucket.org/gyrokinetics/gx
- **Language:** Fortran/CUDA
- **Role:** GPU-native gyrokinetic code, mature coupling with `Trinity3D`

### GENE / GENE-3D (Alternative)
- **Website:** https://genecode.org
- **Language:** Fortran
- **Role:** High-fidelity grid-based Eulerian gyrokinetic code

### Installation & Platform

**`SPECTRAX-GK`:** Install via the Pixi environment. From the repo root:

```
pixi install --environment stage-4-spectrax
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `GX` and `GENE`.

---

## Input Specification

Reference: `stellarator_io_reference.tex`, Sections 3.9-3.10.

### SPECTRAX-GK Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| TOML config | file | Grid, geometry, physics toggles, time integration | User-specified |
| Geometry | analytic or `*.eik.nc` | Magnetic geometry (can be VMEC-derived) | Stage 1/2 |
| Species profiles | in config | Density, temperature, gradients per species | User-specified |
| Collisionality | in config | Collision parameters | User-specified |
| Beta | in config | Electromagnetic parameter | User-specified |

### `GX` Inputs (Alternative)

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `run_name.in` | Input file | Geometry, species, domain, time stepping, diagnostics, resolution | User-specified |
| VMEC geometry | via geometry module | Field-line geometry from wout | Stage 1 |
| `omega=true` | flag | Enable growth-rate diagnostics | Config |
| `fluxes=true` | flag | Enable flux diagnostics | Config |

### `GENE` Inputs (Alternative)

Installation-dependent. Key physics contract: geometry from VMEC/Boozer, species profiles/gradients, collisionality, electromagnetic parameters, numerical grid settings.

### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Output Specification

Reference: `stellarator_io_reference.tex`, Sections 3.9-3.10.

### SPECTRAX-GK Outputs

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `gamma` | scalar/array | Linear growth rate | Objective / screening |
| `omega` | scalar/array | Real frequency | Diagnostic |
| `gamma_t` | 1D array (time) | Growth rate time trace | Convergence check |
| `omega_t` | 1D array (time) | Frequency time trace | Convergence check |
| `Wg_t` | 1D array (time) | Free energy (g) trace | Diagnostic |
| `Wphi_t` | 1D array (time) | Free energy (phi) trace | Diagnostic |
| `Wapar_t` | 1D array (time) | Free energy (A_parallel) trace | Diagnostic |
| `heat_flux_t` | 1D array (time) | Heat flux time trace | **Transport input** |
| `particle_flux_t` | 1D array (time) | Particle flux time trace | **Transport input** |

Optional CSV output: time, growth rate, frequency, free energy, species-resolved heat and particle flux.

The natural downstream contract is the same as `GX`: turbulent heat and particle flux (steady-state values).

### `GX` Outputs (Alternative)

| File | Description |
|------|-------------|
| `run_name.out.nc` | Linear run output |
| `run_name.nc` | Nonlinear run output |
| `run_name.big.nc` | Saved field diagnostics |
| `run_name.restart.nc` | Restart data |

Key NetCDF groups: `Grids`, `Geometry`, `Diagnostics`, `Inputs`

| Field | Location | Description | Used As |
|-------|----------|-------------|---------|
| `ParticleFlux_st` | `Diagnostics/` | Particle flux (species, time) | **Transport input** (`Trinity3D`) |
| `HeatFlux_st` | `Diagnostics/` | Heat flux (species, time) | **Transport input** (`Trinity3D`) |
| `pflux` | `Fluxes/` | Particle flux (alternative location) | Transport input |
| `qflux` | `Fluxes/` | Heat flux (alternative location) | Transport input |
| `ParticleFlux_zst` | `Diagnostics/` | Zeta-resolved particle flux (stellarator) | Transport input |
| `HeatFlux_zst` | `Diagnostics/` | Zeta-resolved heat flux (stellarator) | Transport input |
| `omega_v_time` | `Special/` | Linear growth rate vs time | Screening |

`GX` spectral representation: Hermite-Laguerre velocity-space basis:

$$h_s = \sum_{\ell,m,k_x,k_y} \hat{h}_{s,\ell,m}(z,t)\, e^{i(k_x x + k_y y)} H_m\left(\frac{v_\parallel}{v_{ts}}\right) L_\ell\left(\frac{v_\perp^2}{v_{ts}^2}\right) F_{Ms}$$

### `GENE` Outputs (Alternative)

Installation-dependent filenames. Key outputs: linear growth rates, real frequencies, eigenfunctions, nonlinear species heat/particle fluxes, spectra, time histories.

### Subset Handed to Next Stage

For transport coupling, the critical handoff is the **turbulent flux vector** (steady-state heat and particle flux per species). For screening, only linear gamma and omega may be retained.

`Trinity3D` obtains flux Jacobians by rerunning `GX` on perturbed gradients and finite-differencing.

### Outputs Used as Objectives

- Linear gamma, omega: rapid screening
- Nonlinear heat flux, particle flux: high-fidelity design objectives
- Heat flux is BOTH an objective AND a transport input (dual-role output)

### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Governing Equations

Generic delta-f gyrokinetic equation:

$$\frac{\partial h_s}{\partial t} + v_\parallel \mathbf{b}\cdot\nabla h_s + \mathbf{v}_{Ds}\cdot\nabla h_s + \mathbf{v}_E\cdot\nabla h_s - C[h_s] = -\frac{Z_s e F_{Ms}}{T_s}\frac{\partial\langle\chi\rangle}{\partial t} - \mathbf{v}_\chi\cdot\nabla F_{Ms}$$

Closed by quasineutrality and (for electromagnetic calculations) appropriate field equations.

Reference: `stellarator_workflow.tex`, Section 4.7.

---

## Convergence & Validity

> [!TODO]
> Document convergence criteria, resolution requirements, known failure modes, and benchmark comparisons.

---

## API Documentation

> [!TODO]
> Document key entry-point functions, programmatic usage, JAX differentiation, and configuration effects.

---

## Scripts & Workflows

**`SPECTRAX-GK` (via Pixi):** From the repo root:

```
pixi run stage-4-spectrax
```

which executes something morally equivalent to:

```
spectrax-gk run --config ./stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry.toml --out stage4-turbulence/output/hsx_run
```

**Input:** `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc` + `stages/stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry_quickrun.toml`
**Output:** `stages/stage4-turbulence/output/hsx_run_quickrun.summary.json` + `stages/stage4-turbulence/output/hsx_run_quickrun.diagnostics.csv`

> [!NOTE]
> The TOML's `vmec_file` points into `stage1-equilibrium/output/`. Populate this directory by running `pixi run stage-1-vmec` first. The VMEC geometry path also requires `booz_xform_jax` at runtime (lazy dependency).

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and workflows for `GX` and `GENE`.

---

## W&B Tracking

**Project:** `stellaforge-stage4-turbulence`

> [!TODO]
> Set up W&B tracking.

---

## Container Specification (Phase 2)

**`SPECTRAX-GK`:** Built from the single templated `Dockerfile` using build arguments:

```
docker build --build-arg ENVIRONMENT=stage-4-spectrax .        # CPU
docker build --build-arg ENVIRONMENT=stage-4-spectrax-gpu --build-arg CUDA_VERSION=12 .  # GPU
```

Published to GHCR as `ghcr.io/rkhashmani/stellaforge:stage-4-spectrax-cpu` and `stage-4-spectrax-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `GX` and `GENE`.

---

## Tests (Phase 2)

> [!TODO]
> Write unit, regression, benchmark, and integration tests (including Stage 5 coupling).
> See [guide](../guide.md#writing-tests) for examples.

---

## Claude Skills

> [!TODO]
> Create dev, operational, and cross-stage Claude skills for SPECTRAX-GK and GX workflows.
> See [guide](../guide.md#step-7-create-claude-skills) for skill types.
