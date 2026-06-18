# Stage 3: Neoclassical Transport

## Overview

Stage 3 computes neoclassical transport properties from the Boozer-coordinate equilibrium. It has two codes:

1. **`NEO` / `NEO_JAX`** -- Computes effective ripple (epsilon_eff), a screening/optimization diagnostic. **NOT a transport state variable** -- does not feed into profile evolution. Runs in parallel with the transport code.
2. **`SFINCS` / `sfincs_jax`** -- Solves the full drift-kinetic equation for neoclassical particle flux, heat flux, bootstrap current, and ambipolar E_r. Feeds `NEOPAX` (Stage 5).

**Position in pipeline:** `NEO_JAX` receives `boozmn_*.nc` from Stage 2 (Boozer). `sfincs_jax` receives `wout_*.nc` directly from Stage 1 (Equilibrium). Stage 3 runs in parallel with Stage 4 (Turbulence).

**Reference:** `stellarator_workflow.tex`, Sections 4.4-4.5; `stellarator_io_reference.tex`, Sections 3.4-3.5.

---

## Sub-Stage 3a: `NEO` / `NEO_JAX` (Effective Ripple)

### Codes

**NEO_JAX (Primary JAX):** <https://github.com/uwplasma/NEO_JAX>

**`NEO` (Legacy, part of `STELLOPT`):** <https://github.com/PrincetonUniversity/STELLOPT>

### Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.4.

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `boozmn_*.nc` | NetCDF file | Boozer-coordinate equilibrium | Stage 2 |
| `neo_in.*` / `neo_param.*` | Control file | Surface list, angular resolution (theta_n, phi_n), Fourier cutoffs, MC controls, accuracy targets, current calculation switch (CALC_CUR) | User-specified |

#### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Output Specification

Reference: `stellarator_io_reference.tex`, Section 3.4.

**Primary output:** `neo_out.*` and `neolog.*`

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `epstot` | 1D array (per surface) | epsilon_eff^{3/2} (total effective ripple) | **Screening objective only** |
| `epspar` | 1D array | Parallel epsilon | Diagnostic |
| `reff` | 1D array | Effective radius | Diagnostic |
| `iota` | 1D array | Rotational transform | Cross-check |
| `b_ref` | scalar | Reference magnetic field | Normalization |
| `r_ref` | scalar | Reference radius | Normalization |
| `ctrone` | 1D array | Contribution from one class | Diagnostic |
| `ctrtot` | 1D array | Total contribution | Diagnostic |
| `bareph` | 1D array | Parallel epsilon (bar) | Diagnostic |
| `barept` | 1D array | Perpendicular epsilon (bar) | Diagnostic |
| `yps` | 1D array | Normalized toroidal flux | Coordinate |

**`NEO_JAX` result objects:** `epsilon_effective`, `epsilon_effective_by_class`

Optional outputs (if CALC_CUR=1): `neo_cur.*`, `current.dat`, `conver.dat`, `diagnostic.dat`, `diagnostic_add.dat`, `diagnostic_bigint.dat`

**Role:** Screening/optimization diagnostic. Usually NO direct transport consumer. epsilon_eff is NOT what `Trinity3D` or `NEOPAX` advances in time.

#### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Governing Equations

Field-line integrals:

$$y_2 = \int d\phi\, B^{-2}, \quad y_3 = \int d\phi\, |\nabla\psi| B^{-2}, \quad y_4 = \int d\phi\, K_G B^{-3}$$

Trapped-particle integrals:

$$I_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}}\, B^{-2}$$

$$H_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}} \left(\frac{4}{B/B_0} - \frac{1}{\eta}\right) \frac{K_G}{\sqrt{\eta}} B^{-2}$$

Class-resolved effective ripple:

$$\epsilon_{\text{eff}}^{3/2}(m) = C_\epsilon \frac{y_2}{y_3^2} \text{BigInt}(m), \quad C_\epsilon = \frac{\pi R_0^2 \Delta\eta}{8\sqrt{2}}$$

Total `epstot` is the sum over classes.

**Reference:** `stellarator_workflow.tex`, Section 4.4.

---

## Sub-Stage 3b: `SFINCS` / `sfincs_jax` (Full Neoclassical)

### Codes

**sfincs_jax (Primary JAX):** <https://github.com/uwplasma/sfincs_jax>

**SFINCS (Legacy):** <https://github.com/landreman/sfincs>

### Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.5.

| Field       | Type                    | Description                                                                                                                                                                                                                                                                                                                              | Source         |
| ----------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `wout_*.nc` | NetCDF file             | VMEC equilibrium output. Passed to `sfincs_jax` via `--wout-path` in the Snakemake rule and pixi task (overrides the namelist `equilibriumFile` field both in memory and in the `input.namelist` dataset embedded in `sfincsOutput.h5`). The `sfincs_fortran` backend has no CLI override and reads `equilibriumFile` from the namelist. | Stage 1        |
| `input.*`   | Fortran-style text file | Configuration with species, gradients, resolution, and a fallback `equilibriumFile` path                                                                                                                                                                                                                                                 | User-specified |

Key namelist parameters: species charges/masses, $\hat{n}_s$, $\hat{T}_s$, their gradients, collision model, E_r guess, Phi1 switches, numerical resolution.

#### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Output Specification

Reference: `stellarator_io_reference.tex`, Section 3.5.

**Forward-chain handoff:** `sfincs_jax_flux_profiles.h5` (HDF5) -- aggregated flux profiles (Gamma, Q, Upar vs radius) consumed by `NEOPAX` (Stage 5).

**Native SFINCS output (per surface):** `sfincsOutput.h5` (HDF5) -- the native solver file. The `sfincs_jax` radial scan writes one per flux surface and aggregates them into the handoff above; the standalone `SFINCS` (Fortran) binary writes it directly. Its fields:

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `particleFlux_vm_rN` | array (per species) | Particle flux (vm normalization, rN coord) | **Transport input** (`Trinity3D`) |
| `heatFlux_vm_rN` | array (per species) | Heat flux (vm normalization, rN coord) | **Transport input** (`Trinity3D`) |
| `particleFlux_vd_rN` | array | Particle flux (with Phi1 enabled) | Transport input (alt) |
| `heatFlux_vd_rN` | array | Heat flux (with Phi1 enabled) | Transport input (alt) |
| `FSABjHat` | array | Flux-surface averaged bootstrap current | Transport/equilibrium feedback |
| `FSABFlow` | array | Flux-surface averaged flow | Diagnostic |
| `Phi1Hat` | array | First-order electrostatic potential | Diagnostic |
| `transportMatrix` | 2D array | Full transport matrix | Analysis |

Also: matching momentum-flux arrays, classical fluxes, optional full-f/delta-f exports, `*_vs_x` lineouts. `sfincs_jax` additionally exposes in-memory result dicts and can write `.npy` state vectors.

**Handoff to `Trinity3D`:** The `Trinity3D` adapter reads `particleFlux_vm_rN` and `heatFlux_vm_rN` when `includePhi1` is off, and `particleFlux_vd_rN` / `heatFlux_vd_rN` when on.

#### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Governing Equations

First-order drift-kinetic equation:

$$(v_\parallel \mathbf{b} + \frac{d\Phi_0}{dr}\frac{\mathbf{B}\times\nabla r}{B^2})\cdot\nabla f_{s1} + [\text{mirror/drift terms}]\frac{\partial f_{s1}}{\partial\xi} - (\mathbf{v}_{ms}\cdot\nabla r)\frac{Z_s e}{2T_s x_s}\frac{d\Phi_0}{dr}\frac{\partial f_{s1}}{\partial x_s}$$

$$+ (\mathbf{v}_{ms}\cdot\nabla r)\left[\frac{1}{n_s}\frac{dn_s}{dr} + \frac{Z_s e}{T_s}\frac{d\Phi_0}{dr} + (x_s^2 - \frac{3}{2})\frac{1}{T_s}\frac{dT_s}{dr}\right]f_{sM} = C_s[f_{s1}] + S_s$$

Collision operator: $C_s[f_s] = \sum_b C_{sb}^l[f_s, f_b]$ with Lorentz, energy-diffusion, and field-particle components.

When Phi1 is included, coupled to quasineutrality.

**Reference:** `stellarator_workflow.tex`, Section 4.5.

---

## Installation & Platform

**`sfincs`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-sfincs-fortran
```

**`sfincs_jax`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-sfincs
```

**`neo-jax`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-neo-jax
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `NEO_JAX` and `NEO`.

---

## Convergence & Validity

> [!TODO]
> Document convergence behavior, known failure modes, and recommended tolerances for all three sub-stages.

---

## API Documentation

> [!TODO]
> Document key entry points, configuration parameters, and usage examples for all three sub-stages.

---

## Scripts & Workflows

**`sfincs_jax` (via Pixi):** From the `stages`/ directory:

```
pixi run stage-3-sfincs
```

> [!NOTE]
> The pixi task and the Snakemake `stage3_sfincs` rule both pass the wout path to `sfincs_jax` via `--wout-path`. Populate `outputs/quick_run/stage1_equilibrium/` by running `pixi run stage-1-vmec` first. The namelist's `equilibriumFile` field is retained as a fallback for the `sfincs_fortran` backend and for direct `sfincs_jax` invocations that omit `--wout-path`.

**Input:** `outputs/quick_run/stage1_equilibrium/wout_HSX_vacuum_ns201_quickrun.nc` + `inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun`
**Output:** `outputs/quick_run/stage3_neoclassical/sfincs_jax_flux_profiles.h5`

See `docs/mvp-pipeline.md` for full I/O details.

**`SFINCS` (Fortran, via Pixi):** From the `stages`/ directory:

```
pixi run stage-3-sfincs-fortran
```

Alternative implementation to `sfincs_jax`. Consumes the same input namelist and writes the native `sfincsOutput.h5`, a different file from the `sfincs_jax_flux_profiles.h5` forward-chain handoff and not wired into the Snakemake forward pass; the task stages the namelist as `input.namelist` in the output directory before invocation because the Fortran binary reads that filename from its working directory.

**Input:** same as `sfincs_jax` above.
**Output:** `outputs/quick_run/stage3_neoclassical/sfincsOutput.h5` (the native SFINCS file, separate from the `sfincs_jax` handoff).

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and workflows for `NEO_JAX`, `NEO`, and `SFINCS`.

---

## W&B Tracking

**Project:** `driftless-star-stage3-neoclassical`

> [!TODO]
> Set up W&B tracking for all three sub-stages.

---

## Container Specification (Phase 2)

**`sfincs_jax`:** Built from the single templated `stages/Dockerfile` using build arguments:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-3-sfincs stages/        # CPU
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-3-sfincs-gpu --build-arg CUDA_VERSION=12 stages/  # GPU
```

Published to GHCR as `ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu` and `stage-3-sfincs-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `NEO_JAX`, `NEO`, and `SFINCS`.

---

## Tests (Phase 2)

See [guide](../guide.md#writing-tests) for examples.

> [!TODO]
> Write unit, regression, and integration tests for all three sub-stages.

---

## Claude Skills

See [guide](../guide.md#step-7-create-claude-skills) for skill types.

> [!TODO]
> Create development and operational Claude skills for all three sub-stages.
