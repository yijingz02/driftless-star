# Stage 1: Equilibrium

## Overview

Stage 1 solves the three-dimensional ideal-MHD equilibrium problem, producing the magnetic field geometry and flux-surface profiles that all downstream stages depend on. This is the entry point of the forward-pass pipeline.

**Physics:** Given a plasma boundary shape and profile guesses (pressure, rotational transform or current), find the 3D magnetic equilibrium satisfying force balance: $\nabla p = \mathbf{J} \times \mathbf{B}$, $\nabla \cdot \mathbf{B} = 0$.

**Position in pipeline:** This stage has no upstream dependencies. Its output (`wout_*.nc`) is consumed by Stage 2 (Boozer Transform) and also directly by some turbulence and transport codes.

Reference: `stellarator_workflow/stellarator_workflow.tex`, Section 4.1 (`VMEC++` and `vmec_jax`) and Section 4.2 (`DESC`).

---

## Codes

### vmec_jax (Primary JAX)
- **Repository:** https://github.com/uwplasma/vmec_jax
- **Language:** Python/JAX
- **Role:** JAX-native implementation providing differentiable equilibrium solving with wout-compatible output

### VMEC++ (C++ Alternative)
- **Repository:** https://github.com/proximafusion/vmecpp
- **Documentation:** https://proximafusion.github.io/vmecpp/
- **Language:** C++ with Python bindings
- **Role:** From-scratch C++ reimplementation of `VMEC`. Solves fixed- and free-boundary ideal-MHD equilibria. Preserves the standard `wout` downstream contract.

### DESC (Differentiable Alternative)
- **Repository:** https://github.com/PlasmaControl/DESC
- **Language:** Python/JAX
- **Role:** Differentiable pseudo-spectral equilibrium and optimization suite. Can replace `VMEC++` as the equilibrium engine and also perform some downstream computations (Boozer transform, geometry objectives) internally.

### Installation & Platform

**`vmec_jax`:** Install via the Pixi environment. From the repo root:

```
pixi install --environment stage-1-vmec
```

**`desc-opt`:** Install via the Pixi environment. From the repo root:

```
pixi install --environment stage-1-desc
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `VMEC++` and `DESC`.

---

## Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.1.

### Physical Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `RBC(m,n)` | 2D array (float) | Boundary R cosine Fourier coefficients | User-specified |
| `ZBS(m,n)` | 2D array (float) | Boundary Z sine Fourier coefficients | User-specified |
| `AM` | 1D array (float) | Pressure profile coefficients | User-specified |
| `AM_AUX_*` | arrays (float) | Auxiliary pressure arrays (alternative to AM) | User-specified |
| `AI` | 1D array (float) | Rotational transform iota coefficients (if iota-prescribed) | User-specified |
| `AC` | 1D array (float) | Current profile coefficients (if current-prescribed) | User-specified |
| `AC_AUX_*` | arrays (float) | Auxiliary current arrays | User-specified |
| `PHIEDGE` | scalar (float) | Total toroidal flux (magnetic scale) | User-specified |

### Resolution & Solver Controls

| Field                   | Type          | Description                                                 |
| ----------------------- | ------------- | ----------------------------------------------------------- |
| `NS`                    | int or array  | Number of radial grid points (can be a multi-grid sequence) |
| `MPOL`                  | int           | Maximum poloidal mode number                                |
| `NTOR`                  | int           | Maximum toroidal mode number                                |
| `NITER` / `NITER_ARRAY` | int / array   | Iteration budgets                                           |
| `FTOL` / `FTOL_ARRAY`   | float / array | Convergence tolerances                                      |

### Input Formats
- **INDATA files:** Fortran-style text `input.NAME` format (`vmec_jax` and `VMEC++`)
- **JSON:** Programmatic input (`VMEC++` only)
- **Python objects:** In-memory API (both `VMEC++` and `vmec_jax`)
- **Hot restart:** Previous converged output state as initial guess

### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Output Specification

Reference: `stellarator_io_reference.tex`, Section 3.1.

### Primary Output: `wout_*.nc` (NetCDF)

#### Geometry Scalars

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `aspect` | scalar (float) | Aspect ratio R/a | Objective |
| `Aminor_p` | scalar (float) | Minor radius | Geometry |
| `Rmajor_p` | scalar (float) | Major radius | Geometry |
| `volume_p` | scalar (float) | Plasma volume | Objective |
| `betatotal` | scalar (float) | Total plasma beta | Objective |
| `b0` | scalar (float) | Magnetic field on axis | Geometry |
| `volavgB` | scalar (float) | Volume-averaged |B| | Geometry |
| `fsqr` | scalar (float) | Force residual (radial) | QA signal |
| `fsqz` | scalar (float) | Force residual (vertical) | QA signal |
| `fsql` | scalar (float) | Force residual (lambda) | QA signal |

#### Radial Profiles

| Field | Type | Description |
|-------|------|-------------|
| `presf` | 1D array | Pressure on full mesh |
| `pres` | 1D array | Pressure on half mesh |
| `phi` | 1D array | Toroidal flux |
| `phipf` | 1D array | d(phi)/ds on full mesh |
| `chi` | 1D array | Poloidal flux |
| `chipf` | 1D array | d(chi)/ds on full mesh |
| `iotas` | 1D array | Rotational transform on half mesh |
| `iotaf` | 1D array | Rotational transform on full mesh |
| `q_factor` | 1D array | Safety factor (1/iota) |
| `jcuru` | 1D array | Poloidal current density |
| `jcurv` | 1D array | Toroidal current density |
| `buco` | 1D array | Covariant B_theta (Boozer I) |
| `bvco` | 1D array | Covariant B_zeta (Boozer G) |

#### Spectral Geometry

| Field | Type | Description |
|-------|------|-------------|
| `rmnc` | 2D array (ns x mnmax) | R cosine Fourier coefficients |
| `zmns` | 2D array (ns x mnmax) | Z sine Fourier coefficients |
| `lmns` | 2D array (ns x mnmax) | Lambda sine Fourier coefficients |
| `bmnc` | 2D array (ns x mnmax) | |B| cosine Fourier coefficients |
| `gmnc` | 2D array (ns x mnmax) | Jacobian sqrt(g) cosine coefficients |
| `bsubumnc` | 2D array (ns x mnmax) | B_theta cosine coefficients |
| `bsubvmnc` | 2D array (ns x mnmax) | B_zeta cosine coefficients |
| `bsubsmns` | 2D array (ns x mnmax) | B_s sine coefficients |
| `currumnc` | 2D array (ns x mnmax) | J_theta cosine coefficients |
| `currvmnc` | 2D array (ns x mnmax) | J_zeta cosine coefficients |

#### Python API Objects (`vmec_jax` / `VMEC++`)

| Object | Description |
|--------|-------------|
| `wout` | Full wout data structure |
| `threed1_volumetrics` | 3D volume integrals |
| `jxbout` | J x B force-balance diagnostics |
| `mercier` | Mercier stability criterion |

### Subset Handed to Next Stage

Stage 2 (`BOOZ_XFORM` / `booz_xform_jax`) needs the **full** equilibrium spectrum and profiles in `wout_*.nc`. `GX`, `Trinity3D`, and `NEOPAX` geometry readers also consume wout-level data for field-line geometry, rotational transform, and surface metrics.

### Outputs Used as Objectives

- Aspect ratio, volume, beta, target iota(s): direct design objectives
- Mercier criterion: stability objective
- Residuals `fsqr`, `fsqz`, `fsql`: QA convergence signals, not physics design objectives

### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Governing Equations

The equilibrium satisfies ideal-MHD force balance:

$$\nabla p = \mathbf{J} \times \mathbf{B}, \quad \nabla \cdot \mathbf{B} = 0, \quad \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}$$

`VMEC++` finds the stationary point of the energy functional (Hirshman & Whitson 1983):

$$W = \frac{1}{(2\pi)^2} \int \left( \frac{B^2}{2} + \frac{p}{\gamma - 1} \right) dV$$

In `VMEC++` flux coordinates with the stream function lambda:

$$u = \theta + \lambda(s, \theta, \zeta), \quad \frac{du}{d\zeta} = \iota(s)$$

The contravariant field components are:

$$B^\zeta = \frac{\Phi'(s) + \text{lamscale} \cdot \partial_\theta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

$$B^\theta = \frac{\chi'(s) - \text{lamscale} \cdot \partial_\zeta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

`DESC` solves the same physics in a pseudo-spectral formulation:

$$\mathbf{B} = \frac{\partial_\rho \psi}{2\pi\sqrt{g}} \left[ \left(\iota - \frac{\partial\lambda}{\partial\zeta}\right) \mathbf{e}_\theta + \left(1 + \frac{\partial\lambda}{\partial\theta}\right) \mathbf{e}_\zeta \right]$$

Reference: `stellarator_workflow.tex`, Sections 4.1-4.2.

---

## Convergence & Validity

> [!TODO]
> Document convergence behavior, known failure modes, and recommended tolerances.

---

## API Documentation

> [!TODO]
> Document key entry points, configuration parameters, and usage examples.

---

## Scripts & Workflows

**`vmec_jax` (via Pixi):** From the repo root:

```
pixi run stage-1-vmec
```

**Input:** `stages/stage1-equilibrium/input/input.HSX_vacuum_ns201_quickrun`
**Output:** `stages/stage1-equilibrium/output/wout_HSX_vacuum_ns201_quickrun.nc`

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and debugging workflows for `VMEC++` and `DESC`.

---

## W&B Tracking

**Project:** `stellaforge-stage1-equilibrium`

> [!TODO]
> Set up W&B tracking.

---

## Container Specification (Phase 2)

**`vmec_jax`:** Built from the single templated `Dockerfile` using a build process morally equivalent to:

```
docker build --file Dockerfile --build-arg ENVIRONMENT=stage-1-vmec --platform linux/amd64 --tag ghcr.io/rkhashmani/stellaforge:stage-1-vmec-cpu .  # CPU
docker build --file Dockerfile --build-arg CUDA_VERSION=12 --build-arg ENVIRONMENT=stage-1-vmec-gpu --platform linux/amd64 --tag ghcr.io/rkhashmani/stellaforge:stage-1-vmec-gpu .  # GPU
```

Published to GHCR as `ghcr.io/rkhashmani/stellaforge:stage-1-vmec-cpu` and `stage-1-vmec-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `VMEC++` and `DESC`.

---

## Tests (Phase 2)

> [!TODO]
> Write unit, regression, and integration tests. See [guide](../guide.md#writing-tests) for examples.

---

## Claude Skills

> [!TODO]
> Create development and operational Claude skills. See [guide](../guide.md#step-7-create-claude-skills) for skill types.
