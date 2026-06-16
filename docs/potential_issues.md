# Potential Issues

## Cross-Stage Infrastructure

- [ ] Maybe a location where standard input/outputs are stored and logged? This has the added benefit that we can use this as a "cache" to quickly retrieve outputs that have already been processed, kind of like a database.
## Documentation

- [ ] Add more details for each software, e.g. for Spectrax-gk's `omega_t` what are units, what is it normalized to (gyro freq, etc), scale lengths. 
- [ ] `docs/mvp-pipeline.md` per-stage I/O tables are stale relative to `config.yaml`: the Stage 3/4/5 output filenames (`sfincs_jax_flux_profiles.h5`, `neopax_fluxes.h5`, `transport_solution.h5`) and the per-`run_name` output subdirectories differ from the documented `sfincsOutput_quickrun.h5` / `hsx_run_quickrun.*` / `NEOPAX_output_quickrun.h5`. Refresh the per-stage tables (deferred from the closed-loop docs change, which only refreshed the Snakemake section).

## Code Quality / Tooling

- [ ] No type checker is configured, so the type hints being added across the codebase (e.g. `stages/stage4-turbulence/spectrax_gk_radial_scan.py`) are unverified. Adding one (e.g. `ty`, mypy, or pyright) to CI and/or pre-commit would catch incorrect annotations. Raised on PR #78; deferred to a future PR.

## Stage 1 -- Equilibrium

- [ ] vmec/vmec_jax and DESC do not have directly compatible inputs; an adapter or input translation layer will be needed to support both implementations behind the same pipeline entry point
- [ ] vmec_jax only consumes a subset of the full VMEC INDATA file; need to document which fields are supported/ignored, or validate inputs to warn when unsupported fields are present
- [ ] DESC can output Boozer coordinates directly, so with the right flag/argument it can handle both Stage 1 and Stage 2; the pipeline should support this shortcut path

## Stage 2 -- Boozer Transform

- [ ] Future boundary condition optimization can be added as additional functions in Stage 2

## Stage 3 -- Neoclassical

- [ ] sfincs/sfincs_jax and NEO_JAX do not have directly compatible inputs; same adapter/translation issue as Stage 1.
- [ ] NEO_JAX is fast, but its output can't be used for future stages. sfincs is slower, but more accurate.
- [ ] NEO_JAX is excluded from the MVP, but should be included in the final pipeline as an optional stage; its effective ripple output is valuable as a figure of merit even though it does not feed later stages

## Stage 4 -- Turbulence

- [ ] SPECTRAX-GK/GX, and GENE likely do not have directly compatible inputs; same adapter/translation issue as Stages 1 and 3

## W&B / Output Tracking

- [ ] Decide whether W&B dashboards are internal (maintainers only) or public-facing.
- [ ] Eventually it'll be a public challenge SDK. For the official submission: we don't need to worry about API keys. 

## Workflow Engine

- [ ] Container registry for external collaborators: where do external contributors host their alternative stage implementations? Maybe their own GHCR or Docker Hub, with the workflow engine pulling from there.
- [ ] How to expose the workflow engine to external users.
- [ ] How to validate that externally submitted containers are not a security threat.
- [ ] Parallelization strategy: when a stage's internal loop can be parallelized across threads or compute nodes, should Snakemake handle the distribution or should the stage software manage it internally?
	- [ ] Consider an intermediate orchestration layer between Snakemake and the stage software that handles parallelization.
