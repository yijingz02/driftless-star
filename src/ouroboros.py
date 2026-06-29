"""Ouroboros: closed-loop driver (Stage 5 -> Stage 1 pressure feedback).

Runs the forward pass repeatedly, feeding each iteration's evolved Stage 1 boundary
into the next. Every iteration is self-contained under ``<output_dir>/loop/iter_N/``:
``input/`` holds that pass's seeded inputs and ``output/`` holds all of its stage
outputs (including the evolved boundary and the convergence signal). The committed
inputs under ``input_dir`` are only ever read, never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import yaml

from .utils import resolve_pipeline_paths

logger = logging.getLogger(__name__)


def _abs(repo_root: Path, path: str) -> Path:
    """Resolve a pipeline path against ``repo_root`` (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _seed(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst``, creating ``dst``'s parent, to materialize an iteration input."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _seed_iteration_inputs(
    *,
    repo_root: Path,
    base_p: dict[str, str],
    iter_p: dict[str, str],
    s1_source: str,
    config_path: Path,
) -> None:
    """Populate one iteration's ``input/`` directory.

    The Stage 3/4/5 configs are copied unchanged from the base ``input_dir`` every
    iteration, the run config is copied alongside them for reproducibility, and the
    Stage 1 boundary comes from ``s1_source`` (the base boundary on iteration 1, the
    previous iteration's evolved boundary thereafter).

    Parameters
    ----------
    repo_root : Path
        Repository root; pipeline paths are resolved against it.
    base_p, iter_p : dict[str, str]
        ``resolve_pipeline_paths`` output for the base config and for this iteration.
    s1_source : str
        Path to the Stage 1 boundary to seed as this iteration's ``s1_input``.
    config_path : Path
        Run config file, copied into the iteration input dir as a record.
    """
    for key in ("s3_config", "s4_config", "s5_config"):
        _seed(_abs(repo_root, base_p[key]), _abs(repo_root, iter_p[key]))
    _seed(config_path, _abs(repo_root, iter_p["input_dir"]) / config_path.name)
    _seed(_abs(repo_root, s1_source), _abs(repo_root, iter_p["s1_input"]))


def run_forward_pass(
    *,
    target: str,
    input_dir: str,
    output_dir: str,
    cores: int,
    config_path: Path,
    repo_root: Path,
) -> None:
    """
    Run one Snakemake forward pass for an iteration
    """
    logger.info("Forward pass [%s]: snakemake %s --cores %d", output_dir, target, cores)
    subprocess.run(
        ["snakemake", target, "--cores", str(cores),
         "--configfile", str(config_path),
         "--config", f"input_dir={input_dir}", f"output_dir={output_dir}"],
        cwd=repo_root,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop driver (Stage 5 -> Stage 1 pressure feedback)."
    )
    parser.add_argument("--config", type=Path, default=Path("inputs/quick_run/config.yaml"),
                        help="Pipeline run config (default: inputs/quick_run/config.yaml).")
    parser.add_argument("--max-iters", type=int, default=3,
                        help="Number of iterations (independent forward passes) to run (default: 3).")
    parser.add_argument("--cores", type=int, default=4,
                        help="Cores passed to 'snakemake --cores' (default: 4).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.max_iters < 1:
        raise ValueError(f"--max-iters must be >= 1, got {args.max_iters}")

    repo_root = Path(__file__).resolve().parent.parent
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = yaml.safe_load(config_path.read_text())
    base_out = config["output_dir"]
    base_p = resolve_pipeline_paths(config)

    # Each iteration is an independent forward pass under '<output_dir>/loop/iter_N/'.
    # The Stage 1 boundary comes from the previous iteration's evolved boundary. The
    # other inputs are reseeded from the base each time. Distinct iter_N trees mean every
    # stage recomputes each iteration instead of reusing a cache.
    prev_p: dict[str, str] | None = None
    n = 0
    for n in range(1, args.max_iters + 1):
        iter_in = f"{base_out}/loop/iter_{n}/input"
        iter_out = f"{base_out}/loop/iter_{n}/output"
        logger.info("=== Iteration %d of %d (output=%s) ===", n, args.max_iters, iter_out)
        iter_p = resolve_pipeline_paths(config, input_dir=iter_in, output_dir=iter_out)

        s1_source = base_p["s1_input"] if n == 1 else prev_p["s1_feedback"]
        _seed_iteration_inputs(repo_root=repo_root, base_p=base_p, iter_p=iter_p,
                               s1_source=s1_source, config_path=config_path)

        run_forward_pass(target=iter_p["s5_signal"], input_dir=iter_in, output_dir=iter_out,
                         cores=args.cores, config_path=config_path, repo_root=repo_root)

        prev_p = iter_p  # post-processing wrote iter_p['s1_feedback']; it seeds iteration n+1
        signal = json.loads(_abs(repo_root, iter_p["s5_signal"]).read_text())
        if signal.get("halt"):
            logger.warning("Iteration %d: Stage 5 signalled a halt (pressure not sustained); "
                           "stopping. Restart from different initial conditions.", n)
            break
        if signal.get("converged"):
            logger.info("Converged at iteration %d; stopping.", n)
            break
    logger.info("Loop finished after %d iteration(s); records under %s/loop/", n, base_out)


if __name__ == "__main__":
    main()
