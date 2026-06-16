"""Ouroboros: closed-loop driver (Stage 5 -> Stage 1 pressure feedback)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _resolve_paths(config: dict, run_name: str, repo_root: Path) -> dict[str, Path]:
    """Resolve the per-``run_name`` files the driver reads, seeds, or chains.

    Mirrors the Snakefile's ``{run_name}``-substitution of ``directories`` and
    ``filenames`` so the driver and Snakemake agree on every path for a run.

    Parameters
    ----------
    config : dict
        Parsed ``config.yaml`` (must contain ``directories`` and ``filenames``).
    run_name : str
        Run identifier used to fill ``{run_name}`` placeholders.
    repo_root : Path
        Repository root; resolved paths are returned absolute under it.

    Returns
    -------
    dict[str, Path]
        Absolute paths for ``s1_input`` (equilibrium input / feedback target),
        ``s3_config``/``s4_config`` (stage 3/4 templates to seed per iteration),
        ``s5_output`` (transport solution), and ``s5_signal`` (loop status).
    """
    dirs = config["directories"]
    names = config["filenames"]

    def _p(dir_key: str, name_key: str) -> Path:
        return (
            repo_root
            / dirs[dir_key].format(run_name=run_name)
            / names[name_key].format(run_name=run_name)
        )

    return {
        "s1_input": _p("stage1_input", "s1_input"),
        "s3_config": _p("stage3_input", "s3_config"),
        "s4_config": _p("stage4_input", "s4_config"),
        "s5_output": _p("stage5_output", "s5_output"),
        "s5_signal": _p("stage5_post_processing_output", "s5_signal"),
    }


def _seed(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` (creating ``dst``'s parent) to materialize a
    run-namespaced input file for an iteration."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_forward_pass(*, target: str, run_name: str, cores: int, config_path: Path, repo_root: Path) -> None:
    """Run one full Snakemake forward pass for ``run_name``, up to ``target``.

    ``--config run_name=...`` overrides the configfile, so each iteration writes
    into its own ``{run_name}`` output tree. ``target`` is the repo-relative
    signal file, so Snakemake runs the whole chain Stage 1 -> ... -> Stage 5 ->
    ``stage5_post_processing`` (the in-container fit + convergence check).

    Targeting the signal is deliberate: a bare ``snakemake`` builds the default
    target (``rule all`` = ``S5_OUTPUT``, the transport solution) and stops at
    Stage 5, so a plain forward pass never runs the fit or mutates a Stage 1
    input. Naming the signal -- which depends on ``S5_OUTPUT`` -- opts into the
    feedback step, building the full chain plus ``stage5_post_processing``.
    """
    logger.info("Forward pass [%s]: snakemake %s --cores %d", run_name, target, cores)
    subprocess.run(
        ["snakemake", target, "--cores", str(cores),
         "--configfile", str(config_path), "--config", f"run_name={run_name}"],
        cwd=repo_root,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop driver (Stage 5 -> Stage 1 pressure feedback)."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                        help="Pipeline config file (default: config.yaml).")
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
    base = config["run_name"]
    loop_dir = repo_root / "stages" / "loop-output" / base
    loop_dir.mkdir(parents=True, exist_ok=True)
    base_paths = _resolve_paths(config, base, repo_root)

    # Each iteration is an independent run '{base}_iter_N': the driver seeds that
    # run's inputs (the equilibrium input chained from the previous iteration's
    # fit; the stage 3/4 templates copied from the base), runs the pass, and the
    # in-container post-processing overwrites the equilibrium input with the new
    # fit -> the seed for the next iteration. Distinct '{base}_iter_N' output
    # trees mean stages 3/4 recompute every iteration instead of reusing a cache.
    prev_paths: dict[str, Path] | None = None
    n = 0
    for n in range(1, args.max_iters + 1):
        run_name = f"{base}_iter_{n}"
        logger.info("=== Iteration %d of %d (run_name=%s) ===", n, args.max_iters, run_name)
        p = _resolve_paths(config, run_name, repo_root)

        # Seed this iteration's run-namespaced inputs.
        src_s1 = base_paths["s1_input"] if n == 1 else prev_paths["s1_input"]
        _seed(src_s1, p["s1_input"])
        _seed(base_paths["s3_config"], p["s3_config"])
        _seed(base_paths["s4_config"], p["s4_config"])

        # Snapshot the input feeding this pass before the rule overwrites it.
        iter_dir = loop_dir / f"iter_{n}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p["s1_input"], iter_dir / p["s1_input"].name)

        # Lets the output dir point outside the repository directory (e.g. an HTCondor scratch filesystem).
        try:
            target = str(p["s5_signal"].relative_to(repo_root))
        except ValueError:
            target = str(p["s5_signal"])
        run_forward_pass(target=target, run_name=run_name, cores=args.cores,
                         config_path=config_path, repo_root=repo_root)

        shutil.copy2(p["s5_output"], iter_dir / p["s5_output"].name)
        prev_paths = p  # the rule overwrote p["s1_input"] with the new fit -> seeds n+1
        if json.loads(p["s5_signal"].read_text()).get("converged"):
            logger.info("Converged at iteration %d; stopping.", n)
            break
    logger.info("Loop finished: %d iteration(s) ('%s_iter_*'); snapshots in %s", n, base, loop_dir)


if __name__ == "__main__":
    main()
