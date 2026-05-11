# Command-line driver for the Stage 2 Boozer transform.

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stage 2 Boozer transform from a VMEC wout file.")
    parser.add_argument("--wout", required=True, type=Path, help="Path to the Stage 1 VMEC wout NetCDF file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to the output boozmn NetCDF file.")
    return parser


def run_boozer_transform(wout_path: Path, output_path: Path) -> None:
    if not wout_path.is_file():
        raise FileNotFoundError(f"Stage 2 input wout file does not exist: {wout_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import booz_xform_jax as bx

    booz = bx.Booz_xform()
    logger.info("Reading wout file: %s", wout_path)
    booz.read_wout(wout_path)
    logger.info("Running Boozer transform")
    booz.run()
    logger.info("Writing boozmn file: %s", output_path)
    booz.write_boozmn(output_path)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    try:
        run_boozer_transform(args.wout.resolve(), args.output.resolve())
    except Exception:
        logger.exception("Stage 2 Boozer transform failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
