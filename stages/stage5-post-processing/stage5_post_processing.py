"""Stage 5 Post-Processing: emit the closed-loop convergence signal.

Runs in Stage 5's container immediately after the pressure fit has written the
new Stage 1 input. It decides whether the loop has converged ("Stage 5 output
unchanged") and writes that verdict to a small JSON signal file. The external
loop driver reads this file to decide whether to run another forward pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def converged(transport: Path) -> bool:
    """Return whether the transport solution has stopped changing between passes.
    """
    return False #todo: Update with criteria.


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 Post-Processing: write the closed-loop convergence signal.")
    parser.add_argument("--transport", type=Path, required=True,
                        help="This pass's transport_solution.h5.")
    parser.add_argument("--signal", type=Path, required=True,
                        help="Output path for the convergence-status JSON the driver reads.")
    args = parser.parse_args()

    status = {"converged": converged(args.transport)}
    args.signal.parent.mkdir(parents=True, exist_ok=True)
    args.signal.write_text(json.dumps(status) + "\n")
    print(f"# converge_status: {status}")
    print(f"# wrote_signal: {args.signal}")


if __name__ == "__main__":
    main()
