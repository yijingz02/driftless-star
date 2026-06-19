"""Stage 5 Post-Processing: emit the closed-loop convergence signal.

Runs in Stage 5's container immediately after the pressure fit has written the
new Stage 1 input. It decides whether the loop has converged (the transport
pressure profile has reached steady state between the initial and final time
slices) and writes that verdict to a small JSON signal file. The external loop
driver reads this file to decide whether to run another forward pass.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

# Reuse the stage's pressure loader (sibling script in the same Stage 5 container).
from fit_vmec_pressure_from_transport_h5 import _load_total_pressure

logger = logging.getLogger(__name__)


def pressure_converged(transport: Path, *, rel_tol: float) -> bool:
    """Return whether Stage 5's transport evolution has reached steady state.

    Compares the total pressure profile `P(rho)` at the final time slice of `transport`
    against the initial time slice. A small final-vs-initial change means the transport
    evolution barely moved the input profile, so the boundary fed back to Stage 1 is
    self-consistent. 

    Parameters
    ----------
    transport : Path
        This pass's ``transport_solution.h5`` (species-resolved ``pressure`` or
        ``temperature`` and ``density``, plus ``rho``).
    rel_tol : float
        Relative RMS tolerance; convergence requires the relative change below it.

    Returns
    -------
    bool
        ``True`` if the relative RMS change is below ``rel_tol``, else ``False``.

    Notes
    -----
    If the solution has fewer than two distinct time slices (a static profile, or a
    single saved step) the initial and final slices coincide, so convergence cannot
    be assessed; this returns ``False`` with a warning so the loop never stops on a
    non-evolving profile.
    """
    _, p_initial, idx_initial = _load_total_pressure(transport, time_index=0, final_time=False)
    _, p_final, idx_final = _load_total_pressure(transport, time_index=-1, final_time=True)

    if idx_initial == idx_final:
        logger.warning(
            "%s has fewer than two distinct time slices (initial index %s == final index %s); "
            "cannot assess convergence, reporting not converged.",
            transport, idx_initial, idx_final,
        )
        return False

    initial_norm = float(np.linalg.norm(p_initial))
    if initial_norm == 0.0:
        logger.warning(
            "%s has a zero-norm initial pressure profile; cannot assess convergence, "
            "reporting not converged.",
            transport,
        )
        return False

    rel_change = float(np.linalg.norm(p_final - p_initial)) / initial_norm
    logger.info("pressure relative RMS change = %.3e (rel_tol = %.3e)", rel_change, rel_tol)
    return rel_change < rel_tol


# --- Alternative Convergence Criteria ---
def return_converged_false(transport: Path) -> bool:
    """Always report not converged (placeholder criterion, kept as an alternative)."""
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 Post-Processing: write the closed-loop convergence signal.")
    parser.add_argument("--transport", type=Path, required=True,
                        help="This pass's transport_solution.h5.")
    parser.add_argument("--signal", type=Path, required=True,
                        help="Output path for the convergence-status JSON the driver reads.")
    parser.add_argument("--pressure-rel-tol", type=float, required=True,
                        help="Relative RMS tolerance on the final-vs-initial total pressure profile.")
    args = parser.parse_args()

    status = {"converged": pressure_converged(args.transport, rel_tol=args.pressure_rel_tol)}
    args.signal.parent.mkdir(parents=True, exist_ok=True)
    args.signal.write_text(json.dumps(status) + "\n")
    print(f"# converge_status: {status}")
    print(f"# wrote_signal: {args.signal}")


if __name__ == "__main__":
    main()
