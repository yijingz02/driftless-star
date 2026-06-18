"""Stage 5 (NEOPAX transport) workflow helpers.

NEOPAX is configured via a TOML file rather than CLI flags. :func:`prepare_neopax_config`
writes a path-resolved copy of the shared ``common_input`` template under the run's output
directory (leaving the committed template untouched); the Snakefile then runs NEOPAX on that
copy. Called at Snakefile parse time.
"""

from __future__ import annotations

from pathlib import Path

from .utils import apply_assignments


def prepare_neopax_config(
    *,
    s5_config_template: str,
    s5_resolved_config: str,
    s1_output: str,
    s2_output: str,
    s3_output: str,
    s4_output: str,
    s5_output_dir: str,
) -> None:
    """Write a path-resolved copy of the NEOPAX template for the current run.

    Writes a path-resolved copy of ``s5_config_template`` to ``s5_resolved_config``, with
    its five path fields rewritten relative to the copy's own directory (NEOPAX runs
    there). The committed template is never modified.

    Parameters
    ----------
    s5_config_template : str
        Path to the shared NEOPAX template (``inputs/<run>/common_input.toml``).
    s5_resolved_config : str
        Path of the resolved copy to write (under ``outputs/<run>/stage5_transport/``).
    s1_output, s2_output, s3_output, s4_output : str
        Paths to Stage 1-4 output artifacts referenced by NEOPAX.
    s5_output_dir : str
        Stage 5 output directory (where NEOPAX writes ``transport_solution.h5``).
    """
    resolved = Path(s5_resolved_config)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    base = resolved.parent.resolve()

    def _rel(p: str) -> str:
        return str(Path(p).resolve().relative_to(base, walk_up=True))

    assignments = {
        "vmec_file": f'"{_rel(s1_output)}"',
        "boozer_file": f'"{_rel(s2_output)}"',
        "neoclassical_file": f'"{_rel(s3_output)}"',
        "turbulence_file": f'"{_rel(s4_output)}"',
        "transport_output_dir": f'"{_rel(s5_output_dir)}/"',
    }
    template_text = Path(s5_config_template).read_bytes().decode("utf-8")
    resolved.write_bytes(apply_assignments(template_text, assignments).encode("utf-8"))
