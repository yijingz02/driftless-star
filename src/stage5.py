"""Stage 5 (NEOPAX transport) workflow helpers.

NEOPAX is configured via a TOML file rather than CLI flags, so the Snakefile
must keep the TOML's path fields aligned with the actual artifacts produced by
upstream stages and with the run-namespaced output directory. These rewrites
happen at Snakefile parse time via :func:`prepare_neopax_config`.
"""

from __future__ import annotations

from pathlib import Path

from .utils import set_assignment


def prepare_neopax_config(
    *,
    s5_config: str,
    s5_output_dir: str,
    s1_output: str,
    s2_output: str,
    s3_output: str,
    s4_output: str,
) -> None:
    """Rewrite NEOPAX's TOML path fields so they match the current pipeline run.

    Updates the five fields NEOPAX reads to locate inputs and write outputs:

    - ``vmec_file``, ``boozer_file``, ``neoclassical_file``, ``turbulence_file``
      are set to the relative paths of the corresponding Stage 1-4 output
      artifacts.
    - ``transport_output_dir`` is set to the relative path of the Stage 5 output
      directory (with a trailing slash, matching NEOPAX's convention).

    All paths are written relative to the directory containing the TOML, since
    NEOPAX is launched with that directory as CWD.

    Parameters
    ----------
    s5_config : str
        Path to the Stage 5 NEOPAX TOML config to rewrite.
    s5_output_dir : str
        Stage 5 output directory (already ``{run_name}``-substituted).
    s1_output, s2_output, s3_output, s4_output : str
        Paths to Stage 1-4 output artifacts referenced by NEOPAX.
    """
    toml_dir = Path(s5_config).parent.resolve()

    def _rel(p: str) -> str:
        return str(Path(p).resolve().relative_to(toml_dir, walk_up=True))

    set_assignment(s5_config, "vmec_file",            f'"{_rel(s1_output)}"')
    set_assignment(s5_config, "boozer_file",          f'"{_rel(s2_output)}"')
    set_assignment(s5_config, "neoclassical_file",    f'"{_rel(s3_output)}"')
    set_assignment(s5_config, "turbulence_file",      f'"{_rel(s4_output)}"')
    set_assignment(s5_config, "transport_output_dir", f'"{_rel(s5_output_dir)}/"')
