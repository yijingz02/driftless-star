"""Cross-stage helper utilities for the StellaForge Snakemake workflow."""

from __future__ import annotations

import re
from pathlib import Path


def set_assignment(file_path: str | Path, key: str, value: str) -> None:
    """Rewrite the right-hand side of ``key = ...`` in ``file_path`` to ``value``.

    Used to push pipeline-derived paths and run identifiers into upstream
    config files (TOML/INI-style) for tools that do not expose CLI overrides.

    Parameters
    ----------
    file_path : str or Path
        Path to the config file. If the file does not exist, the call is a
        no-op.
    key : str
        The configuration key whose value should be replaced. Matched as a
        line-anchored literal (``key = ...``) so substring keys do not collide.
    value : str
        The replacement right-hand side, inserted verbatim (no quoting added).
        Callers are responsible for any quoting required by the target syntax.
    """
    path = Path(file_path)
    if not path.exists():
        return
    text = path.read_bytes().decode("utf-8")
    new_text = re.sub(
        rf"^({re.escape(key)}[ \t]*=[ \t]*)[^\r\n]*",
        lambda m: f"{m.group(1)}{value}",
        text,
        flags=re.MULTILINE,
    )
    if new_text != text:
        path.write_bytes(new_text.encode("utf-8"))
