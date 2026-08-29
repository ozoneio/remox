"""Shared on-disk cache location.

Both vocal separation and media downloads are expensive enough to be worth
persisting between runs, and both key their entries on immutable inputs, so
the cache never needs invalidating.
"""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir(*parts: str) -> Path:
    """Per-user cache directory, optionally with sub-paths appended.

    Honours ``REMOX_CACHE_DIR`` first so tests and CI can redirect it.
    """
    override = os.environ.get("REMOX_CACHE_DIR")
    if override:
        base = Path(override)
    elif os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        base = Path(root) / "remox" / "cache"
    else:
        root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        base = Path(root) / "remox"
    return base.joinpath(*parts)
