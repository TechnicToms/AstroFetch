"""Disposable local cache for reprojected layer windows.

Never load-bearing (AGENTS rule 4): every entry is re-fetchable from the archive
and safe to delete at any time. A key is a hash of the layer id and the exact
target grid, so an identical request skips the network on a warm cache while any
change to bbox, size, or CRS misses cleanly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from astrofetch.data.grid import TargetGrid


def default_cache_dir() -> Path:
    """Cache root: ``$ASTROFETCH_CACHE`` if set, else an XDG cache subdir."""
    override = os.environ.get("ASTROFETCH_CACHE")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(base) / "astrofetch"


class WindowCache:
    """Filesystem cache of ``(image, mask)`` arrays keyed by layer and grid.

    Args:
        root: cache directory; defaults to :func:`default_cache_dir`.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()

    def _key(self, layer: str, grid: TargetGrid) -> str:
        west, south, east, north = grid.bbox
        raw = (
            f"{layer}|{grid.crs}|{grid.width}x{grid.height}"
            f"|{west:.8f},{south:.8f},{east:.8f},{north:.8f}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, layer: str, grid: TargetGrid) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the cached ``(image, mask)`` for ``layer``/``grid``, or None."""
        path = self._path(self._key(layer, grid))
        if not path.exists():
            return None
        with np.load(path) as data:
            return data["image"], data["mask"]

    def put(self, layer: str, grid: TargetGrid, image: np.ndarray, mask: np.ndarray) -> None:
        """Store ``(image, mask)`` for ``layer``/``grid``, replacing any entry."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(self._key(layer, grid))
        # Write to a temp file then atomically replace, so an interrupted write
        # never leaves a half-written cache entry that a later read would trust.
        tmp = self.root / f"{path.stem}.{os.getpid()}.tmp.npz"
        np.savez(tmp, image=image, mask=mask)
        os.replace(tmp, path)

    def clear(self) -> None:
        """Delete every cache entry. The cache is disposable; this is always safe."""
        if not self.root.exists():
            return
        for entry in self.root.glob("*.npz"):
            entry.unlink()
