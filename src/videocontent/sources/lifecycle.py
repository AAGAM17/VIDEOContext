"""Temporary-media lifecycle.

Downloaded objects live under a confined temp root with:

* cleanup on success *and* failure (context manager, never orphan on exception)
* TTL sweeps (``sweep_stale``) so crashed runs cannot grow the disk forever
* a free-space pre-check so a 6 GB fetch onto a 1 GB disk fails fast, not halfway
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..logging import get_logger
from .types import VideoAsset, VideoSource

log = get_logger("sources.lifecycle")

TEMP_PREFIX = "vctx-source-"


def temp_root() -> Path:
    root = Path(tempfile.gettempdir()) / "videocontent-sources"
    root.mkdir(parents=True, exist_ok=True)
    return root


def free_bytes(path: Path) -> int | None:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def check_space(dest_dir: Path, needed: int | None) -> None:
    if needed is None:
        return
    free = free_bytes(dest_dir)
    if free is not None and needed > free:
        from ..errors import DownloadError

        raise DownloadError(
            f"not enough disk space: need ~{needed / 1e6:.0f} MB, "
            f"{free / 1e6:.0f} MB free in {dest_dir}",
            retryable=False,
            hint="Free disk space or raise temp capacity before ingesting remote media.",
        )


@contextmanager
def materialized(
    source: VideoSource, *, config: Any = None, workdir: Path | None = None,
) -> Iterator[VideoAsset]:
    """Resolve ``source`` to a local file, cleaning up temp downloads on exit.

    Local files pass through untouched (``asset.temporary`` is False, nothing is
    deleted). Remote downloads are removed unless ``sources.keep_downloads`` is set.
    """
    from ..config import ProcessingConfig
    from .adapters import adapter_for

    cfg = config if isinstance(config, ProcessingConfig) else None
    enabled = tuple(cfg.sources.enabled_sources) if cfg else ("local", "http")
    adapter = adapter_for(source.locator, enabled=enabled)

    if adapter.name == "local":
        asset = adapter.materialize(source, Path(source.metadata.get("path", ".")).parent
                                    if source.metadata.get("path") else Path.cwd())
        yield asset
        return

    # Remote: confined temp dir per asset, swept by TTL even if we crash elsewhere.
    keep = bool(cfg and cfg.sources.keep_downloads)
    directory = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=str(temp_root())))
    # Best-effort quota hint from inspection (may be None; failures here must not
    # block materialization — the download enforces limits while streaming).
    try:
        inspection = adapter.inspect(source, cfg)
        check_space(directory, inspection.estimated_size_bytes)
    except Exception as exc:
        log.debug("sources.quota_hint_skipped", extra={"error": str(exc)})
    asset = adapter.materialize(source, directory, cfg)
    try:
        yield asset
    finally:
        if not keep:
            try:
                adapter.cleanup(asset)
                shutil.rmtree(directory, ignore_errors=True)
            except Exception as exc:  # cleanup must never mask the run's own error
                log.warning("sources.cleanup_failed", extra={"error": str(exc)})


def sweep_stale(*, ttl_h: float = 24.0) -> int:
    """Delete temp download dirs older than ``ttl_h`` hours. Returns count removed."""
    root = temp_root()
    cutoff = time.time() - ttl_h * 3600
    removed = 0
    for child in root.glob(f"{TEMP_PREFIX}*"):
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        log.info("sources.swept", extra={"removed": removed})
    return removed


__all__ = ["materialized", "sweep_stale", "temp_root"]
