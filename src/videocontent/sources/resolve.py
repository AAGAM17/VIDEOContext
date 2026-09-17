"""Source resolution: strings in, :class:`VideoSource` out.

``resolve()`` is the single entry point the SDK, CLI, API and MCP share — the
security boundary (prompt §82) lives here, not duplicated in each surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import SourceError, SourceNotFoundError, UnsupportedSourceError
from .adapters import adapter_for
from .types import VideoSource


def resolve(locator: str | Path | VideoSource, *, config: Any = None) -> VideoSource:
    """Normalize any accepted reference into a :class:`VideoSource`.

    Accepts an existing :class:`VideoSource` (returned as-is), a local path, a
    ``file://`` URL, or an ``http(s)`` URL. Anything else raises a structured
    :class:`SourceError` with an actionable hint.
    """
    from ..config import ProcessingConfig

    if isinstance(locator, VideoSource):
        return locator
    text = str(locator).strip()
    if not text:
        raise SourceError("empty video source", code="INVALID_SOURCE",
                          hint="Pass a file path or an http(s) URL.")
    cfg = config if isinstance(config, ProcessingConfig) else None
    enabled = tuple(cfg.sources.enabled_sources) if cfg else ("local", "http")
    adapter = adapter_for(text, enabled=enabled)
    try:
        return adapter.to_source(text)
    except (SourceError, UnsupportedSourceError, SourceNotFoundError):
        raise
    except Exception as exc:
        raise SourceError(f"could not resolve source {text!r}: {exc}",
                          code="INVALID_SOURCE") from exc


def inspect_source(locator: str | Path | VideoSource, *, config: Any = None) -> Any:
    """Describe a source without downloading it (metadata, capabilities, access)."""
    from ..config import ProcessingConfig

    source = resolve(locator, config=config)
    cfg = config if isinstance(config, ProcessingConfig) else None
    enabled = tuple(cfg.sources.enabled_sources) if cfg else ("local", "http")
    adapter = adapter_for(source.locator, enabled=enabled)
    return adapter.inspect(source, cfg)


__all__ = ["inspect_source", "resolve"]
