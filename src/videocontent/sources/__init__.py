"""Universal source layer — ``Source → Adapter → Asset → pipeline``.

.. code-block:: python

    from videocontent.sources import resolve, inspect_source

    source = resolve("https://example.com/talk.mp4")
    info = inspect_source(source)   # no download
"""

from __future__ import annotations

from . import security
from .adapters import DirectURLAdapter, LocalFileAdapter, adapter_for
from .lifecycle import materialized, sweep_stale, temp_root
from .resolve import inspect_source, resolve
from .types import (
    SUPPORTED_SOURCE_TYPES,
    MediaCapabilities,
    SourceInspection,
    SourceType,
    VideoAsset,
    VideoSource,
    stable_id,
)

__all__ = [
    "SUPPORTED_SOURCE_TYPES",
    "DirectURLAdapter",
    "LocalFileAdapter",
    "MediaCapabilities",
    "SourceInspection",
    "SourceType",
    "VideoAsset",
    "VideoSource",
    "adapter_for",
    "inspect_source",
    "materialized",
    "resolve",
    "security",
    "stable_id",
    "sweep_stale",
    "temp_root",
]
