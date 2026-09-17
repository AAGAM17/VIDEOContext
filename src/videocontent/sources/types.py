"""Video source model — the vocabulary for "where did this video come from?".

A :class:`VideoSource` describes an *unresolved reference* (a path the user typed, a URL).
A :class:`VideoAsset` describes *media ready to process* (a local file plus provenance).

The processing pipeline only ever sees the asset's ``local_path``; everything about
acquisition lives behind the :class:`~videocontent.interfaces.SourceAdapter` protocol
so a new provider (S3, YouTube, catalog, …) can be added without touching the engine.

Only two source types are *supported* today: ``local_file`` and ``direct_url``.
The remaining :class:`SourceType` members reserve the taxonomy so future adapters do
not require a rewrite — resolving one raises :class:`UnsupportedSourceError` with a
clear message rather than silently misbehaving.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    LOCAL_FILE = "local_file"
    DIRECT_URL = "direct_url"
    PUBLIC_VIDEO = "public_video"
    OBJECT_STORAGE = "object_storage"
    CATALOG_REFERENCE = "catalog_reference"
    VCTX = "vctx"
    STREAM = "stream"
    LIVE_SOURCE = "live_source"
    CUSTOM = "custom"


SUPPORTED_SOURCE_TYPES = (SourceType.LOCAL_FILE, SourceType.DIRECT_URL)


class MediaCapabilities(BaseModel):
    """What an adapter can do for a given source — the engine branches on these."""

    model_config = ConfigDict(extra="ignore")

    supports_metadata: bool = True
    supports_seek: bool = True
    supports_streaming: bool = False
    supports_partial_download: bool = False
    supports_range_requests: bool = False
    supports_subtitles: bool = False
    supports_audio: bool = True
    supports_video: bool = True
    requires_authentication: bool = False
    supports_live: bool = False
    supports_collections: bool = False


class VideoSource(BaseModel):
    """An unresolved reference to video media."""

    model_config = ConfigDict(extra="allow")

    source_id: str = Field(description="Stable id: 'src_' + 12 hex chars of canonical id.")
    source_type: SourceType
    provider: str = Field(description="Adapter name that owns this source, e.g. 'local'.")
    locator: str = Field(description="Original user-supplied reference (redacted for display).")
    canonical_id: str = Field(description="Dedup identity; never contains secrets.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    capabilities: MediaCapabilities = Field(default_factory=MediaCapabilities)


class VideoAsset(BaseModel):
    """Normalized media ready for the pipeline — source-independent by construction."""

    model_config = ConfigDict(extra="allow")

    asset_id: str
    source: VideoSource
    local_path: str = Field(description="Local file the pipeline should decode.")
    temporary: bool = Field(default=False, description="True when lifecycle owns cleanup.")
    size_bytes: int | None = None
    content_hash: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    media_metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SourceInspection(BaseModel):
    """Answer to 'what is this, can we process it?' — without downloading the object."""

    model_config = ConfigDict(extra="allow")

    source: VideoSource
    accessible: bool
    reason: str | None = None
    media: dict[str, Any] = Field(default_factory=dict)
    capabilities: MediaCapabilities = Field(default_factory=MediaCapabilities)
    requires_authentication: bool = False
    estimated_size_bytes: int | None = None
    warnings: list[str] = Field(default_factory=list)


def stable_id(prefix: str, canonical: str) -> str:
    return f"{prefix}_{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"


__all__ = [
    "SUPPORTED_SOURCE_TYPES",
    "MediaCapabilities",
    "SourceInspection",
    "SourceType",
    "VideoAsset",
    "VideoSource",
    "stable_id",
]
