"""The Video Context Format (``.vctx``).

Import surface for the schema layer::

    from videocontent.schema import VideoContextDocument, load, save, validate
"""

from __future__ import annotations

from .io import dumps, json_schema, load, loads, migrate, save
from .v1 import (
    VCTX_MEDIA_TYPE,
    VCTX_VERSION,
    BBox,
    Chapter,
    DetectedObject,
    EmbeddingIndex,
    Event,
    EventType,
    Frame,
    Metrics,
    ObjectInstance,
    OCRText,
    Producer,
    Scene,
    Segment,
    StageRecord,
    StageStatus,
    SubtitleTrack,
    TimeSpan,
    Utterance,
    VideoContextDocument,
    VideoInfo,
    VisionNote,
    Word,
)
from .validate import Finding, is_valid, validate

__all__ = [
    "VCTX_MEDIA_TYPE",
    "VCTX_VERSION",
    "BBox",
    "Chapter",
    "DetectedObject",
    "EmbeddingIndex",
    "Event",
    "EventType",
    "Finding",
    "Frame",
    "Metrics",
    "OCRText",
    "ObjectInstance",
    "Producer",
    "Scene",
    "Segment",
    "StageRecord",
    "StageStatus",
    "SubtitleTrack",
    "TimeSpan",
    "Utterance",
    "VideoContextDocument",
    "VideoInfo",
    "VisionNote",
    "Word",
    "dumps",
    "is_valid",
    "json_schema",
    "load",
    "loads",
    "migrate",
    "save",
    "validate",
]
