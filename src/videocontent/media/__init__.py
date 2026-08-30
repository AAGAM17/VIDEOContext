"""Media layer — the only place FFmpeg is invoked.

    from videocontent.media import probe, extract_audio, extract_plan, Workspace
"""

from __future__ import annotations

from . import ffmpeg
from .audio import (
    ASR_SAMPLE_RATE,
    SilenceSpan,
    detect_silence,
    extract_audio,
    extract_subtitles,
    speech_spans,
)
from .frames import SamplePlan, SampleWindow, extract_plan, extract_single, merge_windows
from .probe import content_hash, probe, raw_probe, verify_decodable
from .workspace import Workspace, sanitize, scratch

__all__ = [
    "ASR_SAMPLE_RATE",
    "SamplePlan",
    "SampleWindow",
    "SilenceSpan",
    "Workspace",
    "content_hash",
    "detect_silence",
    "extract_audio",
    "extract_plan",
    "extract_single",
    "extract_subtitles",
    "ffmpeg",
    "merge_windows",
    "probe",
    "raw_probe",
    "sanitize",
    "scratch",
    "speech_spans",
    "verify_decodable",
]
