"""Ingestion: validate an input file and probe it into a :class:`VideoInfo`.

Everything the rest of the pipeline knows about the media comes from here, and it comes from
``ffprobe`` — never from the file extension. A ``.mp4`` that is really a Matroska file, or a
``.txt`` that is really an MP4, both behave correctly because identity is established by
decoding the container.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..config import LimitsConfig
from ..errors import (
    CorruptMediaError,
    MediaTooLargeError,
    UnsupportedMediaError,
)
from ..logging import get_logger
from ..schema.v1 import Chapter, SubtitleTrack, VideoInfo
from . import ffmpeg

log = get_logger("media.probe")

HASH_CHUNK = 4 * 1024 * 1024


def content_hash(path: str | os.PathLike[str], *, chunk: int = HASH_CHUNK) -> str:
    """``sha256:…`` over the file's bytes — the identity used for caching.

    Streamed in chunks so an 8 GB file does not become an 8 GB allocation. Path and mtime
    are deliberately excluded: two copies of the same video must share cache entries.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _rational(value: str | None) -> float | None:
    """FFprobe reports frame rates as ``'30000/1001'``."""
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _num(value, cast=float, default=None):
    if value in (None, "", "N/A"):
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def raw_probe(source: str | os.PathLike[str], *, timeout: float = 60.0) -> dict:
    """Run ``ffprobe -print_format json`` and return the parsed payload."""
    args = [
        "-print_format", "json",
        "-show_format", "-show_streams", "-show_chapters",
        *ffmpeg.input_args(source),
    ]
    result = ffmpeg.run(args, binary="ffprobe", timeout=timeout, check=False)
    if result.returncode != 0:
        raise CorruptMediaError(
            f"ffprobe could not read {Path(str(source)).name}",
            hint=(result.stderr.strip().splitlines() or ["unknown ffprobe failure"])[-1],
        )
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise CorruptMediaError(f"ffprobe returned invalid JSON for {source}") from exc


def probe(
    source: str | os.PathLike[str],
    *,
    limits: LimitsConfig | None = None,
    compute_hash: bool = True,
) -> VideoInfo:
    """Validate and describe a media file.

    Raises :class:`UnsupportedMediaError`, :class:`CorruptMediaError` or
    :class:`MediaTooLargeError` — all with actionable hints — before any expensive work.
    """
    limits = limits or LimitsConfig()
    remote = ffmpeg.is_remote(source)
    path = Path(str(source))

    size_bytes: int | None = None
    if not remote:
        if not path.exists():
            raise UnsupportedMediaError(
                f"no such file: {path}", hint="Check the path, or pass an http(s) URL."
            )
        if path.is_dir():
            raise UnsupportedMediaError(f"{path} is a directory, not a video file")
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise CorruptMediaError(f"{path.name} is empty (0 bytes)")
        max_bytes = limits.max_file_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise MediaTooLargeError(
                f"{path.name} is {size_bytes / 1e6:.0f} MB, above the "
                f"{limits.max_file_size_mb} MB limit",
                hint="Raise limits.max_file_size_mb, or split the file.",
            )

    payload = raw_probe(source, timeout=limits.probe_timeout_s)
    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []
    if not streams:
        raise CorruptMediaError(
            f"{path.name} contains no decodable streams",
            hint="The container may be truncated. Verify with: ffmpeg -v error -i FILE -f null -",
        )

    container = fmt.get("format_name")
    if container and limits.allowed_containers:
        allowed = {c.strip() for c in limits.allowed_containers}
        if container not in allowed and not (allowed & set(container.split(","))):
            raise UnsupportedMediaError(
                f"container {container!r} is not allowed",
                hint=f"Permitted containers: {', '.join(sorted(allowed))}",
            )

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

    # Cover art is a 1-frame video stream; it is not a video track.
    real_video = [
        s for s in video_streams
        if s.get("disposition", {}).get("attached_pic", 0) != 1
    ]
    vstream = real_video[0] if real_video else None
    astream = audio_streams[0] if audio_streams else None

    duration = _num(fmt.get("duration"), float)
    if duration is None and vstream is not None:
        duration = _num(vstream.get("duration"), float)
    if duration is None:
        duration = 0.0

    if duration > limits.max_duration_s:
        raise MediaTooLargeError(
            f"duration {duration / 60:.1f} min exceeds the "
            f"{limits.max_duration_s / 60:.0f} min limit",
            hint="Raise limits.max_duration_s, or process a clip.",
        )
    if not vstream and not astream:
        raise CorruptMediaError(f"{path.name} has neither a video nor an audio stream")

    fps = None
    frame_count = None
    if vstream is not None:
        fps = _rational(vstream.get("avg_frame_rate")) or _rational(vstream.get("r_frame_rate"))
        frame_count = _num(vstream.get("nb_frames"), int)
        if frame_count is None and fps and duration:
            frame_count = round(fps * duration)

    info = VideoInfo(
        id=_short_id(str(source)),
        filename=path.name,
        path=None if remote else str(path.resolve()),
        content_hash=(
            content_hash(path) if (compute_hash and not remote) else None
        ),
        size_bytes=size_bytes,
        duration=round(duration, 3),
        container=container,
        fps=round(fps, 4) if fps else None,
        width=_num(vstream.get("width"), int) if vstream else None,
        height=_num(vstream.get("height"), int) if vstream else None,
        frame_count=frame_count,
        bitrate=_num(fmt.get("bit_rate"), int),
        video_codec=vstream.get("codec_name") if vstream else None,
        audio_codec=astream.get("codec_name") if astream else None,
        audio_channels=_num(astream.get("channels"), int) if astream else None,
        audio_sample_rate=_num(astream.get("sample_rate"), int) if astream else None,
        has_audio=astream is not None,
        has_video=vstream is not None,
        subtitle_tracks=[
            SubtitleTrack(
                index=_num(s.get("index"), int, 0) or 0,
                language=(s.get("tags") or {}).get("language"),
                codec=s.get("codec_name"),
                title=(s.get("tags") or {}).get("title"),
            )
            for s in sub_streams
        ],
        chapters=[
            Chapter(
                start=_num(c.get("start_time"), float, 0.0) or 0.0,
                end=_num(c.get("end_time"), float, 0.0) or 0.0,
                title=(c.get("tags") or {}).get("title"),
            )
            for c in (payload.get("chapters") or [])
        ],
        tags={
            k: str(v) for k, v in (fmt.get("tags") or {}).items()
            if k in {"title", "artist", "comment", "encoder", "creation_time", "language"}
        },
    )

    log.info(
        "probe.ok",
        extra={
            "duration_s": info.duration, "fps": info.fps,
            "resolution": f"{info.width}x{info.height}" if info.width else None,
            "has_audio": info.has_audio, "container": info.container,
        },
    )
    return info


def _short_id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def verify_decodable(source: str | os.PathLike[str], *, timeout: float = 120.0) -> list[str]:
    """Decode the whole file to null, returning FFmpeg's complaints.

    Used by ``videocontent doctor`` and the corrupt-media tests; too slow for the default
    path, which relies on probe plus per-stage error handling instead.
    """
    result = ffmpeg.run(
        [*ffmpeg.input_args(source), "-f", "null", "-"],
        timeout=timeout, check=False,
    )
    return [line for line in result.stderr.splitlines() if line.strip()]


__all__ = ["content_hash", "probe", "raw_probe", "verify_decodable"]
