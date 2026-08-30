"""Audio extraction and analysis.

16 kHz mono PCM WAV is the canonical ASR input (what Whisper-family models want), produced
in a single FFmpeg pass. Silence detection comes from FFmpeg's ``silencedetect`` rather than
a Python DSP loop — it is one pass over the audio and costs nothing extra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger
from . import ffmpeg

log = get_logger("media.audio")

ASR_SAMPLE_RATE = 16000


@dataclass(frozen=True, slots=True)
class SilenceSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def extract_audio(
    source: str | Path,
    out_path: str | Path,
    *,
    sample_rate: int = ASR_SAMPLE_RATE,
    timeout: float = 1800.0,
    normalize: bool = False,
) -> Path | None:
    """Extract mono PCM WAV. Returns ``None`` when the input has no audio stream."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    filters = ["aresample=async=1:first_pts=0"]
    if normalize:
        # Loudness normalization helps quiet lecture recordings; off by default because it
        # changes the signal the model sees and costs another filter pass.
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    args = [
        *ffmpeg.input_args(source),
        "-vn", "-sn", "-dn",
        "-map", "0:a:0?",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-af", ",".join(filters),
        str(out),
    ]
    ffmpeg.run(args, timeout=timeout)
    if not out.exists() or out.stat().st_size <= 44:  # 44 = WAV header only
        log.info("audio.absent", extra={"reason": "no_audio_stream"})
        out.unlink(missing_ok=True)
        return None
    log.info("audio.extracted", extra={"size_kb": out.stat().st_size // 1024, "sr": sample_rate})
    return out


_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silence(
    source: str | Path,
    *,
    noise_db: float = -35.0,
    min_duration: float = 0.8,
    timeout: float = 900.0,
) -> list[SilenceSpan]:
    """Find silent spans — the basis for ``speaker_started`` / ``silence_started`` events."""
    args = [
        *ffmpeg.input_args(source),
        "-vn",
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    result = ffmpeg.run(args, timeout=timeout, check=False, loglevel="info")
    # silencedetect logs at info level, so read both streams regardless of loglevel.
    text = f"{result.stderr}\n{result.stdout}"

    spans: list[SilenceSpan] = []
    pending: float | None = None
    for line in text.splitlines():
        if (m := _SILENCE_START.search(line)) is not None:
            pending = max(0.0, float(m.group(1)))
        elif (m := _SILENCE_END.search(line)) is not None:
            end = float(m.group(1))
            start = pending if pending is not None else max(0.0, end - min_duration)
            if end > start:
                spans.append(SilenceSpan(round(start, 3), round(end, 3)))
            pending = None
    log.debug("audio.silence", extra={"spans": len(spans)})
    return spans


def speech_spans(
    silences: list[SilenceSpan], duration: float, *, min_duration: float = 0.3
) -> list[tuple[float, float]]:
    """Complement of the silence spans: where sound (usually speech) is present."""
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for silence in sorted(silences, key=lambda s: s.start):
        if silence.start - cursor >= min_duration:
            spans.append((round(cursor, 3), round(min(silence.start, duration), 3)))
        cursor = max(cursor, silence.end)
    if duration - cursor >= min_duration:
        spans.append((round(cursor, 3), round(duration, 3)))
    return spans


def extract_subtitles(
    source: str | Path, out_path: str | Path, *, stream_index: int = 0, timeout: float = 300.0
) -> Path | None:
    """Extract an embedded subtitle track to SRT — the ASR fallback when no model exists."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = [*ffmpeg.input_args(source), "-map", f"0:s:{stream_index}?", "-c:s", "srt", str(out)]
    try:
        ffmpeg.run(args, timeout=timeout)
    except Exception as exc:
        log.debug("subtitles.absent", extra={"error": type(exc).__name__})
        out.unlink(missing_ok=True)
        return None
    if not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        return None
    return out


__all__ = [
    "ASR_SAMPLE_RATE",
    "SilenceSpan",
    "detect_silence",
    "extract_audio",
    "extract_subtitles",
    "speech_spans",
]
