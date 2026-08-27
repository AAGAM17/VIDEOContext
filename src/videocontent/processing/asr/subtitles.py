"""Subtitles as a transcript source.

A video that ships an embedded subtitle track already contains a human-checked, precisely
timed transcript — better evidence than anything a speech model would produce from the same
audio, and free. So subtitles are not only the documented fallback for "no ASR model
installed" (ARCHITECTURE §7): where a track exists, this engine is the *better* choice.

**Why this engine takes a video path.** The :class:`~videocontent.interfaces.ASREngine`
protocol hands an engine the extracted 16 kHz WAV, which by construction carries no subtitle
stream. Rather than widen the protocol for one engine, the caller passes the container
through ``source=``; the ``transcribe`` argument is then only a fallback, and a caller that
hands over a bare WAV gets an empty result with a warning that says why.

No new dependency: the two formats that matter differ from each other in the decimal
separator and a handful of inline tags, which is less code than a parser library's import
line. Both are treated as untrusted input — cue text is never interpreted, only stripped.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from ...config import ASRConfig
from ...interfaces import ASROutput, FrameContext
from ...logging import get_logger
from ...media.audio import extract_subtitles
from ...media.ffmpeg import available as ffmpeg_available
from .normalize import finalize, utterance

log = get_logger("asr.subtitles")

SUBTITLE_SUFFIXES = frozenset({".srt", ".vtt", ".webvtt", ".sbv"})

#: A subtitle file is untrusted input (§31). Real tracks for a feature film run well under a
#: megabyte; this only exists so a hostile file cannot be read into memory in full.
MAX_SUBTITLE_BYTES = 32 * 1024 * 1024

#: ``HH:MM:SS,mmm`` (SRT), ``HH:MM:SS.mmm`` (WebVTT) and WebVTT's short ``MM:SS.mmm``.
_TIME = re.compile(r"(?:(\d+):)?([0-5]?\d):([0-5]?\d)[.,](\d{1,3})")
_ARROW = "-->"

#: ``<v Speaker Name>`` — WebVTT's voice span, the one real speaker label in these formats.
_VOICE = re.compile(r"<v(?:\.[^\s>]+)*\s+([^>]+)>", re.IGNORECASE)
#: Inline markup (``<i>``, ``<c.loud>``, ``</v>``) and ASS override blocks (``{\an8}``).
_TAG = re.compile(r"<[^>]*>")
_OVERRIDE = re.compile(r"\{[^}]*\}")


def _seconds(match: re.Match[str]) -> float:
    hours, minutes, seconds, frac = match.groups()
    millis = int(frac.ljust(3, "0"))
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000.0


def _cue_times(line: str) -> tuple[float, float] | None:
    """Parse a cue timing line, ignoring any WebVTT cue settings that follow it."""
    if _ARROW not in line:
        return None
    left, _, right = line.partition(_ARROW)
    start, end = _TIME.search(left), _TIME.search(right)
    if start is None or end is None:
        return None
    return _seconds(start), _seconds(end)


def _clean_text(lines: list[str]) -> tuple[str, str | None]:
    """Flatten a cue's lines into searchable text, returning any voice label separately."""
    speaker: str | None = None
    parts: list[str] = []
    for raw in lines:
        if (voice := _VOICE.search(raw)) is not None and speaker is None:
            speaker = voice.group(1).strip() or None
        # Tags are removed, not replaced by a space: none of the markup in these formats is a
        # word separator, and `<i>demo</i>.` must not become `demo .` — that would put a
        # phrase into the transcript that the speaker never said in that shape. Real line
        # breaks are separators, and they are handled by the join below.
        text = _OVERRIDE.sub("", _TAG.sub("", raw))
        # A cue line broken for display is one sentence, so lines join with a space rather
        # than a newline: the searchable projection should not contain the line wrapping.
        parts.append(text)
    return " ".join(" ".join(parts).split()), speaker


def parse_cues(payload: str) -> list[tuple[float, float, str, str | None]]:
    """Parse SRT/WebVTT into ``(start, end, text, speaker)`` tuples, in file order.

    Blocks without a timing line — the ``WEBVTT`` header, ``NOTE``, ``STYLE`` and ``REGION``
    — fall out naturally rather than needing to be enumerated.

    Blank lines separate blocks, but the *timing* line is what starts a cue. Files in the wild
    do omit the blank line between cues, and a parser that only split on blank lines would
    then read the next cue's timing line as dialogue — putting a literal ``00:00:03,000 -->``
    into the transcript. Fabricating words nobody said is the one failure this format cannot
    tolerate (spec §7), so a second timing line inside a block ends the first cue and opens
    the next.
    """
    cues: list[tuple[float, float, str, str | None]] = []
    for block in re.split(r"\r?\n[ \t]*\r?\n", payload):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        timings = [
            (i, times)
            for i, ln in enumerate(lines)
            if _ARROW in ln and (times := _cue_times(ln)) is not None
        ]
        for n, (at, times) in enumerate(timings):
            stop = timings[n + 1][0] if n + 1 < len(timings) else len(lines)
            body = lines[at + 1 : stop]
            # A bare number immediately before the next cue's timing line is that cue's SRT
            # index, not dialogue. Only checked when another cue follows in the same block,
            # so a normal cue ending in a number keeps it.
            if stop < len(lines) and body and body[-1].isdigit():
                body = body[:-1]
            text, speaker = _clean_text(body)
            if text:
                cues.append((times[0], times[1], text, speaker))
    return cues


def read_subtitle_file(path: Path) -> str:
    """Read a subtitle file defensively: size-capped, BOM-tolerant, never failing on bytes.

    ``utf-8-sig`` matters in practice — a byte-order mark left on the first line makes the
    first cue's index unparseable, silently costing the first line of dialogue.
    """
    size = path.stat().st_size
    if size > MAX_SUBTITLE_BYTES:
        raise ValueError(f"subtitle file is {size} bytes, over the {MAX_SUBTITLE_BYTES} limit")
    return path.read_text(encoding="utf-8-sig", errors="replace")


class SubtitleASR:
    """Transcript from an embedded or sidecar subtitle track."""

    name = "subtitles"
    version = "1.0.0"
    remote = False

    def __init__(
        self,
        config: ASRConfig | None = None,
        *,
        source: str | Path | None = None,
        stream_index: int = 0,
        timeout: float = 300.0,
    ) -> None:
        self.config = config or ASRConfig()
        self.source = Path(source) if source else None
        self.stream_index = stream_index
        self.timeout = timeout

    def available(self) -> bool:
        """True when a track *could* be read.

        Whether a given video actually carries one cannot be answered without probing it, so
        this reports the capability — a sidecar file needs nothing, an embedded track needs
        FFmpeg — and ``transcribe`` reports the absence of a track as a warning.
        """
        if self.source and self.source.suffix.lower() in SUBTITLE_SUFFIXES:
            return self.source.is_file()
        return ffmpeg_available()

    def transcribe(self, audio_path: Path, ctx: FrameContext) -> ASROutput:
        target = self.source or Path(audio_path)
        warnings: list[str] = []

        if target.suffix.lower() in SUBTITLE_SUFFIXES:
            payload = self._read(target, warnings)
        else:
            payload = self._extract(target, warnings)

        cues = parse_cues(payload) if payload else []
        utterances = finalize(
            [
                utterance(text, start, end, speaker=speaker, language=ctx.language)
                for start, end, text, speaker in cues
            ],
            duration=ctx.duration,
            language=ctx.language,
        )
        if not utterances and not warnings:
            warnings.append(f"no subtitle cues found in {target.name}")

        log.info(
            "asr.subtitles",
            extra={"source": target.name, "cues": len(cues), "utterances": len(utterances)},
        )
        return ASROutput(
            utterances=utterances,
            language=ctx.language,
            model=f"{self.name}:{target.suffix.lstrip('.') or 'embedded'}",
            duration_s=ctx.duration,
            warnings=warnings,
        )

    # -- sources ------------------------------------------------------------

    def _read(self, path: Path, warnings: list[str]) -> str:
        try:
            return read_subtitle_file(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"could not read {path.name}: {exc}")
            return ""

    def _extract(self, media: Path, warnings: list[str]) -> str:
        """Pull an embedded track out to a temp SRT, which never outlives this call (§32)."""
        if not ffmpeg_available():
            warnings.append("ffmpeg is required to read an embedded subtitle track")
            return ""
        with tempfile.TemporaryDirectory(prefix="vctx-subs-") as tmp:
            out = extract_subtitles(
                media,
                Path(tmp) / "track.srt",
                stream_index=self.stream_index,
                timeout=self.timeout,
            )
            if out is None:
                warnings.append(f"{media.name} has no embedded subtitle track")
                return ""
            return self._read(out, warnings)


__all__ = [
    "MAX_SUBTITLE_BYTES",
    "SUBTITLE_SUFFIXES",
    "SubtitleASR",
    "parse_cues",
    "read_subtitle_file",
]
