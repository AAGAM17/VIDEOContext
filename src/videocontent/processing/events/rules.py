"""Rule-based event detection over an already-extracted document.

Five rule families, each deriving events from facts the document already contains:

============================  ====================================================
``scene_changed``             a boundary between two detected scenes
``text_appeared``/``…gone``   the endpoints of a temporal OCR event's lifespan
``command_entered``           OCR text that reads as a shell prompt plus a command
``error_shown``               OCR text carrying an error signature
``silence_started``/``…ended``  a gap between utterances long enough to be silence
============================  ====================================================

**Why rules at all, when a model could do this.** These five are exactly the events whose
evidence is already in the document and whose derivation is checkable by reading it. A model
asked "what happened here" would produce events nobody can trace — the opposite of what §11's
``refs`` field is for. Anything needing genuine visual understanding (*did the presenter point
at the chart?*) belongs to a vision provider, which can then attach its own events with its own
frame references and its own confidence.

**What is deliberately not here.** No ``speaker_started``/``speaker_stopped``: one pair per
utterance restates the transcript in a lossier form and buries the events that carry
information. No ``slide_changed``: distinguishing a slide advance from any other screen change
needs to know that the screen *is* a slide, which this layer cannot see. Emitting nothing beats
emitting a guess that a downstream consumer would read as a fact.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

from ...logging import get_logger
from ...schema.v1 import Event, EventType

log = get_logger("events.rules")

#: Shell prompt at the start of an OCR line: ``$``, ``>``, ``%``, ``PS>``, or a
#: ``user@host:~/path$`` style prefix. The command is whatever follows.
#:
#: A bare ``#`` is deliberately absent. It is a root prompt, but on the slides this tool reads
#: it is far more often a markdown heading — and ``# Introduction`` recorded as the command
#: ``Introduction`` is a fabricated event. It is still honoured inside the ``user@host:~#``
#: form, where the surrounding text disambiguates it.
_PROMPT = re.compile(
    r"""^(?:
        (?P<host>[\w.\-]+@[\w.\-]+:\S*)\s*[$#%]      # user@host:~/dir$
      | (?:PS\s*)?(?P<path>[A-Za-z]:\\\S*)\s*>       # PS C:\dir>
      | [$>%]                                         # bare prompt
    )\s+(?P<command>\S.*)$""",
    re.VERBOSE,
)

#: ``> Note:`` is a blockquote, not a command. A leading token ending in a colon is the cheap
#: discriminator, and it costs nothing real: no executable name ends in one.
_NOT_A_COMMAND = re.compile(r"^\S*:$")

#: Error signatures that appear in terminals, consoles and error dialogs. Matched
#: case-sensitively where capitalisation carries the signal (``Error`` not ``error``), because
#: "an error occurred" in prose is not an error being *shown*.
#:
#: **Ordered most specific first**, and it has to stay that way: the first match wins and
#: becomes the event's ``detector`` suffix, which is a provenance claim. ``500 Internal Server
#: Error`` contains the substring ``Error``, so with ``exception`` earlier it would be recorded
#: as an exception — an HTTP response labelled as a thrown error. New patterns go above
#: anything broader than themselves.
_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("traceback", re.compile(r"\bTraceback\b")),
    ("http_status", re.compile(r"\b[45]\d{2}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")),
    ("exception", re.compile(r"\b\w*(?:Error|Exception)\b")),
    ("failure", re.compile(r"\b(?:FAILED|FAIL|ERROR|Failed)\b")),
)

#: A gap between utterances shorter than this is a breath, not silence worth an event.
DEFAULT_MIN_SILENCE_S = 2.0

#: Below this many frames an OCR reading is a flicker; the dedup stage already marks the
#: distinction as ``stable``, and only stable text gets appearance events.
_STABLE_ONLY = True


def _clip(value: float, duration: float) -> float:
    if duration <= 0:
        return max(0.0, value)
    return min(max(0.0, value), duration)


class RuleEventDetector:
    """Derives events from a document's own scenes, OCR events and transcript."""

    name = "rules"
    version = "1.0.0"

    def __init__(
        self,
        *,
        min_silence_s: float = DEFAULT_MIN_SILENCE_S,
        detect_commands: bool = True,
        detect_errors: bool = True,
    ) -> None:
        self.min_silence_s = max(0.0, min_silence_s)
        self.detect_commands = detect_commands
        self.detect_errors = detect_errors

    def detect(self, doc: Any, ctx: Any = None) -> list[Event]:
        """Return events in timeline order, with IDs assigned once that order is known."""
        duration = float(getattr(getattr(doc, "video", None), "duration", 0.0) or 0.0)
        found: list[Event] = []
        found += self._scene_changes(doc)
        found += self._text_lifespans(doc)
        found += self._terminal_events(doc)
        found += self._silences(doc, duration)

        found.sort(key=lambda e: (e.start, e.end, e.type))
        for index, event in enumerate(found):
            event.id = f"evt_{index:04d}"

        log.debug(
            "events.detected",
            extra={"total": len(found), "types": sorted({e.type for e in found})},
        )
        return found

    # -- rules --------------------------------------------------------------

    def _scene_changes(self, doc: Any) -> list[Event]:
        """One event per boundary *between* scenes — not one per scene.

        The first scene's start is the video starting, which is not a change. The event is an
        instant at the boundary rather than a span, because that is precisely what the scene
        detector claimed; widening it to the surrounding scenes would invent a duration.
        """
        scenes = list(getattr(doc, "scenes", []) or [])
        events: list[Event] = []
        for previous, scene in pairwise(scenes):
            events.append(
                Event(
                    id="",
                    type=EventType.SCENE_CHANGED.value,
                    start=scene.start,
                    end=scene.start,
                    description=f"Scene boundary at {scene.start:.2f}s",
                    confidence=scene.confidence,
                    source=["visual"],
                    detector=f"rule:scene-change/{scene.detector or 'unknown'}",
                    attributes={
                        "signals": list(scene.signals),
                        "change_score": scene.change_score,
                    },
                    refs={"scenes": [previous.id, scene.id]},
                )
            )
        return events

    def _text_lifespans(self, doc: Any) -> list[Event]:
        """``text_appeared`` and ``text_disappeared`` at the endpoints of an OCR lifespan.

        Only for text the dedup stage marked ``stable`` — seen in more than one frame. A
        single-frame reading is as likely to be a compression artefact as a real caption, and
        an event claiming text appeared is a stronger statement than the reading supports.
        """
        events: list[Event] = []
        for item in getattr(doc, "ocr", []) or []:
            if _STABLE_ONLY and not item.stable:
                continue
            refs = {"ocr": [item.id]}
            for kind, ts in (
                (EventType.TEXT_APPEARED, item.first_frame_ts or item.start),
                (EventType.TEXT_DISAPPEARED, item.last_frame_ts or item.end),
            ):
                events.append(
                    Event(
                        id="",
                        type=kind.value,
                        start=ts,
                        end=ts,
                        description=item.text,
                        confidence=item.confidence,
                        source=["ocr"],
                        detector="rule:text-lifespan",
                        attributes={"frame_count": item.frame_count},
                        refs=refs,
                    )
                )
        return events

    def _terminal_events(self, doc: Any) -> list[Event]:
        """``command_entered`` and ``error_shown`` from the text of OCR events.

        The span is the OCR event's own span: the command was on screen for exactly as long as
        the text was, and that is the only duration there is evidence for.
        """
        events: list[Event] = []
        for item in getattr(doc, "ocr", []) or []:
            text = item.text.strip()
            if not text:
                continue

            if self.detect_commands and (command := _command_in(text)) is not None:
                events.append(
                    Event(
                        id="",
                        type=EventType.COMMAND_ENTERED.value,
                        start=item.start,
                        end=item.end,
                        description=f"Command entered: {command}",
                        confidence=item.confidence,
                        source=["ocr"],
                        detector="rule:shell-prompt",
                        attributes={"command": command, "line": text},
                        refs={"ocr": [item.id]},
                    )
                )

            if self.detect_errors and (signal := _error_in(text)) is not None:
                events.append(
                    Event(
                        id="",
                        type=EventType.ERROR_SHOWN.value,
                        start=item.start,
                        end=item.end,
                        description=text,
                        confidence=item.confidence,
                        source=["ocr"],
                        detector=f"rule:error-signature/{signal}",
                        attributes={"signal": signal},
                        refs={"ocr": [item.id]},
                    )
                )
        return events

    def _silences(self, doc: Any, duration: float) -> list[Event]:
        """Silence as the *gaps* in the transcript, bounded by the utterances around them.

        Derived from the transcript rather than from the audio on purpose: a gap between two
        utterances is exactly what "the ASR stage found no speech here" means, and both ends
        are referenced, so the claim is checkable. Leading and trailing silence are included
        only when the media duration is known — otherwise the trailing gap has no end.
        """
        transcript = list(getattr(doc, "transcript", []) or [])
        if not transcript or self.min_silence_s <= 0:
            return []

        gaps: list[tuple[float, float, list[str]]] = []
        if transcript[0].start >= self.min_silence_s:
            gaps.append((0.0, transcript[0].start, [transcript[0].id]))
        for previous, current in pairwise(transcript):
            if current.start - previous.end >= self.min_silence_s:
                gaps.append((previous.end, current.start, [previous.id, current.id]))
        if duration > 0 and duration - transcript[-1].end >= self.min_silence_s:
            gaps.append((transcript[-1].end, duration, [transcript[-1].id]))

        events: list[Event] = []
        for start, end, refs in gaps:
            start, end = _clip(start, duration), _clip(end, duration)
            for kind, ts in (
                (EventType.SILENCE_STARTED, start),
                (EventType.SILENCE_ENDED, end),
            ):
                events.append(
                    Event(
                        id="",
                        type=kind.value,
                        start=ts,
                        end=ts,
                        description=f"{end - start:.2f}s without speech",
                        # No confidence: this is derived from an absence, and a number here
                        # would look like a scored detection rather than an inference.
                        source=["transcript"],
                        detector="rule:transcript-gap",
                        attributes={"duration_s": round(end - start, 3)},
                        refs={"transcript": refs},
                    )
                )
        return events


def _command_in(text: str) -> str | None:
    """Return the command after a shell prompt, or ``None`` if this is not a prompt line."""
    for line in text.splitlines() or [text]:
        if (match := _PROMPT.match(line.strip())) is not None:
            command = match.group("command").strip()
            if command and not _NOT_A_COMMAND.match(command.split()[0]):
                return command
    return None


def _error_in(text: str) -> str | None:
    """Return the name of the matching error signature, or ``None``."""
    for label, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return label
    return None


__all__ = ["DEFAULT_MIN_SILENCE_S", "RuleEventDetector"]
