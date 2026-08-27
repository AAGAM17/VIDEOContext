"""Utterance normalisation — the part of ASR that must not differ between engines.

Every engine produces the same three mistakes in its own way: a span that runs past the end
of the media, a fragment whose text is only whitespace, and word timings that disagree with
the utterance that contains them. Fixing that in each adapter would mean three
implementations that drift, so :func:`finalize` owns it and the adapters call it last.

It is deliberately **idempotent**: running it twice changes nothing. That lets the pipeline
call it defensively on whatever a third-party engine returned without needing to trust that
the plugin author called it.

What it does *not* do is invent anything. It never widens a span to look nicer, never merges
utterances that a model emitted separately, and never fills in a confidence that was not
reported. A timestamp in a ``.vctx`` document has to be traceable to something the engine
actually said (spec §7), so the only edits here are clamps, drops and orderings.
"""

from __future__ import annotations

from ...logging import get_logger
from ...schema.v1 import Utterance, Word

log = get_logger("asr.normalize")

#: Word probabilities and utterance confidences are reported on this scale.
_CONF_LO, _CONF_HI = 0.0, 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _confidence(value: float | None) -> float | None:
    """Keep a reported confidence, on scale, or keep the fact that none was reported."""
    if value is None:
        return None
    return round(_clamp(float(value), _CONF_LO, _CONF_HI), 4)


def _clean_words(words: list[Word], start: float, end: float) -> list[Word]:
    """Tidy word timings against the utterance that contains them.

    Whisper-family models emit word timestamps from a separate alignment pass, so a word can
    land a few milliseconds outside its own segment. Clamping to the utterance keeps the
    "find the exact moment the phrase was said" query (spec §7) from returning a word that
    claims to precede the sentence it belongs to.
    """
    out: list[Word] = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        w_start = _clamp(word.start, start, end)
        w_end = _clamp(word.end, w_start, end)
        out.append(
            Word(
                text=text,
                start=round(w_start, 3),
                end=round(w_end, 3),
                confidence=_confidence(word.confidence),
            )
        )
    out.sort(key=lambda w: (w.start, w.end))
    return out


def finalize(
    utterances: list[Utterance],
    *,
    duration: float = 0.0,
    language: str | None = None,
) -> list[Utterance]:
    """Return well-formed, timeline-ordered, id-assigned utterances.

    ``duration`` is the media duration; when it is positive, spans are clamped into
    ``[0, duration]``. Passing ``0.0`` (unknown duration) clamps only against negatives,
    because clamping to a duration we do not know would be the invention this module exists
    to avoid.
    """
    limit = duration if duration > 0 else None
    kept: list[Utterance] = []
    dropped_blank = 0
    clamped = 0
    reordered = 0

    for utt in utterances:
        text = " ".join(utt.text.split())
        if not text:
            dropped_blank += 1
            continue

        start, end = float(utt.start), float(utt.end)
        if end < start:
            # A model that reports end < start has told us nothing about which is which;
            # the ordering is recoverable, the intent is not, so swap and record it.
            start, end = end, start
            reordered += 1

        original = (start, end)
        start = _clamp(start, 0.0, limit if limit is not None else start)
        end = _clamp(end, start, limit if limit is not None else max(end, start))
        if (start, end) != original:
            clamped += 1

        kept.append(
            utt.model_copy(
                update={
                    "id": "",  # assigned below, once the timeline order is known
                    "text": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "language": utt.language or language,
                    "confidence": _confidence(utt.confidence),
                    "words": _clean_words(utt.words, start, end),
                }
            )
        )

    kept.sort(key=lambda u: (u.start, u.end))
    for index, utt in enumerate(kept):
        utt.id = f"utt_{index:04d}"

    if dropped_blank or clamped or reordered:
        log.debug(
            "asr.normalized",
            extra={
                "kept": len(kept),
                "dropped_blank": dropped_blank,
                "clamped": clamped,
                "reordered": reordered,
            },
        )
    return kept


def utterance(
    text: str,
    start: float,
    end: float,
    *,
    confidence: float | None = None,
    language: str | None = None,
    speaker: str | None = None,
    no_speech_prob: float | None = None,
    words: list[Word] | None = None,
) -> Utterance:
    """Build one un-numbered :class:`Utterance`; :func:`finalize` assigns the id.

    Engines use this so that adding an adapter does not mean learning the id convention or
    which fields are optional. ``end`` is raised to ``start`` only to satisfy the schema's
    ordering rule — a degenerate instant is kept rather than dropped, because the text is
    still real evidence that something was said at that moment.
    """
    return Utterance(
        id="",
        text=text,
        start=max(0.0, float(start)),
        end=max(float(start), float(end)),
        confidence=confidence,
        language=language,
        speaker=speaker,
        no_speech_prob=no_speech_prob,
        words=words or [],
    )


__all__ = ["finalize", "utterance"]
