"""The fourth scene signal: on-screen text changes.

Frame differencing cannot see that a slide changed when the new slide is visually similar to
the old one — same template, same background, different words. Text is the signal that
separates them, but it is only available *after* OCR, which itself depends on sampling, which
depends on scenes. So this runs as a refinement pass rather than inside the detector:
scenes are detected, frames are sampled, text is read, and then scenes are split where the
words on screen changed but the pixels did not.

Enabled by ``config.scenes.use_ocr_signal``.
"""

from __future__ import annotations

from ...interfaces import SceneSpan
from ...logging import get_logger
from ...schema.v1 import OCRText
from .util import Boundary, boundaries_to_spans

log = get_logger("scenes.refine")


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def text_change_boundaries(
    ocr: list[OCRText],
    *,
    duration: float,
    similarity_threshold: float = 0.35,
) -> list[Boundary]:
    """Instants where the set of words on screen turned over substantially.

    Each distinct OCR fact start time is a candidate. The words visible just before it are
    compared with the words visible just after; a low overlap means the screen's content
    changed even if its appearance did not.
    """
    if not ocr:
        return []

    def active(ts: float) -> set[str]:
        words: set[str] = set()
        for fact in ocr:
            if fact.start <= ts < max(fact.end, fact.start + 1e-3):
                words |= _tokens(fact.text)
        return words

    starts = sorted({round(fact.start, 3) for fact in ocr if fact.start > 0.0})
    boundaries: list[Boundary] = []
    for ts in starts:
        before = active(max(0.0, ts - 0.25))
        after = active(ts + 0.25)
        if not before and not after:
            continue
        similarity = _jaccard(before, after)
        if similarity <= similarity_threshold:
            # Score reflects how complete the turnover was, so budgeting and
            # min-duration arbitration can compare it against difference scores.
            boundaries.append(Boundary(ts, round(1.0 - similarity, 4), ["text"]))
    return [b for b in boundaries if 0.0 < b.ts < duration]


def refine_with_text(
    scenes: list[SceneSpan],
    ocr: list[OCRText],
    *,
    duration: float,
    min_scene_duration: float = 1.5,
    similarity_threshold: float = 0.35,
) -> list[SceneSpan]:
    """Split scenes at text turnovers that the visual pass missed.

    Existing boundaries are always preserved — this only ever adds detail, so a refinement
    pass can never destroy a cut that the visual signals were confident about.
    """
    if not scenes or not ocr:
        return scenes

    existing = [Boundary(scene.start, scene.score or 0.0, list(scene.signals) or ["cut"])
                for scene in scenes if scene.start > 0.0]
    text_cuts = text_change_boundaries(
        ocr, duration=duration, similarity_threshold=similarity_threshold
    )

    known = [scene.start for scene in scenes] + [duration]
    added = [
        cut for cut in text_cuts
        if all(abs(cut.ts - ts) >= min_scene_duration for ts in known)
    ]
    if not added:
        return scenes

    refined = boundaries_to_spans(existing + added, duration)
    log.info(
        "scenes.refined",
        extra={"before": len(scenes), "after": len(refined), "text_cuts": len(added)},
    )
    return refined


__all__ = ["refine_with_text", "text_change_boundaries"]
