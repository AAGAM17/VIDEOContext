"""Segmentation — fusing the per-modality timelines into the unit retrieval scores.

A segment is a time window plus every fact that overlaps it, resolved by reference (spec §12).
It exists because none of the extracted modalities is a good retrieval unit on its own: an
utterance has no idea what was on screen while it was spoken, and an OCR event that lasted
nine seconds has no idea what was said over it. Search operates on the fusion, which is why
"when was the pricing slide shown?" can be answered by a spoken word and "what command was
typed?" by on-screen text, from the same index.

Two properties are load-bearing and both are tested:

* **every fact lands in at least one segment.** A fact the fusion drops is a fact search can
  never return, and the document would then contain evidence that is unreachable.
* **``text`` is a projection, never a source.** It is the denormalized concatenation of the
  facts the segment references, in timeline order, so a consumer can build an index or a prompt
  without resolving references. Nothing appears in it that is not also reachable through an ID,
  which is what keeps an answer built from ``text`` traceable back to a timestamp.

Windows follow scene boundaries where there are any, because a scene change is the strongest
available signal that the subject changed. They are then subdivided to respect
``max_duration`` — a 3-minute window retrieves as one blob and pins nothing down — and short
windows are merged so that a half-second cut does not become a segment of its own.
"""

from __future__ import annotations

import math
from itertools import pairwise

from ..config import SegmentConfig
from ..logging import get_logger
from ..schema.v1 import Segment, VideoContextDocument

log = get_logger("processing.segments")

#: How far a subdivision point may move to land in a gap between utterances. Beyond this the
#: even split is kept: dragging a boundary several seconds to avoid clipping one sentence
#: distorts the window more than the clipped sentence costs.
SNAP_TOLERANCE_S = 3.0


def _overlaps(start: float, end: float, w_start: float, w_end: float, *, last: bool) -> bool:
    """Half-open ``[w_start, w_end)`` overlap, with instants treated as points.

    Spans are half-open so a fact ending exactly at a boundary belongs to the earlier window
    only. An instant (``end == start``) has no width to overlap with, so it is tested for
    containment instead — otherwise every ``scene_changed`` event, which is deliberately
    zero-length, would belong to no segment at all. The final window includes its own end so
    that a fact at exactly the media duration is not orphaned.
    """
    if end <= start:
        return w_start <= start <= w_end if last else w_start <= start < w_end
    return start < w_end and end > w_start


def _scene_windows(doc: VideoContextDocument) -> list[tuple[float, float]]:
    duration = doc.video.duration
    scenes = sorted(doc.scenes, key=lambda s: (s.start, s.end))
    if not scenes:
        return [(0.0, duration)] if duration > 0 else []
    windows = [(s.start, s.end) for s in scenes if s.end > s.start]
    if not windows:
        return [(0.0, duration)] if duration > 0 else []
    # Scenes are expected to tile the timeline, but a detector is not required to guarantee
    # it. Closing the gaps here means no fact can fall between two segments.
    closed: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in windows:
        if start > cursor + 1e-6:
            closed.append((cursor, start))
        closed.append((max(start, cursor), end))
        cursor = end
    if duration > cursor + 1e-6:
        closed.append((cursor, duration))
    return closed


def _utterance_gaps(doc: VideoContextDocument) -> list[float]:
    """Midpoints of the silences between utterances — the natural places to cut."""
    ordered = sorted(doc.transcript, key=lambda u: u.start)
    return [
        (previous.end + current.start) / 2
        for previous, current in pairwise(ordered)
        if current.start > previous.end
    ]


def _snap(point: float, gaps: list[float], tolerance: float) -> float:
    if not gaps:
        return point
    nearest = min(gaps, key=lambda g: abs(g - point))
    return nearest if abs(nearest - point) <= tolerance else point


def _subdivide(
    window: tuple[float, float], cfg: SegmentConfig, gaps: list[float]
) -> list[tuple[float, float]]:
    start, end = window
    span = end - start
    if span <= cfg.max_duration:
        return [window]
    parts = math.ceil(span / cfg.max_duration)
    step = span / parts
    cuts = [start + step * i for i in range(1, parts)]
    if cfg.split_on_utterance_boundary:
        cuts = [_snap(c, gaps, SNAP_TOLERANCE_S) for c in cuts]
    # Snapping can reorder or collapse cuts; sorting and dropping degenerate pieces keeps the
    # result a valid tiling regardless of where the gaps happened to be.
    edges = [start, *sorted(c for c in cuts if start < c < end), end]
    return [(a, b) for a, b in pairwise(edges) if b - a > 1e-6]


def _merge_short(
    windows: list[tuple[float, float]], min_duration: float
) -> list[tuple[float, float]]:
    """Absorb sub-``min_duration`` windows into a neighbour.

    A window can end up longer than ``max_duration`` as a result. That is the intended
    trade: a 0.4-second segment retrieves nothing useful and pollutes every ranking it
    appears in, while a slightly over-long one merely retrieves coarsely.
    """
    if not windows:
        return []
    merged: list[list[float]] = [list(windows[0])]
    for start, end in windows[1:]:
        # Two reasons to extend the open window instead of starting a new one: this piece is
        # too short to stand alone, or the window it would follow is. Same action either way.
        too_short = end - start < min_duration
        previous_too_short = merged[-1][1] - merged[-1][0] < min_duration
        if too_short or previous_too_short:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    # The final window may still be short if it is the only one, or if everything before it
    # was already long enough; fold it back rather than emitting a runt.
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < min_duration:
        merged[-2][1] = merged[-1][1]
        merged.pop()
    return [(a, b) for a, b in merged]


def plan_windows(
    doc: VideoContextDocument, config: SegmentConfig | None = None
) -> list[tuple[float, float]]:
    """The segment boundaries, before any facts are attached."""
    cfg = config or SegmentConfig()
    base = _scene_windows(doc) if cfg.align_to_scenes else []
    if not base:
        duration = doc.video.duration
        if duration <= 0:
            return []
        base = [(0.0, duration)]
    gaps = _utterance_gaps(doc) if cfg.split_on_utterance_boundary else []
    subdivided = [piece for window in base for piece in _subdivide(window, cfg, gaps)]
    return _merge_short(subdivided, cfg.min_duration)


def build_segments(
    doc: VideoContextDocument, config: SegmentConfig | None = None
) -> list[Segment]:
    """Build the fused segments for ``doc`` from the modalities it already contains."""
    cfg = config or SegmentConfig()
    windows = plan_windows(doc, cfg)
    segments: list[Segment] = []

    for index, (start, end) in enumerate(windows):
        last = index == len(windows) - 1

        def within(items, *, point=False, _s=start, _e=end, _last=last):
            if point:
                return [i for i in items if _overlaps(i.ts, i.ts, _s, _e, last=_last)]
            return [i for i in items if _overlaps(i.start, i.end, _s, _e, last=_last)]

        scenes = within(doc.scenes)
        utterances = within(doc.transcript)
        texts = within(doc.ocr)
        notes = within(doc.vision)
        objects = within(doc.objects)
        events = within(doc.events)
        frames = within(doc.frames, point=True)

        languages = sorted(
            {u.language for u in utterances if u.language}
            | {t.language for t in texts if t.language}
        )
        segments.append(
            Segment(
                id=f"segment_{index:04d}",
                start=round(start, 3),
                end=round(end, 3),
                scene_ids=[s.id for s in scenes],
                transcript_ids=[u.id for u in utterances],
                ocr_ids=[t.id for t in texts],
                vision_ids=[n.id for n in notes],
                object_ids=[o.id for o in objects],
                event_ids=[e.id for e in events],
                frame_ids=[f.id for f in frames],
                text=project_text(utterances, texts, notes),
                summary=None,
                # Left empty deliberately. Retrieval scores `text` directly, so a keyword list
                # would be a second, lossier copy of the same signal — and every cheap
                # extractor available without a new dependency produces noise on exactly this
                # material (shell commands, URLs, prices). A summarizing provider can fill
                # both this and `summary` later.
                keywords=[],
                languages=languages,
            )
        )

    log.debug(
        "segments.built",
        extra={"total": len(segments), "windows": len(windows)},
    )
    return segments


def project_text(utterances, texts, notes) -> str:
    """The searchable projection: every referenced fact's words, in timeline order.

    Deduplicated on exact repeats — an OCR event and an utterance often carry the same phrase,
    and doubling it would let a lexical scorer rank a segment higher for saying something once.
    No modality prefixes: a ``[ocr]`` marker in the text would break a phrase query that
    happened to span the join.
    """
    parts = sorted(
        [(u.start, u.text) for u in utterances]
        + [(t.start, t.text) for t in texts]
        + [(n.start, n.description) for n in notes],
        key=lambda item: item[0],
    )
    seen: set[str] = set()
    lines: list[str] = []
    for _, text in parts:
        cleaned = " ".join(text.split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            lines.append(cleaned)
    return "\n".join(lines)


__all__ = ["SNAP_TOLERANCE_S", "build_segments", "plan_windows", "project_text"]
