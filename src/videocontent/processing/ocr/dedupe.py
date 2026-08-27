"""Temporal OCR deduplication — per-frame text observations become events with lifespans.

This is the difference between an OCR dump and video *context*. Sampling a 40-second slide at
1 fps yields forty copies of its title. Forty rows is not forty facts: it is one fact — "the
title *Quarterly Business Review* was on screen from 00:03 to 00:43" — recorded forty times.
Storing the rows makes the document forty times larger, makes retrieval return forty
near-identical hits for one query, and never answers *when* anything appeared or left.

So observations are tracked across frames and collapsed into :class:`OCRText` events.

**Matching.** A track continues when the new text is similar enough (OCR output jitters — a
letter flips, a comma appears) *and* its box still overlaps the track's. The box gate is what
keeps two different on-screen elements that happen to share a word from being merged into one
event. Its cost is that text which physically moves — a scrolling terminal — starts a new
event at each position; that is the honest reading of "a different thing at a different place",
and lowering ``iou_threshold`` relaxes it for footage where it is the wrong call.

**Gap bridging.** OCR sometimes fails to read text that is plainly still there — a cursor
blinks over it, JPEG noise lands badly. A track therefore survives up to ``max_gap_s`` of
absence before it is closed, so one bad frame does not split one event into two.

**Where the span ends.** Text seen at 10.0 s and missing at 11.0 s disappeared *somewhere in
between*; nothing observed says where. ``end`` is the midpoint, which is the minimum-error
estimate and bounded by the sampling interval. Reporting 10.0 instead would under-claim every
event by up to one interval and make "what was on screen at 10.5 s" answer "nothing" for an
instant that plainly showed the text. The raw evidence stays available and unrounded in
``first_frame_ts``/``last_frame_ts``, so a caller that needs only what was literally seen can
use those and ignore the interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ...config import OCRConfig
from ...interfaces import OCRObservation
from ...logging import get_logger
from ...schema.v1 import OCRText

log = get_logger("ocr.dedupe")

BBox = tuple[float, float, float, float]


def normalize(text: str) -> str:
    """Comparison form: case- and whitespace-insensitive, content otherwise intact."""
    return " ".join(text.lower().split())


def similarity(a: str, b: str) -> float:
    """Ratio in [0, 1]. Cheap length gate first — most candidate pairs are nowhere close."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) / len(longer) < 0.5:
        return 0.0
    return SequenceMatcher(None, shorter, longer).ratio()


def iou(a: BBox | None, b: BBox | None) -> float:
    """Intersection over union. Returns 1.0 when either box is unknown, i.e. no evidence
    against a match — the text gate then decides alone."""
    if a is None or b is None:
        return 1.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class _Track:
    """One piece of on-screen text, followed across frames while it stays there."""

    observations: list[OCRObservation] = field(default_factory=list)
    key: str = ""
    bbox: BBox | None = None
    last_ts: float = 0.0
    absent_ts: float | None = None

    def extend(self, obs: OCRObservation) -> None:
        self.observations.append(obs)
        self.key = normalize(obs.text)
        self.bbox = obs.bbox  # follow drift: match against the most recent position
        self.last_ts = obs.ts
        self.absent_ts = None

    @property
    def first_ts(self) -> float:
        return self.observations[0].ts

    def representative(self) -> OCRObservation:
        """The reading to publish: most confident, longest as a tie-break.

        Longest matters — a partially-recognised "Quarterly Business" and a complete
        "Quarterly Business Review" can carry the same confidence, and the complete one is
        the text a user will search for.
        """
        return max(
            self.observations,
            key=lambda o: (o.confidence if o.confidence is not None else 0.0, len(o.text)),
        )


def deduplicate(
    observations: list[OCRObservation],
    *,
    config: OCRConfig | None = None,
    duration: float = 0.0,
    frame_ts: list[float] | None = None,
    engine: str | None = None,
) -> list[OCRText]:
    """Collapse per-frame observations into temporal OCR events.

    ``frame_ts`` is the list of timestamps that were actually looked at. It is what makes
    absence meaningful: without it, "no text at 12 s" cannot be distinguished from "12 s was
    never sampled", and every span would have to end where it started.
    """
    cfg = config or OCRConfig()
    if not observations:
        return []

    by_frame: dict[float, list[OCRObservation]] = {}
    for obs in observations:
        by_frame.setdefault(obs.ts, []).append(obs)
    timeline = sorted(by_frame)

    # Frames with no text still carry information: they are where text ended.
    sampled = sorted(set(frame_ts or [])) or timeline
    span_end = max(duration, sampled[-1] if sampled else 0.0)

    if not cfg.dedupe:
        events = [
            _event(_single(obs), sampled, span_end, cfg, engine) for obs in observations
        ]
        return _finalize(events)

    open_tracks: list[_Track] = []
    closed: list[_Track] = []

    for ts in sampled:
        frame_obs = by_frame.get(ts, [])
        matched = _match(open_tracks, frame_obs, cfg)

        for t_index, o_index in matched.items():
            open_tracks[t_index].extend(frame_obs[o_index])
        claimed = set(matched.values())
        for o_index, obs in enumerate(frame_obs):
            if o_index in claimed:
                continue
            fresh = _Track()
            fresh.extend(obs)
            open_tracks.append(fresh)

        still_open: list[_Track] = []
        for track in open_tracks:
            if track.last_ts == ts:
                still_open.append(track)
                continue
            if track.absent_ts is None:
                track.absent_ts = ts
            if ts - track.last_ts > cfg.max_gap_s:
                closed.append(track)
            else:
                still_open.append(track)
        open_tracks = still_open

    closed.extend(open_tracks)
    events = [_event(track, sampled, span_end, cfg, engine) for track in closed]
    result = _finalize(events)
    log.info(
        "ocr.deduplicated",
        extra={
            "observations": len(observations),
            "events": len(result),
            "reduction": round(1 - len(result) / len(observations), 3),
        },
    )
    return result


def _match(
    tracks: list[_Track], frame_obs: list[OCRObservation], cfg: OCRConfig
) -> dict[int, int]:
    """Greedy one-to-one assignment of this frame's observations to open tracks.

    Returns ``{track index: observation index}``. Indices rather than objects: two identical
    lines in one frame are equal by value but are not the same element, and a value-keyed map
    would silently merge them.

    Greedy by best score rather than first-fit: when a frame contains two similar lines, the
    better pairing must win both times, or one line steals the other's history.
    """
    scored: list[tuple[float, float, int, int]] = []
    for t_index, track in enumerate(tracks):
        for o_index, obs in enumerate(frame_obs):
            text_score = similarity(track.key, normalize(obs.text))
            if text_score < cfg.similarity_threshold:
                continue
            overlap = iou(track.bbox, obs.bbox)
            if overlap < cfg.iou_threshold:
                continue
            scored.append((text_score, overlap, t_index, o_index))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    used_tracks: set[int] = set()
    used_obs: set[int] = set()
    matched: dict[int, int] = {}
    for _text_score, _overlap, t_index, o_index in scored:
        if t_index in used_tracks or o_index in used_obs:
            continue
        used_tracks.add(t_index)
        used_obs.add(o_index)
        matched[t_index] = o_index
    return matched


def _single(obs: OCRObservation) -> _Track:
    track = _Track()
    track.extend(obs)
    return track


def _event(
    track: _Track,
    sampled: list[float],
    span_end: float,
    cfg: OCRConfig,
    engine: str | None,
) -> OCRText:
    rep = track.representative()
    first, last = track.first_ts, track.last_ts

    if track.absent_ts is not None:
        end = (last + track.absent_ts) / 2.0
    elif sampled and last >= sampled[-1] - 1e-6:
        # Still on screen in the last frame we looked at; it lasts as long as the video does.
        end = span_end
    else:
        end = _midpoint_to_next(last, sampled, cfg.max_gap_s, span_end)

    confidences = [o.confidence for o in track.observations if o.confidence is not None]
    width, height = rep.frame_width, rep.frame_height
    normalized = None
    if rep.bbox and width and height:
        normalized = [
            round(rep.bbox[0] / width, 5), round(rep.bbox[1] / height, 5),
            round(rep.bbox[2] / width, 5), round(rep.bbox[3] / height, 5),
        ]

    return OCRText(
        id="",  # assigned in _finalize, once events are in timeline order
        start=round(first, 3),
        end=round(max(first, min(end, span_end)), 3),
        text=rep.text,
        confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        language=rep.language,
        bbox=list(rep.bbox) if rep.bbox else None,
        bbox_normalized=normalized,
        frame_count=len(track.observations),
        first_frame_ts=round(first, 3),
        last_frame_ts=round(last, 3),
        stable=len(track.observations) > 1,
        engine=engine,
        block_index=rep.block_index,
    )


def _midpoint_to_next(
    last: float, sampled: list[float], max_gap: float, span_end: float
) -> float:
    """Half-way to the next frame we looked at, for a track closed without an absence record."""
    for ts in sampled:
        if ts > last + 1e-6:
            return min((last + ts) / 2.0, last + max_gap)
    return min(span_end, last + max_gap)


def _finalize(events: list[OCRText]) -> list[OCRText]:
    ordered = sorted(events, key=lambda e: (e.start, e.block_index or 0, e.text))
    for index, event in enumerate(ordered):
        event.id = f"ocr_{index:04d}"
    return ordered


__all__ = ["deduplicate", "iou", "normalize", "similarity"]
