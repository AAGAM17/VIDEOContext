"""Temporal intelligence primitives — relations, windows, changes, chapters, UI states.

Everything here is a pure function over a :class:`VideoContextDocument`: no media
decoding, no models, no network. That is what makes temporal reasoning cheap and
testable — the expensive extraction already happened, and these views only relate
facts the document already contains.

The temporal invariant holds throughout: every object returned here carries
``start``/``end`` copied from document facts, never generated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Any

from .logging import get_logger
from .timecode import format_timecode, parse_timecode

log = get_logger("temporal")

_EPS = 1e-6


class TemporalRelation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    CONTAINS = "contains"
    NEAR = "near"
    CO_OCCURS = "co_occurs"
    STARTS_WITHIN = "starts_within"
    ENDS_WITHIN = "ends_within"
    PRECEDES = "precedes"
    FOLLOWS = "follows"


def relate(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
    *,
    tolerance: float = 1.0,
    near_s: float = 5.0,
) -> set[TemporalRelation]:
    """Interval relations between ``[a_start, a_end]`` and ``[b_start, b_end]``.

    Float-safe (``1e-6`` epsilon). ``tolerance`` softens boundary relations
    (DURING/CONTAINS/BEFORE/AFTER/STARTS_WITHIN/ENDS_WITHIN/CO_OCCURS) so a caption
    starting 0.3 s before a scene still counts as starting within it; strict
    ``OVERLAPS``/``PRECEDES``/``FOLLOWS`` use no tolerance so ordering stays exact.
    """
    rel: set[TemporalRelation] = set()
    overlaps = a_start < b_end - _EPS and b_start < a_end - _EPS
    if overlaps:
        rel.add(TemporalRelation.OVERLAPS)
    if b_start - tolerance <= a_start and a_end <= b_end + tolerance:
        rel.add(TemporalRelation.DURING)
    if a_start - tolerance <= b_start and b_end <= a_end + tolerance:
        rel.add(TemporalRelation.CONTAINS)
    if a_end <= b_start + tolerance:
        rel.add(TemporalRelation.BEFORE)
    if a_start >= b_end - tolerance:
        rel.add(TemporalRelation.AFTER)
    if a_end <= b_start + _EPS:
        rel.add(TemporalRelation.PRECEDES)
    if a_start >= b_end - _EPS:
        rel.add(TemporalRelation.FOLLOWS)
    gap = max(0.0, max(a_start, b_start) - min(a_end, b_end))
    if gap <= near_s:
        rel.add(TemporalRelation.NEAR)
    if overlaps or gap <= tolerance:
        rel.add(TemporalRelation.CO_OCCURS)
    if b_start - tolerance <= a_start <= b_end + tolerance:
        rel.add(TemporalRelation.STARTS_WITHIN)
    if b_start - tolerance <= a_end <= b_end + tolerance:
        rel.add(TemporalRelation.ENDS_WITHIN)
    return rel


@dataclass(frozen=True)
class TemporalWindow:
    """A clamped time range — the unit of evidence expansion."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def label(self) -> str:
        return f"{format_timecode(self.start)} → {format_timecode(self.end)}"


def window_at(ts: float, radius: float, duration: float) -> TemporalWindow:
    """``radius`` seconds around instant ``ts``, clamped to ``[0, duration]``."""
    return TemporalWindow(start=max(0.0, ts - radius), end=min(duration, ts + radius))


def window_around(start: float, end: float, radius: float, duration: float) -> TemporalWindow:
    """``radius`` seconds around span ``[start, end]``, clamped to ``[0, duration]``."""
    return TemporalWindow(start=max(0.0, start - radius), end=min(duration, end + radius))


def merge_windows(windows: list[TemporalWindow]) -> list[TemporalWindow]:
    """Collapse overlapping/adjacent windows, sorted by start."""
    ordered = sorted(windows, key=lambda w: (w.start, w.end))
    merged: list[TemporalWindow] = []
    for window in ordered:
        if merged and window.start <= merged[-1].end + _EPS:
            last = merged.pop()
            merged.append(TemporalWindow(start=last.start, end=max(last.end, window.end)))
        else:
            merged.append(window)
    return merged


# ---------------------------------------------------------------------------
# text helpers (cheap signals only)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "it", "its", "as", "at", "by", "from", "you", "we", "they", "he", "she",
    "them", "his", "her", "our", "your", "their", "not", "no", "yes", "so",
    "if", "then", "there", "here", "what", "when", "where", "which", "who",
    "how", "why", "will", "would", "can", "could", "should", "shall", "may",
    "might", "do", "does", "did", "done", "have", "has", "had", "having",
    "let", "us", "lets", "now", "today", "video",
])


def _terms(text: str) -> list[str]:
    return [
        token.strip(".,;:!?()[]{}\"'").lower()
        for token in re.split(r"\s+", text or "")
        if len(token.strip(".,;:!?()[]{}\"'")) >= 3
    ]


def _content_terms(text: str) -> set[str]:
    return {t for t in _terms(text) if t not in _STOPWORDS and not t.isdigit()}


def _clip(text: str, limit: int = 160) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# change detection — what changed between adjacent regions?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    """A content turnover between two adjacent temporal regions.

    ``before``/``after`` are truncated quotes of the underlying facts, and
    ``evidence_ids`` name the facts — a change with no evidence is not emitted.
    ``observed`` distinguishes measured turnover (scene/OCR/speech signals) from
    semantically inferred change; this detector only emits observed change.
    """

    id: str
    ts: float
    change_type: str
    before: str
    after: str
    evidence_ids: tuple[str, ...] = ()
    method: str = "signal_turnover"
    observed: bool = True
    confidence: float | None = None

    @property
    def timecode(self) -> str:
        return format_timecode(self.ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "timecode": self.timecode,
            "change_type": self.change_type,
            "before": self.before,
            "after": self.after,
            "evidence_ids": list(self.evidence_ids),
            "method": self.method,
            "observed": self.observed,
            "confidence": self.confidence,
        }


def _facts_in(doc: Any, start: float, end: float) -> dict[str, list[Any]]:
    """Facts overlapping ``[start, end]`` — defensive about ordering and empties."""
    out: dict[str, list[Any]] = {"scenes": [], "transcript": [], "ocr": []}
    for scene in sorted(getattr(doc, "scenes", []), key=lambda s: (s.start, s.end)):
        if scene.start < end - _EPS and start < scene.end - _EPS:
            out["scenes"].append(scene)
    for item in getattr(doc, "transcript", []):
        if item.start < end - _EPS and start < item.end - _EPS:
            out["transcript"].append(item)
    for item in getattr(doc, "ocr", []):
        if item.start < end - _EPS and start < item.end - _EPS:
            out["ocr"].append(item)
    return out


def detect_changes(doc: Any) -> list[Change]:
    """Cheap change detection between adjacent scenes (or halves when sceneless).

    Signals, in cost order: scene boundaries (already detected), OCR term turnover,
    transcript speech/silence transitions. No vision calls, no embeddings.
    """
    duration = max(0.0, float(getattr(getattr(doc, "video", None), "duration", 0.0) or 0.0))
    scenes = sorted(getattr(doc, "scenes", []), key=lambda s: (s.start, s.end))
    if len(scenes) >= 2:
        regions = [(0.0, scenes[0].end)]
        regions += [(scenes[i + 1].start, scenes[i + 1].end) for i in range(len(scenes) - 1)]
    elif duration > 0:
        regions = [(0.0, duration / 2), (duration / 2, duration)]
    else:
        return []

    changes: list[Change] = []
    for index in range(len(regions) - 1):
        before = _facts_in(doc, *regions[index])
        after = _facts_in(doc, *regions[index + 1])
        ts = regions[index + 1][0]
        before_ocr = {t for o in before["ocr"] for t in _content_terms(o.text)}
        after_ocr = {t for o in after["ocr"] for t in _content_terms(o.text)}
        before_speech = bool(before["transcript"])
        after_speech = bool(after["transcript"])
        evidence = tuple(
            [s.id for s in before["scenes"] + after["scenes"]]
            + [o.id for o in before["ocr"] + after["ocr"]]
            + [u.id for u in before["transcript"] + after["transcript"]]
        )
        if not evidence:
            continue
        before_txt = _clip(" ".join(o.text for o in before["ocr"][:3]) or
                           " ".join(u.text for u in before["transcript"][:2]))
        after_txt = _clip(" ".join(o.text for o in after["ocr"][:3]) or
                          " ".join(u.text for u in after["transcript"][:2]))
        if before_ocr and after_ocr and before_ocr.isdisjoint(after_ocr):
            changes.append(Change(f"chg_{len(changes):04d}", ts, "text_changed",
                                  before_txt, after_txt, evidence))
        elif before_speech != after_speech:
            kind = "speech_started" if after_speech else "speech_ended"
            changes.append(Change(f"chg_{len(changes):04d}", ts, kind,
                                  before_txt, after_txt, evidence))
        elif scenes:
            changes.append(Change(f"chg_{len(changes):04d}", ts, "scene_transition",
                                  before_txt, after_txt, evidence))
    return changes


# ---------------------------------------------------------------------------
# chapters — extractive, deterministic, honestly marked as derived
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chapter:
    id: str
    start: float
    end: float
    title: str
    inferred: bool = True
    signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "start": self.start, "end": self.end,
            "timecode": f"{format_timecode(self.start)} → {format_timecode(self.end)}",
            "title": self.title, "inferred": self.inferred,
            "signals": list(self.signals),
        }


def build_chapters(doc: Any, *, target_s: float = 300.0, max_chapters: int = 12) -> list[Chapter]:
    """Group scenes into ~``target_s`` chapters with extractive keyword titles.

    Titles are the top-3 frequent content terms in the chapter's text — derived, so
    ``inferred`` is always True. Deterministic for a given document.
    """
    duration = max(0.0, float(getattr(getattr(doc, "video", None), "duration", 0.0) or 0.0))
    if duration <= 0:
        return []
    scenes = sorted(getattr(doc, "scenes", []), key=lambda s: (s.start, s.end))
    cuts: list[float] = [0.0]
    if scenes:
        for scene in scenes:
            if scene.start - cuts[-1] >= target_s and len(cuts) < max_chapters:
                cuts.append(scene.start)
    else:
        # No scenes: tile the duration so chapters still tile the timeline.
        n = max(1, min(max_chapters, round(duration / target_s) or 1))
        cuts = [round(i * duration / n, 3) for i in range(n)]
    cuts.append(duration)
    # Merge runt chapters (<60 s) into their predecessor.
    merged = [cuts[0]]
    for cut in cuts[1:-1]:
        if cut - merged[-1] < 60.0 and len(merged) > 0:
            continue
        merged.append(cut)
    merged.append(cuts[-1])

    chapters: list[Chapter] = []
    for index in range(len(merged) - 1):
        start, end = merged[index], merged[index + 1]
        if end - start <= _EPS:
            continue
        facts = _facts_in(doc, start, end)
        corpus = " ".join(u.text for u in facts["transcript"]) + " " + \
                 " ".join(o.text for o in facts["ocr"])
        counts: dict[str, int] = {}
        for term in _terms(corpus):
            if term in _STOPWORDS or term.isdigit() or not any(c.isalpha() for c in term):
                continue  # prices/timestamps are change signals, not chapter titles
            counts[term] = counts.get(term, 0) + 1
        top = sorted(counts, key=lambda t: (-counts[t], t))[:3]
        title = " · ".join(t.capitalize() for t in top) if top else f"Part {index + 1}"
        signals = ["scenes" if scenes else "duration_tiling"]
        if facts["transcript"]:
            signals.append("transcript_terms")
        if facts["ocr"]:
            signals.append("ocr_terms")
        chapters.append(Chapter(f"chapter_{index:04d}", round(start, 3), round(end, 3),
                                title, True, tuple(signals)))
    return chapters


# ---------------------------------------------------------------------------
# UI states — stable on-screen text defines what "screen" persisted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UIState:
    id: str
    start: float
    end: float
    elements: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    layout_signature: str = ""
    app_hint: str | None = None
    app_hint_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "start": self.start, "end": self.end,
            "timecode": f"{format_timecode(self.start)} → {format_timecode(self.end)}",
            "elements": list(self.elements),
            "evidence_ids": list(self.evidence_ids),
            "layout_signature": self.layout_signature,
            "app_hint": self.app_hint,
            "app_hint_confidence": self.app_hint_confidence,
        }


#: Observed keyword → inferred application. Hints only: confidence 0.5, always marked.
_APP_HINTS: tuple[tuple[str, str], ...] = (
    ("browser", "chrome|firefox|safari|edge|address bar|tab|bookmark|url"),
    ("vscode", "vs code|vscode|explorer pane|extensions pane|terminal panel"),
    ("terminal", "terminal|shell|bash|zsh|command prompt|powershell"),
    ("editor", "editor|unsaved|untitled-"),
    ("dashboard", "dashboard|sidebar|widget|metrics overview"),
    ("email", "inbox|compose|subject:|from:"),
    ("video_call", "mute|unmute|share screen|participants|leave meeting"),
)


def _app_hint(elements: tuple[str, ...]) -> tuple[str | None, float | None]:
    blob = " ".join(elements).lower()
    for app, pattern in _APP_HINTS:
        if re.search(pattern, blob):
            return app, 0.5
    return None, None


def _layout_signature(active: list[Any]) -> str:
    """Coarse observed layout: element count + text-position buckets.

    Uses OCR bboxes when present (3-by-3 grid cells), else text-length buckets.
    Two screens with the same signature merely look alike — it is a grouping
    key, not an identity.
    """
    cells: list[str] = []
    for item in active:
        bbox = getattr(item, "bbox_normalized", None) or getattr(item, "bbox", None)
        if bbox and len(bbox) == 4:
            try:
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
                scale = 1.0 if max(bbox) <= 1.5 else 1000.0
                cells.append(f"{min(2, int(x / scale * 3))}{min(2, int(y / scale * 3))}")
                continue
            except (TypeError, ZeroDivisionError):
                pass
        cells.append(f"L{min(9, len(getattr(item, 'text', '')) // 20)}")
    return f"{len(active)}:" + ",".join(sorted(cells))


def ui_states(doc: Any, *, min_state_s: float = 2.0) -> list[UIState]:
    """Maximal intervals with an unchanged set of stable on-screen texts.

    Returns [] when the document has no OCR to define states with — the caller
    should say so rather than invent states.
    """
    runs = [o for o in getattr(doc, "ocr", [])
            if getattr(o, "stable", False) or getattr(o, "frame_count", 1) > 1]
    if not runs:
        return []
    points = sorted({p for o in runs
                     for p in (o.first_frame_ts if o.first_frame_ts is not None else o.start,
                               o.last_frame_ts if o.last_frame_ts is not None else o.end)})
    states: list[UIState] = []
    for a, b in pairwise(points):
        if b - a <= _EPS:
            continue
        active = [o for o in runs
                  if (o.first_frame_ts if o.first_frame_ts is not None else o.start) <= a + _EPS
                  and (o.last_frame_ts if o.last_frame_ts is not None else o.end) >= b - _EPS]
        active.sort(key=lambda o: o.id)
        if not active:
            continue
        active_ids = tuple(o.id for o in active)
        elements = tuple(_clip(o.text, 80) for o in active)
        signature = _layout_signature(active)
        hint, hint_conf = _app_hint(elements)
        if states and states[-1].evidence_ids == active_ids \
                and abs(states[-1].end - a) <= _EPS:
            prev = states.pop()
            states.append(UIState(prev.id, prev.start, b, prev.elements, prev.evidence_ids,
                                  prev.layout_signature, prev.app_hint,
                                  prev.app_hint_confidence))
        else:
            states.append(UIState(
                f"state_{len(states):04d}", a, b, elements, active_ids,
                signature, hint, hint_conf))
    # Fold runt states into their predecessor (a flicker is not a state).
    folded: list[UIState] = []
    for state in states:
        if folded and state.end - state.start < min_state_s:
            prev = folded.pop()
            folded.append(UIState(prev.id, prev.start, state.end, prev.elements,
                                  prev.evidence_ids + state.evidence_ids,
                                  prev.layout_signature, prev.app_hint,
                                  prev.app_hint_confidence))
        else:
            folded.append(state)
    return folded


def ui_timeline(doc: Any, **kw: Any) -> list[dict[str, Any]]:
    """UI states interleaved with change markers, in time order.

    Each entry is ``{"ts", "kind" ("state"|"change"), "label", "ref"}`` — the
    screen-recording storyline: what screen persisted, and what broke it.
    """
    entries: list[dict[str, Any]] = [
        {"ts": state.start, "kind": "state",
         "label": " + ".join(state.elements[:2]) or state.id, "ref": state.id}
        for state in ui_states(doc, **kw)
    ]
    entries.extend(
        {"ts": change.ts, "kind": "change",
         "label": f"{change.change_type}", "ref": change.id}
        for change in detect_changes(doc)
    )
    return sorted(entries, key=lambda entry: (entry["ts"], entry["kind"]))


# ---------------------------------------------------------------------------
# event chains and change chains — temporal sequences, never causal claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventChain:
    """Consecutive events in timestamp order, labeled TEMPORAL_SEQUENCE.

    Adjacency in time is not causation: the chain says B followed A, never that
    A caused B.
    """

    id: str
    events: tuple[Any, ...]
    gaps_s: tuple[float, ...] = ()
    method: str = "temporal_adjacency"

    @property
    def start(self) -> float | None:
        return self.events[0].start if self.events else None

    @property
    def end(self) -> float | None:
        return self.events[-1].end if self.events else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "method": self.method,
            "sequence": "TEMPORAL_SEQUENCE",
            "events": [e.id for e in self.events],
            "gaps_s": list(self.gaps_s),
            "start": self.start, "end": self.end,
        }


def event_chains(doc: Any, *, max_gap_s: float = 120.0) -> list[EventChain]:
    """Split time-ordered events into chains, breaking on gaps over ``max_gap_s``.

    Only chains of ≥2 events are returned — an isolated event is a fact, not a
    sequence, and callers should read it from the event list itself.
    """
    ordered = sorted(getattr(doc, "events", []), key=lambda e: (e.start, e.end, e.id))
    chains: list[EventChain] = []
    current: list[Any] = []
    gaps: list[float] = []
    for event in ordered:
        if current and event.start - current[-1].end > max_gap_s:
            if len(current) >= 2:
                chains.append(EventChain(f"chain_{len(chains):04d}", tuple(current),
                                         tuple(gaps)))
            current, gaps = [], []
        if current:
            gaps.append(round(event.start - current[-1].end, 3))
        current.append(event)
    if len(current) >= 2:
        chains.append(EventChain(f"chain_{len(chains):04d}", tuple(current), tuple(gaps)))
    return chains


@dataclass(frozen=True)
class ChangeChain:
    """state_t0 → change → state_t1 linked by temporal adjacency.

    ``observed`` is True only when both states were observed and the change was
    measured (the only kind this detector emits); the *link* itself is structural
    adjacency, recorded in ``method``.
    """

    id: str
    before_state: UIState | None
    change: Change
    after_state: UIState | None
    method: str = "temporal_adjacency"
    observed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "method": self.method, "observed": self.observed,
            "before_state": self.before_state.id if self.before_state else None,
            "change": self.change.to_dict(),
            "after_state": self.after_state.id if self.after_state else None,
        }


def change_chains(doc: Any, *, tolerance_s: float = 2.0, **kw: Any) -> list[ChangeChain]:
    """Attach each detected change to the UI states it sits between."""
    states = ui_states(doc, **kw)
    chains: list[ChangeChain] = []
    for index, change in enumerate(detect_changes(doc)):
        # A change inside one stable state (e.g. speech turnover) keeps that
        # state on both sides rather than inventing a transition.
        before = next((s for s in reversed(states)
                       if s.end <= change.ts + tolerance_s and s.start <= change.ts), None)
        after = next((s for s in states
                      if s.start >= change.ts - tolerance_s and s.end >= change.ts), None)
        chains.append(ChangeChain(f"cchain_{index:04d}", before, change, after))
    return chains


# ---------------------------------------------------------------------------
# temporal query parsing — small, explicit, no NLP framework
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalQuery:
    """A query split into *what* (anchor) and *when* (relation/range/first)."""

    raw: str
    anchor: str
    relation: TemporalRelation | None = None
    start: float | None = None
    end: float | None = None
    first_only: bool = False
    last_only: bool = False
    anchor_b: str | None = None
    max_gap_s: float | None = None
    radius_s: float = 5.0

    @property
    def is_temporal(self) -> bool:
        return self.relation is not None or self.start is not None \
            or self.end is not None or self.first_only or self.last_only \
            or self.anchor_b is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw, "anchor": self.anchor,
            "relation": self.relation.value if self.relation else None,
            "start": self.start, "end": self.end, "first_only": self.first_only,
            "last_only": self.last_only, "anchor_b": self.anchor_b,
            "max_gap_s": self.max_gap_s, "radius_s": self.radius_s,
        }


_TC_RE = re.compile(r"\b(?:(\d+):)?([0-5]?\d):([0-5]\d(?:\.\d+)?)\b")
_TIMECODE = r"(?:\d+:)?[0-5]?\d:[0-5]\d(?:\.\d+)?"
_BETWEEN_RE = re.compile(
    rf"\bbetween\s+({_TIMECODE})\s+and\s+({_TIMECODE})", re.IGNORECASE)
_BEFORE_RE = re.compile(
    r"(?:^|\b)(?:what happened|what was|show me|find)\s+"
    r"(?:(?:just|immediately|right|shortly)\s+)?before\s+(.+?)\??$", re.IGNORECASE)
_AFTER_RE = re.compile(
    r"(?:^|\b)(?:what happened|what was|show me|find)\s+"
    r"(?:(?:just|immediately|right|shortly)\s+)?after\s+(.+?)\??$", re.IGNORECASE)
_WHILE_RE = re.compile(r"\bwhile\s+(.+?)\??$", re.IGNORECASE)
_DURING_RE = re.compile(r"\bduring\s+(.+?)\??$", re.IGNORECASE)
_AROUND_RE = re.compile(r"\baround\s+(.+?)\??$", re.IGNORECASE)
_FIRST_RE = re.compile(r"\bfirst\b", re.IGNORECASE)
_LAST_RE = re.compile(r"\b(last|final|most recent|disappeared|disappear)\b", re.IGNORECASE)
_GAP_WORDS = (
    (re.compile(r"\bimmediately\b|\bright\s+(after|before)\b", re.IGNORECASE), 10.0),
    (re.compile(r"\bshortly\b", re.IGNORECASE), 30.0),
)
_BETWEEN_ANCHORS_RE = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)\??$", re.IGNORECASE)
_FROM_TO_RE = re.compile(
    rf"\bfrom\s+({_TIMECODE})\s+to\s+({_TIMECODE})", re.IGNORECASE)
_QUESTION_LEAD_RE = re.compile(
    r"^(when did|where was|what was|which|who|show me|find|tell me)\b\s*", re.IGNORECASE)


def _tc_to_seconds(text: str) -> float:
    return parse_timecode(text)


def _gap_hint(raw: str) -> float | None:
    """'immediately after' → 10 s, 'shortly before' → 30 s, else None."""
    for pattern, seconds in _GAP_WORDS:
        if pattern.search(raw):
            return seconds
    return None


def parse_temporal_query(query: str) -> TemporalQuery:
    """Split a natural query into anchor text + temporal constraint.

    Plain queries pass through untouched (``is_temporal`` False) — this only fires
    on explicit temporal phrasing, so normal search behavior never changes.
    """
    raw = query.strip()
    anchor = raw
    start: float | None = None
    end: float | None = None
    relation: TemporalRelation | None = None

    between = _BETWEEN_RE.search(raw)
    if between:
        try:
            start, end = _tc_to_seconds(between.group(1)), _tc_to_seconds(between.group(2))
        except ValueError:
            start = end = None
        if start is not None and end is not None and end > start:
            anchor = (raw[:between.start()] + raw[between.end():]).strip(" ?,-")
            anchor = re.sub(r"(?i)\b(what changed|what happened)\b", "", anchor).strip(" ?,-")
            return TemporalQuery(raw, anchor or raw, None, start, end)

    from_to = _FROM_TO_RE.search(raw)
    if from_to:
        try:
            start, end = _tc_to_seconds(from_to.group(1)), _tc_to_seconds(from_to.group(2))
        except ValueError:
            start = end = None
        if start is not None and end is not None and end > start:
            anchor = (raw[:from_to.start()] + raw[from_to.end():]).strip(" ?,-")
            return TemporalQuery(raw, anchor or raw, None, start, end)

    between_anchors = _BETWEEN_ANCHORS_RE.search(raw)
    if between_anchors and not _TC_RE.search(raw):
        # "what happened between the login and the error": two text anchors.
        return TemporalQuery(raw, between_anchors.group(1).strip(" ?"),
                             None, None, None, False, False,
                             between_anchors.group(2).strip(" ?"))

    before = _BEFORE_RE.search(raw)
    if before:
        return TemporalQuery(raw, before.group(1).strip(" ?"), TemporalRelation.BEFORE,
                             None, None, False, False, None, _gap_hint(raw))
    after = _AFTER_RE.search(raw)
    if after:
        return TemporalQuery(raw, after.group(1).strip(" ?"), TemporalRelation.AFTER,
                             None, None, False, False, None, _gap_hint(raw))
    while_m = _WHILE_RE.search(raw)
    if while_m and len(raw.split()) > 3:
        anchor_text = while_m.group(1).strip(" ?")
        head = raw[:while_m.start()].strip()
        return TemporalQuery(raw, f"{head} {anchor_text}".strip(),
                             TemporalRelation.CO_OCCURS)
    during_m = _DURING_RE.search(raw)
    if during_m and len(raw.split()) > 2:
        anchor_text = during_m.group(1).strip(" ?")
        head = raw[:during_m.start()].strip()
        return TemporalQuery(raw, f"{head} {anchor_text}".strip() or anchor_text,
                             TemporalRelation.DURING)
    around_m = _AROUND_RE.search(raw)
    if around_m and not _TC_RE.search(raw) and len(raw.split()) > 2:
        return TemporalQuery(raw, around_m.group(1).strip(" ?"),
                             TemporalRelation.CO_OCCURS, None, None,
                             False, False, None, None, 30.0)
    if _FIRST_RE.search(raw):
        cleaned = _FIRST_RE.sub("", raw)
        cleaned = _QUESTION_LEAD_RE.sub("", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?")
        return TemporalQuery(raw, cleaned or raw, None, None, None, True)
    if _LAST_RE.search(raw):
        cleaned = _LAST_RE.sub("", raw)
        cleaned = _QUESTION_LEAD_RE.sub("", cleaned).strip()
        cleaned = re.sub(r"(?i)\b(occurrence|appearance|mention|time)\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?")
        return TemporalQuery(raw, cleaned or raw, None, None, None, False, True)

    # Bare timecodes constrain the range ("pricing at 12:20", "12:20-13:10").
    tcs = [m.group(0) for m in _TC_RE.finditer(raw)]
    if tcs:
        try:
            secs = sorted(_tc_to_seconds(t) for t in tcs)
        except ValueError:
            secs = []
        if secs:
            anchor = _TC_RE.sub("", raw).strip(" ?,-")
            if len(secs) >= 2:
                return TemporalQuery(raw, anchor or raw, None, secs[0], secs[-1])
            return TemporalQuery(raw, anchor or raw, None, max(0.0, secs[0] - 30.0),
                                 secs[0] + 30.0)
    return TemporalQuery(raw, anchor, relation, start, end)


__all__ = [
    "Change",
    "ChangeChain",
    "Chapter",
    "EventChain",
    "TemporalQuery",
    "TemporalRelation",
    "TemporalWindow",
    "UIState",
    "build_chapters",
    "change_chains",
    "detect_changes",
    "event_chains",
    "merge_windows",
    "parse_temporal_query",
    "relate",
    "ui_states",
    "ui_timeline",
    "window_around",
    "window_at",
]
