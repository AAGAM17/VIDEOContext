"""Boundary bookkeeping shared by every scene detector.

Detectors differ in how they *find* cut points; turning cut points into a clean, contiguous
list of scenes is identical work and lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...interfaces import SceneSpan


@dataclass(slots=True)
class Boundary:
    """A candidate cut at ``ts``, with the signals that voted for it."""

    ts: float
    score: float = 0.0
    signals: list[str] = field(default_factory=list)

    def merge(self, other: Boundary) -> None:
        self.score = max(self.score, other.score)
        for signal in other.signals:
            if signal not in self.signals:
                self.signals.append(signal)


def dedupe_boundaries(boundaries: list[Boundary], *, tolerance: float = 0.05) -> list[Boundary]:
    """Collapse boundaries that different signals reported at essentially the same instant.

    This is where multi-signal detection pays off: a cut confirmed by both a frame-difference
    spike and a black frame keeps both labels, so the document records *why* it is a boundary.
    """
    if not boundaries:
        return []
    ordered = sorted(boundaries, key=lambda b: b.ts)
    out = [Boundary(ordered[0].ts, ordered[0].score, list(ordered[0].signals))]
    for candidate in ordered[1:]:
        if candidate.ts - out[-1].ts <= tolerance:
            out[-1].merge(candidate)
        else:
            out.append(Boundary(candidate.ts, candidate.score, list(candidate.signals)))
    return out


def enforce_min_duration(
    boundaries: list[Boundary],
    *,
    min_duration: float,
    duration: float,
    strong_score: float = 0.9,
) -> list[Boundary]:
    """Drop boundaries that would create a scene shorter than ``min_duration``.

    Two cases arbitrate differently when boundaries are too close together:

    *Transitions.* Two **strong** cuts a fraction of a second apart are one event — a flash,
    wipe or cross-fade — not two scenes. The real content boundary is the *second* cut, where
    the transition ends, so the later one wins and the pair is labelled ``flash``. Keeping the
    first instead would date the incoming scene to the transition artifact.

    *Everything else.* The better-evidenced boundary wins (more signals, then higher score),
    so a hard cut is never discarded in favour of a weak one that merely came first.
    """
    kept: list[Boundary] = []
    for candidate in dedupe_boundaries(boundaries):
        if candidate.ts <= 0.0 or candidate.ts >= duration:
            continue
        if not kept:
            kept.append(candidate)
            continue
        previous = kept[-1]
        if candidate.ts - previous.ts >= min_duration:
            kept.append(candidate)
            continue

        if previous.score >= strong_score and candidate.score >= strong_score:
            merged = Boundary(candidate.ts, candidate.score, list(candidate.signals))
            for signal in (*previous.signals, "flash"):
                if signal not in merged.signals:
                    merged.signals.append(signal)
            kept[-1] = merged
            continue

        if (len(candidate.signals), candidate.score) > (len(previous.signals), previous.score):
            kept[-1] = candidate
    return kept


def boundaries_to_spans(
    boundaries: list[Boundary], duration: float, *, min_duration: float = 0.0
) -> list[SceneSpan]:
    """Convert cut points into contiguous, gapless scenes covering ``[0, duration)``.

    Gapless coverage is a spec guarantee: ``doc.at(t)`` must always land in exactly one
    scene, so scenes tile the timeline rather than marking only "interesting" regions.
    """
    if duration <= 0:
        return []
    if min_duration > 0:
        cuts = enforce_min_duration(
            boundaries, min_duration=min_duration, duration=duration
        )
    else:
        cuts = dedupe_boundaries(boundaries)

    spans: list[SceneSpan] = []
    start = 0.0
    incoming: list[str] = ["start"]
    incoming_score: float | None = None
    for cut in cuts:
        if cut.ts <= start:
            continue
        spans.append(
            SceneSpan(
                start=round(start, 3),
                end=round(cut.ts, 3),
                score=incoming_score,
                signals=list(incoming),
                keyframe_ts=round(start + min(0.5, (cut.ts - start) / 2), 3),
            )
        )
        start = cut.ts
        incoming = list(cut.signals) or ["cut"]
        incoming_score = cut.score
    if duration - start > 1e-3:
        spans.append(
            SceneSpan(
                start=round(start, 3),
                end=round(duration, 3),
                score=incoming_score,
                signals=list(incoming),
                keyframe_ts=round(start + min(0.5, (duration - start) / 2), 3),
            )
        )
    return spans


def budget_boundaries(
    candidates: list[Boundary],
    *,
    duration: float,
    absolute_threshold: float,
    max_per_minute: float,
) -> list[Boundary]:
    """Select cuts under a density budget instead of one universal threshold.

    A slide deck's cuts score ~0.1 because consecutive slides share a background; handheld
    footage scores >0.3 on ordinary camera motion. No single absolute threshold serves both,
    so anything at or above ``absolute_threshold`` is always a cut, and the remaining budget
    is spent on the highest-scoring weak candidates. Recall stays high on low-contrast
    content without letting noisy content emit thousands of scenes.
    """
    strong = [c for c in candidates if c.score >= absolute_threshold]
    weak = [c for c in candidates if c.score < absolute_threshold]
    budget = max(1, round((duration / 60.0) * max_per_minute))
    room = max(0, budget - len(strong))
    if room and weak:
        weak.sort(key=lambda c: c.score, reverse=True)
        strong.extend(weak[:room])
    return sorted(strong, key=lambda c: c.ts)


__all__ = [
    "Boundary",
    "boundaries_to_spans",
    "budget_boundaries",
    "dedupe_boundaries",
    "enforce_min_duration",
]
