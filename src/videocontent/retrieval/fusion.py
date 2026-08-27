"""Fusion — combining ranked lists into one answer, and the temporal boost that is the point.

Two jobs live here.

**Reciprocal-rank fusion.** The retrievers produce lists whose scores are not comparable:
BM25 is unbounded and corpus-dependent, cosine similarity is bounded and model-dependent, and
each modality's BM25 has its own statistics. Any attempt to combine them by arithmetic on the
raw scores encodes an arbitrary exchange rate. RRF discards the magnitudes and keeps only the
ranks — ``Σ weight / (k + rank)`` — which is why it survives adding an embedding retriever
later without a re-tuning pass. The consequence to know: a fused score is a *relative*
quantity, meaningful for ordering results within one query and meaningless compared across
queries or corpora.

**Temporal co-occurrence.** This is the part a text search engine cannot do, and the reason
this system indexes a video rather than a transcript. When a query's words appear in the
transcript *and* on screen at the same moment, that moment is stronger evidence than either
signal alone: the speaker was talking about the thing that was being shown. The bonus is flat
rather than scaled by how many modalities agree, because vision notes and rule-derived events
co-occur with speech as a matter of course, and a multiplier would let that structural
overlap outrank a genuine match.

Both operations preserve the invariant the whole design rests on: a candidate's span is always
the union of spans that exist in the document. Fusion re-orders and merges evidence; it never
invents a timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .index import Record

#: RRF's rank offset. 60 is the value from the original paper and the one every comparable
#: implementation uses; at this size it makes the difference between ranks 1 and 2 modest and
#: the difference between 30 and 31 negligible, which is the intended shape.
RRF_K = 60

#: How far apart a spoken match and an on-screen match may be and still count as the same
#: moment. Speech about a slide routinely starts a beat before the slide appears, and OCR span
#: ends are interpolated to a sampling-interval midpoint, so requiring strict overlap would
#: miss the co-occurrence this exists to reward.
COOCCURRENCE_TOLERANCE_S = 1.0


@dataclass(frozen=True)
class Candidate:
    """One piece of evidence under consideration, backed by the records that produced it."""

    records: tuple[Record, ...]
    score: float
    matched: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    coverage: float = 0.0
    """Share of the query's IDF mass this candidate matched, carried through fusion.

    RRF gives the top record of every modality the same score, so without this the first
    result of a query is chosen by timestamp — a match on one word of five can lead a match on
    five of five purely by appearing earlier in the video. Coverage breaks those ties without
    contaminating ``score``, which stays a pure rank-fusion quantity.
    """

    @property
    def modality(self) -> str:
        return self.records[0].modality

    @property
    def start(self) -> float:
        return min(record.start for record in self.records)

    @property
    def end(self) -> float:
        return max(record.end for record in self.records)

    @property
    def text(self) -> str:
        return " ".join(record.text for record in self.records if record.text)


def rrf(
    lists: Sequence[tuple[float, Sequence[str]]],
    *,
    k: int = RRF_K,
) -> dict[str, float]:
    """Weighted reciprocal-rank fusion of ``(weight, ranked keys)`` lists.

    A key absent from a list contributes nothing from it, rather than contributing a
    worst-possible rank: a record that a retriever never saw is not the same as one it ranked
    last, and penalising absence would punish every modality-specific match.
    """
    fused: dict[str, float] = {}
    for weight, keys in lists:
        if weight <= 0.0:
            continue
        for rank, key in enumerate(keys, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
    return fused


def _near(a: Candidate, b: Candidate, tolerance: float) -> bool:
    return a.start - tolerance < b.end and b.start - tolerance < a.end


def boost_cooccurrence(
    candidates: Sequence[Candidate],
    *,
    boost: float,
    tolerance: float = COOCCURRENCE_TOLERANCE_S,
) -> list[Candidate]:
    """Reward candidates corroborated by a *different* modality at the same moment.

    Comparisons are against the incoming set, so the bonus is not itself compounded by other
    candidates having just received it.
    """
    if boost <= 0.0 or len(candidates) < 2:
        return list(candidates)

    boosted: list[Candidate] = []
    for candidate in candidates:
        others = sorted(
            {
                other.modality
                for other in candidates
                if other.modality != candidate.modality and _near(candidate, other, tolerance)
            }
        )
        if not others:
            boosted.append(candidate)
            continue
        boosted.append(
            replace(
                candidate,
                score=candidate.score * (1.0 + boost),
                reasons=(*candidate.reasons, f"also matched in {', '.join(others)}"),
            )
        )
    return boosted


def merge_adjacent(candidates: Sequence[Candidate], *, gap: float) -> list[Candidate]:
    """Collapse same-modality candidates separated by at most ``gap`` seconds.

    Without this, one sentence split across three subtitle cues occupies three of the ten
    result slots and pushes the corroborating on-screen text off the page — the ranking looks
    broken when the real problem is that the same evidence was counted three times. The merged
    candidate keeps the best score rather than the sum, because concatenating adjacent text
    does not make the match stronger, and keeps every source id so the evidence stays
    resolvable to individual facts.
    """
    if gap < 0.0 or len(candidates) < 2:
        return list(candidates)

    by_modality: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_modality.setdefault(candidate.modality, []).append(candidate)

    merged: list[Candidate] = []
    for group in by_modality.values():
        group.sort(key=lambda c: (c.start, c.end))
        current = group[0]
        for candidate in group[1:]:
            if candidate.start - current.end <= gap:
                # Deduped by key rather than by value: a record is identified by its id, and
                # `Record` carries a dict field so it is not hashable in the first place.
                unique = {record.key: record for record in (*current.records, *candidate.records)}
                records = sorted(unique.values(), key=lambda r: (r.start, r.key))
                current = Candidate(
                    records=tuple(records),
                    score=max(current.score, candidate.score),
                    matched=tuple(dict.fromkeys((*current.matched, *candidate.matched))),
                    reasons=tuple(dict.fromkeys((*current.reasons, *candidate.reasons))),
                    coverage=max(current.coverage, candidate.coverage),
                )
            else:
                merged.append(current)
                current = candidate
        merged.append(current)

    merged.sort(key=lambda c: (-c.score, -c.coverage, c.start))
    return merged


__all__ = [
    "COOCCURRENCE_TOLERANCE_S",
    "RRF_K",
    "Candidate",
    "boost_cooccurrence",
    "merge_adjacent",
    "rrf",
]
