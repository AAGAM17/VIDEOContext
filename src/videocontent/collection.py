"""Multi-video intelligence — collections without merging files.

A collection is an index *over* ``.vctx`` documents, never a merged super-document:
each member keeps its own identity, and every returned span is tagged with the
``video_id`` it came from so provenance survives the merge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .entities import extract_entities
from .logging import get_logger
from .retrieval.query import EvidenceSpan, Retriever

log = get_logger("collection")


@dataclass(frozen=True)
class ScoredVideo:
    video_id: str
    score: float
    spans: tuple[EvidenceSpan, ...] = ()


@dataclass(frozen=True)
class CollectionResult:
    query: str
    spans: tuple[EvidenceSpan, ...] = ()
    per_video: tuple[ScoredVideo, ...] = ()
    videos_searched: int = 0
    total: int = 0

    def __iter__(self):  # iterate spans, like SearchResult
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "spans": [s.to_dict() for s in self.spans],
            "per_video": [
                {"video_id": v.video_id, "score": v.score,
                 "spans": [s.to_dict() for s in v.spans]}
                for v in self.per_video
            ],
            "videos_searched": self.videos_searched,
            "total": self.total,
        }


class CollectionIndex:
    """Search across videos while preserving per-video provenance."""

    def __init__(self, docs: Mapping[str, Any]) -> None:
        self._docs = dict(docs)
        self._retrievers = {vid: Retriever(doc) for vid, doc in self._docs.items()}

    @property
    def video_ids(self) -> list[str]:
        return list(self._docs)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        per_video_k: int = 5,
        modalities: Any = None,
    ) -> CollectionResult:
        """Search every member; merge by score; tag each span with its video."""
        per_video: list[ScoredVideo] = []
        for video_id, retriever in self._retrievers.items():
            result = retriever.search(query, modalities=modalities, top_k=per_video_k)
            tagged = tuple(
                EvidenceSpan(
                    start=s.start, end=s.end, modality=s.modality, text=s.text,
                    score=s.score, ref_ids=s.ref_ids, segment_ids=s.segment_ids,
                    matched_terms=s.matched_terms, reason=f"{s.reason} [from {video_id}]",
                    confidence=s.confidence, language=s.language, kind=s.kind,
                    video_id=video_id,
                )
                for s in result.spans
            )
            if tagged:
                per_video.append(ScoredVideo(video_id, max(s.score for s in tagged), tagged))
        per_video.sort(key=lambda v: -v.score)
        merged = sorted((s for v in per_video for s in v.spans),
                        key=lambda s: -s.score)
        total = len(merged)
        kept = tuple(merged[:top_k] if top_k > 0 else merged)
        return CollectionResult(query, kept, tuple(per_video), len(self._retrievers), total)


@dataclass(frozen=True)
class CompareResult:
    video_a: str
    video_b: str
    shared_entities: tuple[str, ...] = ()
    unique_a: tuple[str, ...] = ()
    unique_b: tuple[str, ...] = ()
    events_a: dict[str, int] = field(default_factory=dict)
    events_b: dict[str, int] = field(default_factory=dict)
    coverage_a: dict[str, int] = field(default_factory=dict)
    coverage_b: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_a": self.video_a, "video_b": self.video_b,
            "shared_entities": list(self.shared_entities),
            "unique_a": list(self.unique_a), "unique_b": list(self.unique_b),
            "events_a": dict(self.events_a), "events_b": dict(self.events_b),
            "coverage_a": dict(self.coverage_a), "coverage_b": dict(self.coverage_b),
        }


def _event_histogram(doc: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in getattr(doc, "events", []):
        counts[event.type] = counts.get(event.type, 0) + 1
    return counts


def _coverage(doc: Any) -> dict[str, int]:
    return {
        "transcript": len(getattr(doc, "transcript", [])),
        "ocr": len(getattr(doc, "ocr", [])),
        "vision": len(getattr(doc, "vision", [])),
        "events": len(getattr(doc, "events", [])),
        "scenes": len(getattr(doc, "scenes", [])),
        "segments": len(getattr(doc, "segments", [])),
    }


def compare(video_a: str, doc_a: Any, video_b: str, doc_b: Any) -> CompareResult:
    """Evidence-based comparison of two videos: shared/unique entities, event mix.

    No embeddings, no LLM — set overlap over extracted entities plus event-type
    histograms. Every name traces back to timestamped occurrences via the entity
    timelines.
    """
    names_a = {e.name for e in extract_entities(doc_a)}
    names_b = {e.name for e in extract_entities(doc_b)}
    return CompareResult(
        video_a, video_b,
        tuple(sorted(names_a & names_b)),
        tuple(sorted(names_a - names_b)),
        tuple(sorted(names_b - names_a)),
        _event_histogram(doc_a), _event_histogram(doc_b),
        _coverage(doc_a), _coverage(doc_b),
    )


__all__ = ["CollectionIndex", "CollectionResult", "CompareResult", "ScoredVideo", "compare"]
