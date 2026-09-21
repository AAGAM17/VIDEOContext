"""Multi-video intelligence — collections without merging files.

A collection is an index *over* ``.vctx`` documents, never a merged super-document:
each member keeps its own identity, and every returned span is tagged with the
``video_id`` it came from so provenance survives the merge. Timestamps are
video-local unless a query explicitly asks otherwise — two recordings share no
clock, and the API never pretends they do.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .entities import extract_entities, normalize_name
from .logging import get_logger
from .packages import build_package
from .retrieval.query import EvidenceSpan, Retriever
from .temporal import build_chapters, detect_changes, ui_states

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
        self._entities: dict[str, list[Any]] | None = None

    @property
    def video_ids(self) -> list[str]:
        return list(self._docs)

    def _entities_for(self, video_id: str) -> list[Any]:
        """Entities per video, extracted once and memoized (pure over the doc)."""
        if self._entities is None:
            self._entities = {vid: extract_entities(doc)
                              for vid, doc in self._docs.items()}
        return self._entities[video_id]

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

    def entities(self) -> dict[str, dict[str, Any]]:
        """Aggregate entities per video: ``{video_id: {name: entity dict}}``."""
        return {video_id: {entity.name: entity.to_dict()
                           for entity in self._entities_for(video_id)}
                for video_id in self._docs}

    def link_entities(self) -> dict[str, list[dict[str, Any]]]:
        """Collection-level identity over video-local evidence.

        ``{canonical name: [{video_id, entity}]}}`` — local occurrences and
        provenance stay attached to each entry; nothing is merged away.
        """
        linked: dict[str, list[dict[str, Any]]] = {}
        for video_id in self._docs:
            for entity in self._entities_for(video_id):
                linked.setdefault(normalize_name(entity.name), []).append(
                    {"video_id": video_id, "entity": entity.to_dict()})
        return linked

    def occurrences(self, name: str) -> list[dict[str, Any]]:
        """Every occurrence of ``name`` across videos, in (video, time) order."""
        key = normalize_name(name)
        out: list[dict[str, Any]] = []
        for video_id in self._docs:
            for entity in self._entities_for(video_id):
                if normalize_name(entity.name) != key:
                    continue
                for occurrence in entity.timeline():
                    out.append({"video_id": video_id, "entity": entity.name,
                                "type": entity.type, **occurrence.to_dict()})
        return sorted(out, key=lambda o: (o["video_id"], o["start"]))

    def videos_with(self, *names: str) -> dict[str, list[str]]:
        """Which videos contain each name — answers 'which videos contain X?'."""
        return {name: sorted({video_id for video_id in self._docs
                              for entity in self._entities_for(video_id)
                              if normalize_name(entity.name) == normalize_name(name)})
                for name in names}

    def videos_with_all(self, *names: str) -> list[str]:
        """Videos containing every named entity ('both A and B')."""
        sets = [set(videos) for videos in self.videos_with(*names).values()]
        return sorted(set.intersection(*sets)) if sets else []

    def events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Events across videos, tagged and time-ordered per video."""
        out: list[dict[str, Any]] = []
        for video_id, doc in self._docs.items():
            for event in getattr(doc, "events", []):
                if event_type is not None and event.type != event_type:
                    continue
                out.append({"video_id": video_id, "id": event.id, "type": event.type,
                            "start": event.start, "end": event.end,
                            "description": event.description,
                            "refs": dict(getattr(event, "refs", {}) or {})})
        return sorted(out, key=lambda e: (e["video_id"], e["start"]))

    def changes(self) -> dict[str, list[dict[str, Any]]]:
        """Changes per video — 'how did X change across recordings' starts here."""
        return {video_id: [c.to_dict() for c in detect_changes(doc)]
                for video_id, doc in self._docs.items()}

    def timeline(self, start: float = 0.0, end: float | None = None, *,
                 top_k: int = 100) -> CollectionResult:
        """Per-video timelines merged with video tags (timestamps stay local)."""
        per_video: list[ScoredVideo] = []
        for video_id, retriever in self._retrievers.items():
            result = retriever.timeline(start, end, top_k=0)
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
                per_video.append(ScoredVideo(video_id, 1.0, tagged))
        merged = [s for v in per_video for s in v.spans]
        total = len(merged)
        kept = tuple(merged[:top_k] if top_k > 0 else merged)
        return CollectionResult(f"{start}-{end}", kept, tuple(per_video),
                                len(self._retrievers), total)

    def context(self, task: str, *, max_spans: int = 10,
                per_video_k: int = 5) -> Any:
        """Collection-level context package: per-video evidence under one budget.

        Searches each member, merges by score, then optimizes once globally so
        the budget binds the whole answer, not each video separately.
        """
        result = self.search(task, top_k=max_spans * 2, per_video_k=per_video_k)
        return build_package(
            _CollectionDoc(self._docs), task, list(result.spans),
            intent="multi_video_search", max_spans=max_spans,
            budget={"max_spans": max_spans, "per_video_k": per_video_k},
            warnings=[f"timestamps are video-local across {len(self._docs)} videos"])


class _CollectionDoc:
    """Minimal document facade so packages can describe a collection."""

    def __init__(self, docs: dict[str, Any]) -> None:
        from types import SimpleNamespace

        self.id = f"collection:{len(docs)}"
        self.vctx_version = "1.0"
        self.video = SimpleNamespace(duration=0.0)
        self.producer = None
        self.source = None
        self.stages: list[Any] = []


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
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    uncertain: tuple[str, ...] = ()
    structure_a: dict[str, int] = field(default_factory=dict)
    structure_b: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_a": self.video_a, "video_b": self.video_b,
            "shared_entities": list(self.shared_entities),
            "unique_a": list(self.unique_a), "unique_b": list(self.unique_b),
            "events_a": dict(self.events_a), "events_b": dict(self.events_b),
            "coverage_a": dict(self.coverage_a), "coverage_b": dict(self.coverage_b),
            "added": list(self.added), "removed": list(self.removed),
            "changed": list(self.changed), "unchanged": list(self.unchanged),
            "uncertain": list(self.uncertain),
            "structure_a": dict(self.structure_a),
            "structure_b": dict(self.structure_b),
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


def _structure(doc: Any) -> dict[str, int]:
    return {
        "chapters": len(build_chapters(doc)),
        "changes": len(detect_changes(doc)),
        "states": len(ui_states(doc)),
    }


def compare(video_a: str, doc_a: Any, video_b: str, doc_b: Any) -> CompareResult:
    """Evidence-based comparison of two videos.

    No embeddings, no LLM — set overlap over extracted entities plus event-type
    histograms, chapter/change/state counts. ``changed`` means a shared entity
    whose occurrence count differs twofold or more, or whose modalities differ
    (a factual temporal difference, never labeled a regression). ``uncertain`` names shared
    entities where either side's link is ambiguous. Every name traces back to
    timestamped occurrences via the entity timelines.
    """
    entities_a = {e.name: e for e in extract_entities(doc_a)}
    entities_b = {e.name: e for e in extract_entities(doc_b)}
    names_a, names_b = set(entities_a), set(entities_b)
    changed, unchanged, uncertain = [], [], []
    for name in sorted(names_a & names_b):
        a, b = entities_a[name], entities_b[name]
        if a.ambiguous or b.ambiguous:
            uncertain.append(name)
        elif max(len(a.occurrences), 1) >= 2 * max(len(b.occurrences), 1) \
                or max(len(b.occurrences), 1) >= 2 * max(len(a.occurrences), 1) \
                or set(a.linked_modalities) != set(b.linked_modalities):
            changed.append(name)
        else:
            unchanged.append(name)
    return CompareResult(
        video_a, video_b,
        tuple(sorted(names_a & names_b)),
        tuple(sorted(names_a - names_b)),
        tuple(sorted(names_b - names_a)),
        _event_histogram(doc_a), _event_histogram(doc_b),
        _coverage(doc_a), _coverage(doc_b),
        tuple(sorted(names_b - names_a)), tuple(sorted(names_a - names_b)),
        tuple(changed), tuple(unchanged), tuple(uncertain),
        _structure(doc_a), _structure(doc_b),
    )


__all__ = ["CollectionIndex", "CollectionResult", "CompareResult", "ScoredVideo", "compare"]
