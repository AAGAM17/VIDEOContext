"""The query API — search a ``.vctx`` document and get back evidence, not prose.

This module is the contract the rest of the project is built to keep: **every timestamp
returned here was copied out of the document, never generated.** A span's ``start`` and ``end``
come from the records that produced it, its ``ref_ids`` name those records, and a caller can
follow them back into the document and check. That is what makes "when was the pricing slide
shown?" a verifiable answer rather than a plausible one, and it is why the answer type is a
span with references instead of a string.

The shape:

    query ─┬─▶ lexical BM25, per modality        (always available)
           ├─▶ vector cosine similarity          (when embeddings exist)
           ├─▶ time / modality filters
           └─▶ RRF + temporal co-occurrence + adjacent merge
                         │
                         ▼
                   SearchResult ─▶ EvidenceSpan[]

An embedding retriever slots into the same fusion when one is configured; its absence is
reported in ``SearchResult.notes`` rather than hidden, because "no semantic matches" and "no
semantic retriever" are different answers to the same empty result and the user needs to know
which one they got.

``at()`` is the other half of the API and is not a search: "what was visible on screen at
03:21" is a timeline lookup, and running it through a ranker would be a category error.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import RetrievalConfig
from ..logging import get_logger
from ..schema.v1 import VideoContextDocument
from ..temporal import TemporalQuery, TemporalRelation, parse_temporal_query
from ..timecode import format_span, format_timecode
from .fusion import Candidate, boost_cooccurrence, merge_adjacent, rrf
from .index import MODALITIES, Record, build_records
from .lexical import LexicalHit, LexicalIndex

log = get_logger("retrieval.query")


@dataclass(frozen=True)
class EvidenceSpan:
    """A moment in the video that supports an answer, with its sources named.

    ``ref_ids`` is not optional decoration. A consumer that shows a user "03:21 — Pricing" must
    be able to prove it, and an agent that puts this span in a prompt must be able to cite it.
    Both need the ids.
    """

    start: float
    end: float
    modality: str
    text: str
    score: float
    ref_ids: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    reason: str = ""
    confidence: float | None = None
    language: str | None = None
    kind: str | None = None
    video_id: str | None = None
    """Owning video in collection search; None for single-video retrieval."""

    @property
    def timecode(self) -> str:
        """``HH:MM:SS.mmm`` of the span's start — what a user is shown and can seek to."""
        return format_timecode(self.start)

    @property
    def span(self) -> str:
        return format_span(self.start, self.end)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready, with the rendered timecode included for consumers that only display."""
        data = asdict(self)
        data.update(ref_ids=list(self.ref_ids), segment_ids=list(self.segment_ids),
                    matched_terms=list(self.matched_terms), timecode=self.timecode)
        return data


@dataclass(frozen=True)
class SearchResult:
    """Spans plus the story of how they were found.

    Iterating a result yields its spans, so the common case reads as
    ``for hit in video.search("pricing")`` while the diagnostics stay available to anyone
    debugging a query that went wrong.
    """

    query: str
    spans: tuple[EvidenceSpan, ...] = ()
    modalities: tuple[str, ...] = ()
    total: int = 0
    """Candidates found before ``top_k`` truncation — how much was left on the floor."""

    took_ms: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)
    """Retrievers that did not participate, and why. Never silently empty-handed."""

    def __iter__(self) -> Iterator[EvidenceSpan]:
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, index: int) -> EvidenceSpan:
        return self.spans[index]

    def __bool__(self) -> bool:
        return bool(self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "spans": [span.to_dict() for span in self.spans],
            "modalities": list(self.modalities),
            "total": self.total,
            "took_ms": round(self.took_ms, 2),
            "notes": list(self.notes),
        }


class Retriever:
    """A document prepared for querying.

    Construction does the work that does not depend on the query — projecting facts into
    records and inverting them — so repeated searches over one document pay for it once. Hold
    one of these if you are running more than a single query; ``search(doc, ...)`` builds and
    discards one if you are not.

    The index always covers every modality present in the document. ``RetrievalConfig.
    modalities`` and the per-call argument *narrow* what is searched; neither can widen it past
    what the document contains.
    """

    def __init__(
        self,
        doc: VideoContextDocument,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.doc = doc
        self.config = config or RetrievalConfig()
        self.records: tuple[Record, ...] = tuple(build_records(doc, MODALITIES))
        self._lexical = LexicalIndex(self.records)
        self._by_key = {record.key: record for record in self.records}
        self._vector_store = None
        self._vector_ready = False
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """Initialise vector store if embeddings are present in the document."""
        if self.doc.embeddings is None:
            return
        if self.doc.embeddings.external:
            # External store (Qdrant, etc.) — would need a separate client
            return
        if not self.doc.embeddings.vectors:
            return
        try:
            from ..storage.faiss_store import FAISSStore

            dim = self.doc.embeddings.dim or 384
            self._vector_store = FAISSStore(dim=dim)
            # Add all vectors
            ids = list(self.doc.embeddings.vectors.keys())
            vectors = list(self.doc.embeddings.vectors.values())
            self._vector_store.add(ids, vectors)
            self._vector_ready = True
            log.info(
                "retrieval.vector_ready",
                extra={"vectors": len(ids), "dim": dim},
            )
        except Exception as exc:
            log.warning("retrieval.vector_init_failed", extra={"error": str(exc)})

    def _selected(self, modalities: Sequence[str] | None) -> tuple[str, ...]:
        wanted = list(modalities if modalities is not None else self.config.modalities)
        present = {record.modality for record in self.records}
        return tuple(name for name in MODALITIES if name in wanted and name in present)

    def _notes(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.config.vector_weight > 0.0:
            if self.doc.embeddings is None:
                notes.append("no embedding index in this document — lexical retrieval only")
            elif not self._vector_ready:
                notes.append("embedding index present but vector store not initialised — lexical retrieval only")
        return tuple(notes)

    def search(
        self,
        query: str,
        *,
        modalities: Sequence[str] | None = None,
        start: float | None = None,
        end: float | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> SearchResult:
        """Rank the document's facts against ``query``.

        ``start``/``end`` restrict the search to a time window; a fact is kept if it overlaps
        it at all, so a caption spanning the boundary is not lost.
        """
        began = time.perf_counter()
        config = self.config
        selected = self._selected(modalities)
        limit = config.top_k if top_k is None else top_k
        floor = config.min_score if min_score is None else min_score

        # Lexical search
        lexical_hits = [
            hit
            for hit in self._lexical.search(query)
            if hit.record.modality in selected and _within(hit.record, start, end)
        ]

        # Vector search
        vector_hits: list[LexicalHit] = []
        if config.vector_weight > 0.0 and self._vector_ready and self._vector_store:
            vector_hits = self._vector_search(query, selected, start, end)

        # Fuse
        candidates = self._fuse(lexical_hits, vector_hits, config)
        candidates = boost_cooccurrence(candidates, boost=config.cooccurrence_boost)
        candidates = merge_adjacent(candidates, gap=config.merge_adjacent_s)
        candidates = [c for c in candidates if c.score >= floor]
        candidates.sort(key=lambda c: (-c.score, -c.coverage, c.start))

        total = len(candidates)
        kept = candidates if limit <= 0 else candidates[:limit]
        result = SearchResult(
            query=query,
            spans=tuple(_to_span(candidate) for candidate in kept),
            modalities=selected,
            total=total,
            took_ms=(time.perf_counter() - began) * 1000.0,
            notes=self._notes(),
        )
        log.debug(
            "search.complete",
            extra={
                "query_terms": len(query.split()),
                "modalities": list(selected),
                "candidates": total,
                "returned": len(result.spans),
            },
        )
        return result

    def _vector_search(
        self, query: str, selected: tuple[str, ...], start: float | None, end: float | None
    ) -> list[LexicalHit]:
        """Search the vector store and convert results to LexicalHit-like objects."""
        from ..embeddings.local import LocalEmbeddings

        # Embed the query
        try:
            embeddings = LocalEmbeddings()
            query_vector = embeddings.embed([query])[0]
        except Exception as exc:
            log.warning("retrieval.vector_query_failed", extra={"error": str(exc)})
            return []

        # Search vector store
        results = self._vector_store.query(query_vector, top_k=100)

        # Convert to LexicalHit-like objects
        from .lexical import LexicalHit

        hits: list[LexicalHit] = []
        for emb_id, score in results:
            record = self._by_key.get(emb_id)
            if record is None:
                continue
            if record.modality not in selected:
                continue
            if not _within(record, start, end):
                continue
            hits.append(
                LexicalHit(
                    record=record,
                    matched=(),
                    coverage=0.0,
                    phrase=False,
                )
            )
        return hits

    def _fuse(
        self,
        lexical_hits: Sequence[LexicalHit],
        vector_hits: Sequence[LexicalHit],
        config: RetrievalConfig,
    ) -> list[Candidate]:
        """Fuse lexical and vector results via RRF."""
        from .fusion import Candidate

        # Lexical fusion (per modality)
        by_modality: dict[str, list[LexicalHit]] = {}
        for hit in lexical_hits:
            by_modality.setdefault(hit.record.modality, []).append(hit)

        lists = [
            (config.lexical_weight, [hit.record.key for hit in group])
            for group in by_modality.values()
        ]

        # Vector fusion (as a separate "modality")
        if vector_hits and config.vector_weight > 0.0:
            lists.append((config.vector_weight, [hit.record.key for hit in vector_hits]))

        scores = rrf(lists)
        detail = {hit.record.key: hit for hit in (*lexical_hits, *vector_hits)}
        return [
            Candidate(
                records=(self._by_key[key],),
                score=score,
                matched=detail[key].matched,
                reasons=_reasons(detail[key]) if key in detail else ("vector match",),
                coverage=detail[key].coverage if key in detail else 0.0,
            )
            for key, score in scores.items()
        ]

    def at(
        self,
        ts: float,
        *,
        window: float = 0.0,
        modalities: Sequence[str] | None = None,
    ) -> SearchResult:
        """Everything the document knows about the instant ``ts``.

        Not ranked — there is nothing to rank, every returned span demonstrably covers the
        instant. Results are ordered by modality then time so the output reads like a snapshot:
        what was said, what was on screen, what was seen, what happened.
        """
        began = time.perf_counter()
        selected = self._selected(modalities)
        order = {name: position for position, name in enumerate(MODALITIES)}
        covering = [
            record
            for record in self.records
            if record.modality in selected
            and record.start - window <= ts <= record.end + window
        ]
        covering.sort(key=lambda record: (order[record.modality], record.start, record.key))
        label = format_timecode(ts)
        spans = tuple(
            _to_span(
                Candidate(
                    records=(record,),
                    score=1.0,
                    matched=(),
                    reasons=(f"covers {label}",),
                )
            )
            for record in covering
        )
        return SearchResult(
            query=label,
            spans=spans,
            modalities=selected,
            total=len(spans),
            took_ms=(time.perf_counter() - began) * 1000.0,
        )

    def timeline(
        self,
        start: float = 0.0,
        end: float | None = None,
        *,
        modalities: Sequence[str] | None = None,
        top_k: int = 100,
    ) -> SearchResult:
        """Everything the document knows about ``[start, end]``, in timeline order.

        Like :meth:`at` but for a range: unranked, one span per overlapping fact,
        ordered by modality then time. ``top_k`` bounds the output (0 for all) —
        a 2-hour window can hold thousands of facts, and an agent must not drown.
        """
        began = time.perf_counter()
        selected = self._selected(modalities)
        stop = self.doc.video.duration if end is None else end
        order = {name: position for position, name in enumerate(MODALITIES)}
        covering = [
            record
            for record in self.records
            if record.modality in selected and _within(record, start, stop)
        ]
        covering.sort(key=lambda record: (order[record.modality], record.start, record.key))
        total = len(covering)
        kept = covering if top_k <= 0 else covering[:top_k]
        label = format_span(start, stop)
        spans = tuple(_record_to_span(record, 1.0, f"within {label}") for record in kept)
        notes = ()
        if total > len(kept):
            notes = (f"showing {len(kept)} of {total} facts in range — narrow the window",)
        return SearchResult(
            query=label, spans=spans, modalities=selected, total=total,
            took_ms=(time.perf_counter() - began) * 1000.0, notes=notes,
        )

    def query_temporal(
        self,
        question: str,
        *,
        modalities: Sequence[str] | None = None,
        top_k: int | None = None,
        expand_s: float = 5.0,
    ) -> tuple[TemporalQuery, SearchResult]:
        """Answer a temporal question with an explicit, inspectable plan.

        Returns ``(plan, result)``: the plan says what was understood (anchor,
        relation, range), the result carries per-span reasons naming the temporal
        match. Non-temporal questions behave exactly like :meth:`search` — the
        parser only fires on explicit temporal phrasing.
        """
        began = time.perf_counter()
        config = self.config
        limit = config.top_k if top_k is None else top_k
        plan = parse_temporal_query(question)
        if not plan.is_temporal:
            return plan, self.search(question, modalities=modalities, top_k=top_k)

        if plan.first_only and not plan.start and not plan.end and not plan.relation:
            result = self.search(plan.anchor, modalities=modalities,
                                 top_k=max(limit * 5, 25) if limit > 0 else 0)
            ordered = sorted(result.spans, key=lambda s: (s.start, s.end))
            kept = ordered if limit <= 0 else ordered[:limit]
            first = kept[0] if kept else None
            spans = tuple(
                EvidenceSpan(
                    start=s.start, end=s.end, modality=s.modality, text=s.text,
                    score=s.score, ref_ids=s.ref_ids, segment_ids=s.segment_ids,
                    matched_terms=s.matched_terms,
                    reason=f"{s.reason}; first occurrence of {plan.anchor!r}"
                    if s.reason else f"first occurrence of {plan.anchor!r}",
                    confidence=s.confidence, language=s.language, kind=s.kind,
                    video_id=s.video_id,
                )
                for s in kept
            )
            note = (f"first occurrence of {plan.anchor!r}"
                    + (f" at {first.timecode}" if first else "; nothing matched"),)
            return plan, SearchResult(
                query=question, spans=spans, modalities=result.modalities,
                total=len(ordered), took_ms=(time.perf_counter() - began) * 1000.0,
                notes=result.notes + note)

        if plan.relation in (TemporalRelation.BEFORE, TemporalRelation.AFTER):
            return plan, self._before_after(plan, modalities, limit, began)

        # Ranged query (explicit timecodes, or co-occurrence expansion around a match).
        anchor_text = plan.anchor or question
        base = self.search(anchor_text, modalities=modalities,
                           top_k=max(limit * 5, 50) if limit > 0 else 0,
                           start=plan.start, end=plan.end)
        if plan.start is not None or plan.end is not None:
            window_label = format_span(plan.start or 0.0,
                                       plan.end if plan.end is not None else 0.0)
            if not base.spans:
                # Honest fallback: no lexical match in range — show the range itself
                # rather than an empty answer.
                windowed = self.timeline(plan.start or 0.0, plan.end,
                                         modalities=modalities, top_k=limit)
                return plan, SearchResult(
                    query=question, spans=windowed.spans, modalities=windowed.modalities,
                    total=windowed.total,
                    took_ms=(time.perf_counter() - began) * 1000.0,
                    notes=(f"no text matched in range; showing timeline {window_label}",))
        if plan.start is not None or plan.end is not None:
            window_label = format_span(plan.start or 0.0,
                                       plan.end if plan.end is not None else 0.0)
            spans = tuple(
                EvidenceSpan(
                    start=s.start, end=s.end, modality=s.modality, text=s.text,
                    score=s.score, ref_ids=s.ref_ids, segment_ids=s.segment_ids,
                    matched_terms=s.matched_terms,
                    reason=f"{s.reason}; within {window_label}" if s.reason
                    else f"within {window_label}",
                    confidence=s.confidence, language=s.language, kind=s.kind,
                    video_id=s.video_id,
                )
                for s in (base.spans if limit <= 0 else base.spans[:limit])
            )
            return plan, SearchResult(
                query=question, spans=spans, modalities=base.modalities,
                total=base.total, took_ms=(time.perf_counter() - began) * 1000.0,
                notes=(*base.notes, f"constrained to {window_label}"))
        # Co-occurrence ("while"): expand each match into its temporal neighborhood.
        expanded = self._expand_around(base.spans, expand_s, modalities, limit)
        return plan, SearchResult(
            query=question, spans=expanded, modalities=base.modalities,
            total=len(expanded), took_ms=(time.perf_counter() - began) * 1000.0,
            notes=(*base.notes, f"expanded ±{expand_s:g}s around matches"))

    def _before_after(
        self,
        plan: TemporalQuery,
        modalities: Sequence[str] | None,
        limit: int,
        began: float,
    ) -> SearchResult:
        """Evidence strictly before/after the anchor's first occurrence."""
        selected = self._selected(modalities)
        anchors = self.search(plan.anchor, modalities=modalities, top_k=5)
        if not anchors:
            return SearchResult(
                query=plan.raw, spans=(), modalities=selected, total=0,
                took_ms=(time.perf_counter() - began) * 1000.0,
                notes=(f"anchor {plan.anchor!r} matched nothing — "
                       "cannot reason about before/after",))
        pivot = anchors.spans[0]  # top-scored anchor: the most relevant occurrence,
        # not merely the earliest — "before the error" means before the error match
        # the retriever is most confident about.
        before = plan.relation is TemporalRelation.BEFORE
        pool = [
            record for record in self.records
            if record.modality in selected
            and ((record.end <= pivot.start + 1e-6) if before
                 else (record.start >= pivot.end - 1e-6))
        ]
        pool.sort(key=lambda r: (-r.end if before else r.start, r.key))
        kept = pool if limit <= 0 else pool[:limit]
        pivot_tc = format_timecode(pivot.start if before else pivot.end)
        spans = tuple(
            _record_to_span(
                record,
                1.0 / (1.0 + abs((pivot.start - record.end) if before
                                 else (record.start - pivot.end))),
                f"{'ends' if before else 'starts'} "
                f"{abs((pivot.start - record.end) if before else (record.start - pivot.end)):.1f}s "
                f"{'before' if before else 'after'} {plan.anchor!r} at {pivot_tc}",
            )
            for record in kept
        )
        return SearchResult(
            query=plan.raw, spans=spans, modalities=selected, total=len(pool),
            took_ms=(time.perf_counter() - began) * 1000.0,
            notes=(f"{'before' if before else 'after'} top {plan.anchor!r} match "
                   f"at {pivot_tc} ({len(anchors.spans)} anchor matches considered)",))

    def _expand_around(
        self,
        spans: Sequence[EvidenceSpan],
        radius: float,
        modalities: Sequence[str] | None,
        limit: int,
    ) -> tuple[EvidenceSpan, ...]:
        """Each match plus the facts co-occurring in its neighborhood, deduped."""
        from ..temporal import merge_windows, window_around

        selected = self._selected(modalities)
        duration = self.doc.video.duration
        windows = merge_windows([window_around(s.start, s.end, radius, duration) for s in spans])
        seen: set[str] = set()
        out: list[EvidenceSpan] = []
        for span in spans:  # the matches themselves first, in rank order
            key = (span.modality, span.ref_ids)
            seen.add(str(key))
            out.append(span)
        for window in windows:
            for record in self.records:
                if record.modality not in selected:
                    continue
                if not _within(record, window.start, window.end):
                    continue
                key = (record.modality, (record.id,))
                if str(key) in seen:
                    continue
                seen.add(str(key))
                out.append(_record_to_span(
                    record, 0.5,
                    f"co-occurs within ±{radius:g}s of evidence at "
                    f"{format_timecode(window.start)}"))
        return tuple(out if limit <= 0 else out[: max(limit, len(spans))])


def _within(record: Record, start: float | None, end: float | None) -> bool:
    if start is not None and record.end < start:
        return False
    return not (end is not None and record.start > end)


def _reasons(hit: LexicalHit) -> tuple[str, ...]:
    reasons = ["exact phrase"] if hit.phrase else []
    reasons.append(f"matched {hit.coverage:.0%} of query weight")
    return tuple(reasons)


def _record_to_span(record: Record, score: float, reason: str) -> EvidenceSpan:
    """One fact as a span — used by timeline and temporal queries, never ranked."""
    return EvidenceSpan(
        start=record.start,
        end=record.end,
        modality=record.modality,
        text=record.text,
        score=score,
        ref_ids=(record.id,),
        segment_ids=tuple(record.segment_ids),
        matched_terms=(),
        reason=reason,
        confidence=record.confidence,
        language=record.language,
        kind=record.kind,
    )


def _to_span(candidate: Candidate) -> EvidenceSpan:
    """Copy a candidate's evidence into a span. Every value here comes from a record."""
    first = candidate.records[0]
    confidences = [r.confidence for r in candidate.records if r.confidence is not None]
    languages = [r.language for r in candidate.records if r.language]
    return EvidenceSpan(
        start=candidate.start,
        end=candidate.end,
        modality=candidate.modality,
        text=candidate.text,
        score=candidate.score,
        ref_ids=tuple(record.id for record in candidate.records),
        segment_ids=tuple(dict.fromkeys(s for r in candidate.records for s in r.segment_ids)),
        matched_terms=candidate.matched,
        reason="; ".join(candidate.reasons),
        # The minimum, not the mean: a merged span is only as trustworthy as its worst source.
        confidence=min(confidences) if confidences else None,
        language=languages[0] if languages else None,
        kind=first.kind,
    )


def search(
    doc: VideoContextDocument,
    query: str,
    *,
    config: RetrievalConfig | None = None,
    **kw: Any,
) -> SearchResult:
    """One-shot search. Build a :class:`Retriever` instead for repeated queries."""
    return Retriever(doc, config).search(query, **kw)


def at(
    doc: VideoContextDocument,
    ts: float,
    *,
    config: RetrievalConfig | None = None,
    **kw: Any,
) -> SearchResult:
    """One-shot timeline lookup. See :meth:`Retriever.at`."""
    return Retriever(doc, config).at(ts, **kw)


def timeline(
    doc: VideoContextDocument,
    start: float = 0.0,
    end: float | None = None,
    *,
    config: RetrievalConfig | None = None,
    **kw: Any,
) -> SearchResult:
    """One-shot range lookup. See :meth:`Retriever.timeline`."""
    return Retriever(doc, config).timeline(start, end, **kw)


def query_temporal(
    doc: VideoContextDocument,
    question: str,
    *,
    config: RetrievalConfig | None = None,
    **kw: Any,
) -> tuple[Any, SearchResult]:
    """One-shot temporal query. See :meth:`Retriever.query_temporal`."""
    return Retriever(doc, config).query_temporal(question, **kw)


__all__ = ["EvidenceSpan", "Retriever", "SearchResult", "at", "query_temporal", "search",
           "timeline"]
