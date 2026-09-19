"""The SDK facade — three names that cover everything most callers need.

``Video`` is deliberately thin. It owns no logic of its own: processing belongs to
:class:`~videocontent.processing.pipeline.Pipeline`, search to
:class:`~videocontent.retrieval.query.Retriever`, serialisation to :mod:`videocontent.schema.io`.
What it adds is the two things a facade should: it keeps the document and the retriever together
so the second query over a video does not rebuild the index, and it gives the ergonomics one
obvious spelling::

    video = videocontent.process("demo.mp4")     # runs the pipeline, keeps the document
    video.save()                                 # → demo.vctx
    for hit in video.search("pricing"):
        print(hit.timecode, hit.text)

    video = videocontent.load("demo.vctx")       # or start from a document someone else made
    print(video.at("03:21").spans)

    answer = video.ask("What was the revenue?")  # → Answer with evidence
    context = video.context_for("Recreate the design")  # → Optimized AI context
    profile = video.profile("ui_design")         # → UI Design profile

Processing is never implicit. ``Video("demo.mp4").search(...)`` raises rather than quietly
spending thirty seconds of CPU on attribute access — a property that transcodes a video is a
property that will be called inside a loop by someone who did not read this docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ProcessingConfig
from .errors import VideoContextError
from .logging import get_logger
from .schema import io
from .schema.v1 import VideoContextDocument
from .timecode import parse_timecode

if TYPE_CHECKING:  # pragma: no cover - import cost paid only when searching
    from .retrieval.query import Retriever, SearchResult
    from .routing import TaskClassification

log = get_logger("sdk")


@dataclass(frozen=True)
class AnswerTrace:
    """The full path from question to answer — debugging, research, benchmarking.

    ``selected_spans``/``omitted_spans`` name reference IDs (small); the spans
    themselves travel on the Answer. ``answer_support`` names the evidence IDs
    the answer text was built from (here: all selected evidence — the LLM cites
    by number, and fabrication beyond the evidence is a prompt violation, not data).
    """

    query: str
    plan: dict[str, Any] = field(default_factory=dict)
    retrieved_evidence: int = 0
    graph_operations: tuple[str, ...] = ()
    temporal_operations: tuple[str, ...] = ()
    selected_spans: tuple[str, ...] = ()
    omitted_spans: tuple[str, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    answer_support: tuple[str, ...] = ()
    llm: dict[str, Any] = field(default_factory=dict)
    outcome: str = "unknown"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan,
            "retrieved_evidence": self.retrieved_evidence,
            "graph_operations": list(self.graph_operations),
            "temporal_operations": list(self.temporal_operations),
            "selected_spans": list(self.selected_spans),
            "omitted_spans": list(self.omitted_spans),
            "budget": self.budget,
            "coverage": self.coverage,
            "answer_support": list(self.answer_support),
            "llm": self.llm,
            "outcome": self.outcome,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class Answer:
    """An answer to a question about a video, with traceable evidence."""

    question: str
    answer: str
    confidence: float
    evidence: list[Any] = field(default_factory=list)
    spans: list[Any] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    """How the answer was built: query plan, retrieval stats, expansion, LLM used."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence": [span.to_dict() if hasattr(span, "to_dict") else str(span) for span in self.evidence],
            "trace": self.trace,
        }


@dataclass(frozen=True)
class OptimizedContext:
    """Optimized AI context for a specific task.

    Contains the task classification, selected semantic profiles,
    evidence, representative frames, and a packed context string
    ready to send to an LLM.
    """

    task: str
    task_type: Any  # TaskClassification enum
    confidence: float
    context: str
    profiles: dict[str, Any]
    evidence: list[Any]
    frames: list[Any]
    global_context: Any | None
    summaries: dict[str, str]
    token_estimate: int = 0
    budget_notes: list[str] = field(default_factory=list)
    """What the budgeter cut or expanded, and why — the debugger's entry point."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "task_type": self.task_type.value if hasattr(self.task_type, "value") else str(self.task_type),
            "confidence": self.confidence,
            "context": self.context,
            "profiles": {k: v.model_dump() if hasattr(v, "model_dump") else str(v) for k, v in self.profiles.items()},
            "evidence": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in self.evidence],
            "frames": self.frames,
            "global_context": self.global_context.model_dump() if hasattr(self.global_context, "model_dump") else self.global_context,
            "summaries": self.summaries,
            "token_estimate": self.token_estimate,
            "budget_notes": self.budget_notes,
        }

#: What ``save()`` writes when given no path. ``demo.mp4`` → ``demo.vctx`` (§50).
VCTX_SUFFIX = ".vctx"


class NotProcessedError(VideoContextError):
    """A document was asked for before one existed."""


class Video:
    """A video and whatever is known about it.

    Constructed from a media file, a URL, a :class:`VideoSource` or a
    :class:`VideoAsset`, the ``Video`` has no document until :meth:`process` runs.
    Constructed by :func:`load`, it has a document and no media — which is the normal case for
    querying, and the reason nothing here requires the original file to still exist.

    ``Video("demo.mp4")`` keeps working exactly as before; ``Video("https://…/v.mp4")``
    and ``videocontent.open(source)`` are the universal entry points for any source.
    """

    def __init__(
        self,
        source: str | Path | Any,
        *,
        config: ProcessingConfig | None = None,
    ) -> None:
        from .sources.resolve import resolve as _resolve
        from .sources.types import VideoAsset as _Asset
        from .sources.types import VideoSource as _Source

        self.config: ProcessingConfig = config or ProcessingConfig()
        self._doc: VideoContextDocument | None = None
        self._retriever: Retriever | None = None
        self._graph: Any = None
        self._graph_for: Any = None
        self._entities: Any = None
        self._entities_for: Any = None
        self.path: Path | None = None
        """Where the document was loaded from or last saved to, if anywhere."""
        if isinstance(source, _Asset):
            self._asset: Any = source
            self.source_ref: Any = source.source
            self.locator: str = source.source.locator
        elif isinstance(source, _Source):
            self._asset = None
            self.source_ref = source
            self.locator = source.locator
        else:
            self._asset = None
            self.source_ref = _resolve(str(source), config=self.config)
            self.locator = str(source)
        # Backward-compatible display path: local files keep their path; URLs collapse
        # to their basename so ``.name`` / ``default_path()`` keep working.
        if self.source_ref.source_type.value == "local_file":
            self.source: Path = Path(self.locator)
        else:
            base = self.locator.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or "remote-video"
            self.source = Path(base)

    # -- construction ------------------------------------------------------

    @classmethod
    def open(cls, source: str | Path | Any, *, config: ProcessingConfig | None = None) -> Video:
        """Universal entry point: local path, URL, ``VideoSource`` or ``VideoAsset``."""
        return cls(source, config=config)

    def inspect(self) -> Any:
        """Describe this video's source without downloading or processing it."""
        from .sources.resolve import inspect_source as _inspect

        return _inspect(self.source_ref, config=self.config)

    @classmethod
    def from_document(
        cls,
        doc: VideoContextDocument,
        *,
        config: ProcessingConfig | None = None,
        path: str | Path | None = None,
    ) -> Video:
        """Wrap an existing document. ``source`` is taken from the document's own metadata."""
        video = cls(doc.video.filename, config=config)
        video._doc = doc
        video.path = Path(path) if path is not None else None
        # Restore source provenance when the document carries it (new documents do).
        try:
            if getattr(doc, "source", None) is not None:
                from .sources.types import SourceType as _ST
                from .sources.types import VideoSource as _VS

                rec = doc.source
                video.source_ref = _VS(
                    source_id=rec.source_id,
                    source_type=_ST(rec.source_type),
                    provider=rec.provider,
                    locator=rec.locator_redacted or doc.video.filename,
                    canonical_id=rec.canonical_id or rec.source_id,
                    metadata={},
                )
        except Exception as exc:
            log.debug("sdk.source_restore_skipped", extra={"error": str(exc)})
        return video

    # -- state -------------------------------------------------------------

    @property
    def processed(self) -> bool:
        return self._doc is not None

    @property
    def document(self) -> VideoContextDocument:
        """The ``.vctx`` document. Raises if nothing has produced one yet."""
        if self._doc is None:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process(), or load an existing .vctx with videocontent.load()",
            )
        return self._doc

    @property
    def duration(self) -> float:
        return self.document.video.duration

    def __repr__(self) -> str:
        state = "processed" if self.processed else "unprocessed"
        return f"Video({self.source.name!r}, {state})"

    # -- processing --------------------------------------------------------

    def process(self, *, force: bool = False) -> VideoContextDocument:
        """Run the pipeline over ``source`` and keep the result.

        Returns the existing document unless ``force``, so calling this twice on one object is
        cheap. Stage-level caching between *runs* is the pipeline's own business
        (ARCHITECTURE §9) and happens whether or not this object is reused.
        """
        if self._doc is not None and not force:
            return self._doc

        from .processing.pipeline import Pipeline

        self._doc = Pipeline(self.config).run(self._asset or self.source_ref)
        self._retriever = None
        self._graph = None
        self._entities = None
        return self._doc

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path | None = None, **kw: Any) -> Path:
        """Write the document. Defaults to the video's own name with a ``.vctx`` suffix."""
        target = Path(path) if path is not None else self.default_path()
        self.path = io.save(self.document, target, **kw)
        log.info("document saved", extra={"path": self.path.name, "bytes": self.path.stat().st_size})
        return self.path

    def default_path(self) -> Path:
        """``…/demo.mp4`` → ``…/demo.vctx``, alongside the video."""
        return self.source.with_suffix(VCTX_SUFFIX)

    # -- retrieval ---------------------------------------------------------

    @property
    def retriever(self) -> Retriever:
        """The document's index, built once and reused across queries."""
        if self._retriever is None:
            from .retrieval.query import Retriever

            self._retriever = Retriever(self.document, self.config.retrieval)
        return self._retriever

    def search(self, query: str, **kw: Any) -> SearchResult:
        """Ranked evidence for ``query``. See :meth:`Retriever.search` for the options."""
        return self.retriever.search(query, **kw)

    def at(self, ts: float | str, **kw: Any) -> SearchResult:
        """Everything known about one instant. Accepts seconds or ``HH:MM:SS.mmm``."""
        return self.retriever.at(_seconds(ts), **kw)

    def timeline(self, start: float | str = 0.0, end: float | str | None = None,
                 **kw: Any) -> SearchResult:
        """Everything known about ``[start, end]`` in timeline order."""
        stop = _seconds(end) if isinstance(end, str) else end
        return self.retriever.timeline(_seconds(start), stop, **kw)

    def chapters(self) -> list[Any]:
        """Extractive, deterministic chapters with honestly-marked derived titles."""
        from .temporal import build_chapters

        return build_chapters(self.document)

    def entities(self) -> list[Any]:
        """Timestamp-grounded entities with cross-modal links and uncertainty flags."""
        from .entities import extract_entities

        if self._entities is None or self._entities_for is not self._doc:
            self._entities = extract_entities(self.document)
            self._entities_for = self._doc
        return self._entities

    def entity_timeline(self, name: str) -> Any:
        """All occurrences of ``name`` in time order, plus neighborhood helpers.

        Exact normalized match first, then the highest-confidence entity whose
        name contains the query (or vice versa) — so ``ConnectionError`` finds
        ``E ConnectionError: refused …`` without merging distinct entities.
        """
        from .entities import EntityTimeline, normalize_name

        key = normalize_name(name)
        entities = self.entities()
        for entity in entities:
            if normalize_name(entity.name) == key:
                return EntityTimeline(entity)
        fallbacks = sorted(
            (entity for entity in entities
             if key and (key in normalize_name(entity.name)
                         or normalize_name(entity.name) in key)),
            key=lambda entity: (-entity.confidence, entity.first_seen or 0.0))
        return EntityTimeline(fallbacks[0]) if fallbacks else None

    def changes(self) -> list[Any]:
        """Cheap change detection between adjacent regions, evidence-backed."""
        from .temporal import detect_changes

        return detect_changes(self.document)

    def graph(self) -> Any:
        """The derived evidence graph (memoized per document)."""
        from .graph import build_graph

        if self._graph is None or self._graph_for is not self._doc:
            self._graph = build_graph(self.document)
            self._graph_for = self._doc
        return self._graph

    def query_plan(self, question: str) -> dict[str, Any]:
        """Inspectable plan for one question: intent, entities, strategy, coverage."""
        from .queryplan import build_plan

        return build_plan(question, doc=self.document).to_dict()

    def explain(self, node_or_edge_id: str) -> dict[str, Any] | None:
        """Explain a graph node (supporting evidence) or edge (construction rule)."""
        graph = self.graph()
        if node_or_edge_id in graph.edges:
            return graph.explain(node_or_edge_id)
        node = graph.get_node(node_or_edge_id)
        if node is None:
            return None
        neighbors = graph.neighbors(node_or_edge_id)
        return {
            "node": node.to_dict(),
            "supporting_evidence": [n.to_dict()
                                    for n in graph.supporting_evidence(node_or_edge_id)],
            "neighbors": [{"node": n.to_dict(), "edge": e.to_dict()}
                          for n, e in neighbors[:20]],
            "neighbor_count": len(neighbors),
        }

    def context_package(self, task: str, **kw: Any) -> Any:
        """First-class agent context package: planned, optimized, budgeted."""
        from .packages import build_package
        from .queryplan import build_plan

        if not self.processed:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process() first",
            )
        plan = build_plan(task, doc=self.document)
        selection = self.context(task, **kw)
        graph = self.graph()
        entities = [e.to_dict() for e in self.entities()
                    if any(o.ref_id in {r for s in selection.evidence for r in s.ref_ids}
                           for o in e.occurrences)][:20]
        return build_package(
            self.document, task, list(selection.evidence),
            plan=plan.to_dict(), intent=plan.intent.value,
            frames=list(selection.frames), entities=entities,
            graph_summary=graph.stats(),
            budget={"max_tokens": kw.get("max_tokens", 4000),
                    "notes": selection.budget_notes},
            warnings=[*plan.warnings],
            max_spans=kw.get("max_spans"), max_frames=kw.get("max_frames"))

    def plan(self, task: str) -> dict[str, Any]:
        """What processing would ``task`` need? Plan + coverage against this document.

        This is the *processing* plan (stages, providers). For the per-question
        *query* plan (intent, entities, retrieval strategy), see :meth:`query_plan`.
        """
        from .plans import check_coverage, plan_for_task

        if not self.processed:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process() first",
            )
        plan = plan_for_task(task)
        return {"plan": plan.to_dict(), "coverage": check_coverage(self.document, plan).to_dict()}

    def receipt(self) -> dict[str, Any]:
        """Processing receipt: how this context was generated, from the artifact itself.

        Tells downstream intelligence which modalities exist or are missing, which
        stages ran (and where — local vs remote), which timestamps are
        trustworthy, and which costs were measured. Nothing is invented: absent
        data reads as absent.
        """
        doc = self.document
        source = getattr(doc, "source", None)
        modalities = {
            "transcript": len(doc.transcript), "ocr": len(doc.ocr),
            "vision": len(doc.vision), "events": len(doc.events),
            "scenes": len(doc.scenes), "segments": len(doc.segments),
            "frames": len(doc.frames), "objects": len(getattr(doc, "objects", [])),
        }
        trust = {stage.name: {"status": stage.status.value
                              if hasattr(stage.status, "value") else stage.status,
                              "provider": stage.provider, "remote": stage.remote,
                              "error": stage.error}
                 for stage in doc.stages}
        return {
            "video_id": doc.id,
            "vctx_version": doc.vctx_version,
            "producer": doc.producer.model_dump(mode="json"),
            "source": source.model_dump(mode="json") if source is not None else None,
            "stages": [s.model_dump(mode="json") for s in doc.stages],
            "metrics": doc.metrics.model_dump(mode="json"),
            "modalities": modalities,
            "missing_modalities": sorted(mod for mod, count in modalities.items()
                                         if count == 0),
            "trust": trust,
            "timestamps_trustworthy": bool(doc.segments) or bool(doc.scenes),
            "derived_available": {
                "entities": True, "chapters": True, "changes": True,
                "states": bool(doc.ocr),
            },
        }

    # -- Q&A -----------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        modalities: list[str] | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> "Answer":
        """Answer a question using planned, graph-aware retrieval + an LLM.

        Pipeline: QueryPlan → coverage check → graph-aware retrieval → evidence
        selection → ContextPackage → LLM → Answer + AnswerTrace. If the plan needs
        modalities the document lacks, the answer says so (with a reprocessing
        suggestion) instead of failing silently — and nothing is ever processed
        implicitly to fill the gap.
        """
        from .llm import NullLLM, OpenAILLM
        from .queryplan import build_plan

        if not self.processed:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process() first",
            )

        plan = build_plan(question, doc=self.document)
        budget = {"top_k": top_k, "modalities": modalities, "min_score": min_score}
        warnings: list[str] = list(plan.warnings)
        outcome_hint = ""
        if plan.coverage.get("missing") and plan.intent.value not in (
                "fact_lookup", "general_summary"):
            # Coverage-aware planning: attempt retrieval anyway (partial evidence
            # may suffice), but name the gap and how to fill it. Never process
            # implicitly to fill it.
            outcome_hint = "insufficient_coverage"
            warnings.append("missing modalities: "
                            + ", ".join(f"{mod} ({hint})" for mod, hint in zip(
                                plan.coverage["missing"],
                                plan.coverage.get("suggestions", []))))

        search_result, explanation = self.retriever.search_graph(
            question, modalities=modalities, top_k=top_k, graph=self.graph())
        selected = list(search_result.spans)
        if min_score is not None:
            selected = [s for s in selected if s.score >= min_score]

        def _trace(outcome: str, llm: dict[str, Any]) -> dict[str, Any]:
            return AnswerTrace(
                query=question, plan=plan.to_dict(),
                retrieved_evidence=search_result.total,
                graph_operations=tuple(
                    f"{e['relation']}:{e['from']}→{e['to']}"
                    for e in explanation.graph_expansions),
                temporal_operations=tuple(explanation.temporal_operations),
                selected_spans=tuple(ref for span in selected for ref in span.ref_ids),
                omitted_spans=tuple(o["ref_ids"][0] if o["ref_ids"] else "?"
                                      for o in explanation.omitted),
                budget=budget, coverage=plan.coverage,
                answer_support=tuple(ref for span in selected for ref in span.ref_ids),
                llm=llm, outcome=outcome, warnings=tuple(warnings)).to_dict()

        if not selected:
            outcome = outcome_hint or "insufficient_evidence"
            if outcome_hint:
                answer_text = ("I couldn't answer from this video: it lacks "
                               f"{', '.join(plan.coverage['missing'])}. "
                               + " ".join(plan.coverage.get("suggestions", [])))
            elif min_score is not None:
                answer_text = ("Retrieved evidence did not meet the minimum score — "
                               "the video may not contain an answer.")
            else:
                answer_text = ("I couldn't find any relevant information in the video "
                               "to answer this question.")
            return Answer(question=question, answer=answer_text, confidence=0.0,
                          evidence=[], spans=[], trace=_trace(outcome, {}))

        # Build context from evidence
        context_parts = []
        for i, span in enumerate(selected, 1):
            context_parts.append(
                f"[{i}] {span.timecode} ({span.modality}): {span.text[:500]}"
            )
        context = "\n".join(context_parts)

        # Get LLM provider
        llm_provider = self.config.llm.provider or "null"
        if llm_provider == "openai":
            llm = OpenAILLM(self.config.llm)
        elif llm_provider == "local":
            from .llm import LocalLLM
            llm = LocalLLM(self.config.llm)
        else:
            llm = NullLLM(self.config.llm)

        # Build prompt
        system_prompt = (
            "You are a helpful assistant that answers questions about a video "
            "using ONLY the provided evidence. Each piece of evidence has a "
            "timestamp and source modality. Cite evidence by its number [1], [2], "
            "etc. If the evidence doesn't contain the answer, say so. "
            "Never invent timestamps or facts not in the evidence."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Evidence:\n{context}\n\n"
            f"Answer the question based only on the evidence above. "
            f"Cite evidence using [1], [2], etc."
        )

        try:
            answer_text = llm.complete(user_prompt, system=system_prompt)
        except Exception as exc:
            # If LLM fails, return evidence-only answer
            log.warning("ask.llm_failed", extra={"error": str(exc)})
            return Answer(
                question=question,
                answer=f"LLM unavailable ({type(exc).__name__}). Here is the relevant evidence:\n" + context,
                confidence=0.5,
                evidence=list(selected),
                spans=list(selected),
                trace=_trace("llm_failed", {"provider": llm_provider,
                                            "error": type(exc).__name__}),
            )

        # Calculate confidence based on evidence quality
        confidences = [s.confidence for s in selected if s.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return Answer(
            question=question,
            answer=answer_text,
            confidence=min(avg_confidence, 1.0),
            evidence=list(selected),
            spans=list(selected),
            trace=_trace("answered", {"provider": llm_provider,
                                      "model": self.config.llm.model}),
        )

    # -- Semantic Context ----------------------------------------------------

    def context(
        self,
        task: str,
        *,
        max_tokens: int = 4000,
        modalities: list[str] | None = None,
        max_spans: int | None = None,
        max_frames: int | None = None,
        max_seconds: float | None = None,
        expand_s: float = 0.0,
    ) -> "OptimizedContext":
        """Get optimized AI context for a specific task.

        This is the main entry point for multi-resolution context.
        It classifies the task, selects optimal representations,
        and returns packed context ready for an LLM.

        Args:
            task: Natural language description of what you want to do.
                  Examples:
                  - "What was the revenue?"
                  - "Recreate the website design"
                  - "How does this application work?"
                  - "Describe the animations"
            max_tokens: Maximum tokens for the returned context.
            modalities: Optional modality filter for evidence retrieval.
            max_spans: Cap evidence spans (most relevant kept).
            max_frames: Cap representative frames.
            max_seconds: Cap total evidence span duration.
            expand_s: Pull ±N seconds of co-occurring evidence around each match.

        Returns:
            OptimizedContext with task classification, selected profiles,
            evidence, frames, and packed context string.
        """
        from .routing import (
            classify_task,
            select_context,
            ContextBudget,
            pack_context,
            TaskClassification,
        )

        if not self.processed:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process() first",
            )

        task_classification: TaskClassification = classify_task(task, self.document)
        budget = ContextBudget(max_tokens=max_tokens, max_spans=max_spans,
                               max_frames=max_frames, max_seconds=max_seconds)

        selection = select_context(
            self.document,
            task_classification,
            budget,
            query=task if task_classification.requires_evidence else None,
            expand_s=expand_s,
            modalities=modalities,
        )

        # Pack into LLM-ready context
        packed_context = pack_context(selection, task_classification.task_type.value, task)

        return OptimizedContext(
            task=task,
            task_type=task_classification.task_type,
            confidence=task_classification.confidence,
            context=packed_context,
            profiles=selection["profiles"],
            evidence=selection["evidence"],
            frames=selection["frames"],
            global_context=selection["global_context"],
            summaries=selection["summaries"],
            token_estimate=selection["token_estimate"],
            budget_notes=selection.get("budget_notes", []),
        )

    def context_for(
        self,
        task: str,
        *,
        max_tokens: int = 4000,
        modalities: list[str] | None = None,
        **kw: Any,
    ) -> "OptimizedContext":
        """Alias for context() - more natural for task-oriented usage."""
        return self.context(task, max_tokens=max_tokens, modalities=modalities, **kw)

    def profile(
        self,
        profile_name: str,
        *,
        force: bool = False,
    ) -> Any:
        """Get a specific semantic profile by name.

        Available profiles: ui_design, application, product_demo, tutorial

        Args:
            profile_name: Name of the profile to generate.
            force: If True, rebuild even if cached.

        Returns:
            The profile object (Pydantic model) or None if not applicable.
        """
        from .profiles import get_profile_builder, ProfileContext

        if not self.processed:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process() first",
            )

        try:
            builder_cls = get_profile_builder(profile_name)
            builder = builder_cls()
            profile_ctx = ProfileContext(doc=self.document, config=self.config.model_dump())
            return builder.build(profile_ctx)
        except ValueError:
            available = ", ".join(["ui_design", "application", "product_demo", "tutorial"])
            raise ValueError(f"Unknown profile: {profile_name}. Available: {available}")

    def profiles(self) -> dict[str, Any]:
        """Get all available semantic profiles for this video.

        Returns:
            Dict mapping profile names to their generated objects.
        """
        from .profiles import list_profiles, get_profile_builder, ProfileContext

        if not self.processed:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process() first",
            )

        results = {}
        profile_ctx = ProfileContext(doc=self.document, config=self.config.model_dump())

        for name in list_profiles():
            try:
                builder_cls = get_profile_builder(name)
                builder = builder_cls()
                if builder.supports(profile_ctx):
                    results[name] = builder.build(profile_ctx)
            except Exception:
                pass

        return results


def _seconds(ts: float | str) -> float:
    """A timestamp as a number, from either a number or a timecode a person typed."""
    return parse_timecode(ts) if isinstance(ts, str) else float(ts)


def load(path: str | Path, *, config: ProcessingConfig | None = None) -> Video:
    """Open an existing ``.vctx`` (or ``.vctx.gz``) document, ready to query."""
    return Video.from_document(io.load(path), config=config, path=path)


def open(source: str | Path | Any, *, config: ProcessingConfig | None = None) -> Video:
    """Universal entry point: local path, URL, ``VideoSource`` or ``VideoAsset``.

    Returns an unprocessed :class:`Video` — call :meth:`Video.inspect` to describe the
    source without fetching it, or :meth:`Video.process` to materialize and extract.
    """
    return Video.open(source, config=config)


def inspect_source(source: str | Path | Any, *, config: ProcessingConfig | None = None) -> Any:
    """Describe a source without downloading or processing it."""
    from .sources.resolve import inspect_source as _inspect

    return _inspect(source, config=config or ProcessingConfig())


def process(
    source: str | Path | Any,
    *,
    config: ProcessingConfig | None = None,
    output: str | Path | bool | None = None,
) -> Video:
    """Process a video and return it, ready to query.

    ``output`` writes the document as well: a path writes there, ``True`` writes to the default
    ``.vctx`` beside the video, and the default of ``None`` writes nothing — a caller who only
    wants to ask a question should not have a file appear next to their video for it.

    For the raw document without the facade, use
    :func:`videocontent.processing.pipeline.process`.
    """
    video = Video(source, config=config)
    video.process()
    if output is not None and output is not False:
        video.save(None if output is True else output)
    return video


__all__ = [
    "VCTX_SUFFIX",
    "Answer",
    "AnswerTrace",
    "NotProcessedError",
    "OptimizedContext",
    "Video",
    "inspect_source",
    "load",
    "open",
    "process",
]
