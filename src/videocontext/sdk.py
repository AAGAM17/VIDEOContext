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

from dataclasses import asdict, dataclass, field
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
    from .routing import TaskClassification, ContextBudget

log = get_logger("sdk")


@dataclass(frozen=True)
class Answer:
    """An answer to a question about a video, with traceable evidence."""

    question: str
    answer: str
    confidence: float
    evidence: list[Any] = field(default_factory=list)
    spans: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence": [span.to_dict() if hasattr(span, "to_dict") else str(span) for span in self.evidence],
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
        }

#: What ``save()`` writes when given no path. ``demo.mp4`` → ``demo.vctx`` (§50).
VCTX_SUFFIX = ".vctx"


class NotProcessedError(VideoContextError):
    """A document was asked for before one existed."""


class Video:
    """A video and whatever is known about it.

    Constructed from a media file, the ``Video`` has no document until :meth:`process` runs.
    Constructed by :func:`load`, it has a document and no media — which is the normal case for
    querying, and the reason nothing here requires the original file to still exist.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        config: ProcessingConfig | None = None,
    ) -> None:
        self.source: Path = Path(source)
        self.config: ProcessingConfig = config or ProcessingConfig()
        self._doc: VideoContextDocument | None = None
        self._retriever: Retriever | None = None
        self.path: Path | None = None
        """Where the document was loaded from or last saved to, if anywhere."""

    # -- construction ------------------------------------------------------

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

        self._doc = Pipeline(self.config).run(self.source)
        self._retriever = None
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

    # -- Q&A -----------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        modalities: list[str] | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> "Answer":
        """Answer a question using retrieved evidence + an LLM.

        The pipeline: search → select top evidence → build context → LLM → answer.
        Every answer carries its evidence spans so the caller can verify timestamps.
        """
        from .llm import NullLLM, OpenAILLM
        from .retrieval.query import SearchResult

        if not self.processed:
            raise NotProcessedError(
                f"{self.source.name} has not been processed",
                hint="call video.process() first",
            )

        # Search for relevant evidence
        search_result: SearchResult = self.search(
            question,
            modalities=modalities,
            top_k=top_k,
            min_score=min_score,
        )

        if not search_result:
            return Answer(
                question=question,
                answer="I couldn't find any relevant information in the video to answer this question.",
                confidence=0.0,
                evidence=[],
                spans=[],
            )

        # Build context from evidence
        context_parts = []
        for i, span in enumerate(search_result.spans, 1):
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
                evidence=list(search_result.spans),
                spans=list(search_result.spans),
            )

        # Calculate confidence based on evidence quality
        confidences = [s.confidence for s in search_result.spans if s.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return Answer(
            question=question,
            answer=answer_text,
            confidence=min(avg_confidence, 1.0),
            evidence=list(search_result.spans),
            spans=list(search_result.spans),
        )

    # -- Semantic Context ----------------------------------------------------

    def context(
        self,
        task: str,
        *,
        max_tokens: int = 4000,
        modalities: list[str] | None = None,
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
        budget = ContextBudget(max_tokens=max_tokens)

        selection = select_context(
            self.document,
            task_classification,
            budget,
            query=task if task_classification.requires_evidence else None,
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
        )

    def context_for(
        self,
        task: str,
        *,
        max_tokens: int = 4000,
        modalities: list[str] | None = None,
    ) -> "OptimizedContext":
        """Alias for context() - more natural for task-oriented usage."""
        return self.context(task, max_tokens=max_tokens, modalities=modalities)

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


def process(
    source: str | Path,
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


__all__ = ["VCTX_SUFFIX", "NotProcessedError", "Video", "load", "process", "Answer", "OptimizedContext"]
