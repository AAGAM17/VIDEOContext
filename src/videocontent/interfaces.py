"""Plugin contracts.

Every capability in VideoContext sits behind a ``typing.Protocol`` declared here.
Protocols are structural: a plugin author never inherits from our classes, and existing
objects can be adapted without a wrapper hierarchy.

Two families of types live in this module:

* **transport types** (:class:`FrameImage`, :class:`OCRObservation`, …) — the plain data
  extractors exchange, *before* it becomes part of a ``.vctx`` document
* **protocols** — the interfaces the pipeline resolves through the registry

Extractors are pure: media + config in, typed results out. They never write documents,
never cache, and never log stage status. That belongs to the pipeline (ARCHITECTURE §2-3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid an import cycle; only needed for annotations
    from .schema.v1 import Utterance

# ---------------------------------------------------------------------------
# transport types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FrameImage:
    """One sampled frame on disk, with the sampler's justification for choosing it."""

    ts: float
    path: Path
    index: int | None = None
    width: int | None = None
    height: int | None = None
    reason: str = "fixed"
    diff_score: float | None = None
    phash: str | None = None

    @property
    def id(self) -> str:
        return f"frame_{(self.index if self.index is not None else 0):04d}"


@dataclass(slots=True)
class OCRObservation:
    """A single text block seen in a single frame — the input to temporal deduplication."""

    text: str
    ts: float
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    language: str | None = None
    block_index: int | None = None
    frame_width: int | None = None
    frame_height: int | None = None


@dataclass(slots=True)
class SceneSpan:
    """A detected shot/scene boundary pair, pre-schema."""

    start: float
    end: float
    score: float | None = None
    signals: list[str] = field(default_factory=list)
    keyframe_ts: float | None = None


@dataclass(slots=True)
class SampleWindow:
    """A time range to sample at a given rate."""

    start: float
    end: float
    fps: float
    reason: str = "fixed"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SamplePlan:
    """What the sampler decided, and why — auditable before a single frame is decoded.

    A plan is *windows*, not a list of timestamps: each window becomes one FFmpeg decode
    pass. Returning bare timestamps would force one seek per frame, which is the dominant
    cost mistake in video pipelines (ARCHITECTURE §6). ``explicit`` carries the handful of
    exact instants (scene keyframes) that deserve their own frame regardless of rate.
    """

    windows: list[SampleWindow] = field(default_factory=list)
    explicit: list[tuple[float, str]] = field(default_factory=list)
    duration: float = 0.0

    def estimated_frames(self) -> int:
        total = sum(max(1, int(w.duration * w.fps)) for w in self.windows)
        return total + len(self.explicit)

    def describe(self) -> str:
        parts = [f"{w.start:.1f}-{w.end:.1f}@{w.fps:g}fps({w.reason})" for w in self.windows]
        if self.explicit:
            parts.append(f"{len(self.explicit)} explicit")
        return ", ".join(parts)


@dataclass(slots=True)
class ASROutput:
    """Result of a transcription run."""

    utterances: list[Utterance]
    language: str | None = None
    model: str | None = None
    duration_s: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VisionOutput:
    """Result of one vision-provider call over a batch of frames."""

    description: str
    start: float
    end: float
    entities: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    ui: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    frame_ids: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(slots=True)
class FrameContext:
    """Ambient facts an extractor may use (resolution, fps, duration, language hints)."""

    duration: float
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    language: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Plugin(Protocol):
    """Common identity every plugin exposes; recorded as ``.vctx`` provenance."""

    name: str

    @property
    def version(self) -> str: ...


@runtime_checkable
class FrameSampler(Protocol):
    """Chooses *which* timestamps to look at — the single biggest cost lever."""

    name: str

    def plan(self, ctx: FrameContext, scenes: list[SceneSpan] | None = None) -> SamplePlan:
        """Return the sampling plan. Must be deterministic for a given (ctx, scenes)."""


@runtime_checkable
class SceneDetector(Protocol):
    name: str

    def detect(self, video_path: Path, ctx: FrameContext) -> list[SceneSpan]: ...


@runtime_checkable
class OCREngine(Protocol):
    name: str

    def available(self) -> bool:
        """Cheap check for the underlying binary/model, used by ``videocontent doctor``."""

    def extract(self, frames: list[FrameImage], ctx: FrameContext) -> list[OCRObservation]: ...


@runtime_checkable
class ASREngine(Protocol):
    name: str
    remote: bool

    def available(self) -> bool: ...

    def transcribe(self, audio_path: Path, ctx: FrameContext) -> ASROutput: ...


@runtime_checkable
class VisionProvider(Protocol):
    name: str
    remote: bool

    def available(self) -> bool: ...

    def describe(self, frames: list[FrameImage], ctx: FrameContext) -> list[VisionOutput]: ...


@runtime_checkable
class EventDetector(Protocol):
    name: str

    def detect(self, doc: Any, ctx: FrameContext) -> list[Any]:
        """Derive typed events from already-extracted modalities in the document."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    remote: bool
    dim: int

    def available(self) -> bool: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def add(self, ids: list[str], vectors: list[list[float]]) -> None: ...

    def query(self, vector: list[float], top_k: int) -> list[tuple[str, float]]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Used by ``ask()``; must never be required for processing or search."""

    name: str
    remote: bool

    def available(self) -> bool: ...

    def complete(self, prompt: str, *, system: str | None = None, **kw: Any) -> str: ...


@runtime_checkable
class StorageProvider(Protocol):
    name: str

    def put(self, key: str, data: bytes) -> str: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...

    def exists(self, key: str) -> bool: ...


__all__ = [
    "ASREngine",
    "ASROutput",
    "EmbeddingProvider",
    "EventDetector",
    "FrameContext",
    "FrameImage",
    "FrameSampler",
    "LLMProvider",
    "OCREngine",
    "OCRObservation",
    "Plugin",
    "SamplePlan",
    "SampleWindow",
    "SceneDetector",
    "SceneSpan",
    "StorageProvider",
    "VectorStore",
    "VisionOutput",
    "VisionProvider",
]
