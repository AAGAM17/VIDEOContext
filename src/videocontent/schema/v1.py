"""Video Context Format (``.vctx``) — schema v1.

Pydantic v2 models implementing ``docs/VIDEO_CONTEXT_SPEC.md``. This module is the
foundation of the project and deliberately depends on nothing else in ``videocontent``:
a ``.vctx`` document must be readable without the machinery that produced it.

Conventions enforced here (spec §2):

* times are ``float`` seconds on the media clock, intervals are half-open ``[start, end)``
* ``confidence`` is ``None`` when unknown — never a fabricated number
* unknown keys are preserved (``extra="allow"``) so a v1.0 reader survives a v1.3 document
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VCTX_VERSION = "1.0"
VCTX_MEDIA_TYPE = "application/vnd.videocontext+json"

Confidence = float | None
BBox = list[float]  # [x1, y1, x2, y2]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VctxModel(BaseModel):
    """Base model: forward-compatible (keeps unknown keys) and immutable-ish by default."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        ser_json_timedelta="float",
        validate_assignment=True,
    )


class TimeSpan(VctxModel):
    """A half-open interval on the media clock. Instants have ``end == start``."""

    start: float = Field(ge=0.0, description="Seconds from media start, inclusive.")
    end: float = Field(ge=0.0, description="Seconds from media start, exclusive.")

    @model_validator(mode="after")
    def _ordered(self):
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: TimeSpan, tolerance: float = 0.0) -> bool:
        return self.start < other.end + tolerance and other.start < self.end + tolerance

    def contains(self, ts: float) -> bool:
        return self.start <= ts < self.end or (self.end == self.start and ts == self.start)


# ---------------------------------------------------------------------------
# §4 video
# ---------------------------------------------------------------------------


class SubtitleTrack(VctxModel):
    index: int
    language: str | None = None
    codec: str | None = None
    title: str | None = None


class Chapter(TimeSpan):
    title: str | None = None


class VideoInfo(VctxModel):
    """Source identity and container facts, as probed — never guessed from the filename."""

    id: str
    filename: str
    path: str | None = None
    content_hash: str | None = Field(
        default=None, description="'sha256:…' over media bytes; the identity used for caching."
    )
    size_bytes: int | None = None

    duration: float = Field(ge=0.0)
    container: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    bitrate: int | None = None

    video_codec: str | None = None
    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None
    has_audio: bool = False
    has_video: bool = True

    subtitle_tracks: list[SubtitleTrack] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# §5 stages
# ---------------------------------------------------------------------------


class StageStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


class StageRecord(VctxModel):
    """Provenance for one pipeline stage.

    Makes absence interpretable: an empty ``ocr`` array with ``status=ok`` means the video
    has no on-screen text; with ``status=skipped`` it means nobody looked.
    """

    name: str
    status: StageStatus
    provider: str | None = None
    provider_version: str | None = None
    stage_version: str = "1"
    config_hash: str | None = None
    started_at: datetime | None = None
    duration_s: float | None = None
    cached: bool = False
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    remote: bool = Field(
        default=False, description="True if this stage sent data off-machine (privacy audit)."
    )
    counts: dict[str, int] = Field(default_factory=dict)

    @property
    def ran(self) -> bool:
        return self.status in (StageStatus.OK, StageStatus.PARTIAL)


# ---------------------------------------------------------------------------
# §6-§13 modalities
# ---------------------------------------------------------------------------


class Scene(TimeSpan):
    id: str
    confidence: Confidence = None
    detector: str | None = None
    keyframe_ts: float | None = None
    change_score: float | None = None
    signals: list[str] = Field(default_factory=list)


class Word(TimeSpan):
    text: str
    confidence: Confidence = None


class Utterance(TimeSpan):
    id: str
    text: str
    language: str | None = None
    confidence: Confidence = None
    speaker: str | None = Field(default=None, description="None = diarization not attempted.")
    no_speech_prob: float | None = None
    words: list[Word] = Field(default_factory=list)


class OCRText(TimeSpan):
    """A *temporal* OCR event: one piece of text with a lifespan, not a per-frame row."""

    id: str
    text: str
    confidence: Confidence = None
    language: str | None = None
    bbox: BBox | None = Field(default=None, description="[x1,y1,x2,y2] in source pixels.")
    bbox_normalized: BBox | None = None
    frame_count: int = 1
    first_frame_ts: float | None = None
    last_frame_ts: float | None = None
    stable: bool = False
    engine: str | None = None
    block_index: int | None = None

    @field_validator("bbox", "bbox_normalized")
    @classmethod
    def _four_numbers(cls, v: BBox | None) -> BBox | None:
        if v is not None and len(v) != 4:
            raise ValueError("bbox must have exactly 4 values [x1, y1, x2, y2]")
        return v


class VisionNote(TimeSpan):
    id: str
    description: str
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    ui: dict[str, Any] = Field(default_factory=dict)
    confidence: Confidence = None
    provider: str | None = None
    model: str | None = None
    frame_ids: list[str] = Field(default_factory=list)
    language: str | None = None


class ObjectInstance(VctxModel):
    ts: float
    bbox: BBox | None = None
    confidence: Confidence = None


class DetectedObject(TimeSpan):
    id: str
    label: str
    confidence: Confidence = None
    track_id: str | None = None
    detector: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    instances: list[ObjectInstance] = Field(default_factory=list)


class EventType(str, Enum):
    """Core taxonomy (spec §11). The taxonomy is *open* — arbitrary strings are valid."""

    SCENE_CHANGED = "scene_changed"
    SCREEN_CHANGED = "screen_changed"
    SLIDE_CHANGED = "slide_changed"
    TEXT_APPEARED = "text_appeared"
    TEXT_DISAPPEARED = "text_disappeared"
    TEXT_CHANGED = "text_changed"
    SPEAKER_STARTED = "speaker_started"
    SPEAKER_STOPPED = "speaker_stopped"
    SILENCE_STARTED = "silence_started"
    SILENCE_ENDED = "silence_ended"
    PERSON_ENTERED = "person_entered"
    PERSON_LEFT = "person_left"
    OBJECT_APPEARED = "object_appeared"
    OBJECT_DISAPPEARED = "object_disappeared"
    BUTTON_CLICKED = "button_clicked"
    COMMAND_ENTERED = "command_entered"
    ERROR_SHOWN = "error_shown"


class Event(TimeSpan):
    id: str
    type: str = Field(description="Core taxonomy value or a namespaced custom type.")
    description: str | None = None
    confidence: Confidence = None
    source: list[str] = Field(default_factory=list, description="Contributing modalities.")
    detector: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    refs: dict[str, list[str]] = Field(
        default_factory=dict, description="IDs of the facts that produced this event."
    )


class Frame(VctxModel):
    id: str
    ts: float = Field(ge=0.0)
    index: int | None = None
    path: str | None = None
    width: int | None = None
    height: int | None = None
    reason: str = "fixed"
    sharpness: float | None = None
    diff_score: float | None = None
    phash: str | None = None
    data_uri: str | None = None


class Segment(TimeSpan):
    """The fused retrieval unit: a window plus every modality overlapping it, by reference."""

    id: str
    scene_ids: list[str] = Field(default_factory=list)
    transcript_ids: list[str] = Field(default_factory=list)
    ocr_ids: list[str] = Field(default_factory=list)
    vision_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)

    text: str = Field(default="", description="Denormalized searchable projection.")
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    embeddings: dict[str, str] = Field(default_factory=dict)


class EmbeddingIndex(VctxModel):
    model: str | None = None
    dim: int | None = None
    normalized: bool = True
    space: str = "text"
    quantization: str = "none"
    vectors: dict[str, list[float]] = Field(default_factory=dict)
    external: dict[str, Any] | None = None


class Metrics(VctxModel):
    processing_time_s: float | None = None
    video_duration_s: float | None = None
    realtime_factor: float | None = None
    frames_sampled: int | None = None
    frames_skipped: int | None = None
    stage_times: dict[str, float] = Field(default_factory=dict)
    tokens: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float | None = None
    cache_hits: int = 0
    peak_memory_mb: float | None = None


# ---------------------------------------------------------------------------
# Multi-Resolution Context (v1.1+)
# ---------------------------------------------------------------------------


class GlobalContext(VctxModel):
    """Global semantic understanding of the entire video."""

    summary: str | None = None
    one_line: str | None = None
    domain: str | None = None
    major_topics: list[str] = Field(default_factory=list)
    visual_language: list[str] = Field(default_factory=list)
    interaction_language: list[str] = Field(default_factory=list)


class VisualStyleProfile(VctxModel):
    """Visual style characteristics."""

    overall: list[str] = Field(default_factory=list)
    color_characteristics: list[str] = Field(default_factory=list)
    surface_style: list[str] = Field(default_factory=list)
    confidence: Confidence = None
    evidence: list[TimeSpan] = Field(default_factory=list)


class TypographyProfile(VctxModel):
    """Typography characteristics."""

    hierarchy: str | None = None
    heading_style: str | None = None
    density: str | None = None
    confidence: Confidence = None
    evidence: list[TimeSpan] = Field(default_factory=list)


class LayoutProfile(VctxModel):
    """Layout patterns."""

    patterns: list[str] = Field(default_factory=list)
    confidence: Confidence = None
    evidence: list[TimeSpan] = Field(default_factory=list)


class ComponentProfile(VctxModel):
    """UI component pattern."""

    component_id: str
    type: str
    first_seen: float
    last_seen: float
    visual_characteristics: list[str] = Field(default_factory=list)
    content_structure: list[str] = Field(default_factory=list)
    confidence: Confidence = None
    evidence: list[TimeSpan] = Field(default_factory=list)


class InteractionProfile(VctxModel):
    """Interaction pattern."""

    type: str | None = None
    pattern: str | None = None
    confidence: Confidence = None
    evidence: list[TimeSpan] = Field(default_factory=list)


class MotionProfile(VctxModel):
    """Motion/animation pattern."""

    motion_id: str
    element: str | None = None
    type: str
    direction: str | None = None
    style: str | None = None
    duration_category: str | None = None
    confidence: Confidence = None
    evidence: list[TimeSpan] = Field(default_factory=list)


class UIDesignProfile(VctxModel):
    """Complete UI/Design profile for interface videos."""

    visual_style: VisualStyleProfile = Field(default_factory=VisualStyleProfile)
    typography: TypographyProfile = Field(default_factory=TypographyProfile)
    layout: LayoutProfile = Field(default_factory=LayoutProfile)
    components: list[ComponentProfile] = Field(default_factory=list)
    interaction: InteractionProfile = Field(default_factory=InteractionProfile)
    motion: list[MotionProfile] = Field(default_factory=list)


class ApplicationProfile(VctxModel):
    """Application understanding profile."""

    overview: str | None = None
    screen_hierarchy: list[dict[str, Any]] = Field(default_factory=list)
    user_flows: list[dict[str, Any]] = Field(default_factory=list)
    state_transitions: list[dict[str, Any]] = Field(default_factory=list)
    important_interactions: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[TimeSpan] = Field(default_factory=list)


class ProductDemoProfile(VctxModel):
    """Product demonstration profile."""

    product_overview: str | None = None
    features_shown: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    ui_walkthrough: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[TimeSpan] = Field(default_factory=list)


class TutorialProfile(VctxModel):
    """Tutorial/educational profile."""

    topic: str | None = None
    learning_objectives: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    evidence: list[TimeSpan] = Field(default_factory=list)


class SemanticProfiles(VctxModel):
    """Container for all semantic profiles."""

    ui_design: UIDesignProfile | None = None
    application: ApplicationProfile | None = None
    product_demo: ProductDemoProfile | None = None
    tutorial: TutorialProfile | None = None


class InteractionNode(VctxModel):
    """A state in the interaction graph."""

    state_id: str
    name: str
    start: float
    end: float
    description: str | None = None


class InteractionEdge(VctxModel):
    """A transition between states."""

    from_state: str
    to_state: str
    action: str
    start: float
    end: float
    confidence: Confidence = None


class InteractionGraph(VctxModel):
    """State transition graph for the video."""

    nodes: list[InteractionNode] = Field(default_factory=list)
    edges: list[InteractionEdge] = Field(default_factory=list)


class VisualState(VctxModel):
    """A persistent visual state detected over a time interval."""

    state_id: str
    name: str
    start: float
    end: float
    persistent_elements: list[str] = Field(default_factory=list)
    changes: list[dict[str, Any]] = Field(default_factory=list)
    confidence: Confidence = None


class ContextSummaries(VctxModel):
    """Multi-level summaries for context budgeting."""

    one_line: str | None = None
    short: str | None = None
    detailed: str | None = None
    structured: dict[str, Any] = Field(default_factory=dict)


class Producer(VctxModel):
    name: str = "videocontent"
    version: str = "0.1.0"
    config_hash: str | None = None


# ---------------------------------------------------------------------------
# §3 document
# ---------------------------------------------------------------------------


class VideoContextDocument(VctxModel):
    """The complete semantic representation of one video."""

    vctx_version: str = VCTX_VERSION
    id: str
    created_at: datetime = Field(default_factory=_utcnow)
    producer: Producer = Field(default_factory=Producer)

    video: VideoInfo
    stages: list[StageRecord] = Field(default_factory=list)

    scenes: list[Scene] = Field(default_factory=list)
    transcript: list[Utterance] = Field(default_factory=list)
    ocr: list[OCRText] = Field(default_factory=list)
    vision: list[VisionNote] = Field(default_factory=list)
    objects: list[DetectedObject] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    frames: list[Frame] = Field(default_factory=list)

    embeddings: EmbeddingIndex | None = None
    metrics: Metrics = Field(default_factory=Metrics)

    # Multi-resolution context (v1.1+)
    global_context: GlobalContext | None = None
    semantic_profiles: SemanticProfiles | None = None
    interaction_graph: InteractionGraph | None = None
    visual_states: list[VisualState] = Field(default_factory=list)
    context_summaries: ContextSummaries | None = None

    # -- convenience -------------------------------------------------------

    def stage(self, name: str) -> StageRecord | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def stage_status(self, name: str) -> StageStatus | None:
        rec = self.stage(name)
        return rec.status if rec else None

    def ran(self, name: str) -> bool:
        """True when the stage produced results — the guard against conflating absence."""
        rec = self.stage(name)
        return bool(rec and rec.ran)

    def by_id(self, oid: str) -> Any | None:
        return self._id_map().get(oid)

    def _id_map(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for coll in (
            self.scenes,
            self.transcript,
            self.ocr,
            self.vision,
            self.objects,
            self.events,
            self.segments,
            self.frames,
        ):
            for item in coll:
                out[item.id] = item
        return out

    def transcript_text(self) -> str:
        return " ".join(u.text.strip() for u in self.transcript if u.text.strip())

    def at(self, ts: float) -> dict[str, list[Any]]:
        """Everything overlapping an instant — the primitive behind "what was on screen at T"."""
        probe = TimeSpan(start=ts, end=ts)
        return {
            "scenes": [s for s in self.scenes if s.overlaps(probe)],
            "transcript": [u for u in self.transcript if u.overlaps(probe)],
            "ocr": [o for o in self.ocr if o.overlaps(probe)],
            "vision": [v for v in self.vision if v.overlaps(probe)],
            "events": [e for e in self.events if e.overlaps(probe)],
            "objects": [o for o in self.objects if o.overlaps(probe)],
            "segments": [g for g in self.segments if g.overlaps(probe)],
        }

    def window(self, start: float, end: float) -> dict[str, list[Any]]:
        probe = TimeSpan(start=start, end=end)
        return {
            "scenes": [s for s in self.scenes if s.overlaps(probe)],
            "transcript": [u for u in self.transcript if u.overlaps(probe)],
            "ocr": [o for o in self.ocr if o.overlaps(probe)],
            "vision": [v for v in self.vision if v.overlaps(probe)],
            "events": [e for e in self.events if e.overlaps(probe)],
            "objects": [o for o in self.objects if o.overlaps(probe)],
            "segments": [g for g in self.segments if g.overlaps(probe)],
        }


TimedCollection = Literal[
    "scenes", "transcript", "ocr", "vision", "objects", "events", "segments"
]

__all__ = [
    "VCTX_MEDIA_TYPE",
    "VCTX_VERSION",
    "BBox",
    "Chapter",
    "Confidence",
    "DetectedObject",
    "EmbeddingIndex",
    "Event",
    "EventType",
    "Frame",
    "Metrics",
    "OCRText",
    "ObjectInstance",
    "Producer",
    "Scene",
    "Segment",
    "StageRecord",
    "StageStatus",
    "SubtitleTrack",
    "TimeSpan",
    "Utterance",
    "VideoContextDocument",
    "VideoInfo",
    "VisionNote",
    "Word",
    # Multi-resolution (v1.1+)
    "GlobalContext",
    "VisualStyleProfile",
    "TypographyProfile",
    "LayoutProfile",
    "ComponentProfile",
    "InteractionProfile",
    "MotionProfile",
    "UIDesignProfile",
    "ApplicationProfile",
    "ProductDemoProfile",
    "TutorialProfile",
    "SemanticProfiles",
    "InteractionNode",
    "InteractionEdge",
    "InteractionGraph",
    "VisualState",
    "ContextSummaries",
]
