"""The pipeline — the only component that knows the order of things.

Extractors are pure (media + config in, typed results out). Everything stateful lives here:
resolving providers, sequencing stages, degrading when one fails, recording what happened, and
assembling the ``.vctx`` document.

**Stages are independent and declare their dependencies.** A stage lists the stages whose
output it needs; if one of those did not run, this one is *skipped with a reason* rather than
crashing on an empty list. That is what makes the degradation matrix in ARCHITECTURE §7
mechanical instead of a pile of ``if`` statements: no audio track skips ``audio``, which skips
``asr``, and the document says so.

**Every stage failure is recorded, never raised.** The single exception is the probe: a
container we cannot decode produces no document at all, because a document whose duration and
stream layout are unknown would have nothing to anchor a timestamp to. Everything after that
point degrades — a missing Tesseract binary costs you on-screen text, not the transcript.

**The document is assembled before the last two stages, not after all of them.** Events and
segments are derived *from* the document (an event's ``refs`` point at IDs that must already
exist), so they receive the real thing rather than a stand-in that would have to be kept in
sync with the schema.

What is deliberately not here yet: reading and writing the stage cache. ``config_hash`` is
recorded on every stage so a cache can be keyed later without a format change, but a run
currently recomputes everything (``cached`` is always ``False``) — V0.2 per the roadmap. The
honest field beats a cache that silently serves stale frames.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .. import registry
from ..config import ProcessingConfig
from ..errors import StageError, VideoContextError
from ..interfaces import (
    FrameContext,
    FrameImage,
    OCRObservation,
    SamplePlan,
    SceneSpan,
)
from ..logging import get_logger, timed
from ..media.audio import extract_audio
from ..media.frames import extract_plan
from ..media.probe import probe
from ..media.workspace import Workspace
from ..schema.v1 import (
    Event,
    Frame,
    Metrics,
    OCRText,
    Producer,
    Scene,
    StageRecord,
    StageStatus,
    Utterance,
    VideoContextDocument,
    VideoInfo,
    VisionNote,
)
from .asr import finalize as finalize_transcript
from .asr import resolve_engine
from .cache import StageCache
from .ocr import deduplicate
from .scenes import refine_with_text
from .segments import build_segments

log = get_logger("pipeline")

#: Bumped when a stage's *output* changes meaning for the same input and config. It is part of
#: the cache key, so an increment is what invalidates previously cached artifacts.
STAGE_VERSIONS = {
    "scenes": "1",
    "frames": "1",
    "ocr": "1",
    "scene_refine": "1",
    "audio": "1",
    "asr": "1",
    "vision": "1",
    "events": "1",
    "segments": "1",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StageResult:
    """What a stage reports back. Never an exception — the runner turns those into failures."""

    status: StageStatus
    provider: str | None = None
    provider_version: str | None = None
    remote: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage:
    """One unit of work, plus the stages it cannot run without."""

    name: str
    run: Callable[[Run], StageResult]
    requires: tuple[str, ...] = ()
    #: ``ProcessingConfig`` section names whose values affect this stage's output. Hashed into
    #: ``StageRecord.config_hash``, which is how a later cache decides two runs are the same.
    sections: tuple[str, ...] = ()


@dataclass
class Run:
    """Mutable state for one pass over one video.

    Kept out of ``Pipeline`` so a pipeline object is reusable and stages have exactly one
    place to read and write intermediate results.
    """

    source: Path
    config: ProcessingConfig
    video: VideoInfo
    workspace: Workspace
    ctx: FrameContext

    scenes: list[SceneSpan] = field(default_factory=list)
    plan: SamplePlan | None = None
    frames: list[FrameImage] = field(default_factory=list)
    observations: list[OCRObservation] = field(default_factory=list)
    ocr: list[OCRText] = field(default_factory=list)
    audio: Path | None = None
    transcript: list[Utterance] = field(default_factory=list)
    vision: list[VisionNote] = field(default_factory=list)

    doc: VideoContextDocument | None = None
    records: list[StageRecord] = field(default_factory=list)
    stage_times: dict[str, float] = field(default_factory=dict)

    def ran(self, name: str) -> bool:
        return any(r.name == name and r.ran for r in self.records)


def _accepts_config(target: Any) -> bool:
    """Whether ``target(config=…)`` is a call this callable will accept."""
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # builtins, C extensions — assume the convention holds
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    param = parameters.get("config")
    return param is not None and param.kind is not inspect.Parameter.POSITIONAL_ONLY


def build_provider(capability: str, name: str | None, config: Any = None) -> Any:
    """Instantiate a provider, passing ``config`` only if it will be accepted.

    §25 requires that a developer can add a provider without modifying core, which means core
    cannot insist on a constructor shape. ``NullSceneDetector`` is the built-in proof: it takes
    no arguments, and a pipeline that always passed ``config=`` would be unable to use its own
    fallback detector.
    """
    target = registry.factory(capability, name)
    if config is not None and _accepts_config(target):
        return target(config=config)
    return target()


def _provider_fields(provider: Any) -> dict[str, Any]:
    """Identity of a provider for the record.

    ``version`` is read with ``getattr`` because it is a class attribute on most providers but
    a *property* on two of them — ``TesseractOCR`` shells out to ``tesseract --version`` and
    ``FasterWhisperASR`` reads installed package metadata. Both can raise if the dependency
    vanished between the availability check and here, and a provenance field is not worth
    failing a stage over.
    """
    try:
        version = getattr(provider, "version", None)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("pipeline.version_unavailable", extra={"error": str(exc)})
        version = None
    return {
        "provider": getattr(provider, "name", None),
        "provider_version": str(version) if version is not None else None,
        "remote": bool(getattr(provider, "remote", False)),
    }


def _evenly(items: list[Any], limit: int) -> list[Any]:
    """At most ``limit`` items, spread across the whole list rather than truncated.

    Truncation would describe the first minute of a video and nothing else; even spacing keeps
    the frames a vision provider sees representative of the timeline it is describing.
    """
    if limit <= 0 or len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[min(len(items) - 1, int(i * step))] for i in range(limit)]


def _significant(value: float, digits: int = 4) -> float:
    """``value`` kept to ``digits`` significant figures.

    Used for ``realtime_factor``, where a fixed number of decimal places is wrong at one end of
    the range: 3 dp reads well for a normal run (``0.4612``) but floors anything faster than
    ~0.0005 to ``0.0``, which is indistinguishable from "not measured". That is precisely the
    shape of a cached re-run of a long video, which is the case the metric exists to show off.
    """
    return float(f"{value:.{digits}g}")


def _require_doc(run: Run, stage: str) -> VideoContextDocument:
    """The document a derived stage operates on.

    The runner assembles it before the first derived stage, so this cannot fire in normal
    operation — but a subclass that reorders :meth:`Pipeline.stages` could put ``events``
    first, and a ``StageError`` names the mistake where an ``AttributeError`` on ``None``
    would not. It is raised rather than asserted so that ``python -O`` cannot turn a clear
    failure into a crash three frames deeper.
    """
    if run.doc is None:
        raise StageError(
            f"stage {stage!r} needs an assembled document",
            hint="Derived stages must come after the extraction stages in Pipeline.stages().",
        )
    return run.doc


class Pipeline:
    """Runs the stage sequence over one video and returns a ``.vctx`` document."""

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        self.config = config or ProcessingConfig()

    # -- public API ---------------------------------------------------------

    def stages(self) -> list[Stage]:
        """The stage sequence, in order. ``requires`` names earlier stages."""
        return [
            Stage("scenes", self._scenes, sections=("scenes",)),
            Stage("frames", self._frames, sections=("sampling",)),
            Stage("ocr", self._ocr, requires=("frames",), sections=("ocr",)),
            Stage(
                "scene_refine",
                self._scene_refine,
                requires=("scenes", "ocr"),
                sections=("scenes", "ocr"),
            ),
            Stage("audio", self._audio, sections=("asr",)),
            Stage("asr", self._asr, requires=("audio",), sections=("asr",)),
            Stage("vision", self._vision, requires=("frames",), sections=("vision",)),
            Stage("events", self._events),
            Stage("segments", self._segments, sections=("segments",)),
        ]

    def run(self, source: str | Path) -> VideoContextDocument:
        """Process ``source`` and return the document. Artifacts land in the workspace."""
        started = time.monotonic()
        cfg = self.config

        # Outside the degradable loop on purpose: §7 says a container we cannot decode fails
        # fast with no partial document. Size, duration and container limits are enforced here
        # too, before a single frame is decoded (§31).
        video = probe(source, limits=cfg.limits, compute_hash=True)

        workspace = Workspace.for_video(source, workdir=cfg.workdir, key=cfg.full_hash())

        # Initialize stage cache if enabled
        cache = StageCache(workspace.cache_dir) if cfg.cache_enabled else None

        run = Run(
            source=Path(str(source)),
            config=cfg,
            video=video,
            workspace=workspace,
            ctx=FrameContext(
                duration=video.duration,
                fps=video.fps,
                width=video.width,
                height=video.height,
                language=cfg.asr.language,
            ),
        )
        log.info(
            "pipeline.start",
            extra={
                "video": video.filename,
                "duration_s": round(video.duration, 3),
                "workspace": str(workspace.root),
                "config_hash": cfg.full_hash(),
                "cache_enabled": cfg.cache_enabled,
            },
        )

        stages = self.stages()
        derived = {"events", "segments"}
        for stage in stages:
            if stage.name in derived and run.doc is None:
                run.doc = self._assemble(run)
            self._execute(stage, run, cache)

        if run.doc is None:  # only if the stage list was emptied by a subclass
            run.doc = self._assemble(run)

        run.doc.stages = run.records
        run.doc.metrics = self._metrics(run, time.monotonic() - started)
        log.info(
            "pipeline.done",
            extra={
                "duration_s": round(run.doc.metrics.processing_time_s or 0.0, 3),
                "realtime_factor": run.doc.metrics.realtime_factor,
                "segments": len(run.doc.segments),
                "events": len(run.doc.events),
            },
        )
        return run.doc

    # -- runner -------------------------------------------------------------

    def _execute(self, stage: Stage, run: Run, cache: Optional[StageCache]) -> None:
        missing = [name for name in stage.requires if not run.ran(name)]
        if missing:
            run.records.append(
                StageRecord(
                    name=stage.name,
                    status=StageStatus.SKIPPED,
                    stage_version=STAGE_VERSIONS.get(stage.name, "1"),
                    config_hash=run.config.stage_hash(*stage.sections) if stage.sections else None,
                    warnings=[f"requires {', '.join(missing)}, which did not run"],
                )
            )
            log.info("stage.skipped", extra={"stage": stage.name, "missing": missing})
            return

        # Try cache first
        stage_version = STAGE_VERSIONS.get(stage.name, "1")
        config_hash = run.config.stage_hash(*stage.sections) if stage.sections else None
        video_hash = run.video.content_hash or ""

        cached_result = None
        if cache and config_hash and video_hash:
            cached_entry = cache.get(video_hash, stage.name, stage_version, config_hash)
            if cached_entry:
                cached_result = cached_entry.data
                log.info("cache.hit", extra={"stage": stage.name})

        started_at = _now()
        clock = time.monotonic()

        if cached_result is not None:
            # Use cached result - apply it to run
            self._apply_cached_result(stage.name, cached_result, run)
            elapsed = time.monotonic() - clock
            run.stage_times[stage.name] = round(elapsed, 3)
            run.records.append(
                StageRecord(
                    name=stage.name,
                    status=StageStatus.OK,
                    provider="cached",
                    stage_version=stage_version,
                    config_hash=config_hash,
                    started_at=started_at,
                    duration_s=round(elapsed, 3),
                    cached=True,
                    counts=self._get_cached_counts(stage.name, cached_result),
                )
            )
            log.info("stage.cached", extra={"stage": stage.name, "duration_s": round(elapsed, 3)})
            return

        try:
            with timed(log, f"stage.{stage.name}"):
                result = stage.run(run)
        except VideoContextError as exc:
            # Expected failure modes: a missing binary, an unreadable stream, a bad provider
            # name. The stage is recorded as failed and the run continues degraded.
            result = StageResult(StageStatus.FAILED, error=exc.message)
            log.warning("stage.failed", extra={"stage": stage.name, "error": exc.message})
        except Exception as exc:
            # A provider bug must not take the document down with it: everything extracted so
            # far is still valid evidence. The type is kept in the message because "it failed"
            # without it is not actionable.
            result = StageResult(StageStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            log.exception("stage.crashed", extra={"stage": stage.name})

        elapsed = time.monotonic() - clock
        run.stage_times[stage.name] = round(elapsed, 3)

        # Store in cache if successful
        if cache and config_hash and video_hash and result.status == StageStatus.OK:
            cached_data = self._get_cached_data(stage.name, run)
            if cached_data is not None:
                cache.set(video_hash, stage.name, stage_version, config_hash, cached_data)

        run.records.append(
            StageRecord(
                name=stage.name,
                status=result.status,
                provider=result.provider,
                provider_version=result.provider_version,
                stage_version=STAGE_VERSIONS.get(stage.name, "1"),
                config_hash=run.config.stage_hash(*stage.sections) if stage.sections else None,
                started_at=started_at,
                duration_s=round(elapsed, 3),
                cached=False,
                error=result.error,
                warnings=result.warnings,
                remote=result.remote,
                counts=result.counts,
            )
        )

    # -- stages -------------------------------------------------------------

    def _scenes(self, run: Run) -> StageResult:
        cfg = run.config
        if not cfg.sampling.scene_detection:
            return StageResult(
                StageStatus.SKIPPED, warnings=["sampling.scene_detection is disabled"]
            )
        detector = build_provider("scene_detector", cfg.scenes.detector, cfg.scenes)
        run.scenes = detector.detect(run.source, run.ctx)
        return StageResult(
            StageStatus.OK, **_provider_fields(detector), counts={"scenes": len(run.scenes)}
        )

    def _frames(self, run: Run) -> StageResult:
        cfg = run.config
        sampler = build_provider("sampler", cfg.sampling.mode, cfg.sampling)
        run.plan = sampler.plan(run.ctx, run.scenes or None)
        log.info(
            "frames.plan",
            extra={"plan": run.plan.describe(), "estimated": run.plan.estimated_frames()},
        )
        run.frames = extract_plan(
            run.source,
            run.plan,
            run.workspace.frames_dir,
            scale_width=cfg.sampling.scale_width,
            quality=cfg.sampling.jpeg_quality,
            timeout=cfg.limits.ffmpeg_timeout_s,
            max_frames=cfg.sampling.max_frames,
        )
        if not run.frames:
            # A video with visual content that yielded no frames is a failure, not an empty
            # result: every downstream visual stage would report "nothing on screen" for a
            # screen nobody managed to look at.
            status = StageStatus.OK if not run.video.has_video else StageStatus.FAILED
            return StageResult(
                status,
                provider=sampler.name,
                error=None if status is StageStatus.OK else "no frames were extracted",
                counts={"frames": 0},
            )
        return StageResult(
            StageStatus.OK,
            **_provider_fields(sampler),
            counts={
                "frames": len(run.frames),
                "planned": run.plan.estimated_frames(),
            },
        )

    def _ocr(self, run: Run) -> StageResult:
        cfg = run.config
        if not cfg.ocr.enabled:
            return StageResult(StageStatus.SKIPPED, warnings=["ocr.enabled is false"])
        engine = build_provider("ocr", cfg.ocr.provider, cfg.ocr)
        if not engine.available():
            # §7: a missing OCR binary is a skip, not a failure. The distinction is what the
            # document needs — "no text on screen" and "nobody read the screen" are different
            # facts, and only `status` can tell them apart.
            return StageResult(
                StageStatus.SKIPPED,
                **_provider_fields(engine),
                warnings=[f"OCR provider {engine.name!r} is not available on this machine"],
            )
        run.observations = engine.extract(run.frames, run.ctx)
        run.ocr = deduplicate(
            run.observations,
            config=cfg.ocr,
            duration=run.video.duration,
            frame_ts=[f.ts for f in run.frames],
            engine=engine.name,
        )
        return StageResult(
            StageStatus.OK,
            **_provider_fields(engine),
            counts={
                "observations": len(run.observations),
                "events": len(run.ocr),
                "stable": sum(1 for t in run.ocr if t.stable),
            },
        )

    def _scene_refine(self, run: Run) -> StageResult:
        cfg = run.config
        if not cfg.scenes.use_ocr_signal:
            return StageResult(StageStatus.SKIPPED, warnings=["scenes.use_ocr_signal is false"])
        if not run.scenes or not run.ocr:
            return StageResult(
                StageStatus.SKIPPED, warnings=["no scenes or no on-screen text to combine"]
            )
        before = len(run.scenes)
        run.scenes = refine_with_text(
            run.scenes,
            run.ocr,
            duration=run.video.duration,
            min_scene_duration=cfg.scenes.min_scene_duration,
        )
        return StageResult(
            StageStatus.OK,
            provider="text-signal",
            counts={"before": before, "after": len(run.scenes)},
        )

    def _audio(self, run: Run) -> StageResult:
        cfg = run.config
        if not cfg.asr.enabled:
            return StageResult(StageStatus.SKIPPED, warnings=["asr.enabled is false"])
        if not run.video.has_audio:
            return StageResult(StageStatus.SKIPPED, warnings=["the video has no audio stream"])
        run.audio = extract_audio(
            run.source,
            run.workspace.audio_dir / "audio.wav",
            timeout=cfg.limits.ffmpeg_timeout_s,
        )
        if run.audio is None:
            return StageResult(StageStatus.FAILED, error="audio extraction produced no output")
        return StageResult(
            StageStatus.OK,
            provider="ffmpeg",
            counts={"bytes": run.audio.stat().st_size},
        )

    def _asr(self, run: Run) -> StageResult:
        cfg = run.config
        engine, warnings = resolve_engine(cfg.asr, source=run.source)
        output = engine.transcribe(run.audio, run.ctx)
        run.transcript = finalize_transcript(
            output.utterances, duration=run.video.duration, language=output.language
        )
        fields = _provider_fields(engine)
        # A resolved-to-null engine means the chain ran out of providers, so nothing listened
        # to the audio. Reporting that as `ok` with an empty transcript would assert silence.
        status = StageStatus.SKIPPED if engine.name == "null" else StageStatus.OK
        return StageResult(
            status,
            **fields,
            warnings=[*warnings, *output.warnings],
            counts={
                "utterances": len(run.transcript),
                "words": sum(len(u.words) for u in run.transcript),
            },
        )

    def _vision(self, run: Run) -> StageResult:
        cfg = run.config
        if not cfg.vision.enabled:
            # The default. §32: nothing leaves the machine unless the operator asked.
            return StageResult(StageStatus.SKIPPED, warnings=["vision.enabled is false"])
        provider = build_provider("vision", cfg.vision.provider, cfg.vision)
        if not provider.available():
            return StageResult(
                StageStatus.SKIPPED,
                **_provider_fields(provider),
                warnings=[f"vision provider {provider.name!r} is not available"],
            )
        selected = _evenly(run.frames, cfg.vision.max_frames)
        outputs = provider.describe(selected, run.ctx)
        run.vision = [
            VisionNote(
                id=f"vis_{index:04d}",
                start=out.start,
                end=out.end,
                description=out.description,
                entities=list(out.entities),
                actions=list(out.actions),
                ui=dict(out.ui),
                confidence=out.confidence,
                provider=provider.name,
                model=cfg.vision.model,
                frame_ids=list(out.frame_ids),
            )
            for index, out in enumerate(outputs)
        ]
        fields = _provider_fields(provider)
        return StageResult(
            StageStatus.OK,
            **fields,
            warnings=(
                []
                if not fields["remote"]
                else [f"{len(selected)} frames were sent to {provider.name!r}"]
            ),
            counts={"frames_described": len(selected), "notes": len(run.vision)},
        )

    def _events(self, run: Run) -> StageResult:
        doc = _require_doc(run, "events")
        detector = build_provider("event_detector", "rules")
        events: list[Event] = list(detector.detect(doc, run.ctx))
        doc.events = events
        return StageResult(
            StageStatus.OK,
            **_provider_fields(detector),
            counts={
                "events": len(events),
                **{
                    f"type:{name}": sum(1 for e in events if e.type == name)
                    for name in sorted({e.type for e in events})
                },
            },
        )

    def _segments(self, run: Run) -> StageResult:
        doc = _require_doc(run, "segments")
        doc.segments = build_segments(doc, run.config.segments)
        covered = sum(1 for s in doc.segments if s.text)
        return StageResult(
            StageStatus.OK,
            provider="scene-aligned",
            counts={"segments": len(doc.segments), "with_text": covered},
        )

    # -- assembly -----------------------------------------------------------

    def _assemble(self, run: Run) -> VideoContextDocument:
        """Turn the extracted transport objects into the document's schema models."""
        cfg = run.config
        scenes = [
            Scene(
                id=f"scene_{index:04d}",
                start=span.start,
                end=span.end,
                confidence=span.score,
                detector=cfg.scenes.detector,
                keyframe_ts=span.keyframe_ts,
                change_score=span.score,
                signals=list(span.signals),
            )
            for index, span in enumerate(run.scenes)
        ]
        frames = [
            Frame(
                id=image.id,
                ts=image.ts,
                index=image.index,
                # Relative to the workspace, per spec §13: an absolute path leaks the
                # operator's directory layout into a document meant to be shared.
                path=run.workspace.relative(image.path),
                width=image.width,
                height=image.height,
                reason=image.reason,
                diff_score=image.diff_score,
                phash=image.phash,
            )
            for image in run.frames
        ]
        return VideoContextDocument(
            id=run.video.id,
            producer=Producer(config_hash=cfg.full_hash()),
            video=run.video,
            scenes=scenes,
            transcript=run.transcript,
            ocr=run.ocr,
            vision=run.vision,
            frames=frames,
        )

    def _apply_cached_result(self, stage_name: str, cached_data: Any, run: Run) -> None:
        """Apply cached stage data to the run object."""
        if stage_name == "scenes":
            run.scenes = cached_data
        elif stage_name == "frames":
            run.frames = cached_data
            run.plan = cached_data.get("plan")
        elif stage_name == "ocr":
            run.ocr = cached_data.get("ocr", [])
            run.observations = cached_data.get("observations", [])
        elif stage_name == "scene_refine":
            run.scenes = cached_data
        elif stage_name == "audio":
            run.audio = cached_data
        elif stage_name == "asr":
            run.transcript = cached_data
        elif stage_name == "vision":
            run.vision = cached_data
        elif stage_name == "events":
            if run.doc:
                run.doc.events = cached_data
        elif stage_name == "segments":
            if run.doc:
                run.doc.segments = cached_data

    def _get_cached_data(self, stage_name: str, run: Run) -> Any:
        """Extract cacheable data from the run object for a stage."""
        if stage_name == "scenes":
            return run.scenes
        elif stage_name == "frames":
            return {"frames": run.frames, "plan": run.plan}
        elif stage_name == "ocr":
            return {"ocr": run.ocr, "observations": run.observations}
        elif stage_name == "scene_refine":
            return run.scenes
        elif stage_name == "audio":
            return run.audio
        elif stage_name == "asr":
            return run.transcript
        elif stage_name == "vision":
            return run.vision
        elif stage_name == "events":
            return run.doc.events if run.doc else None
        elif stage_name == "segments":
            return run.doc.segments if run.doc else None
        return None

    def _get_cached_counts(self, stage_name: str, cached_data: Any) -> dict[str, int]:
        """Generate counts for a cached stage result."""
        if stage_name == "scenes":
            return {"scenes": len(cached_data)}
        elif stage_name == "frames":
            return {"frames": len(cached_data.get("frames", []))}
        elif stage_name == "ocr":
            return {"events": len(cached_data.get("ocr", [])), "observations": len(cached_data.get("observations", []))}
        elif stage_name == "scene_refine":
            return {"scenes": len(cached_data)}
        elif stage_name == "audio":
            return {"bytes": cached_data.stat().st_size} if cached_data else {}
        elif stage_name == "asr":
            return {"utterances": len(cached_data)}
        elif stage_name == "vision":
            return {"notes": len(cached_data)}
        elif stage_name == "events":
            return {"events": len(cached_data)}
        elif stage_name == "segments":
            return {"segments": len(cached_data)}
        return {}

    def _metrics(self, run: Run, elapsed: float) -> Metrics:
        duration = run.video.duration
        planned = run.plan.estimated_frames() if run.plan else 0
        return Metrics(
            processing_time_s=round(elapsed, 3),
            video_duration_s=duration,
            realtime_factor=_significant(elapsed / duration) if duration > 0 else None,
            frames_sampled=len(run.frames),
            # Frames the plan asked for that the extractor did not emit: dropped as duplicates
            # or clipped by the cap. Negative would mean the extractor invented frames, so it
            # is floored rather than reported as a nonsense saving.
            frames_skipped=max(0, planned - len(run.frames)),
            stage_times=dict(run.stage_times),
            cache_hits=sum(1 for r in run.records if r.cached),
        )


def process(source: str | Path, config: ProcessingConfig | None = None) -> VideoContextDocument:
    """One-shot convenience wrapper: ``Pipeline(config).run(source)``."""
    return Pipeline(config).run(source)


__all__ = [
    "STAGE_VERSIONS",
    "Pipeline",
    "Run",
    "Stage",
    "StageResult",
    "build_provider",
    "process",
]
