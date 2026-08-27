"""The stage runner: what happens when a stage cannot run, and what the document says about it.

The pipeline's job is not to extract anything — the stages do that, and they have their own
tests. Its job is to keep going when one of them cannot, and to leave behind a record that
distinguishes *nothing was there* from *nobody looked*. Those two produce identical empty
arrays, and only ``StageRecord.status`` separates them, so that distinction is what most of
these tests are about. It matters most for the stages that are genuinely absent on a normal
machine — a Whisper model, a Tesseract binary — which is exactly where a silent empty result
would be least noticed.

No media is decoded. The three functions that touch real files are patched and the providers
are fakes, because a test that needed FFmpeg to prove "a failed OCR stage does not lose the
transcript" would be testing FFmpeg. The real thing runs end-to-end in ``tests/integration``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videocontent import registry
from videocontent.config import ProcessingConfig
from videocontent.errors import DependencyMissingError, ProviderError
from videocontent.interfaces import (
    ASROutput,
    FrameImage,
    OCRObservation,
    SamplePlan,
    SampleWindow,
    SceneSpan,
    VisionOutput,
)
from videocontent.processing import pipeline as pipe
from videocontent.processing.asr import utterance
from videocontent.processing.pipeline import Pipeline, build_provider
from videocontent.schema.v1 import StageStatus, VideoInfo

# The fake sampler has to be registered *over* a built-in name: ``sampling.mode`` is a Literal,
# so ``cfg.sampling.mode = "fake-sampler"`` is rejected by config validation before the registry
# is ever consulted. Every other capability is selected by plain string.
SAMPLER_NAME = "adaptive"


# --------------------------------------------------------------------------- fakes


class FakeScenes:
    name = "fake-scenes"
    version = "9.9"

    def __init__(self, config=None) -> None:
        self.config = config

    def detect(self, video_path, ctx):
        half = ctx.duration / 2
        return [
            SceneSpan(0.0, half, score=0.8, signals=["fake"]),
            SceneSpan(half, ctx.duration, score=0.6, signals=["fake"]),
        ]


class NoArgsScenes:
    """A provider with no ``__init__`` at all — the shape ``NullSceneDetector`` has."""

    name = "no-args"
    version = "1"

    def detect(self, video_path, ctx):
        return [SceneSpan(0.0, ctx.duration, signals=["none"])]


class FakeSampler:
    name = "fake-sampler"
    version = "1"

    def __init__(self, config=None) -> None:
        self.config = config

    def plan(self, ctx, scenes=None):
        return SamplePlan(
            windows=[SampleWindow(0.0, ctx.duration, fps=1.0)], duration=ctx.duration
        )


class FakeOCR:
    name = "fake-ocr"
    version = "1"

    def __init__(self, config=None, *, usable: bool = True) -> None:
        self.config = config
        self.usable = usable

    def available(self) -> bool:
        return self.usable

    def extract(self, frames, ctx):
        return [
            OCRObservation(text="Pricing", ts=f.ts, confidence=0.9, block_index=0) for f in frames
        ]


class UnavailableOCR(FakeOCR):
    name = "missing-ocr"

    def __init__(self, config=None) -> None:
        super().__init__(config, usable=False)


class ExplodingOCR:
    """Fails the way a real adapter fails: a typed error the pipeline knows how to record."""

    name = "exploding-ocr"
    version = "1"

    def __init__(self, config=None) -> None:
        self.config = config

    def available(self) -> bool:
        return True

    def extract(self, frames, ctx):
        raise ProviderError("the OCR binary segfaulted")


class CrashingOCR(ExplodingOCR):
    """Fails the way a *bug* fails: an exception nothing in core anticipated."""

    name = "crashing-ocr"

    def extract(self, frames, ctx):
        raise RuntimeError("a bug nobody anticipated")


class FakeASR:
    name = "fake-asr"
    version = "1"
    remote = False

    def __init__(self, config=None) -> None:
        self.config = config

    def available(self) -> bool:
        return True

    def transcribe(self, audio_path, ctx):
        return ASROutput(
            utterances=[utterance("hello there", 1.0, 4.0, confidence=0.9)],
            language="en",
            model="fake",
        )


class RemoteVision:
    name = "remote-vision"
    version = "1"
    remote = True

    def __init__(self, config=None) -> None:
        self.config = config

    def available(self) -> bool:
        return True

    def describe(self, frames, ctx):
        return [
            VisionOutput(
                description="a slide with a chart",
                start=frames[0].ts,
                end=frames[-1].ts,
                entities=["chart"],
                frame_ids=[f.id for f in frames],
            )
        ]


PROVIDERS = (
    ("scene_detector", "fake-scenes", FakeScenes),
    ("scene_detector", "no-args", NoArgsScenes),
    ("sampler", SAMPLER_NAME, FakeSampler),
    ("ocr", "fake-ocr", FakeOCR),
    ("ocr", "missing-ocr", UnavailableOCR),
    ("ocr", "exploding-ocr", ExplodingOCR),
    ("ocr", "crashing-ocr", CrashingOCR),
    ("asr", "fake-asr", FakeASR),
    ("vision", "remote-vision", RemoteVision),
)


@pytest.fixture(autouse=True)
def fake_providers():
    for capability, name, factory in PROVIDERS:
        registry.register(capability, name, factory, override=True)
    yield
    # The built-ins come back on the next lookup via `_bootstrap`, so clearing is enough to keep
    # these fakes — and the overridden sampler name — out of another module's registry.
    registry.clear()


@pytest.fixture
def media(monkeypatch):
    """Patch the three functions that touch real media, and record what they were handed."""
    calls: dict[str, object] = {}

    def fake_probe(source, *, limits=None, compute_hash=True):
        calls["probe_limits"] = limits
        calls["probe_hash"] = compute_hash
        return VideoInfo(
            id="vid_fake",
            filename=Path(str(source)).name,
            duration=60.0,
            fps=30.0,
            width=1280,
            height=720,
            has_audio=True,
            has_video=True,
            content_hash="sha256:" + "0" * 64,
        )

    def fake_extract_plan(source, plan, outdir, **kw):
        calls["frames_kwargs"] = kw
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        return [
            FrameImage(
                ts=float(index) * 6.0,
                path=directory / f"frame_{index:04d}.jpg",
                index=index,
                width=1280,
                height=720,
                reason="fixed",
            )
            for index in range(10)
        ]

    def fake_extract_audio(source, out_path, **kw):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFfake")
        return path

    monkeypatch.setattr(pipe, "probe", fake_probe)
    monkeypatch.setattr(pipe, "extract_plan", fake_extract_plan)
    monkeypatch.setattr(pipe, "extract_audio", fake_extract_audio)
    return calls


def missing_stream(monkeypatch, *, duration: float = 30.0, **fields):
    """Re-patch the probe to report a video that lacks one of the two streams."""

    def probe(source, *, limits=None, compute_hash=True):
        return VideoInfo(id="vid_x", filename="x.mp4", duration=duration, **fields)

    monkeypatch.setattr(pipe, "probe", probe)


def config(tmp_path) -> ProcessingConfig:
    cfg = ProcessingConfig(workdir=tmp_path / "work")
    cfg.scenes.detector = "fake-scenes"
    cfg.ocr.provider = "fake-ocr"
    cfg.asr.provider = "fake-asr"
    return cfg


def record(doc, name):
    return next(r for r in doc.stages if r.name == name)


# --------------------------------------------------------------------------- tests


class TestHappyPath:
    def test_it_produces_a_document(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert doc.id == "vid_fake"
        assert doc.video.filename == "demo.mp4"

    def test_every_stage_is_recorded_once_in_order(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert [s.name for s in doc.stages] == [s.name for s in Pipeline().stages()]

    def test_the_modalities_are_populated(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert len(doc.scenes) == 2
        assert len(doc.frames) == 10
        assert len(doc.transcript) == 1
        assert doc.ocr and doc.events and doc.segments

    def test_provenance_is_recorded(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        scenes = record(doc, "scenes")
        assert scenes.provider == "fake-scenes"
        assert scenes.provider_version == "9.9"
        assert scenes.started_at is not None
        assert scenes.duration_s is not None

    def test_the_config_hash_is_per_stage_not_global(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        # Two stages reading different config sections must not share a hash, or a cache keyed
        # on it would serve one stage's artifacts to the other.
        assert record(doc, "scenes").config_hash != record(doc, "ocr").config_hash
        assert record(doc, "events").config_hash is None  # reads no config section

    def test_nothing_is_marked_cached_yet(self, media, tmp_path):
        # Cache persistence is V0.2. Recording `cached=True` before it exists would make the
        # metric lie in exactly the direction that flatters the project.
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert not any(s.cached for s in doc.stages)
        assert doc.metrics.cache_hits == 0

    def test_frame_paths_are_relative_to_the_workspace(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        # Spec §13: an absolute path would leak the operator's directory layout into a document
        # meant to be handed to someone else.
        assert all(not Path(f.path).is_absolute() for f in doc.frames)
        assert doc.frames[0].path.startswith("frames")

    def test_the_producer_records_the_config_hash(self, media, tmp_path):
        cfg = config(tmp_path)
        doc = Pipeline(cfg).run("demo.mp4")
        assert doc.producer.config_hash == cfg.full_hash()

    def test_the_limits_reach_the_probe(self, media, tmp_path):
        cfg = config(tmp_path)
        Pipeline(cfg).run("demo.mp4")
        # §31: size, duration and container limits are enforced before a frame is decoded.
        # Dropping the argument in a refactor would disable every one of them silently.
        assert media["probe_limits"] is cfg.limits


class TestDependencies:
    def test_no_audio_skips_audio_and_asr(self, media, monkeypatch, tmp_path):
        missing_stream(monkeypatch, has_audio=False)
        doc = Pipeline(config(tmp_path)).run("silent.mp4")

        assert record(doc, "audio").status is StageStatus.SKIPPED
        assert "no audio stream" in record(doc, "audio").warnings[0]
        assert record(doc, "asr").status is StageStatus.SKIPPED
        assert doc.transcript == []

    def test_a_skipped_dependency_names_itself(self, media, monkeypatch, tmp_path):
        missing_stream(monkeypatch, has_audio=False)
        doc = Pipeline(config(tmp_path)).run("silent.mp4")
        # The reason has to survive into the document: "asr: skipped" with no explanation is
        # not something an operator can act on.
        assert "requires audio" in record(doc, "asr").warnings[0]

    def test_a_failed_dependency_skips_the_dependent_stage(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.ocr.provider = "exploding-ocr"
        doc = Pipeline(cfg).run("demo.mp4")
        assert record(doc, "ocr").status is StageStatus.FAILED
        assert record(doc, "scene_refine").status is StageStatus.SKIPPED

    def test_an_unrelated_failure_does_not_stop_the_run(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.ocr.provider = "exploding-ocr"
        doc = Pipeline(cfg).run("demo.mp4")
        # The whole point of the degradation design: losing on-screen text must not cost the
        # transcript, the scenes or the segments.
        assert doc.transcript and doc.scenes and doc.segments
        assert record(doc, "asr").status is StageStatus.OK


class TestFailureContainment:
    def test_a_known_error_is_recorded_with_its_message(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.ocr.provider = "exploding-ocr"
        doc = Pipeline(cfg).run("demo.mp4")
        assert record(doc, "ocr").error == "the OCR binary segfaulted"

    def test_an_unexpected_exception_is_contained_and_typed(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.ocr.provider = "crashing-ocr"
        doc = Pipeline(cfg).run("demo.mp4")
        # A provider bug must not take the document down with it, and "it failed" without the
        # exception type is not actionable.
        assert record(doc, "ocr").error == "RuntimeError: a bug nobody anticipated"
        assert doc.segments

    def test_an_unknown_provider_name_is_a_stage_failure(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.scenes.detector = "does-not-exist"
        doc = Pipeline(cfg).run("demo.mp4")
        # A typo in one provider name costs that stage, not the run.
        assert record(doc, "scenes").status is StageStatus.FAILED
        assert doc.frames and doc.transcript

    def test_a_missing_binary_is_skipped_not_failed(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.ocr.provider = "missing-ocr"
        doc = Pipeline(cfg).run("demo.mp4")
        ocr = record(doc, "ocr")
        # §7: an absent dependency is a skip. `ok` with an empty list would assert that the
        # screen was read and had no text on it.
        assert ocr.status is StageStatus.SKIPPED
        assert ocr.error is None
        assert "not available" in ocr.warnings[0]

    def test_an_undecodable_container_produces_no_document(self, media, monkeypatch, tmp_path):
        def bad_probe(source, *, limits=None, compute_hash=True):
            raise DependencyMissingError("ffprobe could not read the container")

        monkeypatch.setattr(pipe, "probe", bad_probe)
        # §7: fail fast. A document whose duration is unknown has nothing to anchor a timestamp
        # to, so a partial one would be worse than none.
        with pytest.raises(DependencyMissingError):
            Pipeline(config(tmp_path)).run("broken.mp4")

    def test_no_frames_from_a_video_track_is_a_failure(self, media, monkeypatch, tmp_path):
        monkeypatch.setattr(pipe, "extract_plan", lambda *a, **kw: [])
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        # Not an empty result: every visual stage downstream would otherwise report "nothing on
        # screen" for a screen nobody managed to look at.
        assert record(doc, "frames").status is StageStatus.FAILED
        assert record(doc, "ocr").status is StageStatus.SKIPPED

    def test_no_frames_from_an_audio_only_file_is_not_a_failure(
        self, media, monkeypatch, tmp_path
    ):
        missing_stream(monkeypatch, has_video=False, has_audio=True)
        monkeypatch.setattr(pipe, "extract_plan", lambda *a, **kw: [])
        doc = Pipeline(config(tmp_path)).run("podcast.m4a")
        # Zero frames is the correct outcome here, not a fault, and the transcript still lands.
        assert record(doc, "frames").status is StageStatus.OK
        assert doc.transcript


class TestPrivacy:
    def test_the_default_run_is_entirely_local(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        # §32 as a checkable fact rather than a README promise: no stage in a default run
        # reports having sent anything off the machine.
        assert not any(s.remote for s in doc.stages)

    def test_vision_is_skipped_by_default(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert record(doc, "vision").status is StageStatus.SKIPPED
        assert doc.vision == []

    def test_a_remote_provider_is_flagged_and_says_what_it_sent(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.vision.enabled = True
        cfg.vision.provider = "remote-vision"
        doc = Pipeline(cfg).run("demo.mp4")
        vision = record(doc, "vision")
        assert vision.remote is True
        assert "frames were sent" in vision.warnings[0]
        assert doc.vision[0].description == "a slide with a chart"

    def test_the_frame_budget_is_respected(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.vision.enabled = True
        cfg.vision.provider = "remote-vision"
        cfg.vision.max_frames = 3
        doc = Pipeline(cfg).run("demo.mp4")
        # An upload budget that is not enforced is not a budget.
        assert record(doc, "vision").counts["frames_described"] == 3


class TestProviderConstruction:
    def test_a_provider_that_takes_no_arguments_still_works(self, media, tmp_path):
        # §25: core cannot demand a constructor shape. A pipeline that always passed `config=`
        # would be unable to use its own fallback detector.
        cfg = config(tmp_path)
        cfg.scenes.detector = "no-args"
        doc = Pipeline(cfg).run("demo.mp4")
        assert record(doc, "scenes").status is StageStatus.OK
        assert len(doc.scenes) == 1

    def test_config_reaches_a_provider_that_accepts_it(self):
        assert build_provider("scene_detector", "fake-scenes", "sentinel").config == "sentinel"

    def test_config_is_withheld_from_one_that_does_not(self):
        assert build_provider("scene_detector", "no-args", "sentinel").name == "no-args"

    def test_the_builtin_null_detector_is_constructible(self):
        # The regression that motivated the tolerance: NullSceneDetector defines no __init__,
        # so `create("scene_detector", "null", config=...)` raises TypeError.
        assert build_provider("scene_detector", "null", object()).name == "null"


class TestSpeechDegradation:
    def test_a_null_engine_is_a_skip_not_an_empty_transcript(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.asr.provider = "null"
        cfg.asr.fallback_to_subtitles = False
        doc = Pipeline(cfg).run("demo.mp4")
        # Reporting `ok` with no utterances would assert the video is silent. Nobody listened.
        assert record(doc, "asr").status is StageStatus.SKIPPED

    def test_disabling_speech_skips_the_audio_stage_too(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.asr.enabled = False
        doc = Pipeline(cfg).run("demo.mp4")
        # No point paying for a WAV nobody will transcribe.
        assert record(doc, "audio").status is StageStatus.SKIPPED
        assert record(doc, "asr").status is StageStatus.SKIPPED

    def test_disabling_ocr_skips_it(self, media, tmp_path):
        cfg = config(tmp_path)
        cfg.ocr.enabled = False
        doc = Pipeline(cfg).run("demo.mp4")
        assert record(doc, "ocr").status is StageStatus.SKIPPED
        assert doc.ocr == []


class TestMetrics:
    def test_only_stages_that_ran_are_timed(self, media, monkeypatch, tmp_path):
        missing_stream(monkeypatch, has_audio=False)
        doc = Pipeline(config(tmp_path)).run("silent.mp4")
        timed = set(doc.metrics.stage_times)
        assert timed == {s.name for s in doc.stages if s.duration_s is not None}
        # `asr` never got invoked — a fabricated 0.0 would read as "it ran and was instant".
        assert "asr" not in timed
        assert "audio" in timed  # invoked; it decided to skip itself

    def test_the_realtime_factor_is_measured(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        m = doc.metrics
        assert m.video_duration_s == 60.0
        assert m.processing_time_s is not None and m.processing_time_s > 0
        # The regression: this run takes milliseconds against a nominal 60 seconds, and at three
        # decimal places the factor rounded to 0.0 — reporting "not measured" for the fastest
        # possible run. Significant figures keep it. No comparison against `processing_time_s`,
        # which is itself rounded and would make the assertion a rounding-error test.
        assert 0.0 < m.realtime_factor < 1.0

    def test_a_zero_duration_video_has_no_realtime_factor(self, media, monkeypatch, tmp_path):
        missing_stream(monkeypatch, duration=0.0, has_video=False)
        monkeypatch.setattr(pipe, "extract_plan", lambda *a, **kw: [])
        doc = Pipeline(config(tmp_path)).run("empty.mp4")
        # None, not 0.0: there is nothing to divide by, which is a different fact from "fast".
        assert doc.metrics.realtime_factor is None
        assert doc.segments == []  # no timeline to place a segment on

    def test_frames_skipped_is_the_planned_shortfall(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        frames = record(doc, "frames")
        assert doc.metrics.frames_sampled == 10
        assert doc.metrics.frames_skipped == frames.counts["planned"] - 10

    def test_frames_skipped_is_never_negative(self, media, monkeypatch, tmp_path):
        # An extractor emitting more frames than planned must not report a nonsense saving.
        real = pipe.extract_plan
        monkeypatch.setattr(
            pipe,
            "extract_plan",
            lambda source, plan, outdir, **kw: real(source, plan, outdir, **kw) * 200,
        )
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert doc.metrics.frames_sampled == 2000
        assert doc.metrics.frames_skipped == 0

    def test_counts_are_recorded_per_stage(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert record(doc, "scenes").counts == {"scenes": 2}
        assert record(doc, "frames").counts["frames"] == 10
        assert record(doc, "events").counts["events"] == len(doc.events)
        assert record(doc, "segments").counts["segments"] == len(doc.segments)


class TestDerivedStages:
    def test_events_cite_facts_that_exist(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        known = {
            "scenes": {s.id for s in doc.scenes},
            "ocr": {t.id for t in doc.ocr},
            "transcript": {u.id for u in doc.transcript},
        }
        for event in doc.events:
            # Spec §11: an event with no refs is an unsupported claim about a timestamp, which
            # is the one thing the evidence-first contract exists to rule out.
            assert event.refs, f"{event.id} cites nothing"
            for kind, refs in event.refs.items():
                assert kind in known, f"{event.id} cites unknown kind {kind!r}"
                assert set(refs) <= known[kind], f"{event.id} cites a missing {kind} id"

    def test_segments_cite_facts_that_exist(self, media, tmp_path):
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        for segment in doc.segments:
            assert set(segment.scene_ids) <= {s.id for s in doc.scenes}
            assert set(segment.event_ids) <= {e.id for e in doc.events}
            assert set(segment.frame_ids) <= {f.id for f in doc.frames}
            assert set(segment.transcript_ids) <= {u.id for u in doc.transcript}

    def test_segments_see_the_events_from_the_stage_before(self, media, tmp_path):
        # Ordering regression: if `segments` ran before `events`, every event_ids list would be
        # empty and nothing else about the document would look wrong.
        doc = Pipeline(config(tmp_path)).run("demo.mp4")
        assert any(s.event_ids for s in doc.segments)

    def test_derived_stages_still_run_when_extraction_degraded(self, media, monkeypatch, tmp_path):
        missing_stream(monkeypatch, has_audio=False)
        cfg = config(tmp_path)
        cfg.ocr.provider = "crashing-ocr"
        doc = Pipeline(cfg).run("silent.mp4")
        # Scenes and frames alone are still evidence, and must still be segmented.
        assert record(doc, "segments").status is StageStatus.OK
        assert doc.segments


class TestEvenSpread:
    def test_it_returns_everything_below_the_limit(self):
        assert pipe._evenly([1, 2, 3], 10) == [1, 2, 3]

    def test_it_spans_the_whole_list(self):
        # Truncation would describe the first minute of a video and nothing else.
        picked = pipe._evenly(list(range(100)), 5)
        assert len(picked) == 5
        assert picked[0] == 0
        assert picked[-1] >= 80

    def test_it_never_exceeds_the_limit(self):
        assert len(pipe._evenly(list(range(1000)), 7)) == 7

    def test_it_stays_in_range(self):
        items = list(range(10))
        assert all(x in items for x in pipe._evenly(items, 4))
