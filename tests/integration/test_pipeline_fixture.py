"""The whole pipeline against the real fixture: are the document's own promises kept?

Every other test either stubs the media or exercises one stage. This one runs the real thing —
FFmpeg, Tesseract, faster-whisper — and then checks the *structural* guarantees that the format
makes and that nothing else can check. They are the invariants a consumer relies on and that no
individual stage owns:

* **no dangling reference.** Every id a segment or an event cites resolves to an object in the
  same document. A dangling ref is an answer whose evidence cannot be produced.
* **no orphaned fact.** Every scene, utterance, OCR event and frame is cited by at least one
  segment. An orphan is evidence search can never reach, which looks like a ranking bug and
  is not one.
* **every event cites something** (spec §11). An event with no refs is an unsupported claim
  about a timestamp — the one thing the evidence-first contract exists to forbid.
* **segments tile the timeline.** No gaps, no overlaps, ending at the media duration. A gap is
  a stretch of video that cannot be retrieved at all.
* **the document round-trips.** ``load(dump(doc))`` is the document, or the ``.vctx`` file is
  not a faithful record of what was extracted.
* **no absolute paths.** A frame path that names the operator's home directory leaks it into a
  document meant to be shared (spec §13).

Deliberately *not* asserted here: how many scenes, how many OCR events, or which strings were
read. Those are the recognition-quality claims, they belong to ``test_ocr_fixture.py`` and
``test_asr_fixture.py`` where they are tied to the manifest, and pinning them again here would
turn a change in Tesseract's version into a failure of the pipeline's contract.

Marked ``slow``: this decodes a 62-second video and loads a Whisper model.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from conftest import needs_demo, needs_ffmpeg
from videocontent.config import ProcessingConfig
from videocontent.processing.pipeline import Pipeline
from videocontent.schema.v1 import StageStatus, VideoContextDocument

pytestmark = [pytest.mark.integration, pytest.mark.slow, needs_ffmpeg, needs_demo]

TOLERANCE_S = 0.01


@pytest.fixture(scope="module")
def document(demo_video, tmp_path_factory) -> VideoContextDocument:
    """One real run, shared by every assertion below — it costs ~30 seconds."""
    config = ProcessingConfig(workdir=tmp_path_factory.mktemp("workspace"))
    return Pipeline(config).run(demo_video)


class TestItRan:
    def test_no_stage_failed(self, document):
        failed = {s.name: s.error for s in document.stages if s.status is StageStatus.FAILED}
        assert failed == {}

    def test_the_visual_and_audio_tracks_were_both_read(self, document):
        # Not a quality claim, a coverage one: if either came back empty the invariants below
        # would hold trivially and prove nothing.
        assert document.scenes and document.frames
        assert document.segments and document.events

    def test_the_run_is_recorded_as_local(self, document):
        # §32: the default configuration sends nothing off the machine, and the document says so
        # rather than asking the reader to trust the README.
        assert not any(s.remote for s in document.stages)

    def test_every_stage_reports_a_provider_or_a_reason(self, document):
        for stage in document.stages:
            if stage.status is StageStatus.OK:
                assert stage.provider, f"{stage.name} ran anonymously"
            elif stage.status is StageStatus.SKIPPED:
                assert stage.warnings, f"{stage.name} was skipped with no reason given"


class TestReferentialIntegrity:
    def ids(self, document):
        return {
            "scenes": {s.id for s in document.scenes},
            "transcript": {u.id for u in document.transcript},
            "ocr": {t.id for t in document.ocr},
            "vision": {n.id for n in document.vision},
            "events": {e.id for e in document.events},
            "frames": {f.id for f in document.frames},
        }

    def test_segments_cite_nothing_that_does_not_exist(self, document):
        known = self.ids(document)
        for segment in document.segments:
            for kind, field in (
                ("scenes", "scene_ids"),
                ("transcript", "transcript_ids"),
                ("ocr", "ocr_ids"),
                ("vision", "vision_ids"),
                ("events", "event_ids"),
                ("frames", "frame_ids"),
            ):
                dangling = set(getattr(segment, field)) - known[kind]
                assert not dangling, f"{segment.id}.{field} cites missing {kind}: {dangling}"

    def test_every_event_cites_its_evidence(self, document):
        known = self.ids(document)
        for event in document.events:
            assert event.refs, f"{event.id} ({event.type}) cites nothing"
            for kind, refs in event.refs.items():
                assert kind in known, f"{event.id} cites unknown kind {kind!r}"
                assert not set(refs) - known[kind], f"{event.id} cites a missing {kind}"

    def test_no_fact_is_orphaned(self, document):
        known = self.ids(document)
        cited = {
            "scenes": set(),
            "transcript": set(),
            "ocr": set(),
            "vision": set(),
            "events": set(),
            "frames": set(),
        }
        for segment in document.segments:
            cited["scenes"] |= set(segment.scene_ids)
            cited["transcript"] |= set(segment.transcript_ids)
            cited["ocr"] |= set(segment.ocr_ids)
            cited["vision"] |= set(segment.vision_ids)
            cited["events"] |= set(segment.event_ids)
            cited["frames"] |= set(segment.frame_ids)
        for kind, expected in known.items():
            assert not expected - cited[kind], f"{kind} unreachable from any segment"


class TestTimeline:
    def test_segments_tile_the_video(self, document):
        segments = document.segments
        assert segments[0].start == pytest.approx(0.0, abs=TOLERANCE_S)
        assert segments[-1].end == pytest.approx(document.video.duration, abs=TOLERANCE_S)
        for earlier, later in pairwise(segments):
            # A gap is a stretch of video no query can reach; an overlap double-counts evidence.
            assert later.start == pytest.approx(earlier.end, abs=TOLERANCE_S)

    def test_no_span_runs_backwards(self, document):
        for items in (
            document.scenes,
            document.transcript,
            document.ocr,
            document.segments,
            document.events,
        ):
            for item in items:
                assert item.end >= item.start, f"{item.id} ends before it starts"

    def test_nothing_is_timestamped_past_the_video(self, document):
        limit = document.video.duration + TOLERANCE_S
        for items in (document.scenes, document.transcript, document.ocr, document.segments):
            for item in items:
                assert item.end <= limit, f"{item.id} ends after the video does"
        for frame in document.frames:
            assert frame.ts <= limit

    def test_ocr_evidence_lies_inside_its_own_span(self, document):
        # `start`/`end` are sampling-derived estimates; `first_frame_ts`/`last_frame_ts` are the
        # raw evidence. An estimate that excludes its own evidence is the wrong way round.
        for text in document.ocr:
            if text.first_frame_ts is not None:
                assert text.start <= text.first_frame_ts + TOLERANCE_S
            if text.last_frame_ts is not None:
                assert text.last_frame_ts <= text.end + TOLERANCE_S


class TestSerialization:
    def test_the_document_round_trips(self, document, tmp_path):
        path = tmp_path / "demo.vctx"
        path.write_text(document.model_dump_json())
        reloaded = VideoContextDocument.model_validate_json(path.read_text())
        # model_dump rather than == so a field added without an equality update still counts.
        assert reloaded.model_dump() == document.model_dump()

    def test_no_frame_path_is_absolute(self, document):
        # Spec §13. The check is on the string: an absolute POSIX path also has to fail here on
        # a machine whose own separator is different.
        for frame in document.frames:
            assert frame.path is None or not frame.path.startswith("/")

    def test_the_segment_text_is_reachable_evidence(self, document):
        # `text` is a projection, never a source: anything in it must also be citable. A segment
        # with text but no ids cannot support the answer it would produce.
        for segment in document.segments:
            if segment.text:
                assert segment.transcript_ids or segment.ocr_ids or segment.vision_ids
