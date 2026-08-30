"""Segmentation: does the fusion keep every fact, and put it in the right window?

Segments are the unit retrieval scores, so two failures here are invisible in every other test
and fatal to the product:

* **a dropped fact** — an utterance or OCR event that lands in no segment can never be
  returned by search, no matter how good the ranking is. The document still contains it, which
  makes the loss look like a retrieval bug rather than a fusion bug.
* **a fact in the wrong window** — a segment that claims text it did not overlap produces a
  confident answer with a timestamp that is wrong, which is precisely what the evidence-first
  contract exists to prevent.

So the assertions below are mostly about *coverage* and *boundaries*, not about counts. The
windows are checked as a tiling (no gaps, no overlaps, ends at the duration), and every fact in
the document is checked to appear in at least one segment's reference lists.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from videocontext.config import SegmentConfig
from videocontext.processing.segments import (
    build_segments,
    plan_windows,
    project_text,
)
from videocontext.schema.v1 import (
    Event,
    Frame,
    OCRText,
    Scene,
    Utterance,
    VideoContextDocument,
    VideoInfo,
    VisionNote,
)

DURATION = 60.0


def scene(index: int, start: float, end: float) -> Scene:
    return Scene(id=f"scene_{index:04d}", start=start, end=end, detector="test")


def utt(index: int, text: str, start: float, end: float, language: str | None = "en") -> Utterance:
    return Utterance(id=f"utt_{index:04d}", text=text, start=start, end=end, language=language)


def ocr(index: int, text: str, start: float, end: float, language: str | None = "en") -> OCRText:
    return OCRText(
        id=f"ocr_{index:04d}",
        text=text,
        start=start,
        end=end,
        language=language,
        frame_count=3,
        stable=True,
    )


def event(index: int, ts: float, type_: str = "scene_changed") -> Event:
    # Zero-length on purpose: instants are how the rule detector represents a boundary, and
    # they are the case a naive overlap test silently drops.
    return Event(id=f"evt_{index:04d}", type=type_, start=ts, end=ts, refs={"scenes": ["x"]})


def frame(index: int, ts: float) -> Frame:
    return Frame(id=f"frame_{index:04d}", ts=ts, index=index)


def doc(*, duration: float = DURATION, **kw) -> VideoContextDocument:
    return VideoContextDocument(
        id="vid_test",
        video=VideoInfo(id="vid_test", filename="demo.mp4", duration=duration, has_audio=True),
        scenes=list(kw.get("scenes", ())),
        transcript=list(kw.get("transcript", ())),
        ocr=list(kw.get("ocr", ())),
        vision=list(kw.get("vision", ())),
        events=list(kw.get("events", ())),
        frames=list(kw.get("frames", ())),
    )


def tiles(windows: list[tuple[float, float]], duration: float) -> bool:
    """Contiguous, ordered, covering exactly ``[0, duration]``."""
    if not windows:
        return False
    if abs(windows[0][0]) > 1e-6 or abs(windows[-1][1] - duration) > 1e-6:
        return False
    return all(abs(b[0] - a[1]) < 1e-6 for a, b in pairwise(windows))


class TestWindows:
    def test_windows_follow_scenes(self):
        scenes = [scene(0, 0, 20), scene(1, 20, 40), scene(2, 40, 60)]
        assert plan_windows(doc(scenes=scenes)) == [(0, 20), (20, 40), (40, 60)]

    def test_no_scenes_gives_the_whole_video_as_the_base_window(self):
        # Subdivision still applies on top: the default max_duration is 45 s, so a 60 s video
        # with no scene information is two windows, not one. Both facts are asserted here
        # because pinning a literal list would silently encode that unrelated default.
        config = SegmentConfig(max_duration=DURATION)
        assert plan_windows(doc(), config) == [(0.0, DURATION)]
        assert tiles(plan_windows(doc()), DURATION)

    def test_a_zero_length_video_gives_no_windows(self):
        # Nothing to segment, and a [0, 0) segment would be a span that cannot contain a fact.
        assert plan_windows(doc(duration=0.0)) == []

    def test_windows_tile_the_timeline(self):
        scenes = [scene(0, 0, 12), scene(1, 12, 33), scene(2, 33, 60)]
        assert tiles(plan_windows(doc(scenes=scenes)), DURATION)

    def test_a_gap_between_scenes_is_closed(self):
        # The detector is not required to tile; a fact in the gap must still land somewhere.
        windows = plan_windows(doc(scenes=[scene(0, 0, 10), scene(1, 20, 60)]))
        assert tiles(windows, DURATION)
        assert (10.0, 20.0) in windows

    def test_a_scene_past_the_duration_still_tiles(self):
        windows = plan_windows(doc(scenes=[scene(0, 0, 20)], duration=20.0))
        assert tiles(windows, 20.0)

    def test_ignoring_scenes_drops_the_scene_boundaries(self):
        scenes = [scene(0, 0, 20), scene(1, 20, 60)]
        config = SegmentConfig(align_to_scenes=False, split_on_utterance_boundary=False)
        windows = plan_windows(doc(scenes=scenes), config)
        boundaries = {start for start, _ in windows}
        assert 20.0 not in boundaries
        assert tiles(windows, DURATION)


class TestSubdivision:
    def test_a_long_scene_is_split(self):
        config = SegmentConfig(max_duration=20.0, split_on_utterance_boundary=False)
        windows = plan_windows(doc(scenes=[scene(0, 0, 60)]), config)
        assert windows == [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]

    def test_no_piece_exceeds_the_maximum(self):
        config = SegmentConfig(max_duration=7.0, split_on_utterance_boundary=False)
        windows = plan_windows(doc(scenes=[scene(0, 0, 60)]), config)
        assert max(end - start for start, end in windows) <= 7.0 + 1e-6

    def test_subdivision_still_tiles(self):
        config = SegmentConfig(max_duration=9.0, split_on_utterance_boundary=False)
        assert tiles(plan_windows(doc(scenes=[scene(0, 0, 60)]), config), DURATION)

    def test_a_cut_moves_into_a_nearby_silence(self):
        # The even cut is 30.0; the gap between utterances is 29.0-31.0, midpoint 30.0... so
        # place the gap off-centre to prove the snap actually moved the boundary.
        transcript = [utt(0, "first half", 0, 27.5), utt(1, "second half", 32.0, 60)]
        config = SegmentConfig(max_duration=30.0)
        windows = plan_windows(doc(scenes=[scene(0, 0, 60)], transcript=transcript), config)
        assert windows[0][1] == pytest.approx(29.75)  # midpoint of 27.5-32.0

    def test_a_distant_silence_does_not_drag_the_boundary(self):
        # The only gap is 20 s from the cut. Snapping to it would distort the window far more
        # than clipping one sentence does, so the even cut stands.
        transcript = [utt(0, "a", 0, 9.0), utt(1, "b", 12.0, 60)]
        config = SegmentConfig(max_duration=30.0)
        windows = plan_windows(doc(scenes=[scene(0, 0, 60)], transcript=transcript), config)
        assert windows[0][1] == pytest.approx(30.0)

    def test_snapping_never_breaks_the_tiling(self):
        transcript = [utt(i, f"line {i}", i * 6.0, i * 6.0 + 4.0) for i in range(10)]
        config = SegmentConfig(max_duration=11.0)
        document = doc(scenes=[scene(0, 0, 60)], transcript=transcript)
        assert tiles(plan_windows(document, config), 60.0)


class TestShortWindows:
    def test_a_short_scene_is_absorbed(self):
        # A 0.4 s cut is not a retrievable unit; it belongs to the shot it interrupts.
        scenes = [scene(0, 0, 30), scene(1, 30, 30.4), scene(2, 30.4, 60)]
        windows = plan_windows(doc(scenes=scenes))
        assert len(windows) == 2
        assert tiles(windows, DURATION)

    def test_absorbing_may_exceed_the_maximum(self):
        # Documented trade: a slightly over-long window retrieves coarsely, a runt pollutes
        # every ranking it appears in.
        scenes = [scene(0, 0, 44), scene(1, 44, 45.5), scene(2, 45.5, 60)]
        config = SegmentConfig(max_duration=45.0, split_on_utterance_boundary=False)
        windows = plan_windows(doc(scenes=scenes), config)
        assert tiles(windows, DURATION)

    def test_a_trailing_runt_folds_back(self):
        # The 2.1 s tail scene must not survive as its own window, whatever subdivision did to
        # the long scene before it.
        scenes = [scene(0, 0, 57.9), scene(1, 57.9, 60)]
        windows = plan_windows(doc(scenes=scenes))
        assert 57.9 not in {start for start, _ in windows}
        assert min(end - start for start, end in windows) >= SegmentConfig().min_duration
        assert tiles(windows, DURATION)

    def test_a_video_shorter_than_the_minimum_is_still_one_segment(self):
        # Nothing to merge into. A 2-second clip must not produce zero segments.
        windows = plan_windows(doc(duration=2.0))
        assert windows == [(0.0, 2.0)]


class TestReferences:
    def build(self):
        return doc(
            scenes=[scene(0, 0, 20), scene(1, 20, 40), scene(2, 40, 60)],
            transcript=[utt(0, "hello there", 1, 5), utt(1, "pricing", 25, 30)],
            ocr=[ocr(0, "Slide one", 2, 18), ocr(1, "Pricing", 22, 38)],
            vision=[VisionNote(id="vis_0000", start=40, end=60, description="a chart")],
            events=[event(0, 20.0), event(1, 40.0), event(2, 60.0)],
            frames=[frame(i, float(i) * 5) for i in range(13)],
        )

    def test_a_segment_references_only_what_it_overlaps(self):
        segments = build_segments(self.build())
        first = segments[0]
        assert first.transcript_ids == ["utt_0000"]
        assert first.ocr_ids == ["ocr_0000"]
        assert first.vision_ids == []

    def test_every_fact_lands_in_a_segment(self):
        document = self.build()
        segments = build_segments(document)
        for field, items in (
            ("scene_ids", document.scenes),
            ("transcript_ids", document.transcript),
            ("ocr_ids", document.ocr),
            ("vision_ids", document.vision),
            ("event_ids", document.events),
            ("frame_ids", document.frames),
        ):
            referenced = {ref for s in segments for ref in getattr(s, field)}
            assert {i.id for i in items} <= referenced, f"{field} lost a fact"

    def test_an_instant_event_is_not_dropped(self):
        # The regression this guards: `start < w_end and end > w_start` is false for every
        # window when start == end, so a purely-instant event set would vanish entirely.
        document = doc(scenes=[scene(0, 0, 30), scene(1, 30, 60)], events=[event(0, 30.0)])
        segments = build_segments(document)
        assert [s.event_ids for s in segments].count(["evt_0000"]) == 1

    def test_a_fact_at_the_final_boundary_is_kept(self):
        document = doc(events=[event(0, DURATION)], frames=[frame(0, DURATION)])
        segments = build_segments(document)
        assert segments[-1].event_ids == ["evt_0000"]
        assert segments[-1].frame_ids == ["frame_0000"]

    def test_a_boundary_fact_belongs_to_one_segment_only(self):
        # Half-open spans: text ending exactly at 20.0 is in the first window, not both.
        document = doc(
            scenes=[scene(0, 0, 20), scene(1, 20, 60)],
            ocr=[ocr(0, "title", 5, 20)],
        )
        segments = build_segments(document)
        assert [s.ocr_ids for s in segments] == [["ocr_0000"], []]

    def test_text_spanning_two_segments_is_referenced_by_both(self):
        # Not a duplicate: the text really was on screen during both windows.
        document = doc(
            scenes=[scene(0, 0, 20), scene(1, 20, 60)],
            ocr=[ocr(0, "persistent footer", 5, 55)],
        )
        segments = build_segments(document)
        assert all(s.ocr_ids == ["ocr_0000"] for s in segments)

    def test_ids_are_sequential_and_ordered(self):
        segments = build_segments(self.build())
        assert [s.id for s in segments] == [f"segment_{i:04d}" for i in range(len(segments))]
        assert all(a.start <= b.start for a, b in pairwise(segments))

    def test_it_does_not_mutate_the_document(self):
        document = self.build()
        before = document.model_dump()
        build_segments(document)
        assert document.model_dump() == before

    def test_it_is_deterministic(self):
        document = self.build()
        assert [s.model_dump() for s in build_segments(document)] == [
            s.model_dump() for s in build_segments(document)
        ]


class TestProjection:
    def test_text_is_in_timeline_order(self):
        text = project_text(
            [utt(0, "spoken second", 10, 12)],
            [ocr(0, "shown first", 1, 5)],
            [],
        )
        assert text.splitlines() == ["shown first", "spoken second"]

    def test_a_repeat_across_modalities_appears_once(self):
        # Otherwise a lexical scorer rewards a segment twice for one phrase.
        text = project_text([utt(0, "Pricing", 1, 2)], [ocr(0, "pricing", 1, 9)], [])
        assert text.splitlines() == ["Pricing"]

    def test_vision_descriptions_are_included(self):
        text = project_text(
            [], [], [VisionNote(id="vis_0000", start=0, end=5, description="a bar chart")]
        )
        assert text == "a bar chart"

    def test_whitespace_is_collapsed(self):
        # OCR line joins and SRT cues both introduce newlines mid-phrase; a phrase query has to
        # survive them.
        assert project_text([], [ocr(0, "log  in\nnow", 1, 5)], []) == "log in now"

    def test_empty_text_is_not_a_blank_line(self):
        assert project_text([utt(0, "   ", 1, 2)], [ocr(0, "real", 3, 4)], []) == "real"

    def test_no_modality_prefixes(self):
        # A "[ocr]" marker would break any phrase query spanning the join.
        text = project_text([utt(0, "hello", 1, 2)], [ocr(0, "world", 1, 2)], [])
        assert "[" not in text and "]" not in text

    def test_the_projection_only_contains_referenced_facts(self):
        document = doc(
            scenes=[scene(0, 0, 30), scene(1, 30, 60)],
            transcript=[utt(0, "first window only", 1, 5)],
        )
        segments = build_segments(document)
        assert segments[0].text == "first window only"
        assert segments[1].text == ""


class TestLanguages:
    def test_languages_are_the_union_of_the_facts(self):
        document = doc(
            transcript=[utt(0, "hola", 1, 5, language="es")],
            ocr=[ocr(0, "Sign in", 1, 9, language="en")],
        )
        assert build_segments(document)[0].languages == ["en", "es"]

    def test_unknown_languages_are_omitted_not_nulled(self):
        document = doc(ocr=[ocr(0, "text", 1, 9, language=None)])
        assert build_segments(document)[0].languages == []


class TestDeferredFields:
    """``summary`` and ``keywords`` are empty by decision, not by omission."""

    def test_summary_is_absent_until_a_model_writes_one(self):
        assert build_segments(doc(ocr=[ocr(0, "x", 1, 5)]))[0].summary is None

    def test_keywords_are_empty(self):
        assert build_segments(doc(ocr=[ocr(0, "x", 1, 5)]))[0].keywords == []
