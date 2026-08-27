"""Temporal OCR deduplication: the per-frame -> lifespan collapse.

The behaviours asserted here are the ones the format's timestamps depend on. A span that ends
in the wrong place is not a cosmetic flaw — it is an unsupported claim about when text was on
screen, which is the one thing the evidence-first contract forbids.
"""

from __future__ import annotations

from videocontent.config import OCRConfig
from videocontent.interfaces import OCRObservation
from videocontent.processing.ocr import deduplicate, iou, normalize, similarity


def obs(text: str, ts: float, bbox=(100.0, 200.0, 300.0, 240.0), conf=0.9, **kw):
    return OCRObservation(
        text=text, ts=ts, confidence=conf, bbox=bbox,
        frame_width=kw.pop("frame_width", 1280), frame_height=kw.pop("frame_height", 720),
        **kw,
    )


class TestNormalize:
    def test_case_and_whitespace_insensitive(self):
        assert normalize("  Quarterly   BUSINESS\treview ") == "quarterly business review"

    def test_empty(self):
        assert normalize("   ") == ""


class TestSimilarity:
    def test_identical_is_one(self):
        assert similarity("pricing", "pricing") == 1.0

    def test_empty_is_zero(self):
        assert similarity("", "pricing") == 0.0

    def test_ocr_jitter_stays_high(self):
        # One flipped character is the common case this threshold exists to absorb.
        assert similarity("connectionerror: refused", "connectionerror: refusod") > 0.9

    def test_length_gate_short_circuits(self):
        # Less than half the length: cannot be the same text, and must not cost a diff.
        assert similarity("a", "a much longer line of text") == 0.0

    def test_different_text_of_similar_length_is_low(self):
        assert similarity("revenue rs 42l", "competitor pricing") < 0.5


class TestIou:
    def test_unknown_box_does_not_veto(self):
        # No geometry is no evidence against a match; the text gate then decides alone.
        assert iou(None, (0.0, 0.0, 10.0, 10.0)) == 1.0
        assert iou((0.0, 0.0, 10.0, 10.0), None) == 1.0

    def test_identical(self):
        assert iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0

    def test_disjoint(self):
        assert iou((0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)) == 0.0

    def test_partial_overlap(self):
        # Quarter overlap of two equal boxes: 25 / (100 + 100 - 25).
        assert iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) == 25.0 / 175.0


class TestDeduplicate:
    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_repeats_collapse_to_one_event(self):
        events = deduplicate(
            [obs("Quarterly Business Review", t) for t in (0.0, 1.0, 2.0)],
            duration=4.0, frame_ts=[0.0, 1.0, 2.0, 3.0],
        )
        assert len(events) == 1
        event = events[0]
        assert event.text == "Quarterly Business Review"
        assert event.frame_count == 3
        assert event.stable is True
        assert event.start == 0.0
        assert event.first_frame_ts == 0.0
        assert event.last_frame_ts == 2.0

    def test_span_ends_midway_to_the_frame_that_lost_it(self):
        # Seen at 2.0, absent at 3.0 -> it left somewhere between; the midpoint is the
        # minimum-error estimate, and the raw evidence stays in last_frame_ts.
        events = deduplicate(
            [obs("Pricing", t) for t in (0.0, 1.0, 2.0)],
            duration=10.0, frame_ts=[0.0, 1.0, 2.0, 3.0],
        )
        assert events[0].end == 2.5
        assert events[0].last_frame_ts == 2.0

    def test_text_present_in_final_frame_runs_to_the_end(self):
        events = deduplicate(
            [obs("VideoContext demo fixture", t) for t in (0.0, 1.0, 2.0)],
            duration=10.0, frame_ts=[0.0, 1.0, 2.0],
        )
        assert events[0].end == 10.0

    def test_end_never_exceeds_the_span(self):
        events = deduplicate(
            [obs("Agenda", 0.0)], duration=5.0, frame_ts=[0.0, 1.0],
        )
        assert events[0].end <= 5.0

    def test_short_absence_is_bridged(self):
        # A blinking cursor or a bad JPEG must not split one event in two.
        cfg = OCRConfig(max_gap_s=3.0)
        events = deduplicate(
            [obs("$ pytest -q tests/", t) for t in (0.0, 1.0, 3.0)],
            config=cfg, duration=4.0, frame_ts=[0.0, 1.0, 2.0, 3.0],
        )
        assert len(events) == 1
        assert events[0].frame_count == 3

    def test_absence_beyond_max_gap_splits(self):
        cfg = OCRConfig(max_gap_s=1.0)
        events = deduplicate(
            [obs("Sign in", t) for t in (0.0, 5.0)],
            config=cfg, duration=6.0, frame_ts=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert len(events) == 2

    def test_same_text_in_two_places_stays_two_events(self):
        # The box gate is what stops two distinct elements sharing a word from merging.
        events = deduplicate(
            [obs("Pricing", 0.0, bbox=(90.0, 70.0, 300.0, 130.0)),
             obs("Pricing", 1.0, bbox=(900.0, 600.0, 1100.0, 660.0))],
            duration=2.0, frame_ts=[0.0, 1.0],
        )
        assert len(events) == 2

    def test_moving_text_is_matched_by_following_drift(self):
        # A track matches against the most recent position, so text that creeps a little each
        # frame stays one event even though frame 1 and frame 4 no longer overlap.
        boxes = [(100.0 + 30 * i, 200.0, 300.0 + 30 * i, 240.0) for i in range(4)]
        events = deduplicate(
            [obs("collected 12 items", float(i), bbox=b) for i, b in enumerate(boxes)],
            duration=5.0, frame_ts=[0.0, 1.0, 2.0, 3.0],
        )
        assert len(events) == 1
        assert events[0].frame_count == 4

    def test_representative_prefers_the_confident_reading(self):
        # Same line, one frame read it badly: publish the good reading, not the first one.
        events = deduplicate(
            [obs("Revenue Rs 42|", 0.0, conf=0.55),
             obs("Revenue Rs 42L", 1.0, conf=0.97)],
            duration=2.0, frame_ts=[0.0, 1.0],
        )
        assert len(events) == 1, "near-identical text should be one event"
        assert events[0].text == "Revenue Rs 42L"

    def test_representative_breaks_ties_on_length(self):
        # Equal confidence, one reading clipped a character: the complete text is what a
        # user will search for.
        events = deduplicate(
            [obs("Revenue Rs 42", 0.0, conf=0.9),
             obs("Revenue Rs 42L", 1.0, conf=0.9)],
            duration=2.0, frame_ts=[0.0, 1.0],
        )
        assert len(events) == 1
        assert events[0].text == "Revenue Rs 42L"

    def test_badly_truncated_read_does_not_merge(self):
        # Documents a real limit rather than an intention. "Quarterly Business" against
        # "Quarterly Business Review" scores 0.84, under the 0.88 gate, so a frame that drops
        # a whole word yields a second event instead of extending the first. Loosening the
        # gate to catch it would start merging genuinely different lines, so the trade is
        # deliberate — but a caller seeing two overlapping events for one slide is seeing this.
        events = deduplicate(
            [obs("Quarterly Business", 0.0, conf=0.9),
             obs("Quarterly Business Review", 1.0, conf=0.9)],
            duration=2.0, frame_ts=[0.0, 1.0],
        )
        assert len(events) == 2

    def test_confidence_is_averaged_over_observations(self):
        events = deduplicate(
            [obs("Revenue", 0.0, conf=0.8), obs("Revenue", 1.0, conf=1.0)],
            duration=2.0, frame_ts=[0.0, 1.0],
        )
        assert events[0].confidence == 0.9

    def test_single_observation_is_not_stable(self):
        events = deduplicate([obs("TRANSITION", 0.0)], duration=2.0, frame_ts=[0.0, 1.0])
        assert events[0].stable is False
        assert events[0].frame_count == 1

    def test_bbox_normalized_against_frame_size(self):
        events = deduplicate(
            [obs("Revenue Rs 42L", 0.0, bbox=(128.0, 72.0, 640.0, 360.0))],
            duration=1.0, frame_ts=[0.0],
        )
        assert events[0].bbox == [128.0, 72.0, 640.0, 360.0]
        assert events[0].bbox_normalized == [0.1, 0.1, 0.5, 0.5]

    def test_ids_are_assigned_in_timeline_order(self):
        events = deduplicate(
            [obs("second", 5.0, bbox=(0.0, 0.0, 10.0, 10.0)),
             obs("first", 0.0, bbox=(0.0, 0.0, 10.0, 10.0))],
            config=OCRConfig(max_gap_s=1.0), duration=6.0,
            frame_ts=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert [e.id for e in events] == ["ocr_0000", "ocr_0001"]
        assert events[0].text == "first"

    def test_dedupe_disabled_keeps_every_observation(self):
        cfg = OCRConfig(dedupe=False)
        events = deduplicate(
            [obs("Agenda", t) for t in (0.0, 1.0, 2.0)],
            config=cfg, duration=3.0, frame_ts=[0.0, 1.0, 2.0],
        )
        assert len(events) == 3

    def test_engine_is_recorded_as_provenance(self):
        events = deduplicate([obs("Agenda", 0.0)], duration=1.0,
                             frame_ts=[0.0], engine="tesseract")
        assert events[0].engine == "tesseract"

    def test_two_identical_lines_in_one_frame_are_not_merged(self):
        # Value-equal but not the same element: an index-keyed assignment must keep both.
        events = deduplicate(
            [obs("Q1", 0.0, bbox=(100.0, 600.0, 130.0, 630.0)),
             obs("Q1", 0.0, bbox=(400.0, 600.0, 430.0, 630.0))],
            duration=1.0, frame_ts=[0.0],
        )
        assert len(events) == 2
