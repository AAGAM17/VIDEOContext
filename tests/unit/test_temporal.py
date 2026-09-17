"""Temporal primitives: relations, windows, changes, chapters, UI states, query parsing.

All views are pure functions over hand-built documents — no media, no models. The
invariant under test throughout: every timestamp returned is copied from a fact
the test itself placed in the document.
"""

from __future__ import annotations

import pytest

from videocontent.schema.v1 import (
    OCRText,
    Scene,
    Utterance,
    VideoContextDocument,
    VideoInfo,
)
from videocontent.temporal import (
    TemporalRelation,
    build_chapters,
    detect_changes,
    merge_windows,
    parse_temporal_query,
    relate,
    ui_states,
    window_around,
    window_at,
)


def utt(i, text, s, e, **kw):
    return Utterance(id=f"utt_{i:04d}", text=text, start=s, end=e, **kw)


def ocr(i, text, s, e, **kw):
    return OCRText(id=f"ocr_{i:04d}", text=text, start=s, end=e,
                   first_frame_ts=s, last_frame_ts=e, **kw)


def scene(i, s, e, **kw):
    return Scene(id=f"scene_{i:04d}", start=s, end=e, **kw)


def doc(duration=120.0, **kw):
    return VideoContextDocument(
        id="vid_t",
        video=VideoInfo(id="vid_t", filename="t.mp4", duration=duration, has_audio=True),
        scenes=list(kw.get("scenes", ())),
        transcript=list(kw.get("transcript", ())),
        ocr=list(kw.get("ocr", ())),
        events=list(kw.get("events", ())),
        segments=list(kw.get("segments", ())),
    )


class TestRelate:
    def test_disjoint_order(self):
        rel = relate(0.0, 5.0, 10.0, 15.0)
        assert TemporalRelation.BEFORE in rel
        assert TemporalRelation.PRECEDES in rel
        assert TemporalRelation.OVERLAPS not in rel
        assert TemporalRelation.AFTER not in rel

    def test_reverse_order(self):
        rel = relate(10.0, 15.0, 0.0, 5.0)
        assert TemporalRelation.AFTER in rel
        assert TemporalRelation.FOLLOWS in rel

    def test_overlap_and_containment(self):
        assert TemporalRelation.OVERLAPS in relate(0.0, 10.0, 5.0, 15.0)
        assert TemporalRelation.DURING in relate(5.0, 7.0, 0.0, 10.0)
        assert TemporalRelation.CONTAINS in relate(0.0, 10.0, 5.0, 7.0)

    def test_boundary_tolerance(self):
        # Tolerance softens boundary relations (DURING/BEFORE/CO_OCCURS) while
        # strict ordering (OVERLAPS/PRECEDES/FOLLOWS) stays exact.
        assert TemporalRelation.DURING in relate(9.5, 19.0, 10.0, 20.0)
        assert TemporalRelation.DURING not in relate(9.5, 19.0, 10.0, 20.0, tolerance=0.0)
        assert TemporalRelation.CO_OCCURS in relate(0.0, 9.5, 10.0, 20.0)
        assert TemporalRelation.OVERLAPS not in relate(0.0, 9.5, 10.0, 20.0)
        assert TemporalRelation.PRECEDES in relate(0.0, 9.5, 10.0, 20.0)
        assert TemporalRelation.PRECEDES in relate(0.0, 10.0, 10.0, 20.0)

    def test_near_and_starts_within(self):
        assert TemporalRelation.NEAR in relate(0.0, 5.0, 8.0, 12.0)
        assert TemporalRelation.NEAR not in relate(0.0, 5.0, 100.0, 105.0)
        assert TemporalRelation.STARTS_WITHIN in relate(12.0, 20.0, 10.0, 15.0)
        assert TemporalRelation.ENDS_WITHIN in relate(5.0, 12.0, 10.0, 15.0)

    def test_instants(self):
        rel = relate(10.0, 10.0, 10.0, 10.0)
        assert TemporalRelation.DURING in rel and TemporalRelation.CONTAINS in rel


class TestWindows:
    def test_clamped(self):
        w = window_at(3.0, 5.0, 120.0)
        assert (w.start, w.end) == (0.0, 8.0)
        w = window_around(115.0, 120.0, 10.0, 120.0)
        assert (w.start, w.end) == (105.0, 120.0)

    def test_merge(self):
        merged = merge_windows([window_at(10.0, 5.0, 120.0), window_at(12.0, 5.0, 120.0),
                                window_at(100.0, 2.0, 120.0)])
        assert len(merged) == 2
        assert merged[0].start == 5.0 and merged[0].end == 17.0


class TestChanges:
    def test_text_turnover(self):
        d = doc(scenes=[scene(0, 0.0, 60.0), scene(1, 60.0, 120.0)],
                ocr=[ocr(0, "Pricing plans overview", 10.0, 50.0, stable=True, frame_count=4),
                     ocr(1, "Checkout payment form", 70.0, 110.0, stable=True, frame_count=4)],
                transcript=[utt(0, "here is pricing", 10.0, 20.0),
                            utt(1, "now checkout", 70.0, 80.0)])
        changes = detect_changes(d)
        assert changes, "disjoint OCR across a scene boundary must be reported"
        assert changes[0].change_type == "text_changed"
        assert changes[0].ts == 60.0
        known = {o.id for o in d.ocr} | {u.id for u in d.transcript} | \
            {s.id for s in d.scenes}
        for change in changes:
            assert set(change.evidence_ids) <= known

    def test_speech_transition(self):
        d = doc(scenes=[scene(0, 0.0, 60.0), scene(1, 60.0, 120.0)],
                transcript=[utt(0, "hello there", 70.0, 80.0)])
        changes = detect_changes(d)
        assert [c.change_type for c in changes] == ["speech_started"]

    def test_empty_document(self):
        assert detect_changes(doc(duration=60.0)) == []
        assert detect_changes(doc(duration=0.0)) == []

    def test_timestamps_come_from_facts(self):
        d = doc(scenes=[scene(0, 0.0, 30.0), scene(1, 30.0, 60.0)],
                ocr=[ocr(0, "alpha beta", 5.0, 25.0), ocr(1, "gamma delta", 35.0, 55.0)])
        for change in detect_changes(d):
            assert change.ts in (30.0,)


class TestChapters:
    def test_tiling_without_scenes(self):
        d = doc(duration=600.0, transcript=[utt(0, "revenue revenue pricing", 10.0, 20.0)])
        chapters = build_chapters(d, target_s=300.0)
        assert len(chapters) == 2
        assert chapters[0].start == 0.0 and chapters[-1].end == 600.0
        assert all(c.inferred for c in chapters)

    def test_keyword_titles(self):
        d = doc(duration=600.0,
                transcript=[utt(0, "revenue revenue revenue quarter", 10.0, 20.0),
                            utt(1, "testing testing testing suite", 400.0, 410.0)])
        chapters = build_chapters(d, target_s=300.0)
        assert "Revenue" in chapters[0].title
        assert "Testing" in chapters[1].title

    def test_titles_rank_by_frequency(self):
        d = doc(duration=300.0,
                transcript=[utt(0, "alpha beta alpha alpha gamma beta alpha", 10.0, 20.0)])
        chapters = build_chapters(d, target_s=300.0)
        assert len(chapters) == 1
        first = chapters[0].title.split(" · ")[0]
        assert first == "Alpha", f"most frequent term must lead, got {chapters[0].title!r}"

    def test_deterministic(self):
        d = doc(duration=600.0, transcript=[utt(0, "alpha beta gamma", 10.0, 20.0)])
        assert [c.to_dict() for c in build_chapters(d)] == \
               [c.to_dict() for c in build_chapters(d)]

    def test_empty(self):
        assert build_chapters(doc(duration=0.0)) == []


class TestUIStates:
    def test_stable_runs_define_states(self):
        d = doc(ocr=[ocr(0, "Homepage nav", 0.0, 30.0, stable=True, frame_count=5),
                     ocr(1, "Pricing grid", 30.0, 60.0, stable=True, frame_count=5)])
        states = ui_states(d)
        assert len(states) == 2
        assert states[0].elements == ("Homepage nav",)
        assert states[1].elements == ("Pricing grid",)
        assert states[0].evidence_ids == ("ocr_0000",)

    def test_no_ocr_no_states(self):
        assert ui_states(doc(transcript=[utt(0, "hi", 1.0, 2.0)])) == []


class TestParseTemporalQuery:
    def test_plain_passthrough(self):
        q = parse_temporal_query("pricing page")
        assert not q.is_temporal and q.anchor == "pricing page"

    def test_before_after(self):
        q = parse_temporal_query("what happened before the error")
        assert q.relation is TemporalRelation.BEFORE and q.anchor == "the error"
        q = parse_temporal_query("what happened after the login")
        assert q.relation is TemporalRelation.AFTER and q.anchor == "the login"

    def test_between(self):
        q = parse_temporal_query("what changed between 10:20 and 11:00")
        assert (q.start, q.end) == (620.0, 660.0)

    def test_first(self):
        q = parse_temporal_query("when did Stripe first appear")
        assert q.first_only and "Stripe" in q.anchor and "first" not in q.anchor.lower()

    def test_bare_timecode(self):
        q = parse_temporal_query("pricing at 12:20")
        assert q.start == pytest.approx(710.0) and "pricing" in q.anchor
