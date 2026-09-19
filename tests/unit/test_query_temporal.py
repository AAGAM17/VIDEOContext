"""Temporal queries and timelines return ordered, explained, grounded evidence."""

from __future__ import annotations

from videocontent.retrieval import Retriever, query_temporal, search, timeline
from videocontent.schema.v1 import (
    Event,
    OCRText,
    Segment,
    Utterance,
    VideoContextDocument,
    VideoInfo,
)


def utt(i, text, s, e, **kw):
    return Utterance(id=f"utt_{i:04d}", text=text, start=s, end=e, **kw)


def ocr(i, text, s, e, **kw):
    return OCRText(id=f"ocr_{i:04d}", text=text, start=s, end=e, **kw)


def event(i, type_, s, e, **kw):
    return Event(id=f"evt_{i:04d}", type=type_, start=s, end=e,
                 refs={"ocr": ["ocr_0000"]}, **kw)


def doc(**kw):
    facts = {
        "transcript": list(kw.get("transcript", ())),
        "ocr": list(kw.get("ocr", ())),
        "events": list(kw.get("events", ())),
    }
    segments = [Segment(id=f"segment_{i:04d}", start=float(i * 60), end=float((i + 1) * 60),
                        transcript_ids=[f.id for f in facts["transcript"]
                                        if i * 60 <= f.start < (i + 1) * 60],
                        ocr_ids=[f.id for f in facts["ocr"]
                                 if i * 60 <= f.start < (i + 1) * 60])
                for i in range(3)]
    return VideoContextDocument(
        id="vid_q",
        video=VideoInfo(id="vid_q", filename="q.mp4", duration=180.0, has_audio=True),
        segments=segments, **facts)


def sample():
    return doc(
        transcript=[
            utt(0, "welcome to the demo", 5.0, 10.0),
            utt(1, "the checkout failed with an error", 100.0, 108.0),
            utt(2, "let me fix the checkout error now", 120.0, 128.0),
        ],
        ocr=[
            ocr(0, "Welcome Dashboard", 4.0, 30.0),
            ocr(1, "Error Payment declined", 100.0, 130.0),
        ],
        events=[event(0, "error_shown", 100.0, 108.0,
                       description="Error Payment declined")],
    )


class TestTimeline:
    def test_range_contents(self):
        result = timeline(sample(), 0.0, 60.0)
        assert result.spans
        assert all(s.start < 60.0 and s.end > 0.0 for s in result.spans)
        assert all(s.reason for s in result.spans)

    def test_bounded(self):
        result = timeline(sample(), 0.0, 180.0, top_k=2)
        assert len(result.spans) == 2
        assert result.total > 2
        assert result.notes, "truncation must be disclosed"

    def test_empty_range(self):
        assert len(timeline(sample(), 150.0, 160.0).spans) == 0

    def test_inverted_range_says_so(self):
        result = timeline(sample(), 60.0, 10.0)
        assert len(result.spans) == 0
        assert any("inverted" in note for note in result.notes)


class TestTemporalQueries:
    def test_plain_query_unchanged(self):
        plan, result = query_temporal(sample(), "checkout")
        assert not plan.is_temporal
        assert [s.text for s in result.spans] == \
               [s.text for s in search(sample(), "checkout").spans]

    def test_before(self):
        plan, result = query_temporal(sample(), "what happened before the error")
        assert plan.is_temporal
        assert result.spans, "the welcome precedes the error and must be found"
        assert all(s.end <= 100.0 + 1e-6 for s in result.spans)
        assert all("before" in s.reason for s in result.spans)
        # Most recent first.
        assert result.spans[0].end >= result.spans[-1].end

    def test_after(self):
        _plan, result = query_temporal(sample(), "what happened after the welcome")
        assert result.spans
        assert all(s.start >= 10.0 - 1e-6 for s in result.spans)
        assert all("after" in s.reason for s in result.spans)

    def test_missing_anchor_says_so(self):
        _plan, result = query_temporal(sample(), "what happened before the zebra")
        assert len(result.spans) == 0
        assert any("zebra" in note for note in result.notes)

    def test_first(self):
        plan, result = query_temporal(sample(), "when did checkout first appear")
        assert plan.first_only
        assert result.spans
        assert result.spans[0].start == min(s.start for s in result.spans)
        assert all("first occurrence" in s.reason for s in result.spans)

    def test_range_filters(self):
        plan, result = query_temporal(sample(), "error between 01:30 and 02:00")
        assert (plan.start, plan.end) == (90.0, 120.0)
        assert result.spans
        # Touching the boundary counts (same overlap rule as search itself).
        assert all(s.start <= 120.0 and s.end >= 90.0 for s in result.spans)
        assert all("within" in s.reason for s in result.spans)

    def test_range_fallback_to_timeline(self):
        _plan, result = query_temporal(sample(), "between 00:00 and 00:03")
        assert result.spans or "timeline" in " ".join(result.notes)

    def test_timestamps_are_copied(self):
        facts = {}
        d = sample()
        for group in (d.transcript, d.ocr, d.events):
            for item in group:
                facts[item.id] = item
        _, result = query_temporal(d, "what happened before the error")
        for span in result.spans:
            assert span.ref_ids, "temporal spans must cite their facts"
            for ref in span.ref_ids:
                assert ref in facts
                assert facts[ref].start <= span.end and span.start <= facts[ref].end


class TestRetrieverMethod:
    def test_method_matches_function(self):
        d = sample()
        via_method = Retriever(d).timeline(0.0, 60.0)
        via_fn = timeline(d, 0.0, 60.0)
        assert [s.ref_ids for s in via_method] == [s.ref_ids for s in via_fn]
