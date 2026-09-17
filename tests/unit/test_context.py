"""Context budgets, expansion, frame bias, and the SDK intelligence surface."""

from __future__ import annotations

from videocontent.retrieval import EvidenceSpan
from videocontent.routing import (
    ContextBudget,
    classify_task,
    dedupe_spans,
    enforce_budgets,
    expand_evidence,
    select_context,
    select_representative_frames,
)
from videocontent.schema.v1 import Frame, OCRText, Utterance, VideoContextDocument, VideoInfo
from videocontent.sdk import Video


def span(modality, ref, s, e, text):
    return EvidenceSpan(start=s, end=e, modality=modality, text=text, score=1.0,
                        ref_ids=(ref,))


def doc():
    return VideoContextDocument(
        id="vid_c",
        video=VideoInfo(id="vid_c", filename="c.mp4", duration=200.0, has_audio=True),
        transcript=[Utterance(id="utt_0000", text="the checkout failed", start=50.0, end=56.0),
                    Utterance(id="utt_0001", text="Error on screen again", start=58.0, end=62.0),
                    Utterance(id="utt_0002", text="welcome back", start=5.0, end=9.0)],
        ocr=[OCRText(id="ocr_0000", text="Error Payment declined", start=50.0, end=70.0,
                     first_frame_ts=50.0, last_frame_ts=70.0, stable=True, frame_count=4)],
        frames=[Frame(id=f"frame_{i:04d}", ts=float(i * 20), reason="fixed")
                for i in range(10)],
    )


class TestBudgets:
    def test_max_spans(self):
        sel = {"evidence": [span("ocr", f"r{i}", 1.0 * i, 2.0 * i, "t") for i in range(8)],
               "frames": [], "budget_notes": []}
        out = enforce_budgets(sel, ContextBudget(max_spans=3))
        assert len(out["evidence"]) == 3
        assert any("max_spans" in n for n in out["budget_notes"])

    def test_max_seconds(self):
        sel = {"evidence": [span("ocr", "a", 0.0, 10.0, "t"),
                            span("ocr", "b", 60.0, 120.0, "t"),
                            span("ocr", "c", 130.0, 140.0, "t")],
               "frames": [], "budget_notes": []}
        out = enforce_budgets(sel, ContextBudget(max_seconds=30.0))
        # Rank order wins: keep leading spans until the budget is exhausted.
        assert [s.ref_ids for s in out["evidence"]] == [("a",), ("c",)]
        assert any("max_seconds" in n for n in out["budget_notes"])

    def test_max_frames(self):
        sel = {"evidence": [], "frames": [{"id": f"f{i}"} for i in range(6)],
               "budget_notes": []}
        out = enforce_budgets(sel, ContextBudget(max_frames=2))
        assert len(out["frames"]) == 2

    def test_none_is_unbounded(self):
        sel = {"evidence": [span("ocr", "a", 0.0, 5.0, "t")], "frames": [{"id": "f"}],
               "budget_notes": []}
        out = enforce_budgets(sel, ContextBudget())
        assert len(out["evidence"]) == 1 and len(out["frames"]) == 1
        assert out["budget_notes"] == []


class TestExpansion:
    def test_cooccurring_facts_pulled_in(self):
        d = doc()
        base = [span("transcript", "utt_0000", 50.0, 56.0, "the checkout failed")]
        extra = expand_evidence(d, base, 5.0)
        assert any(s.ref_ids == ("ocr_0000",) for s in extra)
        assert all("co-occurs" in s.reason for s in extra)

    def test_dedupe(self):
        a = span("ocr", "r", 1.0, 2.0, "t")
        assert dedupe_spans([a, a, span("ocr", "s", 1.0, 2.0, "t")]).__len__() == 2


class TestSelectContext:
    def test_budgets_flow_through(self):
        d = doc()
        task = classify_task("What was the error?")
        sel = select_context(d, task, ContextBudget(max_tokens=4000, max_spans=0),
                             query="error")
        assert sel["evidence"] == []
        assert any("max_spans" in n for n in sel["budget_notes"])

    def test_expansion_flows_through(self):
        d = doc()
        task = classify_task("What was the error?")
        plain = select_context(d, task, ContextBudget(), query="checkout")
        expanded = select_context(d, task, ContextBudget(), query="checkout", expand_s=10.0)
        assert len(expanded["evidence"]) >= len(plain["evidence"])
        assert any("expansion" in n for n in expanded["budget_notes"])

    def test_modalities_respected(self):
        d = doc()
        task = classify_task("What was the error?")
        sel = select_context(d, task, ContextBudget(), query="checkout",
                             modalities=["transcript"])
        assert sel["evidence"]
        assert all(s.modality == "transcript" for s in sel["evidence"])


class TestFrameBias:
    def test_query_bias_prefers_matching_frames(self):
        d = doc()
        plain = select_representative_frames(d, max_frames=10)
        biased = select_representative_frames(d, max_frames=10, query="Payment declined")
        assert biased[0]["ts"] == 60.0  # frame overlapping the matching OCR
        assert [f["id"] for f in plain] == [f["id"] for f in
            select_representative_frames(d, max_frames=10, query=None)]


class TestSDKIntel:
    def test_timeline_chapters_entities_changes(self):
        video = Video.from_document(doc())
        assert len(video.timeline(0.0, 200.0).spans) > 0
        assert len(video.timeline(150.0, 160.0).spans) == 0
        assert video.chapters()  # tiling always yields chapters
        assert all(c.inferred for c in video.chapters())
        names = [e.name for e in video.entities()]
        assert any("error" in n.lower() or "payment" in n.lower() for n in names)
        assert isinstance(video.changes(), list)

    def test_receipt(self):
        receipt = Video.from_document(doc()).receipt()
        assert receipt["video_id"] == "vid_c"
        assert receipt["modalities"]["ocr"] == 1
        assert "producer" in receipt and "stages" in receipt

    def test_plan(self):
        report = Video.from_document(doc()).plan("What was said?")
        names = [s["name"] for s in report["plan"]["stages"]]
        assert "asr" in names or "audio" in names
        assert "coverage" in report

    def test_ask_trace(self):
        video = Video.from_document(doc())
        answer = video.ask("What failed?")
        assert answer.trace["executor"] == "search"
        assert answer.trace["retrieval"]["spans"] >= 0
        temporal = video.ask("what happened before the checkout failed")
        assert temporal.trace["executor"] == "temporal"
        assert "query_plan" in temporal.trace

    def test_ask_no_evidence(self):
        video = Video.from_document(doc())
        answer = video.ask("quantum zebras teleporting")
        assert answer.confidence == 0.0
        assert answer.trace["outcome"] == "insufficient_evidence"

    def test_context_budgets(self):
        video = Video.from_document(doc())
        ctx = video.context("What was the error?", max_spans=1)
        assert len(ctx.evidence) <= 1
        assert isinstance(ctx.budget_notes, list)
