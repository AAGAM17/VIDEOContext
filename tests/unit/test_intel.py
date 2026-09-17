"""Entities, collections, and processing plans over hand-built documents."""

from __future__ import annotations

from videocontent.collection import CollectionIndex, compare
from videocontent.entities import entity_timeline, extract_entities, normalize_name
from videocontent.plans import check_coverage, plan_for_task
from videocontent.schema.v1 import (
    Event,
    OCRText,
    StageRecord,
    StageStatus,
    Utterance,
    VideoContextDocument,
    VideoInfo,
)


def utt(i, text, s, e, **kw):
    return Utterance(id=f"utt_{i:04d}", text=text, start=s, end=e, **kw)


def ocr(i, text, s, e, **kw):
    return OCRText(id=f"ocr_{i:04d}", text=text, start=s, end=e, **kw)


def event(i, type_, s, e, **kw):
    return Event(id=f"evt_{i:04d}", type=type_, start=s, end=e, refs={"ocr": ["ocr_0000"]},
                 **kw)


def doc(duration=120.0, **kw):
    return VideoContextDocument(
        id="vid_e",
        video=VideoInfo(id="vid_e", filename="e.mp4", duration=duration, has_audio=True),
        transcript=list(kw.get("transcript", ())),
        ocr=list(kw.get("ocr", ())),
        vision=list(kw.get("vision", ())),
        events=list(kw.get("events", ())),
    )


class TestCrossModalLinking:
    def test_stripe_links_ocr_and_transcript(self):
        d = doc(
            ocr=[ocr(0, "Stripe API Dashboard", 861.0, 867.0)],
            transcript=[utt(0, "Now let's configure Stripe for checkout", 858.0, 869.0)],
        )
        entities = extract_entities(d)
        stripe = next((e for e in entities if normalize_name(e.name) == "stripe api dashboard"
                       or "stripe" in e.name.lower()), None)
        assert stripe is not None
        assert set(stripe.linked_modalities) >= {"ocr", "transcript"}
        assert stripe.confidence == 0.85
        assert not stripe.ambiguous
        timeline = entity_timeline(stripe)
        assert [o.start for o in timeline] == sorted(o.start for o in stripe.occurrences)

    def test_single_mention_is_not_an_entity(self):
        d = doc(ocr=[ocr(0, "Ephemeral Banner", 5.0, 9.0)])
        assert all(e.name != "Ephemeral Banner" for e in extract_entities(d))

    def test_repeated_single_modality_is_uncertain(self):
        d = doc(transcript=[utt(0, "Welcome to Acme Portal today", 1.0, 5.0),
                            utt(1, "Back in Acme Portal again", 10.0, 14.0),
                            utt(2, "Finally Acme Portal ships", 20.0, 24.0)])
        entities = extract_entities(d)
        acme = next(e for e in entities if "acme" in e.name.lower())
        assert acme.ambiguous and acme.confidence < 0.85

    def test_error_and_command_entities(self):
        d = doc(
            ocr=[ocr(0, "TypeError: Cannot read properties of undefined", 100.0, 110.0),
                 ocr(1, "$ npm run deploy", 50.0, 60.0)],
            events=[
                event(0, "error_shown", 100.0, 110.0,
                      description="TypeError: Cannot read properties of undefined",
                      confidence=0.9),
                event(1, "command_entered", 50.0, 60.0,
                      attributes={"command": "npm run deploy"}),
            ],
        )
        by_type = {e.type: e for e in extract_entities(d)}
        assert by_type["ERROR"].method == "error_signature"
        assert by_type["COMMAND"].name == "npm run deploy"

    def test_deterministic(self):
        d = doc(ocr=[ocr(0, "Stripe API", 1.0, 5.0)],
                transcript=[utt(0, "Stripe API here", 1.0, 5.0)])
        assert [e.to_dict() for e in extract_entities(d)] == \
               [e.to_dict() for e in extract_entities(d)]

    def test_empty(self):
        assert extract_entities(doc()) == []


class TestCollection:
    def _docs(self):
        a = doc(transcript=[utt(0, "OAuth login flow explained", 5.0, 12.0)])
        b = doc(transcript=[utt(0, "cooking pasta tonight", 5.0, 12.0)])
        return {"a": a, "b": b}

    def test_search_preserves_provenance(self):
        index = CollectionIndex(self._docs())
        result = index.search("OAuth login")
        assert result.videos_searched == 2
        assert all(s.video_id == "a" for s in result.spans)
        assert all("[from a]" in s.reason for s in result.spans)

    def test_per_video_scores(self):
        result = CollectionIndex(self._docs()).search("flow", per_video_k=5)
        assert result.per_video[0].video_id == "a"

    def test_compare(self):
        docs = self._docs()
        cmp = compare("a", docs["a"], "b", docs["b"])
        assert cmp.video_a == "a" and cmp.video_b == "b"
        assert cmp.coverage_a["transcript"] == 1
        d = cmp.to_dict()
        assert set(d) == {"video_a", "video_b", "shared_entities", "unique_a", "unique_b",
                          "events_a", "events_b", "coverage_a", "coverage_b",
                          "added", "removed", "changed", "unchanged", "uncertain",
                          "structure_a", "structure_b"}
        assert set(cmp.added) == set(cmp.unique_b)
        assert set(cmp.removed) == set(cmp.unique_a)


class TestPlans:
    def test_speech_task(self):
        plan = plan_for_task("What was said about revenue?")
        assert set(plan.required_stages) >= {"audio", "asr", "segments"}
        assert all(s.why for s in plan.stages)

    def test_visual_task_selects_vision(self):
        plan = plan_for_task("Recreate the website design and colors")
        assert "vision" in plan.required_stages

    def test_default_plan_keeps_vision_optional(self):
        plan = plan_for_task("summarize this")
        assert "vision" not in plan.required_stages
        assert plan.notes, "the vision omission must be explained"

    def test_coverage(self):
        d = doc()
        d.stages = [StageRecord(name="asr", status=StageStatus.OK),
                    StageRecord(name="audio", status=StageStatus.OK)]
        plan = plan_for_task("What was said?")
        report = check_coverage(d, plan)
        assert "asr" in report.satisfied
        assert not report.complete  # segments never ran
        assert any(g.stage == "segments" for g in report.missing)
        assert all(g.hint for g in report.missing)
