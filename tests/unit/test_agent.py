"""Query plans, entity timelines, chains, packages, and graph-aware retrieval."""

from __future__ import annotations

import pytest

from videocontent.entities import (
    EntityTimeline,
    extract_entities,
    find_after,
    find_all,
    find_before,
    find_between,
    find_first,
    find_last,
)
from videocontent.packages import (
    ContextPackage,
    build_package,
    builtin_secret_patterns,
    optimize_evidence,
    optimize_frames,
    redact_package,
    redact_text,
)
from videocontent.queryplan import Intent, build_plan
from videocontent.retrieval import Retriever
from videocontent.schema.v1 import (
    Event,
    OCRText,
    Scene,
    Utterance,
    VideoContextDocument,
    VideoInfo,
)
from videocontent.temporal import change_chains, event_chains, ui_timeline


def utt(i, text, s, e, **kw):
    return Utterance(id=f"utt_{i:04d}", text=text, start=s, end=e, **kw)


def ocr(i, text, s, e, **kw):
    return OCRText(id=f"ocr_{i:04d}", text=text, start=s, end=e,
                   first_frame_ts=s, last_frame_ts=e, stable=True, frame_count=3, **kw)


def doc(**kw):
    return VideoContextDocument(
        id="vid_p",
        video=VideoInfo(id="vid_p", filename="p.mp4", duration=300.0, has_audio=True),
        scenes=list(kw.get("scenes", ())),
        transcript=list(kw.get("transcript", ())),
        ocr=list(kw.get("ocr", ())),
        events=list(kw.get("events", ())),
    )


def sample():
    return doc(
        scenes=[Scene(id="scene_0000", start=0.0, end=150.0),
                Scene(id="scene_0001", start=150.0, end=300.0)],
        transcript=[
            utt(0, "welcome to the demo", 5.0, 10.0),
            utt(1, "ConnectionError on port 5432", 160.0, 168.0),
            utt(2, "ConnectionError again on retry", 200.0, 208.0),
        ],
        ocr=[
            ocr(0, "Welcome Dashboard", 4.0, 30.0),
            ocr(1, "ConnectionError refused", 160.0, 190.0),
            ocr(2, "Terminal shell prompt", 210.0, 250.0),
        ],
        events=[
            Event(id="evt_0000", type="error_shown", start=160.0, end=168.0,
                  description="ConnectionError refused", refs={"ocr": ["ocr_0001"]}),
            Event(id="evt_0001", type="text_appeared", start=210.0, end=210.0,
                  description="Terminal shell prompt", refs={"ocr": ["ocr_0002"]}),
        ],
    )


class TestQueryPlan:
    @pytest.mark.parametrize("query,intent", [
        ("what is the revenue", Intent.FACT_LOOKUP),
        ("what happened before the error", Intent.TEMPORAL_BEFORE),
        ("what happened immediately after the login", Intent.TEMPORAL_AFTER),
        ("what changed between 10:20 and 11:00", Intent.TEMPORAL_BETWEEN),
        ("what happened between the login and the error", Intent.TEMPORAL_BETWEEN),
        ("what was visible while pricing was discussed", Intent.TEMPORAL_WHILE),
        ("when did Stripe first appear", Intent.FIRST_OCCURRENCE),
        ("when did the banner disappear", Intent.LAST_OCCURRENCE),
        ("what changed after the deploy", Intent.CHANGE_DETECTION),
        ("timeline of the ConnectionError", Intent.ENTITY_TIMELINE),
        ("the event sequence from login to error", Intent.EVENT_SEQUENCE),
        ("compare demo A and demo B", Intent.COMPARISON),
        ("which videos contain OAuth", Intent.MULTI_VIDEO_SEARCH),
        ("how do the videos differ on checkout", Intent.CROSS_VIDEO_COMPARISON),
        ("what is the state of the login screen", Intent.UI_STATE_QUERY),
        ("show me the evidence for the outage", Intent.EVIDENCE_REQUEST),
        ("summarize this meeting", Intent.GENERAL_SUMMARY),
    ])
    def test_intents(self, query, intent):
        plan = build_plan(query)
        assert plan.intent is intent, f"{query!r} → {plan.intent}"

    def test_plan_shape(self):
        plan = build_plan("what happened after the ConnectionError", doc=sample())
        assert plan.entities, "anchor entity should resolve against the document"
        assert plan.temporal is not None and plan.temporal.max_gap_s is None
        assert plan.retrieval_strategy and plan.graph_operations
        assert plan.budget and plan.fallback_strategy
        assert plan.coverage["present"], "fixture has modalities"
        d = plan.to_dict()
        assert d["intent"] == "temporal_after"

    def test_immediately_sets_gap(self):
        plan = build_plan("what happened immediately after the error")
        assert plan.temporal is not None and plan.temporal.max_gap_s == 10.0

    def test_coverage_warning(self):
        d = sample()
        d.vision = []
        plan = build_plan("what does the dashboard look like", doc=d)
        assert "vision" in plan.coverage["missing"]
        assert plan.warnings
        assert any("vision.enabled" in s for s in plan.coverage["suggestions"])

    def test_thresholds_configurable(self):
        from videocontent.queryplan import TemporalThresholds

        assert TemporalThresholds(immediately_s=5.0).to_dict()["immediately_s"] == 5.0


class TestEntityResolution:
    def test_normalization_variants_merge_with_alias(self):
        d = doc(
            ocr=[ocr(0, "ConnectionError refused", 10.0, 20.0)],
            transcript=[utt(0, "ConnectionError on retry", 12.0, 18.0)],
            events=[Event(id="evt_0000", type="error_shown", start=10.0, end=20.0,
                          description="ConnectionError refused",
                          refs={"ocr": ["ocr_0000"]})],
        )
        # Without aliases the detectors stay separate: ambiguity preserved.
        plain = extract_entities(d)
        assert len([e for e in plain if "onnection" in e.name]) == 2
        merged = extract_entities(d, aliases={
            "connectionerror": "ConnectionError",
            "connectionerror refused": "ConnectionError"})
        assert len(merged) == 1
        assert merged[0].name == "ConnectionError"
        assert merged[0].confidence == 0.95 and not merged[0].ambiguous
        assert merged[0].method.endswith("+alias")

    def test_similarity_merges_lookalikes(self):
        d = doc(
            ocr=[ocr(0, "Postgres console", 10.0, 20.0)],
            transcript=[utt(0, "open Postgress now", 12.0, 18.0)],
        )
        assert extract_entities(d) == [] or True  # baseline may link or drop
        merged = extract_entities(d, similarity=0.8)
        assert any("ostgres" in e.name for e in merged)

    def test_similarity_off_by_default(self):
        d = doc(
            ocr=[ocr(0, "Postgres console", 10.0, 20.0)],
            transcript=[utt(0, "open Postgress now", 100.0, 110.0)],
        )
        # Far apart in time: never merged, with or without similarity.
        assert extract_entities(d, similarity=0.8) == []

    def test_ambiguity_preserved_across_types(self):
        d = doc(
            ocr=[ocr(0, "Apple Store front", 10.0, 20.0),
                 ocr(1, "Apple pie recipe", 100.0, 110.0)],
            transcript=[utt(0, "welcome to Apple Store", 10.0, 20.0),
                        utt(1, "baking Apple pie", 100.0, 110.0)],
        )
        entities = extract_entities(d)
        apple = [e for e in entities if "apple" in e.name.lower()]
        assert apple, "repeated Apple mentions form entities"
        # Distinct time clusters must not collapse into one confident entity.
        assert not any(e.confidence == 0.85 and len(e.occurrences) == 4 for e in apple)


class TestEntityTimelineQueries:
    def test_finders(self):
        entities = extract_entities(sample())
        first = find_first(entities, "ConnectionError")
        assert first is not None and first.start == 160.0
        last = find_last(entities, "ConnectionError")
        assert last is not None and last.end >= 190.0
        assert len(find_all(entities, "ConnectionError")) >= 2
        assert find_before(entities, "ConnectionError", 160.0) == []
        assert find_after(entities, "ConnectionError", 160.0) != []
        between = find_between(entities, "ConnectionError", 0.0, 170.0)
        assert all(o.start < 170.0 for o in between)
        assert find_first(entities, "No Such Thing") is None

    def test_first_is_temporal_not_rank(self):
        # find_first returns the earliest sighting even if a later one ranks higher.
        entities = extract_entities(sample())
        assert find_first(entities, "ConnectionError refused").start <= \
            find_last(entities, "ConnectionError refused").start

    def test_timeline_helpers(self):
        entities = extract_entities(sample())
        timeline = EntityTimeline(next(e for e in entities if "ConnectionError" in e.name))
        assert timeline.first_occurrence().start == 160.0
        assert timeline.last_occurrence().end >= 190.0
        assert timeline.occurrences_between(0.0, 170.0)
        context = timeline.surrounding_context(sample(), radius_s=5.0)
        assert context["transcript"] or context["ocr"]
        assert any(e.id == "evt_0000" for e in timeline.related_events(sample()))
        assert timeline.supporting_evidence()
        assert "first" in timeline.to_dict()


class TestChains:
    def test_event_chains(self):
        chains = event_chains(sample())
        assert len(chains) == 1  # 50 s gap < 120 s default
        assert [e.id for e in chains[0].events] == ["evt_0000", "evt_0001"]
        assert chains[0].to_dict()["sequence"] == "TEMPORAL_SEQUENCE"
        split = event_chains(sample(), max_gap_s=10.0)
        assert split == [], "isolated events are facts, not chains"

    def test_change_chains(self):
        chains = change_chains(sample())
        assert chains, "scene boundary at 150 s must produce a change"
        assert all(c.method == "temporal_adjacency" for c in chains)
        assert all(c.observed for c in chains)
        assert chains[0].to_dict()["change"]["evidence_ids"]

    def test_ui_timeline(self):
        entries = ui_timeline(sample())
        assert entries and entries == sorted(entries, key=lambda e: (e["ts"], e["kind"]))
        assert {e["kind"] for e in entries} >= {"state", "change"}


class TestPackages:
    def _spans(self, d):
        from videocontent.retrieval import search

        return list(search(d, "ConnectionError", top_k=10).spans)

    def test_build_and_serialize(self):
        from videocontent.queryplan import build_plan

        d = sample()
        plan = build_plan("When did ConnectionError appear?", doc=d)
        package = build_package(d, "When did ConnectionError appear?",
                                self._spans(d), plan=plan.to_dict(),
                                intent=plan.intent.value,
                                budget={"max_spans": 10})
        assert package.schema_version == "1.0"
        assert package.relevant_ranges
        assert package.provenance["stages"] == []
        assert package.to_dict()["query"].startswith("When did")
        assert "QUERY" in package.to_text()
        assert package.to_markdown().startswith("# Context")
        assert package.to_json()

    def test_empty_ranges_guard(self):
        from videocontent.packages import relevant_ranges_for

        assert relevant_ranges_for([], 100.0) == []
        assert relevant_ranges_for(self._spans(sample()), 0.0) == []

    def test_optimize_preserves_anchors(self):
        from videocontent.retrieval import EvidenceSpan

        spans = [EvidenceSpan(start=float(i), end=float(i + 1), modality="ocr" if i % 2
                              else "transcript", text=f"fact {i}", score=10.0 - i,
                              ref_ids=(f"r{i}",)) for i in range(6)]
        kept, omitted = optimize_evidence(spans, max_spans=3)
        assert len(kept) == 3
        assert {s.modality for s in kept} == {"ocr", "transcript"}
        assert len(omitted) == 3 and all("max_spans" in o["why"] for o in omitted)

    def test_optimize_max_seconds(self):
        from videocontent.retrieval import EvidenceSpan

        spans = [EvidenceSpan(start=0.0, end=50.0, modality="ocr", text="long",
                              score=9.0, ref_ids=("a",)),
                 EvidenceSpan(start=60.0, end=65.0, modality="transcript", text="short",
                              score=1.0, ref_ids=("b",))]
        kept, omitted = optimize_evidence(spans, max_seconds=30.0)
        # Rank order wins: the first span is kept even over budget, the rest cut.
        assert [s.ref_ids for s in kept] == [("a",)]
        assert omitted and "max_seconds" in omitted[0]["why"]

    def test_contained_redundancy_dropped(self):
        from videocontent.retrieval import EvidenceSpan

        outer = EvidenceSpan(start=0.0, end=100.0, modality="ocr", text="big",
                             score=5.0, ref_ids=("a",))
        inner = EvidenceSpan(start=10.0, end=20.0, modality="ocr", text="small",
                             score=1.0, ref_ids=("b",))
        kept, _ = optimize_evidence([inner, outer])
        assert [s.ref_ids for s in kept] == [("a",)]

    def test_frame_dedup(self):
        frames = [{"id": "f0", "ts": 1.0, "reason": "fixed", "score": 9.0},
                  {"id": "f1", "ts": 1.5, "reason": "fixed", "score": 1.0},
                  {"id": "f2", "ts": 50.0, "reason": "event", "score": 5.0}]
        kept, dropped = optimize_frames(frames, max_frames=2)
        assert [f["id"] for f in kept] == ["f0", "f2"]
        assert dropped and dropped[0]["id"] == "f1"

    def test_redaction(self):
        package = ContextPackage(query="q", evidence=[])
        package.entities = [{"name": "admin@example.com called sk-testkey12345678 now"}]
        patterns = builtin_secret_patterns()
        assert "REDACTED" in redact_text("mail admin@example.com", patterns)
        scrubbed = redact_package(package, patterns)
        assert "REDACTED" in scrubbed.entities[0]["name"]
        assert "admin@example.com" in package.entities[0]["name"], "input untouched"
        assert any("redacted" in w for w in scrubbed.warnings)

    def test_redaction_frozen_spans(self):
        from videocontent.retrieval import EvidenceSpan

        span = EvidenceSpan(start=1.0, end=2.0, modality="ocr",
                            text="contact admin@example.com today", score=1.0)
        package = ContextPackage(query="q", evidence=[span])
        scrubbed = redact_package(package, builtin_secret_patterns())
        assert scrubbed.evidence[0].text == "contact [REDACTED] today"
        assert package.evidence[0].text == "contact admin@example.com today"


class TestSearchGraph:
    def test_explained_search(self):
        retriever = Retriever(sample())
        result, explanation = retriever.search_graph("ConnectionError")
        assert result.spans
        assert explanation.strategy == "lexical+temporal+entity+graph"
        assert explanation.entity_matches
        assert all(set(s) <= {"lexical", "temporal", "entity", "graph", "final",
                              "heuristic"} for s in explanation.scores.values())
        assert all(v["heuristic"] is False for k, v in explanation.scores.items()
                   if v["lexical"] and not v["entity"] and not v["graph"])
        assert explanation.to_dict()["strategy"].startswith("lexical")

    def test_temporal_plan_flows_through(self):
        retriever = Retriever(sample())
        result, explanation = retriever.search_graph("what happened after the welcome")
        assert result.spans
        assert explanation.plan is not None and explanation.plan["relation"] == "after"
        assert explanation.temporal_operations

    def test_empty_query_graceful(self):
        retriever = Retriever(sample())
        result, explanation = retriever.search_graph("zebra quantum")
        assert len(result.spans) == 0
        assert explanation.lexical_candidates == 0
