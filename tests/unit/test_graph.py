"""EvidenceGraph: nodes, rule-cited edges, bounded traversal, explanations."""

from __future__ import annotations

from videocontent.graph import RELATIONS, build_graph
from videocontent.schema.v1 import (
    Event,
    OCRText,
    Scene,
    Segment,
    Utterance,
    VideoContextDocument,
    VideoInfo,
)


def utt(i, text, s, e, **kw):
    return Utterance(id=f"utt_{i:04d}", text=text, start=s, end=e, **kw)


def ocr(i, text, s, e, **kw):
    return OCRText(id=f"ocr_{i:04d}", text=text, start=s, end=e,
                   first_frame_ts=s, last_frame_ts=e, stable=True, frame_count=3, **kw)


def doc(**kw):
    facts = {
        "transcript": list(kw.get("transcript", ())),
        "ocr": list(kw.get("ocr", ())),
        "events": list(kw.get("events", ())),
        "scenes": list(kw.get("scenes", ())),
    }
    return VideoContextDocument(
        id="vid_g",
        video=VideoInfo(id="vid_g", filename="g.mp4", duration=200.0, has_audio=True),
        segments=list(kw.get("segments", ())),
        **facts,
    )


def sample():
    return doc(
        scenes=[Scene(id="scene_0000", start=0.0, end=100.0),
                Scene(id="scene_0001", start=100.0, end=200.0)],
        transcript=[
            utt(0, "welcome to the demo", 5.0, 10.0),
            utt(1, "ConnectionError on port 5432", 120.0, 128.0),
        ],
        ocr=[
            ocr(0, "Welcome Dashboard", 4.0, 30.0),
            ocr(1, "ConnectionError refused", 120.0, 150.0),
        ],
        events=[Event(id="evt_0000", type="error_shown", start=120.0, end=128.0,
                      description="ConnectionError refused", refs={"ocr": ["ocr_0001"]})],
        segments=[Segment(id="segment_0000", start=0.0, end=100.0,
                          transcript_ids=["utt_0000"], ocr_ids=["ocr_0000"]),
                  Segment(id="segment_0001", start=100.0, end=200.0,
                          transcript_ids=["utt_0001"], ocr_ids=["ocr_0001"],
                          event_ids=["evt_0000"])],
    )


class TestNodes:
    def test_fact_and_derived_nodes(self):
        graph = build_graph(sample())
        kinds = {node.kind for node in graph.node_list()}
        assert {"transcript", "ocr", "event", "scene", "segment"} <= kinds
        assert "entity" in kinds  # ConnectionError links ocr + transcript
        assert "change" in kinds  # disjoint OCR across the scene boundary
        facts = [n for n in graph.node_list() if n.observed]
        derived = [n for n in graph.node_list() if not n.observed]
        assert facts and derived
        assert all(n.start <= n.end for n in graph.node_list())

    def test_empty_document(self):
        graph = build_graph(doc())
        # Nothing observed; only the duration-tiling chapter exists, with no edges.
        assert [n.kind for n in graph.node_list()] == ["chapter"]
        assert graph.edge_list() == []

    def test_bounds(self):
        graph = build_graph(sample(), max_nodes=5, max_edges=7)
        assert len(graph.node_list()) <= 5
        assert len(graph.edge_list()) <= 7


class TestEdges:
    def test_relations_are_documented(self):
        graph = build_graph(sample())
        assert set(RELATIONS) >= {"BEFORE", "SAME_ENTITY", "OBSERVED_IN", "DERIVED_FROM"}
        for edge in graph.edge_list():
            assert edge.relation in RELATIONS
            assert edge.rule, "every edge must cite its construction rule"
            assert edge.provenance, "every edge must carry provenance"

    def test_observed_in_and_derived_from(self):
        graph = build_graph(sample())
        assert any(e.relation == "OBSERVED_IN" and e.source == "utt_0000"
                   for e in graph.edge_list())
        derived = [e for e in graph.edge_list()
                   if e.relation == "DERIVED_FROM" and e.source == "evt_0000"]
        assert derived and derived[0].provenance == ("ocr_0001",)

    def test_no_duplicate_edges(self):
        graph = build_graph(sample())
        keys = [(e.source, e.target, e.relation) for e in graph.edge_list()]
        assert len(keys) == len(set(keys))

    def test_explain(self):
        graph = build_graph(sample())
        edge = next(e for e in graph.edge_list() if e.relation == "DERIVED_FROM")
        explanation = graph.explain(edge.id)
        assert explanation["rule"] == edge.rule
        assert "AI determined" not in explanation["rule"]
        assert explanation["provenance"] == list(edge.provenance)
        assert graph.explain("edge_99999") is None


class TestTraversal:
    def test_neighbors_and_related(self):
        graph = build_graph(sample())
        neighbors = graph.neighbors("evt_0000")
        assert any(node.id == "ocr_0001" for node, _ in neighbors)
        assert graph.related("evt_0000", "DERIVED_FROM") != []
        assert graph.neighbors("missing") == []

    def test_path(self):
        graph = build_graph(sample())
        trail = graph.path("utt_0001", "segment_0001")
        assert trail, "utterance reaches its segment in one hop"
        assert graph.path("utt_0001", "missing") is None
        # Depth bound respected.
        assert graph.path("utt_0000", "utt_0001", max_depth=0) is None

    def test_temporal_neighbors(self):
        graph = build_graph(sample())
        near = graph.temporal_neighbors("utt_0001", radius_s=15.0)
        assert {n.id for n in near} >= {"ocr_0001", "evt_0000"}
        assert all(n.observed for n in near)
        assert graph.temporal_neighbors("missing") == []

    def test_entity_occurrences(self):
        graph = build_graph(sample())
        entities = [n for n in graph.node_list() if n.kind == "entity"]
        assert entities
        occs = graph.entity_occurrences(entities[0].id)
        assert len(occs) >= 1
        assert [o.start for o in occs] == sorted(o.start for o in occs)

    def test_supporting_evidence(self):
        graph = build_graph(sample())
        events = [n for n in graph.node_list() if n.kind == "event"]
        support = graph.supporting_evidence(events[0].id)
        assert support and all(n.observed for n in support)
        # Observed facts support themselves.
        assert graph.supporting_evidence("utt_0000")[0].id == "utt_0000"
        assert graph.supporting_evidence("missing") == []

    def test_stats(self):
        stats = build_graph(sample()).stats()
        assert stats["nodes"] == len(build_graph(sample()).node_list())
        assert sum(stats["kinds"].values()) == stats["nodes"]
