"""Intelligence-layer benchmark: exact/temporal/first/entity/change/sequence/graph.

Usage: python bench_intel.py <video.vctx> [output.json]

Measures latency, candidate counts, selected evidence, context size, and graph
shape for the ten standard intelligence cases. No models, no network — pure
derived-view and retrieval work, so numbers are comparable across machines.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from videocontent.collection import CollectionIndex, compare
from videocontent.entities import extract_entities
from videocontent.graph import build_graph
from videocontent.queryplan import build_plan
from videocontent.retrieval import Retriever
from videocontent.schema import io
from videocontent.sdk import Video
from videocontent.temporal import detect_changes, event_chains


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bench_intel.py <video.vctx> [output.json]")
        raise SystemExit(2)
    doc = io.load(sys.argv[1])
    video = Video.from_document(doc)
    retriever = Retriever(doc)
    graph = build_graph(doc)
    entities = extract_entities(doc)
    anchor = entities[0].name if entities else "the"
    collection = CollectionIndex({"a": doc, "b": doc})

    cases = []
    start = time.perf_counter()
    exact = retriever.search(anchor, top_k=5)
    cases.append({"case": "exact_lookup", "latency_ms": _ms(start),
                  "query": anchor, "result": {"spans": len(exact.spans)}})

    def _temporal():
        planned = build_plan(f"what happened after {anchor}", doc=doc)
        _, found = retriever.query_temporal(f"what happened after {anchor}")
        return planned, found

    start = time.perf_counter()
    plan, temporal = _temporal()
    cases.append({"case": "temporal_lookup", "latency_ms": _ms(start),
                  "query": plan.user_query,
                  "result": {"intent": plan.intent.value,
                             "spans": len(temporal.spans)}})
    start = time.perf_counter()
    first = video.entity_timeline(anchor)
    cases.append({"case": "first_occurrence", "latency_ms": _ms(start),
                  "result": {"found": first is not None,
                             "occurrences": len(first.occurrences) if first else 0}})
    start = time.perf_counter()
    changes = detect_changes(doc)
    cases.append({"case": "change_detection", "latency_ms": _ms(start),
                  "result": {"changes": len(changes)}})
    start = time.perf_counter()
    chains = event_chains(doc)
    cases.append({"case": "event_sequence", "latency_ms": _ms(start),
                  "result": {"chains": len(chains),
                             "events": sum(len(c.events) for c in chains)}})
    start = time.perf_counter()
    node = next(iter(graph.nodes), None)
    neighbors = graph.neighbors(node) if node else []
    path = None
    nodes = list(graph.nodes)
    if len(nodes) >= 2:
        path = graph.path(nodes[0], nodes[-1])
    cases.append({"case": "graph_traversal", "latency_ms": _ms(start),
                  "result": {"nodes": len(graph.nodes), "edges": len(graph.edges),
                             "neighbors": len(neighbors),
                             "path_found": path is not None}})
    start = time.perf_counter()
    found = collection.search(anchor, top_k=10)
    cases.append({"case": "collection_search", "latency_ms": _ms(start),
                  "result": {"spans": len(found.spans),
                             "videos": found.videos_searched}})
    start = time.perf_counter()
    diff = compare("a", doc, "b", doc)
    cases.append({"case": "comparison", "latency_ms": _ms(start),
                  "result": {"shared": len(diff.shared_entities),
                             "changed": len(diff.changed)}})
    start = time.perf_counter()
    package = video.context_package(f"what is {anchor}", max_spans=5)
    raw_tokens = sum(len(u.text or "") for u in doc.transcript) // 4
    cases.append({"case": "context_compression", "latency_ms": _ms(start),
                  "result": {"evidence": len(package.evidence),
                             "package_chars": len(package.to_text()),
                             "raw_transcript_tokens_est": raw_tokens}})

    print(f"{'case':<20} {'ms':>10}  result")
    for case in cases:
        print(f"{case['case']:<20} {case['latency_ms']:>10.2f}  {case['result']}")
    if len(sys.argv) >= 3:
        Path(sys.argv[2]).write_text(json.dumps({"video": sys.argv[1], "cases": cases},
                                                indent=2, default=str))
        print(f"\nResults saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()
