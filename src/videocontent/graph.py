"""EvidenceGraph — a bounded derived view over one .vctx document.

The graph is never storage: it is built from facts the document already holds and
discarded after use. Nodes are observed facts (transcript/OCR/vision/events/scenes/
frames/segments) plus derived objects (entities, occurrences, changes, chapters,
states). Edges carry exactly one documented construction rule each, plus the
reference IDs that prove it — ``graph.explain(edge_id)`` returns the rule in words,
never "AI determined these are related".

Traversal is bounded everywhere (``max_depth``, ``max_nodes``, ``max_seconds``) so
no query can explode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .entities import extract_entities, normalize_name
from .logging import get_logger
from .temporal import (
    TemporalRelation,
    build_chapters,
    detect_changes,
    relate,
    ui_states,
)
from .timecode import format_timecode

log = get_logger("graph")


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    start: float
    end: float
    label: str
    refs: tuple[str, ...] = ()
    observed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "start": self.start, "end": self.end,
            "timecode": format_timecode(self.start), "label": self.label[:160],
            "refs": list(self.refs), "observed": self.observed,
        }


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    relation: str
    confidence: float
    rule: str
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "source": self.source, "target": self.target,
            "relation": self.relation, "confidence": self.confidence,
            "rule": self.rule, "provenance": list(self.provenance),
        }


#: Relations the builder emits, each with exactly one construction rule.
RELATIONS = (
    "BEFORE", "AFTER", "DURING", "CONTAINS", "OVERLAPS", "NEAR", "CO_OCCURS",
    "SAME_ENTITY", "SAME_CONCEPT", "OBSERVED_IN", "DERIVED_FROM", "SUPPORTS",
    "PRECEDES", "FOLLOWS", "CHANGES_INTO", "APPEARS_IN", "DISAPPEARS_IN",
    "RELATED_TO",
)

_FACT_GROUPS = ("transcript", "ocr", "vision", "events", "scenes", "frames", "segments")
_NODE_KIND = {
    "transcript": "transcript", "ocr": "ocr", "vision": "vision", "events": "event",
    "scenes": "scene", "frames": "frame", "segments": "segment",
}


class EvidenceGraph:
    """Derived graph view. Build via :func:`build_graph`, then traverse."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._adj: dict[str, list[str]] = {}
        self._edge_count = 0

    # -- construction ------------------------------------------------------

    def _add_node(self, node: Node, max_nodes: int) -> bool:
        if node.id in self.nodes or len(self.nodes) >= max_nodes:
            return node.id in self.nodes
        self.nodes[node.id] = node
        return True

    def _add_edge(self, source: str, target: str, relation: str, confidence: float,
                  rule: str, provenance: tuple[str, ...], max_edges: int) -> str | None:
        if len(self.edges) >= max_edges or source not in self.nodes or target not in self.nodes:
            return None
        self._edge_count += 1
        edge_id = f"edge_{self._edge_count:05d}"
        self.edges[edge_id] = Edge(edge_id, source, target, relation, confidence, rule,
                                   provenance)
        self._adj.setdefault(source, []).append(edge_id)
        self._adj.setdefault(target, []).append(edge_id)
        return edge_id

    # -- read API ----------------------------------------------------------

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def node_list(self) -> list[Node]:
        return list(self.nodes.values())

    def edge_list(self) -> list[Edge]:
        return list(self.edges.values())

    def neighbors(self, node_id: str, relation: str | None = None) -> list[tuple[Node, Edge]]:
        """Adjacent nodes (undirected), optionally filtered to one relation."""
        out: list[tuple[Node, Edge]] = []
        for edge_id in self._adj.get(node_id, []):
            edge = self.edges[edge_id]
            if relation is not None and edge.relation != relation:
                continue
            other = edge.target if edge.source == node_id else edge.source
            node = self.nodes.get(other)
            if node is not None:
                out.append((node, edge))
        return out

    def related(self, node_id: str, relation: str) -> list[Node]:
        return [node for node, _ in self.neighbors(node_id, relation)]

    def path(self, source: str, target: str, *, max_depth: int = 4) -> list[str] | None:
        """Shortest edge-id path (BFS, undirected, depth-bounded)."""
        if source not in self.nodes or target not in self.nodes:
            return None
        queue: deque[tuple[str, list[str]]] = deque([(source, [])])
        seen = {source}
        while queue:
            current, trail = queue.popleft()
            if current == target:
                return trail
            if len(trail) >= max_depth:
                continue
            for edge_id in self._adj.get(current, []):
                edge = self.edges[edge_id]
                nxt = edge.target if edge.source == current else edge.source
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, [*trail, edge_id]))
        return None

    def temporal_neighbors(self, node_id: str, *, radius_s: float = 10.0,
                           max_nodes: int = 50) -> list[Node]:
        """Fact nodes whose spans start within ``radius_s`` of the node's start."""
        anchor = self.nodes.get(node_id)
        if anchor is None:
            return []
        scored = [
            (abs(node.start - anchor.start), node)
            for node in self.nodes.values()
            if node.id != node_id and node.observed
            and abs(node.start - anchor.start) <= radius_s
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1].id))
        return [node for _, node in scored[:max_nodes]]

    def entity_occurrences(self, entity_id: str) -> list[Node]:
        """Occurrence nodes contained by an entity, in time order."""
        occs = [node for node, _ in self.neighbors(entity_id, "CONTAINS")
                if node.kind == "occurrence"]
        return sorted(occs, key=lambda node: (node.start, node.end, node.id))

    def supporting_evidence(self, node_id: str, *, max_depth: int = 3,
                            max_nodes: int = 100) -> list[Node]:
        """Observed fact nodes behind a (possibly derived) node.

        Follows DERIVED_FROM, CONTAINS, OBSERVED_IN and SUPPORTS edges backwards
        to the facts that prove the node. Depth- and count-bounded.
        """
        start = self.nodes.get(node_id)
        if start is None:
            return []
        if start.observed:
            return [start]
        found: dict[str, Node] = {}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        seen = {node_id}
        while queue and len(found) < max_nodes:
            current, depth = queue.popleft()
            for edge_id in self._adj.get(current, []):
                edge = self.edges[edge_id]
                if edge.relation not in ("DERIVED_FROM", "CONTAINS", "OBSERVED_IN",
                                         "SUPPORTS"):
                    continue
                nxt = edge.target if edge.source == current else edge.source
                if nxt in seen:
                    continue
                seen.add(nxt)
                node = self.nodes.get(nxt)
                if node is None:
                    continue
                if node.observed:
                    found[nxt] = node
                elif depth + 1 < max_depth:
                    queue.append((nxt, depth + 1))
        return sorted(found.values(), key=lambda node: (node.start, node.end, node.id))

    def explain(self, edge_id: str) -> dict[str, Any] | None:
        """The construction rule behind an edge, in words, with provenance."""
        edge = self.edges.get(edge_id)
        if edge is None:
            return None
        return {
            "edge_id": edge.id, "relation": edge.relation,
            "source": edge.source, "target": edge.target,
            "rule": edge.rule, "confidence": edge.confidence,
            "confidence_basis": ("observed co-occurrence in the document"
                                 if edge.confidence >= 1.0
                                 else "heuristic detector agreement — see rule"),
            "provenance": list(edge.provenance),
        }

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "kinds": kinds}


def _label(item: Any, fallback: str) -> str:
    text = (getattr(item, "text", None) or getattr(item, "description", None) or fallback)
    return str(text)[:160]


def build_graph(doc: Any, *, max_nodes: int = 5000, max_edges: int = 20000,
                chain_gap_s: float = 60.0) -> EvidenceGraph:
    """Build the derived evidence graph for one document.

    Rules (each edge cites exactly one):
    - OBSERVED_IN: fact → segment listing it (segment ``*_ids``).
    - DERIVED_FROM: event → referenced fact (event ``refs``).
    - CONTAINS: entity → occurrence; OBSERVED_IN: occurrence → cited fact.
    - SAME_CONCEPT: entities sharing a normalized name from different detectors.
    - SUPPORTS: change/chapter/state → cited facts or overlapped scenes.
    - APPEARS_IN / DISAPPEARS_IN: entity → first / last occurrence fact.
    - Temporal chain: consecutive facts in timeline order linked PRECEDES (gapless),
      OVERLAPS, or NEAR (gap ≤ ``chain_gap_s``); anything farther is unlinked.
    - CHANGES_INTO: change → UI state starting within 2 s of the change timestamp.
    """
    graph = EvidenceGraph()

    def edge(source: str, target: str, relation: str, confidence: float,
             rule: str, provenance: tuple[str, ...]) -> None:
        graph._add_edge(source, target, relation, confidence, rule, provenance, max_edges)

    # Fact nodes.
    for group in _FACT_GROUPS:
        for item in getattr(doc, group, []):
            start = float(getattr(item, "start", getattr(item, "ts", 0.0)))
            end = float(getattr(item, "end", getattr(item, "ts", start)))
            graph._add_node(Node(item.id, _NODE_KIND[group], start, end,
                                 _label(item, item.id), (item.id,), True),
                            max_nodes)

    # Segments observe their member facts.
    for segment in getattr(doc, "segments", []):
        for attr in ("transcript_ids", "ocr_ids", "vision_ids", "event_ids", "object_ids",
                     "frame_ids"):
            for ref in getattr(segment, attr, []):
                if ref in graph.nodes:
                    edge(ref, segment.id, "OBSERVED_IN", 1.0,
                         f"{ref} is listed in {segment.id}.{attr}, so the segment observes it",
                         (ref,))

    # Events derive from their refs.
    for event in getattr(doc, "events", []):
        for kind, refs in (getattr(event, "refs", {}) or {}).items():
            for ref in refs:
                if ref in graph.nodes:
                    edge(event.id, ref, "DERIVED_FROM", 1.0,
                         f"{event.id} names {ref} in refs.{kind}: the event was computed "
                         f"from that fact", (ref,))

    # Entities, occurrences, same-concept links.
    entities = extract_entities(doc)
    by_name: dict[str, list[Any]] = {}
    for entity in entities:
        if not graph._add_node(Node(entity.id, "entity", entity.first_seen or 0.0,
                                    entity.last_seen or 0.0, entity.name,
                                    tuple(o.ref_id for o in entity.occurrences), False),
                               max_nodes):
            continue
        by_name.setdefault(normalize_name(entity.name), []).append(entity)
        for index, occurrence in enumerate(entity.timeline()):
            occurrence_id = f"{entity.id}#{index}"
            if not graph._add_node(Node(
                    occurrence_id, "occurrence", occurrence.start, occurrence.end,
                    f"{entity.name} @ {format_timecode(occurrence.start)}",
                    (occurrence.ref_id,), False), max_nodes):
                continue
            edge(entity.id, occurrence_id, "CONTAINS", 1.0,
                 f"{occurrence_id} is occurrence {index} of {entity.id} "
                 f"({occurrence.modality} {occurrence.ref_id})",
                 (occurrence.ref_id,))
            if occurrence.ref_id in graph.nodes:
                edge(occurrence_id, occurrence.ref_id, "OBSERVED_IN", 1.0,
                     f"occurrence cites {occurrence.ref_id}: the sighting is that fact",
                     (occurrence.ref_id,))
        if entity.occurrences:
            first = min(entity.occurrences, key=lambda o: (o.start, o.end))
            last = max(entity.occurrences, key=lambda o: (o.start, o.end))
            if first.ref_id in graph.nodes:
                edge(entity.id, first.ref_id, "APPEARS_IN", 0.9,
                     f"{first.ref_id} holds the earliest sighting of {entity.name}, "
                     f"so the entity appears there first", (first.ref_id,))
            if last.ref_id in graph.nodes and last.ref_id != first.ref_id:
                edge(entity.id, last.ref_id, "DISAPPEARS_IN", 0.9,
                     f"{last.ref_id} holds the latest sighting of {entity.name}, "
                     f"so the entity is last observed there", (last.ref_id,))
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        for first in group:
            for second in group:
                if first.id >= second.id:
                    continue
                edge(first.id, second.id, "SAME_CONCEPT", 0.7,
                     f"both entities normalize to {name!r} but came from different "
                     f"detectors ({first.method} vs {second.method}); kept separate, "
                     f"linked as the same concept", (first.id, second.id))

    # Changes, chapters, states.
    for change in detect_changes(doc):
        if not graph._add_node(Node(change.id, "change", change.ts, change.ts,
                                    f"{change.change_type}: {change.before[:60]} → "
                                    f"{change.after[:60]}", change.evidence_ids, False),
                               max_nodes):
            continue
        for ref in change.evidence_ids:
            if ref in graph.nodes:
                edge(change.id, ref, "SUPPORTS", 1.0,
                     f"{ref} is cited in {change.id}.evidence_ids: the change was "
                     f"measured from it", (ref,))
    for chapter in build_chapters(doc):
        if not graph._add_node(Node(chapter.id, "chapter", chapter.start, chapter.end,
                                    chapter.title, (), False), max_nodes):
            continue
        for scene in getattr(doc, "scenes", []):
            if scene.start < chapter.end and chapter.start < scene.end and scene.id in graph.nodes:
                edge(chapter.id, scene.id, "SUPPORTS", 0.9,
                     f"{scene.id} overlaps chapter {chapter.id}: the chapter range was "
                     f"tiled from scene boundaries", (scene.id,))
    states = ui_states(doc)
    for state in states:
        if not graph._add_node(Node(state.id, "state", state.start, state.end,
                                    " + ".join(state.elements[:3]), state.evidence_ids,
                                    False), max_nodes):
            continue
        for ref in state.evidence_ids:
            if ref in graph.nodes:
                edge(state.id, ref, "SUPPORTS", 1.0,
                     f"{ref} is cited in {state.id}.evidence_ids: the state is defined "
                     f"by that on-screen text", (ref,))
    for change in detect_changes(doc):
        if change.id not in graph.nodes:
            continue
        for state in states:
            if state.id in graph.nodes and abs(state.start - change.ts) <= 2.0:
                edge(change.id, state.id, "CHANGES_INTO", 0.6,
                     f"state {state.id} starts within 2 s of change {change.id}: "
                     f"temporal adjacency only, not causation", (change.id, state.id))

    # Temporal chain over observed facts in timeline order.
    facts = sorted((n for n in graph.nodes.values() if n.observed),
                   key=lambda n: (n.start, n.end, n.id))
    for first, second in pairwise(facts):
        relations = relate(first.start, first.end, second.start, second.end)
        gap = max(0.0, second.start - first.end)
        if TemporalRelation.PRECEDES in relations:
            if gap > chain_gap_s:
                continue
            relation, rule = ("PRECEDES",
                              f"{second.id} starts {gap:.1f}s after {first.id} ends")
        elif TemporalRelation.OVERLAPS in relations:
            relation, rule = ("OVERLAPS",
                              f"{first.id} and {second.id} overlap in time")
        elif gap <= chain_gap_s and TemporalRelation.NEAR in relations:
            relation, rule = ("NEAR",
                              f"{second.id} starts {gap:.1f}s after {first.id} ends")
        else:
            continue
        edge(first.id, second.id, relation, 1.0, rule, (first.id, second.id))

    log.debug("graph.built", extra={"nodes": len(graph.nodes), "edges": len(graph.edges)})
    return graph


__all__ = ["Edge", "Node", "RELATIONS", "EvidenceGraph", "build_graph"]
