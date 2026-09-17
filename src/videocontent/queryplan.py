"""QueryPlan — inspectable intent + strategy for one question.

A plan separates *understanding the question* (intent, entities, temporal
constraints, modalities) from *executing it* (retrieval strategy, graph
operations, budget, fallback). Agents and developers read the plan; the
executor in :mod:`videocontent.retrieval.query` runs it. No LLM, no network —
keyword rules over the query text plus optional resolution against the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .entities import candidate_terms, extract_entities, normalize_name
from .logging import get_logger
from .temporal import TemporalQuery, parse_temporal_query

log = get_logger("queryplan")


class Intent(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    TEMPORAL_BETWEEN = "temporal_between"
    TEMPORAL_WHILE = "temporal_while"
    FIRST_OCCURRENCE = "first_occurrence"
    LAST_OCCURRENCE = "last_occurrence"
    CHANGE_DETECTION = "change_detection"
    ENTITY_TIMELINE = "entity_timeline"
    EVENT_SEQUENCE = "event_sequence"
    CO_OCCURRENCE = "co_occurrence"
    COMPARISON = "comparison"
    MULTI_VIDEO_SEARCH = "multi_video_search"
    CROSS_VIDEO_COMPARISON = "cross_video_comparison"
    UI_STATE_QUERY = "ui_state_query"
    EVIDENCE_REQUEST = "evidence_request"
    GENERAL_SUMMARY = "general_summary"


@dataclass(frozen=True)
class TemporalThresholds:
    """Configurable gap semantics: 'immediately' vs 'shortly' vs 'around'."""

    immediately_s: float = 10.0
    shortly_s: float = 30.0
    around_s: float = 30.0
    near_s: float = 5.0
    expand_s: float = 5.0

    def to_dict(self) -> dict[str, float]:
        return {
            "immediately_s": self.immediately_s, "shortly_s": self.shortly_s,
            "around_s": self.around_s, "near_s": self.near_s, "expand_s": self.expand_s,
        }


@dataclass(frozen=True)
class QueryPlan:
    user_query: str
    intent: Intent
    entities: tuple[str, ...] = ()
    temporal: TemporalQuery | None = None
    modalities: tuple[str, ...] = ()
    retrieval_strategy: tuple[str, ...] = ()
    graph_operations: tuple[str, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    fallback_strategy: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query, "intent": self.intent.value,
            "entities": list(self.entities),
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "modalities": list(self.modalities),
            "retrieval_strategy": list(self.retrieval_strategy),
            "graph_operations": list(self.graph_operations),
            "budget": dict(self.budget),
            "fallback_strategy": self.fallback_strategy,
            "coverage": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                         for k, v in self.coverage.items()},
            "warnings": list(self.warnings),
        }


_COLLECTION_CUES = ("which videos", "across videos", "across all", "every video", "all videos",
                    "both videos", "in which video")
_COMPARISON_CUES = ("compare", "difference", "differ", " vs ", "versus", "regression",
                    "old vs", "before vs", "before/after")
_SUMMARY_CUES = ("summarize", "summary", "overview of", "tldr", "tl;dr", "recap")
_ENTITY_TIMELINE_CUES = ("timeline of", "timeline for", "occurrences of", "every time",
                         "all mentions", "all occurrences", "when did")
_EVENT_SEQUENCE_CUES = ("sequence", "in order", "step by step", "what happened next",
                        "then what", "chain of events")
_EVIDENCE_CUES = ("show me the evidence", "prove it", "evidence for", "cite", "source of")
_CHANGE_CUES = ("what changed", "changed after", "changed between", "difference between",
                "transition", "became visible", "stopped appearing")
_UI_CUES = ("screen", "login page", "button", "state of", "visible on screen", "ui ",
            "interface", "modal", "homepage", "dashboard")
_MODALITY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("transcript", ("said", "saying", "spoken", "quote", "mention", "tell me", "hear")),
    ("ocr", ("shown", "visible", "screen", "written", "display", "slide", "text")),
    ("vision", ("look like", "appearance", "color", "layout", "animation", "show")),
    ("events", ("happen", "event", "click", "navigate", "error", "change")),
)

#: Intent → modalities that can answer it (empty = any present modality).
_INTENT_MODALITIES: dict[Intent, tuple[str, ...]] = {
    Intent.CHANGE_DETECTION: ("ocr", "transcript", "scenes"),
    Intent.EVENT_SEQUENCE: ("events",),
    Intent.UI_STATE_QUERY: ("ocr",),
    Intent.EVIDENCE_REQUEST: (),
    Intent.GENERAL_SUMMARY: (),
}
_MODALITY_HINTS = {
    "transcript": "reprocess with asr.enabled=true",
    "ocr": "reprocess with ocr.enabled=true (requires tesseract)",
    "vision": "reprocess with vision.enabled=true and a configured provider",
    "events": "reprocess normally — events derive from extracted modalities",
    "scenes": "reprocess with sampling.scene_detection=true",
}

_STRATEGIES: dict[Intent, tuple[str, ...]] = {
    Intent.FACT_LOOKUP: ("lexical retrieval", "co-occurrence boost",
                         "rank and select"),
    Intent.TEMPORAL_BEFORE: ("resolve anchor entity", "locate anchor range",
                             "retrieve preceding facts", "rank by recency",
                             "apply budget"),
    Intent.TEMPORAL_AFTER: ("resolve anchor entity", "locate anchor range",
                            "retrieve subsequent facts", "expand neighborhood",
                            "rank by recency", "apply budget"),
    Intent.TEMPORAL_BETWEEN: ("resolve both anchors", "constrain to range",
                              "retrieve within range", "apply budget"),
    Intent.TEMPORAL_WHILE: ("retrieve anchor matches",
                            "expand co-occurring neighborhood", "apply budget"),
    Intent.FIRST_OCCURRENCE: ("retrieve anchor matches", "return earliest span"),
    Intent.LAST_OCCURRENCE: ("retrieve anchor matches", "return latest span"),
    Intent.CHANGE_DETECTION: ("detect adjacent-region turnover",
                              "attach before/after evidence", "apply budget"),
    Intent.ENTITY_TIMELINE: ("resolve entity", "list all occurrences in time order",
                             "attach surrounding context"),
    Intent.EVENT_SEQUENCE: ("order events by timestamp", "split on large gaps"),
    Intent.CO_OCCURRENCE: ("retrieve anchor matches", "expand neighborhood",
                           "keep co-occurring modalities"),
    Intent.COMPARISON: ("extract entities per video", "diff occurrences and events"),
    Intent.MULTI_VIDEO_SEARCH: ("search each video", "merge by score",
                                "preserve video provenance"),
    Intent.CROSS_VIDEO_COMPARISON: ("link entities across videos",
                                    "diff per-video evidence"),
    Intent.UI_STATE_QUERY: ("list stable UI states", "match overlapping states"),
    Intent.EVIDENCE_REQUEST: ("retrieve anchor matches", "return with references"),
    Intent.GENERAL_SUMMARY: ("chapter overview", "global context", "key evidence"),
}
_GRAPH_OPS: dict[Intent, tuple[str, ...]] = {
    Intent.TEMPORAL_BEFORE: ("temporal_neighbors", "supporting_evidence"),
    Intent.TEMPORAL_AFTER: ("temporal_neighbors", "supporting_evidence"),
    Intent.TEMPORAL_BETWEEN: ("supporting_evidence",),
    Intent.TEMPORAL_WHILE: ("temporal_neighbors",),
    Intent.ENTITY_TIMELINE: ("entity_occurrences", "supporting_evidence"),
    Intent.FIRST_OCCURRENCE: ("entity_occurrences",),
    Intent.LAST_OCCURRENCE: ("entity_occurrences",),
    Intent.EVENT_SEQUENCE: ("temporal_neighbors",),
    Intent.CO_OCCURRENCE: ("temporal_neighbors",),
    Intent.EVIDENCE_REQUEST: ("supporting_evidence",),
    Intent.CHANGE_DETECTION: ("supporting_evidence",),
    Intent.UI_STATE_QUERY: ("temporal_neighbors", "supporting_evidence"),
}
_FALLBACKS: dict[Intent, str] = {
    Intent.TEMPORAL_BEFORE: "timeline window before the best anchor match",
    Intent.TEMPORAL_AFTER: "timeline window after the best anchor match",
    Intent.TEMPORAL_BETWEEN: "full timeline of the range",
    Intent.ENTITY_TIMELINE: "lexical search for the name",
    Intent.FIRST_OCCURRENCE: "lexical search for the name",
    Intent.LAST_OCCURRENCE: "lexical search for the name",
}


def _intent_for(query: str, temporal: TemporalQuery) -> Intent:
    lowered = query.lower()
    if any(cue in lowered for cue in _COLLECTION_CUES) or (
            "video" in lowered and any(cue in lowered for cue in _COMPARISON_CUES)):
        if any(cue in lowered for cue in _COMPARISON_CUES):
            return Intent.CROSS_VIDEO_COMPARISON
        return Intent.MULTI_VIDEO_SEARCH
    if any(cue in lowered for cue in _COMPARISON_CUES):
        return Intent.COMPARISON
    if any(cue in lowered for cue in _SUMMARY_CUES):
        return Intent.GENERAL_SUMMARY
    if temporal.relation is not None:
        from .temporal import TemporalRelation

        if temporal.relation is TemporalRelation.BEFORE:
            return Intent.TEMPORAL_BEFORE
        if temporal.relation is TemporalRelation.AFTER:
            return Intent.TEMPORAL_AFTER
        if temporal.relation is TemporalRelation.CO_OCCURS \
                and re.search(r"\bwhile\b", lowered):
            return Intent.TEMPORAL_WHILE
        if temporal.relation in (TemporalRelation.CO_OCCURS, TemporalRelation.DURING):
            return Intent.CO_OCCURRENCE
    if temporal.anchor_b is not None or (temporal.start is not None
                                         and temporal.end is not None):
        return Intent.TEMPORAL_BETWEEN
    if temporal.first_only:
        return Intent.FIRST_OCCURRENCE
    if temporal.last_only:
        return Intent.LAST_OCCURRENCE
    if temporal.start is not None or temporal.end is not None:
        return Intent.TEMPORAL_BETWEEN
    if any(cue in lowered for cue in _EVENT_SEQUENCE_CUES):
        return Intent.EVENT_SEQUENCE
    if any(cue in lowered for cue in _EVIDENCE_CUES):
        return Intent.EVIDENCE_REQUEST
    if any(cue in lowered for cue in _CHANGE_CUES):
        return Intent.CHANGE_DETECTION
    if any(cue in lowered for cue in _ENTITY_TIMELINE_CUES):
        return Intent.ENTITY_TIMELINE
    if any(cue in lowered for cue in _UI_CUES):
        return Intent.UI_STATE_QUERY
    return Intent.FACT_LOOKUP


def _modalities_for(query: str) -> tuple[str, ...]:
    lowered = query.lower()
    return tuple(mod for mod, cues in _MODALITY_CUES if any(cue in lowered for cue in cues))


def _resolve_entities(query: str, doc: Any | None) -> tuple[str, ...]:
    if doc is None:
        return tuple(dict.fromkeys(candidate_terms(query)))
    known = {normalize_name(entity.name): entity.name for entity in extract_entities(doc)}
    resolved: list[str] = []
    for term in candidate_terms(query):
        key = normalize_name(term)
        resolved.append(known.get(key, term))
    return tuple(dict.fromkeys(resolved))


def _coverage(doc: Any, modalities: tuple[str, ...]) -> dict[str, Any]:
    present = {
        "transcript": bool(getattr(doc, "transcript", [])),
        "ocr": bool(getattr(doc, "ocr", [])),
        "vision": bool(getattr(doc, "vision", [])),
        "events": bool(getattr(doc, "events", [])),
        "scenes": bool(getattr(doc, "scenes", [])),
    }
    missing = [mod for mod in modalities if not present.get(mod, False)]
    return {
        "present": sorted(mod for mod, ok in present.items() if ok),
        "missing": missing,
        "suggestions": [_MODALITY_HINTS[mod] for mod in missing],
    }


def build_plan(query: str, *, doc: Any | None = None,
               thresholds: TemporalThresholds | None = None,
               budget: dict[str, Any] | None = None) -> QueryPlan:
    """Plan one question: intent, entities, temporal constraints, strategy.

    ``doc`` is optional — with it, entity names resolve against extracted entities
    and modality coverage is checked; without it the plan is query-only.
    """
    from .routing import ContextBudget

    temporal = parse_temporal_query(query)
    intent = _intent_for(query, temporal)
    entities = _resolve_entities(temporal.anchor, doc)
    if temporal.anchor_b:
        entities = tuple(dict.fromkeys([*entities, *_resolve_entities(temporal.anchor_b, doc)]))
    modalities = _modalities_for(query) or _INTENT_MODALITIES.get(intent, ())
    warnings: list[str] = []
    coverage: dict[str, Any] = {}
    if doc is not None and modalities:
        coverage = _coverage(doc, modalities)
        if coverage["missing"]:
            warnings.append("query needs modalities absent from this document: "
                            + ", ".join(coverage["missing"]))
    default_budget = ContextBudget()
    return QueryPlan(
        user_query=query,
        intent=intent,
        entities=entities,
        temporal=temporal if temporal.is_temporal else None,
        modalities=modalities,
        retrieval_strategy=_STRATEGIES[intent],
        graph_operations=_GRAPH_OPS.get(intent, ()),
        budget=budget or default_budget.to_dict(),
        fallback_strategy=_FALLBACKS.get(intent, "ranked lexical search"),
        coverage=coverage,
        warnings=tuple(warnings),
    )


__all__ = ["Intent", "QueryPlan", "TemporalThresholds", "build_plan"]
