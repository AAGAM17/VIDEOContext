"""ContextPackage — the agent-native unit of video context.

A package is everything a downstream model needs for one task and nothing else:
planned query, budgeted evidence, entities/events/changes in play, the temporal
relations between them, representative frames, graph summary, provenance, and an
honest account of what was omitted. Stable serializable schema (``schema_version``)
with JSON, compact-text and Markdown renderings.

Optimization never removes the only supporting evidence for a retained modality
unless the budget forces it — and then the omission is recorded, not silent.
Redaction is opt-in, pattern-based, and non-destructive (originals preserved).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .logging import get_logger
from .temporal import merge_windows, window_around
from .timecode import format_timecode

log = get_logger("packages")

PACKAGE_SCHEMA_VERSION = "1.0"


@dataclass
class ContextPackage:
    query: str
    evidence: list[Any] = field(default_factory=list)
    schema_version: str = PACKAGE_SCHEMA_VERSION
    intent: str | None = None
    query_plan: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    relevant_ranges: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    temporal_relations: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    graph_summary: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    omitted_information: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def dump(span: Any) -> Any:
            return span.to_dict() if hasattr(span, "to_dict") else span

        return {
            "schema_version": self.schema_version,
            "query": self.query,
            "intent": self.intent,
            "query_plan": self.query_plan,
            "metadata": self.metadata,
            "relevant_ranges": self.relevant_ranges,
            "evidence": [dump(s) for s in self.evidence],
            "entities": self.entities,
            "events": self.events,
            "changes": self.changes,
            "temporal_relations": self.temporal_relations,
            "frames": self.frames,
            "graph_summary": self.graph_summary,
            "provenance": self.provenance,
            "warnings": self.warnings,
            "budget": self.budget,
            "omitted_information": self.omitted_information,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        """Compact text rendering for small context windows."""
        lines = [f"QUERY: {self.query}"]
        if self.intent:
            lines.append(f"INTENT: {self.intent}")
        for i, span in enumerate(self.evidence, 1):
            tc = span.timecode if hasattr(span, "timecode") else "?"
            mod = getattr(span, "modality", "?")
            text = getattr(span, "text", str(span))[:200]
            lines.append(f"[{i}] {tc} ({mod}): {text}")
        if self.omitted_information:
            lines.append(f"[omitted {len(self.omitted_information)} spans — see package]")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [f"# Context: {self.query}", ""]
        if self.intent:
            lines.append(f"**Intent:** `{self.intent}`")
        if self.relevant_ranges:
            ranges = ", ".join(
                f"{format_timecode(r['start'])} → {format_timecode(r['end'])}"
                for r in self.relevant_ranges)
            lines.append(f"**Ranges:** {ranges}")
        lines.append("")
        lines.append("## Evidence")
        for i, span in enumerate(self.evidence, 1):
            tc = span.timecode if hasattr(span, "timecode") else "?"
            mod = getattr(span, "modality", "?")
            text = getattr(span, "text", str(span))[:300]
            reason = getattr(span, "reason", "")
            lines.append(f"{i}. **{tc}** ({mod}) — {text}")
            if reason:
                lines.append(f"   - *why:* {reason}")
        if self.entities:
            lines.append("")
            lines.append("## Entities")
            for entity in self.entities:
                lines.append(f"- **{entity.get('name')}** "
                             f"({entity.get('type')}) — first seen "
                             f"{entity.get('first_seen', '?')}")
        if self.warnings:
            lines.append("")
            lines.append("## Warnings")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


def optimize_evidence(spans: list[Any], *, max_spans: int | None = None) -> tuple[
        list[Any], list[dict[str, Any]]]:
    """Dedupe, drop contained redundancies, enforce the span cap.

    Anchor preservation: modalities present in the input keep at least one span
    whenever the cap allows; anything dropped is recorded in ``omitted``.
    """
    from .routing import dedupe_spans

    spans = dedupe_spans(list(spans))
    # Drop spans fully contained in a higher-score span of the same modality.
    kept: list[Any] = []
    for span in sorted(spans, key=lambda s: (-s.score, s.start)):
        if any(other.modality == span.modality and other.start <= span.start
               and span.end <= other.end and other is not span for other in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: (-s.score, s.start))
    omitted: list[dict[str, Any]] = []
    if max_spans is not None and len(kept) > max_spans:
        by_modality: dict[str, list[Any]] = {}
        for span in kept:
            by_modality.setdefault(span.modality, []).append(span)
        # One anchor per input modality first, then fill by score.
        anchors = [group[0] for group in by_modality.values()]
        rest = sorted((s for group in by_modality.values() for s in group[1:]),
                      key=lambda s: (-s.score, s.start))
        final = anchors + [s for s in rest if s not in anchors]
        dropped = final[max_spans:]
        kept = final[:max_spans]
        omitted = [{"ref_ids": list(s.ref_ids), "why": f"max_spans={max_spans}"}
                   for s in dropped]
    return kept, omitted


def optimize_frames(frames: list[dict[str, Any]], *,
                    max_frames: int | None = None) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop near-duplicate frames (same reason within 1 s, lower score first)."""
    ordered = sorted(frames, key=lambda f: (-f.get("score", 0.0), f.get("ts", 0.0)))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for frame in ordered:
        if any(other.get("reason") == frame.get("reason")
               and abs(other.get("ts", 0.0) - frame.get("ts", 0.0)) < 1.0
               for other in kept):
            dropped.append({"id": frame.get("id"), "why": "near-duplicate frame"})
            continue
        kept.append(frame)
    if max_frames is not None and len(kept) > max_frames:
        dropped.extend({"id": f.get("id"), "why": f"max_frames={max_frames}"}
                       for f in kept[max_frames:])
        kept = kept[:max_frames]
    return kept, dropped


def relevant_ranges_for(spans: list[Any], duration: float,
                        *, radius_s: float = 5.0) -> list[dict[str, Any]]:
    """Merged temporal windows covering the evidence — 'where to look'."""
    windows = merge_windows([window_around(s.start, s.end, radius_s, duration)
                             for s in spans])
    return [{"start": round(w.start, 3), "end": round(w.end, 3),
             "label": w.label} for w in windows]


def build_package(doc: Any, query: str, spans: list[Any], *,
                  plan: dict[str, Any] | None = None,
                  intent: str | None = None,
                  frames: list[dict[str, Any]] | None = None,
                  entities: list[dict[str, Any]] | None = None,
                  events: list[dict[str, Any]] | None = None,
                  changes: list[dict[str, Any]] | None = None,
                  temporal_relations: list[dict[str, Any]] | None = None,
                  graph_summary: dict[str, Any] | None = None,
                  budget: dict[str, Any] | None = None,
                  max_spans: int | None = None,
                  max_frames: int | None = None,
                  warnings: list[str] | None = None) -> ContextPackage:
    """Assemble an optimized, budgeted, provenance-preserving package."""
    evidence, omitted_spans = optimize_evidence(spans, max_spans=max_spans)
    kept_frames, omitted_frames = optimize_frames(frames or [], max_frames=max_frames)
    omitted = omitted_spans + omitted_frames
    video = getattr(doc, "video", None)
    source = getattr(doc, "source", None)
    package = ContextPackage(
        query=query,
        evidence=evidence,
        intent=intent,
        query_plan=plan,
        metadata={
            "video_id": getattr(doc, "id", None),
            "duration_s": getattr(video, "duration", None),
            "vctx_version": getattr(doc, "vctx_version", None),
        },
        relevant_ranges=relevant_ranges_for(
            evidence, float(getattr(video, "duration", 0.0) or 0.0)),
        entities=entities or [],
        events=events or [],
        changes=changes or [],
        temporal_relations=temporal_relations or [],
        frames=kept_frames,
        graph_summary=graph_summary or {},
        provenance={
            "producer": getattr(getattr(doc, "producer", None), "model_dump",
                                lambda **_: None)(mode="json")
            if getattr(doc, "producer", None) else None,
            "source": source.model_dump(mode="json") if source is not None else None,
            "stages": [(s.name, s.status.value if hasattr(s.status, "value") else s.status)
                       for s in getattr(doc, "stages", [])],
        },
        warnings=warnings or [],
        budget=budget or {},
        omitted_information=omitted,
    )
    return package


# ---------------------------------------------------------------------------
# redaction — opt-in, pattern-based, non-destructive
# ---------------------------------------------------------------------------


def builtin_secret_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Common secret shapes. Opt-in: pass to :func:`redact_package` explicitly."""
    return [
        ("api_key", re.compile(r"\b(sk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})\b")),
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
        ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ]


def redact_text(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> str:
    redacted = text
    for _name, pattern in patterns:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_package(package: ContextPackage,
                   patterns: list[tuple[str, re.Pattern[str]]]) -> ContextPackage:
    """Return a redacted copy: evidence/element strings scrubbed, originals kept.

    The input package is never mutated; ``warnings`` records that redaction ran.
    """
    import copy

    clone = copy.deepcopy(package)
    for span in clone.evidence:
        if hasattr(span, "text"):
            try:
                span.text = redact_text(span.text, patterns)
            except (AttributeError, TypeError):
                pass
    for element_list in (clone.entities, clone.events, clone.changes):
        for item in element_list:
            if isinstance(item, dict):
                for key in ("name", "text", "description", "before", "after"):
                    if isinstance(item.get(key), str):
                        item[key] = redact_text(item[key], patterns)
    clone.warnings = [*clone.warnings,
                      f"redacted with {len(patterns)} pattern(s); originals preserved"]
    return clone


__all__ = ["PACKAGE_SCHEMA_VERSION", "ContextPackage", "build_package", "builtin_secret_patterns",
           "optimize_evidence", "optimize_frames", "redact_package", "redact_text",
           "relevant_ranges_for"]
