"""Context Router — Task Classification and Context Selection.

This module implements the brain of the multi-resolution system:
- Classifies user tasks/queries into types
- Selects optimal context representation
- Applies token budgeting
- Packs context for LLM consumption
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..schema.v1 import VideoContextDocument, TimeSpan
from ..profiles import list_profiles, get_profile_builder, ProfileContext
from ..retrieval import search, SearchResult, EvidenceSpan


class TaskType(str, Enum):
    """Classification of user tasks."""
    FACTUAL_RETRIEVAL = "factual_retrieval"
    TEMPORAL_RETRIEVAL = "temporal_retrieval"
    GLOBAL_UNDERSTANDING = "global_understanding"
    DESIGN_ANALYSIS = "design_analysis"
    UI_RECREATION = "ui_recreation"
    APPLICATION_UNDERSTANDING = "application_understanding"
    PRODUCT_ANALYSIS = "product_analysis"
    INTERACTION_ANALYSIS = "interaction_analysis"
    VISUAL_STYLE_ANALYSIS = "visual_style_analysis"
    SUMMARY = "summary"
    CUSTOM = "custom"


@dataclass
class TaskClassification:
    """Result of task classification."""
    task_type: TaskType
    confidence: float
    suggested_profiles: list[str]
    requires_evidence: bool
    requires_frames: bool
    requires_global: bool
    reasoning: str


@dataclass
class ContextBudget:
    """Token budget for context selection."""
    max_tokens: int = 4000
    reserve_tokens: int = 500  # For system prompt, etc.

    @property
    def available_tokens(self) -> int:
        return max(0, self.max_tokens - self.reserve_tokens)


@dataclass
class ContextSelection:
    """Selected context for a task."""
    task_type: TaskType
    profiles: dict[str, Any] = field(default_factory=dict)
    evidence: list[Any] = field(default_factory=list)
    frames: list[Any] = field(default_factory=list)
    global_context: Any | None = None
    summaries: dict[str, str] = field(default_factory=dict)
    token_estimate: int = 0
    compression_stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task Classification
# ---------------------------------------------------------------------------


def classify_task(query: str, doc: VideoContextDocument | None = None) -> TaskClassification:
    """Classify a user query/task into a task type."""
    query_lower = query.lower()

    # Keyword-based classification (can be enhanced with ML later)
    # Rules are checked in order; more specific patterns should come first
    classification_rules = [
        # Summary - check first for explicit summary requests
        (["summarize", "summary", "tl;dr", "tldr", "brief", "short"],
         TaskType.SUMMARY, ["requires_global", "requires_summary"]),

        # UI Recreation / Design Analysis - very specific patterns
        (["recreate the design", "replicate the design", "clone the design",
          "copy the design", "build the same design", "design language",
          "visual style", "aesthetic", "recreate this", "build this"],
         TaskType.UI_RECREATION, ["ui_design", "requires_frames", "requires_evidence"]),

        # Design Analysis
        (["design", "style", "look", "visual", "aesthetic", "recreate", "replicate",
          "build", "clone", "copy", "similar", "color", "typography", "layout",
          "component", "ui", "interface", "website", "website design", "web design"],
         TaskType.DESIGN_ANALYSIS, ["ui_design", "requires_frames", "requires_evidence"]),

        # Application Understanding
        (["how does", "how it works", "how the app works", "how the application works",
          "application", "app", "software", "workflow", "user flow", "screen",
          "navigation", "feature", "functionality", "architecture"],
         TaskType.APPLICATION_UNDERSTANDING, ["application", "requires_evidence"]),

        # Product Analysis
        (["product", "demo", "demonstration", "showcase", "feature", "use case",
          "walkthrough", "capability", "product overview"],
         TaskType.PRODUCT_ANALYSIS, ["product_demo", "requires_evidence"]),

        # Interaction Analysis
        (["interaction", "animation", "transition", "motion", "animate",
          "click", "hover", "scroll", "drag", "transition", "micro-interaction"],
         TaskType.INTERACTION_ANALYSIS, ["ui_design", "requires_frames"]),

        # Visual Style Analysis
        (["color", "theme", "dark mode", "light mode", "branding", "visual identity",
          "style guide", "design system", "colors", "theme"],
         TaskType.VISUAL_STYLE_ANALYSIS, ["ui_design"]),

        # Temporal Retrieval - specific temporal queries
        (["at what time", "timestamp", "at minute", "at second",
          "during", "between", "from", "to"],
         TaskType.TEMPORAL_RETRIEVAL, ["requires_evidence"]),

        # When - could be temporal or factual
        (["when was", "when did", "when does"],
         TaskType.TEMPORAL_RETRIEVAL, ["requires_evidence"]),

        # Global Understanding - broad questions
        (["what is this video about", "what is this video", "what happens in this video",
          "what happened in this video", "main topic", "general", "big picture",
          "what is this about", "video about"],
         TaskType.GLOBAL_UNDERSTANDING, ["requires_global", "requires_summary"]),

        # Factual Retrieval - specific facts, numbers, quotes (more specific first)
        (["what was the revenue", "what was the price", "what was the cost",
          "what is the revenue", "what is the price", "what is the cost",
          "how much", "how many", "revenue", "price", "cost", "number",
          "figure", "statistic", "metric", "what was", "what is"],
         TaskType.FACTUAL_RETRIEVAL, ["requires_evidence"]),

        # General "when" - temporal
        (["when",],
         TaskType.TEMPORAL_RETRIEVAL, ["requires_evidence"]),
    ]

    # Score each rule
    scores: dict[TaskType, float] = {}
    matched_keywords: dict[TaskType, list[str]] = {}

    for keywords, task_type, flags in classification_rules:
        score = 0
        matched = []
        for kw in keywords:
            if kw in query.lower():
                score += 1
                matched.append(kw)

        if score > 0:
            scores[task_type] = scores.get(task_type, 0) + score
            matched_keywords[task_type] = matched

    # Default to global understanding if no match
    if not scores:
        return TaskClassification(
            task_type=TaskType.GLOBAL_UNDERSTANDING,
            confidence=0.3,
            suggested_profiles=[],
            requires_evidence=True,
            requires_frames=False,
            requires_global=True,
            reasoning="No specific task pattern matched; defaulting to global understanding",
        )

    # Get top task type
    top_task = max(scores, key=scores.get)
    confidence = min(1.0, scores[top_task] / 5.0)

    # Determine requirements
    requires_evidence = False
    requires_frames = False
    requires_global = False
    suggested_profiles = []

    # Map task types to profiles and requirements
    task_config = {
        TaskType.FACTUAL_RETRIEVAL: {"evidence": True, "global": False, "frames": False, "profiles": []},
        TaskType.TEMPORAL_RETRIEVAL: {"evidence": True, "global": False, "frames": False, "profiles": []},
        TaskType.GLOBAL_UNDERSTANDING: {"evidence": False, "global": True, "frames": True, "profiles": ["ui_design", "application"]},
        TaskType.DESIGN_ANALYSIS: {"evidence": True, "global": True, "frames": True, "profiles": ["ui_design"]},
        TaskType.UI_RECREATION: {"evidence": True, "global": True, "frames": True, "profiles": ["ui_design"]},
        TaskType.APPLICATION_UNDERSTANDING: {"evidence": True, "global": True, "frames": False, "profiles": ["application"]},
        TaskType.PRODUCT_ANALYSIS: {"evidence": True, "global": True, "frames": True, "profiles": ["product_demo"]},
        TaskType.INTERACTION_ANALYSIS: {"evidence": True, "global": False, "frames": True, "profiles": ["ui_design"]},
        TaskType.VISUAL_STYLE_ANALYSIS: {"evidence": True, "global": False, "frames": True, "profiles": ["ui_design"]},
        TaskType.SUMMARY: {"evidence": False, "global": True, "frames": False, "profiles": []},
    }

    config = task_config.get(top_task, {"evidence": True, "global": True, "frames": True, "profiles": []})
    requires_evidence = config["evidence"]
    requires_frames = config["frames"]
    requires_global = config["global"]
    suggested_profiles = config["profiles"]

    return TaskClassification(
        task_type=top_task,
        confidence=confidence,
        suggested_profiles=suggested_profiles,
        requires_evidence=requires_evidence,
        requires_frames=requires_frames,
        requires_global=requires_global,
        reasoning=f"Matched keywords: {matched_keywords.get(top_task, [])}",
    )


# ---------------------------------------------------------------------------
# Context Selection
# ---------------------------------------------------------------------------


def select_context(
    doc: VideoContextDocument,
    task: TaskClassification,
    budget: ContextBudget,
    query: str | None = None,
) -> dict[str, Any]:
    """Select optimal context for the task within token budget."""
    from ..profiles import get_profile_builder, ProfileContext

    selection = {
        "profiles": {},
        "evidence": [],
        "frames": [],
        "global_context": None,
        "summaries": {},
        "token_estimate": 0,
    }

    profile_ctx = ProfileContext(doc=doc)

    # 1. Build requested profiles
    for profile_name in task.suggested_profiles:
        try:
            builder = get_profile_builder(profile_name)
            profile_ctx = ProfileContext(doc=doc)
            profile_result = builder.build(profile_ctx)
            selection["profiles"][profile_name] = profile_result
        except Exception:
            pass

    # 2. Get evidence if needed
    if task.requires_evidence and query:
        evidence_result: SearchResult = search(doc, query, top_k=10)
        selection["evidence"] = evidence_result.spans

    # 3. Select representative frames if needed
    if task.requires_frames:
        selection["frames"] = select_representative_frames(doc, max_frames=10)

    # 4. Get global context if needed
    if task.requires_global:
        selection["global_context"] = get_global_context(doc)

    # 5. Get multi-level summaries
    selection["summaries"] = get_summaries(doc)

    # 6. Estimate tokens and trim if needed
    selection["token_estimate"] = estimate_tokens(selection)
    if selection["token_estimate"] > budget.available_tokens:
        selection = trim_to_budget(selection, budget)

    return selection


def select_representative_frames(doc: VideoContextDocument, max_frames: int = 10) -> list[dict[str, Any]]:
    """Select the most representative frames."""
    if not doc.frames:
        return []

    # Score frames by:
    # - Scene change frames (highest)
    # - Frames with vision descriptions
    # - Frames with OCR
    # - Frames at event boundaries
    # - Visual uniqueness

    scored_frames = []
    for frame in doc.frames:
        score = 0.0

        if frame.reason == "scene_change":
            score += 10
        elif frame.reason == "boundary_burst":
            score += 8
        elif frame.reason == "motion":
            score += 5
        elif frame.reason == "ocr_density":
            score += 5
        elif frame.reason == "event":
            score += 7
        elif frame.reason == "keyframe":
            score += 6

        # Bonus for having vision description
        vision_notes = [v for v in doc.vision if frame.id in v.frame_ids]
        if vision_notes:
            score += 3

        # Bonus for having OCR
        ocr_notes = [o for o in doc.ocr if o.first_frame_ts and o.first_frame_ts <= frame.ts <= (o.last_frame_ts or float('inf'))]
        if ocr_notes:
            score += 2

        scored_frames.append((score, frame))

    # Sort by score descending
    scored_frames.sort(key=lambda x: x[0], reverse=True)

    # Select top frames with temporal diversity
    selected = []
    selected_times = []

    for score, frame in scored_frames:
        # Ensure temporal diversity (at least 5s apart)
        if not any(abs(frame.ts - t) < 5 for t in selected_times):
            selected.append({
                "id": frame.id,
                "ts": frame.ts,
                "path": frame.path,
                "reason": frame.reason,
                "score": score,
            })
            selected_times.append(frame.ts)

            if len(selected) >= max_frames:
                break

    return selected


def get_global_context(doc: VideoContextDocument) -> dict[str, Any] | None:
    """Get or generate global context."""
    if doc.global_context:
        return doc.global_context.model_dump()

    # Generate basic global context from existing data
    return {
        "summary": doc.global_context.summary if doc.global_context else None,
        "one_line": doc.global_context.one_line if doc.global_context else None,
        "domain": doc.global_context.domain if doc.global_context else None,
        "major_topics": doc.global_context.major_topics if doc.global_context else [],
    }


def get_summaries(doc: VideoContextDocument) -> dict[str, str]:
    """Get multi-level summaries."""
    if doc.context_summaries:
        return {
            "one_line": doc.context_summaries.one_line,
            "short": doc.context_summaries.short,
            "detailed": doc.context_summaries.detailed,
        }

    # Generate basic summaries
    return {
        "one_line": doc.global_context.one_line if doc.global_context else "Video content",
        "short": doc.global_context.summary if doc.global_context else "No summary available",
        "detailed": "Detailed summary not yet generated",
    }


def estimate_tokens(selection: dict[str, Any]) -> int:
    """Rough token estimation for selected context."""
    # Rough estimates
    tokens = 0

    # Profiles
    for profile_data in selection["profiles"].values():
        tokens += len(str(profile_data)) // 4

    # Evidence
    for span in selection["evidence"]:
        tokens += len(span.text) // 4
        tokens += 50  # metadata overhead

    # Frames
    tokens += len(selection["frames"]) * 200  # ~200 tokens per frame description

    # Global context
    if selection["global_context"]:
        tokens += len(str(selection["global_context"])) // 4

    # Summaries
    for summary in selection["summaries"].values():
        if summary:
            tokens += len(summary) // 4

    return tokens


def trim_to_budget(selection: dict[str, Any], budget: ContextBudget) -> dict[str, Any]:
    """Trim selection to fit within token budget."""
    # Priority order for trimming:
    # 1. Reduce evidence spans
    # 2. Reduce frames
    # 3. Use shorter summaries
    # 4. Trim profile detail

    while estimate_tokens(selection) > budget.available_tokens:
        # Try trimming evidence first
        if len(selection["evidence"]) > 3:
            selection["evidence"] = selection["evidence"][:3]
            continue

        # Then frames
        if len(selection["frames"]) > 3:
            selection["frames"] = selection["frames"][:3]
            continue

        # Use only one_line summary
        if "detailed" in selection["summaries"]:
            selection["summaries"]["detailed"] = None
            continue

        if "short" in selection["summaries"]:
            selection["summaries"]["short"] = None
            continue

        # Trim profiles
        for profile_name, profile_data in selection["profiles"].items():
            if hasattr(profile_data, "model_dump"):
                # For Pydantic models, we can't easily trim, so remove
                pass

        # If still over budget, remove frames entirely
        if len(selection["frames"]) > 0:
            selection["frames"] = []
            continue

        # Last resort: remove evidence
        selection["evidence"] = []
        break

    return selection


# ---------------------------------------------------------------------------
# Context Packing
# ---------------------------------------------------------------------------


def pack_context(selection: dict[str, Any], task_type: str, query: str | None = None) -> str:
    """Pack selected context into LLM-ready format."""
    parts = []

    # Task header
    parts.append(f"TASK: {task_type.replace('_', ' ').title()}")
    if query:
        parts.append(f"QUERY: {query}")
    parts.append("")

    # Global context first (high level)
    if selection["global_context"]:
        gc = selection["global_context"]
        if isinstance(gc, dict):
            parts.append("=== GLOBAL CONTEXT ===")
            if gc.get("one_line"):
                parts.append(f"One-line: {gc['one_line']}")
            if gc.get("summary"):
                parts.append(f"Summary: {gc['summary']}")
            if gc.get("domain"):
                parts.append(f"Domain: {gc['domain']}")
            if gc.get("major_topics"):
                parts.append(f"Topics: {', '.join(gc['major_topics'])}")
            parts.append("")

    # Profiles
    if selection["profiles"]:
        parts.append("=== SEMANTIC PROFILES ===")
        for name, profile in selection["profiles"].items():
            parts.append(f"--- {name.upper()} ---")
            parts.append(format_profile(profile))
            parts.append("")

    # Summaries
    if selection["summaries"]:
        parts.append("=== SUMMARIES ===")
        for level, summary in selection["summaries"].items():
            if summary:
                parts.append(f"{level}: {summary}")
        parts.append("")

    # Evidence
    if selection["evidence"]:
        parts.append("=== EVIDENCE (with timestamps) ===")
        for i, span in enumerate(selection["evidence"], 1):
            parts.append(f"[{i}] {span.timecode} ({span.modality}): {span.text[:300]}")
        parts.append("")

    # Frames
    if selection["frames"]:
        parts.append("=== REPRESENTATIVE FRAMES ===")
        for frame in selection["frames"]:
            parts.append(f"Frame {frame['id']} at {frame['ts']:.1f}s: {frame['reason']}")
        parts.append("")

    return "\n".join(parts)


def format_profile(profile: Any) -> str:
    """Format a profile object for display."""
    if hasattr(profile, "model_dump"):
        data = profile.model_dump()
    else:
        data = profile

    lines = []

    def format_dict(d: dict, indent: int = 0):
        prefix = "  " * indent
        for key, value in d.items():
            if value is None:
                continue
            if isinstance(value, list):
                if value:
                    lines.append(f"{prefix}{key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                format_dict(value, indent + 1)
            else:
                lines.append(f"{prefix}{key}: {value}")

    if hasattr(profile, "__dict__"):
        format_dict(profile.__dict__)
    elif isinstance(profile, dict):
        format_dict(profile)

    return "\n".join(lines)


__all__ = [
    "TaskType",
    "TaskClassification",
    "ContextBudget",
    "ContextSelection",
    "classify_task",
    "select_context",
    "select_representative_frames",
    "get_global_context",
    "get_summaries",
    "estimate_tokens",
    "trim_to_budget",
    "pack_context",
    "format_profile",
]