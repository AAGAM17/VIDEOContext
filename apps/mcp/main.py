"""VideoContext MCP Server.

Exposes video querying capabilities to AI agents via the Model Context Protocol.

Tools:
- inspect_video: Describe a processed video (metadata, modalities, counts)
- search_video: Search across all modalities
- search_transcript: Search speech only
- search_ocr: Search on-screen text only
- find_event: Find events by type
- find_object: Find detected objects
- get_entities: List timestamp-grounded entities with uncertainty
- find_changes: Show what changed between adjacent regions
- get_chapters: List extractive chapters (titles marked as derived)
- get_segment: Get a segment by ID
- get_frame: Get a frame by ID
- get_timeline: Get timeline entries
- ask_video: Ask a question about the video
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from videocontent import load
from videocontent.collection import CollectionIndex, compare
from videocontent.schema.v1 import VideoContextDocument
from videocontent.sdk import Video
from videocontent.timecode import format_timecode

# Agent-facing caps: tools return minimal relevant context, never whole documents.
_MAX_ITEMS = 30
_MAX_TEXT = 200

# Global document cache
_docs: dict[str, VideoContextDocument] = {}

# Named multi-video collections: id -> video ids. Timestamps stay video-local.
_collections: dict[str, list[str]] = {}


def _get_doc(video_id: str) -> VideoContextDocument:
    """Get a document by video ID, loading if necessary."""
    if video_id not in _docs:
        # Try to find .vctx file
        search_paths = [
            Path(f"/tmp/videocontent_outputs/{video_id}.vctx"),
            Path(f"./.videocontent/{video_id}.vctx"),
            Path(f"{video_id}.vctx"),
        ]
        for path in search_paths:
            if path.exists():
                _docs[video_id] = load(path)
                break
        else:
            raise ValueError(f"Video {video_id} not found. Process it first.")
    return _docs[video_id]


def _format_span(span) -> str:
    """Format a search span for display."""
    return f"[{span.timecode}] ({span.modality}) {span.text[:_MAX_TEXT]}"


def _bounded(value: int, default: int) -> int:
    """Clamp agent-supplied limits so one call cannot dump the whole document."""
    try:
        return max(1, min(int(value), _MAX_ITEMS))
    except (TypeError, ValueError):
        return default


def _gap_of(span, word: str) -> float | None:
    """Gap seconds parsed from a before/after reason; None when absent."""
    match = re.search(r"(ends|starts) ([\d.]+)s (before|after)", span.reason or "")
    if match and match.group(3) == word:
        try:
            return float(match.group(2))
        except ValueError:
            return None
    return None


async def main():
    server = Server("videocontent")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="inspect_video",
                description="Describe a processed video: duration, modalities available, "
                            "fact counts, processing stages. Call this first.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                    },
                    "required": ["video_id"],
                },
            ),
            Tool(
                name="search_video",
                description="Search video content across all modalities (speech, on-screen text, vision, events)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 10, "description": "Max results"},
                        "modalities": {"type": "array", "items": {"type": "string"}, "description": "Modalities to search"},
                    },
                    "required": ["video_id", "query"],
                },
            ),
            Tool(
                name="search_transcript",
                description="Search speech transcript only",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["video_id", "query"],
                },
            ),
            Tool(
                name="search_ocr",
                description="Search on-screen text (OCR) only",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["video_id", "query"],
                },
            ),
            Tool(
                name="find_event",
                description="Find events by type (scene_changed, slide_changed, text_appeared, etc.)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "event_type": {"type": "string", "description": "Event type to find"},
                    },
                    "required": ["video_id", "event_type"],
                },
            ),
            Tool(
                name="find_object",
                description="Find detected objects by label",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "label": {"type": "string", "description": "Object label to find"},
                    },
                    "required": ["video_id", "label"],
                },
            ),
            Tool(
                name="get_entities",
                description="List timestamp-grounded entities (errors, commands, linked "
                            "concepts) with occurrence timestamps and uncertainty flags",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "top_k": {"type": "integer", "default": 20,
                                  "description": "Max entities"},
                    },
                    "required": ["video_id"],
                },
            ),
            Tool(
                name="find_changes",
                description="Show what changed between adjacent regions: text turnover, "
                            "speech transitions, scene boundaries — with evidence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                    },
                    "required": ["video_id"],
                },
            ),
            Tool(
                name="get_chapters",
                description="List extractive chapters. Titles are derived keywords "
                            "(marked as such), ranges are factual.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                    },
                    "required": ["video_id"],
                },
            ),
            Tool(
                name="get_segment",
                description="Get a segment by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "segment_id": {"type": "string", "description": "Segment ID"},
                    },
                    "required": ["video_id", "segment_id"],
                },
            ),
            Tool(
                name="get_frame",
                description="Get a frame by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "frame_id": {"type": "string", "description": "Frame ID"},
                    },
                    "required": ["video_id", "frame_id"],
                },
            ),
            Tool(
                name="get_timeline",
                description="Get timeline entries for a time range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "start": {"type": "number", "description": "Start time (seconds)"},
                        "end": {"type": "number", "description": "End time (seconds)"},
                    },
                    "required": ["video_id"],
                },
            ),
            Tool(
                name="ask_video",
                description="Ask a question about the video using retrieved evidence + LLM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "question": {"type": "string", "description": "Question to ask"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["video_id", "question"],
                },
            ),
            Tool(
                name="get_video_context",
                description="Get optimized AI context for a specific task (e.g., 'recreate the design', 'understand the application')",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "task": {"type": "string", "description": "Task description (e.g., 'recreate the design language', 'understand the application')"},
                        "max_tokens": {"type": "integer", "default": 4000, "description": "Max tokens for context"},
                    },
                    "required": ["video_id", "task"],
                },
            ),
            Tool(
                name="get_video_profile",
                description="Get a specific semantic profile (ui_design, application, product_demo, tutorial)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "profile_name": {"type": "string", "description": "Profile name (ui_design, application, product_demo, tutorial)"},
                    },
                    "required": ["video_id", "profile_name"],
                },
            ),
            Tool(
                name="list_video_profiles",
                description="List all available semantic profiles for a video",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                    },
                    "required": ["video_id"],
                },
            ),
            Tool(
                name="get_entity_timeline",
                description="Every occurrence of an entity in time order, with evidence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "entity": {"type": "string", "description": "Entity name"},
                    },
                    "required": ["video_id", "entity"],
                },
            ),
            Tool(
                name="get_events",
                description="Events in a time range, optionally filtered by type",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "event_type": {"type": "string", "description": "Only this type"},
                        "start": {"type": "number", "description": "Range start (seconds)"},
                        "end": {"type": "number", "description": "Range end (seconds)"},
                        "top_k": {"type": "integer", "default": 30},
                    },
                    "required": ["video_id"],
                },
            ),
            Tool(
                name="get_evidence",
                description="Exact facts behind reference IDs (inspect search hits)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "refs": {"type": "array", "items": {"type": "string"},
                                 "description": "Fact IDs, e.g. from span ref_ids"},
                    },
                    "required": ["video_id", "refs"],
                },
            ),
            Tool(
                name="get_context",
                description="Budgeted context package for a task (markdown or json)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "task": {"type": "string", "description": "Task description"},
                        "max_spans": {"type": "integer", "default": 5},
                        "max_tokens": {"type": "integer", "default": 2000},
                        "format": {"type": "string", "default": "markdown",
                                   "description": "markdown or json"},
                    },
                    "required": ["video_id", "task"],
                },
            ),
            Tool(
                name="explain_evidence",
                description="Why a node/fact is trusted: supporting evidence and relations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "ref": {"type": "string", "description": "Node or fact ID"},
                    },
                    "required": ["video_id", "ref"],
                },
            ),
            Tool(
                name="explain_relation",
                description="The construction rule behind a graph edge ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "edge_id": {"type": "string", "description": "Edge ID"},
                    },
                    "required": ["video_id", "edge_id"],
                },
            ),
            Tool(
                name="find_before",
                description="Evidence before an anchor phrase ('immediately' supported)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "anchor": {"type": "string", "description": "Anchor phrase"},
                        "max_gap_s": {"type": "number",
                                      "description": "Only this close (seconds)"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["video_id", "anchor"],
                },
            ),
            Tool(
                name="find_after",
                description="Evidence after an anchor phrase ('immediately' supported)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "Video ID"},
                        "anchor": {"type": "string", "description": "Anchor phrase"},
                        "max_gap_s": {"type": "number",
                                      "description": "Only this close (seconds)"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["video_id", "anchor"],
                },
            ),
            Tool(
                name="compare_videos",
                description="Added/removed/changed/unchanged entities plus structure",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "First video ID"},
                        "other_video_id": {"type": "string", "description": "Second video ID"},
                    },
                    "required": ["video_id", "other_video_id"],
                },
            ),
            Tool(
                name="register_collection",
                description="Name a set of videos for collection search (timestamps stay local)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_id": {"type": "string", "description": "Name to register"},
                        "video_ids": {"type": "array", "items": {"type": "string"},
                                      "description": "Video IDs (2+)"},
                    },
                    "required": ["collection_id", "video_ids"],
                },
            ),
            Tool(
                name="search_collection",
                description="Search across videos; every span keeps its video_id",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_id": {"type": "string",
                                         "description": "Registered collection"},
                        "video_ids": {"type": "array", "items": {"type": "string"},
                                      "description": "Or ad-hoc video IDs"},
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        if name == "register_collection":
            collection_id = arguments.get("collection_id", "")
            video_ids = arguments.get("video_ids", [])
            if not collection_id or len(video_ids) < 2:
                return CallToolResult(content=[TextContent(
                    type="text",
                    text="Error: collection_id and 2+ video_ids are required")])
            missing = [vid for vid in video_ids if vid not in _docs
                       and not any(p.exists() for p in (
                           Path(f"/tmp/videocontent_outputs/{vid}.vctx"),
                           Path(f"./.videocontent/{vid}.vctx"),
                           Path(f"{vid}.vctx")))]
            _collections[collection_id] = list(video_ids)
            note = f" (unloaded: {', '.join(missing)} — process them first)" if missing else ""
            return CallToolResult(content=[TextContent(
                type="text",
                text=f"Registered collection {collection_id}: "
                     f"{', '.join(video_ids)}{note}")])

        if name == "search_collection":
            query = arguments.get("query", "")
            top_k = _bounded(arguments.get("top_k", 10), 10)
            collection_id = arguments.get("collection_id", "")
            video_ids = list(arguments.get("video_ids", [])
                             or _collections.get(collection_id, []))
            if len(video_ids) < 2:
                return CallToolResult(content=[TextContent(
                    type="text",
                    text="Error: give 2+ video_ids or a registered collection_id")])
            try:
                docs = {vid: _get_doc(vid) for vid in video_ids}
                result = CollectionIndex(docs).search(query, top_k=top_k)
                lines = [f"[{s.video_id} {s.timecode}] ({s.modality}) "
                         f"{s.text[:_MAX_TEXT]}" for s in result.spans]
                return CallToolResult(content=[TextContent(
                    type="text",
                    text=f"Found {result.total} spans across "
                         f"{result.videos_searched} videos:\n" + "\n".join(lines))])
            except Exception as exc:
                return CallToolResult(content=[TextContent(
                    type="text", text=f"Error: {type(exc).__name__}: {exc}")])

        video_id = arguments.get("video_id")
        if not video_id:
            return CallToolResult(content=[TextContent(type="text", text="Error: video_id is required")])

        try:
            doc = _get_doc(video_id)
            video = Video.from_document(doc)

            if name == "inspect_video":
                receipt = video.receipt()
                lines = [
                    f"duration: {doc.video.duration:.1f}s",
                    f"modalities: " + ", ".join(
                        f"{k}={v}" for k, v in receipt["modalities"].items()),
                    "stages: " + ", ".join(
                        f"{s['name']}:{s['status']}" for s in receipt["stages"]),
                ]
                return CallToolResult(content=[TextContent(
                    type="text", text="Video context:\n" + "\n".join(lines))])

            if name == "search_video":
                query = arguments.get("query", "")
                top_k = _bounded(arguments.get("top_k", 10), 10)
                modalities = arguments.get("modalities")
                result = video.search(query, modalities=modalities, top_k=top_k)
                hits = "\n".join(_format_span(h) for h in result.spans)
                return CallToolResult(content=[TextContent(type="text", text=f"Found {result.total} matches:\n{hits}")])

            elif name == "search_transcript":
                query = arguments.get("query", "")
                top_k = _bounded(arguments.get("top_k", 10), 10)
                result = video.search(query, modalities=["transcript"], top_k=top_k)
                hits = "\n".join(_format_span(h) for h in result.spans)
                return CallToolResult(content=[TextContent(type="text", text=f"Found {result.total} transcript matches:\n{hits}")])

            elif name == "search_ocr":
                query = arguments.get("query", "")
                top_k = _bounded(arguments.get("top_k", 10), 10)
                result = video.search(query, modalities=["ocr"], top_k=top_k)
                hits = "\n".join(_format_span(h) for h in result.spans)
                return CallToolResult(content=[TextContent(type="text", text=f"Found {result.total} OCR matches:\n{hits}")])

            elif name == "find_event":
                event_type = arguments.get("event_type", "")
                events = [e for e in doc.events if e.type == event_type]
                if not events:
                    return CallToolResult(content=[TextContent(type="text", text=f"No events of type '{event_type}' found")])
                lines = [f"[{format_timecode(e.start)}] {e.description or e.type}"
                         for e in events[:_MAX_ITEMS]]
                return CallToolResult(content=[TextContent(type="text", text=f"Found {len(events)} events:\n" + "\n".join(lines))])

            elif name == "find_object":
                label = arguments.get("label", "").lower()
                objects = [o for o in doc.objects if label in o.label.lower()]
                if not objects:
                    return CallToolResult(content=[TextContent(type="text", text=f"No objects with label '{label}' found")])
                lines = [f"[{o.start:.2f}-{o.end:.2f}] {o.label} (conf: {o.confidence:.2f})" for o in objects]
                return CallToolResult(content=[TextContent(type="text", text=f"Found {len(objects)} objects:\n" + "\n".join(lines))])

            elif name == "get_segment":
                segment_id = arguments.get("segment_id", "")
                segment = next((s for s in doc.segments if s.id == segment_id), None)
                if not segment:
                    return CallToolResult(content=[TextContent(type="text", text=f"Segment {segment_id} not found")])
                return CallToolResult(content=[TextContent(type="text", text=json.dumps({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "transcript_ids": segment.transcript_ids,
                    "ocr_ids": segment.ocr_ids,
                    "event_ids": segment.event_ids,
                }, indent=2))])

            elif name == "get_frame":
                frame_id = arguments.get("frame_id", "")
                frame = next((f for f in doc.frames if f.id == frame_id), None)
                if not frame:
                    return CallToolResult(content=[TextContent(type="text", text=f"Frame {frame_id} not found")])
                return CallToolResult(content=[TextContent(type="text", text=json.dumps({
                    "id": frame.id,
                    "ts": frame.ts,
                    "path": frame.path,
                    "reason": frame.reason,
                }, indent=2))])

            elif name == "get_timeline":
                start = arguments.get("start", 0)
                end = arguments.get("end", doc.video.duration)
                top_k = _bounded(arguments.get("top_k", 30), 30)
                result = video.timeline(start, end, top_k=top_k)
                entries = [_format_span(span) for span in result.spans]
                suffix = f"\n({result.total - len(result.spans)} more — narrow the range)" \
                    if result.total > len(result.spans) else ""
                return CallToolResult(content=[TextContent(type="text", text=f"Timeline ({start:.1f}-{end:.1f}s):\n" + "\n".join(entries) + suffix)])

            elif name == "get_entities":
                top_k = _bounded(arguments.get("top_k", 20), 20)
                entities = video.entities()[:top_k]
                if not entities:
                    return CallToolResult(content=[TextContent(type="text", text="No entities found")])
                lines = []
                for e in entities:
                    seen = format_timecode(e.first_seen) if e.first_seen is not None else "?"
                    flag = " (uncertain)" if e.ambiguous else ""
                    lines.append(f"{e.name} [{e.type}]{flag} first seen {seen} "
                                 f"×{len(e.occurrences)} [{','.join(e.linked_modalities)}]")
                return CallToolResult(content=[TextContent(type="text", text="Entities:\n" + "\n".join(lines))])

            elif name == "find_changes":
                changes = video.changes()[:_MAX_ITEMS]
                if not changes:
                    return CallToolResult(content=[TextContent(type="text", text="No changes detected")])
                lines = [f"[{format_timecode(c.ts)}] {c.change_type}: "
                         f"{c.before[:120]} → {c.after[:120]}" for c in changes]
                return CallToolResult(content=[TextContent(type="text", text="Changes:\n" + "\n".join(lines))])

            elif name == "get_chapters":
                chapters = video.chapters()
                if not chapters:
                    return CallToolResult(content=[TextContent(type="text", text="No chapters")])
                lines = [f"[{format_timecode(c.start)} → {format_timecode(c.end)}] "
                         f"{c.title} (derived title)" for c in chapters]
                return CallToolResult(content=[TextContent(type="text", text="Chapters:\n" + "\n".join(lines))])

            elif name == "ask_video":
                question = arguments.get("question", "")
                top_k = _bounded(arguments.get("top_k", 5), 5)
                answer = video.ask(question, top_k=top_k)
                evidence = "\n".join(_format_span(e) for e in answer.evidence)
                return CallToolResult(content=[TextContent(type="text", text=f"Q: {answer.question}\nA: {answer.answer}\nConfidence: {answer.confidence:.0%}\n\nEvidence:\n{evidence}")])

            elif name == "get_video_context":
                task = arguments.get("task", "")
                max_tokens = arguments.get("max_tokens", 4000)
                context = video.context_for(task, max_tokens=max_tokens)
                evidence = "\n".join(_format_span(e) for e in context.evidence)
                profiles = ", ".join(context.profiles.keys()) if context.profiles else "none"
                return CallToolResult(content=[TextContent(type="text", text=f"Task: {context.task}\nType: {context.task_type.value if hasattr(context.task_type, 'value') else context.task_type}\nToken estimate: {context.token_estimate}\nProfiles: {profiles}\n\nContext:\n{context.context}")])

            elif name == "get_video_profile":
                profile_name = arguments.get("profile_name", "")
                try:
                    profile = video.profile(profile_name)
                    if hasattr(profile, "model_dump"):
                        profile_data = profile.model_dump()
                    else:
                        profile_data = str(profile)
                    return CallToolResult(content=[TextContent(type="text", text=json.dumps(profile_data, indent=2, default=str))])
                except ValueError as e:
                    return CallToolResult(content=[TextContent(type="text", text=f"Error: {e}")])

            elif name == "list_video_profiles":
                profiles = video.profiles()
                names = list(profiles.keys())
                return CallToolResult(content=[TextContent(type="text", text=f"Available profiles: {', '.join(names) if names else 'none'}")])

            elif name == "get_entity_timeline":
                entity = arguments.get("entity", "")
                timeline = video.entity_timeline(entity)
                if timeline is None:
                    return CallToolResult(content=[TextContent(type="text", text=f"No entity named '{entity}'")])
                lines = [f"[{o.start:.1f}] ({o.modality}) {o.text[:_MAX_TEXT]}"
                         for o in timeline.occurrences[:_MAX_ITEMS]]
                return CallToolResult(content=[TextContent(
                    type="text",
                    text=f"{timeline.entity.name} [{timeline.entity.type}] "
                         f"×{len(timeline.occurrences)}:\n" + "\n".join(lines))])

            elif name == "get_events":
                event_type = arguments.get("event_type")
                start = arguments.get("start", 0.0)
                end = arguments.get("end", doc.video.duration)
                top_k = _bounded(arguments.get("top_k", 30), 30)
                try:
                    start, end = float(start), float(end)
                except (TypeError, ValueError):
                    return CallToolResult(content=[TextContent(
                        type="text", text="Error: start/end must be numbers")])
                events = [e for e in doc.events
                          if (event_type is None or e.type == event_type)
                          and e.start < end and start < e.end]
                events.sort(key=lambda e: (e.start, e.end))
                lines = [f"[{format_timecode(e.start)}] {e.type}: "
                         f"{(e.description or '')[:_MAX_TEXT]}" for e in events[:top_k]]
                suffix = f"\n({len(events) - top_k} more)" if len(events) > top_k else ""
                return CallToolResult(content=[TextContent(
                    type="text", text=f"Events ({start:.1f}-{end:.1f}s):\n"
                    + "\n".join(lines) + suffix if lines else "No events in range")])

            elif name == "get_evidence":
                refs = arguments.get("refs", [])
                if not isinstance(refs, list) or not refs:
                    return CallToolResult(content=[TextContent(
                        type="text", text="Error: refs must be a non-empty list of IDs")])
                found = doc.by_id if hasattr(doc, "by_id") else None
                lines = []
                for ref in refs[:_MAX_ITEMS]:
                    item = found(ref) if found else None
                    if item is None:
                        lines.append(f"{ref}: not found")
                        continue
                    text = getattr(item, "text", None) or getattr(item, "description",
                                                                   None) or ""
                    lines.append(f"{ref} [{getattr(item, 'start', '?')}] {text[:_MAX_TEXT]}")
                return CallToolResult(content=[TextContent(
                    type="text", text="Evidence:\n" + "\n".join(lines))])

            elif name == "get_context":
                task = arguments.get("task", "")
                max_spans = _bounded(arguments.get("max_spans", 5), 5)
                try:
                    max_tokens = max(256, min(int(arguments.get("max_tokens", 2000)), 8000))
                except (TypeError, ValueError):
                    max_tokens = 2000
                package = video.context_package(task, max_spans=max_spans,
                                                max_tokens=max_tokens)
                fmt = arguments.get("format", "markdown")
                body = package.to_json() if fmt == "json" else package.to_markdown()
                return CallToolResult(content=[TextContent(type="text", text=body)])

            elif name == "explain_evidence":
                ref = arguments.get("ref", "")
                explanation = video.explain(ref)
                if explanation is None:
                    return CallToolResult(content=[TextContent(
                        type="text", text=f"No node or fact {ref!r}")])
                if "rule" in explanation:
                    return CallToolResult(content=[TextContent(
                        type="text",
                        text=f"{explanation['relation']}: {explanation['source']} → "
                             f"{explanation['target']}\nRule: {explanation['rule']}")])
                node = explanation["node"]
                support = "\n".join(
                    f"[{s['start']:.1f}] ({s['kind']}) {s['label'][:_MAX_TEXT]}"
                    for s in explanation["supporting_evidence"][:10])
                return CallToolResult(content=[TextContent(
                    type="text",
                    text=f"{node['id']} ({node['kind']}): {node['label'][:_MAX_TEXT]}\n"
                         f"Observed fact: {node['observed']}\nSupporting evidence:\n{support}")])

            elif name == "explain_relation":
                edge_id = arguments.get("edge_id", "")
                graph = video.graph()
                explanation = graph.explain(edge_id)
                if explanation is None:
                    return CallToolResult(content=[TextContent(
                        type="text", text=f"No edge {edge_id!r}")])
                return CallToolResult(content=[TextContent(
                    type="text",
                    text=f"{explanation['relation']}: {explanation['source']} → "
                         f"{explanation['target']}\nRule: {explanation['rule']}\n"
                         f"Provenance: {', '.join(explanation['provenance'])}")])

            elif name in ("find_before", "find_after"):
                anchor = arguments.get("anchor", "")
                top_k = _bounded(arguments.get("top_k", 5), 5)
                try:
                    max_gap = (None if arguments.get("max_gap_s") is None
                               else float(arguments.get("max_gap_s")))
                except (TypeError, ValueError):
                    return CallToolResult(content=[TextContent(
                        type="text", text="Error: max_gap_s must be a number")])
                word = "before" if name == "find_before" else "after"
                _, result = video.retriever.query_temporal(
                    f"what happened {word} {anchor}", top_k=top_k)
                if max_gap is not None:
                    filtered = [s for s in result.spans
                                if _gap_of(s, word) is not None and _gap_of(s, word) <= max_gap]
                else:
                    filtered = list(result.spans)
                hits = "\n".join(_format_span(h) for h in filtered)
                return CallToolResult(content=[TextContent(
                    type="text",
                    text=f"{word.capitalize()} '{anchor}':\n{hits}" if hits
                    else f"Nothing found {word} '{anchor}'")])

            elif name == "compare_videos":
                other_id = arguments.get("other_video_id", "")
                if not other_id:
                    return CallToolResult(content=[TextContent(
                        type="text", text="Error: other_video_id is required")])
                try:
                    other_doc = _get_doc(other_id)
                except ValueError as exc:
                    return CallToolResult(content=[TextContent(
                        type="text", text=f"Error: {exc}")])
                result = compare(video_id, doc, other_id, other_doc)
                lines = [f"added in {other_id}: {', '.join(result.added[:10]) or '—'}",
                         f"removed: {', '.join(result.removed[:10]) or '—'}",
                         f"changed: {', '.join(result.changed[:10]) or '—'}",
                         f"unchanged: {len(result.unchanged)}",
                         f"uncertain: {', '.join(result.uncertain[:10]) or '—'}"]
                return CallToolResult(content=[TextContent(
                    type="text", text="Comparison:\n" + "\n".join(lines))])

            else:
                return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])

        except Exception as exc:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: {type(exc).__name__}: {exc}")])

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())