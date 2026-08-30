"""VideoContext MCP Server.

Exposes video querying capabilities to AI agents via the Model Context Protocol.

Tools:
- search_video: Search across all modalities
- search_transcript: Search speech only
- search_ocr: Search on-screen text only
- find_event: Find events by type
- find_object: Find detected objects
- get_segment: Get a segment by ID
- get_frame: Get a frame by ID
- get_timeline: Get timeline entries
- ask_video: Ask a question about the video
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from videocontext import load
from videocontext.schema.v1 import VideoContextDocument

# Global document cache
_docs: dict[str, VideoContextDocument] = {}


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
    return f"[{span.timecode}] ({span.modality}) {span.text[:200]}"


async def main():
    server = Server("videocontent")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
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
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        video_id = arguments.get("video_id")
        if not video_id:
            return CallToolResult(content=[TextContent(type="text", text="Error: video_id is required")])

        try:
            doc = _get_doc(video_id)
            video = load(doc=doc)

            if name == "search_video":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 10)
                modalities = arguments.get("modalities")
                result = video.search(query, modalities=modalities, top_k=top_k)
                hits = "\n".join(_format_span(h) for h in result.spans)
                return CallToolResult(content=[TextContent(type="text", text=f"Found {result.total} matches:\n{hits}")])

            elif name == "search_transcript":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 10)
                result = video.search(query, modalities=["transcript"], top_k=top_k)
                hits = "\n".join(_format_span(h) for h in result.spans)
                return CallToolResult(content=[TextContent(type="text", text=f"Found {result.total} transcript matches:\n{hits}")])

            elif name == "search_ocr":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 10)
                result = video.search(query, modalities=["ocr"], top_k=top_k)
                hits = "\n".join(_format_span(h) for h in result.spans)
                return CallToolResult(content=[TextContent(type="text", text=f"Found {result.total} OCR matches:\n{hits}")])

            elif name == "find_event":
                event_type = arguments.get("event_type", "")
                events = [e for e in doc.events if e.type == event_type]
                if not events:
                    return CallToolResult(content=[TextContent(type="text", text=f"No events of type '{event_type}' found")])
                lines = [f"[{e.timecode}] {e.description or e.type}" for e in events]
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
                timeline = video.at(start, window=0).spans if start == end else video.search("", start=start, end=end).spans
                entries = []
                for span in timeline:
                    entries.append(f"[{span.timecode}] ({span.modality}) {span.text[:200]}")
                return CallToolResult(content=[TextContent(type="text", text=f"Timeline ({start:.1f}-{end:.1f}s):\n" + "\n".join(entries))])

            elif name == "ask_video":
                question = arguments.get("question", "")
                top_k = arguments.get("top_k", 5)
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

            else:
                return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])

        except Exception as exc:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: {type(exc).__name__}: {exc}")])

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())