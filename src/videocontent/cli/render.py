"""Terminal rendering. Every function here takes data and returns a renderable; none of them
print, and none of them decide anything.

Two rules shape the output.

**Evidence is shown with its provenance.** A search result prints the fact ids and the segment
ids underneath the matched text, because a tool whose claim is "this timestamp is real" has to
make the reference checkable at the point where the user reads the answer, not in a ``--json``
flag they will never pass.

**Nothing about a stage is inferred from silence.** An empty OCR section can mean "no text on
screen" or "OCR never ran", and those are different facts about the video. The stage table
prints ``skipped`` and ``failed`` as loudly as ``ok``, and it prints ``local``/``remote`` per
stage because §32 requires a user to be able to see which stages sent data anywhere.
"""

from __future__ import annotations

import re
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..retrieval.query import SearchResult
from ..schema.v1 import StageStatus, VideoContextDocument
from ..timecode import format_span, format_timecode

#: Two console settings that are correctness, not taste.
#:
#: ``markup=False``: recognised text is arbitrary content from an untrusted file (§31), and rich
#: would read ``[INFO] starting`` in an OCR'd log line as a style tag and print ``starting``.
#: Everything styled here is therefore built as :class:`~rich.text.Text`, never interpolated
#: into a markup string.
#:
#: ``highlight=False``: rich would otherwise colour numbers and URLs *inside* OCR text and
#: transcript lines, which makes recognised content look like tool output.
console = Console(highlight=False, soft_wrap=False, markup=False)
errors = Console(stderr=True, highlight=False, style="red", markup=False)

_STATUS_STYLE = {
    StageStatus.OK: "green",
    StageStatus.PARTIAL: "yellow",
    StageStatus.SKIPPED: "dim",
    StageStatus.FAILED: "bold red",
}

_WHITESPACE = re.compile(r"\s+")


def shorten(text: str, width: int = 96) -> str:
    """Collapse whitespace and clip. OCR text arrives with newlines inside a single event."""
    flat = _WHITESPACE.sub(" ", text).strip()
    return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"


def _dim(*parts: str) -> Text:
    return Text(" · ".join(p for p in parts if p), style="dim")


def bold(text: str) -> Text:
    """Emphasis for callers, who cannot use ``[bold]`` markup — see the console note above."""
    return Text(text, style="bold")


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


# -- search ----------------------------------------------------------------


def spans_table(result: SearchResult) -> Table:
    """Ranked evidence, one row per span, provenance under each match."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=False)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("TIME", no_wrap=True, style="cyan")
    table.add_column("WHERE", no_wrap=True)
    table.add_column("SCORE", justify="right", no_wrap=True, style="dim")
    table.add_column("EVIDENCE", overflow="fold")

    for position, span in enumerate(result.spans, start=1):
        where = span.modality if not span.kind else f"{span.modality}/{span.kind}"
        evidence = Group(
            Text(shorten(span.text, 120) or "—"),
            _dim(span.reason, " ".join(span.ref_ids), " ".join(span.segment_ids)),
        )
        table.add_row(
            str(position),
            format_span(span.start, span.end, millis=True),
            where,
            f"{span.score:.4f}",
            evidence,
        )
    return table


def search_footer(result: SearchResult) -> Text:
    shown = len(result.spans)
    parts = [
        f"{shown} shown of {result.total} match{'' if result.total == 1 else 'es'}",
        f"{result.took_ms:.1f} ms",
        f"searched: {', '.join(result.modalities) or 'nothing'}",
    ]
    footer = Text(" · ".join(parts), style="dim")
    for note in result.notes:
        footer.append(f"\nnote: {note}", style="yellow")
    return footer


def snapshot_table(result: SearchResult) -> Table:
    """What the document knows about one instant. No scores — nothing here is ranked."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    table.add_column("TIME", no_wrap=True, style="cyan")
    table.add_column("WHERE", no_wrap=True)
    table.add_column("EVIDENCE", overflow="fold")
    for span in result.spans:
        where = span.modality if not span.kind else f"{span.modality}/{span.kind}"
        table.add_row(
            format_span(span.start, span.end),
            where,
            Group(Text(shorten(span.text, 120) or "—"), _dim(" ".join(span.ref_ids))),
        )
    return table


# -- document --------------------------------------------------------------


def overview(doc: VideoContextDocument) -> Panel:
    """Metadata a user needs before trusting anything else in the document."""
    video = doc.video
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(overflow="fold")

    geometry = (
        f"{video.width}x{video.height}" if video.width and video.height else "unknown size"
    )
    tracks = ", ".join(
        part
        for part in (
            f"video {video.video_codec}" if video.has_video else None,
            f"audio {video.audio_codec}" if video.has_audio else "no audio track",
        )
        if part
    )
    rows: list[tuple[str, str]] = [
        ("file", video.filename),
        ("duration", f"{format_timecode(video.duration)}  ({video.duration:.2f}s)"),
        ("media", f"{geometry} · {_num(video.fps, 3)} fps · {video.container or '?'} · {tracks}"),
        ("content hash", (video.content_hash or "—")[:16]),
        ("vctx", f"{doc.vctx_version} · {doc.producer.name} {doc.producer.version}"),
        ("created", doc.created_at.isoformat(timespec="seconds")),
        ("config", doc.producer.config_hash or "—"),
        ("processing", "local only" if not any(s.remote for s in doc.stages) else "REMOTE STAGES"),
    ]
    for label, value in rows:
        grid.add_row(label, value)
    return Panel(grid, title="video", title_align="left", box=box.ROUNDED, expand=False)


def counts_table(doc: VideoContextDocument) -> Table:
    """How much of each kind of fact the document holds, and whether anyone looked.

    The ``status`` column is the load-bearing one: ``0`` next to ``ok`` means the video has no
    on-screen text, and ``0`` next to ``skipped`` means nobody read it.
    """
    by_stage = {stage.name: stage for stage in doc.stages}
    sections: list[tuple[str, int, tuple[str, ...]]] = [
        ("scenes", len(doc.scenes), ("scenes", "scene_refine")),
        ("transcript", len(doc.transcript), ("asr",)),
        ("ocr", len(doc.ocr), ("ocr",)),
        ("vision", len(doc.vision), ("vision",)),
        ("objects", len(doc.objects), ("objects",)),
        ("events", len(doc.events), ("events",)),
        ("segments", len(doc.segments), ("segments",)),
        ("frames", len(doc.frames), ("frames",)),
    ]
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=False)
    table.add_column("SECTION", no_wrap=True)
    table.add_column("N", justify="right", no_wrap=True)
    table.add_column("STATUS", no_wrap=True)
    for label, count, stage_names in sections:
        stage = next((by_stage[n] for n in stage_names if n in by_stage), None)
        status = stage.status if stage else None
        table.add_row(
            label,
            str(count),
            Text(status.value if status else "not run", style=_STATUS_STYLE.get(status, "dim")),
        )
    return table


def stages_table(doc: VideoContextDocument) -> Table:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=False)
    table.add_column("STAGE", no_wrap=True)
    table.add_column("STATUS", no_wrap=True)
    table.add_column("PROVIDER", no_wrap=True)
    table.add_column("WHERE", no_wrap=True)
    table.add_column("TIME", justify="right", no_wrap=True, style="dim")
    table.add_column("NOTES", overflow="fold")

    for stage in doc.stages:
        notes = Text(no_wrap=False)
        if stage.cached:
            notes.append("cached")
        if stage.counts:
            _append(notes, " ".join(f"{k}={v}" for k, v in stage.counts.items()))
        if stage.error:
            _append(notes, shorten(stage.error, 80), style="red")
        for warning in stage.warnings:
            _append(notes, shorten(warning, 80), style="yellow")
        table.add_row(
            stage.name,
            Text(stage.status.value, style=_STATUS_STYLE.get(stage.status, "")),
            stage.provider or "—",
            Text("remote", style="yellow") if stage.remote else Text("local", style="green"),
            f"{stage.duration_s:.2f}s" if stage.duration_s is not None else "—",
            notes,
        )
    return table


def _append(target: Text, part: str, *, style: str = "") -> None:
    """Append one ``·``-separated note, styled, without ever parsing ``part`` as markup."""
    if target.plain:
        target.append(" · ", style="dim")
    target.append(part, style=style)


def metrics_table(doc: VideoContextDocument) -> Table:
    m = doc.metrics
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    grid.add_row("processing time", f"{_num(m.processing_time_s)}s")
    grid.add_row("realtime factor", _num(m.realtime_factor, 3))
    grid.add_row("frames sampled", str(m.frames_sampled if m.frames_sampled is not None else "—"))
    grid.add_row("frames skipped", str(m.frames_skipped if m.frames_skipped is not None else "—"))
    grid.add_row("cache hits", str(m.cache_hits))
    if m.stage_times:
        slowest = sorted(m.stage_times.items(), key=lambda kv: -kv[1])[:4]
        grid.add_row("slowest stages", " · ".join(f"{k} {v:.2f}s" for k, v in slowest))
    return grid


# -- fact sections ---------------------------------------------------------


def _facts_table(*columns: str) -> Table:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=False)
    table.add_column("TIME", no_wrap=True, style="cyan")
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


def transcript_table(doc: VideoContextDocument, limit: int | None) -> Table:
    table = _facts_table("SPEAKER", "TEXT")
    for utterance in _clip(doc.transcript, limit):
        table.add_row(
            format_span(utterance.start, utterance.end),
            utterance.speaker or "—",
            shorten(utterance.text, 120),
        )
    return table


def ocr_table(doc: VideoContextDocument, limit: int | None) -> Table:
    table = _facts_table("TEXT", "EVIDENCE")
    for item in _clip(doc.ocr, limit):
        # first/last frame timestamps are the raw observations; start/end are estimates derived
        # from them, and showing both is what makes the estimate auditable.
        seen = "—"
        if item.first_frame_ts is not None:
            seen = format_timecode(item.first_frame_ts, millis=True)
            if item.last_frame_ts is not None and item.last_frame_ts != item.first_frame_ts:
                seen += f"→{format_timecode(item.last_frame_ts, millis=True)}"
        flags = " ".join(part for part in ("stable" if item.stable else "", seen) if part)
        table.add_row(
            format_span(item.start, item.end),
            shorten(item.text, 96),
            _dim(f"{item.frame_count} frames", flags),
        )
    return table


def scenes_table(doc: VideoContextDocument, limit: int | None) -> Table:
    table = _facts_table("SIGNALS", "DETAIL")
    for scene in _clip(doc.scenes, limit):
        table.add_row(
            format_span(scene.start, scene.end),
            ", ".join(scene.signals) or "—",
            _dim(
                scene.detector or "",
                f"score {_num(scene.change_score, 3)}",
                f"keyframe {format_timecode(scene.keyframe_ts)}"
                if scene.keyframe_ts is not None
                else "",
            ),
        )
    return table


def events_table(doc: VideoContextDocument, limit: int | None) -> Table:
    table = _facts_table("TYPE", "DESCRIPTION", "REFS")
    for item in _clip(doc.events, limit):
        refs = " ".join(f"{kind}:{len(ids)}" for kind, ids in sorted(item.refs.items()))
        table.add_row(
            format_span(item.start, item.end),
            item.type,
            shorten(item.description or "—", 80),
            _dim(refs, ",".join(item.source)),
        )
    return table


def vision_table(doc: VideoContextDocument, limit: int | None) -> Table:
    table = _facts_table("DESCRIPTION", "ENTITIES")
    for note in _clip(doc.vision, limit):
        table.add_row(
            format_span(note.start, note.end),
            shorten(note.description, 96),
            shorten(", ".join(note.entities), 48),
        )
    return table


def segments_table(doc: VideoContextDocument, limit: int | None) -> Table:
    table = _facts_table("ID", "CONTENTS", "TEXT")
    for segment in _clip(doc.segments, limit):
        contents = " ".join(
            f"{label}={len(ids)}"
            for label, ids in (
                ("sc", segment.scene_ids),
                ("tr", segment.transcript_ids),
                ("ocr", segment.ocr_ids),
                ("vis", segment.vision_ids),
                ("ev", segment.event_ids),
            )
            if ids
        )
        table.add_row(
            format_span(segment.start, segment.end),
            segment.id,
            _dim(contents),
            shorten(segment.text, 72),
        )
    return table


def _clip(items: list[Any], limit: int | None) -> list[Any]:
    return items if limit is None or limit <= 0 else items[:limit]


# -- environment -----------------------------------------------------------


def doctor_table(checks: list[dict[str, Any]]) -> Table:
    """What is installed, and — when something is missing — the command that installs it.

    A missing optional component is not a failure: it means one stage will be recorded as
    ``skipped``. Only the required ones are red.
    """
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=False)
    table.add_column("COMPONENT", no_wrap=True)
    table.add_column("", no_wrap=True)
    table.add_column("DETAIL", overflow="fold")
    for check in checks:
        if check["ok"]:
            mark = Text("ok", style="green")
        elif check["required"]:
            mark = Text("missing", style="bold red")
        else:
            mark = Text("absent", style="yellow")
        table.add_row(str(check["name"]), mark, str(check["detail" if check["ok"] else "hint"]))
    return table


def providers_grid(providers: dict[str, list[str]]) -> Table:
    """Every capability the registry knows and the providers registered against it."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(overflow="fold")
    for capability, names in sorted(providers.items()):
        grid.add_row(capability, ", ".join(sorted(names)) or "—")
    return grid


def truncation_note(shown: int, total: int, flag: str) -> RenderableType | None:
    if shown >= total:
        return None
    return Text(f"… {total - shown} more — pass {flag} to see all", style="dim")


__all__ = [
    "bold",
    "console",
    "counts_table",
    "doctor_table",
    "errors",
    "events_table",
    "metrics_table",
    "ocr_table",
    "overview",
    "providers_grid",
    "scenes_table",
    "search_footer",
    "segments_table",
    "shorten",
    "snapshot_table",
    "spans_table",
    "stages_table",
    "transcript_table",
    "truncation_note",
    "vision_table",
]
