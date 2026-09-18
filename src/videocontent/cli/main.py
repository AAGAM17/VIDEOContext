"""The command line — the surface most people will meet this project through.

Six commands, each mapping to a question someone actually has:

    videocontent process demo.mp4              → demo.vctx
    videocontent inspect demo.vctx             what is in it, and did every stage run?
    videocontent search  demo.vctx "pricing"   ranked, timestamped evidence
    videocontent at      demo.vctx 03:21       what was on screen at that moment
    videocontent doctor                        which providers can actually run here
    videocontent schema                        the format, for consumers that are not Python

Three conventions hold across all of them.

**Every command has a ``--json`` twin.** The text output is for a person and may be re-styled;
the JSON is the contract, and an agent or a shell script should use it.

**An empty result is a successful run.** ``search`` exits 0 when nothing matched, because "the
speaker never said that" is a correct answer and a non-zero exit would make a caller treat it as
breakage. Exit 1 means the tool failed; exit 2 means the invocation was wrong.

**Errors print their remediation.** ``VideoContextError`` carries a ``hint``, and the point of
that field is this handler: "OCR failed" without "brew install tesseract" wastes an afternoon.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.text import Text

from .. import __version__
from ..config import ProcessingConfig, _assign, load_config
from ..errors import VideoContextError
from ..logging import configure as configure_logging
from ..schema import io
from ..timecode import parse_timecode
from . import render
from .render import console, errors

app = typer.Typer(
    name="videocontent",
    help="Turn video into timestamped, searchable context. Runs locally by default.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

F = TypeVar("F", bound=Callable[..., Any])

#: Sections ``inspect`` can print, and the flag that asks for each in full.
_SECTIONS = ("scenes", "transcript", "ocr", "vision", "events", "segments")

#: How many rows of each section the default ``inspect`` view shows.
_PREVIEW = 8


class _State:
    """Options shared by every command, resolved once in the root callback."""

    def __init__(self) -> None:
        self.config: ProcessingConfig = ProcessingConfig()


state = _State()


def friendly(fn: F) -> F:
    """Turn a library error into a message and an exit code instead of a traceback.

    Applied per command rather than around ``app()`` so that the behaviour is reachable from
    ``CliRunner`` and therefore testable.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kw: Any) -> Any:
        try:
            return fn(*args, **kw)
        except VideoContextError as exc:
            errors.print(f"error: {exc.message}")
            if exc.hint:
                errors.print(f"hint: {exc.hint}", style="yellow")
            raise typer.Exit(1) from None
        except KeyboardInterrupt:  # pragma: no cover - interactive only
            errors.print("interrupted")
            raise typer.Exit(130) from None

    return wrapper  # type: ignore[return-value]


def _emit(payload: dict[str, Any]) -> None:
    """JSON to stdout, one document, no rich styling — this is the machine-readable path."""
    print(json.dumps(payload, indent=2, default=str))


def _overrides(pairs: list[str]) -> dict[str, Any]:
    """``--set sampling.mode=adaptive`` into a nested overrides mapping."""
    layers: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"--set expects KEY=VALUE, got {pair!r}")
        _assign(layers, key.strip(), value)
    return layers


def _timestamp(value: str) -> float:
    """A timecode a person typed, as seconds. Usage errors exit 2, not 1."""
    try:
        return parse_timecode(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _optional_timestamp(value: str | None) -> float | None:
    return None if value is None else _timestamp(value)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"videocontent {__version__}")
        raise typer.Exit(0)


@app.callback()
def root(
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="YAML config file. Default: search upward for one.",
        exists=True, dir_okay=False, readable=True,
    ),
    set_: list[str] = typer.Option(
        [], "--set", "-s", metavar="KEY=VALUE",
        help="Override one config value, e.g. --set sampling.mode=adaptive. Repeatable.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log at DEBUG."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Log errors only."),
    log_json: bool = typer.Option(False, "--log-json", help="Structured logs on stderr."),
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    level = "DEBUG" if verbose else "ERROR" if quiet else "INFO"
    configure_logging(level=level, fmt="json" if log_json else "text", force=True)
    try:
        state.config = load_config(_overrides(set_), config_file=config_file)
    except VideoContextError as exc:
        errors.print(f"error: {exc.message}")
        raise typer.Exit(1) from None


# -- process ---------------------------------------------------------------


@app.command()
@friendly
def process(
    video: str = typer.Argument(
        ..., help="Video file path or http(s) URL to process.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Where to write the .vctx. Default: alongside the video.",
    ),
    compress: bool = typer.Option(False, "--gzip", help="Write gzip-compressed .vctx."),
    as_json: bool = typer.Option(False, "--json", help="Print a machine-readable summary."),
) -> None:
    """Process a video into a .vctx document.

    Accepts a local file path or a direct http(s) media URL. Remote URLs are fetched
    through the source security boundary (SSRF protection, redirect revalidation,
    size/timeout limits) into a temp file, then processed by the same pipeline —
    use `source inspect` first to check access without downloading.

    Local by default: FFmpeg for decoding, Tesseract for on-screen text, faster-whisper for
    speech. Nothing is uploaded unless a remote provider is explicitly configured, and
    ``inspect`` will tell you per stage which ones ran where.
    """
    from ..sdk import Video

    handle = Video(video, config=state.config)
    handle.process()
    target = output or handle.default_path()
    if compress and target.suffix != ".gz":
        target = target.with_name(target.name + ".gz")
    path = handle.save(target, compress=compress or None)
    doc = handle.document

    if as_json:
        _emit(
            {
                "output": str(path),
                "id": doc.id,
                "vctx_version": doc.vctx_version,
                "duration_s": doc.video.duration,
                "counts": {
                    "scenes": len(doc.scenes), "transcript": len(doc.transcript),
                    "ocr": len(doc.ocr), "vision": len(doc.vision),
                    "events": len(doc.events), "segments": len(doc.segments),
                    "frames": len(doc.frames),
                },
                "metrics": doc.metrics.model_dump(mode="json"),
                "stages": [
                    {"name": s.name, "status": s.status.value, "provider": s.provider,
                     "remote": s.remote, "duration_s": s.duration_s, "error": s.error}
                    for s in doc.stages
                ],
            }
        )
        return

    console.print(render.overview(doc))
    console.print(render.stages_table(doc))
    console.print(render.counts_table(doc))
    console.print(render.metrics_table(doc))
    console.print(
        render.bold(f"\nwrote {path}") + Text(f" ({path.stat().st_size / 1024:.1f} KiB)")
    )
    failed = [s.name for s in doc.stages if s.status.value == "failed"]
    if failed:
        errors.print(f"warning: {len(failed)} stage(s) failed: {', '.join(failed)}")


# -- inspect ---------------------------------------------------------------


@app.command()
@friendly
def inspect(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    scenes: bool = typer.Option(False, "--scenes", help="Show every scene."),
    transcript: bool = typer.Option(False, "--transcript", "-t", help="Show every utterance."),
    ocr: bool = typer.Option(False, "--ocr", help="Show every on-screen text event."),
    vision: bool = typer.Option(False, "--vision", help="Show every vision note."),
    events: bool = typer.Option(False, "--events", help="Show every event."),
    segments: bool = typer.Option(False, "--segments", help="Show every segment."),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show every section in full."),
    limit: int = typer.Option(
        0, "--limit", "-n", metavar="N",
        help=f"Rows per section. Default: {_PREVIEW} in the overview, all with a section flag.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the selection as JSON."),
) -> None:
    """Show what a .vctx document contains.

    With no section flags you get the overview: metadata, what each stage did, how many facts of
    each kind there are, and a short preview of each section. Section flags print that section
    in full — the counts table is the thing to read first, because it distinguishes "no text on
    screen" from "OCR never ran".
    """
    doc = io.load(document)
    asked = {
        "scenes": scenes, "transcript": transcript, "ocr": ocr,
        "vision": vision, "events": events, "segments": segments,
    }
    focused = [name for name, wanted in asked.items() if wanted]
    selected = _SECTIONS if (show_all or not focused) else tuple(focused)
    full = show_all or bool(focused)
    #: ``None`` means every row. An explicit ``--limit`` always wins; otherwise a section flag
    #: means "all of it" and the bare overview means "a preview".
    rows = limit if limit > 0 else (None if full else _PREVIEW)

    if as_json:
        # JSON is the contract, so it is not truncated unless the caller asked for it — a
        # consumer reading `payload["transcript"]` must not silently receive eight of a hundred
        # utterances because a display default happened to apply.
        cap = limit if limit > 0 else None
        payload: dict[str, Any] = {
            "id": doc.id,
            "vctx_version": doc.vctx_version,
            "video": doc.video.model_dump(mode="json"),
            "producer": doc.producer.model_dump(mode="json"),
            "created_at": doc.created_at,
            "metrics": doc.metrics.model_dump(mode="json"),
            "stages": [s.model_dump(mode="json") for s in doc.stages],
            "counts": {name: len(getattr(doc, name)) for name in _SECTIONS},
            "truncated": cap is not None,
        }
        for name in selected:
            items = getattr(doc, name)
            payload[name] = [
                item.model_dump(mode="json")
                for item in (items if cap is None else items[:cap])
            ]
        _emit(payload)
        return

    console.print(render.overview(doc))
    console.print(render.stages_table(doc))
    console.print(render.counts_table(doc))

    renderers = {
        "scenes": render.scenes_table,
        "transcript": render.transcript_table,
        "ocr": render.ocr_table,
        "vision": render.vision_table,
        "events": render.events_table,
        "segments": render.segments_table,
    }
    flags = {
        "scenes": "--scenes", "transcript": "--transcript", "ocr": "--ocr",
        "vision": "--vision", "events": "--events", "segments": "--segments",
    }
    for name in selected:
        items = getattr(doc, name)
        if not items:
            continue
        console.print(render.bold(f"\n{name}") + Text(f" ({len(items)})"))
        console.print(renderers[name](doc, rows))
        shown = len(items) if rows is None else min(rows, len(items))
        note = render.truncation_note(shown, len(items), flags[name])
        if note is not None:
            console.print(note)

    if not full:
        console.print(render.metrics_table(doc))


# -- search ----------------------------------------------------------------


@app.command()
@friendly
def search(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    query: str = typer.Argument(..., help="Words to look for, in speech and on screen."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Results to show. 0 for all."),
    modality: list[str] = typer.Option(
        [], "--modality", "-m",
        help="Restrict to transcript / ocr / vision / events. Repeatable.",
    ),
    start: str | None = typer.Option(None, "--from", help="Only after this timecode."),
    end: str | None = typer.Option(None, "--to", help="Only before this timecode."),
    min_score: float | None = typer.Option(None, "--min-score", help="Drop weaker matches."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
    temporal: bool = typer.Option(False, "--temporal",
                                  help="Plan temporal questions (before/after/first/range)."),
    explain: bool = typer.Option(False, "--explain", help="Show the query plan and notes."),
) -> None:
    """Find timestamped evidence for a query.

    Ranking is lexical (BM25) with a bonus when the words appear in speech *and* on screen at
    the same moment. Every result carries the ids of the facts it came from, so a timestamp can
    always be traced back into the document — nothing here is generated.

    Exits 0 with no results when nothing matched.
    """
    from ..sdk import load as load_video

    video = load_video(document, config=state.config)
    plan = None
    if temporal:
        plan, result = video.retriever.query_temporal(
            query,
            modalities=modality or None,
            top_k=top_k,
        )
        # Re-apply the time/score filters the plain path supports.
        if start is not None or end is not None or min_score is not None:
            kept = [s for s in result.spans
                    if (start is None or s.end >= _timestamp(start))
                    and (end is None or s.start <= _timestamp(end))
                    and (min_score is None or s.score >= min_score)]
            from ..retrieval.query import SearchResult as _SR

            result = _SR(query=result.query, spans=tuple(kept), modalities=result.modalities,
                         total=len(kept), took_ms=result.took_ms, notes=result.notes)
    else:
        result = video.search(
            query,
            modalities=modality or None,
            start=_optional_timestamp(start),
            end=_optional_timestamp(end),
            top_k=top_k,
            min_score=min_score,
        )
    if as_json:
        payload = result.to_dict()
        if plan is not None:
            payload["query_plan"] = plan.to_dict()
        _emit(payload)
        return
    if not result:
        console.print(Text("no matches for ") + render.bold(query))
        console.print(render.search_footer(result))
        return
    console.print(render.spans_table(result))
    console.print(render.search_footer(result))
    if explain:
        lines = []
        if plan is not None:
            lines.append(f"plan: {plan.to_dict()}")
        lines.extend(f"note: {note}" for note in result.notes)
        console.print(render.explain_block("query", lines or ["lexical search, no plan"]))


# -- at --------------------------------------------------------------------

@app.command()
@friendly
def at(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    timecode: str = typer.Argument(..., help="An instant: 201.45, 3:21 or 00:03:21.450."),
    window: float = typer.Option(
        0.0, "--window", "-w", help="Also include facts within this many seconds.",
    ),
    modality: list[str] = typer.Option(
        [], "--modality", "-m", help="Restrict to one or more modalities.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Show everything the document knows about one instant.

    Not a search: every span returned demonstrably covers the timestamp, so there is nothing to
    rank. Output is ordered speech, screen, vision, events — read it as a snapshot.
    """
    from ..sdk import load as load_video

    video = load_video(document, config=state.config)
    result = video.at(_timestamp(timecode), window=window, modalities=modality or None)
    if as_json:
        _emit(result.to_dict())
        return
    if not result:
        console.print(Text("nothing recorded at ") + render.bold(result.query))
        return
    console.print(render.bold(result.query))
    console.print(render.snapshot_table(result))


# -- ask ---------------------------------------------------------------------


@app.command()
@friendly
def ask(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    question: str = typer.Argument(..., help="Question to ask about the video."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Max evidence spans to use."),
    modality: list[str] = typer.Option(
        [], "--modality", "-m", help="Restrict search to these modalities.",
    ),
    min_score: float | None = typer.Option(None, "--min-score", help="Drop weaker matches."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
    explain: bool = typer.Option(False, "--explain", help="Show the query plan and trace."),
) -> None:
    """Answer a question about the video using retrieved evidence and an LLM.

    The pipeline: query plan → retrieval → evidence → context → LLM → answer.
    Temporal questions ("before the error", "when did X first appear") are planned
    explicitly. Every answer carries its evidence spans so you can verify timestamps.
    """
    from ..sdk import load as load_video

    video = load_video(document, config=state.config)
    answer = video.ask(
        question,
        modalities=modality or None,
        top_k=top_k,
        min_score=min_score,
    )
    if as_json:
        _emit(answer.to_dict())
        return
    console.print(render.bold(f"Q: {answer.question}"))
    console.print(render.bold(f"A: {answer.answer}"))
    console.print(render.bold(f"Confidence: {answer.confidence:.2f}"))
    if answer.evidence:
        console.print(render.bold("\nEvidence:"))
        for i, span in enumerate(answer.evidence, 1):
            console.print(f"  [{i}] {span.timecode} ({span.modality}): {span.text[:120]}")
    if explain and answer.trace:
        trace = answer.trace
        lines = [f"plan: {trace.get('query_plan', {})}",
                 f"executor: {trace.get('executor', '?')}"]
        retrieval = trace.get("retrieval", {})
        lines.append(f"retrieval: {retrieval.get('spans', 0)} spans "
                     f"(total {retrieval.get('total', 0)})")
        lines.extend(f"note: {note}" for note in retrieval.get("notes", ()))
        lines.append(f"outcome: {trace.get('outcome', '?')}")
        console.print(render.explain_block("trace", lines))


# -- timeline / events / entities / changes / chapters / context ---------------


def _load_video(document: Path):
    from ..sdk import load as load_video

    return load_video(document, config=state.config)


@app.command()
@friendly
def timeline(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    start: str | None = typer.Option(None, "--from", help="Range start timecode."),
    end: str | None = typer.Option(None, "--to", help="Range end timecode."),
    modality: list[str] = typer.Option([], "--modality", "-m", help="Restrict modalities."),
    top_k: int = typer.Option(100, "--top-k", "-k", help="Max facts to show. 0 for all."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Show everything the document knows about a time range, in timeline order."""
    video = _load_video(document)
    result = video.timeline(_optional_timestamp(start) or 0.0,
                            _optional_timestamp(end), modalities=modality or None,
                            top_k=top_k)
    if as_json:
        _emit(result.to_dict())
        return
    if not result:
        console.print(Text("nothing recorded in ") + render.bold(result.query))
        return
    console.print(render.bold(result.query))
    console.print(render.snapshot_table(result))
    console.print(render.search_footer(result))


@app.command()
@friendly
def events(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    type_: str | None = typer.Option(None, "--type", "-t", help="Only this event type."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """List typed temporal events with the evidence behind each one."""
    video = _load_video(document)
    matched = [e for e in video.document.events if type_ is None or e.type == type_]
    if as_json:
        _emit({"events": [e.model_dump(mode="json") for e in matched]})
        return
    if not matched:
        console.print(Text("no events") + (Text(f" of type {type_}") if type_ else Text("")))
        return
    doc = video.document
    console.print(render.bold(f"{len(matched)} events"))
    console.print(render.events_table(doc, events=matched))


@app.command()
@friendly
def entities(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    type_: str | None = typer.Option(None, "--type", "-t",
                                     help="Only this entity type (ERROR, COMMAND, CONCEPT)."),
    top_k: int = typer.Option(50, "--top-k", "-k", help="Max entities to show."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """List timestamp-grounded entities with cross-modal links and uncertainty."""
    video = _load_video(document)
    matched = video.entities()
    if type_ is not None:
        matched = [e for e in matched if e.type == type_.upper()]
    matched = matched[:top_k] if top_k > 0 else matched
    if as_json:
        _emit({"entities": [e.to_dict() for e in matched]})
        return
    if not matched:
        console.print(Text("no entities found"))
        return
    console.print(render.entities_table(matched))


@app.command()
@friendly
def changes(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Show what changed between adjacent regions (text, speech, scenes)."""
    video = _load_video(document)
    found = video.changes()
    if as_json:
        _emit({"changes": [c.to_dict() for c in found]})
        return
    if not found:
        console.print(Text("no changes detected"))
        return
    console.print(render.changes_table(found))


@app.command()
@friendly
def chapters(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    target: float = typer.Option(300.0, "--target-s", help="Target seconds per chapter."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Show extractive chapters. Titles are derived keywords, marked as such."""
    from ..temporal import build_chapters

    video = _load_video(document)
    found = build_chapters(video.document, target_s=target)
    if as_json:
        _emit({"chapters": [c.to_dict() for c in found]})
        return
    if not found:
        console.print(Text("no chapters (unknown duration)"))
        return
    console.print(render.chapters_table(found))


@app.command(name="context")
@friendly
def context_cmd(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    task: str = typer.Argument(..., help="Task to build context for."),
    max_tokens: int = typer.Option(4000, "--max-tokens", help="Token budget."),
    max_spans: int | None = typer.Option(None, "--max-spans", help="Cap evidence spans."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Cap frames."),
    expand: float = typer.Option(0.0, "--expand-s",
                                 help="Pull ±N seconds around each match."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
    explain: bool = typer.Option(False, "--explain", help="Show budget decisions."),
) -> None:
    """Build a minimal, budgeted AI context package for a task."""
    video = _load_video(document)
    ctx = video.context(task, max_tokens=max_tokens, max_spans=max_spans,
                        max_frames=max_frames, expand_s=expand)
    if as_json:
        _emit(ctx.to_dict())
        return
    console.print(render.bold(f"task: {ctx.task}"))
    console.print(f"tokens ≈ {ctx.token_estimate} · "
                  f"{len(ctx.evidence)} spans · {len(ctx.frames)} frames")
    if ctx.evidence:
        from ..retrieval.query import SearchResult as _SR

        console.print(render.spans_table(_SR(query=task, spans=tuple(ctx.evidence),
                                            total=len(ctx.evidence))))
    if explain and ctx.budget_notes:
        console.print(render.explain_block("budget", ctx.budget_notes))


# -- agent intelligence --------------------------------------------------------


@app.command()
@friendly
def graph(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    node: str | None = typer.Option(None, "--node", "-n",
                                    help="Show neighbors of this node ID."),
    entity: str | None = typer.Option(None, "--entity", "-e",
                                      help="Show occurrences of this entity name."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Show the derived evidence graph: stats, node neighbors, entity occurrences."""
    video = _load_video(document)
    graph = video.graph()
    if as_json:
        if node is not None:
            _emit({"node": node,
                   "neighbors": [{"node": n.to_dict(), "edge": e.to_dict()}
                                 for n, e in graph.neighbors(node)[:30]]})
        elif entity is not None:
            timeline = video.entity_timeline(entity)
            _emit({"entity": entity,
                   "timeline": timeline.to_dict() if timeline else None})
        else:
            _emit({"stats": graph.stats()})
        return
    console.print(render.bold(f"graph: {len(graph.nodes)} nodes · {len(graph.edges)} edges"))
    if node is not None:
        for neighbor, edge in graph.neighbors(node)[:30]:
            console.print(f"  {edge.relation} → {neighbor.id} ({neighbor.kind})")
    elif entity is not None:
        timeline = video.entity_timeline(entity)
        if timeline is None:
            console.print(Text(f"no entity named {entity!r}"))
            return
        for occurrence in timeline.occurrences:
            console.print(f"  [{occurrence.start:.1f}] ({occurrence.modality}) "
                          f"{occurrence.text[:100]}")
    else:
        for kind, count in sorted(graph.stats()["kinds"].items()):
            console.print(f"  {kind}: {count}")


@app.command(name="entity-timeline")
@friendly
def entity_timeline_cmd(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    name: str = typer.Argument(..., help="Entity name, e.g. ConnectionError."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Every occurrence of an entity in time order, with supporting evidence."""
    video = _load_video(document)
    timeline = video.entity_timeline(name)
    if timeline is None:
        if as_json:
            _emit({"entity": name, "occurrences": []})
            return
        console.print(Text(f"no entity named {name!r}"))
        return
    if as_json:
        _emit(timeline.to_dict())
        return
    console.print(render.bold(f"{timeline.entity.name} "
                              f"[{timeline.entity.type}] x{len(timeline.occurrences)}"))
    for occurrence in timeline.occurrences:
        console.print(f"  [{occurrence.start:.1f}] ({occurrence.modality}) "
                      f"{occurrence.text[:100]}")


@app.command()
@friendly
def plan(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    question: str = typer.Argument(..., help="Question to plan for."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Show the inspectable query plan: intent, entities, strategy, coverage."""
    video = _load_video(document)
    planned = video.query_plan(question)
    if as_json:
        _emit(planned)
        return
    console.print(render.bold(f"intent: {planned['intent']}"))
    if planned["entities"]:
        console.print(f"entities: {', '.join(planned['entities'])}")
    if planned["temporal"]:
        console.print(f"temporal: {planned['temporal']}")
    console.print("strategy:")
    for step in planned["retrieval_strategy"]:
        console.print(f"  • {step}")
    if planned["coverage"].get("missing"):
        errors.print("coverage gap: " + ", ".join(planned["coverage"]["missing"]))
    if planned["warnings"]:
        for warning in planned["warnings"]:
            errors.print(f"warning: {warning}")


@app.command()
@friendly
def explain(
    document: Path = typer.Argument(
        ..., help="A .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    ref: str = typer.Argument(..., help="Node or edge ID, e.g. evt_0000 or edge_00001."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Explain a graph node (supporting evidence) or edge (construction rule)."""
    video = _load_video(document)
    explanation = video.explain(ref)
    if explanation is None:
        if as_json:
            _emit({"ref": ref, "found": False})
            return
        errors.print(f"error: no node or edge {ref!r}")
        raise typer.Exit(1)
    if as_json:
        _emit(explanation)
        return
    if "rule" in explanation:
        console.print(render.bold(f"{explanation['relation']}: "
                                  f"{explanation['source']} → {explanation['target']}"))
        console.print(f"rule: {explanation['rule']}")
        console.print(f"provenance: {', '.join(explanation['provenance'])}")
    else:
        node = explanation["node"]
        console.print(render.bold(f"{node['id']} ({node['kind']}) — {node['label'][:100]}"))
        console.print(f"supporting evidence ({len(explanation['supporting_evidence'])}):")
        for support in explanation["supporting_evidence"][:10]:
            console.print(f"  [{support['start']:.1f}] ({support['kind']}) "
                          f"{support['label'][:100]}")


@app.command()
@friendly
def compare(
    first: Path = typer.Argument(
        ..., help="First .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    second: Path = typer.Argument(
        ..., help="Second .vctx file.", exists=True, dir_okay=False, readable=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Compare two videos: added/removed/changed/unchanged entities plus structure."""
    from ..collection import compare as _compare
    from ..sdk import load as load_video

    doc_a = load_video(first, config=state.config).document
    doc_b = load_video(second, config=state.config).document
    result = _compare(first.stem, doc_a, second.stem, doc_b)
    if as_json:
        _emit(result.to_dict())
        return
    console.print(render.bold(f"{first.stem} vs {second.stem}"))
    if result.added:
        console.print(f"added in {second.stem}: {', '.join(result.added[:10])}")
    if result.removed:
        console.print(f"only in {first.stem}: {', '.join(result.removed[:10])}")
    if result.changed:
        console.print(f"changed: {', '.join(result.changed[:10])}")
    if result.uncertain:
        console.print(f"uncertain: {', '.join(result.uncertain[:10])}")
    console.print(f"structure: {result.structure_a} vs {result.structure_b}")


collection_app = typer.Typer(
    name="collection",
    help="Multi-video intelligence over .vctx files (never merges them).",
    no_args_is_help=True,
)
app.add_typer(collection_app, name="collection")


@collection_app.command("search")
@friendly
def collection_search(
    query: str = typer.Argument(..., help="Query to run across videos."),
    documents: list[Path] = typer.Argument(..., help="Two or more .vctx files."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Max merged spans."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Search across videos; every span keeps its video_id provenance."""
    from ..collection import CollectionIndex
    from ..sdk import load as load_video

    if len(documents) < 2:
        errors.print("error: collection search needs at least two .vctx files")
        raise typer.Exit(2)
    docs = {}
    for path in documents:
        if not path.is_file():
            errors.print(f"error: no such file: {path}")
            raise typer.Exit(2)
        docs[path.stem] = load_video(path, config=state.config).document
    result = CollectionIndex(docs).search(query, top_k=top_k)
    if as_json:
        _emit(result.to_dict())
        return
    for span in result.spans:
        console.print(f"  [{span.video_id} {span.timecode}] ({span.modality}): "
                      f"{span.text[:120]}")
    console.print(render.bold(f"{result.total} spans across "
                              f"{result.videos_searched} videos"))


@collection_app.command("entities")
@friendly
def collection_entities(
    documents: list[Path] = typer.Argument(..., help="Two or more .vctx files."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Cross-video entity links; video-local evidence stays attached."""
    from ..collection import CollectionIndex
    from ..sdk import load as load_video

    docs = {p.stem: load_video(p, config=state.config).document for p in documents
            if p.is_file()}
    linked = CollectionIndex(docs).link_entities()
    if as_json:
        _emit({"links": linked})
        return
    shared = {name: entries for name, entries in linked.items() if len(entries) > 1}
    console.print(render.bold(f"{len(linked)} distinct entities, "
                              f"{len(shared)} shared across videos"))
    for name, entries in sorted(shared.items())[:20]:
        console.print(f"  {name}: {', '.join(e['video_id'] for e in entries)}")


# -- source ----------------------------------------------------------------


source_app = typer.Typer(
    name="source",
    help="Inspect and resolve video sources (local files, URLs).",
    no_args_is_help=True,
)
app.add_typer(source_app, name="source")


@source_app.command("inspect")
@friendly
def source_inspect(
    source: str = typer.Argument(..., help="File path or http(s) URL to describe."),
    as_json: bool = typer.Option(False, "--json", help="Emit the inspection as JSON."),
) -> None:
    """Describe a video source without downloading or processing it.

    Answers: what is this, where does it come from, can VIDEOContext access it, what
    media is available, does it need auth, what processing is possible.
    """
    from ..sources.resolve import inspect_source, resolve

    src = resolve(source, config=state.config)
    info = inspect_source(src, config=state.config)
    if as_json:
        _emit({
            "source_id": src.source_id,
            "source_type": src.source_type.value,
            "provider": src.provider,
            "canonical_id": src.canonical_id,
            "accessible": info.accessible,
            "reason": info.reason,
            "media": info.media,
            "capabilities": info.capabilities.model_dump(mode="json"),
            "requires_authentication": info.requires_authentication,
            "estimated_size_bytes": info.estimated_size_bytes,
            "warnings": info.warnings,
        })
        if not info.accessible:
            raise typer.Exit(1)
        return
    console.print(render.bold(f"{src.source_type.value} ") + Text(f"({src.provider})"))
    console.print(f"  source_id:    {src.source_id}")
    console.print(f"  canonical_id: {src.canonical_id}")
    console.print(f"  accessible:   {'yes' if info.accessible else 'no'}")
    if info.reason:
        console.print(f"  reason:       {info.reason}")
    if info.media:
        console.print(render.bold("  media:"))
        for key, value in info.media.items():
            console.print(f"    {key}: {value}")
    if info.estimated_size_bytes:
        console.print(f"  estimated size: {info.estimated_size_bytes / 1e6:.1f} MB")
    if info.warnings:
        for warning in info.warnings:
            errors.print(f"  warning: {warning}")
    if not info.accessible:
        raise typer.Exit(1)


@source_app.command("resolve")
@friendly
def source_resolve(
    source: str = typer.Argument(..., help="File path or http(s) URL to resolve."),
    as_json: bool = typer.Option(False, "--json", help="Emit the resolution as JSON."),
) -> None:
    """Show the canonical identity VIDEOContext assigns to a source (dedup key)."""
    from ..sources.resolve import resolve

    src = resolve(source, config=state.config)
    if as_json:
        _emit(src.model_dump(mode="json"))
        return
    console.print(f"source_id:    {src.source_id}")
    console.print(f"source_type:  {src.source_type.value}")
    console.print(f"provider:     {src.provider}")
    console.print(f"canonical_id: {src.canonical_id}")


# -- doctor ----------------------------------------------------------------


@app.command()
@friendly
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Check what can actually run on this machine.

    FFmpeg is the only hard requirement. Everything else degrades: without Tesseract there is no
    OCR, without faster-whisper there is no transcript, and the document records the stage as
    ``skipped`` rather than pretending the video had no text or no speech.
    """
    from .. import registry
    from ..media import ffmpeg
    from ..processing.ocr import tesseract

    engine = tesseract.TesseractOCR()
    checks: list[dict[str, Any]] = [
        {
            "name": "ffmpeg",
            "required": True,
            "ok": ffmpeg.available(),
            "detail": ffmpeg.version() or "not on PATH",
            "hint": "install ffmpeg (brew install ffmpeg / apt install ffmpeg)",
        },
        {
            "name": "tesseract",
            "required": False,
            "ok": engine.available(),
            "detail": engine.version,
            "hint": "brew install tesseract — without it, OCR is skipped",
        },
    ]
    for module, label, extra in (
        ("faster_whisper", "faster-whisper", "asr"),
        ("faiss", "faiss", "vectors"),
        ("sentence_transformers", "sentence-transformers", "embeddings"),
    ):
        hint = f"pip install 'videocontent[{extra}]'"
        checks.append({**_probe(module), "name": label, "required": False, "hint": hint})

    providers = registry.all_names()
    ok = all(check["ok"] for check in checks if check["required"])

    if as_json:
        _emit({"ok": ok, "checks": checks, "providers": providers})
        raise typer.Exit(0 if ok else 1)

    console.print(render.doctor_table(checks))
    console.print(render.bold("\nregistered providers"))
    console.print(render.providers_grid(providers))
    if not ok:
        errors.print("\nffmpeg is required and was not found.")
        raise typer.Exit(1)


def _probe(module: str) -> dict[str, Any]:
    """Whether an optional package is installed, without importing it into this process."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):  # pragma: no cover - malformed installs
        spec = None
    return {"ok": spec is not None, "detail": "installed" if spec else "not installed"}


# -- benchmark ---------------------------------------------------------------


@app.command()
@friendly
def benchmark(
    video: Path = typer.Argument(
        ..., help="Video file to benchmark.", exists=True, dir_okay=False, readable=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Run a processing benchmark on a video.

    Measures per-stage timings, frame counts, and estimated costs.
    Useful for comparing configurations and tracking regressions.
    """
    from ..sdk import Video
    from ..config import ProcessingConfig

    # Use minimal config for consistent benchmark
    cfg = ProcessingConfig()
    cfg.vision.enabled = False  # Skip vision for consistent benchmarking

    handle = Video(video, config=cfg)
    handle.process()
    doc = handle.document

    if as_json:
        _emit({
            "video": video.name,
            "duration_s": doc.video.duration,
            "metrics": doc.metrics.model_dump(mode="json"),
            "stages": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "provider": s.provider,
                    "duration_s": s.duration_s,
                }
                for s in doc.stages
            ],
        })
        return

    console.print(render.bold(f"\nBenchmark: {video.name}"))
    console.print(render.bold("=" * 50))
    console.print(render.metrics_table(doc))
    console.print()
    console.print(render.stages_table(doc))


# -- schema ----------------------------------------------------------------


@app.command()
@friendly
def schema(
    output: Path | None = typer.Option(None, "--output", "-o", help="Write here instead."),
) -> None:
    """Print the .vctx JSON Schema: the format contract for non-Python consumers."""
    payload = json.dumps(io.json_schema(), indent=2)
    if output is None:
        print(payload)
        return
    output.write_text(payload + "\n", encoding="utf-8")
    console.print(render.bold(f"wrote {output}"))


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point for both ``videocontent`` and ``vctx``."""
    app(args=argv)


__all__ = ["app", "main"]
