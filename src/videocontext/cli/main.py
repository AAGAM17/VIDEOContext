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
    video: Path = typer.Argument(
        ..., help="Video file to process.", exists=True, dir_okay=False, readable=True,
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Where to write the .vctx. Default: alongside the video.",
    ),
    compress: bool = typer.Option(False, "--gzip", help="Write gzip-compressed .vctx."),
    as_json: bool = typer.Option(False, "--json", help="Print a machine-readable summary."),
) -> None:
    """Process a video into a .vctx document.

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
) -> None:
    """Find timestamped evidence for a query.

    Ranking is lexical (BM25) with a bonus when the words appear in speech *and* on screen at
    the same moment. Every result carries the ids of the facts it came from, so a timestamp can
    always be traced back into the document — nothing here is generated.

    Exits 0 with no results when nothing matched.
    """
    from ..sdk import load as load_video

    video = load_video(document, config=state.config)
    result = video.search(
        query,
        modalities=modality or None,
        start=_optional_timestamp(start),
        end=_optional_timestamp(end),
        top_k=top_k,
        min_score=min_score,
    )
    if as_json:
        _emit(result.to_dict())
        return
    if not result:
        console.print(Text("no matches for ") + render.bold(query))
        console.print(render.search_footer(result))
        return
    console.print(render.spans_table(result))
    console.print(render.search_footer(result))


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
) -> None:
    """Answer a question about the video using retrieved evidence and an LLM.

    The pipeline: search → select top evidence → build context → LLM → answer.
    Every answer carries its evidence spans so you can verify timestamps.
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
