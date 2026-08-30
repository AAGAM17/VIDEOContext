"""Frame extraction.

Two strategies, chosen per plan by estimated cost:

**Single pass.** One decode of the file, with a ``select`` expression that encodes the plan's
piecewise rate function, so a 1 fps baseline and a 4 fps burst come out of the same decode.

**Seeked windows.** One short, input-seeked decode per window. Cheap when the plan touches
only a little of a long video.

The choice matters more than it looks. Measured on this project's fixture, per-invocation
overhead is ~0.65 s (process start plus seek) while decoding runs at roughly 16x realtime.
A 17-window plan over a 62 s video therefore costs ~16 s as seeked windows and ~5 s as one
pass; a 3-window plan over a 3-hour video inverts that completely. Picking one strategy
unconditionally is wrong in one of those two directions, so :func:`extract_plan` estimates
both and picks. What is never acceptable is one ``ffmpeg -ss`` per *frame* — the classic
mistake this module exists to avoid.

Timestamps come from ``showinfo``, which logs the presentation timestamp of every frame it
passes. ``metadata=print`` is the more obvious choice and is silently useless here: it emits
nothing for frames that carry no metadata, so it degrades to guessing timestamps from the
nominal rate, which is exactly wrong on variable-frame-rate sources.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

from ..errors import FFmpegError
from ..interfaces import FrameImage, SamplePlan, SampleWindow
from ..logging import get_logger
from . import ffmpeg

log = get_logger("media.frames")

#: ``showinfo`` line: ``n:  0 pts:      0 pts_time:0       duration:…``
_SHOWINFO = re.compile(r"\bpts_time:(-?[\d.]+)")

MAX_WINDOWS = 64
#: Wall-clock cost of one ffmpeg invocation including seek, measured on this project's
#: fixture. Used only to compare strategies, never as a timeout.
SPAWN_COST_S = 0.65
#: Decode throughput as a multiple of realtime, for the same comparison.
DECODE_SPEED = 16.0
#: Never select more than this many frames from a single explicit-timestamp window.
_EXPLICIT_EPS = 0.02


def merge_windows(windows: list[SampleWindow], *, gap: float = 1.0) -> list[SampleWindow]:
    """Coalesce overlapping/adjacent windows that share a rate."""
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: (w.start, w.end))
    merged = [SampleWindow(ordered[0].start, ordered[0].end, ordered[0].fps, ordered[0].reason)]
    for window in ordered[1:]:
        last = merged[-1]
        if window.start <= last.end + gap and abs(window.fps - last.fps) < 1e-9:
            last.end = max(last.end, window.end)
            if window.reason != last.reason:
                last.reason = "mixed"
        else:
            merged.append(SampleWindow(window.start, window.end, window.fps, window.reason))
    return merged


def flatten_windows(windows: list[SampleWindow]) -> list[SampleWindow]:
    """Turn overlapping windows into a non-overlapping, piecewise-constant rate function.

    Where windows overlap the highest rate wins, so a burst laid over a baseline yields one
    region at burst rate rather than two regions that both get decoded. Without this the
    overlap is decoded twice and the duplicate frames are thrown away afterwards.
    """
    if not windows:
        return []
    edges = sorted({round(edge, 4) for w in windows for edge in (w.start, w.end)})
    pieces: list[SampleWindow] = []
    for left, right in pairwise(edges):
        if right - left <= 1e-6:
            continue
        mid = (left + right) / 2
        covering = [w for w in windows if w.start <= mid < w.end]
        if not covering:
            continue
        best = max(covering, key=lambda w: w.fps)
        pieces.append(SampleWindow(left, right, best.fps, best.reason))

    collapsed: list[SampleWindow] = []
    for piece in pieces:
        if (
            collapsed
            and abs(collapsed[-1].fps - piece.fps) < 1e-9
            and collapsed[-1].reason == piece.reason
            and piece.start - collapsed[-1].end <= 1e-6
        ):
            collapsed[-1].end = piece.end
        else:
            collapsed.append(piece)
    return collapsed


def _scale_filter(width: int | None) -> str | None:
    if not width:
        return None
    # -2 keeps the aspect ratio with an even height, which most encoders require.
    return f"scale='min({width},iw)':-2"


def _num(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _rate_expression(pieces: list[SampleWindow]) -> str:
    """Nested ``if(between(t,a,b), interval, …)`` giving the minimum gap between frames."""
    expr = "1e9"  # outside every window: never select on rate alone
    for piece in reversed(pieces):
        interval = 1.0 / piece.fps if piece.fps > 0 else 1e9
        expr = (
            f"if(between(t,{_num(piece.start)},{_num(piece.end)}),{_num(interval)},{expr})"
        )
    return expr


def _select_expression(pieces: list[SampleWindow], explicit: list[float]) -> str:
    """The whole plan as one ``select`` predicate.

    ``prev_selected_t`` is NaN until something is selected, so the ``isnan`` term seeds the
    first frame; ``+`` acts as boolean OR in FFmpeg expressions.
    """
    terms: list[str] = []
    if pieces:
        terms.append(f"isnan(prev_selected_t)+gte(t-prev_selected_t,{_rate_expression(pieces)})")
    for ts in explicit:
        terms.append(f"between(t,{_num(max(0.0, ts - _EXPLICIT_EPS))},{_num(ts + _EXPLICIT_EPS)})")
    return "+".join(terms) if terms else "0"


def _run_pass(
    source: str | Path,
    *,
    outdir: Path,
    prefix: str,
    filters: list[str],
    seek: tuple[float, float] | None,
    quality: int,
    timeout: float,
) -> list[tuple[float, Path]]:
    """One ffmpeg invocation. Returns (timestamp, path) pairs, empty on failure."""
    outdir.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    offset = 0.0
    if seek is not None:
        start, length = seek
        offset = max(0.0, start)
        args += ["-ss", f"{offset:.3f}"]
        if length > 0:
            args += ["-t", f"{length:.3f}"]
    args += [
        *ffmpeg.input_args(source),
        "-an", "-sn", "-dn",
        "-vf", ",".join([*filters, "showinfo"]),
        "-fps_mode", "passthrough",
        "-q:v", str(quality),
        "-f", "image2",
        str(outdir / f"{prefix}_%05d.jpg"),
    ]
    try:
        # showinfo logs at info level; anything quieter discards the timestamps.
        result = ffmpeg.run(args, timeout=timeout, loglevel="info")
    except FFmpegError as exc:
        log.warning("frames.pass_failed", extra={"prefix": prefix, "error": exc.message})
        return []

    files = sorted(outdir.glob(f"{prefix}_*.jpg"))
    if not files:
        return []
    times = [round(offset + float(m), 3) for m in _SHOWINFO.findall(result.stderr)]
    if len(times) != len(files):
        log.debug("frames.pts_mismatch", extra={"pts": len(times), "files": len(files)})
        times = (times + [None] * len(files))[: len(files)]  # type: ignore[list-item]
        times = [t if t is not None else 0.0 for t in times]
    # Lengths are equalised by the guard above, so strict= is a free invariant check.
    return list(zip(times, files, strict=True))


def _label(
    ts: float, pieces: list[SampleWindow], explicit: list[tuple[float, str]]
) -> str:
    """Why this frame is in the output — the sampler's reason, preserved into the document."""
    for want, reason in explicit:
        if abs(ts - want) <= _EXPLICIT_EPS * 2:
            return reason
    covering = [p for p in pieces if p.start - 1e-3 <= ts < p.end + 1e-3]
    if covering:
        return max(covering, key=lambda p: p.fps).reason
    return "fixed"


def _estimate_costs(
    pieces: list[SampleWindow], explicit_count: int, duration: float
) -> tuple[float, float]:
    """(single-pass cost, seeked-window cost) in seconds. Heuristic, for strategy choice only."""
    coverage = sum(p.duration for p in pieces)
    single = duration / DECODE_SPEED + SPAWN_COST_S
    seeked = (len(pieces) + explicit_count) * SPAWN_COST_S + coverage / DECODE_SPEED
    return single, seeked


def extract_plan(
    source: str | Path,
    plan: SamplePlan,
    outdir: str | Path,
    *,
    scale_width: int | None = 1280,
    quality: int = 4,
    timeout: float = 1800.0,
    max_frames: int = 2000,
    dedupe_tolerance: float = 0.05,
    strategy: str = "auto",
) -> list[FrameImage]:
    """Execute a sampling plan and return the extracted frames, ascending by timestamp."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    duration = plan.duration
    pieces = flatten_windows(plan.windows)
    if len(pieces) > MAX_WINDOWS:
        rate = max((p.fps for p in pieces), default=1.0)
        span_end = max((p.end for p in pieces), default=duration)
        log.info("frames.plan_collapsed", extra={"pieces": len(pieces), "fps": rate})
        pieces = [SampleWindow(0.0, span_end, rate, "collapsed")]

    explicit = sorted(plan.explicit, key=lambda item: item[0])
    single_cost, seeked_cost = _estimate_costs(pieces, len(explicit), duration)
    if strategy == "auto":
        chosen = "single" if single_cost <= seeked_cost else "seeked"
    else:
        chosen = strategy
    log.debug(
        "frames.strategy",
        extra={"chosen": chosen, "single_s": round(single_cost, 2),
               "seeked_s": round(seeked_cost, 2), "pieces": len(pieces)},
    )

    scale = _scale_filter(scale_width)
    collected: list[tuple[float, Path]] = []

    if chosen == "single":
        select = _select_expression(pieces, [ts for ts, _ in explicit])
        filters = [f for f in (scale,) if f]
        filters.append(f"select='{select}'")
        collected = _run_pass(
            source, outdir=out / "pass", prefix="f", filters=filters,
            seek=None, quality=quality, timeout=timeout,
        )
    else:
        for i, piece in enumerate(pieces):
            filters = [f"fps={piece.fps:g}"]
            if scale:
                filters.append(scale)
            collected += _run_pass(
                source, outdir=out / f"w{i:03d}", prefix="f", filters=filters,
                seek=(piece.start, piece.duration), quality=quality, timeout=timeout,
            )
        for ts, _reason in explicit:
            # -frames:v 1 rather than a hair-thin -t window: a 1 ms duration decodes to
            # nothing and ffmpeg exits non-zero.
            filters = [f for f in (scale,) if f]
            collected += _run_pass(
                source, outdir=out / f"x{int(ts * 1000):09d}", prefix="f",
                filters=[*filters, "select='eq(n\\,0)'"],
                seek=(ts, 0.0), quality=quality, timeout=timeout,
            )

    # Prefer specifically-justified frames when two land on the same instant.
    labelled = [(ts, path, _label(ts, pieces, explicit)) for ts, path in collected]
    labelled.sort(key=lambda item: (item[0], item[2] in ("fixed", "collapsed")))

    frames: list[FrameImage] = []
    last_ts = -1e9
    dropped = 0
    for ts, path, reason in labelled:
        if ts - last_ts < dedupe_tolerance or len(frames) >= max_frames:
            path.unlink(missing_ok=True)
            dropped += 1
            continue
        frames.append(FrameImage(ts=ts, path=path, index=len(frames), reason=reason))
        last_ts = ts

    log.info(
        "frames.extracted",
        extra={"frames": len(frames), "strategy": chosen, "pieces": len(pieces),
               "dropped": dropped},
    )
    return frames


def extract_single(
    source: str | Path,
    ts: float,
    out_path: str | Path,
    *,
    scale_width: int | None = None,
    quality: int = 2,
    timeout: float = 120.0,
) -> Path | None:
    """One frame at one timestamp — for point lookups, never for processing."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    filters = [f for f in (_scale_filter(scale_width),) if f]
    args = ["-ss", f"{max(0.0, ts):.3f}", *ffmpeg.input_args(source), "-frames:v", "1"]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-q:v", str(quality), str(out)]
    try:
        ffmpeg.run(args, timeout=timeout)
    except FFmpegError:
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def frame_timestamps(frames: list[FrameImage]) -> list[float]:
    return [f.ts for f in frames]


__all__ = [
    "DECODE_SPEED",
    "MAX_WINDOWS",
    "SPAWN_COST_S",
    "SamplePlan",
    "SampleWindow",
    "extract_plan",
    "extract_single",
    "flatten_windows",
    "frame_timestamps",
    "merge_windows",
]
