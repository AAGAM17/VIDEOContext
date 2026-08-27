"""FFmpeg-based scene detection.

Raw frame differencing alone is not a scene detector — it fires on camera motion and misses
low-contrast cuts such as one dark slide replacing another. This detector combines three
signals in a **single decode pass**:

1. *frame difference* — FFmpeg's ``scene`` score between consecutive analysed frames
2. *black frames* — ``blackdetect``, which catches fades that score low on difference
3. *density budgeting* — how many cuts the video is allowed to have, given its length

and the pipeline may later add a fourth (on-screen text changes) via
:func:`videocontent.processing.scenes.refine.refine_with_text`.

The analysis pass deliberately decodes small and slow (320 px at 8 fps by default). Scene
boundaries do not need full resolution, and this keeps detection at a small fraction of the
cost of decoding the video properly.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...config import SceneConfig
from ...errors import FFmpegError
from ...interfaces import FrameContext, SceneSpan
from ...logging import get_logger
from ...media import ffmpeg
from .util import Boundary, boundaries_to_spans, budget_boundaries, enforce_min_duration

log = get_logger("scenes.ffmpeg")

_PTS_TIME = re.compile(r"pts_time:(-?[\d.]+)")
_SCORE = re.compile(r"lavfi\.scene_score=([\d.]+)")
_BLACK_START = re.compile(r"black_start:\s*([\d.]+)")
_BLACK_END = re.compile(r"black_end:\s*([\d.]+)")


class FFmpegSceneDetector:
    """Detect scenes with FFmpeg filters only — no OpenCV, no model, no GPU."""

    name = "ffmpeg"
    version = "1.0.0"

    def __init__(self, config: SceneConfig | None = None, *, timeout: float = 1800.0) -> None:
        self.config = config or SceneConfig()
        self.timeout = timeout

    # -- public API --------------------------------------------------------

    def available(self) -> bool:
        return ffmpeg.available()

    def detect(self, video_path: Path, ctx: FrameContext) -> list[SceneSpan]:
        duration = ctx.duration
        if duration <= 0:
            return []

        candidates, black_spans = self._analyse(video_path)
        if not candidates and not black_spans:
            # A genuinely single-shot video (or an analysis pass that found nothing) is one
            # scene, not zero — callers rely on scenes tiling the whole timeline.
            log.info("scenes.single_shot", extra={"duration": round(duration, 2)})
            return boundaries_to_spans([], duration)

        selected = budget_boundaries(
            candidates,
            duration=duration,
            absolute_threshold=self.config.threshold,
            max_per_minute=self.config.max_boundaries_per_minute,
        )
        for start, end in black_spans:
            # Both edges of a fade are boundaries: content ends at one and resumes at the other.
            selected.append(Boundary(start, 1.0, ["black"]))
            if end - start > 0.05:
                selected.append(Boundary(end, 1.0, ["black"]))

        kept = enforce_min_duration(
            selected, min_duration=self.config.min_scene_duration, duration=duration
        )
        spans = boundaries_to_spans(kept, duration)
        log.info(
            "scenes.detected",
            extra={
                "scenes": len(spans),
                "candidates": len(candidates),
                "selected": len(kept),
                "black_spans": len(black_spans),
            },
        )
        return spans

    # -- internals ---------------------------------------------------------

    def _filter_chain(self) -> str:
        cfg = self.config
        parts = [f"scale={cfg.analysis_width}:-2", f"fps={cfg.analysis_fps:g}"]
        if cfg.black_detect:
            parts.append("blackdetect=d=0.15:pic_th=0.97")
        # select before metadata=print so only candidate frames are reported.
        parts.append(f"select='gt(scene,{cfg.score_floor:g})'")
        parts.append("metadata=print:file=-")
        return ",".join(parts)

    def _analyse(self, video_path: Path) -> tuple[list[Boundary], list[tuple[float, float]]]:
        """One decode pass yielding difference candidates and black spans."""
        args = [
            *ffmpeg.input_args(video_path),
            "-an", "-sn", "-dn",
            "-vf", self._filter_chain(),
            "-f", "null", "-",
        ]
        try:
            # blackdetect reports through the log, metadata=print through stdout.
            result = ffmpeg.run(args, timeout=self.timeout, loglevel="info")
        except FFmpegError as exc:
            log.warning("scenes.analysis_failed", extra={"error": exc.message})
            return [], []

        candidates = self._parse_scores(result.stdout)
        black = self._parse_black(result.stderr)
        return candidates, black

    @staticmethod
    def _parse_scores(stdout: str) -> list[Boundary]:
        """Pair each ``frame:…pts_time:`` header with the score line that follows it."""
        boundaries: list[Boundary] = []
        pending_ts: float | None = None
        for line in stdout.splitlines():
            if (m := _PTS_TIME.search(line)) is not None:
                pending_ts = float(m.group(1))
                continue
            if (m := _SCORE.search(line)) is not None and pending_ts is not None:
                boundaries.append(Boundary(round(pending_ts, 3), float(m.group(1)), ["diff"]))
                pending_ts = None
        return boundaries

    @staticmethod
    def _parse_black(stderr: str) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        for line in stderr.splitlines():
            start_match = _BLACK_START.search(line)
            end_match = _BLACK_END.search(line)
            if start_match and end_match:
                start, end = float(start_match.group(1)), float(end_match.group(1))
                if end > start:
                    spans.append((round(start, 3), round(end, 3)))
        return spans


__all__ = ["FFmpegSceneDetector"]
