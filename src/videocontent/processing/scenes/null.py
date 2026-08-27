"""The no-op scene detector.

Useful in three real situations: content that is genuinely one continuous shot, a machine
without a working FFmpeg filter pass, and benchmarking the cost of scene detection by
removing it. It returns one scene covering the whole video rather than an empty list, so
downstream code that assumes scenes tile the timeline keeps working.
"""

from __future__ import annotations

from pathlib import Path

from ...interfaces import FrameContext, SceneSpan


class NullSceneDetector:
    name = "null"
    version = "1.0.0"

    def available(self) -> bool:
        return True

    def detect(self, video_path: Path, ctx: FrameContext) -> list[SceneSpan]:
        if ctx.duration <= 0:
            return []
        return [
            SceneSpan(
                start=0.0,
                end=round(ctx.duration, 3),
                score=None,
                signals=["none"],
                keyframe_ts=0.0,
            )
        ]


__all__ = ["NullSceneDetector"]
