"""The no-vision provider — and the default.

Unlike the other ``null`` engines, this one is not primarily a fallback: it is what most runs
should use. A document with scenes, a transcript and temporally deduplicated OCR answers the
queries in the brief's final product test (§51) without a single model call, and this provider
is how that stays the default rather than the degraded path.

It returns no descriptions. It does not return an empty *description* — describing a frame as
"" would be a claim about its contents. Nothing is claimed, and the stage record says the
stage was configured off rather than that it looked and saw nothing (spec §5).
"""

from __future__ import annotations

from ...config import VisionConfig
from ...interfaces import FrameContext, FrameImage, VisionOutput


class NullVision:
    name = "null"
    version = "1.0.0"
    remote = False

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()

    def available(self) -> bool:
        return True

    def describe(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[VisionOutput]:
        return []


__all__ = ["NullVision"]
