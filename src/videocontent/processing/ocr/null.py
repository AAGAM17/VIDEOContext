"""The no-OCR engine.

Not a placeholder. It is how a caller turns OCR off *explicitly* — configuring
``ocr.provider = "null"`` records a stage that ran and found nothing, which the ``.vctx``
document distinguishes from a stage that was skipped or that crashed. It is also the engine
tests use when they need the pipeline to run without Tesseract installed.
"""

from __future__ import annotations

from ...config import OCRConfig
from ...interfaces import FrameContext, FrameImage, OCRObservation


class NullOCR:
    name = "null"
    version = "1.0.0"
    remote = False

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()

    def available(self) -> bool:
        return True

    def extract(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[OCRObservation]:
        return []


__all__ = ["NullOCR"]
