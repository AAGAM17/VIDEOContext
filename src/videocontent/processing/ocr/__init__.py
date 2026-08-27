"""On-screen text extraction.

Two steps, deliberately separate:

1. an :class:`~videocontent.interfaces.OCREngine` reads text out of individual frames, and
2. :func:`deduplicate` turns those per-frame readings into temporal events.

Keeping them apart is what makes the engine replaceable. A contributor adding PaddleOCR,
EasyOCR or a cloud vision API writes step 1 only — a class with ``available()`` and
``extract()`` — and inherits the temporal model, the confidence handling and the span
arithmetic unchanged. Nothing about deduplication is Tesseract-specific.
"""

from __future__ import annotations

from .dedupe import deduplicate, iou, normalize, similarity

__all__ = ["deduplicate", "iou", "normalize", "similarity"]
