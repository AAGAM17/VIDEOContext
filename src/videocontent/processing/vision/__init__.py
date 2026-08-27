"""Vision providers — model-generated visual understanding.

This is the stage the brief is most insistent about keeping at arm's length (§2): the
abstraction is *Video → Context*, not *Video → some vendor → answer*. A vision provider is
one optional contributor to a document that is already useful without it. Everything a
``.vctx`` file needs to be searchable — transcript, temporal OCR, scenes, frames — comes from
local stages, and :class:`~videocontent.processing.vision.null.NullVision` is the default so
that a fresh install produces a complete document with no account, no key and no upload.

A provider implements :class:`~videocontent.interfaces.VisionProvider`: it takes a batch of
sampled frames and returns :class:`~videocontent.interfaces.VisionOutput` per batch. Two
obligations that are not optional for a remote one:

* ``remote = True``, so the stage record marks the run as having sent data off-machine and
  ``videocontent doctor`` can answer "what leaves this machine?" (spec §5, brief §32).
* it must batch. One call per frame is the cost mistake this layer exists to avoid (§27).
"""

from __future__ import annotations

from .null import NullVision
from .remote import GeminiVision, LocalVLMVision, OpenAIVision

__all__ = ["NullVision", "OpenAIVision", "GeminiVision", "LocalVLMVision"]
