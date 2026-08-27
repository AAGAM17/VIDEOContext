"""Temporal events — the typed layer that makes a video queryable rather than described.

An event says *something happened at this time*, and per the spec it must say which facts it
came from: every event carries ``refs`` pointing at the scene, OCR event or utterance IDs that
produced it (spec §11). That is what makes the layer auditable, and it is why the rules here
derive events only from facts already in the document. Nothing is inferred from the video a
second time, so an event can never claim a moment that the modalities beneath it do not
support.

The taxonomy is open — a custom detector may emit any ``type`` string. What it may not do is
emit an event with empty ``refs``.
"""

from __future__ import annotations

from .rules import RuleEventDetector

__all__ = ["RuleEventDetector"]
