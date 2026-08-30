"""Extraction stages.

Each subpackage holds interchangeable adapters for one capability, resolved by name through
:mod:`videocontent.registry`. Stages are pure functions of (media, config): they never write
documents, never decide their own status, and never cache. The pipeline owns all of that
(ARCHITECTURE §2-3), which is what lets a stage fail without invalidating its neighbours.
"""

from __future__ import annotations

__all__: list[str] = []
