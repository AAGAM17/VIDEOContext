"""Base classes for semantic profiles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..schema.v1 import VideoContextDocument


@dataclass
class ProfileContext:
    """Context passed to profile builders - contains all extracted evidence."""

    doc: VideoContextDocument
    config: dict[str, Any] | None = None


class SemanticProfile(ABC):
    """Base class for semantic profiles."""

    name: str
    """Unique profile name (e.g., 'ui_design', 'application')."""

    display_name: str
    """Human-readable name."""

    description: str
    """What this profile captures."""

    @abstractmethod
    def build(self, context: ProfileContext) -> Any:
        """Build the profile from the video context.

        Returns a structured representation (Pydantic model or dict).
        """
        pass

    def supports(self, context: ProfileContext) -> bool:
        """Whether this profile is applicable to the given video.

        Default: always supports. Override to check for domain relevance.
        """
        return True


class ProfileBuilder:
    """Wrapper that handles caching and lazy building."""

    def __init__(self, profile: SemanticProfile):
        self.profile = profile
        self._cache: dict[str, Any] = {}

    def build(self, context: ProfileContext, *, force: bool = False) -> Any:
        """Build the profile, using cache if available."""
        cache_key = self._cache_key(context)
        if not force and cache_key in self._cache:
            return self._cache[cache_key]

        result = self.profile.build(context)
        self._cache[cache_key] = result
        return result

    def _cache_key(self, context: ProfileContext) -> str:
        """Generate a cache key based on video content hash and profile version."""
        video_hash = context.doc.video.content_hash or "unknown"
        return f"{self.profile.name}:{video_hash}:{getattr(self.profile, 'version', '1')}"


__all__ = [
    "SemanticProfile",
    "ProfileContext",
    "ProfileBuilder",
]