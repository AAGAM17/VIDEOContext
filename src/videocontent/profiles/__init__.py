"""Semantic Profiles — Domain-specific understanding of video content.

Profiles provide structured, domain-aware representations built on top of the
raw evidence (transcript, OCR, vision, events, frames). They are generated
lazily when requested and cached in the document.
"""

from __future__ import annotations

from .base import SemanticProfile, ProfileContext
from .ui_design import UIDesignProfileBuilder
from .application import ApplicationProfileBuilder
from .product_demo import ProductDemoProfileBuilder
from .tutorial import TutorialProfileBuilder

# Profile registry
PROFILE_BUILDERS = {
    "ui_design": UIDesignProfileBuilder,
    "application": ApplicationProfileBuilder,
    "product_demo": ProductDemoProfileBuilder,
    "tutorial": TutorialProfileBuilder,
}


def get_profile_builder(profile_name: str):
    """Get a profile builder by name."""
    builder_cls = PROFILE_BUILDERS.get(profile_name)
    if builder_cls is None:
        raise ValueError(f"Unknown profile: {profile_name}")
    return builder_cls


def list_profiles() -> list[str]:
    """List available profile names."""
    return list(PROFILE_BUILDERS.keys())


__all__ = [
    "SemanticProfile",
    "ProfileContext",
    "PROFILE_BUILDERS",
    "get_profile_builder",
    "list_profiles",
    "UIDesignProfileBuilder",
    "ApplicationProfileBuilder",
    "ProductDemoProfileBuilder",
    "TutorialProfileBuilder",
]