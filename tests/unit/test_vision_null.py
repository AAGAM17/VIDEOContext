"""The default vision provider: the one that doesn't call anybody.

Small file, load-bearing promise. The brief's core principle (§2) is that a ``.vctx`` document
is useful without a model vendor, and §32 requires that nothing leaves the machine unless the
operator asked for it. Both of those reduce to a checkable fact — the *default* vision provider
is this one, it is marked ``remote = False``, and it returns nothing rather than describing a
frame it never looked at.
"""

from __future__ import annotations

from pathlib import Path

from videocontent.config import ProcessingConfig, VisionConfig
from videocontent.interfaces import FrameContext, FrameImage, VisionProvider
from videocontent.processing.vision import NullVision
from videocontent.registry import create


def frames(count: int = 3) -> list[FrameImage]:
    # The paths are never opened: a provider that returns no descriptions has no reason to read
    # a frame, and a test that had to write real JPEGs to prove that would be testing nothing.
    return [
        FrameImage(ts=float(i) * 2, path=Path(f"frames/frame_{i:04d}.jpg"), index=i)
        for i in range(count)
    ]


def ctx() -> FrameContext:
    return FrameContext(duration=60.0, fps=30.0, width=1280, height=720)


class TestDefaults:
    def test_vision_is_off_by_default(self):
        assert ProcessingConfig().vision.enabled is False

    def test_the_default_provider_is_null(self):
        # If this ever changes, a fresh install starts uploading frames somewhere.
        assert ProcessingConfig().vision.provider == "null"


class TestNullVision:
    def test_it_is_always_available(self):
        # It has no dependency to be missing, which is what makes it usable as the fallback
        # when a configured provider cannot be built.
        assert NullVision().available() is True

    def test_it_is_not_remote(self):
        assert NullVision().remote is False

    def test_it_names_and_versions_itself(self):
        engine = NullVision()
        assert engine.name == "null"
        assert engine.version == "1.0.0"

    def test_it_describes_nothing(self):
        # Not an empty description — no description. Calling a frame "" would be a claim about
        # its contents.
        assert NullVision().describe(frames(), ctx()) == []

    def test_no_frames_is_also_nothing(self):
        assert NullVision().describe([], ctx()) == []

    def test_it_accepts_a_config(self):
        config = VisionConfig(provider="null", max_frames=12)
        assert NullVision(config).config.max_frames == 12

    def test_it_builds_its_own_config_when_given_none(self):
        assert NullVision().config.max_frames == VisionConfig().max_frames

    def test_it_satisfies_the_provider_protocol(self):
        # The plugin contract from §25: a third-party provider is accepted on the strength of
        # its shape, so the built-in has to have that shape too.
        assert isinstance(NullVision(), VisionProvider)


class TestRegistry:
    def test_it_resolves_through_the_registry(self):
        # Regression: the registry declared this module before it existed, so `create` raised
        # ModuleNotFoundError for the default provider.
        engine = create("vision", "null")
        assert engine.name == "null"

    def test_the_registry_passes_the_config_through(self):
        engine = create("vision", "null", config=VisionConfig(max_frames=5))
        assert engine.config.max_frames == 5
