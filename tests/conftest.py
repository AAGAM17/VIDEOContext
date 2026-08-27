"""Shared fixtures and capability gates.

The OCR and media stages call real binaries. Tests that need one are *skipped* rather than
failed when it is absent, so a contributor without Tesseract still gets a meaningful run —
but the gate names the binary, so a skip is never mistaken for a pass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
DEMO_VIDEO = FIXTURES / "demo.mp4"
DEMO_MANIFEST = FIXTURES / "demo.manifest.json"


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


needs_tesseract = pytest.mark.skipif(not have("tesseract"), reason="tesseract not installed")
needs_ffmpeg = pytest.mark.skipif(not have("ffmpeg"), reason="ffmpeg not installed")
needs_demo = pytest.mark.skipif(not DEMO_VIDEO.is_file(), reason="run scripts/make_test_video.py")


@pytest.fixture(scope="session")
def manifest() -> dict:
    if not DEMO_MANIFEST.is_file():
        pytest.skip("demo manifest missing; run scripts/make_test_video.py")
    return json.loads(DEMO_MANIFEST.read_text())


@pytest.fixture(scope="session")
def demo_video() -> Path:
    if not DEMO_VIDEO.is_file():
        pytest.skip("demo video missing; run scripts/make_test_video.py")
    return DEMO_VIDEO
