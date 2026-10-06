"""Regressions for bugs that type-checking found but no test exercised."""

from __future__ import annotations

import pytest

from videocontent.config import VisionConfig
from videocontent.errors import StageError
from videocontent.processing.pipeline import _require_doc
from videocontent.processing.vision.remote import OpenAIVision
from videocontent.sources.adapters import _suffix_for
from videocontent.sources.types import SourceInspection


def test_stage_error_names_stage_when_document_missing():
    class _Run:
        doc = None

    with pytest.raises(StageError) as exc:
        _require_doc(_Run(), "events")  # type: ignore[arg-type]
    assert exc.value.stage == "events"
    assert exc.value.hint


def test_openai_vision_reads_config_without_attribute_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIVision(VisionConfig()).available() is False
    assert OpenAIVision(VisionConfig(api_key="k")).available() is True


def test_source_inspection_hint_is_a_declared_field():
    assert "hint" in SourceInspection.model_fields


def test_suffix_for_uses_url_path():
    assert _suffix_for("https://x.test/a/clip.MP4?token=1", {}) == ".mp4"


def test_select_context_builds_requested_profiles():
    from pathlib import Path

    from videocontent.profiles import list_profiles
    from videocontent.routing import ContextBudget, classify_task, select_context
    from videocontent.schema.io import load

    demo = Path(__file__).parent.parent / "fixtures" / "demo.vctx"
    if not demo.exists():
        pytest.skip("demo fixture missing")
    doc = load(demo)
    task = classify_task("summarize this video")
    task.suggested_profiles = list_profiles()
    sel = select_context(doc, task, ContextBudget())
    assert sel["profiles"], "profiles were requested but none were built"
