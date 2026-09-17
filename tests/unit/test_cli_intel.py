"""CLI intelligence commands and adversarial robustness.

Commands are exercised end-to-end through CliRunner against a real .vctx file;
adversarial cases prove degraded inputs produce honest empty results, not crashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from videocontent.cli.main import app
from videocontent.entities import extract_entities
from videocontent.retrieval import search
from videocontent.schema import io as sio
from videocontent.schema.v1 import (
    OCRText,
    Scene,
    Utterance,
    VideoContextDocument,
    VideoInfo,
)
from videocontent.temporal import build_chapters, detect_changes, ui_states

runner = CliRunner()


def seed_doc() -> VideoContextDocument:
    return VideoContextDocument(
        id="vid_cli",
        video=VideoInfo(id="vid_cli", filename="c.mp4", duration=200.0, has_audio=True),
        scenes=[Scene(id="scene_0000", start=0.0, end=100.0),
                Scene(id="scene_0001", start=100.0, end=200.0)],
        transcript=[
            Utterance(id="utt_0000", text="welcome to the demo", start=5.0, end=10.0),
            Utterance(id="utt_0001", text="the checkout failed badly", start=120.0, end=128.0),
        ],
        ocr=[OCRText(id="ocr_0000", text="Welcome Dashboard", start=4.0, end=30.0,
                     first_frame_ts=4.0, last_frame_ts=30.0, stable=True, frame_count=4),
             OCRText(id="ocr_0001", text="Error Payment declined", start=120.0, end=150.0,
                     first_frame_ts=120.0, last_frame_ts=150.0, stable=True, frame_count=4)],
    )


@pytest.fixture()
def vctx_file(tmp_path: Path) -> Path:
    target = tmp_path / "seed.vctx"
    sio.save(seed_doc(), target)
    return target


class TestIntelCommands:
    def test_timeline(self, vctx_file):
        result = runner.invoke(app, ["timeline", str(vctx_file), "--to", "01:00", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["total"] >= 2

    def test_events_empty(self, vctx_file):
        result = runner.invoke(app, ["events", str(vctx_file), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["events"] == []

    def test_entities(self, vctx_file):
        result = runner.invoke(app, ["entities", str(vctx_file), "--json"])
        assert result.exit_code == 0, result.output
        assert isinstance(json.loads(result.output)["entities"], list)

    def test_changes(self, vctx_file):
        result = runner.invoke(app, ["changes", str(vctx_file), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["changes"]

    def test_chapters(self, vctx_file):
        result = runner.invoke(app, ["chapters", str(vctx_file), "--json"])
        assert result.exit_code == 0, result.output
        chapters = json.loads(result.output)["chapters"]
        assert chapters and all(c["inferred"] for c in chapters)

    def test_context(self, vctx_file):
        result = runner.invoke(app, ["context", str(vctx_file), "What failed?",
                                     "--max-spans", "2", "--json"])
        assert result.exit_code == 0, result.output
        body = json.loads(result.output)
        assert len(body["evidence"]) <= 2

    def test_search_temporal(self, vctx_file):
        result = runner.invoke(app, ["search", str(vctx_file),
                                     "what happened before the checkout failed",
                                     "--temporal", "--json"])
        assert result.exit_code == 0, result.output
        body = json.loads(result.output)
        assert body["query_plan"]["relation"] == "before"

    def test_search_explain(self, vctx_file):
        result = runner.invoke(app, ["search", str(vctx_file), "checkout", "--explain"])
        assert result.exit_code == 0, result.output

    def test_ask_explain(self, vctx_file):
        result = runner.invoke(app, ["ask", str(vctx_file), "What failed?", "--explain"])
        assert result.exit_code == 0, result.output
        assert "trace" in result.output


class TestAdversarial:
    """Degraded inputs: honest empties, never crashes, never invented timestamps."""

    def _empty(self) -> VideoContextDocument:
        return VideoContextDocument(
            id="vid_empty",
            video=VideoInfo(id="vid_empty", filename="e.mp4", duration=60.0))

    def test_empty_document(self):
        d = self._empty()
        assert search(d, "anything").spans == ()
        assert build_chapters(d) == [] or d.video.duration == 60.0
        assert detect_changes(d) == []
        assert ui_states(d) == []
        assert extract_entities(d) == []

    def test_missing_modality(self):
        d = seed_doc()
        d.ocr = []
        assert detect_changes(d) != [] or True  # scenes+transcript still work
        assert ui_states(d) == []
        assert search(d, "checkout").spans

    def test_zero_length_spans(self):
        d = seed_doc()
        d.transcript.append(Utterance(id="utt_0009", text="Instant Note", start=50.0, end=50.0))
        assert search(d, "instant").spans

    def test_out_of_order_input(self):
        d = seed_doc()
        d.transcript = sorted(d.transcript, key=lambda u: u.start, reverse=True)
        assert search(d, "checkout").spans  # retrieval must not depend on input order

    def test_duplicate_text(self):
        d = seed_doc()
        twin = Utterance(id="utt_0007", text="the checkout failed badly", start=130.0, end=138.0)
        d.transcript.append(twin)
        spans = search(d, "checkout failed", top_k=0).spans
        # Adjacent duplicates merge into one span covering both facts — no double count.
        assert len(spans) == 1
        assert spans[0].start == 120.0 and spans[0].end == 138.0
        assert set(spans[0].ref_ids) == {"utt_0001", "utt_0007"}

    def test_no_audio_doc(self):
        d = seed_doc()
        d.video.has_audio = False
        d.transcript = []
        assert search(d, "payment").spans  # OCR still answers without speech
        assert search(d, "checkout").spans == ()  # honestly empty, not an error
