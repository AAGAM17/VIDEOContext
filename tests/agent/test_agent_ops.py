"""The agent operations: resolution, coverage, bounds, labels, redaction, injection-as-data."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from videocontent.agent import ops
from videocontent.schema import io as sio
from videocontent.schema.v1 import OCRText

from .conftest import FAKE_KEY, INJECTION, bug_document


def dumps(payload: dict) -> str:
    return json.dumps(payload, default=str)


class TestLocate:
    def test_video_finds_sidecar(self, bug_video: Path):
        where = ops.locate(str(bug_video))
        assert where.kind == "video"
        assert where.vctx == bug_video.with_suffix(".vctx")

    def test_vctx_directly(self, bug_video: Path):
        where = ops.locate(str(bug_video.with_suffix(".vctx")))
        assert where.kind == "vctx" and where.vctx is not None

    def test_gzip_sidecar(self, tmp_path: Path):
        video = tmp_path / "z.mp4"
        video.write_bytes(b"x")
        sio.save(bug_document(), tmp_path / "z.vctx.gz", compress=True)
        assert ops.locate(str(video)).vctx == tmp_path / "z.vctx.gz"

    def test_unanalyzed_video(self, tmp_path: Path):
        video = tmp_path / "new.mp4"
        video.write_bytes(b"x")
        where = ops.locate(str(video))
        assert where.vctx is None
        assert where.default_output == tmp_path / "new.vctx"

    @pytest.mark.parametrize("ref", ["", "   ", "a\x00b"])
    def test_invalid_reference(self, ref: str):
        with pytest.raises(ops.AgentError):
            ops.locate(ref)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(ops.AgentError, match="no such file"):
            ops.locate(str(tmp_path / "nope.mp4"))

    def test_directory_is_not_a_video(self, tmp_path: Path):
        with pytest.raises(ops.AgentError, match="directory"):
            ops.locate(str(tmp_path))

    def test_url_without_document(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        where = ops.locate("https://example.com/media/talk.mp4?token=secret")
        assert where.kind == "url" and where.vctx is None
        assert where.default_output.name == "talk.vctx"

    def test_url_with_document(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sio.save(bug_document(), tmp_path / "talk.vctx")
        assert ops.locate("https://example.com/media/talk.mp4").vctx is not None

    @pytest.mark.parametrize("url", ["ftp://example.com/v.mp4", "file:///etc/passwd"])
    def test_disallowed_url_schemes_are_not_urls(self, url: str):
        # Not http(s): treated as a path, which does not exist, so it is refused.
        with pytest.raises(ops.AgentError):
            ops.locate(url)

    def test_workspace_boundary(self, bug_video: Path, tmp_path: Path):
        root = tmp_path / "inner"
        root.mkdir()
        with pytest.raises(ops.AgentError, match="outside the allowed workspace"):
            ops.locate(str(bug_video), root=root)
        with pytest.raises(ops.AgentError, match="outside"):
            ops.locate("../bug.mp4", root=root)

    def test_relative_paths_resolve_under_root(self, bug_video: Path):
        where = ops.locate("bug.mp4", root=bug_video.parent)
        assert where.vctx is not None


class TestNeverProcessesImplicitly:
    def test_query_on_unanalyzed_video_raises(self, tmp_path: Path):
        video = tmp_path / "new.mp4"
        video.write_bytes(b"x")
        for call in (lambda: ops.search(str(video), "x"), lambda: ops.ask(str(video), "x"),
                     lambda: ops.timeline(str(video)), lambda: ops.entities(str(video)),
                     lambda: ops.changes(str(video)), lambda: ops.context(str(video), "x")):
            with pytest.raises(ops.NotAnalyzedError) as info:
                call()
            assert "videocontent analyze" in (info.value.hint or "")
        assert not (tmp_path / "new.vctx").exists()

    def test_analyze_no_process(self, tmp_path: Path):
        video = tmp_path / "new.mp4"
        video.write_bytes(b"not really a video")
        report = ops.analyze(str(video), process=False)
        assert report["result"]["status"] == "not_analyzed"
        assert report["result"]["processed"] is False
        assert not (tmp_path / "new.vctx").exists()

    def test_analyze_reuses_existing(self, bug_video: Path):
        report = ops.analyze(str(bug_video))
        assert report["result"]["action"] == "reused"

    def test_inspect_unanalyzed_reports_next_step(self, tmp_path: Path):
        video = tmp_path / "new.mp4"
        video.write_bytes(b"not really a video")
        report = ops.inspect(str(video))
        assert report["result"]["status"] == "not_analyzed"
        assert report["result"]["next"] == f"videocontent analyze {video}"


class TestCoverage:
    def test_full_coverage(self, bug_video: Path):
        cov = ops.inspect(str(bug_video))["result"]["coverage"]
        assert cov["modalities"]["speech"]["count"] == 3
        assert cov["modalities"]["on_screen_text"]["count"] == 7
        missing = {m["capability"] for m in cov["missing"]}
        assert "speech" not in missing
        assert {"visual_descriptions", "objects"} <= missing

    def test_missing_speech_is_named_with_remedy(self, silent_video: Path):
        cov = ops.inspect(str(silent_video))["result"]["coverage"]
        gap = next(m for m in cov["missing"] if m["capability"] == "speech")
        assert gap["status"] == "skipped"
        assert "faster-whisper" in gap["how_to_add"]
        report = ops.analyze(str(silent_video))
        assert any("no speech" in w for w in report["warnings"])

    def test_subtitle_fallback_is_not_silence(self, tmp_path: Path):
        # No speech engine installed: the chain fell back to embedded subtitles, found none,
        # and recorded "ok". The audio was never listened to, so speech must read as missing.
        from videocontent.schema.v1 import StageRecord, StageStatus

        doc = bug_document(with_asr=False)
        doc.video.has_audio = True
        doc.stages = [s for s in doc.stages if s.name != "asr"] + [
            StageRecord(name="asr", status=StageStatus.OK, provider="subtitles")]
        video = tmp_path / "subs.mp4"
        video.write_bytes(b"x")
        sio.save(doc, tmp_path / "subs.vctx")
        cov = ops.inspect(str(video))["result"]["coverage"]
        gap = next(m for m in cov["missing"] if m["capability"] == "speech")
        assert gap["status"] == "not_transcribed"
        assert cov["modalities"]["speech"]["stage"] == "not_transcribed"

    def test_objects_are_never_guessed(self, bug_video: Path):
        cov = ops.inspect(str(bug_video))["result"]["coverage"]
        assert cov["modalities"]["objects"] == {"count": 0, "stage": "not_run"}


class TestEnvelope:
    @pytest.mark.parametrize("name", ["inspect", "analyze", "search", "ask", "timeline",
                                      "entities", "changes", "context", "explain"])
    def test_every_operation_shares_the_envelope(self, bug_video: Path, name: str):
        ref = str(bug_video)
        calls = {
            "inspect": lambda: ops.inspect(ref), "analyze": lambda: ops.analyze(ref),
            "search": lambda: ops.search(ref, "ConnectionError"),
            "ask": lambda: ops.ask(ref, "what happened after the ConnectionError"),
            "timeline": lambda: ops.timeline(ref), "entities": lambda: ops.entities(ref),
            "changes": lambda: ops.changes(ref),
            "context": lambda: ops.context(ref, "fix the migration bug"),
            "explain": lambda: ops.explain(ref, "evt_0001"),
        }
        out = calls[name]()
        assert out["schema"] == ops.AGENT_SCHEMA
        assert out["operation"] == name
        assert out["content_notice"] == ops.UNTRUSTED_NOTICE
        assert isinstance(out["warnings"], list)
        assert out["video"]["id"] == "vid_bug"
        assert out["video"]["vctx"].endswith("bug.vctx")
        json.loads(dumps(out))  # always serializable

    def test_compare_names_both_videos(self, bug_video: Path, silent_video: Path):
        out = ops.compare(str(bug_video), str(silent_video))
        assert [v["filename"] for v in out["videos"]] == ["bug.mp4", "silent.mp4"]
        assert out["result"]["kind"] == "derived"

    def test_compare_same_name_disambiguated(self, bug_video: Path):
        out = ops.compare(str(bug_video), str(bug_video.with_suffix(".vctx")))
        assert out["result"]["a"] != out["result"]["b"]


class TestEvidence:
    def test_search_labels_and_timestamps(self, bug_video: Path):
        result = ops.search(str(bug_video), "ConnectionError")["result"]
        assert result["returned"] >= 1
        top = result["spans"][0]
        assert top["timecode"] == "00:00:30.000"
        assert {s["kind"] for s in result["spans"]} <= {"observed", "detected"}
        assert all(s["ref_ids"] for s in result["spans"])

    def test_after_question_returns_later_evidence(self, bug_video: Path):
        result = ops.ask(str(bug_video), "what happened after the ConnectionError")["result"]
        assert result["temporal_relations"], "the temporal planner must fire"
        starts = [s["start"] for s in result["evidence"]]
        assert starts and min(starts) >= 30.0
        texts = " ".join(s["text"] for s in result["evidence"])
        assert "retry" in texts

    def test_ask_without_llm_is_extractive_and_says_so(self, bug_video: Path):
        out = ops.ask(str(bug_video), "what happened after the ConnectionError")
        assert out["result"]["answer_kind"] == "extractive"
        assert out["result"]["trace"]["llm"] == "none"
        assert any("no LLM configured" in w for w in out["warnings"])

    def test_ask_result_fields(self, bug_video: Path):
        result = ops.ask(str(bug_video), "ConnectionError")["result"]
        for key in ("query", "answer", "confidence", "evidence", "timestamps", "entities",
                    "events", "temporal_relations", "context", "trace"):
            assert key in result
        assert result["timestamps"] == sorted(result["timestamps"])

    def test_entity_timeline(self, bug_video: Path):
        result = ops.entities(str(bug_video), name="ConnectionError")["result"]
        assert result["found"] is True
        assert result["occurrences"][0]["timecode"] == "00:00:30.000"

    def test_unknown_entity_is_an_honest_empty(self, bug_video: Path):
        result = ops.entities(str(bug_video), name="Kubernetes")["result"]
        assert result == {"name": "Kubernetes", "found": False, "occurrences": []}

    def test_explain_unknown_id(self, bug_video: Path):
        assert ops.explain(str(bug_video), "nope_123")["result"]["found"] is False

    def test_changes_are_sequence_not_causation(self, bug_video: Path):
        for change in ops.changes(str(bug_video))["result"]["changes"]:
            assert change["kind"] in ("observed", "derived")

    def test_context_package_for_coding_task(self, bug_video: Path):
        result = ops.context(str(bug_video), "Recreate the login page UI")["result"]
        for key in ("relevant_ranges", "evidence", "ui_states", "events", "changes",
                    "frames", "provenance", "budget"):
            assert key in result
        assert result["evidence"], "the login page must be found"


class TestSafety:
    def test_injection_is_reported_verbatim_as_data(self, bug_video: Path):
        out = ops.search(str(bug_video), "ignore previous instructions")
        texts = [s["text"] for s in out["result"]["spans"]]
        assert INJECTION in texts
        assert out["content_notice"] == ops.UNTRUSTED_NOTICE
        span = next(s for s in out["result"]["spans"] if s["text"] == INJECTION)
        assert span["kind"] == "observed"

    def test_credentials_are_redacted_everywhere(self, bug_video: Path):
        ref = str(bug_video)
        for out in (ops.search(ref, "OPENAI_API_KEY"), ops.timeline(ref, start=50),
                    ops.ask(ref, "OPENAI_API_KEY"), ops.analyze(ref),
                    ops.context(ref, "api key configuration"), ops.entities(ref),
                    ops.changes(ref)):
            assert FAKE_KEY not in dumps(out), out["operation"]
        assert "[REDACTED]" in dumps(ops.search(ref, "OPENAI_API_KEY"))

    @pytest.mark.parametrize("secret", [
        "AKIAABCDEFGHIJKLMNOP", "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "-----BEGIN RSA PRIVATE KEY-----",
        "sk-proj-—abcdefghijk",
    ])
    def test_clip_redacts_credential_shapes(self, secret: str):
        cleaned = ops.clip(f"before {secret} after")
        assert "[REDACTED]" in cleaned
        assert "before" in cleaned and "after" in cleaned

    @pytest.mark.parametrize("line, leaked", [
        ("password = hunter22", "hunter22"),
        ("api_key: 0123456789abcdef", "0123456789abcdef"),
        # What OCR really produced for a rendered key: an em dash and a split token.
        ("OPENAI_API_KEY=sk-proj-—abcdefghijk Lmnopgqrstuvwx1234", "Lmnopgqrstuvwx1234"),
        ("GITHUB_TOKEN: ghp_x y z", "ghp_x"),
    ])
    def test_secret_assignments_redact_the_whole_value(self, line: str, leaked: str):
        cleaned = ops.clip(f"$ export {line}")
        assert leaked not in cleaned
        assert cleaned.startswith("$ export ") and cleaned.endswith("[REDACTED]")

    def test_clip_keeps_ordinary_text(self):
        text = "Revenue Rs 42L · email me at dev@example.com · port 5432"
        assert ops.clip(text) == text

    def test_url_with_embedded_credentials_is_refused(self, tmp_path: Path, monkeypatch):
        from videocontent.errors import VideoContextError

        monkeypatch.chdir(tmp_path)
        with pytest.raises(VideoContextError) as info:
            ops.inspect("https://user:hunter2@example.com/v.mp4")
        assert "hunter2" not in str(info.value)

    def test_url_tokens_never_echoed(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = ops.inspect("https://example.com/v.mp4?token=abc123secret&x=1")
        assert "abc123secret" not in dumps(out)


class TestBounds:
    def test_large_documents_are_capped(self, tmp_path: Path):
        doc = bug_document()
        doc.video.duration = 30100.0
        doc.ocr = [OCRText(id=f"ocr_{i:05d}", text=f"line{i} error " + "x" * 2000,
                           start=i * 10.0, end=i * 10.0 + 0.5, first_frame_ts=i * 10.0,
                           last_frame_ts=i * 10.0 + 0.5, stable=True, frame_count=1)
                   for i in range(3000)]
        video = tmp_path / "big.mp4"
        video.write_bytes(b"x")
        sio.save(doc, tmp_path / "big.vctx")
        ref = str(video)
        search = ops.search(ref, "error", top_k=10_000)["result"]
        assert search["returned"] <= ops.MAX_ITEMS and search["truncated"] is True
        assert all(len(s["text"]) <= ops.MAX_TEXT for s in search["spans"])
        timeline = ops.timeline(ref, top_k=10_000)["result"]
        assert timeline["returned"] <= ops.MAX_ITEMS
        for out in (ops.analyze(ref), ops.context(ref, "error"), ops.changes(ref)):
            assert len(dumps(out)) < 120_000, out["operation"]

    @pytest.mark.parametrize("value, expected", [
        (0, 1), (-5, 1), (10_000, ops.MAX_ITEMS), ("7", 7), ("x", ops.DEFAULT_ITEMS),
        (None, ops.DEFAULT_ITEMS)])
    def test_cap(self, value, expected):
        assert ops.cap(value) == expected

    @pytest.mark.parametrize("bad", ["", "   ", "x" * 5000])
    def test_malformed_queries(self, bug_video: Path, bad: str):
        with pytest.raises(ops.AgentError):
            ops.search(str(bug_video), bad)


class TestStaleness:
    def test_changed_video_is_flagged(self, bug_video: Path):
        vctx = bug_video.with_suffix(".vctx")
        old = time.time() - 100
        os.utime(vctx, (old, old))
        out = ops.search(str(bug_video), "ConnectionError")
        assert out["video"].get("stale") is True
        assert any("changed after" in w for w in out["warnings"])
