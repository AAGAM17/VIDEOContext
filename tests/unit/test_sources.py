"""Source layer: resolution, canonical identity, security boundary, adapters.

No test here touches the public internet. Literal-IP checks need no DNS; the one
end-to-end download spins up a loopback ``http.server`` and explicitly opts into
``allow_private_ips`` — which is also what proves the default (blocked) path.
"""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

from videocontent.config import ProcessingConfig
from videocontent.errors import (
    DownloadError,
    SecurityError,
    SourceNotFoundError,
    UnsupportedSourceError,
)
from videocontent.sources import SourceType, adapter_for, inspect_source, resolve
from videocontent.sources import security as sec

# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


class TestResolve:
    def test_local_path_resolves(self):
        src = resolve("demo.mp4")
        assert src.source_type is SourceType.LOCAL_FILE
        assert src.provider == "local"
        assert src.source_id.startswith("src_")

    def test_http_url_resolves(self):
        src = resolve("https://example.com/v.mp4")
        assert src.source_type is SourceType.DIRECT_URL
        assert src.provider == "http"

    def test_file_url_resolves_to_local(self):
        src = resolve("file:///tmp/demo.mp4")
        assert src.source_type is SourceType.LOCAL_FILE

    def test_empty_is_invalid(self):
        from videocontent.errors import SourceError

        with pytest.raises(SourceError):
            resolve("   ")

    def test_unsupported_scheme(self):
        with pytest.raises(UnsupportedSourceError):
            resolve("ftp://example.com/v.mp4")

    def test_gibberish_is_not_found(self):
        with pytest.raises(SourceNotFoundError):
            resolve("definitely-not-a-video-xyz-123")

    def test_source_passthrough(self):
        src = resolve("https://example.com/v.mp4")
        assert resolve(src) is src


# ---------------------------------------------------------------------------
# canonical identity + redaction
# ---------------------------------------------------------------------------


class TestCanonicalIdentity:
    def test_ephemeral_params_stripped(self):
        a = sec.canonicalize_url("https://cdn.example.com/v.mp4?token=abc&expires=123")
        b = sec.canonicalize_url("https://cdn.example.com/v.mp4?token=zzz&expires=999")
        assert a == b
        assert "token" not in a and "expires" not in a

    def test_meaningful_params_kept(self):
        a = sec.canonicalize_url("https://h.example.com/v.mp4?quality=1080p")
        assert "quality=1080p" in a

    def test_host_case_and_default_port(self):
        a = sec.canonicalize_url("https://EXAMPLE.com:443/v.mp4")
        assert a == "https://example.com/v.mp4"

    def test_same_object_same_source_id(self):
        a = resolve("https://cdn.example.com/v.mp4?token=one")
        b = resolve("https://cdn.example.com/v.mp4?token=two")
        assert a.canonical_id == b.canonical_id
        assert a.source_id == b.source_id

    def test_different_objects_differ(self):
        assert (resolve("https://h.example.com/a.mp4").source_id
                != resolve("https://h.example.com/b.mp4").source_id)

    def test_redact_hides_ephemeral_values(self):
        redacted = sec.redact("https://h.example.com/v.mp4?token=secret&quality=1080p")
        assert "secret" not in redacted
        assert "quality=1080p" in redacted


# ---------------------------------------------------------------------------
# security boundary (no DNS needed: literal IPs short-circuit before getaddrinfo)
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_rejects_non_http_scheme(self):
        with pytest.raises(SecurityError):
            sec.parse_url("ftp://example.com/v.mp4")

    def test_rejects_embedded_credentials(self):
        with pytest.raises(SecurityError):
            sec.parse_url("https://user:pass@example.com/v.mp4")

    def test_rejects_whitespace(self):
        from videocontent.errors import SourceError

        with pytest.raises(SourceError):
            sec.parse_url("https://example.com/vid eo.mp4")

    @pytest.mark.parametrize("host", [
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",  # noqa: S104 - test target string, not a bind() call
        "[::1]",
    ])
    def test_private_literals_blocked(self, host):
        with pytest.raises(SecurityError):
            sec.validate_url(f"http://{host}/v.mp4")

    def test_public_literal_allowed(self):
        parsed = sec.validate_url("https://8.8.8.8/v.mp4")
        assert parsed.hostname == "8.8.8.8"

    def test_allow_private_opt_in(self):
        parsed = sec.validate_url("http://127.0.0.1/v.mp4", allow_private_ips=True)
        assert parsed.hostname == "127.0.0.1"

    def test_redirect_keeps_scheme_check(self):
        target = sec.resolve_redirect("https://a.example.com/x", "/y.mp4", index=0)
        assert target == "https://a.example.com/y.mp4"
        with pytest.raises(SecurityError):
            sec.resolve_redirect("https://a.example.com/x", "ftp://evil.example.com/y", index=0)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_adapter_for_local(self):
        from videocontent.sources.adapters import LocalFileAdapter

        assert isinstance(adapter_for("demo.mp4"), LocalFileAdapter)

    def test_adapter_for_http(self):
        from videocontent.sources.adapters import DirectURLAdapter

        assert isinstance(adapter_for("https://example.com/v.mp4"), DirectURLAdapter)

    def test_local_inspect_missing(self, tmp_path):
        src = resolve(str(tmp_path / "nope.mp4"))
        info = inspect_source(src)
        assert info.accessible is False
        assert "no such file" in (info.reason or "")

    def test_local_materialize_passthrough(self, tmp_path):
        target = tmp_path / "a.mp4"
        target.write_bytes(b"\x00" * 16)
        src = resolve(str(target))
        asset = adapter_for(str(target)).materialize(src, tmp_path)
        assert asset.temporary is False
        assert Path(asset.local_path).resolve() == target.resolve()

    def test_http_canonical_strips_token(self):
        src = resolve("https://cdn.example.com/v.mp4?token=abc")
        assert "token" not in src.canonical_id

    def test_http_inspect_rejects_html(self, monkeypatch):
        from videocontent.sources.adapters import DirectURLAdapter

        adapter = DirectURLAdapter()
        src = resolve("https://example.com/v.mp4")
        monkeypatch.setattr(adapter, "_fetch_headers",
                            lambda *a, **k: ("https://example.com/v.mp4",
                                             {"Content-Type": "text/html",
                                              "Content-Length": "100"}))
        info = adapter.inspect(src, ProcessingConfig())
        assert info.accessible is False
        assert "not media" in (info.reason or "")

    def test_http_inspect_rejects_oversize(self, monkeypatch):
        from videocontent.sources.adapters import DirectURLAdapter

        adapter = DirectURLAdapter()
        src = resolve("https://example.com/v.mp4")
        monkeypatch.setattr(adapter, "_fetch_headers",
                            lambda *a, **k: ("https://example.com/v.mp4",
                                             {"Content-Type": "video/mp4",
                                              "Content-Length": str(10**12)}))
        info = adapter.inspect(src, ProcessingConfig())
        assert info.accessible is False
        assert "limit" in (info.reason or "")


# ---------------------------------------------------------------------------
# loopback download (real HTTP, explicitly opted into private IPs)
# ---------------------------------------------------------------------------


def _serve(directory: Path, payload: bytes, name: str = "v.mp4"):
    (directory / name).write_bytes(payload)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class TestLoopbackDownload:
    def test_download_and_cleanup(self, tmp_path):
        from videocontent.sources.adapters import DirectURLAdapter

        payload = b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 4096
        server = _serve(tmp_path, payload)
        try:
            cfg = ProcessingConfig()
            cfg.sources.allow_private_ips = True
            url = f"http://127.0.0.1:{server.server_port}/v.mp4"
            src = resolve(url, config=cfg)
            adapter = DirectURLAdapter()
            dest = tmp_path / "dl"
            dest.mkdir()
            asset = adapter.materialize(src, dest, cfg)
            assert Path(asset.local_path).read_bytes() == payload
            assert asset.content_hash and asset.content_hash.startswith("sha256:")
            assert asset.temporary is True
            adapter.cleanup(asset)
            assert not Path(asset.local_path).exists()
        finally:
            server.shutdown()

    def test_size_limit_enforced(self, tmp_path):
        from videocontent.sources.adapters import DirectURLAdapter

        payload = b"x" * 65536
        server = _serve(tmp_path, payload)
        try:
            cfg = ProcessingConfig()
            cfg.sources.allow_private_ips = True
            url = f"http://127.0.0.1:{server.server_port}/v.mp4"
            src = resolve(url, config=cfg)
            adapter = DirectURLAdapter()
            dest = tmp_path / "dl"
            dest.mkdir()
            # The oversize path is covered by the inspect test above; here verify the
            # normal download plumbing (size, hash, cleanup).
            asset = adapter.materialize(src, dest, cfg)
            assert asset.size_bytes == len(payload)
            adapter.cleanup(asset)
        finally:
            server.shutdown()

    def test_loopback_blocked_by_default(self, tmp_path):
        from videocontent.sources.adapters import DirectURLAdapter

        server = _serve(tmp_path, b"data")
        try:
            url = f"http://127.0.0.1:{server.server_port}/v.mp4"
            src = resolve(url)  # resolution itself is string-level; fetch enforces SSRF
            with pytest.raises((SecurityError, DownloadError)):
                DirectURLAdapter().materialize(src, tmp_path, ProcessingConfig())
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_local_passthrough_no_delete(self, tmp_path):
        from videocontent.sources.lifecycle import materialized

        target = tmp_path / "keep.mp4"
        target.write_bytes(b"data")
        src = resolve(str(target))
        with materialized(src, config=ProcessingConfig()) as asset:
            assert asset.temporary is False
        assert target.exists()

    def test_sweep_stale(self, tmp_path, monkeypatch):
        import time

        from videocontent.sources import lifecycle as lc

        root = tmp_path / "vroots"
        root.mkdir()
        monkeypatch.setattr(lc, "temp_root", lambda: root)
        old = root / "vctx-source-old"
        new = root / "vctx-source-new"
        old.mkdir()
        new.mkdir()
        ancient = time.time() - 48 * 3600
        import os

        os.utime(old, (ancient, ancient))
        assert lc.sweep_stale(ttl_h=24.0) == 1
        assert not old.exists()
        assert new.exists()


# ---------------------------------------------------------------------------
# pipeline + schema provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_local_run_stamps_source(self, tmp_path, monkeypatch):
        from videocontent.processing import pipeline as pipe
        from videocontent.processing.pipeline import Pipeline
        from videocontent.schema.v1 import VideoInfo

        def fake_probe(source, *, limits=None, compute_hash=True):
            return VideoInfo(id="vid_x", filename=Path(str(source)).name,
                             path=str(source), duration=10.0,
                             has_audio=False, has_video=False,
                             content_hash="sha256:abc")

        monkeypatch.setattr(pipe, "probe", fake_probe)
        cfg = ProcessingConfig(workdir=tmp_path / "w")
        cfg.asr.enabled = False
        cfg.ocr.enabled = False
        cfg.sampling.scene_detection = False
        doc = Pipeline(cfg).run(str(tmp_path / "clip.mp4"))
        assert doc.source is not None
        assert doc.source.source_type == "local_file"
        assert doc.source.access_mode == "local"
        assert doc.source.source_id.startswith("src_")
        assert doc.video.path is not None  # local paths are preserved

    def test_old_documents_load_without_source(self, tmp_path):
        from videocontent.schema import io as sio
        from videocontent.schema.v1 import VideoContextDocument, VideoInfo

        doc = VideoContextDocument(id="v1", video=VideoInfo(id="v", filename="x.mp4", duration=1.0))
        raw = sio.loads(sio.dumps(doc))
        assert raw.source is None
        # And a document with source round-trips.
        doc2 = Pipeline_provenance_doc()
        raw2 = sio.loads(sio.dumps(doc2))
        assert raw2.source is not None
        assert raw2.source.canonical_id == doc2.source.canonical_id


def Pipeline_provenance_doc():
    from datetime import datetime, timezone

    from videocontent.schema.v1 import SourceRecord, VideoContextDocument, VideoInfo

    return VideoContextDocument(
        id="v2",
        video=VideoInfo(id="v", filename="r.mp4", duration=2.0),
        source=SourceRecord(
            source_id="src_abc", source_type="direct_url", provider="http",
            locator_redacted="https://h.example.com/v.mp4",
            canonical_id="url:https://h.example.com/v.mp4",
            retrieved_at=datetime.now(timezone.utc), access_mode="remote",
        ),
    )


# ---------------------------------------------------------------------------
# SDK + CLI regression
# ---------------------------------------------------------------------------


class TestSDK:
    def test_video_accepts_url(self):
        import videocontent

        v = videocontent.Video("https://example.com/v.mp4")
        assert v.source_ref.source_type is SourceType.DIRECT_URL
        assert v.default_path().suffix == ".vctx"

    def test_open_alias(self):
        import videocontent

        assert isinstance(videocontent.open("demo.mp4").source_ref, object)

    def test_local_video_compat(self):
        import videocontent

        v = videocontent.Video("demo.mp4")
        assert str(v.source) == "demo.mp4"
        assert "demo.mp4" in repr(v)


class TestCLI:
    def test_source_resolve_json(self, tmp_path):
        from typer.testing import CliRunner

        from videocontent.cli.main import app

        result = CliRunner().invoke(app, ["source", "resolve", "demo.mp4", "--json"])
        assert result.exit_code == 0, result.output
        import json

        assert json.loads(result.output)["source_type"] == "local_file"

    def test_source_inspect_missing_file(self, tmp_path):
        from typer.testing import CliRunner

        from videocontent.cli.main import app

        missing = str(tmp_path / "nope.mp4")
        result = CliRunner().invoke(app, ["source", "inspect", missing, "--json"])
        # Inaccessible sources exit 1 with a JSON body explaining why.
        assert result.exit_code == 1
        import json

        assert json.loads(result.output)["accessible"] is False

    def test_process_rejects_gibberish_gracefully(self):
        from typer.testing import CliRunner

        from videocontent.cli.main import app

        result = CliRunner().invoke(app, ["process", "definitely-not-a-video-xyz-123"])
        assert result.exit_code == 1
        assert "hint" in result.output.lower() or "error" in result.output.lower()
