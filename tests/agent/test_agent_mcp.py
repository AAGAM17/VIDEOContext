"""The ``videocontent mcp`` server: protocol, tool contract, validation and boundaries."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from videocontent.agent import mcp_server as srv
from videocontent.agent import ops
from videocontent.config import ProcessingConfig

from .conftest import FAKE_KEY, INJECTION


@pytest.fixture()
def server(bug_video: Path) -> srv.Server:
    return srv.Server(srv.Context(bug_video.parent, ProcessingConfig()))


def rpc(server: srv.Server, method: str, params: dict | None = None, msg_id: int = 1):
    message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def call(server: srv.Server, name: str, arguments):
    response = rpc(server, "tools/call", {"name": name, "arguments": arguments})
    result = response["result"]
    return json.loads(result["content"][0]["text"]), result["isError"], result


class TestProtocol:
    @pytest.mark.parametrize("version", srv.PROTOCOL_VERSIONS)
    def test_initialize_echoes_supported_versions(self, server, version):
        result = rpc(server, "initialize", {"protocolVersion": version,
                                            "capabilities": {}})["result"]
        assert result["protocolVersion"] == version
        assert result["capabilities"] == {"tools": {"listChanged": False}}
        assert result["serverInfo"]["name"] == "videocontent"
        assert "untrusted" in result["instructions"]

    def test_unknown_version_gets_latest(self, server):
        result = rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]
        assert result["protocolVersion"] == srv.PROTOCOL_VERSIONS[-1]

    def test_notifications_get_no_reply(self, server):
        assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_unknown_method(self, server):
        assert rpc(server, "sampling/createMessage")["error"]["code"] == -32601

    def test_invalid_requests(self, server):
        assert server.handle({"id": 1, "method": "ping"})["error"]["code"] == -32600
        assert server.handle(["not", "a", "dict"])["error"]["code"] == -32600
        assert rpc(server, "tools/call", {"arguments": {}})["error"]["code"] == -32602

    def test_ping_and_empty_lists(self, server):
        assert rpc(server, "ping")["result"] == {}
        assert rpc(server, "resources/list")["result"] == {"resources": []}
        assert rpc(server, "prompts/list")["result"] == {"prompts": []}

    def test_serve_loop_handles_garbage_and_batches(self, server):
        lines = "\n".join([
            "this is not json",
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
            json.dumps([{"jsonrpc": "2.0", "id": 2, "method": "ping"},
                        {"jsonrpc": "2.0", "method": "notifications/initialized"}]),
            "",
        ])
        out = io.StringIO()
        server.serve(io.StringIO(lines), out)
        replies = [json.loads(line) for line in out.getvalue().splitlines()]
        assert replies[0]["error"]["code"] == -32700
        assert replies[1] == {"jsonrpc": "2.0", "id": 1, "result": {}}
        assert replies[2] == [{"jsonrpc": "2.0", "id": 2, "result": {}}]

    def test_structured_content_only_for_new_protocols(self, server):
        rpc(server, "initialize", {"protocolVersion": "2024-11-05"})
        _, _, raw = call(server, "videocontent_inspect", {"video": "bug.mp4"})
        assert "structuredContent" not in raw
        rpc(server, "initialize", {"protocolVersion": "2025-06-18"})
        _, _, raw = call(server, "videocontent_inspect", {"video": "bug.mp4"})
        assert raw["structuredContent"]["operation"] == "inspect"


class TestTools:
    EXPECTED = ["videocontent_inspect", "videocontent_analyze", "videocontent_search",
                "videocontent_ask", "videocontent_timeline", "videocontent_entities",
                "videocontent_changes", "videocontent_context", "videocontent_compare",
                "videocontent_explain"]

    def test_tool_list(self, server):
        tools = rpc(server, "tools/list")["result"]["tools"]
        assert [t["name"] for t in tools] == self.EXPECTED
        for tool in tools:
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            assert "video" in schema["properties"]
            assert tool["description"]
        writes = [t["name"] for t in tools if not t["annotations"]["readOnlyHint"]]
        assert writes == ["videocontent_analyze"]

    @pytest.mark.parametrize("name, arguments", [
        ("videocontent_inspect", {"video": "bug.mp4"}),
        ("videocontent_analyze", {"video": "bug.mp4"}),
        ("videocontent_search", {"video": "bug.mp4", "query": "ConnectionError"}),
        ("videocontent_ask", {"video": "bug.mp4",
                              "question": "what happened after the ConnectionError"}),
        ("videocontent_timeline", {"video": "bug.mp4", "start": 25, "end": 45}),
        ("videocontent_entities", {"video": "bug.mp4", "name": "ConnectionError"}),
        ("videocontent_changes", {"video": "bug.mp4"}),
        ("videocontent_context", {"video": "bug.mp4", "task": "rebuild the login page"}),
        ("videocontent_compare", {"video": "bug.mp4", "other_video": "silent.mp4"}),
        ("videocontent_explain", {"video": "bug.mp4", "id": "evt_0001"}),
    ])
    def test_every_tool(self, server, silent_video, name, arguments):
        payload, is_error, _ = call(server, name, arguments)
        assert not is_error, payload
        assert payload["schema"] == ops.AGENT_SCHEMA
        assert payload["content_notice"] == ops.UNTRUSTED_NOTICE
        assert FAKE_KEY not in json.dumps(payload)

    def test_analyze_without_consent_never_processes(self, server, bug_video):
        fresh = bug_video.parent / "fresh.mp4"
        fresh.write_bytes(b"x")
        payload, is_error, _ = call(server, "videocontent_analyze", {"video": "fresh.mp4"})
        assert not is_error
        assert payload["result"]["status"] == "not_analyzed"
        assert not (bug_video.parent / "fresh.vctx").exists()

    def test_unanalyzed_query_is_an_error_with_hint(self, server, bug_video):
        (bug_video.parent / "fresh.mp4").write_bytes(b"x")
        payload, is_error, _ = call(server, "videocontent_search",
                                    {"video": "fresh.mp4", "query": "x"})
        assert is_error
        assert payload["error"]["type"] == "NotAnalyzedError"
        assert "analyze" in payload["error"]["hint"]

    def test_injection_stays_data(self, server):
        payload, is_error, _ = call(server, "videocontent_search",
                                    {"video": "bug.mp4", "query": "ignore previous instructions"})
        assert not is_error
        assert INJECTION in [s["text"] for s in payload["result"]["spans"]]


class TestValidation:
    @pytest.mark.parametrize("arguments, message", [
        ({"video": "bug.mp4", "query": "x", "shell": "rm -rf /"}, "unknown argument"),
        ({"video": "bug.mp4"}, "missing required argument: query"),
        ({"video": 3, "query": "x"}, "must be of type string"),
        ({"video": "bug.mp4", "query": "x", "top_k": 0}, "out of range"),
        ({"video": "bug.mp4", "query": "x", "top_k": 51}, "out of range"),
        ({"video": "bug.mp4", "query": "x", "top_k": True}, "must be of type integer"),
        ({"video": "bug.mp4", "query": "x", "top_k": "5"}, "must be of type integer"),
        ({"video": "bug.mp4", "query": "x", "modalities": ["ocr", "brain"]}, "must be in"),
        ({"video": "bug.mp4", "query": "x", "modalities": "ocr"}, "must be of type array"),
    ])
    def test_malformed_arguments(self, server, arguments, message):
        payload, is_error, _ = call(server, "videocontent_search", arguments)
        assert is_error
        assert message in payload["error"]["message"]

    def test_arguments_must_be_an_object(self, server):
        payload, is_error, _ = call(server, "videocontent_inspect", ["bug.mp4"])
        assert is_error and "object" in payload["error"]["message"]

    def test_unknown_tool(self, server):
        payload, is_error, _ = call(server, "run_shell", {"cmd": "id"})
        assert is_error and "unknown tool" in payload["error"]["message"]

    @pytest.mark.parametrize("video", ["../../etc/passwd", "/etc/passwd",
                                       "../outside.mp4", "~/.ssh/id_rsa"])
    def test_workspace_boundary(self, server, video):
        payload, is_error, _ = call(server, "videocontent_inspect", {"video": video})
        assert is_error
        assert payload["error"]["type"] == "AgentError"

    def test_internal_errors_do_not_leak_messages(self, server, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("/secret/path/in/message")

        monkeypatch.setitem(srv.TOOLS["videocontent_inspect"], "handler", boom)
        payload, is_error, _ = call(server, "videocontent_inspect", {"video": "bug.mp4"})
        assert is_error
        assert payload["error"] == {"type": "RuntimeError", "message": "internal error"}

    def test_oversized_results_are_refused(self, server, monkeypatch):
        monkeypatch.setattr(srv, "MAX_RESULT_BYTES", 200)
        payload, is_error, _ = call(server, "videocontent_inspect", {"video": "bug.mp4"})
        assert is_error
        assert "too large" in payload["error"]["message"]


def test_stdio_roundtrip(bug_video: Path):
    """The real process: ``python -m videocontent mcp`` over pipes."""
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "videocontent_ask",
                    "arguments": {"video": "bug.mp4",
                                  "question": "what happened after the ConnectionError"}}},
    ]
    done = subprocess.run(
        [sys.executable, "-m", "videocontent", "mcp", "--root", str(bug_video.parent)],
        input="\n".join(json.dumps(m) for m in messages) + "\n",
        capture_output=True, text=True, timeout=120, check=True)
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3]
    assert replies[0]["result"]["protocolVersion"] == "2025-06-18"
    assert len(replies[1]["result"]["tools"]) == len(srv.TOOLS)
    answer = replies[2]["result"]["structuredContent"]["result"]
    assert answer["evidence"] and min(e["start"] for e in answer["evidence"]) >= 30.0
