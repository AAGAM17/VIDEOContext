"""Clean install, end to end: wheel → fresh venv → skill → analyze → ask → MCP.

This is the path a new developer takes, run for real in an isolated environment:

    clean environment (new venv, new HOME, new working directory)
      → pip install the built wheel (base install: no ASR, no extras)
      → videocontent init-agent           (skill comes from the wheel, not the repo)
      → videocontent inspect demo.mp4     (not analyzed; nothing processed)
      → videocontent analyze demo.mp4     (processes locally, once)
      → videocontent ask demo.mp4 "..."   (evidence-backed answer)
      → videocontent mcp                  (MCP handshake + tool call over stdio)

Opt-in (it builds a wheel and installs dependencies): ``VIDEOCONTENT_INSTALL_TEST=1``.
Needs ``uv``, ffmpeg and tesseract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import DEMO_VIDEO, needs_demo, needs_ffmpeg, needs_tesseract

REPO = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.slow, pytest.mark.integration, needs_ffmpeg, needs_tesseract, needs_demo,
    pytest.mark.skipif(os.environ.get("VIDEOCONTENT_INSTALL_TEST") != "1",
                       reason="set VIDEOCONTENT_INSTALL_TEST=1 to run the clean-install test"),
    pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed"),
]


def sh(argv: list[str], *, env: dict[str, str], cwd: Path, stdin: str | None = None,
       timeout: int = 900) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(argv, env=env, cwd=cwd, input=stdin, capture_output=True,
                          text=True, timeout=timeout, check=False)
    assert done.returncode == 0, f"{argv}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    return done


def test_clean_install_to_evidence(tmp_path: Path):
    dist, venv, home, work = (tmp_path / n for n in ("dist", "venv", "home", "work"))
    for d in (home / ".claude", work):
        d.mkdir(parents=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "LANG": "C.UTF-8",
           "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR",
                                          str(Path.home() / ".cache" / "uv"))}

    sh(["uv", "build", "--wheel", "--out-dir", str(dist)], env=env, cwd=REPO)
    wheel = next(dist.glob("videocontent-*.whl"))
    sh(["uv", "venv", str(venv), "--python", "3.12"], env=env, cwd=tmp_path)
    python = venv / "bin" / "python"
    sh(["uv", "pip", "install", "--python", str(python), str(wheel)], env=env, cwd=tmp_path)
    exe = str(venv / "bin" / "videocontent")
    env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env['PATH']}"

    # The CLI explains itself.
    assert "analyze" in sh([exe, "--help"], env=env, cwd=work).stdout
    assert sh([str(python), "-m", "videocontent", "--version"], env=env, cwd=work).stdout

    # The skill is installed from the wheel into a clean HOME.
    report = json.loads(sh([exe, "init-agent", "--json"], env=env, cwd=work).stdout)
    assert report["skills"][0]["status"] == "installed"
    skill = home / ".claude" / "skills" / "videocontent"
    assert (skill / "SKILL.md").read_text() == (REPO / "skills/videocontent/SKILL.md").read_text()
    source = json.loads(sh([exe, "init-agent", "--check", "--json"], env=env, cwd=work).stdout)
    assert "site-packages" in source["skill_source"], "the skill must ship inside the wheel"

    # Inspect never processes; analyze processes once; ask answers with evidence.
    shutil.copy(DEMO_VIDEO, work / "demo.mp4")
    inspected = json.loads(sh([exe, "-q", "inspect", "demo.mp4", "--agent"], env=env,
                              cwd=work).stdout)
    assert inspected["result"]["status"] == "not_analyzed"
    assert not (work / "demo.vctx").exists()

    analyzed = json.loads(sh([exe, "-q", "analyze", "demo.mp4", "--agent"], env=env,
                             cwd=work).stdout)
    assert analyzed["result"]["action"] == "processed"
    assert (work / "demo.vctx").is_file()
    # Base install has no speech recognition: that must be reported, not hidden.
    missing = {m["capability"] for m in analyzed["result"]["coverage"]["missing"]}
    assert "speech" in missing

    asked = json.loads(sh([exe, "-q", "ask", "demo.mp4",
                           "What happened after the ConnectionError?", "--agent"],
                          env=env, cwd=work).stdout)
    evidence = asked["result"]["evidence"]
    assert evidence and min(e["start"] for e in evidence) >= 48.0
    assert asked["result"]["answer_kind"] == "extractive"

    # MCP over stdio from the installed package (no mcp SDK installed).
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "install-test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "videocontent_search",
                    "arguments": {"video": "demo.mp4", "query": "ConnectionError"}}},
    ]
    served = sh([exe, "mcp"], env=env, cwd=work,
                stdin="\n".join(json.dumps(m) for m in messages) + "\n", timeout=120)
    replies = [json.loads(line) for line in served.stdout.splitlines()]
    assert len(replies[1]["result"]["tools"]) == 10
    spans = replies[2]["result"]["structuredContent"]["result"]["spans"]
    assert spans and 47.0 <= spans[0]["start"] <= 57.4
