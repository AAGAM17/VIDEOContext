"""Agent evaluation: can an agent answer developer questions about a video, correctly and cheaply?

Seven tasks over the demo fixture (ground truth: ``tests/fixtures/demo.manifest.json``) and a
40-second cut of it (for comparison). Two modes:

``engine`` (default, deterministic)
    Runs the reference workflow the skill prescribes for each task through
    :mod:`videocontent.agent.ops` and scores what an agent would receive: evidence inside the
    ground-truth window, every timestamp grounded in a stored fact (no invented timestamps),
    whether anything was (re)processed, output size and latency.

``claude`` (opt-in, uses your Claude Code account)
    Runs ``claude -p`` headless with this repository loaded as a plugin, one task per session,
    and scores the agent itself: did it pick VIDEOContext, which commands, did it force
    reprocessing of an already-analyzed video, are the timecodes in its answer grounded,
    cost and latency.

    python benchmarks/bench_agent.py                 # engine mode
    python benchmarks/bench_agent.py --mode claude   # real agent runs (costs tokens)
    python benchmarks/bench_agent.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "tests" / "fixtures" / "demo.mp4"
MANIFEST = json.loads((REPO / "tests" / "fixtures" / "demo.manifest.json").read_text())


def shot(name: str) -> tuple[float, float]:
    found = next(s for s in MANIFEST["shots"] if s["shot"] == name)
    return float(found["start"]), float(found["end"])


TERMINAL = shot("terminal")      # ConnectionError is on screen here
OUTRO = shot("outro")
BROWSER = shot("browser")        # the login page
REVENUE = (shot("agenda")[0], shot("revenue")[1])


def within(t: float, window: tuple[float, float], slack: float = 1.0) -> bool:
    return window[0] - slack <= t <= window[1] + slack


# ---------------------------------------------------------------------------
# tasks: the prompt, the reference workflow (engine mode) and how to score it
# ---------------------------------------------------------------------------


def _first_error(ops: Any, v: str, _b: str) -> tuple[list[dict], bool, str]:
    out = ops.search(v, "first ConnectionError")
    starts = [s["start"] for s in out["result"]["spans"]]
    ok = bool(starts) and within(min(starts), TERMINAL)
    return [out], ok, f"first at {min(starts) if starts else None}s; truth {TERMINAL}"


def _after_error(ops: Any, v: str, _b: str) -> tuple[list[dict], bool, str]:
    out = ops.ask(v, "what happened after the ConnectionError")
    starts = [e["start"] for e in out["result"]["evidence"]]
    ok = bool(starts) and min(starts) >= TERMINAL[0] - 1 and any(within(s, OUTRO) for s in starts)
    return [out], ok, f"evidence {sorted(starts)}; outro {OUTRO}"


def _compare(ops: Any, v: str, b: str) -> tuple[list[dict], bool, str]:
    out = ops.compare(v, b)
    gone = " ".join(out["result"]["removed_in_b"]).lower()
    ok = "connectionerror" in gone or "pytest" in gone
    return [out], ok, f"removed in cut: {out['result']['removed_in_b'][:5]}"


def _revenue(ops: Any, v: str, _b: str) -> tuple[list[dict], bool, str]:
    out = ops.entities(v, name="Revenue", top_k=50)
    starts = [o["start"] for o in out["result"].get("occurrences", [])]
    ok = bool(starts) and all(within(s, REVENUE) for s in starts)
    return [out], ok, f"{len(starts)} occurrences {sorted(starts)}; truth {REVENUE}"


def _ui(ops: Any, v: str, _b: str) -> tuple[list[dict], bool, str]:
    out = ops.context(v, "Recreate the login page UI shown in this video")
    frames = [f for f in out["result"]["frames"] if "image" in f and within(f["ts"], BROWSER)]
    text = " ".join(e["text"] for e in out["result"]["evidence"]).lower()
    ok = bool(frames) and "login" in text
    return [out], ok, (f"{len(frames)} login-page frame image(s); "
                       f"login in evidence={'login' in text}")


def _timeline(ops: Any, v: str, _b: str) -> tuple[list[dict], bool, str]:
    analyzed = ops.analyze(v)
    out = ops.timeline(v, top_k=50)
    shots = {s["shot"] for s in MANIFEST["shots"] if s["narration"]
             for span in out["result"]["spans"]
             if within(span["start"], (s["start"], s["end"]), 0.0)}
    ok = len(shots) >= 6 and bool(analyzed["result"]["chapters"])
    return [analyzed, out], ok, f"{len(shots)} narrated shots covered"


def _explain(ops: Any, v: str, _b: str) -> tuple[list[dict], bool, str]:
    found = ops.search(v, "ConnectionError")
    ref = found["result"]["spans"][0]["ref_ids"][0]
    out = ops.explain(v, ref)
    fact = out["result"].get("fact") or {}
    ok = bool(out["result"]["found"]) and fact.get("start") is not None \
        and within(float(fact["start"]), TERMINAL)
    return [found, out], ok, f"{ref} → start {fact.get('start')}"


TASKS: list[dict[str, Any]] = [
    {"id": "first-error", "prompt": "Find the first occurrence of ConnectionError in demo.mp4.",
     "expect": {"search", "entity-timeline", "entities", "ask"}, "run": _first_error},
    {"id": "after-error", "prompt": "What happened after the ConnectionError in demo.mp4?",
     "expect": {"ask", "search", "timeline"}, "run": _after_error},
    {"id": "compare", "prompt": "What changed between demo.mp4 and demo_cut.mp4?",
     "expect": {"compare"}, "run": _compare},
    {"id": "revenue", "prompt": "Find every mention of Revenue in demo.mp4.",
     "expect": {"entity-timeline", "search", "entities"}, "run": _revenue},
    {"id": "ui", "prompt": "I want to recreate the login UI shown in demo.mp4. Gather the "
                           "context you would need (don't write code yet) and summarize it.",
     "expect": {"context"}, "run": _ui},
    {"id": "timeline", "prompt": "Generate a timeline of demo.mp4.",
     "expect": {"timeline", "analyze", "chapters"}, "run": _timeline},
    {"id": "explain", "prompt": "In demo.mp4, where does ConnectionError appear, and what "
                                "evidence supports that?",
     "expect": {"search", "explain", "ask", "entity-timeline"}, "run": _explain},
]


# ---------------------------------------------------------------------------
# workspace: demo.mp4 + a 40 s cut, both analyzed once
# ---------------------------------------------------------------------------


def prepare(work: Path) -> tuple[Path, Path, dict[str, float]]:
    from videocontent.agent import ops

    video = work / "demo.mp4"
    shutil.copy(DEMO, video)
    cut = work / "demo_cut.mp4"
    # Fixed argv; the only variable parts are paths this script created.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),  # noqa: S603,S607
                    "-t", "40", "-c", "copy", str(cut)], check=True, timeout=120)
    timings = {}
    for path in (video, cut):
        started = time.perf_counter()
        report = ops.analyze(str(path))
        timings[path.name] = round(time.perf_counter() - started, 2)
        if report["result"]["action"] != "processed":
            raise RuntimeError(f"expected a fresh analysis of {path.name}: {report}")
    return video, cut, timings


def grounded(payloads: list[dict], doc: Any) -> tuple[int, int]:
    """(timestamps checked, timestamps inside the time range of a fact they cite).

    Every evidence item names the facts it came from (``ref_ids`` / ``ref_id``). A timestamp
    is grounded when it lies within one of those facts; an invented timestamp cannot be.
    """
    checked = ok = 0

    def walk(node: Any) -> None:
        nonlocal checked, ok
        if isinstance(node, dict):
            refs = node.get("ref_ids") or ([node["ref_id"]] if node.get("ref_id") else [])
            if refs and "start" in node:
                checked += 1
                start = float(node["start"])
                facts = [doc.by_id(r) for r in refs]
                if any(f is not None and float(f.start) - 0.05 <= start <= float(f.end) + 0.05
                       for f in facts):
                    ok += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for payload in payloads:
        walk(payload)
    return checked, ok


def run_engine(work: Path) -> dict[str, Any]:
    from videocontent import load
    from videocontent.agent import ops

    video, cut, timings = prepare(work)
    vctx = video.with_suffix(".vctx")
    doc = load(vctx).document
    rows = []
    for task in TASKS:
        before = vctx.stat().st_mtime_ns
        started = time.perf_counter()
        payloads, ok, detail = task["run"](ops, str(video), str(cut))
        elapsed = (time.perf_counter() - started) * 1000
        checked, good = grounded(payloads, doc)
        rows.append({
            "task": task["id"], "pass": ok, "detail": detail,
            "latency_ms": round(elapsed, 1),
            "output_bytes": sum(len(json.dumps(p, default=str)) for p in payloads),
            "reprocessed": vctx.stat().st_mtime_ns != before,
            "timestamps_checked": checked, "timestamps_grounded": good,
        })
    return {"mode": "engine", "analyze_seconds": timings, "tasks": rows,
            "passed": sum(r["pass"] for r in rows), "total": len(rows)}


# ---------------------------------------------------------------------------
# claude mode: the real agent, headless
# ---------------------------------------------------------------------------

_TC = re.compile(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\b")


def cited_seconds(text: str) -> list[float]:
    """Timecodes an answer states (``00:00:48.600``, ``0:48``, ``1:02.4``) in seconds."""
    return [int(h or 0) * 3600 + int(m) * 60 + int(s) + float(f"0.{frac or 0}")
            for h, m, s, frac in _TC.findall(text)]


def run_claude(work: Path, budget_usd: float) -> dict[str, Any]:
    from videocontent import load

    if shutil.which("claude") is None:
        sys.exit("claude CLI not found")
    video, cut, timings = prepare(work)
    doc = load(video.with_suffix(".vctx")).document
    cut_doc = load(cut.with_suffix(".vctx")).document
    # Any time a fact starts or ends, in either video: a cited timecode must be one of these
    # (±1 s, since agents round), or it was not taken from the evidence.
    anchors = sorted({round(float(t), 1) for d in (doc, cut_doc)
                      for section in (d.transcript, d.ocr, d.events, d.scenes, d.segments)
                      for x in section for t in (x.start, x.end)}
                     | {0.0, round(float(doc.video.duration), 1)})
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    rows = []
    for task in TASKS:
        started = time.perf_counter()
        done = subprocess.run(  # noqa: S603 - fixed argv; the prompt is our own task text
            ["claude",  # noqa: S607 - the user's installed claude CLI, found on PATH "-p", "--setting-sources", "", "--plugin-dir", str(REPO),
             "--no-session-persistence", "--max-budget-usd", str(budget_usd),
             "--allowedTools", "Bash(videocontent *)", "Read",
             "--output-format", "stream-json", "--verbose", task["prompt"]],
            cwd=work, capture_output=True, text=True, timeout=900, env=env, check=False)
        elapsed = time.perf_counter() - started
        commands: list[str] = []
        skill = False
        answer, cost, turns = "", None, None
        for line in done.stdout.splitlines():
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("type") == "assistant":
                for block in message["message"].get("content", []):
                    if block.get("type") != "tool_use":
                        continue
                    if block["name"] == "Skill" and "videocontent" in json.dumps(block["input"]):
                        skill = True
                    command = str(block["input"].get("command", ""))
                    if block["name"] == "Bash" and "videocontent" in command:
                        commands.append(command)
                    if "videocontent_" in block["name"]:
                        commands.append(block["name"])
            if message.get("type") == "result":
                answer = message.get("result") or ""
                cost, turns = message.get("total_cost_usd"), message.get("num_turns")
        used = {m for c in commands
                for m in re.findall(r"videocontent\s+(?:-q\s+)?([a-z][a-z-]+)", c)}
        used |= {c.split("videocontent_")[-1] for c in commands if "videocontent_" in c}
        forced = sum(1 for c in commands if re.search(r"\banalyze\b.*--force", c))
        cited = cited_seconds(answer)
        ungrounded = [t for t in cited if not any(abs(t - a) <= 1.0 for a in anchors)]
        rows.append({
            "task": task["id"], "skill_invoked": skill, "commands": sorted(used),
            "expected_any_of": sorted(task["expect"]),
            "correct_tool": bool(used & task["expect"]),
            "calls": len(commands), "forced_reprocessing": forced,
            "timestamps_cited": len(cited), "timestamps_ungrounded": ungrounded,
            "cost_usd": cost, "turns": turns, "latency_s": round(elapsed, 1),
            "answer_chars": len(answer), "answer": answer[:4000], "exit": done.returncode,
        })
        print(json.dumps(rows[-1]), file=sys.stderr)
    return {"mode": "claude", "analyze_seconds": timings, "tasks": rows,
            "correct_tool": sum(r["correct_tool"] for r in rows), "total": len(rows)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VIDEOContext agent evaluation")
    parser.add_argument("--mode", choices=["engine", "claude"], default="engine")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget-usd", type=float, default=0.75,
                        help="per-task spending cap in claude mode")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="vctx-agent-eval-") as tmp:
        if args.mode == "engine":
            report = run_engine(Path(tmp))
        else:
            report = run_claude(Path(tmp), args.budget_usd)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
