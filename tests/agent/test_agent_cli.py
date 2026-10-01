"""The CLI as an agent meets it: video paths, --agent envelopes, analyze, init-agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from videocontent.agent import install, ops
from videocontent.cli.main import app

from .conftest import FAKE_KEY

runner = CliRunner()


def run(*args: str):
    return runner.invoke(app, list(args))


class TestVideoPathsAreAccepted:
    @pytest.mark.parametrize("args", [
        ("search", "{v}", "ConnectionError", "--json"),
        ("timeline", "{v}", "--json"),
        ("entities", "{v}", "--json"),
        ("changes", "{v}", "--json"),
        ("chapters", "{v}", "--json"),
        ("events", "{v}", "--json"),
        ("at", "{v}", "0:31", "--json"),
        ("plan", "{v}", "what happened after the error", "--json"),
        ("graph", "{v}", "--json"),
        ("entity-timeline", "{v}", "ConnectionError", "--json"),
        ("explain", "{v}", "evt_0001", "--json"),
        ("context", "{v}", "fix the bug", "--json"),
        ("inspect", "{v}", "--json"),
    ])
    def test_video_or_vctx(self, bug_video: Path, args):
        for ref in (str(bug_video), str(bug_video.with_suffix(".vctx"))):
            result = run(*(a.format(v=ref) for a in args))
            assert result.exit_code == 0, result.output
            json.loads(result.output)

    def test_missing_file_is_a_usage_error(self, tmp_path: Path):
        result = run("search", str(tmp_path / "missing.mp4"), "x")
        assert result.exit_code == 2

    def test_unanalyzed_video_names_the_fix(self, tmp_path: Path):
        video = tmp_path / "new.mp4"
        video.write_bytes(b"x")
        result = run("ask", str(video), "what happened?")
        assert result.exit_code == 1
        assert "videocontent analyze" in result.output
        assert not (tmp_path / "new.vctx").exists()

    def test_inspect_unanalyzed_never_processes(self, tmp_path: Path):
        video = tmp_path / "new.mp4"
        video.write_bytes(b"x")
        result = run("inspect", str(video), "--json")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["result"]["status"] == "not_analyzed"
        assert not (tmp_path / "new.vctx").exists()

    def test_compare_videos(self, bug_video: Path, silent_video: Path):
        result = run("compare", str(bug_video), str(silent_video))
        assert result.exit_code == 0, result.output
        assert "bug vs silent" in result.output


class TestAgentFlag:
    @pytest.mark.parametrize("args, operation", [
        (("inspect", "{v}"), "inspect"),
        (("search", "{v}", "ConnectionError"), "search"),
        (("ask", "{v}", "what happened after the ConnectionError"), "ask"),
        (("timeline", "{v}", "--from", "0:25", "--to", "0:45"), "timeline"),
        (("entities", "{v}", "--type", "ERROR"), "entities"),
        (("entity-timeline", "{v}", "ConnectionError"), "entities"),
        (("changes", "{v}"), "changes"),
        (("context", "{v}", "Recreate the login page"), "context"),
        (("explain", "{v}", "evt_0001"), "explain"),
        (("compare", "{v}", "{v}"), "compare"),
        (("analyze", "{v}"), "analyze"),
    ])
    def test_envelope(self, bug_video: Path, args, operation):
        result = run(*(a.format(v=str(bug_video)) for a in args), "--agent")
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["schema"] == ops.AGENT_SCHEMA
        assert out["operation"] == operation
        assert FAKE_KEY not in result.output

    def test_top_k_zero_means_the_envelope_maximum(self, bug_video: Path):
        result = run("search", str(bug_video), "migrate", "--top-k", "0", "--agent")
        assert result.exit_code == 0, result.output


class TestAskJson:
    def test_original_keys_kept_and_agent_keys_added(self, bug_video: Path):
        result = run("ask", str(bug_video), "what happened after the ConnectionError", "--json")
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        for key in ("question", "answer", "confidence", "evidence", "trace"):
            assert key in out, key
        for key in ("query", "answer_kind", "timestamps", "entities", "events",
                    "temporal_relations", "context", "warnings", "video", "content_notice"):
            assert key in out, key
        assert out["answer_kind"] == "extractive"


class TestAnalyze:
    def test_reuse_human_output(self, bug_video: Path):
        result = run("analyze", str(bug_video))
        assert result.exit_code == 0, result.output
        assert "reused" in result.output
        assert "ConnectionError" in result.output
        assert "next" in result.output

    def test_no_process(self, tmp_path: Path):
        video = tmp_path / "fresh.mp4"
        video.write_bytes(b"x")
        result = run("analyze", str(video), "--no-process", "--json")
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["result"]["status"] == "not_analyzed"
        assert not (tmp_path / "fresh.vctx").exists()

    def test_unknown_profile(self, bug_video: Path):
        result = run("analyze", str(bug_video), "--profile", "nonsense")
        assert result.exit_code == 1
        assert "profiles" in result.output


class TestHelp:
    def test_root_help_points_at_the_quick_path(self):
        result = run("--help")
        assert result.exit_code == 0
        for word in ("analyze", "ask", "init-agent", "mcp"):
            assert word in result.output

    def test_no_args_shows_help(self):
        result = run()
        assert "analyze" in result.output


class TestInitAgent:
    @pytest.fixture()
    def home(self, tmp_path: Path, monkeypatch) -> Path:
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("CODEX_HOME", raising=False)
        monkeypatch.setattr(install.shutil, "which", lambda name: None)
        return home

    def test_dry_run_changes_nothing(self, home: Path):
        result = run("init-agent", "--dry-run", "--json")
        assert result.exit_code == 0, result.output
        steps = json.loads(result.output)["skills"]
        assert [s["status"] for s in steps] == ["would_install"]
        assert not (home / ".claude" / "skills").exists()

    def test_install_then_up_to_date(self, home: Path):
        first = json.loads(run("init-agent", "--json").output)
        assert first["skills"][0]["status"] == "installed"
        skill = home / ".claude" / "skills" / "videocontent"
        assert (skill / "SKILL.md").is_file()
        assert (skill / "references" / "workflows.md").is_file()
        assert (skill / install.MARKER).is_file()
        second = json.loads(run("init-agent", "--json").output)
        assert second["skills"][0]["status"] == "up_to_date"

    def test_all_targets(self, home: Path):
        out = json.loads(run("init-agent", "--agent", "all", "--json").output)
        assert {s["target"] for s in out["skills"]} == {"claude", "codex", "agents"}
        for sub in (".claude/skills", ".codex/skills", ".agents/skills"):
            assert (home / sub / "videocontent" / "SKILL.md").is_file()

    def test_foreign_skill_is_never_overwritten(self, home: Path):
        foreign = home / ".claude" / "skills" / "videocontent"
        foreign.mkdir(parents=True)
        (foreign / "SKILL.md").write_text("mine")
        out = json.loads(run("init-agent", "--json").output)
        assert out["skills"][0]["status"] == "skipped"
        assert (foreign / "SKILL.md").read_text() == "mine"
        forced = json.loads(run("init-agent", "--force", "--json").output)
        assert forced["skills"][0]["status"] == "updated"

    def test_our_older_copy_is_upgraded(self, home: Path):
        run("init-agent")
        skill = home / ".claude" / "skills" / "videocontent"
        (skill / "SKILL.md").write_text("old version")
        out = json.loads(run("init-agent", "--json").output)
        assert out["skills"][0]["status"] == "updated"
        assert (skill / "SKILL.md").read_text() != "old version"

    def test_project_scope(self, home: Path, tmp_path: Path, monkeypatch):
        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.chdir(project)
        run("init-agent", "--agent", "claude", "--scope", "project")
        assert (project / ".claude" / "skills" / "videocontent" / "SKILL.md").is_file()
        assert not (home / ".claude" / "skills").exists()

    def test_mcp_is_only_printed_unless_asked(self, home: Path):
        out = json.loads(run("init-agent", "--agent", "claude", "--json").output)
        assert out["mcp"][0]["status"] == "not_requested"
        assert out["mcp"][0]["command"][:3] == ["claude", "mcp", "add"]

    def test_mcp_skipped_when_agent_cli_missing(self, home: Path):
        out = json.loads(run("init-agent", "--agent", "claude", "--mcp", "--json").output)
        assert out["mcp"][0]["status"] == "skipped"

    def test_unknown_agent(self, home: Path):
        result = run("init-agent", "--agent", "cursorx")
        assert result.exit_code == 1

    def test_check(self, home: Path):
        run("init-agent")
        out = json.loads(run("init-agent", "--check", "--json").output)
        assert out["installed"][0]["current"] is True
        assert out["installed"][0]["managed"] is True
