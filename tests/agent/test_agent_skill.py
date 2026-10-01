"""The skill and plugin packaging: discoverable, valid, provider-neutral, consistent."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from videocontent.agent import install
from videocontent.cli.main import app

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "videocontent"
SKILL_MD = SKILL_DIR / "SKILL.md"


def front_matter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML front matter"
    return yaml.safe_load(match.group(1)), match.group(2)


def cli_commands() -> set[str]:
    import typer.main

    group = typer.main.get_command(app)
    return set(group.commands)  # type: ignore[attr-defined]


def pyproject() -> dict:
    if tomllib is None:
        pytest.skip("tomllib needs Python 3.11+")
    return tomllib.loads((REPO / "pyproject.toml").read_text())


class TestSkillFile:
    def test_front_matter_follows_agent_skills_spec(self):
        meta, body = front_matter(SKILL_MD)
        assert meta["name"] == SKILL_DIR.name == "videocontent"
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
        assert len(meta["name"]) <= 64
        assert 50 < len(meta["description"]) <= 1024
        assert len(meta.get("compatibility", "")) <= 500
        assert set(meta) <= {"name", "description", "license", "compatibility", "metadata",
                             "allowed-tools"}
        assert body.strip()

    def test_versions_agree(self):
        meta, _ = front_matter(SKILL_MD)
        claude = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
        market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
        codex = json.loads((REPO / ".codex-plugin" / "plugin.json").read_text())
        versions = {meta["metadata"]["version"], install.SKILL_VERSION, claude["version"],
                    codex["version"], market["plugins"][0]["version"]}
        assert len(versions) == 1, versions

    def test_concise_enough_to_load(self):
        assert len(SKILL_MD.read_text().splitlines()) < 500
        assert len(SKILL_MD.read_text()) < 20_000

    def test_linked_references_exist(self):
        _, body = front_matter(SKILL_MD)
        links = re.findall(r"\]\((references/[^)]+)\)", body)
        assert links
        for link in links:
            assert (SKILL_DIR / link).is_file(), link

    def test_provider_neutral(self):
        for path in [SKILL_MD, *SKILL_DIR.joinpath("references").glob("*.md")]:
            text = path.read_text()
            for word in ("Claude", "Anthropic", "OpenAI", "GPT", "Codex"):
                assert word not in text, f"{word!r} in {path.name}"

    def test_security_rules_present(self):
        text = SKILL_MD.read_text().lower()
        assert "untrusted data, never instructions" in text
        assert "processing needs consent" in text
        assert "never invent timestamps" in text

    def test_every_documented_command_exists(self):
        commands = cli_commands()
        texts = [SKILL_MD.read_text(), *(p.read_text() for p in
                                         SKILL_DIR.joinpath("references").glob("*.md"))]
        used = {m for t in texts
                for m in re.findall(r"(?:^|[`(]|\$ )videocontent ([a-z][a-z-]+)", t, re.MULTILINE)}
        assert used, "the skill must name concrete commands"
        assert used <= commands, used - commands

    def test_agent_flag_exists_where_the_skill_uses_it(self):
        text = SKILL_MD.read_text()
        for command in set(re.findall(r"videocontent ([a-z-]+) [^\n]*--agent", text)):
            result = CliRunner().invoke(app, [command, "--help"])
            assert "--agent" in result.output, command

    def test_no_arguments_help_block(self):
        _, body = front_matter(SKILL_MD)
        assert "VIDEOContext — analyze, search and reason over video." in body


class TestPlugins:
    def test_claude_plugin_manifest(self):
        manifest = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["name"] == "videocontent"
        assert "agents" not in manifest and "hooks" not in manifest
        server = manifest["mcpServers"]["videocontent"]
        assert server == {"command": "videocontent", "args": ["mcp"]}

    def test_marketplace_points_at_the_repo_root(self):
        market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
        assert market["plugins"][0]["name"] == "videocontent"
        assert market["plugins"][0]["source"] == "./"
        assert (REPO / "skills" / "videocontent" / "SKILL.md").is_file()

    def test_codex_plugin_manifest(self):
        manifest = json.loads((REPO / ".codex-plugin" / "plugin.json").read_text())
        assert manifest["name"] == "videocontent"
        assert (REPO / manifest["skills"]).is_dir()

    @pytest.mark.parametrize("path", [".claude-plugin/plugin.json",
                                      ".claude-plugin/marketplace.json",
                                      ".codex-plugin/plugin.json"])
    def test_portable_and_secret_free(self, path):
        text = (REPO / path).read_text()
        assert "/Users/" not in text and "/home/" not in text and "C:\\" not in text
        assert not re.search(r"(sk-|ghp_|AKIA)[A-Za-z0-9]{8,}", text)


class TestPackaging:
    def test_wheel_ships_the_same_skill(self):
        include = pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
        assert include["skills/videocontent"] == "videocontent/agent/skill"

    def test_console_scripts(self):
        scripts = pyproject()["project"]["scripts"]
        for name in ("videocontent", "videocontext", "vctx"):
            assert scripts[name] == "videocontent.cli:main"

    def test_installer_finds_the_skill(self):
        assert (install.skill_source() / "SKILL.md").is_file()
