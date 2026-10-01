"""``videocontent init-agent`` — put the VIDEOContext skill where coding agents look for it.

What it does, and nothing else:

1. Copies the provider-neutral skill (``SKILL.md`` + ``references/``) into each detected
   agent's skills directory.
2. With ``--mcp``, registers the ``videocontent mcp`` server through the agent's *own* CLI
   (``claude mcp add`` / ``codex mcp add``). Without it, prints those commands.

Safety rules:

* A directory we did not create is never overwritten (``--force`` overrides). Ours carry a
  ``.videocontent-skill`` marker, so upgrades replace only our own files.
* No agent configuration file is edited directly; MCP registration goes through each
  agent's supported ``mcp add`` command, and only when asked.
* ``--dry-run`` reports the plan without touching the filesystem.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import subprocess  # argv-only calls to the agents' own CLIs, opt-in via --mcp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..errors import VideoContextError

SKILL_NAME = "videocontent"

#: Version of the skill instructions. Bumped when agent-visible behaviour changes; the core
#: package version stays authoritative for VIDEOContext itself.
SKILL_VERSION = "1.0.0"

MARKER = ".videocontent-skill"


@dataclass(frozen=True)
class Target:
    """One agent runtime and where it discovers skills."""

    key: str
    label: str
    user_dir: Path
    project_dir: Path
    cli: str | None
    invoke: str

    def skills_dir(self, scope: str, project: Path) -> Path:
        return self.user_dir if scope == "user" else project / self.project_dir

    def detected(self) -> bool:
        return self.user_dir.parent.is_dir() or (self.cli is not None
                                                 and shutil.which(self.cli) is not None)


def targets() -> dict[str, Target]:
    home = Path.home()
    codex_home = Path(os.environ.get("CODEX_HOME") or home / ".codex")
    return {
        "claude": Target("claude", "Claude Code", home / ".claude" / "skills",
                         Path(".claude/skills"), "claude", "/videocontent"),
        "codex": Target("codex", "OpenAI Codex", codex_home / "skills",
                        Path(".codex/skills"), "codex", "$videocontent"),
        "agents": Target("agents", "Agent Skills (.agents)", home / ".agents" / "skills",
                         Path(".agents/skills"), None, "the videocontent skill"),
    }


def skill_source() -> Path:
    """The skill shipped with this install (wheel), or the repository copy (editable)."""
    packaged = Path(__file__).resolve().parent / "skill"
    if (packaged / "SKILL.md").is_file():
        return packaged
    repo = Path(__file__).resolve().parents[3] / "skills" / SKILL_NAME
    if (repo / "SKILL.md").is_file():
        return repo
    raise VideoContextError(
        "the VIDEOContext skill files are missing from this installation",
        hint="reinstall videocontent, or install from the repository "
             "(pip install 'git+https://github.com/AAGAM17/VIDEOContext')")


def _files(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*")
                  if p.is_file() and p.name != MARKER and "__pycache__" not in p.parts)


def _same(src: Path, dest: Path) -> bool:
    if not dest.is_dir() or _files(src) != _files(dest):
        return False
    return all(filecmp.cmp(src / rel, dest / rel, shallow=False) for rel in _files(src))


def _marker(dest: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((dest / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class Step:
    target: str
    label: str
    path: str
    status: str  # installed | updated | up_to_date | skipped | would_install | would_update
    detail: str = ""
    invoke: str = ""


@dataclass
class Report:
    steps: list[Step] = field(default_factory=list)
    mcp: list[dict[str, Any]] = field(default_factory=list)
    skill_version: str = SKILL_VERSION
    package_version: str = __version__

    @property
    def ok(self) -> bool:
        return not any(s.status == "failed" for s in self.steps) and \
            not any(m.get("status") == "failed" for m in self.mcp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "package_version": self.package_version,
            "skill_version": self.skill_version,
            "skills": [s.__dict__ for s in self.steps],
            "mcp": self.mcp,
        }


def install_skill(dest_root: Path, *, force: bool = False,
                  dry_run: bool = False) -> tuple[str, str]:
    """Copy the skill into ``dest_root/videocontent``. Returns ``(status, detail)``."""
    src = skill_source()
    dest = dest_root / SKILL_NAME
    if dest.exists() and not dest.is_dir():
        return "skipped", f"{dest} exists and is not a directory"
    if dest.is_dir():
        if _same(src, dest):
            return "up_to_date", ""
        ours = _marker(dest) is not None
        if not ours and not force:
            return "skipped", ("a skill named videocontent already exists here and was not "
                               "installed by videocontent; re-run with --force to replace it")
        if dry_run:
            return "would_update", ""
        shutil.rmtree(dest)
        status = "updated"
    else:
        if dry_run:
            return "would_install", ""
        status = "installed"
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", MARKER))
    (dest / MARKER).write_text(json.dumps({"managed_by": "videocontent",
                                           "skill_version": SKILL_VERSION,
                                           "package_version": __version__}) + "\n",
                               encoding="utf-8")
    return status, ""


def mcp_command() -> list[str]:
    """How an agent should launch the MCP server on this machine."""
    exe = shutil.which("videocontent")
    if exe:
        return [exe, "mcp"]
    return [sys.executable, "-m", "videocontent", "mcp"]


def mcp_add_argv(target: str, command: list[str], scope: str) -> list[str] | None:
    if target == "claude":
        return ["claude", "mcp", "add", "--scope", "user" if scope == "user" else "project",
                SKILL_NAME, "--", *command]
    if target == "codex":
        return ["codex", "mcp", "add", SKILL_NAME, "--", *command]
    return None


def _registered(target: str) -> bool | None:
    """Whether the agent already knows a ``videocontent`` MCP server (None: unknown)."""
    argv = ["claude", "mcp", "get", SKILL_NAME] if target == "claude" else \
        ["codex", "mcp", "get", SKILL_NAME] if target == "codex" else None
    if argv is None or shutil.which(argv[0]) is None:
        return None
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30,  # noqa: S603
                              check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.returncode == 0


def run(agents: list[str], *, scope: str = "user", project: Path | None = None,
        force: bool = False, dry_run: bool = False, mcp: bool = False) -> Report:
    """Install the skill for ``agents`` (``["auto"]`` = every detected agent)."""
    if scope not in ("user", "project"):
        raise VideoContextError(f"unknown scope {scope!r}", hint="use user or project")
    known = targets()
    if agents in ([], ["auto"]):
        chosen = [key for key, t in known.items() if t.detected()] or ["agents"]
    else:
        unknown = [a for a in agents if a not in known and a != "all"]
        if unknown:
            raise VideoContextError(f"unknown agent(s): {', '.join(unknown)}",
                                    hint=f"choose from: {', '.join(known)}, all, auto")
        chosen = list(known) if "all" in agents else list(dict.fromkeys(agents))
    project = (project or Path.cwd()).resolve()
    report = Report()
    command = mcp_command()
    for key in chosen:
        target = known[key]
        root = target.skills_dir(scope, project)
        try:
            status, detail = install_skill(root, force=force, dry_run=dry_run)
        except OSError as exc:
            status, detail = "failed", f"{type(exc).__name__}: {exc.strerror or exc}"
        report.steps.append(Step(key, target.label, str(root / SKILL_NAME), status, detail,
                                 target.invoke))
        argv = mcp_add_argv(key, command, scope)
        if argv is None:
            continue
        entry: dict[str, Any] = {"target": key, "command": argv}
        if not mcp:
            entry["status"] = "not_requested"
        elif shutil.which(argv[0]) is None:
            entry["status"] = "skipped"
            entry["detail"] = f"{argv[0]} is not on PATH"
        elif dry_run:
            entry["status"] = "would_register"
        elif _registered(key):
            entry["status"] = "already_registered"
        else:
            try:
                done = subprocess.run(argv, capture_output=True, text=True,  # noqa: S603
                                      timeout=60, check=False, cwd=project)
            except (OSError, subprocess.TimeoutExpired) as exc:
                entry["status"], entry["detail"] = "failed", type(exc).__name__
            else:
                entry["status"] = "registered" if done.returncode == 0 else "failed"
                if done.returncode != 0:
                    entry["detail"] = (done.stderr or done.stdout).strip()[:300]
        report.mcp.append(entry)
    return report


def status(*, project: Path | None = None) -> dict[str, Any]:
    """Where the skill is installed, at which version, and whether it is current."""
    src = skill_source()
    project = (project or Path.cwd()).resolve()
    rows = []
    for key, target in targets().items():
        for scope in ("user", "project"):
            dest = target.skills_dir(scope, project) / SKILL_NAME
            if not (dest / "SKILL.md").is_file():
                continue
            marker = _marker(dest) or {}
            rows.append({"target": key, "scope": scope, "path": str(dest),
                         "skill_version": marker.get("skill_version"),
                         "managed": bool(marker), "current": _same(src, dest)})
    return {"package_version": __version__, "skill_version": SKILL_VERSION,
            "skill_source": str(src), "installed": rows,
            "mcp_command": mcp_command()}


__all__ = ["MARKER", "SKILL_NAME", "SKILL_VERSION", "Report", "install_skill", "mcp_command",
           "run", "skill_source", "status", "targets"]
