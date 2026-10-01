# VIDEOContext in OpenAI Codex

## Install

```bash
pipx install "videocontent[agent] @ git+https://github.com/AAGAM17/VIDEOContext"
videocontent init-agent --agent codex          # → $CODEX_HOME/skills/videocontent (default ~/.codex)
videocontent init-agent --agent agents         # → ~/.agents/skills/videocontent (Agent Skills location)
```

Installing to both locations is harmless; use whichever your Codex version reads.
Project-only: add `--scope project` (writes `.codex/skills/` or `.agents/skills/` in the
current directory).

MCP tools (optional — the skill works through the CLI alone):

```bash
codex mcp add videocontent -- videocontent mcp
```

which writes to `~/.codex/config.toml`:

```toml
[mcp_servers.videocontent]
command = "videocontent"
args = ["mcp"]
```

(`videocontent init-agent --agent codex --mcp` runs `codex mcp add` for you.)

A Codex plugin manifest is also provided at [`.codex-plugin/plugin.json`](../../.codex-plugin/plugin.json)
(`skills: ./skills/`), for Codex versions that install plugins from a repository.

## Use

```text
$videocontent analyze demo.mp4
$videocontent ask demo.mp4 "What happened after the error?"
Analyze demo.mp4 and tell me where the login fails.
```

The skill is provider-neutral: the same `SKILL.md` drives Claude Code, Codex and other
Agent Skills–compatible agents.

## What was verified

- `codex mcp add videocontent -- …/videocontent mcp` writes the entry above and
  `codex mcp list` shows it enabled (Codex CLI 0.46, isolated `CODEX_HOME`).
- The MCP server itself was exercised with the official MCP Python client and by Claude Code.

Not verified: a live Codex session. The Codex CLI available during development (0.46) could
not run the account's default model ("requires a newer version of Codex") and predates skill
and plugin support, so skill discovery and `.codex-plugin` installation in Codex are
**untested**. Reports from a current Codex are welcome.
