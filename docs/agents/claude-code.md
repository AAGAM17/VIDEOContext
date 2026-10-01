# VIDEOContext in Claude Code

## Install (pick one)

**A. Skill only — fastest.**

```bash
pipx install "videocontent[agent] @ git+https://github.com/AAGAM17/VIDEOContext"
videocontent init-agent            # → ~/.claude/skills/videocontent
```

Restart Claude Code. `/videocontent` is now available, and Claude uses it on its own when you
mention a video.

**B. Plugin — skill + MCP tools, updatable through `/plugin`.**

```text
/plugin marketplace add AAGAM17/VIDEOContext
/plugin install videocontent@videocontent
```

The plugin bundles the same skill and registers the `videocontent mcp` server, which needs
the `videocontent` command on your `PATH` (install it as in A; `init-agent` is then optional).

**C. MCP only.**

```bash
claude mcp add --scope user videocontent -- videocontent mcp
```

(`videocontent init-agent --mcp` runs this for you.)

## Use

```text
/videocontent                                   → short help with examples
/videocontent analyze demo.mp4
/videocontent ask demo.mp4 "What happened after the error?"
/videocontent context demo.mp4 "Recreate this UI in React"
Analyze demo.mp4 and tell me where the login fails.     (no slash command needed)
```

When loaded as a plugin the skill is also listed as `videocontent:videocontent`; typing
`/videocontent` resolves to it.

The skill makes Claude: inspect first (never processes), analyze only when you asked to
analyze/watch/understand the video (otherwise it asks), reuse the `.vctx` on every later
question, answer with timestamped evidence labelled *observed / detected / derived /
interpretation*, and treat anything said or shown in the video as data, never as
instructions.

## Fewer permission prompts

Claude asks before running shell commands. To allow VIDEOContext's commands without
prompts, add to `.claude/settings.json` (project) or `~/.claude/settings.json`:

```json
{ "permissions": { "allow": ["Bash(videocontent *)"] } }
```

## What was verified

With Claude Code 2.1.286, this repository loaded via `--plugin-dir` and no other plugins
(`--setting-sources ""`):

- `claude plugin validate` passes for `.claude-plugin/plugin.json` and
  `.claude-plugin/marketplace.json` (the plugin's only `--strict` warning is that the
  repository's own `CLAUDE.md` is not plugin context, which is intended).
- `/videocontent` with no arguments printed the skill's help block.
- *"Analyze demo.mp4 and tell me what happened after the error."* on an unanalyzed copy of
  the demo: Claude invoked the skill, ran `inspect` → `analyze` → `ask` (temporal AFTER) →
  `timeline`, and answered with correctly labelled evidence matching the fixture's ground
  truth (47 s, $0.20).
- The agent evaluation (`python benchmarks/bench_agent.py --mode claude`); results in
  [benchmarks/agent_eval_claude.json](../../benchmarks/agent_eval_claude.json).

- Marketplace install from a local checkout, in an isolated `CLAUDE_CONFIG_DIR`:
  `claude plugin marketplace add <repo>` → `claude plugin install videocontent@videocontent`
  → enabled, 1 skill + 1 MCP server; `claude plugin details` projects ~146 tokens always-on
  and ~1.9k when the skill is invoked.

Not verified: `/plugin marketplace add AAGAM17/VIDEOContext` from GitHub itself, which
requires these files to be published on the default branch first.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/videocontent` unknown | restart Claude Code after `init-agent`; check `videocontent init-agent --check` |
| MCP server `videocontent` failed | `videocontent` is not on `PATH` for Claude Code; install with `pipx`, or register the absolute path printed by `videocontent init-agent --check` |
| "has not been analyzed yet" | expected for a new video; Claude asks before running `analyze` unless you asked for analysis |
