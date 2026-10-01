# Analyze a video

**Prompt**

> Analyze demo.mp4.

**Workflow**

1. `inspect` — is there a `.vctx`? (never processes)
2. not analyzed, and the user asked to analyze → `analyze` (local, once)
3. summarize coverage, key moments, entities; offer next questions

**Operations**

```bash
videocontent inspect demo.mp4 --agent     # result.status = "not_analyzed"
videocontent analyze demo.mp4             # the agent uses --agent; human form shown
```

**Output** (first run: 40 s on a laptop CPU; a second `analyze` reuses `demo.vctx` in ~1 s)

```text
demo.mp4 · 62.4s · processed demo.vctx
coverage: speech ✓ 8 | on-screen text ✓ 36 | visual descriptions — skipped |
objects — not_run | events ✓ 96 | scenes ✓ 9 | frames ✓ 91

key moments
  00:00:48.600  ERROR    E ConnectionError: refused on port 5432
  00:00:48.600  COMMAND  pytest -q tests/

chapters (titles are derived keywords)
  00:00:00.000 → 00:01:02.439  Pricing · Competitor · Revenue

entities
  00:00:48.600  ERROR    E ConnectionError: refused on port 5432 x1
  00:00:48.600  COMMAND  pytest -q tests/ x1
  00:00:06.233  CONCEPT  Revenue Rs x4
  00:00:06.233  CONCEPT  Competitor x5
  00:00:00.033  CONCEPT  Pricing x4 (uncertain)

events: 96 · changes: 8
```

**Agent answer** (written from the output)

> demo.mp4 (62 s) is a quarterly business review followed by a live demo. It covers revenue
> (from 00:00:06), pricing, competitors, a login page (00:00:41) and a terminal where
> `pytest -q tests/` fails with `ConnectionError: refused on port 5432` (00:00:48.6).
> Speech and on-screen text were both captured; visual descriptions were not (vision is
> off by default). Ask me about any moment — e.g. what happened after the error.
