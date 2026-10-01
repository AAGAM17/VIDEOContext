# Ask a temporal question

**Prompt**

> Analyze demo.mp4 and tell me what happened after the error.

**Workflow**

1. `inspect` → `analyze` if needed (the prompt asks for analysis, so no confirmation)
2. name the anchor the way the video does (`ConnectionError`)
3. `ask` with explicit temporal phrasing: *what happened after …*
4. if evidence is thin, widen with `timeline` over the range after the anchor
5. answer with labelled evidence; order, not causation

**Operations**

```bash
videocontent inspect demo.mp4 --agent
videocontent analyze demo.mp4 --agent
videocontent ask demo.mp4 "What happened after the ConnectionError?" --agent
videocontent timeline demo.mp4 --from 0:48 --to 1:03 --agent
```

**`ask` result** (fields abridged)

```text
answer_kind: extractive        (no LLM configured — the evidence is the answer)
trace.intent: temporal_after
temporal_relations: after top 'the ConnectionError' match at 00:00:57.466
evidence:
  00:00:57.600 [detected · events] Quarterly Business Review
  00:00:57.600 [observed · ocr]    Product, Pricing and Competition
  00:00:59.480 [detected · events] 2.96s without speech
```

**Agent answer** — actual Claude Code output (headless run, this repository as a plugin):

```markdown
### Answer
The error appears at 00:00:48.600 in a terminal running `pytest -q tests/`. One test fails
with `ConnectionError: refused on port 5432`, and the run ends "1 failed, 11 passed". At
about 00:00:57 the terminal closes, the presenter says "That concludes the review, thank
you.", and the video goes back to the "Quarterly Business Review" title slide until it ends
at 01:02. Nothing in the recording shows anyone fixing or looking into the error.

### Evidence
00:00:48.240 — [observed · speech] "The terminal we run Pytest, and one test fails with the connection error."
00:00:48.600 — [observed · on-screen text] "$ pytest -q tests/" · "E ConnectionError: refused on port 5432"
00:00:57.080 — [observed · speech] "That concludes the review, thank you."
00:00:57.375 — [detected · event] scene boundary
00:00:57.600 — [observed · on-screen text] "Quarterly Business Review" (shown until 01:02.439)

### Source
demo.mp4 (demo.vctx, created by this run on your machine)

### Trace
inspect → not_analyzed → analyze (19s) → ask "what happened after the ConnectionError" → 5 results → timeline 0:48–1:03
```

Ground truth (`tests/fixtures/demo.manifest.json`): terminal 48.4–57.4 s, outro 57.4–62.4 s.
