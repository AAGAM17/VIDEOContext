# Turn a recording into documentation

**Prompt**

> Turn demo.mp4 into developer documentation.

**Workflow**

1. `analyze` → chapters and key moments (reuses the `.vctx`)
2. `timeline` → what was said and shown, in order
3. `context "<section topic>" --max-frames 2` per section → screenshots with timestamps
4. write sections; quote on-screen text and commands verbatim; one timecode per step

**Operations**

```bash
videocontent analyze demo.mp4 --agent
videocontent timeline demo.mp4 --top-k 40 --agent
videocontent context demo.mp4 "live demo: login page and test run" --max-frames 2 --agent
```

**Output** (transcript spans from `timeline`, and UI states from `context`)

```text
00:00:00.000 [observed · transcript] Welcome to the quarterly business review.
00:00:05.710 [observed · transcript] Here is the agenda, revenue, pricing, competitors, and a live demo.
00:00:11.980 [observed · transcript] Revenue reached 42-lock rupees this quarter, up 18%.
00:00:19.790 [observed · transcript] Our pricing starts at $29 for the starter plan, and $499 at scale.
00:00:29.340 [observed · transcript] Competitor pricing is roughly $2 per hour of video.
00:00:41.340 [observed · transcript] Now the demo, the browser opens the login page on localHistPort3000.
00:00:48.240 [observed · transcript] The terminal we run Pytest, and one test fails with the connection error.
00:00:57.080 [observed · transcript] That concludes the review, thank you.

ui_states:
  00:00:06.233 → 00:00:11.966  Agenda · 1. Revenue update · 2. Pricing changes · 3. Competitor analysis
  00:00:12.233 → 00:00:19.966  Revenue · Revenue Rs 42L · up 18% quarter over quarter
  00:00:20.233 → 00:00:29.333  Pricing · Plan Seats Price / month · Starter 3 $29 · Growth 25 $149
  00:00:41.600 → 00:00:48.333  localhost:3000/login · Sign in · email@example.com · Password
  00:00:48.600 → 00:00:57.333  bash — videocontent · $ pytest -q tests/ · collected 12 items
```

**Agent output** (excerpt, written from the evidence)

```markdown
## Running the test suite (00:00:48)

The demo runs the tests from a terminal:

    $ pytest -q tests/

It collects 12 items; one fails with `E ConnectionError: refused on port 5432`
(screenshot: frame at 00:00:48.6). Start the database before running the suite.
```

The single extractive chapter ("Pricing · Competitor · Revenue") is a keyword title over the
whole 62 s video; for short videos the agent structures sections by UI states instead.
Speech is quoted as recognized ("42-lock rupees"); the on-screen text (`Revenue Rs 42L`) is
used where exact values matter.
