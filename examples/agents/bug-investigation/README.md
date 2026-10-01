# Investigate a bug from a recording

**Prompt**

> Here's a screen recording of the failure (demo.mp4). Find where the app starts failing.

**Workflow**

1. `entities --type ERROR` → the earliest `first_seen` is the candidate
2. `ask "what happened before <error>"` → the lead-up
3. `timeline` around the error → full terminal and narration text
4. report error → preceding steps → visible state; a cause is a labelled hypothesis

**Operations and output**

```bash
videocontent entities demo.mp4 --type ERROR --agent
```
```text
00:00:48.600  E ConnectionError: refused on port 5432
```

```bash
videocontent ask demo.mp4 "what happened before the ConnectionError" --agent
```
```text
00:00:48.600 [detected · events] bash — videocontent
00:00:48.600 [detected · events] $ pytest -q tests/
00:00:48.600 [detected · events] collected 12 items
00:00:48.600 [detected · events] E ConnectionError: refused on port 5432
```

```bash
videocontent timeline demo.mp4 --from 0:40 --to 0:58 --agent
```
```text
00:00:41.340 [observed · transcript] Now the demo, the browser opens the login page on localHistPort3000.
00:00:41.600 [observed · ocr] localhost:3000/login
00:00:41.600 [observed · ocr] Sign in
00:00:48.240 [observed · transcript] The terminal we run Pytest, and one test fails with the connection error.
00:00:48.600 [observed · ocr] $ pytest -q tests/
00:00:48.600 [observed · ocr] collected 12 items
```

**Agent answer** (written from the evidence)

> The failure first appears at **00:00:48.600**: `pytest -q tests/` collects 12 items and one
> fails with `E ConnectionError: refused on port 5432` [observed · on-screen text]. Just
> before, from 00:00:41.6, the browser showed `localhost:3000/login` [observed]. Nothing in
> the recording shows the database starting.
>
> *Interpretation:* port 5432 is PostgreSQL's default, so the test likely ran without a
> database listening — check that Postgres is up in the test environment.

Note the transcript quotes ASR as heard ("localHistPort3000"); the on-screen text
(`localhost:3000/login`) is the reliable source for exact strings.
