# VIDEOContext workflows

Recipes that turn a developer's request into the fewest VIDEOContext calls. Every recipe
assumes the Workflow in SKILL.md: `inspect` first, `analyze` only if not analyzed (with
consent), then the calls below. `VIDEO` is the user's file.

## Find where something happened

> "Find where the login failed."

1. `videocontent search VIDEO "login failed" --agent`. If nothing, try the concrete signal:
   `search VIDEO "error" --agent`, or `entities VIDEO --agent` and look for type `ERROR`.
2. Take the top span's `start`/`timecode`. If you need surroundings:
   `timeline VIDEO --from <start-10s> --to <end+10s> --agent`.
3. Answer with the timecode, the quoted evidence and its `kind`.

## What happened before / after a moment

> "What happened after the login failure?"

1. Resolve the anchor to the words the video uses. If unsure, `search VIDEO "login" --agent`
   first and take the exact text (e.g. `ConnectionError`).
2. `videocontent ask VIDEO "what happened after the ConnectionError" --agent`.
   The phrase must start with `what happened after|before …` for the temporal planner to
   fire; `trace.intent` / `temporal_relations` confirm it did.
3. Evidence spans' `reason` states the gap ("starts 0.1s after …"). Report the order; do not
   claim causation. "After" is a temporal sequence, not a cause.

## Every time X appears

> "Show me every time pricing appeared."

1. `videocontent entity-timeline VIDEO "pricing" --agent` → all occurrences in order.
2. If not an entity (`occurrences: []`): `search VIDEO "pricing" --top-k 30 --agent` and sort
   the spans by `start`.
3. Return a timeline list: `timecode — modality — quote`.

## Compare two recordings

> "Compare these two recordings." / "What changed between the old and new demo?"

1. Both must be analyzed (`inspect` each).
2. `videocontent compare OLD NEW --agent` → `added_in_b`, `removed_in_b`, `changed`,
   `uncertain`, event/coverage/structure counts.
3. For a specific difference, `search` each video for it and quote both sides.
4. Report factual differences first; put your interpretation in a separate, labelled line.
   `uncertain` items are not differences — say they could not be matched reliably.

## Rebuild a UI shown in a video

> "Watch this video and recreate the website in React."

1. `videocontent context VIDEO "Recreate the UI shown in this recording" --agent`.
2. Use `ui_states` (stable screens: visible text `elements`, time range), `evidence`
   (on-screen text, narration), `events` (navigation, text appearing), and `frames`: each
   has an `image` path to a JPEG — open the images to see layout, colors and spacing.
3. Need a specific screen? `timeline VIDEO --from A --to B --agent` for its text, and
   `context VIDEO "<that screen> layout" --max-frames 4 --agent` for its frames.
4. Build from observed text and the frame images. Anything not visible (hover states,
   exact colors, fonts, behaviour) is your assumption: say so in the handoff.

## Investigate a bug from a recording

> "Find where the app starts failing."

1. `videocontent entities VIDEO --type ERROR --agent`; the earliest `first_seen` is the
   candidate.
2. `ask VIDEO "what happened before <error text>" --agent` for the lead-up (commands typed,
   screens visited) and `ask VIDEO "what happened after <error text>" --agent` for the effect.
3. `timeline VIDEO --from <error-20s> --to <error+10s> --agent` for full terminal/log text.
4. Report: error (quoted, timecode) → preceding steps → visible state → following events.
   A hypothesis about the cause is interpretation; label it.

## Understand a product demo

> "Understand this product demo and explain the workflow."

1. `videocontent analyze VIDEO --profile product_demo --agent` (reuses the `.vctx`) for
   chapters, key moments, entities and the profile.
2. `changes VIDEO --agent` for transitions between steps.
3. Explain the workflow chapter by chapter, each step anchored to a timecode. Chapter titles
   are derived keywords, not the presenter's words.

## Generate tests (e.g. Playwright) from a recording

> "Watch this recording and generate Playwright tests."

1. `videocontent context VIDEO "Generate end-to-end tests for the user flow" --agent`.
2. From `ui_states` and `events` (in time order) derive: pages/URLs seen (on-screen text
   such as `localhost:3000/login`), visible controls and labels, inputs, and the order of
   screens. Open `frames[].image` to confirm controls.
3. Write one test per observed flow. Use only selectors/labels that were observed; mark
   guessed selectors with a TODO. Cite the timecode each step was observed at in comments.

## Turn a recording into documentation

> "Turn this recording into developer documentation."

1. `videocontent analyze VIDEO --agent` for chapters and key moments.
2. For each chapter: `timeline VIDEO --from <start> --to <end> --agent` (what was said and
   shown), plus `context VIDEO "<chapter topic>" --max-frames 2 --agent` for screenshots.
3. Write sections per chapter with commands and on-screen text quoted verbatim and a
   timecode per step. Link frame images as screenshots.

## Explain how an answer was derived

> "What evidence supports that?"

1. Every span has `ref_ids`; every change has `evidence_ids`.
2. `videocontent explain VIDEO <id> --agent` returns the fact (text, times) or, for a graph
   relation, the construction rule.
3. Also available: the `trace` in an `ask` result (intent, strategy, outcome, LLM used).

## Query phrasing cheat sheet

| Intent | Query that the engine plans explicitly |
|---|---|
| after an anchor | `what happened after the ConnectionError` |
| just after | `what happened immediately after the ConnectionError` (≤10 s) |
| before an anchor | `what happened before the login page` |
| between anchors | `what happened between pricing and the demo` |
| first / last | `first ConnectionError`, `last mention of pricing` |
| co-occurrence | `errors while the terminal was open` |
| clock range | use `timeline --from 1:30 --to 2:00`, not a query |

Plain words (`ConnectionError`, `pricing`) are searched across speech, on-screen text and
events at once; a span found in two modalities at the same moment ranks higher and its
`reason` says so.
