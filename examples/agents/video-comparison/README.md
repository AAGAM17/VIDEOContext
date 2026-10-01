# Compare two recordings

**Prompt**

> What changed between demo.mp4 and demo_cut.mp4?

(`demo_cut.mp4` is the first 40 seconds of the demo: `ffmpeg -i demo.mp4 -t 40 -c copy demo_cut.mp4`.)

**Workflow**

1. both analyzed? (`inspect` each; `analyze` with consent)
2. `compare` → entity differences, event and coverage counts
3. verify each claimed difference with `search` on both videos
4. report observed differences first; `uncertain` items are not differences

**Operations**

```bash
videocontent compare demo.mp4 demo_cut.mp4 --agent
videocontent search demo_cut.mp4 "Quarterly Business Review" --agent
videocontent search demo_cut.mp4 "ConnectionError" --agent
```

**`compare` result**

```text
removed_in_b: Competition, E ConnectionError: refused on port 5432, Product,
              Quarterly Business Review, pytest -q tests/
uncertain:    Pricing
events a: command_entered 1, error_shown 1, scene_changed 8, text_appeared 36, ...
events b: scene_changed 5, text_appeared 24, ...   (no command_entered, no error_shown)
```

**Verification.** `search demo_cut.mp4 "Quarterly Business Review"` returns 4 spans,
starting with the title slide at 00:00:00.033 — so that item is not really removed: in the
cut it is seen once (a sighting, below the entity threshold), while the full video shows it
again at the outro. `search demo_cut.mp4 "ConnectionError"` returns nothing — that
difference is real.

**Agent answer** (written from the evidence)

> **Observed differences:** demo_cut.mp4 ends before the live demo's terminal segment. It
> has no `pytest -q tests/` command and no `ConnectionError` (in demo.mp4 both at
> 00:00:48.6), and no command or error events (demo.mp4: one of each).
>
> **Not a difference:** "Quarterly Business Review" is listed as removed, but the title
> slide is in both (demo_cut.mp4 00:00:00.033); the full video repeats it at the end.
>
> **Uncertain:** "Pricing" could not be matched reliably between the two.
