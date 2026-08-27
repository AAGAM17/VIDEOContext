# Video Context Format (`.vctx`) — Specification v1.0

**Status:** stable draft · **Media type:** `application/vnd.videocontext+json` ·
**Extension:** `.vctx` (JSON, UTF-8) / `.vctx.gz` (gzip)

A `.vctx` document is the complete semantic representation of one video: what was said,
what was shown, what was written on screen, what happened, and when — with every fact
anchored to the media clock.

The format is **vendor-neutral**. Nothing in the schema names a model provider; provider
identity appears only as free-form provenance metadata. A document produced with Whisper +
Tesseract and one produced with a cloud stack are the same shape and are consumed by the
same code.

---

## 1. Goals & non-goals

**Goals**

1. **Self-describing** — readable without the source video or the producing library.
2. **Timestamp-precise** — every fact carries a time anchor; nothing floats free.
3. **Lossless enough to re-index** — a vector index can be rebuilt from the document alone.
4. **Extensible without forks** — new modalities and event types are additive.
5. **Honest about absence** — "not extracted" is distinguishable from "nothing found".

**Non-goals**

- Not a container format: it stores no video or audio bytes (frames are referenced, or
  optionally inlined as data URIs).
- Not a runtime index: embeddings are optional; the search index is a derived artifact.

---

## 2. Conventions

| Rule | Detail |
|------|--------|
| Time | Seconds as `float`, relative to media start (`0.0`). Never frame numbers, never wall clock. |
| Intervals | `[start, end)` — half-open. Instants use `end == start`. |
| Ordering | Every array of timed objects is sorted ascending by `start`, then `end`. |
| IDs | Unique within the document, stable across reprocessing of identical input: `{kind}_{index:04d}` (e.g. `segment_0007`). Treat as opaque. |
| Confidence | `float` in `[0,1]`, or `null` when the producer cannot report one. Never invent a value. |
| Language | BCP-47 tags (`en`, `hi`, `en-IN`) on every textual object that has one. `null` = unknown. |
| Unknown fields | Consumers **must** ignore unrecognized keys. Producers **must not** rely on them. |
| Extensions | Namespaced under `x_` on any object (`x_myorg_foo`). Core keys never begin with `x_`. |

---

## 3. Top-level document

```json
{
  "vctx_version": "1.0",
  "id": "vctx_9f2c1a0b",
  "created_at": "2026-08-26T11:04:22Z",
  "producer": { "name": "videocontent", "version": "0.1.0" },

  "video":      { "...": "§4" },
  "stages":     [ "§5" ],
  "scenes":     [ "§6" ],
  "transcript": [ "§7" ],
  "ocr":        [ "§8" ],
  "vision":     [ "§9" ],
  "objects":    [ "§10" ],
  "events":     [ "§11" ],
  "segments":   [ "§12" ],
  "frames":     [ "§13" ],
  "embeddings": { "...": "§14" },
  "metrics":    { "...": "§15" },
  "x_": {}
}
```

`vctx_version` is the **only** field whose absence is fatal. Every modality array defaults
to `[]`, so a minimal valid document is version + `video` + empty arrays.

### Versioning

`MAJOR.MINOR`. Minor bumps are additive and backwards compatible — a v1.0 reader must
consume a v1.3 document by ignoring unknown fields. A major bump signals a breaking change
and ships with a migration in `videocontent.schema.migrations`. Readers reject a document
whose major version exceeds their own, with a clear error naming both versions.

---

## 4. `video` — source identity

```json
{
  "id": "abc123",
  "filename": "lecture.mp4",
  "path": "/data/lecture.mp4",
  "content_hash": "sha256:1a2b3c…",
  "size_bytes": 734003200,
  "duration": 3720.5,
  "container": "mov,mp4,m4a",
  "fps": 29.97,
  "width": 1920,
  "height": 1080,
  "frame_count": 111503,
  "bitrate": 1578000,
  "video_codec": "h264",
  "audio_codec": "aac",
  "audio_channels": 2,
  "audio_sample_rate": 48000,
  "has_audio": true,
  "subtitle_tracks": [ { "index": 2, "language": "en", "codec": "mov_text" } ],
  "chapters": [ { "start": 0.0, "end": 610.2, "title": "Intro" } ],
  "tags": { "title": "Lecture 4" }
}
```

`content_hash` is the identity that makes caching and reprocessing correct: it is a hash of
the media bytes (streamed, chunked), not the path or mtime. Two copies of the same file
under different names share cache entries.

---

## 5. `stages` — provenance and honesty

The field that makes "absence" interpretable. One entry per pipeline stage attempted.

```json
{
  "name": "ocr",
  "status": "ok",
  "provider": "tesseract",
  "provider_version": "5.5.0",
  "stage_version": "1",
  "config_hash": "7d1f…",
  "started_at": "2026-08-26T11:02:10Z",
  "duration_s": 4.21,
  "cached": false,
  "error": null,
  "warnings": ["3 frames unreadable"],
  "remote": false
}
```

`status ∈ {ok, partial, skipped, failed}` · `remote` records whether the stage sent data
off-machine — the primitive that makes the privacy audit in ARCHITECTURE §8 possible.

Consumer rule: an empty `ocr` array with `status: "ok"` means **the video has no on-screen
text**; with `status: "skipped"` it means **nobody looked**. Never conflate them.

---

## 6. `scenes` — temporal segmentation

```json
{ "id": "scene_0017", "start": 120.2, "end": 183.7,
  "confidence": 0.88, "detector": "ffmpeg-scenescore",
  "keyframe_ts": 122.0, "change_score": 0.61,
  "signals": ["visual", "ocr_change"] }
```

`signals` lists which evidence contributed to the boundary — visual difference alone is
brittle (see ARCHITECTURE §2, Layer 2), so the format records the combination.

---

## 7. `transcript` — speech

```json
{ "id": "utt_0042", "start": 205.2, "end": 209.8,
  "text": "Today we'll discuss backpropagation.",
  "language": "en", "confidence": 0.94,
  "speaker": null, "no_speech_prob": 0.01,
  "words": [ { "text": "Today", "start": 205.2, "end": 205.5, "confidence": 0.99 } ] }
```

`speaker` is reserved for diarization: `null` means not attempted, a string is a stable
label within the document (`"speaker_1"`). `words` is optional (word-level timing enables
"find the exact moment the phrase was said"); when absent, utterance timing is the bound.

---

## 8. `ocr` — on-screen text, temporally deduplicated

**This is not a per-frame dump.** Identical text across adjacent frames collapses into one
*temporal OCR event* with a lifespan — the difference between 40 000 rows and 300 facts.

```json
{ "id": "ocr_0031", "start": 124.2, "end": 127.8,
  "text": "Revenue ₹42L",
  "confidence": 0.96, "language": "en",
  "bbox": [420, 180, 980, 240],
  "bbox_normalized": [0.219, 0.167, 0.510, 0.222],
  "frame_count": 4,
  "first_frame_ts": 124.2, "last_frame_ts": 127.8,
  "stable": true, "engine": "tesseract", "block_index": 2 }
```

- `bbox` is `[x1, y1, x2, y2]` in **source pixels**, origin top-left; `bbox_normalized` is
  the same in `[0,1]` so it survives rescaling. The reported box is the union across the
  lifespan.
- `stable: true` when text persisted across ≥2 sampled frames — a strong signal for slides
  and UI chrome versus transient overlays.
- Merge rule (normative): candidates merge when normalized text matches under the
  producer's similarity threshold, boxes overlap by IoU ≥ threshold, and the time gap is
  within the sampling interval tolerance. Producers record thresholds in the stage
  `config_hash`.

---

## 9. `vision` — model-generated visual understanding

```json
{ "id": "vis_0009", "start": 421.0, "end": 425.0,
  "description": "A terminal window; a pytest run finishes with 3 failures.",
  "entities": ["terminal", "pytest output"],
  "actions": ["command executed"],
  "ui": { "app": "terminal", "state": "test failure" },
  "confidence": 0.8, "provider": "gemini", "model": "…",
  "frame_ids": ["frame_0210"], "language": "en" }
```

Provider and model live here as opaque provenance strings. Nothing in the schema requires
any particular provider — swapping one changes these strings and nothing else.

---

## 10. `objects` — detections and tracks

```json
{ "id": "obj_0004", "label": "car", "start": 88.0, "end": 96.5,
  "confidence": 0.91, "track_id": "track_2",
  "detector": "…", "attributes": { "color": "red" },
  "instances": [ { "ts": 88.0, "bbox": [12, 40, 300, 260], "confidence": 0.9 } ] }
```

A detection without tracking is an entry whose `start == end` and `track_id == null`. This
shape supports "find when the red car appears" without requiring a tracker in the MVP.

---

## 11. `events` — the typed temporal layer

Events are what make a video *queryable* rather than merely *described*.

```json
{ "id": "evt_0112", "type": "screen_changed",
  "start": 421.2, "end": 425.7,
  "description": "Browser navigates to localhost:3000/login",
  "confidence": 0.72, "source": ["ocr", "visual"],
  "detector": "rule:screen-change",
  "attributes": { "from": "editor", "to": "browser" },
  "refs": { "scenes": ["scene_0031"], "ocr": ["ocr_0140"], "frames": ["frame_0210"] } }
```

`refs` is what makes events auditable — every event points at the facts that produced it.

### Core taxonomy (v1.0)

`scene_changed` · `screen_changed` · `slide_changed` · `text_appeared` · `text_disappeared` ·
`text_changed` · `speaker_started` · `speaker_stopped` · `silence_started` · `silence_ended` ·
`person_entered` · `person_left` · `object_appeared` · `object_disappeared` ·
`button_clicked` · `command_entered` · `error_shown`

The taxonomy is **open**: unknown types are valid and must round-trip. Custom types use a
`x_`-prefixed or namespaced string (`myorg.form_submitted`). Consumers filter on `type` as
a string and must not assume the enum is closed.

---

## 12. `segments` — the fused retrieval unit

A segment is a time window with every modality that overlaps it, resolved by reference.
Segments are what retrieval scores and what an LLM is shown.

```json
{ "id": "segment_0007", "start": 240.0, "end": 282.7,
  "scene_ids": ["scene_0019"],
  "transcript_ids": ["utt_0042", "utt_0043"],
  "ocr_ids": ["ocr_0031"],
  "vision_ids": ["vis_0009"],
  "event_ids": ["evt_0112"],
  "object_ids": [],
  "frame_ids": ["frame_0210"],
  "text": "…concatenated searchable text…",
  "summary": null,
  "keywords": ["revenue", "pricing"],
  "languages": ["en"],
  "embeddings": { "text": "emb_0007" } }
```

`text` is the denormalized, searchable projection (transcript + OCR + vision description) —
present so a consumer can build an index or a prompt without resolving references. All
`*_ids` arrays reference IDs elsewhere in the document; a reference to a missing ID is
invalid. Segment boundaries align to scene boundaries where possible, subdivided to respect
a maximum duration and to avoid splitting an utterance.

---

## 13. `frames` — sampled visual anchors

```json
{ "id": "frame_0210", "ts": 421.0, "index": 12630,
  "path": "frames/000210.jpg", "width": 1280, "height": 720,
  "reason": "scene_change", "sharpness": 0.71, "diff_score": 0.42,
  "phash": "9f1c…", "data_uri": null }
```

`reason ∈ {fixed, scene_change, motion, ocr_density, event, keyframe}` — the sampler's
justification, which is what makes an adaptive sampling run auditable and tunable. `path`
is relative to the document's artifact directory; `data_uri` allows a fully self-contained
document at the cost of size. `phash` supports cross-frame dedup.

---

## 14. `embeddings` — optional vectors

```json
{
  "model": "…", "dim": 384, "normalized": true,
  "space": "text", "quantization": "none",
  "vectors": { "emb_0007": [0.013, -0.221] },
  "external": null
}
```

`external` names a vector store instead of inlining (`{"store": "qdrant", "collection": "…"}`)
— because a 2-hour video with per-segment vectors would otherwise triple the document size.
Embeddings are always reconstructible from `segments[].text`, so omitting them is safe.

---

## 15. `metrics` — cost and performance

```json
{ "processing_time_s": 42.3, "video_duration_s": 600.0,
  "realtime_factor": 14.2,
  "frames_sampled": 642, "frames_skipped": 17358,
  "stage_times": { "asr": 8.1, "ocr": 4.2, "vision": 18.4 },
  "tokens": { "input": 12400, "output": 830 },
  "estimated_cost_usd": 0.08, "cache_hits": 3, "peak_memory_mb": 812 }
```

Advisory, not normative — but present in every document so `benchmark` reports real numbers
and cost regressions are visible in review.

---

## 16. Validation rules (normative)

A document is **valid** iff:

1. `vctx_version` present; major version ≤ reader's major version.
2. Every timed object: `0 ≤ start ≤ end ≤ video.duration` (tolerance 0.5 s for container
   duration rounding).
3. Every array of timed objects is sorted ascending by `start`.
4. All IDs unique within the document.
5. Every `*_ids` / `refs` entry resolves to an existing object.
6. `confidence`, when non-null, is in `[0,1]`.
7. Scenes do not overlap. Segments do not overlap. Other modalities may overlap freely.
8. Every stage named in `stages` has a unique `name`.

`videocontent inspect --validate FILE` enforces these and exits non-zero on violation.

---

## 17. Why these choices

| Choice | Rationale |
|--------|-----------|
| JSON, not protobuf/parquet | Inspectable by humans and by every language; the artifact must survive the library. |
| Temporal OCR events, not frame rows | 100× size reduction and a semantically better unit ("text visible 124–128 s"). |
| Separate modality arrays + `segments` referencing them | Facts stay atomic and reusable; the fused view is a derived projection, so resegmenting never loses data. |
| `stages` in the document | Absence becomes interpretable, and the privacy/egress audit is answerable from the file. |
| Open event taxonomy | Domain-specific events (a lecture, a factory line, a UI test) cannot be enumerated in advance. |
| Optional inline embeddings | Keeps documents small and portable while allowing fully self-contained ones. |
| Half-open intervals | Adjacent spans tile the timeline without ambiguity at the boundary. |
