# VideoContext — Architecture

> The open-source semantic layer between video and AI models.

VideoContext turns opaque video bytes into a **temporal semantic index**: timestamped,
searchable, machine-readable context that an LLM or agent can consume with evidence.

This document describes the system as designed, the boundaries between layers, and the
extension points. It is the contract that implementation follows.

---

## 1. Design invariants

These are not negotiable; every design decision below derives from them.

| # | Invariant | Consequence |
|---|-----------|-------------|
| 1 | **Model agnostic** | The core never imports a provider SDK. Providers are adapters resolved at runtime. |
| 2 | **Local first** | The default configuration runs with zero network calls and zero accounts. Cloud is an accelerator, never a requirement. |
| 3 | **Timestamp precise** | Every extracted fact carries `start`/`end` in seconds (float, media-clock relative). Nothing is stored without a time anchor. |
| 4 | **Evidence first** | Retrieval and Q&A return the spans and modalities that support the answer. An answer without traceable evidence is a bug. |
| 5 | **Incremental** | Stages are independent and resumable. A vision failure must not invalidate transcript, OCR or scenes. |
| 6 | **Extensible** | Every capability sits behind a `Protocol` and is discoverable via a registry. Adding a provider requires no core edit. |
| 7 | **Cost efficient** | Work is hashed and cached. Redundant frames, duplicate OCR and repeated model calls are eliminated before they are paid for. |
| 8 | **Untrusted input** | Video is hostile input: validated, size-limited, never shell-interpolated, always processed via argv arrays. |

---

## 2. System overview

```
                                  ┌──────────────────────────────────────────┐
   video file / URL ─────────────▶│ 1. INGESTION                             │
                                  │    validate · probe · normalize · hash   │
                                  └───────────────────┬──────────────────────┘
                                                      │  MediaInfo
                            ┌─────────────────────────┼─────────────────────────┐
                            ▼                         ▼                         ▼
                  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
                  │ 2a. AUDIO TRACK   │    │ 2b. FRAME PLANE   │    │ 2c. CONTAINER     │
                  │  16k mono wav     │    │  adaptive samples │    │  subs · chapters  │
                  └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
                            ▼                        ▼                        │
                  ┌───────────────────┐    ┌───────────────────────────┐      │
                  │ 3. ASR            │    │ 4. SCENES · OCR · VISION  │      │
                  │  segments+words   │    │  spans · text · captions  │      │
                  └─────────┬─────────┘    └─────────┬─────────────────┘      │
                            └────────────┬───────────┴────────────────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 5. EVENT EXTRACTION  │  typed, temporal
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 6. SEGMENTATION      │  fuse modalities on the timeline
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 7. .vctx DOCUMENT    │  the durable artifact
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 8. INDEX             │  lexical + vector + temporal
                              └──────────┬───────────┘
                                         ▼
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
              search() → spans     ask() → answer        MCP tools
                                    + evidence           (agents)
```

**The `.vctx` document is the architectural centre.** Everything upstream produces it;
everything downstream consumes it. It is a plain, versioned, vendor-neutral file — so an
index can be rebuilt, a different LLM can be swapped in, and processing can happen on a
different machine than querying.

---

## 3. Layers

### Layer 0 — `videocontent.schema` (the format)

Pydantic v2 models for the `.vctx` document plus (de)serialization and migration.
Depends on nothing else in the project. See [VIDEO_CONTEXT_SPEC.md](VIDEO_CONTEXT_SPEC.md).

### Layer 1 — `videocontent.media` (the media boundary)

The **only** place that shells out to FFmpeg/FFprobe.

- `ffmpeg.run()` — argv-array invocation, no `shell=True`, timeouts, captured stderr,
  structured error translation. Never interpolates user strings into a command line.
- `probe.probe()` — `ffprobe -print_format json` → `MediaInfo` (duration, fps, resolution,
  codecs, streams, bitrate, container tags, subtitle tracks).
- `frames.extract()` — decode-once frame extraction at requested timestamps.
- `audio.extract()` — 16 kHz mono PCM WAV, the canonical ASR input.

Rationale: media handling is the security surface and the performance floor. Isolating it
makes both auditable and replaceable (a future `VideoDecoder` backed by PyAV changes one
module).

### Layer 2 — `videocontent.processing` (the extractors)

Each capability is a package with an interface, a registry entry and ≥1 implementation:

| Package | Interface | MVP implementations |
|---------|-----------|---------------------|
| `sampling/` | `FrameSampler` | `fixed`, `scene`, `adaptive` |
| `scenes/` | `SceneDetector` | `ffmpeg` (scene-score), `histogram` |
| `ocr/` | `OCREngine` | `tesseract`, `null` |
| `asr/` | `ASREngine` | `faster-whisper`, `subtitles` (embedded track), `null` |
| `vision/` | `VisionProvider` | `null`; adapters for Gemini / OpenAI-compatible / local VLM |
| `events/` | `EventDetector` | rule-based (screen/slide/text/speaker/scene/object) |

Extractors are **pure functions over media + config → typed results**. They never write
files, never mutate the document, and never know about caching. That makes them trivially
testable and cacheable by the layer above.

### Layer 3 — `videocontent.processing.pipeline` (the orchestrator)

Owns staging, ordering, caching, degradation and telemetry.

```
Stage(name, requires, produces, run) ──▶ StageResult(status, data, metrics, error)
```

- **Independence**: a stage declares `requires`. If a dependency is missing or failed, the
  stage is `skipped`, not fatal. The pipeline always emits a document.
- **Caching**: cache key = `sha256(video content hash + stage name + stage version + resolved
  stage config)`. A changed sampling rate invalidates frames+OCR+vision but not ASR.
- **Degradation**: `PartialResult` is a first-class outcome. `document.stages[]` records
  status, duration, provider identity and error for every stage — so consumers can tell
  "no on-screen text" from "OCR never ran".
- **Telemetry**: every stage reports `StageMetrics` (wall time, units processed, units
  skipped, tokens, estimated cost) which feed `videocontent benchmark`.

### Layer 4 — `videocontent.retrieval` (the query engine)

Hybrid by construction — vector-only retrieval fails on the queries video users actually
ask ("what command was typed", exact product names, error strings).

```
query ─┬─▶ lexical (BM25 over transcript · ocr · vision · events)
       ├─▶ vector  (embeddings, when an embedding provider is configured)
       ├─▶ filters (time window · modality · event type · language · confidence)
       └─▶ fusion  (reciprocal-rank fusion + modality co-occurrence boost)
                    │
                    ▼
              EvidenceSpan[]  { start, end, score, modality, text, reason }
```

Temporal co-occurrence is the differentiator: a moment where the phrase appears in *both*
transcript and OCR ranks above either alone, and the `reason` string says so. Spans are
merged when adjacent, and every returned span is guaranteed to exist in the document —
timestamps cannot be hallucinated because they are copied, never generated.

### Layer 5 — `videocontent.sdk` (the facade)

```python
video = Video("lecture.mp4")   # lazy: nothing runs yet
video.process()                # → VideoContextDocument (.vctx)
video.search("pricing")        # → list[EvidenceSpan]
video.ask("what was typed?")   # → Answer(text, confidence, evidence[])
```

Three methods for the common case; `ProcessingConfig` for everything else. The facade is
thin — it composes the layers below and holds no logic of its own.

### Layer 6 — surfaces

- **CLI** (`videocontent`): `process · inspect · search · ask · export · benchmark · doctor`
- **REST API** (`apps/api`, FastAPI): job-based; long work never blocks a request
- **MCP server** (`apps/mcp`): agent tools over an existing `.vctx` — read-only by default
- **Web demo** (`apps/web`, React+TS+Vite+Tailwind): upload → progress → explorer → search

Surfaces are **peers**, all built on the SDK. None contains extraction logic.

---

## 4. Extension model

A plugin is any object satisfying a `Protocol`, registered under a name:

```python
from videocontent.registry import register_ocr
from videocontent.interfaces import OCREngine

@register_ocr("my-ocr")
class MyOCR:
    name = "my-ocr"
    version = "1.0.0"
    def extract(self, frames: list[Frame], ctx: FrameContext) -> list[OCRResult]: ...
```

Resolution order for any capability:

1. explicit object passed to `process(ocr_engine=...)`
2. name in Python/YAML config
3. `VIDEO_CONTEXT_OCR_PROVIDER` environment variable
4. built-in default (local, offline)

Third-party packages register via the `videocontent.plugins` entry-point group; discovery is
lazy so an unused heavy provider never imports its dependencies. **Optional dependencies are
imported inside the adapter, never at package import time** — this is what keeps
`pip install videocontent` small and `import videocontent` fast.

Interfaces are `typing.Protocol` (structural), not ABCs: a plugin author never has to
inherit from our class hierarchy, and adapters can wrap existing objects.

---

## 5. Data & storage boundaries

| Concern | Interface | Local default | Production |
|---------|-----------|---------------|------------|
| Raw media | `ObjectStore` | filesystem | S3-compatible |
| Derived artifacts (frames, audio, stage cache) | `ArtifactStore` | `.videocontent/` workdir | object storage |
| Semantic document | `.vctx` file | filesystem | object storage + Postgres row |
| Vector index | `VectorStore` | in-process / FAISS | Qdrant (pgvector, Pinecone, Weaviate later) |
| Relational metadata & jobs | `MetadataStore` | SQLite | PostgreSQL + Redis |

Derived artifacts are always reconstructible from `raw media + config`, so they are safe to
evict — which is what makes retention policies and "delete the video, keep the context"
both implementable.

---

## 6. Concurrency & performance

- **Decode once.** Sampling computes the full timestamp plan first, then extracts in a
  single FFmpeg pass. Per-frame `ffmpeg -ss` invocations are the classic 10× mistake.
- **Stage-level parallelism.** ASR (CPU/GPU, audio) and the frame plane (I/O + OCR) are
  independent and run concurrently.
- **Bounded fan-out.** Provider calls run through a semaphore + token-bucket per provider,
  with retry/jitter. We never let a 4-hour video DoS an API key.
- **Batching.** OCR and vision consume frame *batches*; deduplication happens before the
  batch is dispatched, so identical frames are paid for once.
- **Profile before optimizing.** `videocontent benchmark` exists in V0.1 precisely so
  optimization targets are measured, never guessed.

---

## 7. Failure & degradation matrix

| Failure | Behaviour |
|---------|-----------|
| Corrupt / unreadable container | Fail fast at ingestion with actionable error; no partial document |
| No audio stream | ASR `skipped(reason="no_audio_stream")`; visual pipeline proceeds |
| No speech in audio | ASR `ok` with zero segments (distinct from skipped) |
| ASR model unavailable | Fall back: embedded subtitles → `null`; document records the substitution |
| OCR binary missing | Stage `skipped`; `doctor` explains the fix |
| Vision provider error / quota | Stage `partial` with the frames that succeeded |
| Embedding provider missing | Index built lexical-only; search degrades in quality, not availability |
| Zero-length / single-frame video | Handled as a 1-segment document |

---

## 8. Security & privacy posture

- FFmpeg invoked with argv arrays, explicit timeouts, and output paths under a workdir we
  own. No user string ever reaches a shell.
- Paths resolved and confined to the workdir (traversal-safe); temp files cleaned via
  context managers even on exception.
- Configurable limits: max file size, max duration, allowed containers, MIME sniffing that
  trusts probe output over file extension.
- **Egress is explicit.** No stage sends data off-machine unless a cloud provider is
  configured; the document records which stages used which provider, so an audit can answer
  "did this video leave the building?" from the artifact itself.
- Logs carry identifiers, timings and counts — never transcript, OCR or frame content.
- API layer: auth, rate limits, tenant-scoped storage prefixes, retention + deletion APIs.

---

## 9. Repository layout

A single Python distribution (`videocontent`) with internal packages that mirror the layer
boundaries, plus separate apps. Splitting into eight pip packages would multiply release
overhead without buying isolation — the brief's warning against abstraction-for-its-own-sake
applies.

```
src/videocontent/
├── schema/         .vctx models, io, migrations
├── media/          ffmpeg boundary: probe, frames, audio
├── processing/     sampling · scenes · ocr · asr · vision · events · pipeline
├── retrieval/      index · lexical · fusion · query
├── embeddings/     providers
├── storage/        object · artifact · metadata
├── cli/            typer commands
├── interfaces.py   every Protocol
├── registry.py     plugin registry + entry points
├── config.py       env · yaml · python config resolution
└── sdk.py          Video facade

apps/api · apps/mcp · apps/web
docs · tests · examples · benchmarks · scripts · docker
```

Optional dependency extras keep the base install lean:
`[asr]`, `[ocr]`, `[vision]`, `[vectors]`, `[api]`, `[all]`.

---

## 10. MVP boundary (V0.1)

**In:** ingestion · probe · adaptive sampling · scene detection · Whisper ASR · Tesseract OCR
with temporal dedup · rule-based events · segmentation · `.vctx` read/write · lexical hybrid
retrieval · SDK · CLI (`process`/`inspect`/`search`/`doctor`) · tests · benchmark harness.

**Out (deliberately):** vision providers, embeddings/FAISS, `ask()`, REST API, MCP, web UI,
Docker, job queue. Each has a designed seam and a milestone in [ROADMAP.md](ROADMAP.md).

The MVP is done when this works offline, on a real file:

```bash
videocontent process demo.mp4 && videocontent search demo.vctx "pricing"
```
