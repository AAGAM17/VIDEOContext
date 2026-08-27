# VideoContext — Roadmap

Ordered by "what unblocks the next thing", not by what is easiest to demo. Every milestone
ends with something a developer can actually run.

Legend: ✅ done · 🚧 in progress · ⬜ planned

---

## V0.1 — The vertical slice (MVP)

**Definition of done:** offline, no account, no GPU:

```bash
videocontent process demo.mp4          # → demo.vctx
videocontent inspect demo.vctx         # transcript · OCR · scenes · events
videocontent search demo.vctx "pricing"
```

| Item | Status |
|------|--------|
| `docs/ARCHITECTURE.md`, `docs/VIDEO_CONTEXT_SPEC.md`, `docs/ROADMAP.md` | ✅ |
| `.vctx` schema (pydantic v2) + reader/writer + validation | 🚧 |
| Media boundary: safe FFmpeg invocation, `ffprobe` → `MediaInfo` | 🚧 |
| Ingestion validation (size, container, MIME-by-probe, corruption) | 🚧 |
| Frame sampling: `fixed`, `scene`, `adaptive` (single-pass extraction) | 🚧 |
| Scene detection (FFmpeg scene score + histogram fallback) | 🚧 |
| ASR: faster-whisper adapter, embedded-subtitle adapter, `null` | 🚧 |
| OCR: Tesseract adapter + **temporal deduplication** | 🚧 |
| Rule-based event extraction (text/slide/screen/speaker/scene/silence) | 🚧 |
| Segmentation (scene-aligned fusion) | 🚧 |
| Lexical hybrid retrieval (BM25 + modality co-occurrence + filters) | 🚧 |
| Stage pipeline: independence, degradation, content-hash caching | 🚧 |
| Python SDK facade (`Video.process/search`) | 🚧 |
| CLI: `process · inspect · search · doctor · benchmark` | 🚧 |
| Unit + integration tests, synthetic test-video generator | 🚧 |

**Explicitly out of V0.1:** vision providers, embeddings, `ask()`, REST API, MCP, web UI,
Docker, job queue. Each has a seam already designed for it.

---

## V0.2 — Answers with evidence

The point at which VideoContext becomes useful to an *application*, not just a developer.

- ⬜ `VisionProvider` adapters: Gemini, OpenAI-compatible, local VLM (Ollama/llama.cpp)
- ⬜ `EmbeddingProvider`: local sentence-transformers + FAISS store; API embeddings optional
- ⬜ True hybrid retrieval: reciprocal-rank fusion of lexical + vector
- ⬜ `video.ask()` — query planning → retrieval → context assembly → LLM → answer
- ⬜ **Evidence guarantees**: every returned timestamp is copied from the document; answers
  citing unsupported spans are rejected before they reach the caller
- ⬜ REST API (FastAPI): upload, process, status, query, timeline, segments, frames
- ⬜ Web demo (React + TS + Vite + Tailwind): upload → live progress → explorer → AI search
      with click-to-seek evidence
- ⬜ Docker: `Dockerfile`, `docker-compose.yml`, `.local`, `.gpu`
- ⬜ Caching across runs + incremental reprocessing of single stages
- ⬜ `videocontent benchmark` reporting per-stage cost and realtime factor

---

## V0.3 — Agents, plugins, scale

- ⬜ MCP server: `search_video · search_transcript · search_ocr · find_event · find_object ·
      get_segment · get_frame · get_timeline · ask_video`
- ⬜ Entry-point plugin discovery + `videocontent plugins list` + a plugin cookbook
- ⬜ Qdrant vector store; `VectorStore` conformance test suite reused by every backend
- ⬜ GPU paths (CUDA/Metal) with automatic detection and graceful CPU fallback
- ⬜ Job system (Redis + worker) so the API never blocks on processing
- ⬜ Batch processing CLI + provider fallback chains and budget caps
- ⬜ Speaker diarization (optional plugin)
- ⬜ Benchmark suite v1: OCR accuracy, WER, Recall@K, timestamp error, event P/R, E2E QA —
      with published, reproducible methodology and no unverified superiority claims

---

## V1.0 — Production

- ⬜ Temporal reasoning queries: "what changed between 10:00 and 20:00", before/after
      comparison, first/last occurrence, co-occurrence across modalities
- ⬜ Multimodal embeddings (joint text+frame space)
- ⬜ Storage backends: S3, PostgreSQL, Redis; retention + deletion APIs
- ⬜ Multi-tenancy: isolation, auth, quotas, audit logs
- ⬜ Horizontal workers with shard-by-time processing for long videos
- ⬜ Advanced CV plugins: faces, tracking, pose, actions, logos, charts/tables, slides
- ⬜ Stability guarantee on `.vctx` v1.x and the SDK surface

---

## Deliberately deferred

| Not doing yet | Why |
|---------------|-----|
| Monetization / hosted tier | Adoption first. The open core stays uncrippled. |
| Custom model training | We connect models; we don't compete with frontier labs. |
| Realtime / live-stream ingestion | Batch semantics must be solid before streaming. |
| Video editing / generation | Out of scope: this is a *read* layer. |
| Frame-perfect video codec work | FFmpeg is the right dependency here. |

---

## Non-negotiables at every milestone

1. `pip install videocontent` then three lines of Python must produce real output — offline.
2. No stage sends data off-machine unless explicitly configured, and the `.vctx` records it.
3. No returned timestamp is ever model-generated; timestamps are copied from evidence.
4. A failing stage degrades the document, never the run.
5. Tests and docs land with the feature, not after it.
