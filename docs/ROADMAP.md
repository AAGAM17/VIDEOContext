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

- 🚧 Temporal reasoning queries: `before`/`after`/`between`/first-/last-occurrence/
  during/around/two-anchor ranges implemented and tested (explicit planner, no NLP
  framework); cross-video temporal joins still planned
- ⬜ Multimodal embeddings (joint text+frame space)
- ⬜ Storage backends: S3, PostgreSQL, Redis; retention + deletion APIs
- ⬜ Multi-tenancy: isolation, auth, quotas, audit logs
- ⬜ Horizontal workers with shard-by-time processing for long videos
- ⬜ Advanced CV plugins: faces, tracking, pose, actions, logos, charts/tables, slides
- ⬜ Stability guarantee on `.vctx` v1.x and the SDK surface

---

## Intelligence layer — shipped (this phase)

- ✅ Temporal core: interval relations, clamped windows, cheap change detection,
  extractive chapters (derived titles), UI states from stable OCR
- ✅ Entities: ERROR/COMMAND from event evidence, CONCEPT terms linked across
  modalities on co-occurrence, explicit uncertainty
- ✅ Retrieval: temporal query planning with inspectable plans, range timelines,
  per-span reasons everywhere, collection index + evidence-based comparison
- ✅ Context packages: expansion → dedup → structural budgets → token trim,
  every cut recorded in `budget_notes`; query-biased frame selection
- ✅ Traceability: `Answer.trace` (plan/executor/retrieval/LLM/outcome),
  processing receipts, vision token accounting in metrics
- ✅ Surfaces: CLI (`timeline/events/entities/changes/chapters/context`,
  `--explain`), MCP (`inspect_video/get_entities/find_changes/get_chapters`,
  capped outputs), API (`/entities/changes/chapters/receipt`; fixed
  `load(doc=...)` call sites), repaired `jobs` package imports
- ✅ Tests with every feature (unit + CLI + API); docs describe behavior only

---

## Agentic intelligence — shipped (this phase)

- ✅ EvidenceGraph derived view (fact/entity/occurrence/change/chapter/state nodes,
  rule-cited edges, bounded traversal, explanations) — no graph database
- ✅ Entity resolution (aliases, opt-in lexical similarity, ambiguity preserved)
  + EntityTimeline (first/last/between/before/after, context, related events)
- ✅ QueryPlan engine (17-intent taxonomy, temporal language with configurable
  gaps, coverage-aware planning that never auto-processes)
- ✅ Graph-aware retrieval (`search_graph` with labeled measured/heuristic scores)
  + AnswerTrace (plan, graph/temporal ops, selected/omitted, budget, coverage)
- ✅ First-class ContextPackage (JSON/text/Markdown, anchor-preserving
  optimization, omitted-information ledger, opt-in redaction)
- ✅ Collection intelligence (entities/events/changes/timeline/occurrences/context,
  cross-video linking, added/removed/changed/uncertain comparison)
- ✅ Event/change chains (`TEMPORAL_SEQUENCE`, observed vs inferred marked),
  UI layout signatures + app hints, receipt trust/coverage inventory
- ✅ Surfaces: CLI (`graph/entity-timeline/plan/explain/compare/collection`),
  MCP (13 new tools, collection registry, caps), API (graph/plan/timeline/
  evidence/explain + `/v1/collections`)
- ✅ `bench_intel.py` (10 cases: lookup/temporal/first/entity/change/sequence/
  graph/collection/comparison/compression)
- ✅ Tests with every feature; docs describe behavior only

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
