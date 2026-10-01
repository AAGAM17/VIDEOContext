# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

VideoContext (Python package `videocontent`, CLI `videocontent` / `vctx`) turns video into a timestamped, searchable `.vctx` document plus derived views for AI agents. `docs/ARCHITECTURE.md` is the design contract; `docs/VIDEO_CONTEXT_SPEC.md` specifies the `.vctx` format.

## Commands

A `.venv` exists in the repo (uv-managed, `uv.lock`). Python ≥3.10; FFmpeg is required at runtime and Tesseract is needed for OCR.

```bash
pip install -e ".[dev]"            # or: uv sync --extra dev   (extras: agent (=ASR), ocr, api, mcp, vectors, ...)

pytest                              # all tests (testpaths=tests, -q --strict-markers)
pytest tests/unit/test_retrieval.py::test_name   # single test
pytest -m "not integration and not slow"         # skip ffmpeg/model-dependent tests
ruff check src tests                # lint (line length 100, includes flake8-bandit "S" rules)
ruff format src tests
mypy                                # configured to check src/videocontent only

python scripts/make_test_video.py   # regenerates tests/fixtures/demo.mp4 + demo.manifest.json
videocontent doctor                 # check ffmpeg/tesseract/provider availability
videocontent process demo.mp4 && videocontent search demo.vctx "pricing"

videocontent analyze demo.mp4              # reuse demo.vctx or process once, then summarize
videocontent mcp                           # built-in MCP server (stdio, no extra deps)
videocontent init-agent --dry-run          # skill install plan for Claude Code / Codex / .agents
uvicorn apps.api.main:app --port 8000      # REST API (needs [api])
python -m apps.mcp.main                    # legacy MCP server (needs [mcp], i.e. mcp<2)

VIDEOCONTENT_INSTALL_TEST=1 pytest tests/install   # wheel → fresh venv → skill → analyze → ask → MCP
python benchmarks/bench_agent.py                   # agent eval (add --mode claude for real agent runs)
cd apps/web && npm install && npm run dev  # React/Vite demo UI (build: npm run build)
```

There is no CI workflow in the repo. Run pytest, ruff and mypy locally. Ruff and mypy have
pre-existing findings (≈188 ruff, ≈101 mypy with `--python-version 3.12`; the configured 3.10
target fails on numpy's stubs), so compare against `main` rather than expecting zero.

Tests that need `ffmpeg`, `tesseract` or the demo fixture are **skipped**, not failed, when those are missing (`needs_ffmpeg`, `needs_tesseract` and `needs_demo` in `tests/conftest.py`). A green run can therefore hide skipped integration coverage, so check the skip count.

## Architecture

The layers are strict. Lower layers never import from higher ones:

1. **`schema/`**: Pydantic v2 models for the `.vctx` document (`v1.py`), plus `io.py` (read/write, `vctx_version` check, `migrate()`). It depends on nothing else in the project. The `.vctx` document is the central artifact: everything upstream produces it and everything downstream reads it.
2. **`sources/`**: the only place that turns a user reference (a path or URL) into a local file. Includes SSRF protection (scheme allowlist, IP checks on every redirect), size and timeout limits, and cleanup of temporary downloads. The pipeline only sees `VideoAsset.local_path`.
3. **`media/`**: the only place that shells out to ffmpeg/ffprobe. Always uses argv arrays and never `shell=True`. The `# noqa: S603` markers at subprocess call sites are intentional; keep a reason on each one.
4. **`processing/`**: extractors (`sampling`, `scenes`, `ocr`, `asr`, `vision`, `events`), each behind a `Protocol` in `interfaces.py` and registered in `registry.py`. Extractors are pure functions: they never write files, never mutate the document and know nothing about caching. `processing/pipeline.py` handles stage ordering, caching (key = content hash + stage name + stage version + stage config), degradation and metrics. A failed or missing dependency marks a stage `skipped`/`partial`, and the pipeline always emits a document with per-stage status in `document.stages`.
5. **`retrieval/`**: hybrid search that combines lexical BM25, an optional vector index, filters and reciprocal-rank fusion, with a boost when modalities co-occur. It returns `EvidenceSpan`s. Timestamps are copied from the document, never generated.
6. **Derived views** (`temporal.py`, `entities.py`, `graph.py`, `queryplan.py`, `collection.py`, `plans.py`, `packages.py`, `routing/`): pure, deterministic functions over a finished document. They are **never stored** in `.vctx` and never trigger reprocessing. The graph is bounded by `max_depth`, `max_nodes` and `max_seconds`. Context packages record every budget cut in `budget_notes`.
7. **`sdk.py`**: the thin `Video` facade (`process`, `search`, `ask`, `graph`, `query_plan`, `context_package`, ...).
8. **Surfaces** (all peers built on the SDK, with no extraction logic): the CLI (`cli/main.py` is typer, `cli/render.py` handles output, and every command supports `--json`), `apps/api` (FastAPI, job-based), `apps/mcp` (legacy MCP server, `video_id`-based, mcp 1.x) and `apps/web`.
9. **Agent layer** (`agent/`, `skills/videocontent/`): `agent/ops.py` turns a video/.vctx/URL reference into one bounded, labelled, credential-redacted JSON envelope (`videocontent.agent/1`) and never processes except in `analyze`; the CLI's `--agent` flag and `agent/mcp_server.py` (`videocontent mcp`, hand-rolled stdio JSON-RPC) both call it. `skills/videocontent/` is the single provider-neutral skill: the Claude Code plugin (`.claude-plugin/`, repo root is the plugin) and Codex plugin (`.codex-plugin/`) load it, and hatch force-includes it into the wheel as `videocontent/agent/skill` for `init-agent`. Keep skill text free of provider names (a test enforces it) and keep versions in `SKILL.md`, `agent/install.py:SKILL_VERSION` and the manifests equal.

When you add a feature that is exposed to users, it usually needs wiring through the SDK and then the CLI, API and MCP surfaces, plus tests for each.

### Extension and config

- Providers are resolved in this order: explicit object passed in → name in Python/YAML config → `VIDEO_CONTEXT_*` env var (e.g. `VIDEO_CONTEXT_OCR_PROVIDER`) → local offline default. Config resolution lives in `config.py`.
- Register plugins with decorators such as `@register_ocr("name")` or through the `videocontent.plugins` entry-point group.
- **Import optional dependencies inside the adapter, never at module import time.** The base install deliberately excludes torch and cloud SDKs, and the core never imports a provider SDK.
- `profiles/` holds semantic profiles (tutorial, product demo, UI design, application).

## Invariants to preserve

- By default, nothing makes network calls. Cloud providers are opt-in, and the document records which provider each stage used.
- Every stored fact has `start`/`end` in seconds. Answers and packages must trace back to evidence spans.
- Derived output labels inference explicitly (for example, chapter titles have `inferred=True`, weak links have `ambiguous=True`, and chains mean temporal sequence, not causation). Do not present inferred output as observed.
- Keep `.vctx` backwards compatible. New fields must be additive (older documents load with `None`), and version changes go through `schema/io.py:migrate`.
- Logs contain identifiers, timings and counts, never transcript, OCR or frame content.
- Agent-facing output treats video text as untrusted data: keep `content_notice`, `ops.clip` redaction and size caps on anything new that returns extracted text.
- CLI query commands take `VIDEO` (a video, its `.vctx`, or a URL); a missing path is exit 2, an unanalyzed video is exit 1 with an `analyze` hint. Existing `--json` shapes are a contract: add fields, never rename or remove.
