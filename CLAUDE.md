# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the `nemo_retriever/` subdirectory unless noted.

**Install (requires Python 3.12 and `uv`):**
```bash
cd nemo_retriever
uv sync  # installs all extras including dev prerelease Nemotron wheels
```

For a specific extras subset (e.g., service-only, no GPU):
```bash
uv pip install -e ".[service]"
```

**Run tests:**
```bash
cd nemo_retriever
pytest                                        # unit tests only (integration excluded by default)
pytest nemo_retriever/tests/test_foo.py       # single test file
pytest nemo_retriever/tests/test_foo.py::test_bar  # single test
pytest -m integration                        # integration tests (require external services/GPUs)
```

The root `pytest.ini` also covers `api/api_tests`, `client/client_tests`, and `tests/service_tests`.

**Lint / format:**
```bash
pre-commit install          # first-time setup
pre-commit run --all-files  # black (120-char) + flake8 (120-char) + uv-lock check
```

**CLI entry point (after install):**
```bash
retriever --help                # root CLI
retriever service start         # run FastAPI service
retriever ingest ...            # ingest documents
retriever pipeline ...          # pipeline subcommands
retriever harness ...           # CI/K8s test harness
```

**Commits require DCO sign-off:**
```bash
git commit --signoff -m "..."
```

## Architecture

### Package dependency direction

```
api/        ← shared types, schemas, no upstream imports
├── client/ ← HTTP client for service mode (depends only on api/)
└── src/    ← core pipeline logic (depends only on api/)

nemo_retriever/  ← top-level library; may import from all three
```

**Never** import upward or sideways between peers. `client/` must never import from `src/` and vice versa. If `api/` needs something from `src/`, extract the shared abstraction into `api/` instead.

### Source layout

The installable package lives in `nemo_retriever/src/nemo_retriever/`. Key packages:

- **`graph/`** — graph-based execution model. `AbstractOperator`, `Node`, `Graph`, `InprocessExecutor` (pandas, local), `RayDataExecutor` (Ray Data, large-scale). Operators are chained with `>>`. Each operator has a `preprocess → process → postprocess` lifecycle.
- **`ingestor/`** — public `create_ingestor()` factory. Dispatches to `GraphIngestor` (`inprocess`/`batch` run modes) or `ServiceIngestor` (`service` run mode). The fluent API: `.files([...]).extract(...).embed().vdb_upload().ingest()`.
- **`operators/`** — pipeline stage implementations (extract, embed, rerank, vdb, dedup, etc.). GPU operators subclass `GPUOperator`; CPU operators subclass `CPUOperator`.
- **`cli/`** — Typer CLI. Subapps are registered lazily so missing optional deps (torch, tritonclient) don't prevent the service from starting.
- **`service/`** — FastAPI service layer. `app.py` is the application factory. `routers/` contains ingest, admin, dashboard, metrics endpoints. `services/` contains job tracker, pipeline pool, event bus. Supports four deploy modes: `standalone`, `gateway`, `realtime`, `batch`.
- **`common/`** — shared Pydantic param models (`EmbedParams`, `ExtractParams`, `StoreParams`, etc.), schemas, VDB abstractions, and Ray resource heuristics.
- **`harness/`** — automated CI/benchmarking harness that drives Helm-deployed K8s clusters, runs ingestion jobs, and collects recall metrics.
- **`tools/`** — evaluation (`eval` CLI), benchmarking (`benchmark` CLI), recall scoring (`recall` CLI), and skill evaluation (`skill-eval` CLI).
- **`models/`** — model registry, HF cache helpers, default model constants (`VL_EMBED_MODEL`, `VL_RERANK_MODEL`).

### Run modes

| Mode | Class | Executor | Use case |
|------|-------|----------|----------|
| `inprocess` | `GraphIngestor` | `InprocessExecutor` (pandas) | < 100 docs, local dev |
| `batch` | `GraphIngestor` | `RayDataExecutor` | Large-scale, GPU cluster |
| `service` | `ServiceIngestor` | Remote HTTP | Talking to a deployed service |

### Ingestion pipeline stages (service mode)

The pipeline in `config/default_pipeline.yaml` defines stage order:

1. **Source** — broker task intake (Redis)
2. **Extraction** — parallel extractors for PDF, audio, DOCX, PPTX, image, HTML, infographic, table, chart
3. **Mutation** — image filter, image dedup, text splitter
4. **Transform** — image captioning (VLM NIM), text embedding (embedding NIM)
5. **Storage** — image storage (S3), embedding storage, broker response sink
6. **Telemetry** — OpenTelemetry tracing

In `inprocess`/`batch` mode, the same logical stages are expressed as `AbstractOperator` subclasses registered in `graph/stages/stage_registry.py` and wired by `ingestor/graph_ingestor.py`.

### VDB and retrieval

LanceDB is the default vector store (included in core deps). The `Retriever` class (`graph/retriever.py`) runs `embed → RetrieveVdbOperator [→ NemotronRerankActor]`. VDB config is flat `{"uri": ..., "table_name": ...}` for LanceDB or nested `{"vdb_op": "lancedb", "vdb_kwargs": {...}}`. The `tabular` extra adds DuckDB and Neo4j backends.

### NIM microservices

Remote NIM endpoints replace local GPU models via `service/config.py` (`NimEndpointsConfig`): `page_elements_invoke_url`, `ocr_invoke_url`, `table_structure_invoke_url`, `graphic_elements_invoke_url`, `embed_invoke_url`, `rerank_invoke_url`, `caption_invoke_url`. Set `NVIDIA_API_KEY` (or `NGC_API_KEY`) for authenticated endpoints.

### Key decorators (pipeline stages)

- **`@traceable()`** — adds entry/exit timing to `IngestControlMessage` metadata. Defined in `api/internal/primitives/tracing/tagging.py`.
- **Node failure decorator** (`try/except` variant) — wraps stage failures as consistent annotations. Import from `api/util/exception_handlers/decorators.py`. By default, `skip_processing_if_failed=True` so already-failed messages are not re-processed.
- **`@filter_by_task`** — skips the stage if the `IngestControlMessage` doesn't contain a required task name/property.

### Testing conventions

- Integration tests that require external services or GPUs must be marked `@pytest.mark.integration`.
- Test files mirror source paths: `nemo_retriever/tests/test_<module>.py`.
- Mock at the boundary (service/NIM client level), not deep in the call stack; use `spec=True` on mocks.
- Tests that only need the service layer use `fastapi.testclient.TestClient` via the `conftest.py` helper `create_test_job()`.
