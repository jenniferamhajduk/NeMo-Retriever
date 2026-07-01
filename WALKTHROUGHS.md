# NeMo Retriever Library — Hands-On Walkthroughs (Single H100 GPU)

This guide is a set of step-by-step, runnable walkthroughs for learning every major
feature of the **NeMo Retriever Library** (NRL), assuming you have **one NVIDIA H100
GPU (80 GB)**. It complements, but does not replace, the official docs:

- Repo quickstart: [`nemo_retriever/README.md`](nemo_retriever/README.md)
- Official docs site: https://docs.nvidia.com/nemo/retriever/extraction/
- Python API reference: [`docs/docs/extraction/nemo-retriever-api-reference.md`](docs/docs/extraction/nemo-retriever-api-reference.md)
- CLI reference: [`nemo_retriever/docs/cli/README.md`](nemo_retriever/docs/cli/README.md)

> **Why an H100 is enough for everything here.** Per the
> [Pre-Requisites & Support Matrix](docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements),
> the core extraction pipeline (page-elements detection, table structure, OCR, VL
> embedding) needs only ~4.8 GiB combined and the reranker fits *concurrently* with
> the core pipeline on any GPU with ≥80 GB VRAM — which an H100 satisfies. Features
> that need a **second, dedicated** GPU on smaller cards (audio/video ASR,
> Nemotron Parse, Omni image captioning) still run fine on a single H100 — you just
> can't run two of those *at the same time* as the core pipeline; run them
> sequentially, or offload them to hosted build.nvidia.com NIMs (Walkthrough 9).

Each walkthrough is self-contained: you can do them in order, or jump to the one
you need.

## Table of contents

1. [Environment setup](#1-environment-setup)
2. [Quickstart: ingest a PDF and inspect extraction](#2-quickstart-ingest-a-pdf-and-inspect-extraction)
3. [The CLI: `retriever ingest` and `retriever query`](#3-the-cli-retriever-ingest-and-retriever-query)
4. [Vector storage and retrieval with LanceDB](#4-vector-storage-and-retrieval-with-lancedb)
5. [Embedding options: text, multimodal (VL), page-image](#5-embedding-options-text-multimodal-vl-page-image)
6. [Reranking](#6-reranking)
7. [Live RAG: retrieve, answer, score, and judge](#7-live-rag-retrieve-answer-score-and-judge)
8. [Ingesting other content: Office docs, images, SVG](#8-ingesting-other-content-office-docs-images-svg)
9. [Audio and video transcription (Parakeet ASR)](#9-audio-and-video-transcription-parakeet-asr)
10. [Alternate PDF extraction: Nemotron Parse](#10-alternate-pdf-extraction-nemotron-parse)
11. [Storing extracted images to disk/S3](#11-storing-extracted-images-to-diskS3)
12. [Metadata filtering](#12-metadata-filtering)
13. [Remote inference: build.nvidia.com NIMs, no local GPU](#13-remote-inference-buildnvidiacom-nims-no-local-gpu)
14. [Scaling out: Ray and batch mode](#14-scaling-out-ray-and-batch-mode)
15. [Agentic retrieval](#15-agentic-retrieval)
16. [Running NeMo Retriever as a service (+ MCP for agents)](#16-running-nemo-retriever-as-a-service--mcp-for-agents)
17. [QA evaluation pipeline (retrieval + answer quality)](#17-qa-evaluation-pipeline-retrieval--answer-quality)
18. [Benchmark harness](#18-benchmark-harness)
19. [Framework integrations: LlamaIndex / LangChain](#19-framework-integrations-llamaindex--langchain)
20. [Troubleshooting cheat sheet](#20-troubleshooting-cheat-sheet)

---

## 1. Environment setup

**Prerequisites** (full list: [prerequisites-support-matrix.md](docs/docs/extraction/prerequisites-support-matrix.md)):

- Linux (Ubuntu 22.04+), NVIDIA driver ≥ 580, CUDA ≥ 13.0 (`libcudart.so.13` must be loadable)
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

Create an isolated environment and install the library with local-GPU support:

```bash
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install "nemo-retriever[local]"
```

PyTorch from PyPI defaults to CPU wheels — override with the CUDA 13.0 build so
local Hugging Face inference can use your H100:

```bash
uv pip uninstall torch torchvision
uv pip install torch==2.10.0 torchvision -i https://download.pytorch.org/whl/cu130
```

Sanity-check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA H100 ...
```

If OCR fails with a `libcudart.so.13` error:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

`uv pip install "nemo-retriever[local]"` pulls vLLM, which JIT-compiles CUDA
kernels and needs Python headers. If you didn't install Python via
`uv python install 3.12`, install headers manually to avoid
`InductorError: fatal error: Python.h: No such file or directory`:

```bash
sudo apt install -y python3.12-dev
```

**Optional extras**, install only what you need:

```bash
uv pip install "nemo-retriever[nemotron-parse]"   # alternate PDF parser (Walkthrough 10)
uv pip install "nemo-retriever[multimedia]"       # audio/video + SVG (Walkthroughs 8 & 9)
uv pip install "nemo-retriever[llm]"              # LLM/judge clients for live RAG (Walkthrough 7)
```

System packages for some file types:

```bash
sudo apt install -y libreoffice   # required to render .pptx/.docx pages as images
sudo apt install -y ffmpeg        # required for audio/video extraction
```

You're ready. All walkthroughs below assume you're in the repo root with this
venv active, and use the bundled sample files in [`data/`](data/) (e.g.
`data/multimodal_test.pdf`, which has text, a table, and a chart).

---

## 2. Quickstart: ingest a PDF and inspect extraction

The core abstraction is the **`Ingestor`**: a lazily-built, chainable pipeline of
tasks (`.files()` → `.extract()` → `.embed()` → `.vdb_upload()`), executed only
when you call `.ingest()`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.io import to_markdown, to_markdown_by_page
from pathlib import Path

documents = [str(Path("data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")

ingestor = (
    ingestor.files(documents)
    .extract(
        # defaults shown explicitly — each content type can be toggled independently
        extract_text=True,
        extract_charts=True,
        extract_tables=True,
        extract_infographics=True,
    )
    .embed()
    .vdb_upload()
)

chunks = ingestor.ingest()  # pandas.DataFrame
```

Inspect what was extracted:

```python
print(chunks.iloc[0]["text"])   # raw page text
print(chunks.iloc[1]["text"])   # the table, rendered as markdown
print(chunks.iloc[2]["text"])   # the chart, OCR'd into text
print(to_markdown(chunks)[:200])              # whole-document markdown
print(to_markdown_by_page(chunks).keys())     # dict[int, str] keyed by page
```

The first run downloads the Hugging Face checkpoints listed in the
[hardware table](docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements)
(page-elements, table-structure, OCR, VL embedder — about 5 GiB total) to your
HF cache. Subsequent runs reuse the cache.

**What just happened:** the PDF was split into pages, page elements were
classified (text/table/chart/infographic), OCR ran over scanned/visual regions,
the results were embedded, and embeddings were written to a local LanceDB table
at `./lancedb/nemo-retriever`.

---

## 3. The CLI: `retriever ingest` and `retriever query`

The same pipeline is available without writing Python, via the `retriever` CLI
(full reference: [`nemo_retriever/docs/cli/README.md`](nemo_retriever/docs/cli/README.md)).

```bash
retriever ingest ./data/multimodal_test.pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --use-table-structure \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

```bash
retriever query "What is in this document?" \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

By default, ingest writes to `lancedb/nemo-retriever` and query reads from the
same table. Try a directory of PDFs instead of a single file — `DOCUMENTS...`
accepts files, directories, or shell globs.

Tune what comes back:

```bash
retriever query "annual revenue by region" \
  --top-k 5 \
  --candidate-k 40 \
  --content-types table
```

- `--candidate-k` controls how many raw hits LanceDB returns *before*
  page-dedup/content-type filtering; `--top-k` is the final count. Set
  `--candidate-k` > `--top-k` if filtering might otherwise starve your results.
- `--content-types` filters by `text`, `table`, `chart`, `image`, `infographic`.

Use `--dry-run` on any ingest invocation to print the resolved plan without
actually running anything — useful while you're learning the flags:

```bash
retriever ingest ./data/multimodal_test.pdf --dry-run
```

---

## 4. Vector storage and retrieval with LanceDB

NRL ships **LanceDB** (embedded, in-process, no extra service to run) as its
first-party vector store ([details](docs/docs/extraction/vdbs.md)). It uses an
`IVF_HNSW_SQ` index over a Lance/Arrow columnar layout.

> **Corpus-size gotcha:** LanceDB's default IVF index needs at least 16 chunks
> to train its 16 k-means partitions. Ingesting a single small PDF can fail at
> the indexing step — point ingestion at a directory with enough documents/pages
> to clear that threshold.

Ingest a directory and land it in a named LanceDB table:

```bash
python -m nemo_retriever.examples.graph_pipeline \
  /path/to/your/pdf-directory \
  --vdb-kwargs-json '{"uri":"lancedb","table_name":"nemo-retriever"}'
```

Query it back from Python with the `Retriever` class:

```python
from nemo_retriever.graph.retriever import Retriever

retriever = Retriever(
    vdb_kwargs={"uri": "lancedb", "table_name": "nemo-retriever"},
    top_k=5,
    rerank=False,
)
hits = retriever.query("Given their activities, which animal is responsible for the typos?")
for h in hits:
    print(h["_distance"], h["text"][:80])
```

You can also drive LanceDB directly when you want precomputed query vectors
instead of letting `Retriever` embed the query string for you:

```python
from nemo_retriever.common.vdb.lancedb import LanceDB

vdb = LanceDB(uri="./lancedb_data", table_name="nemo-retriever", index_type="IVF_HNSW_SQ")
docs = vdb.retrieval(queries, top_k=10)
```

**Important:** when constructing an `Ingestor.vdb_upload()` call directly (not
via `graph_pipeline`), pass `vdb_op="lancedb"` explicitly — omitting it falls
back to a `"milvus"` default string for backward compatibility, which is *not*
LanceDB.

Want a different vector store (Elasticsearch, Pinecone, Teradata, or your own)?
NRL defines a `VDB` abstract interface you can subclass — see
[`examples/building_vdb_operator.ipynb`](examples/building_vdb_operator.ipynb) and
[Vector database partners](docs/docs/extraction/vdbs.md#vector-database-partners).

---

## 5. Embedding options: text, multimodal (VL), page-image

NRL's default embedder treats all extracted content (text, OCR'd tables/charts)
as plain text. You can instead use the
[**Llama Nemotron Embed VL 1B v2**](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)
multimodal model to embed visual regions as *images* rather than OCR'd text —
more accurate for dense tables/charts, at a performance cost.

**Default text embedding (baseline):**

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed()  # embeds everything as text
)
results = ingestor.ingest()
```

**Text + image for structured elements** (tables/charts embedded as images,
everything else as text):

```python
ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(embed_modality="text_image")
)
results = ingestor.ingest()
```

**Whole-page-as-image embedding** (best for infographics, dense diagrams, forms):

```python
ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(embed_modality="image", embed_granularity="page")
)
results = ingestor.ingest()
```

More background: [docs/docs/extraction/embedding.md](docs/docs/extraction/embedding.md).
There is also a text-only NIM
([NeMo Retriever Text Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html))
you can route to instead of/alongside the multimodal flows by passing
`embed_invoke_url`/`model_name` to `.embed()`.

---

## 6. Reranking

A two-stage retrieve-then-rerank pipeline trades a bit of latency for
meaningfully better top-k precision. NRL ships
[`nvidia/llama-nemotron-rerank-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2)
(text) and a VL variant for image-aware reranking.

On an H100 (≥80 GB), the reranker fits **concurrently** with the core
extraction/embedding pipeline — no need to tear anything down (smaller GPUs
like A10G/L40S must run extraction and reranking separately; H100 doesn't).

Enable reranking at query time via the `Retriever`:

```python
from nemo_retriever.graph.retriever import Retriever

retriever = Retriever(
    vdb_kwargs={"uri": "lancedb", "table_name": "nemo-retriever"},
    top_k=5,
    rerank=True,  # loads the local reranker model on first use
)
hits = retriever.query("What is RAG?")
```

Or via the CLI against a hosted rerank endpoint (Cohere-style `/v1/rerank`):

```bash
export NGC_INFERENCE_API_KEY=...

retriever query "What is in this document?" \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 \
  --reranker-invoke-url https://inference-api.nvidia.com/v1/rerank \
  --reranker-model-name nvidia/llama-3.2-nv-rerankqa-1b-v2 \
  --reranker-api-key-env NGC_INFERENCE_API_KEY
```

> **Troubleshooting:** if the vLLM-based VL reranker fails during CUDA graph
> capture with `Python.h: No such file or directory`, install
> `python3.12-dev` (same root cause as the vLLM note in
> [Section 1](#1-environment-setup)) and restart.

---

## 7. Live RAG: retrieve, answer, score, and judge

The retrieve → build-prompt → call-LLM pattern is built into the SDK as
`Retriever.answer()`, so you can skip the boilerplate. Install the LLM extra:

```bash
uv pip install "nemo-retriever[llm]"
export NVIDIA_API_KEY=nvapi-...
```

**Single-query live RAG:**

```python
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.llm import LiteLLMClient

retriever = Retriever(
    vdb_kwargs={"uri": "lancedb", "table_name": "nemo-retriever"},
    top_k=5,
)
llm = LiteLLMClient.from_kwargs(
    model="nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
    temperature=0.0,
    max_tokens=512,
)

result = retriever.answer("What is RAG?", llm=llm)
print(result.answer)
print(f"{result.latency_s:.2f}s on {result.model}, {len(result.chunks)} chunks")
```

**Add scoring + an LLM judge** (requires ground-truth `reference`):

```python
from nemo_retriever.llm import LLMJudge

judge = LLMJudge.from_kwargs(
    model="nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
    temperature=0.1,
    max_tokens=4096,
)
result = retriever.answer(
    "What is RAG?",
    llm=llm,
    judge=judge,
    reference="RAG combines retrieved context with LLM generation.",
)
print(result.token_f1, result.judge_score, result.failure_mode)
```

**Batch RAG over many queries** (operator-graph style, each step optional):

```python
df = (
    retriever.pipeline()
    .generate(llm)
    .score()
    .judge(judge)
    .run(
        queries=["What is RAG?", "What is reranking?"],
        reference=[
            "RAG combines retrieval with generation.",
            "Reranking re-scores retrieved passages.",
        ],
    )
)
print(df[["query", "answer", "token_f1", "judge_score", "failure_mode"]])
```

Scoring tiers, from cheapest to most informative:

| Tier | Fields | Needs |
|---|---|---|
| 1 | `answer_in_context` | `reference` |
| 2 | `token_f1`, `exact_match` | `reference` |
| 3 | `judge_score` | `reference` + `judge` |

`failure_mode` classifies each answer as `correct`, `partial`,
`retrieval_miss`, `generation_miss`, `refused_*`, or `thinking_truncated`.

---

## 8. Ingesting other content: Office docs, images, SVG

PDF isn't the only supported input. PowerPoint/Word need LibreOffice (installed
in Section 1) to render pages as images for the page-elements classifier:

```python
from pathlib import Path
from nemo_retriever import create_ingestor

documents = [str(Path(f"data/multimodal_test{ext}")) for ext in [".pptx", ".docx"]]
images = [str(Path(f"data/multimodal_test{ext}")) for ext in [".png", ".jpeg", ".bmp"]]

ingestor = create_ingestor(run_mode="batch")
ingestor = ingestor.files(documents + images).extract()
results = ingestor.ingest()
```

SVG requires the optional `cairosvg` dependency (network access needed to
install — skip in air-gapped environments):

```bash
uv pip install "nemo-retriever[multimedia]"   # or: uv pip install "cairosvg>=2.7.0"
```

HTML and plain text have no natural page boundaries, so pair them with
`split_config` to keep chunks under your embedder's max sequence length:

```python
documents = [str(Path(f"data/*{ext}")) for ext in [".txt", ".html"]]
ingestor = create_ingestor(run_mode="batch").files(documents).extract(
    split_config={"text": {"max_tokens": 512}, "html": {"max_tokens": 512}}
)
results = ingestor.ingest()
```

Render any single document's results as markdown:

```python
from nemo_retriever.common.io import to_markdown, to_markdown_by_page

markdown_doc = to_markdown(results[0])           # whole-document string
per_page = to_markdown_by_page(results[0])        # dict[int, str]
```

---

## 9. Audio and video transcription (Parakeet ASR)

NRL transcribes speech with NVIDIA's
[`parakeet-1-1b-ctc-en-us`](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-ctc-en-us.html)
ASR NIM, then embeds the transcript like any other text.

```bash
uv pip install "nemo-retriever[multimedia]"
sudo apt-get install -y ffmpeg
```

> **Single-GPU note:** the support matrix says Parakeet "must run on a
> dedicated additional GPU" when self-hosted alongside the core pipeline. On a
> single H100 you have two practical options: (a) run audio extraction as a
> separate step/process *before or after* the core pipeline so the GPU isn't
> shared concurrently, or (b) skip local hosting entirely and call the **hosted**
> Parakeet endpoint on build.nvidia.com (no local GPU contention at all) — shown
> below.

**Hosted Parakeet (no local GPU contention):**

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params.models import ASRParams

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.mp3")
    .extract_audio(
        asr_params=ASRParams(
            audio_endpoints=("grpc.nvcf.nvidia.com:443", None),
            function_id="<function ID from the Parakeet build.nvidia.com page>",
            auth_token="<your API key>",
            segment_audio=True,  # one element per ASR sentence-like segment
        ),
    )
)
results = ingestor.ingest()
```

**Self-hosted (if you free up the GPU or have a second one)** — see
[audio-video.md](docs/docs/extraction/audio-video.md#run-parakeet-on-the-cluster-helm)
for the Helm-based deployment and the matching `audio_endpoints=("audio:50051", None)`
in-cluster invocation.

Video files (`.mp4`, `.mov`, `.mkv`, `.avi`) reuse this same audio path for
their soundtrack, and combine with OCR on video frames for on-screen text — see
[Video and frame OCR](docs/docs/extraction/audio-video.md#video-and-frame-ocr).

---

## 10. Alternate PDF extraction: Nemotron Parse

[Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)
is an alternative to the default `pdfium` layout path, useful for harder visual
layouts. Install its client extra first:

```bash
uv pip install "nemo-retriever[nemotron-parse]"
```

```python
from nemo_retriever import create_ingestor

ingestor = create_ingestor(run_mode="batch").files(documents).extract(method="nemotron_parse")
results = ingestor.ingest()
```

> **Know the trade-off:** Nemotron Parse's semantic classes don't include
> `Chart`/`Infographic` (it labels regions as `Text`, `Table`, `Picture`,
> `Caption`, etc.). With `extract_method="nemotron_parse"`, chart/infographic
> modality rows are never produced — even with `extract_charts=True` — and any
> chart/infographic-filtered query returns nothing. Use the default `pdfium`
> path when you need chart/infographic detection and filtering (Section 3's
> `--content-types chart` example, for instance).

On a single H100, this model fits as an additional dedicated model alongside
the core pipeline per the
[hardware table](docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements)
(~3.5 GiB weights, ~16 GB disk).

---

## 11. Storing extracted images to disk/S3

Use `.store()` after `.embed()` to persist row-level image payloads (citation
assets, thumbnails) anywhere `fsspec` can write — local disk, S3, MinIO, GCS:

```python
ingestor = (
    create_ingestor(run_mode="batch")
    .files(documents)
    .extract()
    .embed()
    .store(
        storage_uri="s3://my-bucket/citation-assets",   # or a local path
        storage_options={"key": "...", "secret": "..."},  # fsspec auth, S3/MinIO only
    )
    .vdb_upload()
)
```

Stored URIs are written back into the result DataFrame so they're available for
VDB upload and for reranking. The CLI equivalent (root ingest) writes the
*image* artifacts produced by the store stage:

```bash
retriever ingest ./data --store-images-uri ./processed_docs/images
```

`--embed-granularity page` stores page images; `--embed-granularity element`
stores individual element images.

---

## 12. Metadata filtering

Every extracted row carries metadata (page number, source path, detected
element counts, content type). You can attach custom metadata at ingest time
and filter on it at query time — full runnable example in
[`examples/nemo_retriever_retriever_query_metadata_filter.ipynb`](examples/nemo_retriever_retriever_query_metadata_filter.ipynb).

Open that notebook for the end-to-end flow; conceptually:

1. Attach custom fields to documents during `.files()`/`.extract()`.
2. Query with a `where` predicate (server-side) or a client-side filter on the
   returned hits' metadata dict.

See also [Vector databases — Metadata and filtering](docs/docs/extraction/vdbs.md#metadata-and-filtering).

---

## 13. Remote inference: build.nvidia.com NIMs, no local GPU

Everything above can run **without** touching your H100, by routing extraction,
OCR, embedding, and reranking calls to NVIDIA-hosted NIM endpoints. Useful when
you want to compare local vs. hosted quality/latency, or save your H100 for
something else.

```bash
export NVIDIA_API_KEY=nvapi-...
```

```python
ingestor = (
    create_ingestor(run_mode="batch")
    .files(documents)
    .extract(
        page_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3",
        ocr_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1",
        table_structure_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1",
    )
    .embed(
        embed_invoke_url="https://integrate.api.nvidia.com/v1/embeddings",
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embed_modality="text",
    )
    .vdb_upload()
)
```

Equivalent CLI form:

```bash
retriever ingest ./data/multimodal_test.pdf \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

**Important:** at query time, point `Retriever`'s `embed_kwargs` at the *same*
embedding endpoint/model used at ingest time — query and stored vectors must
come from the same embedding space.

---

## 14. Scaling out: Ray and batch mode

NRL uses [Ray Data](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)
under the hood for distributed/batch ingestion. On a single-GPU box, the main
reason to start your own Ray cluster is to watch the dashboard or pin the
pipeline to exactly one GPU explicitly.

Start a local Ray cluster with its dashboard:

```bash
ray start --head
# open http://127.0.0.1:8265
```

Then run any pipeline with `--ray-address auto` to attach to it.

Pin Ray to a single GPU explicitly (useful if other processes also use the
H100, or you want deterministic resource accounting):

```bash
CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1
```

Batch ingest with worker tuning flags:

```bash
retriever ingest batch ./data/pdf_corpus \
  --profile fast-text \
  --pdf-extract-workers 4 \
  --embed-workers 2
```

Worker-count heuristics (`page_elements_workers`, `detect_workers`,
`embed_workers`, etc.) are derived from `gpu_count` automatically — see
[Multi-GPU resource heuristics](nemo_retriever/README.md#multi-gpu-resource-heuristics-library-batch-mode)
in the main README. On a single GPU these heuristics correctly collapse to
`gpu_count=1`, so you rarely need to override them — `override_cpu_count` /
`override_gpu_count` exist if you ever do.

---

## 15. Agentic retrieval

Instead of one static dense-retrieval pass, `--agentic` runs an LLM-driven
ReAct loop: the agent issues several retrieval sub-queries, fuses candidates
with reciprocal rank fusion (RRF), and makes a final selection pass. It's a
drop-in alternative to standard `retriever query` — same LanceDB table, same
embedding config.

```bash
retriever query "how does the ingestion pipeline handle tables?" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5
```

Tune the loop (remote agent endpoint, fewer reasoning rounds):

```bash
retriever query "summarize the deployment options" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions \
  --embed-invoke-url http://localhost:8000/v1 \
  --agentic-react-max-steps 5
```

Key knobs: `--agentic-backend-top-k` (default 20, candidates per retrieval
call), `--agentic-reasoning-effort` (default `high`), `--agentic-temperature`
(default `0.0`), `--agentic-text-truncation` (default `0` = no truncation).

Output differs from dense mode: agentic queries return ranked document IDs as
JSON, each tagged with the stage that selected it (`final_results`, `rrf`, or
`selection_agent`), rather than text-enriched hits.

Background reading: [Agentic retrieval workflow](docs/docs/extraction/workflow-agentic-retrieval.md),
[concept doc](docs/docs/extraction/agentic-retrieval-concept.md).

---

## 16. Running NeMo Retriever as a service (+ MCP for agents)

For long-running or multi-client use, run NRL as a service rather than a one-shot
script. The same service exposes a FastMCP endpoint at `/mcp` so agent
frameworks can call ingestion/query/answer tools directly.

Ingest through a running service:

```bash
retriever ingest service ./data/pdf_corpus \
  --service-url http://localhost:7670 \
  --service-concurrency 8
```

```bash
retriever query service "What is in this corpus?" --service-url http://localhost:7670
```

Use `--service-api-token` (or `NEMO_RETRIEVER_API_TOKEN`) when the service
requires a bearer token. Service mode intentionally hides flags that don't
apply remotely (no `--lancedb-uri`, no Ray tuning, no local endpoint/API-key
overrides) — the service owns its own storage and config.

For a local stdio-based agent that talks to an existing service over MCP:

```bash
retriever service mcp-stdio \
  --service-url http://localhost:7670 \
  --api-token "$NEMO_RETRIEVER_API_TOKEN"
```

A remote agent instead points directly at `https://<retriever-service-host>/mcp`.
The `ingest_documents` MCP tool also accepts inline `content_base64` document
bytes for agents that can't see the service's local filesystem.

For containerized standalone service builds, see
[`nemo_retriever/docker.md`](nemo_retriever/docker.md); for Kubernetes/Helm
production deployment (multi-GPU, NIM microservices), see
[`nemo_retriever/helm/README.md`](nemo_retriever/helm/README.md) — out of scope
for a single-GPU walkthrough but worth knowing it exists for production scale-up.

---

## 17. QA evaluation pipeline (retrieval + answer quality)

Beyond ad-hoc queries, NRL ships a full QA evaluation framework
(`nemo_retriever.evaluation`) that measures end-to-end RAG answer quality:
retrieve context → generate answers with one or more LLMs → score against
ground truth with multi-tier scoring + an LLM judge. Full docs:
[`nemo_retriever/src/nemo_retriever/tools/evaluation/README.md`](nemo_retriever/src/nemo_retriever/tools/evaluation/README.md).

```bash
uv venv qa-retriever --python 3.12
source qa-retriever/bin/activate
uv pip install -e "./nemo_retriever[llm]"
export NVIDIA_API_KEY="nvapi-..."
```

Reproduce the bundled **bo767** benchmark (multimodal Q&A over a real PDF set)
end-to-end:

```bash
# 1. Ingest + embed + save Parquet (~45-90 min on modest hardware; faster on H100)
python -m nemo_retriever.examples.graph_pipeline /path/to/bo767 \
  --vdb-kwargs-json '{"uri":"lancedb","table_name":"nemo-retriever"}' \
  --save-intermediate data/bo767_extracted

# 2. Build a full-page markdown index (improves scores on structured content)
retriever eval build-page-index \
  --parquet-dir data/bo767_extracted \
  --output data/bo767_page_markdown.json

# 3. Export retrieval results for the ground-truth question set
retriever eval export \
  --lancedb-uri lancedb \
  --query-csv data/bo767_annotations.csv \
  --output data/eval/bo767_retrieval_fullpage.json \
  --page-index data/bo767_page_markdown.json

# 4. Run QA evaluation (generation + judging + scoring)
retriever eval run --config nemo_retriever/examples/eval_sweep.yaml
```

Bring your own retrieval system instead of NRL's: produce a JSON file matching
the documented retrieval-JSON interface contract and skip straight to step 4 —
the eval framework only cares about that file shape, not how it was produced.

`retriever eval` also supports querying LanceDB **live**, in-memory
(`retrieval.type: "lancedb"` in the eval config), skipping the export step.

---

## 18. Benchmark harness

For repeatable, code-owned benchmarks (recall, latency, throughput) rather than
one-off QA eval runs, use the developer harness:

```bash
retriever harness list --runsets
retriever harness run <benchmark-name>
```

There's also a standalone `recall.ipynb`-style benchmark over the **bo767**
and **bo10k** corpora in [`evaluation/`](evaluation/) (see
`evaluation/bo767_recall.ipynb`, `evaluation/digital_corpora_download.ipynb`)
if you want a notebook-driven recall study instead of the CLI harness.

For research-grade leaderboard reproduction (Vidore V3, BRIGHT) using
NRL's agentic retrieval pipeline, see the separate
[`retrieval-bench/`](retrieval-bench/README.md) package — this is the
benchmarking codebase behind NRL's 1st-place Vidore V3 and 2nd-place BRIGHT
leaderboard submissions, useful if you want to compare NRL's pipeline against
other retrieval systems on standard datasets.

---

## 19. Framework integrations: LlamaIndex / LangChain

Once chunks are in LanceDB, you don't have to use NRL's own `Retriever` —
plug straight into existing RAG frameworks:

- [`examples/llama_index_multimodal_rag.ipynb`](examples/llama_index_multimodal_rag.ipynb)
- [`examples/langchain_multimodal_rag.ipynb`](examples/langchain_multimodal_rag.ipynb)

There's also a hosted, pre-extracted demo you can query without ingesting
anything yourself: the
[multimodal PDF data extraction for enterprise RAG](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
blueprint on build.nvidia.com.

---

## 20. Troubleshooting cheat sheet

| Symptom | Fix |
|---|---|
| OCR fails with `libcudart.so.13` not found | `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64` |
| vLLM: `InductorError: Python.h: No such file or directory` (during ingest or VL reranker startup) | `sudo apt install python3.12-dev`, restart |
| LanceDB indexing fails on a single small PDF | Point ingestion at a directory with ≥16 chunks; IVF index needs that minimum to train |
| `.pptx`/`.docx` extraction fails to render pages | `sudo apt install libreoffice` |
| Audio/video extraction errors about missing binaries | `sudo apt install ffmpeg` (the Python `ffmpeg-python` wrapper does not install the binary) |
| Query results don't match what you expect after switching embedding endpoints | Make sure `Retriever`'s `embed_kwargs`/`--embed-model-name` matches what was used at ingest time — query and stored vectors must share an embedding space |
| `vdb_op` omitted from `.vdb_upload()` and rows didn't land in LanceDB | Always pass `vdb_op="lancedb"` explicitly; the default falls back to `"milvus"` |
| Want to try a chart/infographic-filtered query but get nothing back | You likely ingested with `extract_method="nemotron_parse"`, which doesn't emit chart/infographic rows — re-ingest with the default `pdfium` path |
| Reranker and core pipeline seem to compete for GPU memory | Not an issue on H100 (≥80 GB) — they run concurrently; this only applies to GPUs <80 GB |

Full troubleshooting reference: [docs/docs/extraction/troubleshoot.md](docs/docs/extraction/troubleshoot.md).

---

## Where to go deeper

- [FAQ](docs/docs/extraction/faq.md)
- [Concepts (chunking, splitting, pipeline model)](docs/docs/extraction/concepts.md)
- [Content metadata schema reference](docs/docs/extraction/content-metadata.md)
- [Environment variables reference](docs/docs/extraction/environment-config.md)
- [Performance guide](docs/docs/extraction/performance_guide.md)
- [Release notes](docs/docs/extraction/releasenotes.md)
- [Starter kits / notebooks](docs/docs/extraction/starter-kits.md)
