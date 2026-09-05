# SatQuery AI — Backend

A production backend for the SatQuery AI research notebook: a multi-agent
[LangGraph](https://github.com/langchain-ai/langgraph) pipeline over a
**Qwen2-VL** vision model and a **Groq** LLM that answers natural-language
questions about satellite imagery.

Capabilities (auto-routed by a supervisor agent):

| Task | What it does | Needs |
|------|--------------|-------|
| `image_analysis` | Single-image VQA / scene description (optical or SAR) | 1 image |
| `change_detection` | Bi-temporal "what changed between T1 and T2" | `image_t1` + `image_t2` |
| `cross_modal` | Joint optical + SAR interpretation | `optical` + `sar` |
| `geo_spatial` | CRS / bounds / resolution / pixel-space from the raster | 1 image (rasterio) |
| `retrieval` | Remote-sensing domain-knowledge background | – |

Every visual finding passes a strict **verifier** (strips invented
location/date/sensor/measurements), a **reflection** step decides whether the
evidence is sufficient or a different specialist should retry, and an
**answer-synthesis** step writes the final grounded answer.

```
START ─▶ context_manager ─▶ input_validation ─┬─(invalid)─▶ answer_synthesis ─▶ END
                                              └─(ok)─▶ supervisor
   supervisor ─(router)─▶ image_analysis │ change_detection │ cross_modal │ geo_spatial │ retrieval
        └─▶ evidence_pool ─▶ verification ─▶ reflection ─┬─(retry)────▶ retry ─▶ supervisor
                                                        └─(validated)▶ answer_synthesis ─▶ END
```

`GET /api/v1/graph` returns the live Mermaid diagram.

---

## Project layout

```
satquery/
  config.py            # env-driven settings
  main.py              # FastAPI app factory  (satquery.main:app)
  __main__.py          # `python -m satquery`  -> uvicorn
  api/
    routes/            # /health, /api/v1/analyze[/json], /api/v1/jobs
    schemas.py         # request/response models
    inputs.py          # uploads / URLs / base64  -> files on disk
    deps.py errors.py
  graph/
    builder.py         # StateGraph assembly + compile
    runner.py          # build_initial_state() + run_analysis()
    state.py prompts.py llm.py
    nodes/             # every graph node
  vlm/                 # VLM backend contract + local Qwen2-VL + disabled stub
  geospatial/raster.py # rasterio helpers (optional dep)
  jobs/manager.py      # in-process async job queue
tests/                 # full graph + API, with the VLM & LLM stubbed
scripts/smoke_test.py  # one end-to-end run with stubs
```

---

## Install

Requires **Python 3.11+**.

```bash
python -m venv .venv && source .venv/bin/activate

# 1. Core backend (API + orchestration, CPU-only)
pip install -r requirements.txt && pip install --no-deps -e .

# 2. (optional) local vision model — install torch for your platform first
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-ml.txt

# 3. (optional) geospatial tools
pip install -r requirements-geo.txt
```

`make install`, `make install-ml`, `make install-geo`, `make install-dev` do the same.

---

## Configure

Copy `.env.example` to `.env` and set at least `GROQ_API_KEY`.

| Variable | Default | Notes |
|----------|---------|-------|
| `GROQ_API_KEY` | – | **required** for routing / verification / synthesis |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | any Groq chat model |
| `VLM_BACKEND` | `local` | `local` (Qwen2-VL via transformers) or `disabled` |
| `VLM_MODEL_ID` | `manny2706/satquery-qwen2vl-16bit` | HF checkpoint |
| `VLM_DEVICE_MAP` | `auto` | passed to `from_pretrained` |
| `VLM_LOAD_ON_STARTUP` | `false` | warm the model during app startup |
| `API_KEYS` | – | comma-separated; if set, callers must send `X-API-Key` |
| `CORS_ORIGINS` | `*` | comma-separated origins |
| `MAX_UPLOAD_MB` | `25` | per-file upload cap |
| `DEFAULT_MAX_RETRIES` | `2` | reflection retry budget |
| `MIN_CONFIDENCE` | `0.75` | below this the pipeline retries |
| `WORK_DIR` | `./var/uploads` | where uploads are written |

With `VLM_BACKEND=disabled` (or no `GROQ_API_KEY`) the API still runs — image
specialists return a clear "unavailable" note and the final answer explains
what is missing. `GET /api/v1/status` reports exactly what is configured.

---

## Run

```bash
make dev          # uvicorn with autoreload on :8000
# or
python -m satquery
# or
uvicorn satquery.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Docker

```bash
docker compose up --build          # API only, VLM disabled by default
GROQ_API_KEY=... VLM_BACKEND=disabled docker compose up
```

For in-container GPU inference, enable the ML deps in the `Dockerfile`, switch to
a CUDA base image, and uncomment the GPU block in `docker-compose.yml`.

---

## API

All analysis routes are under `/api/v1` and honour `X-API-Key` when `API_KEYS` is set.

### `POST /api/v1/analyze` — synchronous, multipart

Fields: `query` (required), and any of `optical`, `sar`, `image_t1`,
`image_t2`, `images` (repeatable), plus optional `max_retries`.

```bash
curl -s http://localhost:8000/api/v1/analyze \
  -F 'query=Describe the land cover and any built-up areas.' \
  -F 'optical=@optical1.png'

curl -s http://localhost:8000/api/v1/analyze \
  -F 'query=What changed between the two dates?' \
  -F 'image_t1=@T1.jpg' -F 'image_t2=@T2.jpg'

curl -s http://localhost:8000/api/v1/analyze \
  -F 'query=Compare the optical and SAR views.' \
  -F 'optical=@optical1.png' -F 'sar=@sar1.png'
```

### `POST /api/v1/analyze/json` — synchronous, JSON

```jsonc
{
  "query": "Describe the scene.",
  "optical_image_url": "https://example.com/scene.tif",   // or *_b64
  "max_retries": 2
}
```

### Response (`AnalyzeResult`)

```jsonc
{
  "query": "...",
  "final_answer": "The scene shows ...",
  "confidence": 0.82,
  "current_task": "image_analysis",
  "temporal_mode": "single",
  "modalities": ["optical"],
  "image_count": 1,
  "input_valid": true,
  "validation_errors": [],
  "retry_count": 0,
  "reflection": { "decision": "VALIDATED", "required_action": "NONE", ... },
  "evidence": [
    { "agent": "image_analysis_agent", "task": "image_analysis",
      "finding": "FINDING: ...", "verified_finding": "VALID_FINDING: ...",
      "confidence": 0.82, "status": "ok" }
  ],
  "execution_trace": [ { "node": "context_manager", ... }, ... ],
  "duration_seconds": 7.41
}
```

### Async jobs

VLM inference can take a while; use jobs for non-blocking calls.

```bash
# same multipart fields as /analyze
JOB=$(curl -s http://localhost:8000/api/v1/jobs -F 'query=...' -F 'optical=@o.png' | jq -r .job_id)
curl -s http://localhost:8000/api/v1/jobs/$JOB      # -> status + result when done
curl -s http://localhost:8000/api/v1/jobs           # recent jobs
```

The job queue is in-process and single-worker (one GPU). For horizontal scaling,
put Celery/RQ/Arq behind `satquery.jobs.manager.JobManager`.

### Meta

| Route | Purpose |
|-------|---------|
| `GET /health` | liveness |
| `GET /api/v1/status` | LLM/VLM config, graph nodes, defaults |
| `GET /api/v1/graph` | Mermaid diagram of the compiled graph |

---

## Use the graph directly (no HTTP)

```python
from satquery.graph import run_analysis

state = run_analysis(
    query="Describe the land cover.",
    optical_image="optical1.png",     # path on disk
    # sar_image=..., image_t1=..., image_t2=..., max_retries=2
)
print(state["final_answer"])
```

---

## Tests

```bash
make install-dev
make test          # or: pytest
```

The suite stubs the vision model and the Groq LLM (`satquery.vlm.set_vlm`,
`satquery.graph.llm.set_llm_text_override`), so the **entire graph and API run
offline and deterministically** — no GPU, no API key.

```bash
python scripts/smoke_test.py                 # stubbed end-to-end
SATQUERY_SMOKE_REAL=1 GROQ_API_KEY=... python scripts/smoke_test.py optical.png
```

---

## Notes / deviations from the notebook

- The supervisor is a plain classification call (deterministic rules + one LLM
  fallback) instead of a `deepagents` agent — same routing decision, fewer
  moving parts and dependencies.
- `verification` dispatches to the change / cross-modal / generic verifier
  prompt based on the specialist that produced the evidence (the notebook wired
  only the generic one).
- A `retrieval` specialist and an `input_validation` short-circuit were added so
  the compiled graph is total (every router branch has a node) and unusable
  requests fail fast with a clear message.
- Verifier confidence is parsed from the model output (regex) rather than
  hard-coded.
