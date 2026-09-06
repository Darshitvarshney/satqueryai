# SatQuery AI — API test guide (Postman)

Complete manual/automated test pass for every endpoint and feature.

`docs/postman/`:

| File | Purpose |
|------|---------|
| `SatQuery.postman_collection.json` | 22 requests across 6 folders, 79 assertions |
| `SatQuery.local.postman_environment.json` | `base_url`, `api_key` |
| `samples/` | `optical1.png`, `sar1.png`, `T1.png`, `T2.png` — ready-to-attach test images |

---

## 1. Prerequisites

Start the backend (`python -m satquery`, default `http://localhost:8000`) and
confirm `GET /health`.

The response shape is identical in every deployment mode — only the *content* of
`final_answer` / `evidence[].finding` changes:

| Mode | `.env` | Notes |
|------|--------|-------|
| Full | real `GROQ_API_KEY`, `VLM_BACKEND=local` (+ torch/torchvision) | real answers; `evidence[].status = "ok"`, `verified_finding` filled |
| LLM only | real `GROQ_API_KEY`, `VLM_BACKEND=disabled` | routing/verification/synthesis real; image tasks report "unavailable" |
| Plumbing only | any key, `VLM_BACKEND=disabled` | every endpoint 200 with a graceful `final_answer`; good for CI |

The collection assertions pass in **all three modes** — they check structure and
routing, and only check `verified_finding` when the VLM produced a usable finding.

---

## 2. Import

1. **File → Import** → both JSON files in `docs/postman/`.
2. Environment dropdown → **“SatQuery — local”**. Edit `base_url` if needed.
3. **Sample images** — the multipart requests reference `docs/postman/samples/*.png`.
   Either set **Settings → General → Working directory** to the repo root, or
   attach each file by hand (Body tab → the file row → *Select File*).

---

## 3. Run

- Single request: open, **Send**, read the **Test Results** tab.
- Whole collection: **Collection Runner** → *SatQuery AI Backend* → the
  environment → **Run** (folders are numbered; the Jobs folder stores `job_id`).
- CLI:
  ```bash
  npm i -g newman
  make test-api        # newman run docs/postman/SatQuery.postman_collection.json -e ... --working-dir .
  ```

> **Async:** *04 · Jobs → Get job* may return `queued`/`running` on the first
> send — **Send again** until `succeeded`. Set `VLM_LOAD_ON_STARTUP=true` to
> avoid a slow first job.

---

## 4. Folders

### 00 · Meta

| Request | Asserts |
|---|---|
| `GET /health` | 200, `status:"ok"`, `version` |
| `GET /api/v1/status` | `llm.fallback_chain` list, `vlm` block, `graph_nodes` ⊇ 5 specialists |
| `GET /api/v1/graph` | Mermaid text with `context_manager` … `answer_synthesis` |

### 01 · Analyze — sync (multipart)  →  `POST /api/v1/analyze`

Form fields: `query` (required), `optical`, `sar`, `image_t1`, `image_t2`,
`images` (repeatable), `max_retries`.

**Routing is decided by the query + the image count — not by which field you
use.** The labelled fields only tell a specialist which image is which.

| Request | Sends | Asserts |
|---|---|---|
| Image analysis — optical | `query` + `optical` | `current_task = image_analysis`, `temporal_mode = single`, trace ends `answer_synthesis`, `verified_finding` (if VLM ok) |
| Image analysis — SAR | `query` + `sar` | `current_task = image_analysis`, `modalities = ["sar"]` |
| Change detection — labelled T1/T2 | `query` + `image_t1` + `image_t2` | `temporal_mode = bi_temporal`, `current_task = change_detection` |
| **Change detection — unlabelled `images[]`** | change query + 2× `images` | `image_count = 2`, `current_task = change_detection`, `change_detection` node ran |
| Cross-modal — labelled optical + SAR | `query` + `optical` + `sar` | `current_task = cross_modal`, `modalities = [optical, sar]` |
| **Cross-modal — unlabelled `images[]`** | optical/SAR query + 2× `images` | `image_count = 2`, `current_task = cross_modal`, `cross_modal` node ran |
| Geo-spatial | geometry query + `optical` | `current_task = geo_spatial`, `finding.pixel_space.width_pixels` numeric |
| Retrieval — domain knowledge | "explain in general …" + `optical` | 200 + `final_answer`; task `retrieval` **or** `image_analysis` |
| Validation error — no image | `query` referencing an image, no file | **422**, body mentions "image" |
| Bounded retries — max_retries=1 | `query` + `optical` + `max_retries=1` | `retry_count ≤ 1` |

### 02 · Analyze — JSON  →  `POST /api/v1/analyze/json`

| Request | Asserts |
|---|---|
| From base64 (`optical_image_b64`) | 200, `current_task = image_analysis` |
| From URL (`optical_image_url`, needs internet) | 200 on success, **400** if the URL can't be fetched |

### 03 · Streaming (SSE)  →  `POST /api/v1/analyze/stream`

Asserts `Content-Type: text/event-stream` and `event: start` / `progress` /
`result` frames, plus `final_answer` in the result.

> Postman buffers the whole stream. For live frames: `curl -N .../analyze/stream -F 'query=...' -F 'optical=@...'`

### 04 · Jobs — async

| Request | Asserts |
|---|---|
| Create job (`POST /api/v1/jobs`) | **202**, `job_id`, saved to `{{job_id}}` |
| Get job (`GET /api/v1/jobs/{{job_id}}`) | 200; `succeeded` → `result.final_answer`; `failed` → `error` |
| List jobs (`GET /api/v1/jobs?limit=20`) | 200, array, contains `{{job_id}}` |
| Unknown id | **404** |

### 05 · Auth  (start server with `API_KEYS=test-key`, set env `api_key=test-key`)

| Request | Asserts |
|---|---|
| Missing key → 401 | auth forced off → **401** (or 200 if the server has no `API_KEYS`) |
| Valid key → 200 | inherits `X-API-Key: {{api_key}}` → **200** |

`/health`, `/api/v1/status`, `/api/v1/graph` stay public by design.

---

## 5. Feature → test map

| Feature | Proven by |
|---|---|
| Routing → each of the 5 specialists | 01 · the six analyze requests |
| **Routing by query + image count (not by upload field)** | 01 · "unlabelled `images[]`" change + cross-modal requests |
| Two-image specialist fills its pair from the generic pool | same two requests (`_resolve_pair`) |
| Context detection (`image_count`, `temporal_mode`, `modalities`) | 01 · all (asserted) |
| Input validation short-circuit | 01 · Validation error |
| Evidence verification (`verified_finding`, confidence) | 01 · Image analysis (when VLM ok) |
| Reflection + bounded retry | 01 · Bounded retries |
| Answer synthesis | every 200 analyze response |
| SSE streaming | 03 |
| Async jobs | 04 |
| API-key auth | 05 |
| Image ingest: multipart / base64 / URL | 01 / 02 |
| LLM fallback chain (config) | 00 · Status (`llm.fallback_chain`) |
| Graceful degradation (no VLM / bad key) | any request in *Plumbing only* mode still returns 200 |

---

## 6. Outside Postman

**Actual LLM failover.** Set a bad primary + good fallback and watch the log:
```
GROQ_MODEL=this-model-does-not-exist
GROQ_FALLBACK_MODELS=llama-3.3-70b-versatile
```
(unit test: `tests/test_llm_fallback.py::test_primary_failure_falls_through`.)

**Upload size limit.** Attach a file > `MAX_UPLOAD_MB` (default 25) → **400**.

**CORS.** From an allowed origin, `OPTIONS /api/v1/analyze` returns the
`Access-Control-Allow-*` headers.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Every analyze request 401 | `api_key` env var ≠ server `API_KEYS`. Align or clear both. |
| "file not found" on multipart | set Postman working directory to the repo root, or attach the file. |
| `final_answer` says the vision model is unavailable | `VLM_BACKEND=disabled` or the model failed to load — see README → Troubleshooting (`torchvision`, `bitsandbytes`). |
| Job stuck `running` | first call loads the model; set `VLM_LOAD_ON_STARTUP=true`. |
| `analyze/json` URL test → 400 | no internet / host blocked — use the base64 request. |
