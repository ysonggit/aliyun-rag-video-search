# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RAG semantic search over egocentric (EPIC-KITCHENS-100) kitchen video. Frames are extracted,
captioned with Qwen-VL, embedded with a multimodal model, and stored in DashVector for
text→image vector search with metadata filtering. All AI services are Alibaba Cloud
(DashScope 百炼 + DashVector), deployed serverless to FC in the Singapore region.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run the Streamlit search UI (reads .env)
streamlit run app.py

# End-to-end ingestion (extract → embed → caption → upload OSS → index)
python scripts/ingest_epic.py --fps 2.0 --force --collection scene_frames_2fps

# Verify cloud credentials / connectivity before ingesting
python test_connectivity.py

# Retrieval quality eval (recall@k, MRR) against EPIC ground truth
python tests/test_evaluation.py
DV_COLLECTION_NAME=scene_frames_2fps python tests/test_evaluation.py   # eval a specific collection

# Unit tests (plain unittest/pytest-style scripts, run individually)
python tests/test_frame_extractor.py
python tests/test_retrieval.py
python tests/test_ingestion.py
```

## Configuration

All secrets/regions come from env (`.env` at root; `deploy/.env.cloud` overrides for OSS).
Copy `.env.example` → `.env`. Required: `DASHSCOPE_API_KEY`, `DASHVECTOR_API_KEY`,
`DASHVECTOR_ENDPOINT`. `config.py` centralizes this into frozen dataclasses
(`dashscope_config`, `dashvector_config`, `oss_config`) — read config from there, never
`os.getenv` directly in pipeline code. `config.py` also sets `dashscope.base_http_api_url`
globally; the FC handlers set it again from `DASHSCOPE_BASE_URL` since they don't import config.

Region gotcha: the intl/Singapore DashScope endpoint is `dashscope-intl.aliyuncs.com`
(see `deploy/s.yaml`), different from the default `dashscope.aliyuncs.com` in `config.py`.

## Architecture

Two flows share the `pipeline/` package:

**Ingestion** (`scripts/ingest_epic.py` locally; `deploy/fc/ingest-pipeline/` on OSS trigger):
`FrameExtractor` (OpenCV fixed-interval) → `Embedder` (DashScope MultiModalEmbedding,
1152-dim, batch ≤8) → `Captioner` (Qwen-VL structured JSON: objects/actions/lighting/
occlusion/is_anomaly) → upload frames to OSS → `Indexer` upserts `Doc`s to DashVector.
A `FrameMetadata` dataclass carries state across every stage.

**Query** (`app.py` Streamlit; `deploy/fc/query-service/` HTTP `GET /search`):
text → `Retriever._embed_text` (same embedding model) → DashVector ANN query → results.

### Hybrid filtering — the key non-obvious design

`pipeline/filter_builder.py` splits filters into two tiers because the Singapore DashVector
cluster does NOT support array `contain_any`/`contain_all`:
- **Scalar fields** (lighting, occlusion, category, video_id, is_anomaly, …) → server-side
  SQL-like filter string built by `FilterBuilder`.
- **Array fields** (objects, actions) → client-side post-filter in Python
  (`Retriever._apply_array_filters`).

Because array filtering happens after the vector query, `Retriever.search` over-fetches
(`vector_fetch_multiplier`, default 3×top_k) when a client filter is present, then truncates.

### Schema

`schema.py` defines the DashVector collection `FIELDS_SCHEMA` (cosine metric). It includes
**reserved ego-centric fields** (`episode_id`, `action_label`, `proprioception_ts`,
`view_type`) populated for a future Phase 6, plus `gt_narration` from EPIC used for
evaluation only. When changing stored fields, update `schema.py`, `Indexer.index`'s `fields`
dict, and `Retriever.OUTPUT_FIELDS`/`SearchResult` together.

### Deployment

`deploy/` holds Serverless Devs config (`s.yaml`, FC v3, custom-container) plus separate
Dockerfiles for `ingest`/`query`. FC functions re-import the SAME `pipeline/` modules — keep
pipeline code free of local-only assumptions. `oss_path` field stores a local path during
local ingest and a public OSS URL after upload; `app.py` falls back to "No image" when the
path isn't a local file.

## Conventions

- Pipeline stages mutate `FrameMetadata` in place and return the list; chain them in order.
- API callers (`Embedder`, `Captioner`) implement their own retry w/ exponential backoff and
  write progress to `/tmp/*_progress.txt` to bypass stdout buffering during long runs.
- Qwen-VL has no batch API (one image per call), so `Captioner` captions frames concurrently
  via a thread pool (`max_workers`, default 8, env `CAPTION_WORKERS`); lower it if you hit
  DashScope rate limits. `Embedder` batches ≤8 images per call AND fans those batches out
  concurrently (`max_workers`, default 4, env `EMBED_WORKERS`).
- Ingestion runs embed and caption **concurrently** (disjoint `FrameMetadata` fields) and
  uploads frames to OSS in parallel (`pipeline/oss_uploader.py`, env `UPLOAD_WORKERS`, default
  8). Combined DashScope concurrency ≈ `EMBED_WORKERS + CAPTION_WORKERS` — turn these down
  together if you get throttled.
- `scripts/ingest_epic.py` is resumable: indexed `frame_id`s are appended to
  `storage/manifests/{collection}.txt` (via `Indexer.index`'s `on_batch_indexed` callback) and
  skipped on re-run. It also writes `storage/facets/{collection}.json` (distinct videos/objects/
  categories) which `app.py` reads to populate sidebar filters without a sampling query.
- `embedding_model` must match the `dimension` of the target collection (1152 for
  `tongyi-embedding-vision-plus`); `Embedder.dimension` maps model→dim.
- See `docs/` for design intent: `milestone6_egocentric_design.md` (ego-centric roadmap),
  `phase1_finetune_plan.md` (LoRA fine-tuning plan), `architecture.svg`.
