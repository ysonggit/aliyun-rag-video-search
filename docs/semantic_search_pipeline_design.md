# Semantic Search Pipeline Design

> This document describes the current EPIC-KITCHENS-based RAG video-search architecture and how
> it migrates to an ego-centric (first-person) capture setting — used for failure-case mining and
> event retrieval in an embodied-AI data foundation.

> **Related**: [Embodied-AI Model Training Whitepaper](embodied_ai_training.html) — analysis of the
> embodied-intelligence models trainable on this data pipeline.

---

## Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Qwen-VL vs BGE-VL Model Comparison](#qwen-vl-vs-bge-vl-model-comparison)
3. [Production Streaming Ingestion Pipeline](#production-streaming-ingestion-pipeline)
4. [Query Latency Guarantees](#query-latency-guarantees)

---

## System Architecture Overview

### Production: streaming ingestion + query separation

```
                     ┌──────────────┐
                     │   Camera /   │
                     │   Uploader   │  ◀── continuous video-stream upload
                     └──────┬───────┘
                            │ MP4 upload
                            ▼
                     ┌──────────────┐
                     │   OSS Bucket │  ◀── raw-videos/ (high durability, low cost)
                     │  raw-videos/ │
                     └──────┬───────┘
                            │ OSS Event Notification (ObjectCreated)
                            ▼
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐       ┌──────────┐        ┌──────────┐
   │ FC-1    │       │ FC-2     │        │ FC-3     │
   │Frame    │       │Embedder  │        │Captioner │
   │Extractor│──────▶│b64→1152  │        │Qwen-VL   │
   │0.5-2fps │       │(DashScope│        │image→JSON│
   │OpenCV   │       │ API)     │        │structured│
   │→ frames │       │batch≤8   │        │~2s/frame │
   └────┬────┘       └────┬─────┘        └────┬─────┘
        │                 │                   │
        ▼                 ▼                   ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │ OSS      │    │ MNS/RocketMQ │    │ MNS/RocketMQ │
   │ frames/  │    │ embed_ready  │    │ capt_ready   │
   └──────────┘    └──────┬───────┘    └──────┬───────┘
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                           ┌──────────────┐
                           │ FC-4         │
                           │ Indexer      │  ◀── scheduled trigger (30s)
                           │ merge embed  │      batch upsert
                           │ +caption     │
                           │ →DashVector  │
                           └──────┬───────┘
                                  │ batch upsert
                                  ▼
                    ┌──────────────────────────┐
                    │     DashVector            │  ◀── vector DB (Singapore)
                    │  real-time writes +       │      queryable immediately
                    │  concurrent queries       │      after write
                    │  HNSW auto-updates        │      (no re-index)
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌──────────────┐            ┌──────────────┐
           │ Write (Ingest)│            │ Read (Query)  │
           │ OSS Event→FC  │            │ HTTP→API GW   │
           │ async, batch  │            │ sync, low-lat │
           │ backlog/retry │            │ provisioned   │
           └──────────────┘            └──────────────┘

  Read/write fully decoupled — separate FC functions; DashVector natively
  supports concurrent read/write.

  ─────────────────────────────────────────────────────────────

  Query service (deployed independently, unaffected by ingestion):

  ┌──────────────┐
  │ API Gateway  │  ◀── GET /search?q=take+cup&top_k=10
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ FC-5         │  ◀── Query Service (provisioned, no cold start)
  │ text→embed   │      DashScope text embedding → 200ms
  │ →DashVector  │      ANN vector retrieval     → 50ms
  │ →top-k       │      array post-filter        → 5ms
  │ →frames+meta │      ─────────────────────────
  └──────────────┘      total latency: ~250ms (warm)
```

### Development: local hybrid pipeline (what runs today)

```
  Your Mac (local)                            Alibaba Cloud (Singapore)
  ┌───────────────────────────┐               ┌──────────────────────┐
  │                           │               │                      │
  │ EPIC-KITCHENS/            │               │  DashScope API       │
  │   P01/videos/*.MP4        │               │  (embedding+caption) │
  │     ↓ (OpenCV)            │               │                      │
  │ storage/frames/           │──base64──────▶│  tongyi-emb-vision   │
  │   P01_03/*.jpg (local)    │               │  → 1152-dim vector   │
  │                           │──base64──────▶│  qwen-vl-max         │
  │                           │◀─vector+JSON──│  → structured caption│
  │                           │               │                      │
  │                           │──upload JPG──▶│  OSS frames/         │
  │                           │               │  (frame storage)     │
  │                           │──upsert──────▶│  DashVector          │
  │                           │               │  (vectors+metadata)  │
  └───────────────────────────┘               └──────────────────────┘

  Data flow:
  ① Video on local disk → OpenCV frame extraction → local JPG files
  ② Local JPG → base64 encode → DashScope Embedding API → 1152-dim vector
  ③ Local JPG → DashScope Qwen-VL API → JSON caption
  ④ Local JPG → upload to OSS (frames accessible in the cloud)
  ⑤ vector + caption metadata → DashVector.upsert()

  Key differences vs the production architecture:
  ┌──────────────────┬────────────────────────┬──────────────────────────┐
  │                  │ Current (local hybrid) │ Production target (cloud) │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ Video storage    │ local disk             │ OSS raw-videos/           │
  │ Frame images     │ local disk + OSS       │ OSS frames/ (native)      │
  │ Frame extraction │ local OpenCV           │ FC-1 (OSS event trigger)  │
  │ Embedding input  │ local file base64      │ OSS URL (no base64)       │
  │ Caption input    │ local file path        │ OSS URL                   │
  │ Trigger          │ manual python script   │ OSS ObjectCreated event   │
  │ Elastic scaling  │ none (single machine)  │ FC auto-scaling           │
  │ Offline impact   │ everything halts       │ none (cloud-autonomous)   │
  └──────────────────┴────────────────────────┴──────────────────────────┘

  Changes to reach the production architecture:
  - Local ingest_epic.py → split into FC-1~4 functions (deploy/fc/ already exists)
  - base64 encoding → call DashScope directly with OSS URLs (faster)
  - Manual run → automatic OSS-event trigger
  - Local frame storage → native OSS storage
```

---

## Qwen-VL vs BGE-VL Model Comparison

### Model specs at a glance

| Dimension | Qwen3-VL-Embedding (2B) | Qwen3-VL-Embedding (8B) | BGE-VL-base | BGE-VL-large | BGE-VL-MLLM-S2 |
|:---|:---|:---|:---|:---|:---|
| **Parameters** | 2B | 8B | 0.15B | 0.43B | 7.57B |
| **Vector dim** | 2048 (64–2048) | 4096 (64–4096) | 512 | n/a | n/a |
| **Model size** | ~4 GB | ~16 GB | 299 MB | 855 MB | 15.14 GB |
| **Architecture** | Qwen3-VL Transformer | Qwen3-VL Transformer | CLIP-ViT-B/16 | CLIP-ViT-L | MLLM (multimodal LLM) |
| **Video support** | ✅ (≤50MB) | ✅ (≤50MB) | ❌ | ❌ | ❌ |
| **Text context** | 32K tokens | 32K tokens | 77 tokens (CLIP) | 77 tokens (CLIP) | n/a |
| **Languages** | 33 | 33 | mostly English | mostly English | mostly English |
| **Independent vectors** | ✅ (default) | ✅ (default) | ✅ | ✅ | ✅ |
| **Fused vectors** | ✅ (enable_fusion) | ✅ (enable_fusion) | ✅ | ✅ | ✅ |
| **License** | Apache 2.0 | Apache 2.0 | MIT | MIT | MIT |

### Benchmark comparison

| Benchmark | Qwen3-VL-8B | Qwen3-VL-2B | BGE-VL-MLLM | Notes |
|:---|:---:|:---:|:---:|:---|
| **MMEB-V2 Overall** | **77.9** | 73.4 | n/a | 78-dataset composite |
| MMEB Image Overall | 80.1 | 75.0 | — | image retrieval + QA |
| MMEB Video Overall | 66.1 | 61.1 | — | video retrieval (BGE-VL unsupported) |
| MMEB VisDoc VR | 88.8 | 86.3 | — | visual-document retrieval |
| **MMTEB Mean (Task)** | 67.88 | 63.87 | — | text-only multitask |
| MMTEB Retrieval | 81.08 | 78.50 | — | text retrieval |
| **CIRCO (composed)** | — | — | BGE-VL-base **beats** 50× larger | BGE-VL's core strength |
| MMEB Zero-Shot | — | — | **SOTA** | BGE-VL-MLLM leads |
| MMEB Fine-Tuning | — | — | **SOTA** (+7.1% over prev) | after fine-tuning |

### Deployment comparison

| Deployment | Qwen-VL (DashScope) | Qwen-VL (self-hosted) | BGE-VL |
|:---|:---|:---|:---|
| **API call** | ✅ DashScope | ❌ | ❌ |
| **Local deploy** | ❌ (cloud model) | ✅ GPU required | ✅ GPU required |
| **Min GPU** | none (API) | A10 (24GB) for 2B, A100 (80GB) for 8B | T4 (16GB) for base, A100 for MLLM |
| **Inference stack** | DashScope SDK | vLLM, sentence-transformers | sentence-transformers, transformers |
| **Deploy complexity** | zero (API) | medium (K8s + GPU) | low (pip install + download weights) |
| **Elastic scaling** | automatic (Alibaba-managed) | needs HPA | manual |

### Cost comparison (per 1M images embedded)

| Model | Mode | Cost / 1M frames | Monthly @ 1M frames |
|:---|:---|:---|:---|
| **tongyi-embedding-vision-plus** (current) | API | ¥0.0005/k tokens ~¥500 | ¥500 |
| **Qwen3-VL-Embedding-2B** | API | ¥0.0018/k tokens ~¥1,800 | ¥1,800 |
| **Qwen3-VL-Embedding-2B** | self-hosted (ACK GPU) | GPU instance ~¥3,000/mo | ¥3,000 |
| **BGE-VL-base** | self-hosted (T4 GPU) | ~¥1,500/mo | ¥1,500 |
| **BGE-VL-MLLM** | self-hosted (A100 GPU) | ~¥15,000/mo | ¥15,000 |

### How to choose for our scenario

```
Decision tree (based on our actual needs):

Need video embedding?
├─ Yes → Qwen-VL (BGE-VL has no video support)
│   └─ Also need zero deployment?
│       ├─ Yes → tongyi-embedding-vision-plus (¥0.0005, 1152-dim)
│       └─ No  → Qwen3-VL-Embedding-2B self-hosted (2048-dim)
│
└─ No (image-only embedding)
    ├─ Need zero deploy + low cost?
    │   └─ tongyi-embedding-vision-plus (¥0.0005, 1152-dim)
    │
    ├─ Need very high precision (composed retrieval)?
    │   └─ BGE-VL-base (CIRCO SOTA, 512-dim, MIT, 299MB)
    │
    └─ Need balanced precision vs cost?
        ├─ via API        → Qwen3-VL-Embedding-2B (MMEB 73.4)
        └─ self-hosted    → BGE-VL-large (855MB, GPU)
```

### Our recommendation

```
Phase 1-2 (validation):  tongyi-embedding-vision-plus (API)
                         why: zero deploy, lowest cost, already in use
                         ✓ 1152-dim, ¥0.0005/k tokens
                         ✓ supports image and video

Phase 3 (prod, images):  consider BGE-VL-base (self-hosted)
                         why: CIRCO composed-retrieval SOTA, only 299MB
                         fits "image + text instruction" ("make the background darker")

Phase 3 (prod, video):   keep tongyi-embedding-vision-plus
                         or upgrade to Qwen3-VL-Embedding-2B (API)
                         why: BGE-VL has no video; Qwen is the only option

Phase 6 (ego-centric):   Qwen3-VL-Embedding-8B (API)
                         why: complex ego scenes, 8B is most accurate
                         cost controllable (¥0.0018/k tokens)
```

### Migration path

```
Now:       tongyi-embedding-vision-plus (DashScope API, 1152-dim)
            ↓  no code change, just swap the model param
Upgrade:   qwen3-vl-embedding (DashScope API, 2560-dim)
            ↓  requires re-ingest (new dimension is incompatible)
Optional:  qwen3-vl-embedding + dimension=1152 (downscale to fit existing collection)
            ↓  keep existing DashVector data, only update the model field
Future:    BGE-VL (self-hosted GPU, for specific composed-retrieval scenarios)
```

---

## Production Streaming Ingestion Pipeline

### Design principle: writes and queries fully decoupled

| Concern | Write path | Query path |
|:---|:---|:---|
| Trigger | OSS Event (async) | HTTP request (sync) |
| Processing | batch (batch upsert) | single request (ANN + filter) |
| Scaling | auto-scale by video volume | auto-scale by QPS |
| Failure handling | MNS dead-letter queue + retry | API Gateway timeout + fallback |
| Latency target | minutes (end-to-end) | milliseconds (P99 < 500ms) |
| FC instances | on-demand (cost-optimized) | provisioned (latency-optimized) |

### Write pipeline: event-driven + batched

```
Step 1: OSS ObjectCreated(video.mp4) event
  └─ triggers FC-1 (FrameExtractor)
     ├─ download video to local /tmp
     ├─ adaptive frame extraction (activity-based 0.2–2.0 fps)
     ├─ save frames to OSS frames/{video_id}/
     └─ send MNS message: {video_id, frame_list}

Step 2: MNS message triggers FC-2 (Embedder) + FC-3 (Captioner)
  └─ run in parallel (no dependency)
     ├─ FC-2: per frame base64 → DashScope Embedding API
     │         batch ≤ 8 frames/request (fewer API calls)
     │         results → sidecar JSON: {video_id}_embeddings.json
     │
     └─ FC-3: per frame image → Qwen-VL (sequential, ~2s/frame)
               JSON caption → sidecar: {video_id}_captions.json

Step 3: scheduled trigger (every 30s) starts FC-4 (Indexer)
  └─ scan OSS frames/ for frames with embed+caption done
     ├─ merge embedding + caption → Doc list
     ├─ batch upsert to DashVector (50 docs/batch)
     └─ mark processed (write done marker or move to processed/)
```

### Backlog control

```
Monitoring:
  unprocessed videos in raw-videos/ > 10 → PagerDuty alert
  FC error rate > 5%                      → auto-scale + notify
  DashVector upsert latency > 2s          → degrade (buffer to MNS, retry later)

Dead-letter queue (DLQ):
  FC-1/2/3 fails 3×  → push to MNS DLQ
  → manual inspection: corrupt video? API throttling? OSS permissions?
  → re-deliver after fix, or mark as skip
```

### Cost control

| Strategy | Saves | Effect |
|:---|:---|:---|
| **FC pay-per-use** | no compute cost when idle | $0/mo when idle |
| **Embedding batch (≤8 imgs/request)** | API call count | −87.5% calls |
| **Adaptive frame extraction** | embedding + caption + storage | −60–80% frames |
| **FC provisioned instances (query only)** | eliminates cold-start latency | ~¥30/mo/instance |
| **OSS lifecycle policy** | cold-data storage cost | 30d→IA, 90d→archive, −70% |

### Scalability estimate

| Scale | Videos/day | Frames/day | Embed cost/day | FC concurrency |
|:---|:---|:---|:---|:---|
| Small | 10 (10 min ea) | ~3,000 | ~¥3 | 1–2 |
| Medium | 100 | ~30,000 | ~¥30 | 3–5 |
| Large | 1,000 | ~300,000 | ~¥300 | 10–20 |

---

## Query Latency Guarantees

### Latency breakdown

```
Full-path query (P99):

  API Gateway routing         ~10ms
  FC cold start (first req)    ~500ms    ← eliminated with provisioned instances
  FC warm start (later reqs)   ~0ms
  ──────────────────────────────────────
  DashScope text embedding    ~200ms    ← Singapore endpoint
  DashVector ANN query        ~50ms     ← HNSW index + Singapore
  Scalar filter (server-side) ~10ms
  Array filter (client-side)  ~5ms
  JSON serialize + HTTP resp  ~10ms
  ──────────────────────────────────────
  Total (warm):               ~275ms
  Total (cold):               ~775ms
```

### Optimizations

| Optimization | Latency reduction | Cost |
|:---|:---|:---|
| FC provisioned instances | removes 500ms cold start | ~¥30/mo |
| Query-result cache (Redis) | same query → 0ms | ~¥50/mo |
| Switch to tongyi-embedding-vision-flash | embedding 200→100ms | cheaper |
| Higher-spec DashVector instance | retrieval 50→30ms | ~¥50/mo extra |
| CDN edge deployment | network ~50→5ms | ~¥100/mo |

### Caching strategy

```python
# Redis cache for hot queries
@cache(ttl=300)  # 5 minutes
def cached_search(query: str, top_k: int) -> list:
    return retriever.search(query, top_k=top_k)

# cache key   = hash(query + filters)
# cache value = [frame_id, score, metadata] (messagepack-compressed)
# estimated hit rate: hot queries (60% of traffic) → 80% hit rate
```

### Degradation strategy

```
Normal path:
  query → embedding → DashVector → top-k → return

Degrade 1 (DashVector timeout > 2s):
  return Redis cached result (possibly stale)

Degrade 2 (DashScope API throttled):
  return "service busy" + HTTP 503
  + auto-retry 3× (exponential backoff)

Degrade 3 (no results):
  return empty result + suggest different keywords
  log to SLS → analyze frequent no-result queries → feed back into caption-prompt tuning
```
