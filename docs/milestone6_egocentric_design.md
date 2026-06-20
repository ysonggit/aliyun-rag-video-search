# Milestone 6: Ego-Centric 场景映射设计文档

> 本文档说明如何将当前基于 EPIC-KITCHENS 的 RAG 视频检索架构迁移到 ego-centric（第一人称视角）采集场景，用于 embodied-AI 数据底座中的失败案例挖掘与事件检索。

> **相关文档**: [Embodied-AI 模型训练白皮书](embodied_ai_training.html) — 基于本数据管道可训练的具身智能模型分析

---

## 系统架构总览

### 生产环境：流式摄入 + 查询分离

```
                     ┌──────────────┐
                     │   Camera /   │
                     │   Uploader   │  ◀── 持续上传视频流
                     └──────┬───────┘
                            │ MP4 upload
                            ▼
                     ┌──────────────┐
                     │   OSS Bucket │  ◀── raw-videos/ (高持久性, 低成本)
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
   │0.5-2fps │       │(DashScope│        │图片→JSON │
   │OpenCV   │       │ API)     │        │结构化描述 │
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
                           │ Indexer      │  ◀── 定时触发 (30s)
                           │ 合并embed    │      批量 upsert
                           │ +caption     │
                           │ →DashVector  │
                           └──────┬───────┘
                                  │ batch upsert
                                  ▼
                    ┌──────────────────────────┐
                    │     DashVector            │  ◀── 向量库 (新加坡)
                    │  实时写入 + 并发查询       │      写后可立刻查
                    │  HNSW 索引自动更新         │      无需 re-index
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌──────────────┐            ┌──────────────┐
           │ 写路径 (Ingest)│            │ 读路径 (Query) │
           │ OSS Event→FC  │            │ HTTP→API GW   │
           │ 异步, 批处理   │            │ 同步, 低延迟   │
           │ 可积压, 可重试 │            │ 预留实例       │
           └──────────────┘            └──────────────┘

  读/写完全解耦 — 各走各的 FC 函数, DashVector 原生支持并发读写

  ─────────────────────────────────────────────────────────────

  查询服务 (独立部署, 不受摄入影响):

  ┌──────────────┐
  │ API Gateway  │  ◀── GET /search?q=take+cup&top_k=10
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ FC-5         │  ◀── Query Service (预留实例, 无冷启动)
  │ 文本→embed   │      DashScope text embedding → 200ms
  │ →DashVector  │      ANN 向量检索 → 50ms
  │ →top-k 结果  │      数组后过滤 → 5ms
  │ →返回帧+meta │      ─────────────────
  └──────────────┘      总延迟: ~250ms (热启动)
```

### 开发环境：本地混合管线 (当前实际运行)

```
  你的 Mac (本地)                              阿里云 (新加坡)
  ┌───────────────────────────┐               ┌──────────────────────┐
  │                           │               │                      │
  │ EPIC-KITCHENS/            │               │  DashScope API       │
  │   P01/videos/*.MP4        │               │  (embedding+caption) │
  │     ↓ (OpenCV)            │               │                      │
  │ storage/frames/           │──base64──────▶│  tongyi-emb-vision   │
  │   P01_03/*.jpg (本地磁盘)  │               │  → 1152-dim 向量     │
  │                           │──base64──────▶│  qwen-vl-max         │
  │                           │◀──向量+JSON───│  → 结构化 caption    │
  │                           │               │                      │
  │                           │──upload JPG──▶│  OSS frames/         │
  │                           │               │  (帧图片存储)         │
  │                           │──upsert──────▶│  DashVector          │
  │                           │               │  (向量+metadata)      │
  └───────────────────────────┘               └──────────────────────┘

  数据流:
  ① 视频在本地磁盘 → OpenCV 抽帧 → 本地 JPG 文件
  ② 本地 JPG → base64 编码 → DashScope Embedding API → 1152-dim 向量
  ③ 本地 JPG → DashScope Qwen-VL API → JSON caption
  ④ 本地 JPG → 上传到 OSS (帧图片云端可访问)
  ⑤ 向量 + caption metadata → DashVector.upsert()

  关键区别 vs 生产架构:
  ┌──────────────────┬────────────────────────┬──────────────────────────┐
  │                  │ 当前 (本地混合)          │ 生产目标 (全云端)          │
  ├──────────────────┼────────────────────────┼──────────────────────────┤
  │ 视频存储         │ 本地磁盘                 │ OSS raw-videos/           │
  │ 帧图片           │ 本地磁盘 + OSS (上传后)  │ OSS frames/ (原生)        │
  │ 抽帧             │ 本地 OpenCV              │ FC-1 (OSS event 触发)     │
  │ Embedding 输入   │ 本地文件 base64          │ OSS URL (无需 base64)     │
  │ Caption 输入     │ 本地文件路径             │ OSS URL                   │
  │ 触发方式         │ 手动 python script       │ OSS ObjectCreated 事件    │
  │ 弹性伸缩         │ 无 (单机)                │ FC 自动伸缩               │
  │ 断网影响         │ 全部中断                 │ 无 (云端自治)             │
  └──────────────────┴────────────────────────┴──────────────────────────┘

  迁移到生产架构的改动:
  - 本地 ingest_epic.py → 拆成 FC-1~4 函数 (已有 deploy/fc/ 代码)
  - base64 编码 → 直接用 OSS URL 调 DashScope API (更快)
  - 手动运行 → OSS Event 自动触发
  - 本地帧存储 → OSS 原生存储
```

## Embedding 原理详解

### 什么是多模态 Embedding？

Embedding 是将高维的非结构化数据（图片、文本）压缩为固定长度的稠密向量，
使得**语义相似的内容在向量空间中距离接近**。

```
图片 "一个人拿着杯子"     ──Embedding──▶  [0.12, -0.34, 0.89, ..., 0.05]  (1152维)
文本 "holding a cup"      ──Embedding──▶  [0.11, -0.36, 0.91, ..., 0.03]  (1152维)

cosine_similarity = 0.94  ← 语义高度相似，向量接近！
```

### tongyi-embedding-vision-plus 的工作流程

```
输入帧 (JPG, 1920×1080)
  │
  ├─ Step 1: base64 编码
  │     frame_bytes → "data:image/jpeg;base64,/9j/4AAQ..."
  │
  ├─ Step 2: Vision Transformer (ViT) 编码
  │     图片被切为 16×16 patches → 序列输入 Transformer
  │     → 输出 1152-dim 特征向量
  │     → 经过对比学习训练 (CLIP-style):
  │        正样本: 匹配的 图片-文本 pair 拉近
  │        负样本: 不匹配的 pair 推远
  │
  └─ Step 3: 归一化
        向量 L2-normalize → ||v|| = 1
        便于余弦相似度计算: cos(a,b) = a·b
```

### 为什么选 1152 维？

| 维度 | 语义容量 | 存储/1M向量 | 检索速度 | 适合场景 |
|:---|:---|:---|:---|:---|
| 256 | 低 | ~1 GB | 极快 | 粗筛，大规模 |
| 768 | 中 | ~3 GB | 快 | 一般应用 |
| **1152** | **高** | **~4.6 GB** | **中** | **我们的选择** |
| 2560 | 极高 | ~10 GB | 慢 | 高精度需求 |

**Rationale**：1152 维是 tongyi-embedding-vision-plus 的固定输出维度，
由模型训练时确定，在语义容量和计算效率之间取平衡点。

### 同一个语义空间 — 跨模态检索的核心

所有 tongyi 多模态模型共享同一个语义空间。这意味着：

```
图片 embedding 和 文本 embedding 可以直接比较！

"take cup" ──text embed──▶  v_text  = [0.12, -0.34, ...]
  cup图片  ──image embed─▶  v_image = [0.11, -0.36, ...]
                              cos(v_text, v_image) = 0.94  ← 可比较！
```

因此检索时无需额外对齐：文本 query → text embedding → 直接和图片向量算相似度。

---

## 向量数据库原理

### 核心问题：如何从百万级向量中快速找到 Top-K？

```
暴力搜索:
  query_vec vs 1,000,000 个向量 → 1,000,000 次比较
  → 太慢 (秒级延迟)

近似最近邻 (ANN):
  用索引结构预排序，只搜索候选子集
  → 毫秒级延迟，召回率 > 95%
```

### DashVector 的 HNSW 索引

HNSW (Hierarchical Navigable Small World) — 多层图索引：

```
Layer 2 (稀疏):  ○────○          ← 长距离跳转，快速定位区域
                 │
Layer 1 (中等):  ○──○──○──○      ← 中距离导航
                 │  │  │
Layer 0 (密集):  ○─○─○─○─○─○─○  ← 精确搜索 (所有向量都在 Layer 0)

搜索过程:
  query_vec →
    从顶层入口点开始
    → 逐层下降 (greedy search 每层最近邻)
    → 到达 Layer 0 进行精确搜索
    → 返回 top-K
```

### 混合检索流程

DashVector 支持向量召回 + 标量字段过滤的组合查询：

```
query("take cup", top_k=10, video_id="P01_03")

  Step 1: 向量召回 (ANN)
    ┌──────────────────────────────────┐
    │ text "take cup" → DashScope       │
    │ → 1152-dim query vector           │
    └──────────────┬───────────────────┘
                   ▼
    ┌──────────────────────────────────┐
    │ DashVector HNSW 索引搜索           │
    │ → 找到最相似的所有帧向量            │
    └──────────────┬───────────────────┘
                   ▼
  Step 2: 标量过滤 (server-side)
    ┌──────────────────────────────────┐
    │ filter: video_id = "P01_03"       │
    │ → 过滤掉其他视频的帧               │
    └──────────────┬───────────────────┘
                   ▼
  Step 3: 数组过滤 (client-side)
    ┌──────────────────────────────────┐
    │ objects contain "cup"            │
    │ → Python 端后处理过滤              │
    └──────────────┬───────────────────┘
                   ▼
             top-10 结果
```

### 为什么数组过滤放客户端？

新加坡 DashVector 集群**不支持 `contain_any` / `contain_all` 操作符**。
解决方案：
- **标量字段**（video_id, lighting, category）→ DashVector 服务端过滤
- **数组字段**（objects, actions）→ Python 客户端后过滤

```
Server-side filters (DashVector SQL-like):
  ✅ lighting = "bright"
  ✅ video_id = "P01_03"
  ✅ is_anomaly = false
  ✅ action_label = "take"

Client-side filters (Python):
  ✅ any(obj in frame.objects for obj in query_objects)
  ✅ all(act in frame.actions for act in query_actions)
```

**Rationale**：当前数据集小（164-650 帧），客户端过滤无性能压力。
规模扩大后可用 `objects_str` 逗号分隔字段 + `like` 过滤迁移到服务端。

### 相似度度量

DashVector 使用**余弦相似度**作为默认度量 (metric="cosine")：

```
cosine(a, b) = (a · b) / (||a|| × ||b||)

值域: [-1, 1]
  1.0  = 完全相同方向 (最相关)
  0.0  = 正交 (无关)
 -1.0  = 完全相反 (最不相关)

DashVector 返回的 score = cosine_similarity
```

---

## Qwen-VL vs BGE-VL 模型对比

### 模型规格总览

| 维度 | Qwen3-VL-Embedding (2B) | Qwen3-VL-Embedding (8B) | BGE-VL-base | BGE-VL-large | BGE-VL-MLLM-S2 |
|:---|:---|:---|:---|:---|:---|
| **参数量** | 2B | 8B | 0.15B | 0.43B | 7.57B |
| **向量维度** | 2048 (64~2048) | 4096 (64~4096) | 512 | 未公布 | 未公布 |
| **模型大小** | ~4 GB | ~16 GB | 299 MB | 855 MB | 15.14 GB |
| **架构** | Qwen3-VL Transformer | Qwen3-VL Transformer | CLIP-ViT-B/16 | CLIP-ViT-L | MLLM (多模态大模型) |
| **视频支持** | ✅ (≤50MB) | ✅ (≤50MB) | ❌ | ❌ | ❌ |
| **文本上下文** | 32K tokens | 32K tokens | 77 tokens (CLIP) | 77 tokens (CLIP) | 未公布 |
| **语言** | 33种 | 33种 | 英文为主 | 英文为主 | 英文为主 |
| **独立向量** | ✅ (默认) | ✅ (默认) | ✅ | ✅ | ✅ |
| **融合向量** | ✅ (enable_fusion) | ✅ (enable_fusion) | ✅ | ✅ | ✅ |
| **开源协议** | Apache 2.0 | Apache 2.0 | MIT | MIT | MIT |

### Benchmark 对比

| Benchmark | Qwen3-VL-8B | Qwen3-VL-2B | BGE-VL-MLLM | 说明 |
|:---|:---:|:---:|:---:|:---|
| **MMEB-V2 Overall** | **77.9** | 73.4 | 未公布 | 78个数据集综合 |
| MMEB Image Overall | 80.1 | 75.0 | — | 图像检索+QA |
| MMEB Video Overall | 66.1 | 61.1 | — | 视频检索 (BGE-VL不支持) |
| MMEB VisDoc VR | 88.8 | 86.3 | — | 视觉文档检索 |
| **MMTEB Mean (Task)** | 67.88 | 63.87 | — | 纯文本多任务 |
| MMTEB Retrieval | 81.08 | 78.50 | — | 文本检索 |
| **CIRCO (组合检索)** | — | — | BGE-VL-base **超越** 50× 大模型 | BGE-VL 的核心优势 |
| MMEB Zero-Shot | — | — | **SOTA** | BGE-VL-MLLM 领先 |
| MMEB Fine-Tuning | — | — | **SOTA** (+7.1% over prev) |   经过微调后 |

### 部署方式对比

| 部署方式 | Qwen-VL (DashScope) | Qwen-VL (自建) | BGE-VL |
|:---|:---|:---|:---|
| **API 调用** | ✅ DashScope | ❌ | ❌ |
| **本地部署** | ❌ (云端模型) | ✅ GPU 需要 | ✅ GPU 需要 |
| **最低 GPU** | 无需 (API) | A10 (24GB) for 2B, A100 (80GB) for 8B | T4 (16GB) for base, A100 for MLLM |
| **推理框架** | DashScope SDK | vLLM, sentence-transformers | sentence-transformers, transformers |
| **部署复杂度** | 零（API 调用） | 中（K8s + GPU） | 低（pip install + 下载权重） |
| **弹性伸缩** | 自动（阿里云托管） | 需配置 HPA | 需手动 |

### 成本对比 (每百万张图片 embedding)

| 模型 | 模式 | 成本/百万帧 | 月成本@100万帧 |
|:---|:---|:---|:---|
| **tongyi-embedding-vision-plus** (当前) | API | ¥0.0005/千token ~¥500 | ¥500 |
| **Qwen3-VL-Embedding-2B** | API | ¥0.0018/千token ~¥1,800 | ¥1,800 |
| **Qwen3-VL-Embedding-2B** | 自建 (ACK GPU) | GPU实例费 ~¥3,000/月 | ¥3,000 |
| **BGE-VL-base** | 自建 (T4 GPU) | ~¥1,500/月 | ¥1,500 |
| **BGE-VL-MLLM** | 自建 (A100 GPU) | ~¥15,000/月 | ¥15,000 |

### 在我们的场景中如何选择？

```
决策树 (基于我们的实际需求):

需要视频 embedding?
├─ 是 → Qwen-VL (BGE-VL 不支持视频)
│   └─ 也需零部署?
│       ├─ 是 → tongyi-embedding-vision-plus (¥0.0005, 1152-dim)
│       └─ 否 → Qwen3-VL-Embedding-2B 自建 (2048-dim)
│
└─ 否 (仅图片 embedding)
    ├─ 需零部署 + 低成本?
    │   └─ tongyi-embedding-vision-plus (¥0.0005, 1152-dim)
    │
    ├─ 需极高精度 (组合检索)?
    │   └─ BGE-VL-base (CIRCO SOTA, 512-dim, MIT, 299MB)
    │
    └─ 需平衡精度与成本?
        ├─ 用 API → Qwen3-VL-Embedding-2B (MMEB 73.4)
        └─ 自建 → BGE-VL-large (855MB, GPU)
```

### 我们的推荐

```
Phase 1-2 (验证):     tongyi-embedding-vision-plus (API)
                      理由: 零部署, 最低成本, 当前正在用
                      ✓ 1152-dim, ¥0.0005/千token
                      ✓ 支持图片和视频

Phase 3 (生产, 图片):  考虑 BGE-VL-base (自建)
                      理由: CIRCO 组合检索 SOTA, 仅 299MB
                      适合 "图片+文本指令" 场景 ("把背景变暗")

Phase 3 (生产, 视频):  保持 tongyi-embedding-vision-plus
                      或升级到 Qwen3-VL-Embedding-2B (API)
                      理由: BGE-VL 不支持视频, Qwen 是唯一选择

Phase 6 (ego-centric): Qwen3-VL-Embedding-8B (API)
                      理由: ego 场景复杂图片, 8B 精度最高
                      成本可控 (¥0.0018/千token)
```

### 迁移路径

```
当前:   tongyi-embedding-vision-plus (DashScope API, 1152-dim)
         ↓  代码无需改动, 只换 model 参数
升级:   qwen3-vl-embedding (DashScope API, 2560-dim)
         ↓  需重新 inget (新维度不兼容)
可选:   qwen3-vl-embedding + dimension=1152 (降维兼容现有 collection)
         ↓  保留现有 DashVector 数据, 只更新 model 字段
未来:   BGE-VL (自建 GPU, 用于特定组合检索场景)
```

---

## 生产流式摄入管道

### 设计原则：写入和查询完全解耦

| 关注点 | 写入路径 | 查询路径 |
|:---|:---|:---|
| 触发方式 | OSS Event 异步触发 | HTTP Request 同步触发 |
| 处理模式 | 批处理 (batch upsert) | 单次请求 (ANN + filter) |
| 伸缩策略 | 按视频量自动伸缩 | 按 QPS 自动伸缩 |
| 失败处理 | MNS 死信队列 + 重试 | API Gateway 超时 + fallback |
| 延迟目标 | 分钟级 (端到端) | 毫秒级 (P99 < 500ms) |
| FC 实例 | 按需 (cost-optimized) | 预留 (latency-optimized) |

### 写入管线：事件驱动 + 批处理

```
Step 1: OSS ObjectCreated(video.mp4) 事件
  └─ 触发 FC-1 (FrameExtractor)
     ├─ 下载视频到本地 /tmp
     ├─ 自适应抽帧 (activity-based 0.2-2.0 fps)
     ├─ 帧保存到 OSS frames/{video_id}/
     └─ 发 MNS 消息: {video_id, frame_list}

Step 2: MNS 消息触发 FC-2 (Embedder) + FC-3 (Captioner)
  └─ 并行执行 (无依赖)
     ├─ FC-2: 每帧 base64 → DashScope Embedding API
     │         batch ≤ 8 帧/请求 (减少 API 调用)
     │         结果写入 sidecar JSON: {video_id}_embeddings.json
     │
     └─ FC-3: 每帧图片 → Qwen-VL (sequential, ~2s/frame)
               JSON caption 写入 sidecar: {video_id}_captions.json

Step 3: 定时触发器 (every 30s) 启动 FC-4 (Indexer)
  └─ 扫描 OSS frames/ 中已完成 embed+caption 的帧
     ├─ 合并 embedding + caption → Doc 列表
     ├─ 批量 upsert 到 DashVector (50 docs/batch)
     └─ 标记已处理 (写 done marker 或 move to processed/)
```

### 积压控制

```
监控指标:
  raw-videos/ 未处理视频数 > 10 → PagerDuty 告警
  FC 函数错误率 > 5% → 自动扩容 + 通知
  DashVector upsert 延迟 > 2s → 降级 (写入 MNS 缓存, 稍后重试)

死信队列 (DLQ):
  FC-1/2/3 失败 3 次 → 丢入 MNS DLQ
  → 人工检查: 视频损坏? API 限流? OSS 权限?
  → 修复后重新投递, 或标记为 skip
```

### 成本控制

| 策略 | 节省什么 | 效果 |
|:---|:---|:---|
| **FC 按需计费** | 无视频时不产生计算费 | 空闲时 $0/月 |
| **Embedding batch (≤8 图/请求)** | API 调用次数 | 减少 87.5% 调用量 |
| **自适应抽帧** | embedding + caption + 存储 | 减少 60-80% 帧数 |
| **FC 预留实例 (仅查询服务)** | 消除冷启动延迟 | ~¥30/月/实例 |
| **OSS 生命周期策略** | 冷数据存储费 | 30天→低频, 90天→归档, 降低 70% |

### 扩展性估算

| 规模 | 视频/天 | 帧/天 | 嵌入成本/天 | 所需 FC 并发 |
|:---|:---|:---|:---|:---|
| 小 | 10 (10 min ea) | ~3,000 | ~¥3 | 1-2 |
| 中 | 100 | ~30,000 | ~¥30 | 3-5 |
| 大 | 1,000 | ~300,000 | ~¥300 | 10-20 |

---

## 查询延迟保证

### 延迟分解

```
查询请求全链路 (P99):

  API Gateway 路由            ~10ms
  FC 冷启动 (首次请求)         ~500ms    ← 用预留实例消除
  FC 热启动 (后续请求)         ~0ms
  ──────────────────────────────────────
  DashScope text embedding    ~200ms    ← 新加坡 endpoint
  DashVector ANN query        ~50ms     ← HNSW 索引 + 新加坡
  Scalar filter (server-side) ~10ms
  Array filter (client-side)  ~5ms
  JSON 序列化 + HTTP 响应      ~10ms
  ──────────────────────────────────────
  Total (热启动):             ~275ms
  Total (冷启动):             ~775ms
```

### 优化手段

| 优化 | 延迟降低 | 成本 |
|:---|:---|:---|
| FC 预留实例 | 消 500ms 冷启动 | ~¥30/月 |
| 查询结果缓存 (Redis) | 相同 query → 0ms | ~¥50/月 |
| 换成 tongyi-embedding-vision-flash | embedding 200→100ms | 更便宜 |
| DashVector 更高规格实例 | 检索 50→30ms | ~¥50/月额外 |
| CDN 边缘部署 | 网络延迟 ~50→5ms | ~¥100/月 |

### 缓存策略

```python
# Redis 缓存高频查询
@cache(ttl=300)  # 5分钟
def cached_search(query: str, top_k: int) -> list:
    return retriever.search(query, top_k=top_k)

# 缓存 key = hash(query + filters)
# 缓存 value = [frame_id, score, metadata] (消息pack压缩)
# 命中率估计: 高频查询 (60% of traffic) → 80% hit rate
```

### 降级策略

```
正常路径:
  query → embedding → DashVector → top-k → 返回

降级 1 (DashVector 超时 > 2s):
  返回 Redis 缓存结果 (可能过期)

降级 2 (DashScope API 限流):
  返回 "service busy" + HTTP 503
  + 自动重试 3 次 (exponential backoff)

降级 3 (无结果):
  返回空结果 + 建议更换关键词
  记录到 SLS → 分析高频无结果 query → 反馈给 caption prompt 优化
```

---

## 目录

1. [系统架构总览](#系统架构总览)
   - [生产环境：流式摄入 + 查询分离](#生产环境流式摄入--查询分离)
   - [开发环境：本地验证管线](#开发环境本地验证管线)
2. [Embedding 原理详解](#embedding-原理详解)
3. [向量数据库原理](#向量数据库原理)
4. [生产流式摄入管道](#生产流式摄入管道)
5. [查询延迟保证](#查询延迟保证)
6. [Qwen-VL vs BGE-VL 模型对比](#qwen-vl-vs-bge-vl-模型对比)
7. [总体迁移路线](#1-总体迁移路线)
8. [抽帧策略适配](#2-抽帧策略适配)
9. [Schema 适配与字段填充](#3-schema-适配与字段填充)
10. [检索 Query 模式变化](#4-检索-query-模式变化)
11. [AV 车队数据 vs Ego 数据的异同](#5-av-车队数据-vs-ego-数据的异同)
12. [实施优先级矩阵](#6-实施优先级矩阵)

---

## 1. 总体迁移路线

```
当前 (EPIC-KITCHENS 验证)                     目标 (Ego-Centric Fleet)
┌──────────────────────────┐                ┌──────────────────────────────┐
│  短视频 (1-2 min)        │    ──────▶     │  长视频 (数小时)              │
│  固定间隔抽帧 (0.5fps)   │    ──────▶     │  自适应抽帧 (activity-based) │
│  人工标注 narration      │    ──────▶     │  Qwen-VL 自动标注 + 少样本人工│
│  语义检索 (single-hop)   │    ──────▶     │  时序-语义检索 (multi-hop)     │
│  评测: gt_narration 匹配 │    ──────▶     │  评测: 事件级 recall + 时间精度│
└──────────────────────────┘                └──────────────────────────────┘

核心不变组件:
  ✓ 多模态 embedding (tongyi-embedding-vision-plus → 1152-dim 独立向量)
  ✓ Qwen-VL 结构化 caption (JSON: objects/actions/lighting/occlusion/is_anomaly)
  ✓ DashVector 向量库 + 标量字段过滤
  ✓ Python + DashScope SDK 技术栈

需新增组件:
  + 活动检测器 (activity detector, 控制抽帧密度)
  + 时序索引 (time-range queries, 事件片段检索)
  + Proprioception 对齐层 (IMU/odometry → timestamp mapping)
  + 少样本人工标注管线 (active learning loop)
```

---

## 2. 抽帧策略适配

### 2.1 从固定间隔到自适应

| 维度 | EPIC 阶段 (当前) | Ego-Centric 阶段 (目标) |
|:---|:---|:---|
| 视频时长 | 1-2 分钟 | 数十分钟到数小时 |
| 帧率 | 0.5 fps 固定 | 0.2–2.0 fps 自适应 |
| 场景变化 | 中（厨房操作） | 高（走动 + 头部转动 + 操作） |
| 冗余度 | 低 | 极高（长时间静态场景） |
| 实现 | OSS 截帧 / OpenCV | FC + 活动检测 + 关键帧提取 |

### 2.2 推荐策略：三层自适应采样

```
Layer 1: 活动检测 (Motion Activity Detector)
  ├─ 输入: 连续帧差 + 光流幅度
  ├─ 输出: activity_score ∈ [0, 1]
  └─ 阈值: score > 0.1 → "活动段",  else → "静止段"

Layer 2: 动态抽帧密度
  ├─ 活动段: 1.0–2.0 fps (密集采样，捕捉操作细节)
  ├─ 静止段: 0.1–0.2 fps (稀疏采样，保留场景上下文)
  └─ 过渡段: 0.5 fps (标准采样)

Layer 3: 手-物交互检测 (Hand-Object Interaction, Phase 5d)
  ├─ 输入: Qwen-VL caption 中检测到 hand + object 共现
  ├─ 触发: 提升该段抽帧密度至 3.0+ fps
  └─ 目的: 为后续细粒度 failure case 分析保留足够帧
```

**实现方案**：

```python
# 伪代码：自适应抽帧控制器
def adaptive_sampling(video_path, activity_threshold=0.1):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if prev_frame is not None:
            # 简单活动检测: 帧间差值
            diff = cv2.absdiff(frame, prev_frame)
            activity = np.mean(diff) / 255.0

            if activity > activity_threshold:
                fps = 2.0       # 活动段
            else:
                fps = 0.2       # 静止段

            interval = max(1, int(src_fps / fps))
            if frame_idx % interval == 0:
                yield frame, activity, frame_idx

        prev_frame = frame
```

**Rationale**：固定间隔对长视频不经济 — 99% 的帧是冗余的。活动检测用简单的帧差法（无需 GPU），保证实时性；密度分三层，在精度和成本之间做 trade-off。

### 2.3 成本估算对比

| 策略 | 帧数 / 1h 视频 | Embedding 成本 | 适用场景 |
|:---|:---|:---|:---|
| 固定 0.5fps | 1,800 | ~¥1 | 验证 |
| 固定 2.0fps | 7,200 | ~¥4 | 不考虑 |
| 自适应 0.2–2.0fps | ~1,500–3,000 | ~¥1–2 | **生产推荐** |
| 自适应 + 手物检测 | ~2,000–4,000 | ~¥2–3 | 失败案例挖掘 |

---

## 3. Schema 适配与字段填充

### 3.1 当前 Schema 回顾

我们的 schema 已预留在 6 个 ego-centric 字段（创建于 Phase 1，当前为空或占位值）：

```python
"episode_id": str,           # 当前: participant_id (P01) / 空
"action_label": str,         # 当前: gt_verb (EPIC 标注) / ""
"proprioception_ts": float,  # 当前: 0.0
"view_type": str,            # 当前: "ego" (固定)
"objects": List[str],        # 当前: Qwen-VL 生成
"actions": List[str],        # 当前: Qwen-VL 生成
```

### 3.2 Ego-Centric 字段填充方案

| 字段 | EPIC 阶段 | Ego-Centric 阶段 | 填充方式 |
|:---|:---|:---|:---|
| `episode_id` | `P01` (participant) | `mission_2026_001` 或 AVIN | 采集时写入：唯一标识每次采集任务 |
| `action_label` | `take`, `open`, ... (EPIC gt_verb) | `grasping_cup`, `opening_door` | **Qwen-VL 生成**：扩展 prompt 增加动作分类，同时人工标注少量样本做校准 |
| `proprioception_ts` | `0.0` (无数据) | `1234567890.123` | 采集时写入：cam_ts ↔ imu_ts ↔ odom_ts 对齐表 |
| `view_type` | `"ego"` (固定) | `"ego"` | 固定 |
| `objects` | Qwen-VL 生成 | Qwen-VL 生成 | 无变化 |
| `actions` | Qwen-VL 生成 | Qwen-VL 生成 | prompt 优化：增加 ego-centric action vocabulary |

### 3.3 建议新增字段

基于 ego-centric 场景的实际需求，建议增加：

```python
# --- 新增: Ego-Centric 专属字段 ---
"activity_score": float,      # 活动检测分数 [0,1]，用于检索时过滤高活动帧
"camera_pose_x": float,       # 相机/头部位置 X (需 proprioception 数据)
"camera_pose_y": float,       # 相机/头部位置 Y
"hand_present": bool,         # Qwen-VL 检测到手部存在
"interaction_intensity": str, # "none" | "touch" | "grasp" | "manipulate" (Qwen-VL 生成)
"gaze_target": str,           # 注视目标物体 (Qwen-VL 生成，如 "cup", "door_handle")
```

**Rationale**：
- `activity_score` 和 `hand_present` 用于快速过滤：只检索"人在操作"的帧，排除大量静止/行走帧
- `camera_pose_*` 支持空间查询："在厨房水槽附近的所有帧"
- `interaction_intensity` 对失败案例挖掘关键：区分"看着物体"和"正在操作物体"

### 3.4 Proprioception 对齐设计

```
采集数据流:
  camera.mp4 ──┬── frame_idx, camera_ts
               │          │
  imu.csv ─────┤          ├── alignment_table {camera_ts → imu_ts → odom_ts}
               │          │
  odom.csv ────┘          │
                          ▼
                    ingestion 时合并:
                    proprioception_ts = alignment_table.lookup(camera_ts)
                    camera_pose_*     = odom_data[camera_ts].pose

DashVector 检索时:
  "找我在厨房操作时靠近水槽的所有帧"
  → spatial filter: camera_pose near (x0, y0) radius 2m
  → temporal filter: 2026-06-18 10:00–10:30
  → activity filter: activity_score > 0.3
```

---

## 4. 检索 Query 模式变化

### 4.1 查询类型演进

| 查询类型 | 示例 (EPIC 阶段) | 示例 (Ego-Centric 阶段) |
|:---|:---|:---|
| **语义单跳** | "take cup from cupboard" | "拿杯子" |
| **语义 + 时间** | 不支持 | "厨房操作前 30 秒的帧" |
| **语义 + 空间** | 不支持 | "在水槽附近的操作帧" |
| **时序因果** | 不支持 | "拿杯子 → 杯掉落地面的时间范围内的帧" |
| **失败案例** | "is_anomaly=true" | "物体掉落 + 操作中断 + 重复拾取" |

### 4.2 检索架构升级

```
当前 (Milestone 4):
  text query → embedding → DashVector ANN → top-k 帧

Ego-Centric (Phase 6):
  text query → embedding → DashVector ANN
     │                           │
     ├─ temporal filter ─────────┤ (timestamp range)
     ├─ spatial filter ──────────┤ (camera_pose proximity)
     ├─ activity filter ─────────┤ (activity_score threshold)
     └─ interaction filter ──────┘ (hand_present, interaction_intensity)
              │
              ▼
         top-k 帧
              │
         temporal grouping (合并相邻帧为事件片段)
              │
              ▼
         event segments: [(start_ts, end_ts, summary)]
```

### 4.3 失败案例挖掘的 Query 模式

```
Pattern 1: "异常检测"
  query: "is_anomaly = true"
  → 检索 Qwen-VL 标注为异常的帧
  → 按 episode 聚合 → 识别异常场景

Pattern 2: "失败事件链"
  step 1: 检索 "物体掉落" 帧 → 找到 drop_time
  step 2: 检索 [drop_time-30s, drop_time-1s] 的高活动帧 → 找到操作前兆
  step 3: 检索 [drop_time+1s, drop_time+10s] 的帧 → 找到恢复行为

Pattern 3: "重复操作" (retry loop indicator)
  step 1: 检索所有 "take X" 帧 → 按时间排序
  step 2: 检测同一物体在短时间内被重复 take → flag 为候选失败案例

Pattern 4: "空间-时间热点"
  query: camera_pose near [X,Y] AND activity_score > 0.5
  → 找到高失败率的空间区域 → 机器人/人因工程设计反馈
```

---

## 5. AV 车队数据 vs Ego 数据的异同

### 5.1 对比总览

| 维度 | EPIC-KITCHENS (当前验证) | AV 车队数据 (目标) | 差异影响 |
|:---|:---|:---|:---|
| **视角** | 第一人称（头戴） | 第一人称（头戴/胸戴/车内） | 低 — schema 已预留 `view_type="ego"` |
| **场景** | 单一厨房 | 多样化（仓库/产线/道路/家庭） | 中 — 需扩展 Qwen-VL prompt 覆盖更多 domain |
| **视频时长** | 1-2 分钟 | 数小时（continuous recording） | **高** — 自适应抽帧是关键 |
| **标注** | 全量人工标注（verb/noun/narration） | 极少/无人工标注 | **高** — 必须依赖 Qwen-VL 自动标注 |
| **动作多样性** | 97 verbs × 300 nouns (受控) | 开放词汇 (unbounded) | 中 — 需扩展 caption prompt 的 actions vocabulary |
| **环境光** | 室内可控光 | 室内/室外/夜间 | 中 — `lighting` 字段已有，但 Qwen-VL 在极端光照下准确度未知 |
| **遮挡** | 中（手遮挡常见） | 中-高（车体/货架/人群） | 中 — `occlusion` 字段已有 |
| **运动模式** | 头部运动为主 | 行走 + 头部 + 车辆/机器人 | 高 — 需 activity_score 区分 |
| **相机参数** | 固定 1080p/60fps | 可变分辨率/帧率 | 中 — 抽帧模块需适配动态源帧率 |
| **Proprioception** | 无 | IMU + Odometry + 控制指令 | **新增** — 需设计对齐层 |
| **Post-hoc 分析** | 可 (已标注) | 必须 — 用于失败根因分析 | 架构要求支持时序回溯 |

### 5.2 EPIC → Fleet 迁移清单

```
不变的部分:
  ✅ embedding pipeline (base64 → DashScope → 1152-dim vector)
  ✅ Qwen-VL structured caption (prompt 微调后复用)
  ✅ DashVector schema (核心字段不变，只新增 ego 字段)
  ✅ DashVector query + filter paradigm
  ✅ Python + DashScope SDK 技术栈

需改/新增的部分:
  🔧 自适应抽帧控制器 (activity detection + dynamic fps)
  🔧 Qwen-VL prompt: 增加 ego-centric action vocabulary、domain 场景描述词
  🔧 Proprioception 对齐层 (timestamp mapping table)
  🔧 时序检索接口 (time range + event grouping)
  🔧 少样本人工标注管线 (active learning: 模型标注 → 人工复核 → 反馈)
  🔧 失败案例自动标识 (anomaly scoring heuristic: repetition + duration + object-loss)
```

### 5.3 关键技术风险

| 风险 | 影响 | 缓解 |
|:---|:---|:---|
| Qwen-VL 在极端光照/遮挡下 caption 准确度低 | 检索召回率下降 | 多帧融合 (相邻帧的 caption 投票)、增加 uncertainty 字段 |
| 长视频 embedding + caption 成本过高 | 月成本爆炸 | 自适应抽帧 + 分级存储 (活动段全量、静止段采样) |
| Proprioception 数据时间戳不精确 | 空间查询失败 | 时间戳同步误差容忍 ±50ms、使用 NTP 校准 |
| 开放词汇动作识别难 | 检索 queries 找不到对应帧 | 建立 ego-centric action taxonomy、结合 CLIP 等开放词汇模型辅助 |

---

## 6. 实施优先级矩阵

| 优先级 | 模块 | 理由 | 依赖 |
|:---|:---|:---|:---|
| **P0** | 自适应抽帧 (activity detection) | 不解决就无法规模化 | 无 |
| **P0** | Qwen-VL prompt 扩展 (ego domain) | 提升自动标注准确率 | 少量 ego 样本数据 |
| **P1** | 时序检索接口 | 核心需求: "什么时候发生的" | P0 完成 |
| **P1** | Proprioception 对齐层 | 空间查询的基础 | 采集端提供 IMU/odom 数据 |
| **P1** | 失败案例自动标识 | 核心业务价值 | P0 + 少量 labeled 失败案例 |
| **P2** | 空间查询 (camera_pose filter) | 导航/路径分析 | P1 proprioception |
| **P2** | 少样本人工标注管线 | 持续提升模型准确率 | P0 完成 |
| **P3** | 事件链分析 (causal multi-hop) | 高级分析能力 | P1 时序检索完成 |

---

## 附录: 与当前代码库的对应关系

| 模块 | 当前实现 | Ego 需改的文件 |
|:---|:---|:---|
| 抽帧 | `pipeline/frame_extractor.py` | 新增 `AdaptiveFrameExtractor` 类 |
| Schema | `schema.py` (FIELDS_SCHEMA) | 新增 6 个 ego 字段 |
| Caption | `pipeline/captions.py` | 扩展 `CAPTION_PROMPT` |
| 检索 | `pipeline/retriever.py` | 新增 `search_temporal()` 方法 |
| Filter | `pipeline/filter_builder.py` | 新增 activity/camera_pose 过滤条件 |
| 入 | `scripts/ingest_epic.py` | 新增 episod_id 生成 + proprioception 对齐 |
