# Phase 1 模型微调计划

> 基于 [Embodied-AI 模型训练白皮书](embodied_ai_training.html) 的 Phase 1 训练路线。
> 目标：用 ego-centric 视频数据微调 Qwen3-VL-Embedding-2B，提升检索召回率。

---

## 目标

| 指标 | 当前 (baseline) | 目标 |
|:---|:---|:---|
| Recall@5 | 0.80 | 0.88+ |
| MRR | 0.73 | 0.85+ |
| 帧数 | 164 | 2400+ |

## 架构决策

采用**跨域混合方案**（不迁移新加坡 infra）：

```
Singapore (现有)                    Hangzhou (新增, 仅训练)
┌──────────────────┐               ┌──────────────────┐
│ OSS: frames/     │──── copy ────→│ OSS: train-data/  │
│ DashVector       │               │ PAI-DLC (A10)     │
│ FC: ingest/query │               │   ↓ LoRA 训练     │
│ DashScope API    │               │ Model weights     │
└──────────────────┘               └──────────────────┘
         ↑
  训练完成后评估:
  提升 > 5pp → 部署 PAI-EAS 杭州, FC 跨域调用 (+50ms)
  提升 < 5pp → 保持 DashScope API, 不改架构
```

跨域延迟分析：新加坡→杭州 ~30-50ms，对批量训练无感知，对查询 250ms→300ms 可接受。

## Step 1: 扩展数据量

### 1.1 下载 EPIC 短视频

当前 3 个视频 → 扩展到 10 个：

| 视频 | 时长 | 标注数 | 状态 |
|:---|:---|:---|:---|
| P01_03 | 2.0 min | 42 | ✅ 已有 |
| P01_04 | 1.8 min | 32 | ✅ 已有 |
| P01_08 | 1.6 min | 32 | ✅ 已有 |
| P01_07 | 2.7 min | 57 | 待下载 |
| P01_10 | 2.3 min | 26 | 待下载 |
| P01_12 | 2.9 min | ~30 | 待下载 |
| P01_13 | 1.6 min | ~20 | 待下载 |
| P01_16 | 2.9 min | 73 | 待下载 |
| P02_01 | 3.9 min | ~40 | 待下载 |
| P01_02 | 8.4 min | ~50 | 待下载 |

合计: ~29 min 视频, ~400 标注段, 下载量 ~3GB

### 1.2 提高抽帧频率

从 0.5fps → 2.0fps（覆盖率提升 4 倍）：

```
10 视频 × ~29 min × 2.0 fps ≈ 3480 帧
有标注帧: ~2400 (69%)
OSS 存储: ~3GB 视频 + ~700MB 帧 = ~3.7GB
```

### 1.3 云端入库

```bash
# 下载新增视频
python scripts/download_epic_videos.py  # 更新 SELECTED_VIDEOS 列表

# 2fps 入库 (走新加坡 DashScope + DashVector)
python scripts/ingest_epic.py --fps 2.0 --force --collection scene_frames_2fps
```

成本估算：
- DashScope embedding: 3480 帧 × ¥0.0005/千token ≈ ¥2
- DashScope Qwen-VL caption: 3480 帧 × ~¥0.01/帧 ≈ ¥35
- OSS 存储: 3.7GB × ¥0.12/GB/月 ≈ ¥0.45/月
- **总计: ~¥37 一次性 + ¥0.45/月**

## Step 2: 导出训练数据

从 DashVector 导出 (frame_id, gt_narration, oss_url)，构建对比学习三元组：

```python
# 三元组格式
{
  "query": "take cup from cupboard",          # EPIC gt_narration
  "positive": "oss://.../P01_03_f001547.jpg", # 匹配帧
  "negatives": [                               # 硬负样本
    "oss://.../P01_03_f003000.jpg",            # 同视频不同动作
    "oss://.../P01_04_f006069.jpg",            # 不同视频
    "oss://.../P01_08_f001200.jpg",            # 不同视频
    "oss://.../P01_16_f000500.jpg"             # 不同视频
  ]
}

# 预估: 2400 有标注帧 × 5 条/帧 = ~12000 三元组
```

硬负样本策略：
- 同视频不同 verb 的帧（最难区分）
- 不同视频相同 verb 的帧（跨场景干扰）
- 随机采样帧（easy negative）

## Step 3: 上传到杭州 + PAI 训练

### 3.1 数据上传

```bash
# 导出三元组 JSON → 杭州 OSS
aliyun oss cp train_data.json oss://krones-train-hz/ego-embed/

# 同时上传帧图片（或直接用新加坡 OSS 公网 URL）
# 如果跨域读图片太慢, copy 帧图片到杭州 OSS
```

### 3.2 PAI-DLC 训练任务

```yaml
# PAI-DLC 训练配置
模型: Qwen3-VL-Embedding-2B
微调: LoRA (rank=8, alpha=16)
GPU: 1× A10 (24GB) — 杭州
训练数据: ~12000 三元组
Batch size: 32
Epochs: 3-5
Learning rate: 1e-4 (LoRA)
预计时间: 1-2 小时
```

### 3.3 训练数据格式（PAI 兼容）

```jsonl
{"query": "take cup", "positive_image": "/data/frames/P01_03_f001547.jpg", "negative_image": "/data/frames/P01_04_f006069.jpg"}
{"query": "open door", "positive_image": "/data/frames/P01_03_f000120.jpg", "negative_image": "/data/frames/P01_08_f001200.jpg"}
...
```

## Step 4: 评测与决策

```bash
# 用 fine-tuned 模型重新 embed 3480 帧
# 写入新 collection: scene_frames_finetuned

# 跑 M5 评测脚本
python tests/test_evaluation.py  # 对比 baseline vs fine-tuned
```

决策矩阵：

| Recall@5 提升 | 行动 |
|:---|:---|
| > 8pp | 部署 PAI-EAS 杭州, FC 跨域调用, 值得额外 50ms |
| 3-8pp | 部署 PAI-EAS, 或考虑增加训练数据后重训 |
| < 3pp | 保持 DashScope API, 转向其他优化方向（reranker / 2fps / prompt 优化） |

## Step 5（条件触发）: 部署推理

仅当 Step 4 评测结果满意时执行：

```
杭州 PAI-EAS:
  模型: fine-tuned Qwen3-VL-Embedding-2B
  端点: https://xxx.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/ego_embed
  
新加坡 FC ingest-pipeline 修改:
  embedder.py: 从 DashScope API 改为调 PAI-EAS HTTP API
  新增: PAI_EAS_ENDPOINT 环境变量
  
新加坡 FC query-service 修改:
  retriever.py: _embed_text() 改为调 PAI-EAS
  
延迟影响: +50ms (跨域), 总查询 250ms→300ms
```

## 时间线

| 阶段 | 内容 | 耗时 | 依赖 |
|:---|:---|:---|:---|
| 1.1 下载视频 | 7 个 EPIC 视频 | 20 min | 网络 |
| 1.2 入库 | 2fps ingest 3480 帧 | 60-90 min | DashScope API |
| 2 导出训练数据 | 构建三元组 JSON | 15 min | Step 1 完成 |
| 3.1 上传杭州 | 训练数据 + 帧图片 | 10 min | 杭州 OSS |
| 3.2 PAI 训练 | LoRA 微调 | 1-2 hours | A10 GPU |
| 4 评测 | 重新 embed + M5 | 30 min | Step 3 完成 |
| 5 部署（可选） | PAI-EAS + FC 修改 | 1 hour | Step 4 通过 |

**总计: 3-5 小时**（含等待时间）

## 风险

| 风险 | 概率 | 影响 | 缓解 |
|:---|:---|:---|:---|
| 2400 帧仍不够 | 中 | 提升 < 3pp | 增加 EPIC 视频到 20 个 |
| 跨域图片传输慢 | 低 | 训练数据上传慢 | 预 copy 帧图片到杭州 OSS |
| PAI-DLC 配置复杂 | 中 | 花时间调试 | 用 PAI-DSW 交互式调参 |
| LoRA 不收敛 | 低 | 浪费 GPU 时间 | 先小数据验证, 再全量训练 |
| DashScope 国内 key 不同于 intl | 中 | 杭州 PAI 无法调 DashScope | 用 PAI 本地推理, 不依赖 DashScope |

---

[← 返回训练白皮书](embodied_ai_training.html) | [← 返回技术设计文档](milestone6_egocentric_design.html)
