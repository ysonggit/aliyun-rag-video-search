# RAG 视频检索系统 — 改进工作总结

> 2026-06-17 ~ 2026-06-20 | 从本地原型到云端生产，从 164 帧到 2805 帧，从串行 75min 到并发 20min

---

## 一、里程碑回顾

| 日期 | 里程碑 | 交付物 |
|:---|:---|:---|
| 06-17 | M1 环境 + 连通性 | DashScope SDK + DashVector + 连通性测试 |
| 06-17 | M2 抽帧模块 | OpenCV 抽帧 + EPIC-KITCHENS 标注交叉引用 |
| 06-17 | M3 Embedding + 入库 | 逐帧 embedding + Qwen-VL caption + DashVector upsert |
| 06-17 | M4 检索 | Hybrid 检索（向量 + scalar filter + 数组后过滤）|
| 06-17 | M5 评测 | 20 query 评测：Recall@5=0.80, MRR=0.73 |
| 06-17 | M6 设计文档 | Ego-centric 映射 + 模型对比 + 生产架构 |
| 06-18 | Streamlit UI | 本地搜索界面 |
| 06-18 | 云端部署 | FC ingest + FC query + OSS + ACR (新加坡) |
| 06-18 | 静态 HTML 前端 | web/index.html 纯前端搜索界面 |
| 06-18 | 架构图 | SVG 技术架构图，嵌入 HTML 文档 |
| 06-18 | 训练白皮书 | Embodied-AI 4 层训练场景 + Qwen3 选型 |
| 06-19 | 并发优化 | Claude 优化：embed 4w + caption 8w + oss 8w 并发 |
| 06-19 | Signed URL | OSS 私有 bucket signed URL 方案 |
| 06-19 | 数据扩展 | 9→20 视频, 2→8 参与者, 1762→2805 帧 |
| 06-20 | 微调计划 | Phase 1 Qwen3-VL-Embedding-2B LoRA 方案 |

---

## 二、核心改进

### 2.1 数据规模扩展

| 维度 | 初始 | 当前 | 倍数 |
|:---|:---|:---|:---|
| 视频 | 3 | 20 | 6.7× |
| 参与者 | 1 (P01) | 8 (P01-P08) | 8× |
| 帧数 | 164 | 2,805 | 17.1× |
| 有标注帧 | 121 | ~1,401 | 11.6× |
| 标注段 | 106 | 1,562 | 14.7× |
| 抽帧频率 | 0.5 fps | 1.0 fps | 2× |
| OSS 帧图片 | 0 | 2,805 | — |

**为什么扩展**：初始 3 视频 164 帧仅覆盖 1 个厨房，embedding 模型会过拟合到 P01 的场景布局。20 视频 8 参与者提供跨厨房多样性，使检索结果可泛化。

### 2.2 性能优化

| 步骤 | 串行 (初始) | 并发 (当前) | 加速 | 实现 |
|:---|:---|:---|:---|:---|
| Embedding | 4 min | 1 min | 4× | ThreadPoolExecutor(4) + batch≤8 |
| Captioning | 64 min | 18 min | 3.6× | ThreadPoolExecutor(8) |
| OSS 上传 | 3 min | 0.5 min | 6× | ThreadPoolExecutor(8) |
| Embed+Caption | 68 min (串行) | 18 min (并行) | 3.8× | 两步同时执行 |
| **总管线** | **75 min** | **~20 min** | **3.75×** | — |

**关键改进**：Claude 的优化将 embedding 和 captioning 从串行改为并行执行，同时各自内部多线程并发调 API。

### 2.3 检索质量

| 评测版本 | 帧数 | pure_vector R@5 | hybrid_video R@5 | hybrid_verb R@5 | hybrid_verb R@10 |
|:---|:---|:---|:---|:---|:---|
| v1 strict (164帧) | 164 | 0.25 | 0.30 | — | — |
| v2 soft (164帧) | 164 | 0.70 | 0.75 | 0.80 | 0.80 |
| v2 soft (1762帧) | 1762 | 0.45 | 0.60 | 0.80 | 0.90 |
| **v2 soft (2805帧)** | 2805 | **0.25** | **0.60** | **0.80** | **0.90** |

**关键发现**：
- 数据量增加后 pure_vector Recall@5 下降 (0.70→0.25) — 向量空间更密集，噪声增加
- hybrid_verb 始终最强 — verb 标签过滤有效抑制噪声
- Recall@10=0.90 稳定 — 更大候选池中能找到正确帧
- **结论：hybrid_verb 是生产推荐策略，Phase 1 微调目标推到 Recall@5=0.88+**

### 2.4 安全改进

| 问题 | 初始方案 | 改进方案 |
|:---|:---|:---|
| OSS 图片访问 | public URL (403 Forbidden) | Signed URL (1h 过期, slash_safe=True) |
| API 密钥 | 硬编码风险 | 环境变量 + .env.cloud (override=True) |
| Bucket ACL | 尝试 public-read (被拒) | 保持 private + FC 签名返回 |

### 2.5 架构演进

```
阶段 1 (本地原型):
  Mac → OpenCV → DashScope API → DashVector
  视频/帧在本地磁盘, 断网即停

阶段 2 (云端部署):
  OSS → FC ingest-pipeline → DashScope → DashVector
  FC query-service → Flask :9000 → signed URL 返回
  web/index.html → 纯前端调 FC API
  读/写完全解耦, 云端自治

阶段 3 (优化+扩展, 当前):
  20 视频 8 参与者 → 并发管线 (embed 4w + caption 8w + oss 8w)
  → 2805 帧 OSS + DashVector → signed URL 检索
  → M5 评测 Recall@10=0.90
```

---

## 三、踩坑记录 (已内化到 infra-ops agent)

| # | 问题 | 根因 | 解决 |
|:---|:---|:---|:---|
| 1 | FC exec format error | Mac ARM64 build, FC 需 x86_64 | docker buildx --platform linux/amd64 |
| 2 | FC 函数不启动 | custom-container 需要 HTTP server | Flask :9000 + CMD |
| 3 | DashScope InvalidApiKey | 新加坡端点需显式设置 | dashscope.api_key + base_http_api_url |
| 4 | DashVector contain_any 无效 | 新加坡集群不支持数组过滤 | 标量服务端 + 数组客户端 |
| 5 | OSS trigger SourceARN 缺失 | s.yaml 需指定 bucket ARN | 手动 FC 控制台添加 |
| 6 | FC 镜像不更新 | latest tag 被缓存 | 用 :v2 tag 强制拉取 |
| 7 | Ingest OSS 上传崩溃 | base64 编码后路径变成 URL | 先 embed/caption 本地文件, 再上传 OSS |
| 8 | OSS signed URL 403 | slash_safe 未启用 | bucket.sign_url(slash_safe=True) |
| 9 | Captioning 64min 太慢 | 串行调 API | ThreadPoolExecutor(8) 并发 |
| 10 | Debian libgl1-mesa-glx 缺失 | Trixie 改名 | 改用 libgl1 |

---

## 四、当前系统状态

### 云端资源 (新加坡 ap-southeast-1)

| 组件 | 状态 | 详情 |
|:---|:---|:---|
| OSS Bucket | `rag-videos-krones` (private) | 2805 帧图片, signed URL |
| DashVector | `scene_frames_2fps` | 2805 vectors, 1152-dim, 18 fields |
| FC query-service | Active (v2 镜像) | Flask :9000, 2GB, signed URL |
| FC ingest-pipeline | Active (v2 镜像) | Flask :9000, 4GB, OSS trigger |
| ACR | rag-query:v2 + rag-ingest:v2 | x86_64, 个人版 |
| DashScope | 新加坡 intl 端点 | tongyi-embedding-vision-plus + qwen-vl-max |

### 可用入口

| 入口 | 地址 |
|:---|:---|
| 云端搜索 API | `https://query-service-eftocucisq.ap-southeast-1.fcapp.run/search?q=take+cup` |
| 浏览器 UI | `open web/index.html` |
| 本地 Streamlit | `streamlit run app.py` |
| M5 评测 | `DV_COLLECTION_NAME=scene_frames_2fps python3 tests/test_evaluation.py` |

### 技术文档

| 文档 | 路径 |
|:---|:---|
| 技术设计 (含架构图) | `docs/milestone6_egocentric_design.html` |
| 训练白皮书 | `docs/embodied_ai_training.html` |
| 微调计划 | `docs/phase1_finetune_plan.md` |
| 架构图 SVG | `docs/architecture.svg` |

---

## 五、下一步

| 优先级 | 任务 | 预期 |
|:---|:---|:---|
| P0 | Phase 1 微调 Qwen3-VL-Embedding-2B LoRA | Recall@5 0.80→0.88+ |
| P1 | FC OSS trigger 配置 (全自动摄入) | 上传视频即触发处理 |
| P1 | Web UI 上云 (OSS 静态托管) | 公网可访问搜索界面 |
| P2 | Qwen3-VL-Reranker 精排 | MRR 0.71→0.85+ |
| P2 | 2fps 入库 | 帧数翻倍, Recall +5pp |
| P3 | Ego-centric 适配 (proprioception) | 支持 VLA 训练 |

---

[← 返回技术设计文档](milestone6_egocentric_design.html) | [← 返回训练白皮书](embodied_ai_training.html)
