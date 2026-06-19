# Embodied-AI 模型训练白皮书

> 本文档基于 [RAG 视频检索系统技术设计文档](milestone6_egocentric_design.html) 中构建的数据管道，分析其对具身智能模型训练的支撑能力。
> 数据管道产出物：per-frame 视觉 embedding (1152-dim) + 结构化场景描述 (JSON) + EPIC ground truth narration + 时序元数据 + 异常标记 + 混合检索能力。

---

## 目录

1. [数据管道产出物概览](#1-数据管道产出物概览)
2. [第一层：直接可训练模型](#2-第一层直接可训练模型)
3. [第二层：检索增强具身策略](#3-第二层检索增强具身策略)
4. [第三层：世界模型与前瞻规划](#4-第三层世界模型与前瞻规划)
5. [第四层：数据飞轮与持续学习](#5-第四层数据飞轮与持续学习)
6. [Qwen3 模型选型矩阵](#6-qwen3-模型选型矩阵)
7. [关键差距与补齐路径](#7-关键差距与补齐路径)
8. [推荐训练路线](#8-推荐训练路线)

---

## 1. 数据管道产出物概览

```
Pipeline 产出物 (per frame):
  ┌──────────────────────────────────────────────────────┐
  │ 视觉:   frame_image (OSS JPG) + 1152-dim embedding   │
  │ 语义:   objects[], actions[], lighting, occlusion     │
  │ 语言:   gt_narration ("take cup") + scene_desc JSON  │
  │ 时序:   timestamp, video_id, episode_id              │
  │ 质量:   is_anomaly (bool)                            │
  │ 检索:   vector ANN + scalar filter + keyword filter  │
  │ 预留:   proprioception_ts, camera_pose, action_label │
  └──────────────────────────────────────────────────────┘
```

核心洞察：这不只是一个"搜索引擎"，而是一个**结构化的经验记忆系统**。它把非结构化的视频流转化为可检索、可训练、可推理的数据资产。

---

## 2. 第一层：直接可训练模型

### 2.1 Affordance 预测模型

输入: ego-centric 帧 → 输出: 每个物体的"可操作性"热力图

```
训练数据:
  frame_image ←→ objects[] + actions[]
  
  "cup" 出现时 → action 通常是 "take" / "wash" / "put-down"
  "door" 出现时 → action 通常是 "open" / "close"
  
  → 模型学习: 看到杯子把手 → 高 affordance "grasp"
              看到门把手   → 高 affordance "pull"
```

为什么我们的数据够用：每帧都有 `objects` + `actions` 的共现关系，统计 164 帧就能学到 "物体→可行动作" 的映射。扩大到数千帧后，可以做像素级 affordance grounding。

### 2.2 动作预期模型 (Action Anticipation)

```
输入: 帧序列 [t-3s, t-2s, t-1s, t]
输出: 预测 t+1 时刻的动作

训练数据:
  (P01_03 frames at t=0,2,4,6...) → action at t=8
  
  例: 看到"手伸向cupboard" → 预测"open cupboard"
      看到"拿起了cup" → 预测"pour water" 或 "put down cup"

评估: 用 EPIC gt_narration 做时间对齐的 ground truth
```

我们的优势：`timestamp` 精确到帧级别，`gt_narration` 提供动作标签，可以构建 `(过去N帧, 下一帧动作)` 的监督信号。

### 2.3 场景图生成模型 (Scene Graph)

```
输入: ego-centric 帧
输出: 场景图 {节点=物体, 边=空间/交互关系}

训练数据推导:
  从 objects[] 共现统计:
    "cup" + "table" → ON(cup, table)
    "hand" + "cup" → HOLDING(hand, cup)
    "fridge" + "door" → HAS_PART(fridge, door)
  
  从 actions[] 推导交互:
    "take" + "cup" → INTERACT(hand, cup, take)
```

### 2.4 异常检测 / 安全分类器

```
输入: ego-centric 帧序列
输出: is_anomaly (bool) + 异常类型

训练数据:
  is_anomaly=true 的帧 (Qwen-VL 标注) → 正样本
  is_anomaly=false 的帧 → 负样本

进阶:
  检索 "重复 take" → retry loop (失败模式)
  检索 "物体掉落" → drop event (安全事故)
  检索 "操作中断" → hesitation (不确定状态)
```

---

## 3. 第二层：检索增强具身策略

### 3.1 检索增强策略网络 (RAG for Robotics)

这是最核心的应用 — 我们的 pipeline 本质上是机器人的**情景记忆**：

```
┌─────────────────────────────────────────────────────────────┐
│              机器人决策循环 (with episodic memory)             │
│                                                             │
│  当前观察 (camera frame)                                     │
│      │                                                      │
│      ├──→ Encode → 1152-dim embedding                       │
│      │                                                      │
│      ├──→ DashVector 检索: "上次遇到类似场景是什么?"           │
│      │    → 返回: 相似帧 + 当时的动作 + 结果(成功/失败)        │
│      │                                                      │
│      ├──→ Qwen3-VL 推理: "当前场景 + 历史经验 → 下一步?"      │
│      │                                                      │
│      └──→ Policy: action = f(current_frame, retrieved_exp)  │
│                                                             │
│  执行动作 → 新帧 → 重新入库 → 记忆更新                        │
└─────────────────────────────────────────────────────────────┘
```

与传统 RL 的区别：
- 传统 RL: 策略网络从零学习，靠试错积累
- RAG-RL: 策略网络可以"回忆"过去相似场景的经验，避免重复犯错

我们的 pipeline 独特价值：`is_anomaly` 字段让检索可以区分"成功经验"和"失败经验" — 机器人不仅能回忆"我上次怎么做的"，还能知道"上次这样做失败了"。

### 3.2 模仿学习的示范检索

```
任务: "把杯子从橱柜拿到桌子上"

Step 1: 检索 "take cup from cupboard"
  → 返回 20 帧相关历史 (ego-centric 视角)
  → 按时间排序 → 重构操作序列

Step 2: 检索 "put down cup on table"  
  → 返回 15 帧
  → 重构放置序列

Step 3: 拼接 → 完整 demonstration trajectory
  → 喂给 Behavior Cloning 策略网络

Step 4: 策略网络 + 当前观察 → 执行动作
```

### 3.3 失败恢复策略 (Failure Recovery)

```
检测到异常 (is_anomaly=true)
  │
  ├─ 检索时序上下文: 异常前 30s 的帧
  │   → "什么操作导致了失败?"
  │
  ├─ 检索相似失败: contain_any(actions, ["retry"])
  │   → "上次类似失败后是怎么恢复的?"
  │
  └─ 生成恢复策略:
      → 如果 "cup dropped" → 重新拾取
      → 如果 "door stuck" → 换方向拉
      → 如果 "object misplaced" → 重新定位
```

---

## 4. 第三层：世界模型与前瞻规划

### 4.1 状态转移世界模型

```
输入: (frame_t, action_t)
输出: 预测 frame_{t+1} (或其 embedding)

训练数据:
  从时序帧对构建:
    (P01_03_f000120, "open door") → P01_03_f000180
    (P01_03_f000300, "take cup")  → P01_03_f000360

模型学习:
  "open door" 后 → 门状态变化 (场景外观变化)
  "take cup" 后  → 杯子从桌上消失 (手部出现杯子)
  "pour water" 后 → 杯中出现液体

应用:
  → Model Predictive Control: 在脑中"模拟"多个候选动作
  → 选最优动作执行
  → 不需要物理仿真器 (learned world model)
```

我们的数据优势：embedding 空间天然度量场景相似度，可以在 embedding 空间做状态转移预测（不需要生成像素级图像）。

---

## 5. 第四层：数据飞轮与持续学习

```
                    ┌─────────────────┐
                    │  机器人部署       │
                    │  (持续采集视频)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Ingest Pipeline │
                    │  (抽帧+embed+    │
                    │   caption+入库)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  DashVector      │
                    │  (经验记忆库)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼──────┐  ┌───▼──────────┐
     │ 检索质量    │  │ 稀疏区域    │  │ 低置信度帧    │
     │ 评估        │  │ 检测        │  │ 识别          │
     │ (M5 评测)  │  │ (向量空间)  │  │ (caption 不   │
     └────────┬───┘  └──────┬──────┘  │  确定帧)      │
              │              │         └───┬──────────┘
              └──────────────┼─────────────┘
                             │
                    ┌────────▼────────┐
                    │  人工标注/修正    │
                    │  (active learning)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  模型微调        │
                    │  (Qwen-VL/Embed) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  更新模型 →      │
                    │  重新入库 →      │
                    │  机器人升级      │
                    └─────────────────┘
```

---

## 6. Qwen3 模型选型矩阵

| 训练场景 | 推荐模型 | 参数 | GPU (LoRA) | 训练数据来源 | 预期效果 |
|:---|:---|:---|:---|:---|:---|
| Embedding 质量提升 | Qwen3-VL-Embedding-2B | 2B | 1× A10 | (frame, narration) contrastive | Recall@5: 0.80→0.88 |
| Caption 准确率 | Qwen3-VL-Flash + LoRA | ~7B | 1× A10 | (frame, scene_desc_JSON) SFT | caption 准确率+15% |
| 检索重排序 | Qwen3-VL-Reranker-2B | 2B | 1× T4 | (query, frame, score) | MRR: 0.73→0.85 |
| 动作规划推理 | Qwen3-4B-Thinking | 4B | 1× A10 | (scene, retrieved_exp, action) | 因果推理链 |
| 机器人代码生成 | Qwen3-Coder-7B | 7B | 1× A10 | (instruction+frame, control_code) | ROS2 控制代码 |
| 全模态理解 | Qwen3-Omni | — | 多卡 | (video+audio, description) | 连续视频理解 |

---

## 7. 关键差距与补齐路径

| 训练目标 | 我们有 | 我们缺 | 补齐方式 |
|:---|:---|:---|:---|
| Affordance 模型 | objects, actions | 物体 bounding box | Qwen-VL 3D 定位能力（已有） |
| 动作预期 | timestamp, actions | 长序列标注 | 扩大视频入库量 |
| 场景图 | objects 共现 | 关系标注 (ON/IN/NEAR) | Qwen-VL prompt 扩展 |
| VLA 模型 | frame + language | 机器人动作轨迹 | 接入 ROS2 proprioception（schema 已预留） |
| 世界模型 | 帧序列 | 连续动作信号 | 补充 joint state 数据 |
| RL 策略 | 检索增强 | reward signal | 人工反馈 + 任务成功率 |

最关键的差距：VLA 模型需要 `proprioception` 数据（关节角度、末端执行器位姿）。我们的 schema 已预留了 `proprioception_ts` 和 `camera_pose` 字段 — 一旦采集端接入 IMU + 关节编码器，pipeline 可以直接扩展为 VLA 训练数据平台。

---

## 8. 推荐训练路线

```
Phase 1 (立即, 单卡):
  Qwen3-VL-Embedding-2B + LoRA
  数据: EPIC (frame, narration) contrastive
  GPU: 1× A10 (24GB)
  预期: Recall@5 从 0.80 → 0.88+

Phase 2 (短期, 单卡):
  Qwen3-VL-Flash + LoRA
  数据: (frame, scene_desc_JSON) SFT
  GPU: 1× A10 (24GB)
  预期: caption 准确率提升 → 检索质量提升

Phase 3 (中期, 双卡):
  Qwen3-VL-Reranker-2B + LoRA
  数据: (query, frame, relevance_score)
  GPU: 1× T4 (16GB)
  预期: top-10 精度提升, MRR 从 0.73 → 0.85+

Phase 4 (长期, 多卡):
  Qwen3-4B-Thinking + LoRA
  数据: (scene, retrieved_exp, action_plan)
  GPU: 1× A10
  预期: 动作规划 + 失败案例推理
```

---

## 总结

> 这个 pipeline 的本质是 **embodied-AI 的结构化经验记忆**。它把非结构化的视频流转化为 `(视觉, 语义, 时序, 质量)` 四维结构化数据，使得机器人能够**记住经验、检索相似场景、预测动作后果、识别失败模式、持续迭代进化**。短期可训练 affordance / 动作预期 / 异常检测模型；中期支持检索增强策略和模仿学习；长期接入 proprioception 后可训练完整的 VLA 世界模型。

---

[← 返回技术设计文档](milestone6_egocentric_design.html) | [← 返回项目首页](../)
