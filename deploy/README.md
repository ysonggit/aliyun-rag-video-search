# RAG 视频检索系统 — 阿里云 Serverless 部署指南

## 架构

```
                    ┌─────────────────────────────────────┐
                    │          Alibaba Cloud (Singapore)     │
                    └─────────────────────────────────────┘

   Upload          OSS                    FC Functions              API Gateway
  ┌──────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
  │Video │────▶│ raw-videos/   │────▶│ ingest-pipeline   │     │ GET /search  │
  │.MP4  │     │ (Bucket)      │     │ (OSS trigger)     │     │   q=...      │
  └──────┘     │              │     │ extract+caption    │     └──────┬───────┘
               │ frames/      │     │ +embed+upsert      │            │
               │ (extracted)  │     └────────┬─────────┘            ▼
               └──────────────┘              │              ┌──────────────┐
                                             ▼              │ query-service │
                                      ┌──────────────┐     │ (HTTP trigger) │
                                      │  DashVector   │◀────│ embed→search   │
                                      │  (已有集群)    │     │ →top-k results │
                                      └──────────────┘     └──────────────┘
                                             │
                                      ┌──────────────┐
                                      │  DashScope    │
                                      │  (API, 已有)  │
                                      └──────────────┘
```

## 前置条件

1. 阿里云账号（新加坡 region）
2. 已开通服务：OSS、函数计算 FC、API 网关、RAM
3. 已有 DashVector 集群（当前正在用的）
4. 已有 DashScope API Key（当前正在用的）
5. 安装 [Serverless Devs](https://github.com/Serverless-Devs/Serverless-Devs)

```bash
npm install -g @serverless-devs/s
s config add --AccountID <你的AccountID> \
  --AccessKeyID <你的AK> \
  --AccessKeySecret <你的SK>
```

## 快速部署

```bash
cd deploy
cp .env.cloud.example .env.cloud
# 编辑 .env.cloud 填入你的密钥和端点
./deploy.sh
```

## 资源清单

| 资源 | 类型 | 用途 |
|:---|:---|:---|
| `rag-videos-{uid}` | OSS Bucket | 视频存储 + 帧存储 |
| `ingest-pipeline` | FC Function (Python 3.11) | OSS 触发 → 抽帧 → embed → caption → 入 |
| `query-service` | FC Function (Python 3.11) | HTTP → 文本 query → 检索 → 返回结果 |
| `rag-search-api` | API Gateway | 路由 /search → query-service |
| `fc-oss-role` | RAM Role | FC 访问 OSS/DashVector 的权限 |

## 手动部署步骤

### Step 1: 创建 OSS Bucket

```bash
# 使用阿里云 CLI
aliyun oss mb oss://rag-videos-{你的后缀} \
  --region ap-southeast-1 \
  --acl private
```

### Step 2: 部署 FC 函数

```bash
# 部署查询服务
s deploy --template query-service.yaml

# 部署摄入管线
s deploy --template ingest-pipeline.yaml
```

### Step 3: 配置 OSS 触发器

在 FC 控制台为 `ingest-pipeline` 函数添加 OSS 触发器：
- 事件类型: `oss:ObjectCreated:*`
- Bucket: `rag-videos-{后缀}`
- 前缀: `raw-videos/`
- 后缀: `.mp4`

### Step 4: 测试

```bash
# 上传测试视频
aliyun oss cp test.mp4 oss://rag-videos-{后缀}/raw-videos/test.mp4

# 等 1-2 分钟 (处理时间)
# 查询
curl "https://{api-gateway-url}/search?q=take+cup&top_k=5"
```
