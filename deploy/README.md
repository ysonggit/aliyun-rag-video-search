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

---

## Step 5: 部署 Web UI (Streamlit)

`web-ui` 函数已包含在 `s.yaml` 中,`./deploy.sh` 会自动构建并推送 `rag-webui` 镜像。部署后:

```bash
# 获取 web-ui 的 HTTP 触发器 URL
s info web-ui --output url
# → https://1234567890.ap-southeast-1.fcapp.run
```

**本地测试** (不部署到云,先验证密码门和检索):

```bash
# 设置环境变量后本地运行
export WEBUI_PASSWORD="test-only-password"
streamlit run app.py --server.port=8501
```

若 `WEBUI_PASSWORD` 为空 → 无密码门 (本地开发模式)。生产环境必须设置。

---

## Step 6: 生产安全加固 (Tier 2)

Web UI 和 query-service 默认通过 FC HTTP 触发器暴露 (`*.fcapp.run`)。虽然代码层已有
密码门 + `INTERNAL_TOKEN` 头校验,但生产环境应在前端再加 API Gateway + WAF + 自定义域名 TLS。

### 6.1 生成密钥

```bash
# 三个独立的强随机字符串,写入 .env.cloud (永远不要提交)
openssl rand -hex 32  # → WEBUI_PASSWORD
openssl rand -hex 32  # → INTERNAL_TOKEN   (API Gateway 注入的 X-Internal-Token)
openssl rand -hex 32  # → API_GATEWAY_KEY  (客户端调用 API 时用的 X-API-Key)
```

把这三个值填入 `deploy/.env.cloud`,然后重新部署让 FC 函数读取新环境变量:

```bash
./deploy/deploy.sh
```

### 6.2 创建 API Gateway (保护 query-service 的程序化 API)

API Gateway 做:API Key 鉴权 + 限流配额 + 注入 `X-Internal-Token` 头。

```bash
# 1. 创建 API 分组
GROUP_ID=$(aliyun apigateway CreateApiGroup \
  --GroupName rag-search \
  --Description "RAG video search API" \
  --RegionId ap-southeast-1 \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['GroupId'])")
echo "GroupId: $GROUP_ID"

# 2. 获取 query-service 的 FC HTTP 触发器 URL
QUERY_URL=$(s info query-service --output url)
QUERY_HOST=$(echo "$QUERY_URL" | sed 's|https://||; s|/.*||')

# 3. 定义 /search API (后端 = query-service FC)
API_ID=$(aliyun apigateway CreateApi \
  --GroupId "$GROUP_ID" \
  --ApiName search \
  --Visibility PRIVATE \
  --RequestConfig '{"RequestHttpMethod":"GET","RequestProtocol":"HTTPS","RequestPath":"/search","RequestMode":"MAPPING"}' \
  --ServiceConfig "{\"ServiceType\":\"HTTP\",\"ServiceHttpMethod\":\"GET\",\"ServiceAddress\":\"https://$QUERY_HOST\",\"ServicePath\":\"/search\",\"ServiceTimeout\":30000}" \
  --RequestParameters '[]' \
  --ServiceParameters '[{"ServiceParameterName":"q","Location":"QUERY","ParameterCatalog":"REQUEST"},{"ServiceParameterName":"top_k","Location":"QUERY","ParameterCatalog":"REQUEST"}]' \
  --RegionId ap-southeast-1 \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['ApiId'])")

# 4. 部署 API 到 RELEASE 环境
aliyun apigateway DeployApi --ApiId "$API_ID" --StageName RELEASE --RegionId ap-southeast-1
```

### 6.3 配置 API Key 鉴权 + 限流

```bash
# 1. 创建 App (持有 API Key)
APP_ID=$(aliyun apigateway CreateApp \
  --AppName rag-search-client \
  --Description "Programmatic /search client" \
  --RegionId ap-southeast-1 \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['AppId'])")

# 2. 创建使用计划 (限流: 每秒 5 次, 每天 1000 次)
PLAN_ID=$(aliyun apigateway CreateUsagePlan \
  --UsagePlanName rag-search-quota \
  --ApiSchemes '[{"ApiId":"'$API_ID'","StageName":"RELEASE"}]' \
  --RateLimit 5 \
  --Quota 1000 \
  --RegionId ap-southeast-1 \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['UsagePlanId'])")

# 3. 绑定 App 到使用计划
aliyun apigateway SetAppsForUsagePlan --UsagePlanId "$PLAN_ID" --AppIds "[\"$APP_ID\"]" --RegionId ap-southeast-1

# 4. 授权 App 访问该 API
aliyun apigateway AuthorizeApp --AppId "$APP_ID" --ApiId "$API_ID" --StageName RELEASE --RegionId ap-southeast-1

# 5. 获取该 App 的 API Key (AppCode),客户端调用时用 X-API-Key 头发送
aliyun apigateway DescribeApp --AppId "$APP_ID" --RegionId ap-southeast-1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('AppCode (X-API-Key):', d.get('AppCode','(run DescribeApps to find)'))"
```

调用程序化 API (带 API Key):

```bash
curl -H "X-API-Key: $API_GATEWAY_KEY" \
  "https://{api-group-subdomain}.ap-southeast-1.alicontent.com/search?q=take+cup&top_k=5"
```

### 6.4 注入 X-Internal-Token 头 (defense-in-depth)

让 API Gateway 转发时自动附加 `X-Internal-Token` 头,这样直接访问 FC URL 的请求
会被 query-service 拒绝 (代码层校验,见 `index.py` 的 `_check_internal_token`)。

在 API Gateway 控制台:
1. 进入该 API 的「后端配置」
2. 添加「系统参数」→ 自定义请求头 `X-Internal-Token` = 你在 `.env.cloud` 里设置的 `INTERNAL_TOKEN` 值
3. 重新发布到 RELEASE 环境

> 注意: Alibaba Cloud API Gateway 的「常量参数 / 透传头」配置在控制台 UI 里更直观,
> CLI 对此支持有限。建议用控制台完成这一步,记录到内部文档。

### 6.5 自定义域名 + 托管 TLS (Web UI)

Streamlit 用 WebSocket,所以前面放 **DCDN** (动态加速,原生支持 WS) 而不是普通 CDN。

```bash
# 前提: 你的域名已解析到 Alibaba Cloud,且在 DNS 里加一条 CNAME
#   search.your-domain.com → <dcdn-cname>.aliddns.com

# 1. 创建 DCDN 加速域名
DOMAIN="search.your-domain.com"
aliyun dcdn OpenDcdnService 2>/dev/null  # 若未开通
aliyun dcdn AddDcdnDomain \
  --DomainName "$DOMAIN" \
  --ResourceType dynamic \
  --Sources '[{"content":"<web-ui-fc-trigger-host>","type":"ipaddr","port":443,"priority":"20"}]' \
  --Scope domestic

# 2. 申请托管证书 (或上传自有证书)
#    控制台: DCDN → 域名管理 → HTTPS 配置 → 申请免费证书
```

### 6.6 WAF 规则 (SQLi / XSS / 扫描器拦截)

```bash
# 1. 开通 WAF (若未开通)
aliyun waf-openapi ModifyInstanceInfo 2>/dev/null

# 2. 把域名接入 WAF (代理模式: DNS 指向 WAF CNAME)
aliyun waf-openapi CreateDomain \
  --Domain "$DOMAIN" \
  --InstanceType shareware \
  --Region ap-southeast-1

# 3. 启用内置防护规则集 (默认开启 SQLi/XSS/CC)
#    控制台: WAF → 防护配置 → Web 应用防护 → 启用「规则防护引擎」+「主动防御」
```

完成后的流量路径:

```
浏览器
   │  https://search.your-domain.com
   ▼
DCDN (TLS 终结 + WebSocket 透传 + 全球加速)
   ▼
WAF (SQLi/XSS/扫描器拦截 + IP 限流)
   ▼
FC web-ui (Streamlit,密码门校验 WEBUI_PASSWORD)
   │  直接 import pipeline.retriever (不经 HTTP)
   ▼
DashScope + DashVector (新加坡)

程序化客户端
   │  https://{api-group}.ap-southeast-1.alicontent.com/search
   │  Header: X-API-Key: ...
   ▼
API Gateway (API Key 鉴权 + 限流配额 + 注入 X-Internal-Token)
   ▼
FC query-service (校验 X-Internal-Token 头 → 检索 → 返回)
```

### 6.7 验证加固是否生效

```bash
# 1. 直连 FC URL 应被拒 (缺 token)
curl -i "https://<query-service-fc-url>/search?q=test"
# 期望: HTTP 401 Unauthorized

# 2. 带 token 直连应通过
curl -i -H "X-Internal-Token: $INTERNAL_TOKEN" "https://<query-service-fc-url>/search?q=test&top_k=3"
# 期望: HTTP 200 + JSON 结果

# 3. Web UI 密码门
open "https://<web-ui-fc-url>"
# 期望: 看到 "Sign in required" 表单,输入正确密码后才进主界面

# 4. 超长 query 被拒
python3 -c "print('q=' + 'x'*600)" | xargs -I{} curl -i "https://<query-service-fc-url>/search?{}"
# 期望: HTTP 400 "Query too long"
```

### 安全清单 (部署前过一遍)

- [ ] `.env.cloud` 中 `WEBUI_PASSWORD` / `INTERNAL_TOKEN` / `API_GATEWAY_KEY` 均为 `openssl rand -hex 32` 生成的强随机值
- [ ] `deploy/.env.cloud` 在 `.gitignore` 中 (已在,见根目录 `.gitignore` line 3)
- [ ] FC `query-service` 的 `INTERNAL_TOKEN` 环境变量已设置 (空值 = 关闭防护)
- [ ] FC `web-ui` 的 `WEBUI_PASSWORD` 环境变量已设置 (空值 = 无密码门)
- [ ] API Gateway 已绑定使用计划 (限流) + App 授权
- [ ] WAF 已接入域名 + 启用规则引擎
- [ ] 自定义域名 TLS 证书有效 (DCDN 免费证书或自有证书)
- [ ] `query-service` 异常不再返回内部错误 (`{"error": "Internal server error"}`)

