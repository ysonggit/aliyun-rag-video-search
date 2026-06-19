#!/bin/bash
# Deploy RAG Video Search to Alibaba Cloud
# Prerequisites:
#   - Alibaba Cloud CLI configured
#   - Docker installed
#   - Serverless Devs installed (npm i -g @serverless-devs/s)
#   - .env.cloud configured

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Load env ─────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env.cloud" ]; then
  echo "ERROR: .env.cloud not found. Copy .env.cloud.example and fill in values."
  exit 1
fi
source "$SCRIPT_DIR/.env.cloud"

REGION="${REGION:-ap-southeast-1}"
ACR="${ACR_REGISTRY}"

echo "=== RAG Video Search — Cloud Deploy ==="
echo "Region: $REGION"
echo "ACR:    $ACR"

# ── Step 1: Build Docker images ─────────────────────
echo ""
echo "[1/4] Building Docker images..."

docker build \
  -t rag-query:latest \
  -f "$SCRIPT_DIR/Dockerfile.query" \
  "$PROJECT_DIR"

docker build \
  -t rag-ingest:latest \
  -f "$SCRIPT_DIR/Dockerfile.ingest" \
  "$PROJECT_DIR"

docker build \
  -t rag-webui:latest \
  -f "$SCRIPT_DIR/Dockerfile.webui" \
  "$PROJECT_DIR"

echo "  Images built: rag-query, rag-ingest, rag-webui"

# ── Step 2: Push to ACR ──────────────────────────────
echo ""
echo "[2/4] Pushing to ACR..."

docker tag rag-query:latest "$ACR/rag-query:latest"
docker tag rag-ingest:latest "$ACR/rag-ingest:latest"
docker tag rag-webui:latest "$ACR/rag-webui:latest"

docker push "$ACR/rag-query:latest"
docker push "$ACR/rag-ingest:latest"
docker push "$ACR/rag-webui:latest"

echo "  Images pushed to ACR"

# ── Step 3: Create OSS Bucket ─────────────────────────
echo ""
echo "[3/4] Ensuring OSS bucket..."

python3 -c "
import oss2
auth = oss2.Auth('$OSS_ACCESS_KEY_ID', '$OSS_ACCESS_KEY_SECRET')
service = oss2.Service(auth, 'https://$OSS_ENDPOINT')
existing = [b.name for b in oss2.BucketIterator(service)]
if '$OSS_BUCKET_NAME' in existing:
    print(f'  Bucket $OSS_BUCKET_NAME already exists')
else:
    bucket = oss2.Bucket(auth, 'https://$OSS_ENDPOINT', '$OSS_BUCKET_NAME')
    bucket.create_bucket(oss2.models.BUCKET_ACL_PRIVATE)
    print(f'  Bucket $OSS_BUCKET_NAME created')
"

# ── Step 4: Deploy FC functions ──────────────────────
echo ""
echo "[4/4] Deploying FC functions..."

cd "$SCRIPT_DIR"
s deploy

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Test query:"
echo "  curl \"https://\$(s info query-service --output url)/search?q=take+cup&top_k=5\""
echo ""
echo "Open the web UI:"
echo "  open \"https://\$(s info web-ui --output url)\""
echo ""
echo "Test ingestion:"
echo "  aliyun oss cp test.mp4 oss://$BUCKET/raw-videos/test.mp4"
echo ""
