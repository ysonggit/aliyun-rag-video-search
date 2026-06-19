"""
FC Query Service — HTTP server for Alibaba Cloud FC custom-container.

Listens on port 9000 (FC default). Handles GET /search requests.
"""

import json
import os
import sys
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set DashScope to Singapore/intl endpoint before any DashScope calls
import dashscope
dashscope.base_http_api_url = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

from pipeline.retriever import Retriever

app = Flask(__name__)
_retriever = None

# ── Security config ────────────────────────────────────────
# INTERNAL_TOKEN: shared secret that the API Gateway injects when forwarding.
# Defense-in-depth: even if someone discovers the FC HTTP trigger URL, they
# can't call /search without this header. Leave empty to disable (dev only).
_INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "").strip()
# Sanity caps on user input — prevent pathological queries from burning quota.
_MAX_QUERY_LEN = 500
_MAX_FILTER_VALUES = 20


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            embedding_model=os.environ.get("EMBEDDING_MODEL", "tongyi-embedding-vision-plus"),
            dashvector_api_key=os.environ["DASHVECTOR_API_KEY"],
            dashvector_endpoint=os.environ["DASHVECTOR_ENDPOINT"],
            collection_name=os.environ.get("DV_COLLECTION_NAME", "scene_frames"),
        )
    return _retriever


@app.before_request
def _check_internal_token():
    """Require X-Internal-Token header on all routes except the health check.

    The API Gateway injects this header; it blocks direct hits to the FC URL
    if someone discovers it. Health check (/) stays open so FC can probe it.
    """
    if not _INTERNAL_TOKEN:
        return  # dev mode: no gate
    if request.path == "/" or request.path == "/health":
        return
    sent = request.headers.get("X-Internal-Token", "").strip()
    if not sent or sent != _INTERNAL_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401


@app.route("/", methods=["GET"])
def health():
    """Health check endpoint for FC."""
    return jsonify({"status": "ok"}), 200


@app.route("/health", methods=["GET"])
def health_alias():
    return health()


@app.route("/search", methods=["GET"])
def search():
    """Search endpoint: GET /search?q=take+cup&top_k=10&video_id=P01_03"""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400
    if len(query) > _MAX_QUERY_LEN:
        return jsonify({"error": f"Query too long (max {_MAX_QUERY_LEN} chars)"}), 400

    try:
        top_k = int(request.args.get("top_k", 10))
    except ValueError:
        return jsonify({"error": "Invalid 'top_k'"}), 400
    top_k = min(max(top_k, 1), 50)

    video_id = request.args.get("video_id")
    if video_id is not None:
        video_id = video_id.strip()[:64] or None

    lighting = request.args.get("lighting")
    if lighting is not None:
        lighting = lighting.strip()[:32] or None

    objects_str = request.args.get("objects", "")
    objects = [o.strip()[:64] for o in objects_str.split(",") if o.strip()] or None
    if objects and len(objects) > _MAX_FILTER_VALUES:
        return jsonify({"error": f"Too many object filters (max {_MAX_FILTER_VALUES})"}), 400

    try:
        retriever = get_retriever()
        results = retriever.search(
            query, top_k=top_k,
            video_id=video_id, lighting=lighting, objects=objects,
        )
        return jsonify({
            "query": query,
            "total": len(results),
            "results": [
                {
                    "frame_id": r.frame_id,
                    "score": round(r.score, 4),
                    "video_id": r.video_id,
                    "timestamp": r.timestamp,
                    "frame_path": r.frame_path,
                    "objects": r.objects,
                    "actions": r.actions,
                    "lighting": r.lighting,
                    "occlusion": r.occlusion,
                    "gt_narration": r.gt_narration,
                }
                for r in results
            ],
        })
    except Exception:
        # Log full traceback server-side; never leak internals to the client.
        app.logger.exception("search failed")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("CAPort", 9000))
    app.run(host="0.0.0.0", port=port)
