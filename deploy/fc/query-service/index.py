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


@app.route("/", methods=["GET"])
def health():
    """Health check endpoint for FC."""
    return jsonify({"status": "ok"}), 200


@app.route("/search", methods=["GET"])
def search():
    """Search endpoint: GET /search?q=take+cup&top_k=10&video_id=P01_03"""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    top_k = min(max(int(request.args.get("top_k", 10)), 1), 50)

    video_id = request.args.get("video_id")
    lighting = request.args.get("lighting")
    objects_str = request.args.get("objects", "")
    objects = [o.strip() for o in objects_str.split(",") if o.strip()] or None

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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("CAPort", 9000))
    app.run(host="0.0.0.0", port=port)
