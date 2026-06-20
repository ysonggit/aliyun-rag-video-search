"""
FC Query Service — HTTP server for Alibaba Cloud FC custom-container.

Returns signed OSS URLs so browser can load private bucket images.
"""

import json
import os
import sys
from urllib.parse import urlparse
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set DashScope to Singapore/intl endpoint
import dashscope
dashscope.base_http_api_url = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

from pipeline.retriever import Retriever

# OSS for signed URLs
import oss2

app = Flask(__name__)
_retriever = None
_oss_bucket = None


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


def get_oss_bucket():
    global _oss_bucket
    if _oss_bucket is None:
        ak = os.environ.get("OSS_ACCESS_KEY_ID", "")
        sk = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
        endpoint = os.environ.get("OSS_ENDPOINT", "oss-ap-southeast-1.aliyuncs.com")
        bucket_name = os.environ.get("OSS_BUCKET_NAME", "rag-videos-krones")
        if ak and sk:
            auth = oss2.Auth(ak, sk)
            _oss_bucket = oss2.Bucket(auth, f"https://{endpoint}", bucket_name)
    return _oss_bucket


def sign_oss_url(public_url: str, expires: int = 3600) -> str:
    """Convert a public OSS URL to a signed URL for private bucket access."""
    bucket = get_oss_bucket()
    if not bucket or "aliyuncs.com" not in public_url:
        return public_url

    # Extract key from URL: https://bucket.endpoint/key/path.jpg
    parsed = urlparse(public_url)
    key = parsed.path.lstrip("/")

    # Generate signed URL (slash_safe=True is critical for keys with /)
    return bucket.sign_url("GET", key, expires, slash_safe=True)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/search", methods=["GET"])
def search():
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
                    "frame_path": sign_oss_url(r.frame_path),
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
