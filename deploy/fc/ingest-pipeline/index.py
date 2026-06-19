"""
FC Ingest Pipeline — Flask HTTP server for Alibaba Cloud FC custom-container.

Receives OSS event via HTTP POST on port 9000.
Extracts frames → embeds → captions → upserts to DashVector.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import oss2
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set DashScope to Singapore/intl endpoint
import dashscope
dashscope.base_http_api_url = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

from pipeline.embedder import Embedder
from pipeline.captions import Captioner
from pipeline.indexer import Indexer
from pipeline.frame_metadata import FrameMetadata
from pipeline.oss_uploader import upload_frames

app = Flask(__name__)

EXTRACT_FPS = float(os.environ.get("EXTRACT_FPS", "0.5"))
JPEG_QUALITY = int(os.environ.get("FRAME_QUALITY", "85"))

OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "oss-ap-southeast-1.aliyuncs.com")
OSS_BUCKET = os.environ.get("OSS_BUCKET_NAME", "")
OSS_AK = os.environ.get("OSS_ACCESS_KEY_ID", "")
OSS_SK = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
FRAME_PREFIX = "frames"


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/invoke", methods=["POST"])
def invoke():
    """FC sends event payload as POST body."""
    event = request.get_json(force=True, silent=True) or {}
    return process_event(event)


@app.route("/", methods=["POST"])
def invoke_root():
    """FC may also POST to root."""
    event = request.get_json(force=True, silent=True) or {}
    return process_event(event)


def process_event(event):
    """Process OSS trigger event."""
    app.logger.info(f"Event: {json.dumps(event, default=str)[:500]}")

    events = event.get("events", [])
    if not events:
        # Maybe direct invocation
        events = [{"oss": {"bucket": {"name": event.get("bucket", OSS_BUCKET)},
                           "object": {"key": event.get("object", event.get("key", ""))}}}]

    results = []
    for evt in events:
        oss_info = evt.get("oss", {})
        bucket_name = oss_info.get("bucket", {}).get("name", OSS_BUCKET)
        object_key = oss_info.get("object", {}).get("key", "")

        if not object_key:
            continue
        if not (object_key.endswith(".MP4") or object_key.endswith(".mp4")):
            app.logger.info(f"Skipping non-video: {object_key}")
            continue

        video_id = object_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        app.logger.info(f"Processing: {object_key} → {video_id}")

        try:
            result = process_video(bucket_name, object_key, video_id)
            results.append({"video_id": video_id, "status": "ok", **result})
        except Exception as e:
            app.logger.error(f"FAILED: {video_id}: {e}")
            results.append({"video_id": video_id, "status": "error", "error": str(e)})

    return jsonify({"results": results})


def process_video(bucket_name, object_key, video_id):
    """Full pipeline: download → extract → embed → caption → upload OSS → ingest."""
    auth = oss2.Auth(OSS_AK, OSS_SK)
    bucket = oss2.Bucket(auth, f"https://{OSS_ENDPOINT}", bucket_name)

    # 1. Download video
    local_video = f"/tmp/{video_id}.mp4"
    app.logger.info(f"  Downloading {object_key} → {local_video}")
    bucket.get_object_to_file(object_key, local_video)

    # 2. Extract frames (local paths in /tmp)
    frames = extract_frames(local_video, video_id)
    app.logger.info(f"  Extracted {len(frames)} frames")

    # 3 & 4. Embed and caption concurrently (disjoint fields, same frames)
    embedder = Embedder(
        model=os.environ.get("EMBEDDING_MODEL", "tongyi-embedding-vision-plus"),
        api_key=os.environ["DASHSCOPE_API_KEY"],
        max_workers=int(os.environ.get("EMBED_WORKERS", "4")),
    )
    captioner = Captioner(
        model=os.environ.get("VL_MODEL", "qwen-vl-max"),
        api_key=os.environ["DASHSCOPE_API_KEY"],
        max_workers=int(os.environ.get("CAPTION_WORKERS", "8")),
    )
    with ThreadPoolExecutor(max_workers=2) as stage_pool:
        fut_embed = stage_pool.submit(embedder.embed_frames, frames)
        fut_caption = stage_pool.submit(captioner.caption_frames, frames)
        fut_embed.result()
        fut_caption.result()

    # 5. Upload frames to OSS (after embedding/captioning, so local files are no longer needed)
    upload_frames(
        frames,
        bucket,
        prefix=FRAME_PREFIX,
        max_workers=int(os.environ.get("UPLOAD_WORKERS", "8")),
    )

    # 6. Ingest to DashVector (with OSS URLs as frame_path)
    indexer = Indexer(
        api_key=os.environ["DASHVECTOR_API_KEY"],
        endpoint=os.environ["DASHVECTOR_ENDPOINT"],
        collection_name=os.environ.get("DV_COLLECTION_NAME", "scene_frames"),
        dimension=embedder.dimension,
    )
    indexer.create_collection(if_not_exists=True)
    ingested = indexer.index(
        frames,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "tongyi-embedding-vision-plus"),
        caption_model=os.environ.get("VL_MODEL", "qwen-vl-max"),
    )

    # Cleanup local files
    os.remove(local_video)
    for f in frames:
        local_path = f"/tmp/{f.frame_id}.jpg"
        if os.path.exists(local_path):
            os.remove(local_path)

    return {"frames": len(frames), "ingested": ingested}


def extract_frames(video_path, video_id):
    """Extract frames using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(src_fps / EXTRACT_FPS))
    frames = []
    frame_idx = 0

    while True:
        ret, image = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / src_fps
            frame_id = f"{video_id}_f{frame_idx:06d}"
            frame_path = f"/tmp/{frame_id}.jpg"
            cv2.imwrite(frame_path, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frames.append(FrameMetadata(
                video_id=video_id, frame_id=frame_id,
                frame_path=frame_path, timestamp=timestamp, frame_number=frame_idx,
            ))
        frame_idx += 1

    cap.release()
    return frames


if __name__ == "__main__":
    port = int(os.environ.get("CAPort", 9000))
    app.run(host="0.0.0.0", port=port)
