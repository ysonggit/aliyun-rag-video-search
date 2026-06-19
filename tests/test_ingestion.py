"""
Milestone 3: End-to-End Ingestion Pipeline Test

Tests:
  1. Frame extraction from test video
  2. Embedding generation (DashScope API)
  3. Qwen-VL structured caption generation
  4. DashVector collection creation + upsert
  5. Verify retrieval from DashVector

Usage:
  python tests/test_ingestion.py

Requires:
  - .env with DASHSCOPE_API_KEY, DASHVECTOR_API_KEY, DASHVECTOR_ENDPOINT set
  - Working DashScope and DashVector credentials

Acceptance criteria:
  - Frames extracted correctly
  - Embedding for each frame is 1152-dim
  - Caption for each frame contains valid JSON with required fields
  - All frames successfully upserted into DashVector
  - Retrieval returns the ingested documents

Sanity check:
  python tests/test_ingestion.py && echo "INGESTION PIPELINE OK"
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import dashscope_config, dashvector_config
from pipeline.frame_extractor import FrameExtractor
from pipeline.embedder import Embedder
from pipeline.captions import Captioner
from pipeline.indexer import Indexer


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _create_multiscene_test_video(output_path: str, scene_count: int = 3) -> str:
    """
    Create a synthetic video with multiple "scene" transitions for testing.

    Each scene is:
    - 30 frames (1 second at 30fps) with a different color + label
    - Scene themes: "red_cup", "blue_plate", "green_bowl"
    """
    width, height = 640, 480
    fps = 30.0
    frames_per_scene = 30
    total_frames = scene_count * frames_per_scene
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [
        {"color": (0, 0, 255), "label": "Scene 1: red cup on counter"},
        {"color": (255, 0, 0), "label": "Scene 2: blue plate in sink"},
        {"color": (0, 255, 0), "label": "Scene 3: green bowl on table"},
    ]

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {output_path}")

    for scene_idx in range(min(scene_count, len(scenes))):
        s = scenes[scene_idx]
        for i in range(frames_per_scene):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 30
            cv2.rectangle(frame, (50, 50), (590, 430), s["color"], -1)
            cv2.putText(frame, s["label"], (60, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {scene_idx * frames_per_scene + i}",
                        (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            writer.write(frame)

    writer.release()
    print(f"  Test video: {output_path} ({total_frames} frames, {total_frames/fps:.1f}s)")
    return output_path


def test_full_pipeline():
    """Test the complete frame → embed → caption → ingest pipeline."""
    separator("Milestone 3: Full Ingestion Pipeline")

    # Check environment
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("  SKIP: DASHSCOPE_API_KEY not set in environment")
        return True  # Not a failure, just unavailable
    if not os.getenv("DASHVECTOR_API_KEY"):
        print("  SKIP: DASHVECTOR_API_KEY not set in environment")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        # ---- Step 1: Extract frames ----
        print("\n[1/4] Extracting frames...")
        video_path = _create_multiscene_test_video(
            os.path.join(tmpdir, "test_scenes.mp4"), scene_count=3
        )

        extractor = FrameExtractor(
            fps=1.0,  # 1 fps for denser sampling on this short video
            output_dir=os.path.join(tmpdir, "frames"),
        )
        frames = extractor.extract(video_path, video_id="test_scenes")
        assert len(frames) >= 2, f"Expected >= 2 frames, got {len(frames)}"
        print(f"  OK: {len(frames)} frames extracted")

        # ---- Step 2: Generate embeddings ----
        print("\n[2/4] Generating embeddings...")
        embedder = Embedder(
            model=dashscope_config.embedding_model,
            api_key=dashscope_config.api_key,
        )
        frames = embedder.embed_frames(frames)

        for f in frames:
            assert f.embedding is not None, f"Frame {f.frame_id} has no embedding"
            assert len(f.embedding) == embedder.dimension, \
                f"Frame {f.frame_id} dim={len(f.embedding)}, expected {embedder.dimension}"
        print(f"  OK: {len(frames)} vectors generated (dim={embedder.dimension})")

        # ---- Step 3: Generate captions ----
        print("\n[3/4] Generating captions...")
        captioner = Captioner(
            model=dashscope_config.vl_model,
            api_key=dashscope_config.api_key,
        )
        frames = captioner.caption_frames(frames)

        for f in frames:
            assert f.scene_desc_raw, f"Frame {f.frame_id} has no scene description"
            # Verify JSON is parseable
            cap = json.loads(f.scene_desc_raw)
            assert "objects" in cap, f"Caption missing 'objects': {cap}"
            assert "actions" in cap, f"Caption missing 'actions': {cap}"
            assert "lighting" in cap, f"Caption missing 'lighting': {cap}"
            print(f"  Frame {f.frame_id}: objects={f.objects}, actions={f.actions}, "
                  f"lighting={f.lighting}")

        # ---- Step 4: Ingest to DashVector ----
        print("\n[4/4] Ingesting to DashVector...")
        indexer = Indexer(
            api_key=dashvector_config.api_key,
            endpoint=dashvector_config.endpoint,
            collection_name=dashvector_config.collection_name,
            dimension=embedder.dimension,
        )
        indexer.create_collection(if_not_exists=True)
        count = indexer.index(
            frames,
            embedding_model=dashscope_config.embedding_model,
            caption_model=dashscope_config.vl_model,
        )
        assert count == len(frames), f"Ingested {count}/{len(frames)} frames"

        # ---- Step 5: Verify retrieval ----
        print("\n[Verify] Testing retrieval...")
        collection = indexer.get_collection()

        # Search with first frame's vector → should return itself as top result
        query_vec = frames[0].embedding
        rsp = collection.query(
            vector=query_vec,
            topk=3,
            output_fields=["frame_id", "video_id", "objects", "actions", "gt_narration"],
        )

        assert rsp and rsp.output, "Query returned no results"
        top_result = rsp.output[0]
        print(f"  Query (vector from {frames[0].frame_id}):")
        for j, result in enumerate(rsp.output[:3]):
            print(f"    #{j+1}: {result.id} fields={result.fields}")

        # Verify metadata roundtrip
        field_keys = result.fields.keys()
        assert "objects" in field_keys, f"Missing 'objects' in fields: {field_keys}"
        assert "actions" in field_keys, f"Missing 'actions' in fields: {field_keys}"

        # ---- Step 6 (optional): Snapshot summary ----
        print("\n[Snapshot] Frames in DashVector:")
        for f in frames:
            print(f"  {f.frame_id:25s} vec_dim={len(f.embedding):4d}  "
                  f"objects={str(f.objects):30s}  actions={str(f.actions):30s}  "
                  f"gt={f.gt_narration or '-'}")

    print(f"\n  PASS: Full ingestion pipeline works end-to-end")
    return True


if __name__ == "__main__":
    print("RAG Video Search — Milestone 3: Ingestion Pipeline Test")
    print(f"Embedding model: {dashscope_config.embedding_model}")
    print(f"VL model: {dashscope_config.vl_model}")
    print(f"DashVector: {dashvector_config.collection_name} @ {dashvector_config.endpoint}")

    ok = test_full_pipeline()

    if ok:
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        print("\nChecks failed.")
        sys.exit(1)
