#!/usr/bin/env python
"""
End-to-End EPIC-KITCHENS Video Ingestion Pipeline.

Usage:
  python scripts/ingest_epic.py [--fps 2.0] [--force] [--collection scene_frames_2fps]

Options:
  --fps FLOAT          Frame extraction rate (default: 0.5)
  --force              Force re-extraction even if frames exist
  --collection NAME    DashVector collection name (default: scene_frames)
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "deploy" / ".env.cloud", override=True)  # Override .env placeholders with real OSS creds

from config import dashscope_config, dashvector_config
from data.epic_loader import EpicAnnotations
from pipeline.frame_extractor import FrameExtractor
from pipeline.frame_metadata import FrameMetadata
from pipeline.embedder import Embedder
from pipeline.captions import Captioner
from pipeline.indexer import Indexer
from pipeline.oss_uploader import upload_frames


EPIC_ROOT = ROOT / "EPIC-KITCHENS"
ANNOTATIONS_DIR = os.environ.get("EPIC_ANNOTATIONS_DIR", "/tmp/epic-kitchens-100-annotations")
FRAME_OUTPUT = ROOT / "storage" / "frames"
VIDEOS = [
    "P01_02", "P01_03", "P01_04", "P01_06", "P01_07",
    "P01_08", "P01_10", "P01_16", "P02_01", "P02_02",
]


def find_video_path(video_id: str) -> Path:
    """Find video file in P01 or P02 directories."""
    participant = video_id.split("_")[0]
    path = EPIC_ROOT / participant / "videos" / f"{video_id}.MP4"
    return path


def main():
    parser = argparse.ArgumentParser(description="EPIC-KITCHENS video ingestion")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Frame extraction rate (default: 0.5)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction even if frames exist")
    parser.add_argument("--collection", type=str, default=None,
                        help="DashVector collection name (default: from .env or scene_frames)")
    args = parser.parse_args()

    collection_name = args.collection or dashvector_config.collection_name

    start_time = time.time()
    print("=" * 60)
    print(f"  EPIC-KITCHENS Ingestion: {args.fps} fps → {collection_name}")
    print("=" * 60)
    print(f"  Videos:  {VIDEOS}")
    print(f"  Extract: {args.fps} fps")
    print(f"  Embed:   {dashscope_config.embedding_model}")
    print(f"  Caption: {dashscope_config.vl_model}")
    print(f"  Store:   {collection_name}")

    # Validate
    if not EPIC_ROOT.exists():
        print(f"\n  ERROR: EPIC-KITCHENS dir not found: {EPIC_ROOT}")
        sys.exit(1)
    if not Path(ANNOTATIONS_DIR).exists():
        print(f"\n  ERROR: Annotations not found: {ANNOTATIONS_DIR}")
        sys.exit(1)
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n  ERROR: DASHSCOPE_API_KEY not set")
        sys.exit(1)

    # Load annotations
    print("\n[0] Loading annotations...")
    ann = EpicAnnotations(ANNOTATIONS_DIR)
    for vid in VIDEOS:
        print(f"    {vid}: {len(ann.get_segments(vid))} segments")

    # Step 1: Extract frames
    print(f"\n[1] Extracting frames ({args.fps} fps)...")
    all_frames = []
    extractor = FrameExtractor(fps=args.fps, output_dir=str(FRAME_OUTPUT), annotations=ann)

    for vid in VIDEOS:
        video_path = find_video_path(vid)
        if not video_path.exists():
            print(f"    WARN: {video_path} not found")
            continue

        meta_path = FRAME_OUTPUT / vid / f"{vid}_metadata.json"
        if meta_path.exists() and not args.force:
            import json
            with open(meta_path) as f:
                existing = json.load(f)
            n_existing = len(existing)
            # Estimate expected frame count for this fps
            vc = cv2.VideoCapture(str(video_path))
            total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps = vc.get(cv2.CAP_PROP_FPS)
            vc.release()
            expected = max(1, int(total_frames / src_fps * args.fps))
            if abs(n_existing - expected) <= 3:
                print(f"    {vid}: {n_existing} frames exist (~{expected} expected), skipping")
                for record in existing:
                    fm = FrameMetadata(
                        video_id=record["video_id"],
                        frame_id=record["frame_id"],
                        frame_path=record["frame_path"],
                        timestamp=record["timestamp"],
                        frame_number=record["frame_number"],
                        gt_narration=record.get("gt_narration", ""),
                        gt_verb=record.get("gt_verb", ""),
                        gt_noun=record.get("gt_noun", ""),
                    )
                    all_frames.append(fm)
                continue
            else:
                print(f"    {vid}: {n_existing} frames exist but {expected} expected at {args.fps}fps, re-extracting...")

        try:
            frames = extractor.extract(str(video_path), video_id=vid)
            all_frames.extend(frames)
        except Exception as e:
            print(f"    ERROR extracting {vid}: {e}, skipping")
            continue

    print(f"  Total: {len(all_frames)} frames")
    annotated = sum(1 for f in all_frames if f.gt_narration)
    print(f"  Annotated: {annotated} ({100*annotated/max(1,len(all_frames)):.0f}%)")

    if not all_frames:
        print("  ERROR: No frames")
        sys.exit(1)

    # Checkpoint: skip frames already indexed in a previous run (resume support).
    manifest_path = FRAME_OUTPUT.parent / "manifests" / f"{collection_name}.txt"
    done_ids = set()
    if manifest_path.exists():
        with open(manifest_path) as mf:
            done_ids = {line.strip() for line in mf if line.strip()}
    if done_ids:
        before = len(all_frames)
        all_frames = [f for f in all_frames if f.frame_id not in done_ids]
        print(f"  Resume: {len(done_ids)} frames already indexed, "
              f"{before - len(all_frames)} skipped, {len(all_frames)} remaining")
    if not all_frames:
        print(f"  Nothing to do — all frames already indexed in {collection_name}. Done.")
        return

    # Estimate cost
    est_embedding = len(all_frames) * 0.001  # rough per-frame cost in ¥
    est_caption = len(all_frames) * 0.02     # rough per-frame cost in ¥
    print(f"  Est. cost: embedding ~¥{est_embedding:.0f}, caption ~¥{est_caption:.0f}")

    # Steps 2 & 3: Embed and caption concurrently.
    # They read the same frames but write disjoint fields (embedding vs scene_desc/
    # objects/...), so the two stages overlap instead of running as sequential passes.
    print(f"\n[2+3] Embedding ({dashscope_config.embedding_model}) + "
          f"captioning ({dashscope_config.vl_model}) concurrently...")
    embedder = Embedder(
        model=dashscope_config.embedding_model,
        api_key=dashscope_config.api_key,
        max_workers=int(os.environ.get("EMBED_WORKERS", "4")),
    )
    captioner = Captioner(
        model=dashscope_config.vl_model,
        api_key=dashscope_config.api_key,
        max_workers=int(os.environ.get("CAPTION_WORKERS", "8")),
    )
    with ThreadPoolExecutor(max_workers=2) as stage_pool:
        fut_embed = stage_pool.submit(embedder.embed_frames, all_frames)
        fut_caption = stage_pool.submit(captioner.caption_frames, all_frames)
        fut_embed.result()      # re-raises embedding errors (caption errors fall back to defaults)
        fut_caption.result()

    # Top objects
    obj_counts = {}
    for f in all_frames:
        for obj in f.objects:
            obj_counts[obj] = obj_counts.get(obj, 0) + 1
    top10 = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top objects: {', '.join(f'{o}({c})' for o, c in top10)}")

    # Step 3.5: Upload frames to OSS
    print(f"\n[3.5] Uploading {len(all_frames)} frames to OSS...")
    import oss2
    oss_ak = os.environ.get("OSS_ACCESS_KEY_ID", "")
    oss_sk = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    oss_endpoint = os.environ.get("OSS_ENDPOINT", "oss-ap-southeast-1.aliyuncs.com")
    oss_bucket = os.environ.get("OSS_BUCKET_NAME", "rag-videos-krones")

    if oss_ak and oss_sk:
        auth = oss2.Auth(oss_ak, oss_sk)
        bucket = oss2.Bucket(auth, f"https://{oss_endpoint}", oss_bucket)
        upload_frames(
            all_frames,
            bucket,
            prefix="frames",
            max_workers=int(os.environ.get("UPLOAD_WORKERS", "8")),
        )
    else:
        print(f"  WARN: OSS credentials not set, keeping local paths")

    # Step 4: Ingest
    print(f"\n[4] Ingesting to DashVector ({collection_name})...")
    indexer = Indexer(
        api_key=dashvector_config.api_key,
        endpoint=dashvector_config.endpoint,
        collection_name=collection_name,
        dimension=embedder.dimension,
    )
    indexer.create_collection(if_not_exists=True)

    # Append each successfully upserted batch to the manifest so a failed/partial
    # run resumes without re-spending on the embed/caption APIs.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_lock = threading.Lock()

    def checkpoint(frame_ids):
        with manifest_lock, open(manifest_path, "a") as mf:
            mf.write("\n".join(frame_ids) + "\n")

    ingested = indexer.index(
        all_frames,
        embedding_model=dashscope_config.embedding_model,
        caption_model=dashscope_config.vl_model,
        on_batch_indexed=checkpoint,
    )

    # Write/merge facet values for the search UI (avoids a sampling query on load).
    # Merge with any existing file so resume runs accumulate the full value set.
    facets_path = FRAME_OUTPUT.parent / "facets" / f"{collection_name}.json"
    facets_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if facets_path.exists():
        try:
            existing = json.loads(facets_path.read_text())
        except Exception:
            existing = {}
    facets = {
        "videos": sorted(set(existing.get("videos", [])) | {f.video_id for f in all_frames if f.video_id}),
        "objects": sorted(set(existing.get("objects", [])) | {o for f in all_frames for o in f.objects}),
        "category": sorted(set(existing.get("category", [])) | {f.category for f in all_frames if f.category}),
    }
    facets_path.write_text(json.dumps(facets, ensure_ascii=False, indent=2))
    print(f"  Facets written: {facets_path}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  DONE: {ingested}/{len(all_frames)} frames in {elapsed:.0f}s")
    print(f"  Collection: {collection_name}")
    print(f"  Next: DV_COLLECTION_NAME={collection_name} python tests/test_evaluation.py")


if __name__ == "__main__":
    main()
