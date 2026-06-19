"""
Parallel OSS frame upload.

Uploads extracted frame images to OSS concurrently and rewrites each
FrameMetadata.frame_path to its public OSS URL in place.

Qwen-VL/embedding aside, serial upload of thousands of frames is a major
ingestion bottleneck; each put_object_from_file is an independent HTTP call,
so a thread pool gives near-linear speedup (bounded by OSS/network).

Usage:
    bucket = oss2.Bucket(auth, f"https://{endpoint}", bucket_name)
    upload_frames(frames, bucket, prefix="frames")
    # each frame.frame_path is now "https://{bucket}.{endpoint}/{prefix}/{video_id}/{frame_id}.jpg"
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from pipeline.frame_metadata import FrameMetadata


def upload_frames(
    frames: List[FrameMetadata],
    bucket,
    prefix: str = "frames",
    max_workers: int = 8,
) -> List[FrameMetadata]:
    """
    Upload frame images to OSS concurrently, mutating frame_path in place.

    Args:
        frames: FrameMetadata objects with local frame_path set.
        bucket: An oss2.Bucket instance (caller-constructed). put_object_from_file
                issues an independent HTTP request per call, so it is safe to share
                the bucket across threads.
        prefix: OSS key prefix; objects are stored at "{prefix}/{video_id}/{frame_id}.jpg".
        max_workers: Concurrent uploads. Lower if you hit OSS request limits.

    Returns:
        The same list, with each frame.frame_path replaced by its public OSS URL.
    """
    total = len(frames)
    # bucket.endpoint looks like "https://oss-ap-southeast-1.aliyuncs.com"
    host = bucket.endpoint.split("://", 1)[-1].rstrip("/")
    print(f"\n  Uploading {total} frames to OSS (bucket={bucket.bucket_name}, workers={max_workers})")
    progress_file = "/tmp/upload_progress.txt"

    lock = threading.Lock()
    counters = {"done": 0, "ok": 0, "fail": 0}

    def worker(f: FrameMetadata):
        oss_key = f"{prefix}/{f.video_id}/{f.frame_id}.jpg"
        ok = True
        try:
            bucket.put_object_from_file(oss_key, f.frame_path)
            f.frame_path = f"https://{bucket.bucket_name}.{host}/{oss_key}"
        except Exception as e:
            ok = False
            print(f"  WARN: upload failed for {f.frame_id}: {e}")
        with lock:
            counters["done"] += 1
            counters["ok" if ok else "fail"] += 1
            done = counters["done"]
            if done % 100 == 0 or done == total:
                msg = f"Uploaded {done}/{total} (ok={counters['ok']}, fail={counters['fail']})"
                print(f"  {msg}")
                with open(progress_file, "w") as pf:
                    pf.write(f"{msg}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(worker, frames))

    print(f"  Done: {counters['ok']} uploaded, {counters['fail']} failed")
    return frames
