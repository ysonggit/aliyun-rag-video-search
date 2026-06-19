"""
Multi-modal embedding via DashScope API.

Uses tongyi-embedding-vision-plus (1152-dim, independent vectors only).
Supports batching (max 8 images per API call) and base64 encoding for local files.

Usage:
    embedder = Embedder()
    vectors = embedder.embed_frames(frames)  # frames: List[FrameMetadata]
    # Each FrameMetadata.embedding is now populated with a 1152-dim list
"""

import base64
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import List

import dashscope

from pipeline.frame_metadata import FrameMetadata


class Embedder:
    """
    Batch embedder for frame images using DashScope Multimodal Embedding API.

    Config:
        EMBEDDING_MODEL (env): default "tongyi-embedding-vision-plus"
        DASHSCOPE_API_KEY (env): DashScope API key
    """

    def __init__(
        self,
        model: str = "tongyi-embedding-vision-plus",
        api_key: str = "",
        max_batch_size: int = 8,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_workers: int = 4,
    ):
        """
        Args:
            model: Embedding model name.
            api_key: DashScope API key (falls back to DASHSCOPE env var).
            max_batch_size: Max images per API call (tongyi-*-vision-*: 8).
            max_retries: Retry count on transient errors.
            retry_delay: Initial delay between retries (exponential backoff).
            max_workers: Concurrent batch API calls. Batches are independent, so
                         throughput scales with concurrency. Lower if you hit
                         DashScope rate limits.
        """
        self.model = model
        self.api_key = api_key or dashscope.api_key
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers

    def embed_frames(self, frames: List[FrameMetadata]) -> List[FrameMetadata]:
        """
        Generate embeddings for a list of frames. Populates frame.embedding in-place.

        Args:
            frames: List of FrameMetadata with frame_path set.

        Returns:
            Same list with embedding field populated on each frame.

        Raises:
            RuntimeError: If embedding API returns error after all retries.
        """
        total = len(frames)
        print(f"\n  Embedding {total} frames (model={self.model}, workers={self.max_workers})")
        progress_file = "/tmp/embed_progress.txt"

        # Split into independent batches (≤ max_batch_size images each)
        batches = [
            frames[s : s + self.max_batch_size]
            for s in range(0, total, self.max_batch_size)
        ]
        total_batches = len(batches)

        lock = threading.Lock()
        state = {"cost": 0.0, "done": 0}

        def worker(item):
            batch_num, batch = item
            usage_tokens, batch_cost = self._embed_batch(batch, batch_num, total_batches)
            with lock:
                state["cost"] += batch_cost
                state["done"] += 1
                msg = (f"Batch {state['done']}/{total_batches}: "
                       f"{len(batch)} frames, {usage_tokens} tokens")
                print(f"  {msg}")
                with open(progress_file, "w") as pf:
                    pf.write(f"{msg} (total cost ~¥{state['cost']:.4f})\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            list(pool.map(worker, enumerate(batches, start=1)))

        print(f"  Done. Estimated cost: ¥{state['cost']:.4f}")
        return frames

    def _embed_batch(self, batch, batch_num, total_batches):
        """
        Embed one batch (≤ max_batch_size images), assigning f.embedding in place.

        Returns:
            (usage_tokens, batch_cost) tuple.

        Raises:
            RuntimeError: if the API fails after all retries.
        """
        inputs = []
        for f in batch:
            b64 = self._encode_image(f.frame_path)
            inputs.append({"image": f"data:image/jpeg;base64,{b64}"})

        for attempt in range(self.max_retries):
            resp = dashscope.MultiModalEmbedding.call(
                api_key=self.api_key,
                model=self.model,
                input=inputs,
            )

            if resp.status_code == HTTPStatus.OK:
                embeddings = resp.output["embeddings"]
                usage_tokens = resp.usage.get("total_tokens", 0)
                batch_cost = usage_tokens * 0.0005 / 1000  # ¥0.0005/k token
                for i, f in enumerate(batch):
                    f.embedding = embeddings[i]["embedding"]
                return usage_tokens, batch_cost

            if attempt < self.max_retries - 1:
                wait = self.retry_delay * (2 ** attempt)
                print(f"  Batch {batch_num}: retry {attempt+1}/{self.max_retries} "
                      f"({resp.code}: {resp.message}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Embedding API failed after {self.max_retries} retries: "
                    f"{resp.code} - {resp.message}"
                )

    @staticmethod
    def _encode_image(path: str) -> str:
        """Read image file and return base64 string."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @property
    def dimension(self) -> int:
        """Return the vector dimension for this model."""
        dims = {
            "tongyi-embedding-vision-plus": 1152,
            "tongyi-embedding-vision-flash": 768,
            "qwen3-vl-embedding": 2560,
            "multimodal-embedding-v1": 1024,
        }
        return dims.get(self.model, 1152)
