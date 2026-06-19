"""
DashVector indexer for ingesting frame embeddings with metadata.

Creates collection (if not exists) and upserts FrameMetadata records
as DashVector Doc objects with dense vectors + filterable fields.

Usage:
    indexer = Indexer(config)
    indexer.create_collection()
    count = indexer.index(frames)  # Upsert frames into DashVector
"""

import json
import time
from typing import List

from dashvector import Client, Doc

from pipeline.frame_metadata import FrameMetadata
from schema import FIELDS_SCHEMA, VECTOR_METRIC


class Indexer:
    """
    Manages DashVector collection and document ingestion.

    Config:
        DASHVECTOR_API_KEY, DASHVECTOR_ENDPOINT, DV_COLLECTION_NAME (env)
    """

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        collection_name: str = "scene_frames",
        dimension: int = 1152,
        metric: str = VECTOR_METRIC,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric = metric
        self._client: Client = None
        self._collection = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(
                api_key=self.api_key,
                endpoint=self.endpoint,
            )
        return self._client

    def create_collection(self, if_not_exists: bool = True) -> bool:
        """
        Create the DashVector collection if it doesn't exist.

        Returns:
            True if collection was created or already exists.
        """
        print(f"\n  Creating collection: {self.collection_name} (dim={self.dimension})")

        if if_not_exists:
            existing = self.client.get(self.collection_name)
            if existing:
                print(f"  Collection already exists, using existing.")
                self._collection = existing
                return True

        ret = self.client.create(
            name=self.collection_name,
            dimension=self.dimension,
            metric=self.metric,
            fields_schema=FIELDS_SCHEMA,
        )

        if ret:
            print(f"  Collection created successfully.")
            self._collection = self.client.get(self.collection_name)
            return True
        else:
            print(f"  FAIL: {ret.code} - {ret.message}")
            return False

    def get_collection(self):
        """Get or create the collection reference."""
        if self._collection is None:
            self._collection = self.client.get(self.collection_name)
            if self._collection is None:
                raise RuntimeError(f"Collection '{self.collection_name}' not found. Run create_collection() first.")
        return self._collection

    def index(
        self,
        frames: List[FrameMetadata],
        embedding_model: str = "",
        caption_model: str = "",
        batch_size: int = 50,
    ) -> int:
        """
        Upsert frames into DashVector.

        Args:
            frames: FrameMetadata objects with embedding, caption fields populated.
            embedding_model: Model name used for embedding (recorded in metadata).
            caption_model: Model name used for captioning.
            batch_size: Number of docs per upsert call.

        Returns:
            Number of documents upserted.
        """
        collection = self.get_collection()
        now = time.time()
        total = len(frames)
        ingested = 0

        print(f"\n  Ingesting {total} frames into DashVector...")

        for batch_start in range(0, total, batch_size):
            batch = frames[batch_start : batch_start + batch_size]
            docs = []

            for f in batch:
                if f.embedding is None:
                    print(f"  WARN: Frame {f.frame_id} has no embedding, skipping")
                    continue

                doc = Doc(
                    id=f.frame_id,
                    vector=f.embedding,
                    fields={
                        # Core identification
                        "video_id": f.video_id,
                        "frame_id": f.frame_id,
                        "timestamp": f.timestamp,
                        "oss_path": f.frame_path,  # local path for now

                        # Qwen-VL caption
                        "scene_desc": f.scene_desc_raw,
                        "objects": f.objects,
                        "actions": f.actions,
                        "lighting": f.lighting,
                        "occlusion": f.occlusion,
                        "is_anomaly": f.is_anomaly,

                        # Category
                        "category": f.category,

                        # Ego-centric fields (populated in Phase 6)
                        "episode_id": f.video_id.split("_")[0] if "_" in f.video_id else "",
                        "action_label": f.gt_verb,
                        "proprioception_ts": 0.0,
                        "view_type": "ego",

                        # Ground truth (EPIC-KITCHENS)
                        "gt_narration": f.gt_narration,

                        # Processing metadata
                        "embedding_model": embedding_model,
                        "caption_model": caption_model,
                        "created_at": now,
                    },
                )
                docs.append(doc)

            if docs:
                rsp = collection.upsert(docs)
                if rsp:
                    ingested += len(docs)
                    batch_num = batch_start // batch_size + 1
                    total_batches = (total + batch_size - 1) // batch_size
                    print(f"  Batch {batch_num}/{total_batches}: {len(docs)} docs upserted")
                else:
                    print(f"  WARN: Batch upsert failed: {rsp.code} - {rsp.message}")

        print(f"  Done: {ingested}/{total} frames ingested")
        return ingested

    def delete_collection(self) -> bool:
        """Delete the collection (use with caution)."""
        ret = self.client.delete(self.collection_name)
        if ret:
            print(f"  Collection '{self.collection_name}' deleted")
        return bool(ret)
