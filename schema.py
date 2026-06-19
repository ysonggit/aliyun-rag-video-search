"""
DashVector collection schema definition.
Includes ego-centric reserved fields for Phase 6 migration.

Updated for EPIC-KITCHENS-100 test dataset:
  - Added "category" for coarse-grained scene categorization
  - Added "gt_narration" for EPIC-KITCHENS ground truth (evaluation only)
"""

from typing import List
from dashvector import Doc


# ============================================================
# Metadata Schema for DashVector Collection
# ============================================================
# This schema is used when creating the DashVector collection.
# All fields are searchable via filter syntax (SQL-like).
# ============================================================

FIELDS_SCHEMA = {
    # --- Core identification ---
    "video_id": str,            # Source video identifier (e.g., "P01_03")
    "frame_id": str,            # Frame unique ID (e.g., "P01_03_f0123")
    "timestamp": float,         # Frame position in video (seconds)
    "oss_path": str,            # OSS path to frame image (local path for Phase 1-2)

    # --- Qwen-VL structured scene caption ---
    "scene_desc": str,          # Full JSON caption (compressed string)
    "objects": List[str],       # Detected objects, e.g. ["cup", "table", "person"]
    "actions": List[str],       # Detected actions, e.g. ["cooking", "standing"]
    "lighting": str,            # "bright" | "dim" | "outdoor" | "mixed"
    "occlusion": str,           # "none" | "partial" | "heavy"
    "is_anomaly": bool,         # Whether the scene contains an anomaly

    # --- Category ---
    "category": str,            # Scene category (e.g. "cooking", "cleaning", "assembly")

    # --- Ego-centric reserved fields (Phase 6, populated later) ---
    "episode_id": str,          # Collection episode identifier (e.g., "P01" for EPIC)
    "action_label": str,        # Fine-grained action label (e.g., "grasping_cup")
    "proprioception_ts": float, # Timestamp for proprioception data alignment
    "view_type": str,           # "ego" (first-person) | "third_person"

    # --- Ground truth (for evaluation only, empty in production) ---
    "gt_narration": str,        # Human-written narration from EPIC-KITCHENS (e.g., "take cup")

    # --- Processing metadata ---
    "embedding_model": str,     # Model used for embedding
    "caption_model": str,       # Model used for captioning
    "created_at": float,        # Unix timestamp of processing time
}

# Vector dimension per model:
#   tongyi-embedding-vision-plus → 1152 (fixed)
#   qwen3-vl-embedding          → 256/512/768/1024/1536/2048/2560
VECTOR_METRIC = "cosine"  # cosine similarity for semantic search
