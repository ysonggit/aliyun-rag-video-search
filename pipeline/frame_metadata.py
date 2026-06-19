"""
Frame metadata dataclass.
Used to represent extracted frame information before embedding.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FrameMetadata:
    """Represents a single extracted frame with its metadata."""
    video_id: str               # Source video ID (e.g., "P01_03")
    frame_id: str               # Unique frame ID (e.g., "P01_03_f0123")
    frame_path: str             # Local path to frame image file
    timestamp: float            # Frame timestamp in seconds
    frame_number: int           # Frame number in the original video

    # EPIC-KITCHENS ground truth (empty if not in an annotated segment)
    gt_narration: str = ""      # Human-written narration
    gt_verb: str = ""           # Verb label (ground truth)
    gt_noun: str = ""           # Noun label (ground truth)
    gt_all_nouns: List[str] = field(default_factory=list)

    # Placeholders for ML pipeline output (populated in later milestones)
    embedding: Optional[List[float]] = None
    scene_desc_raw: str = ""    # Raw JSON string from Qwen-VL
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    lighting: str = ""
    occlusion: str = ""
    is_anomaly: bool = False
    category: str = ""
