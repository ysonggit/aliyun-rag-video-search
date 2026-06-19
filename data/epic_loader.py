"""
EPIC-KITCHENS-100 annotation loader.

Loads the CSV annotation files and provides lookup utilities
for cross-referencing frames with ground truth labels.
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class EpicAnnotations:
    """
    Loads and queries EPIC-KITCHENS-100 annotation CSV files.

    Usage:
        ann = EpicAnnotations("/path/to/epic-kitchens-100-annotations")
        segments = ann.get_segments("P01_03")
        gt = ann.get_annotation_at_frame("P01_03", 1500)
    """

    def __init__(self, annotation_dir: str):
        self.ann_dir = Path(annotation_dir)
        self._train_data: Dict[str, List[dict]] = {}  # video_id -> list of annotation dicts
        self._load_train_csv()

    def _load_train_csv(self):
        """Load EPIC_100_train.csv and index by video_id."""
        train_csv = self.ann_dir / "EPIC_100_train.csv"
        if not train_csv.exists():
            raise FileNotFoundError(
                f"EPIC_100_train.csv not found at {train_csv}. "
                f"Clone from: https://github.com/epic-kitchens/epic-kitchens-100-annotations"
            )

        with open(train_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["video_id"]
                if video_id not in self._train_data:
                    self._train_data[video_id] = []
                # Parse all_nouns from string representation
                all_nouns = self._parse_list_field(row.get("all_nouns", "[]"))
                all_noun_classes = self._parse_list_field(row.get("all_noun_classes", "[]"), as_int=True)

                seg = {
                    "narration_id": row["narration_id"],
                    "participant_id": row["participant_id"],
                    "video_id": video_id,
                    "narration_timestamp": row["narration_timestamp"],
                    "start_timestamp": self._parse_timestamp(row["start_timestamp"]),
                    "stop_timestamp": self._parse_timestamp(row["stop_timestamp"]),
                    "start_frame": int(row["start_frame"]),
                    "stop_frame": int(row["stop_frame"]),
                    "narration": row.get("narration", ""),
                    "verb": row.get("verb", ""),
                    "verb_class": int(row.get("verb_class", -1)),
                    "noun": row.get("noun", ""),
                    "noun_class": int(row.get("noun_class", -1)),
                    "all_nouns": all_nouns,
                    "all_noun_classes": all_noun_classes,
                }
                self._train_data[video_id].append(seg)

    @staticmethod
    def _parse_timestamp(ts_str: str) -> float:
        """Convert HH:MM:SS.ms string to seconds (float)."""
        parts = ts_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return float(ts_str)

    @staticmethod
    def _parse_list_field(value: str, as_int: bool = False) -> list:
        """Parse list fields stored as string representations like ['a','b'] or [1,2]."""
        import ast
        try:
            parsed = ast.literal_eval(value)
            if as_int:
                return [int(x) for x in parsed]
            return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            return []

    def get_segments(self, video_id: str) -> List[dict]:
        """Return all annotation segments for a given video_id."""
        return self._train_data.get(video_id, [])

    def get_annotation_at_frame(self, video_id: str, frame_number: int) -> Optional[dict]:
        """
        Find the annotation segment that contains the given frame number.
        Returns the segment dict or None if frame is not in any annotated segment.
        """
        segments = self.get_segments(video_id)
        for seg in segments:
            if seg["start_frame"] <= frame_number <= seg["stop_frame"]:
                return seg
        return None

    def get_annotated_frame_ranges(self, video_id: str) -> List[Tuple[int, int]]:
        """Return list of (start_frame, stop_frame) tuples for a video."""
        segments = self.get_segments(video_id)
        return [(seg["start_frame"], seg["stop_frame"]) for seg in segments]

    @property
    def available_videos(self) -> List[str]:
        """Return list of video_ids with annotations."""
        return sorted(self._train_data.keys())

    def __len__(self) -> int:
        return sum(len(segs) for segs in self._train_data.values())

    def __repr__(self) -> str:
        return f"EpicAnnotations({len(self._train_data)} videos, {len(self)} segments)"
