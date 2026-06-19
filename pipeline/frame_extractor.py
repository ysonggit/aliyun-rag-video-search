"""
Frame extraction from video files.

Strategy: fixed interval (0.5 fps by default) using OpenCV.
For later phases, keyframe extraction via FFmpeg scene detection is planned.

Each extracted frame is saved as JPG and annotated with metadata
(cross-referenced against EPIC-KITCHENS annotations if available).
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pipeline.frame_metadata import FrameMetadata


class FrameExtractor:
    """
    Extract frames from a video at a fixed frame interval.

    Usage:
        extractor = FrameExtractor(annotations=epic_annotations, output_dir="storage/frames")
        frames = extractor.extract("storage/videos/P01_03.MP4", video_id="P01_03")
    """

    def __init__(
        self,
        fps: float = 0.5,
        output_dir: str = "storage/frames",
        annotations: Optional["EpicAnnotations"] = None,
        quality: int = 90,
    ):
        """
        Args:
            fps: Frames per second to extract (default 0.5 = 1 frame every 2 seconds).
            output_dir: Directory to save extracted frame images.
            annotations: Optional EPIC-KITCHENS annotation object for ground truth lookup.
            quality: JPEG compression quality (0-100).
        """
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.annotations = annotations
        self.quality = quality
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: str, video_id: Optional[str] = None) -> List[FrameMetadata]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file (mp4, avi, mov, etc.).
            video_id: Optional identifier (defaults to filename without extension).

        Returns:
            List of FrameMetadata objects for each extracted frame.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if video_id is None:
            video_id = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / src_fps if src_fps > 0 else 0

        # Calculate extraction interval: every Nth source frame
        if src_fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS ({src_fps}) for video: {video_path}")

        frame_interval = max(1, int(src_fps / self.fps))

        print(f"  Video: {video_id}")
        print(f"  Source: {total_frames} frames @ {src_fps:.1f} fps, {duration:.1f}s")
        print(f"  Extract: 1 frame every {frame_interval} source frames (~{self.fps} fps)")
        print(f"  Output: {self.output_dir / video_id}")

        # Create output subdirectory
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)

        frames: List[FrameMetadata] = []
        frame_idx = 0
        extracted = 0

        while True:
            ret, image = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / src_fps

                # Lookup EPIC-KITCHENS ground truth if available
                gt_narration = ""
                gt_verb = ""
                gt_noun = ""
                gt_all_nouns = []

                if self.annotations:
                    seg = self.annotations.get_annotation_at_frame(video_id, frame_idx)
                    if seg:
                        gt_narration = seg["narration"]
                        gt_verb = seg["verb"]
                        gt_noun = seg["noun"]
                        gt_all_nouns = seg["all_nouns"]

                # Save frame image
                frame_id = f"{video_id}_f{frame_idx:06d}"
                frame_filename = f"{frame_id}.jpg"
                frame_path = video_output_dir / frame_filename

                cv2.imwrite(
                    str(frame_path),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                )

                meta = FrameMetadata(
                    video_id=video_id,
                    frame_id=frame_id,
                    frame_path=str(frame_path),
                    timestamp=timestamp,
                    frame_number=frame_idx,
                    gt_narration=gt_narration,
                    gt_verb=gt_verb,
                    gt_noun=gt_noun,
                    gt_all_nouns=gt_all_nouns,
                )
                frames.append(meta)
                extracted += 1

            frame_idx += 1

        cap.release()

        # Save metadata JSON for this video
        self._save_metadata_json(video_id, frames)

        annotated = sum(1 for f in frames if f.gt_narration)
        print(f"  Extracted: {extracted} frames ({annotated} annotated)")

        return frames

    def _save_metadata_json(self, video_id: str, frames: List[FrameMetadata]):
        """Save frame metadata as JSON for inspection and pipeline use."""
        meta_path = self.output_dir / video_id / f"{video_id}_metadata.json"
        records = []
        for f in frames:
            records.append({
                "frame_id": f.frame_id,
                "timestamp": f.timestamp,
                "frame_number": f.frame_number,
                "frame_path": f.frame_path,
                "gt_narration": f.gt_narration,
                "gt_verb": f.gt_verb,
                "gt_noun": f.gt_noun,
                "objects": f.objects,
                "actions": f.actions,
            })

        with open(meta_path, "w") as fp:
            json.dump(records, fp, indent=2)

        print(f"  Metadata saved: {meta_path}")

    @staticmethod
    def create_test_video(output_path: str, duration_sec: float = 5.0, fps: float = 30):
        """
        Create a synthetic test video with timestamp overlay for testing.

        Each frame shows a colored rectangle and frame number.
        """
        import cv2
        import numpy as np

        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        width, height = 640, 480
        total_frames = int(duration_sec * fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create test video: {output_path}")

        colors = [
            (0, 0, 255),   # red
            (0, 255, 0),   # green
            (255, 0, 0),   # blue
            (0, 255, 255), # yellow
            (255, 0, 255), # magenta
        ]

        for i in range(total_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 30
            color = colors[(i // 30) % len(colors)]  # Change color every 30 frames
            cv2.rectangle(frame, (50, 50), (590, 430), color, -1)
            cv2.putText(
                frame, f"Frame {i:04d}", (200, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f"t={i/fps:.2f}s", (200, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            writer.write(frame)

        writer.release()
        print(f"  Test video created: {output_path} ({total_frames} frames, {duration_sec}s)")
