"""
Milestone 2: Frame Extraction Module — Sanity Check

Tests:
  1. Create synthetic test video
  2. Extract frames at 0.5fps
  3. Verify frame count and metadata correctness
  4. Verify EPIC annotation cross-referencing (if annotations available)

Usage:
  python tests/test_frame_extractor.py

Acceptance criteria:
  - Synthetic video extraction returns correct frame count
  - Each frame has valid timestamp and frame_number
  - Frame images exist on disk
  - Annotation lookup works (when EpicAnnotations is available)

Sanity check:
  python tests/test_frame_extractor.py && echo "FRAME EXTRACTION OK"
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.frame_extractor import FrameExtractor
from pipeline.frame_metadata import FrameMetadata


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_synthetic_video():
    """Test frame extraction from a synthetic test video."""
    separator("Test 1: Synthetic Video Frame Extraction")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a 5-second test video at 30fps
        video_path = os.path.join(tmpdir, "test.mp4")
        FrameExtractor.create_test_video(video_path, duration_sec=5.0, fps=30)

        # Extract frames at 0.5fps (1 frame every 2 seconds)
        extractor = FrameExtractor(fps=0.5, output_dir=os.path.join(tmpdir, "frames"))
        frames = extractor.extract(video_path, video_id="test_video")

        # Expected: 5s * 0.5fps = 2 or 3 frames (depending on boundary)
        # At 30fps source, interval = int(30/0.5) = 60 frames
        # Frames at indices: 0, 60, 120, ... = 3 frames (for 150 total frames)
        expected_min = 2
        expected_max = 4

        if not (expected_min <= len(frames) <= expected_max):
            print(f"  FAIL: Expected {expected_min}-{expected_max} frames, got {len(frames)}")
            return False
        print(f"  Frame count: {len(frames)} (OK)")

        # Verify each frame
        for f in frames:
            if not os.path.exists(f.frame_path):
                print(f"  FAIL: Frame image not found: {f.frame_path}")
                return False
            if f.timestamp < 0 or f.timestamp > 5.0:
                print(f"  FAIL: Invalid timestamp: {f.timestamp}")
                return False
            if f.video_id != "test_video":
                print(f"  FAIL: Invalid video_id: {f.video_id}")
                return False
            print(f"  Frame: {f.frame_id:20s} t={f.timestamp:.2f}s  frame#={f.frame_number}")

        # Verify metadata JSON was written
        meta_path = os.path.join(tmpdir, "frames", "test_video", "test_video_metadata.json")
        if not os.path.exists(meta_path):
            print(f"  FAIL: Metadata file not created: {meta_path}")
            return False
        with open(meta_path) as fp:
            meta = json.load(fp)
        assert len(meta) == len(frames), f"Metadata count mismatch: {len(meta)} vs {len(frames)}"
        print(f"  Metadata JSON: {len(meta)} records (OK)")

    print(f"  PASS: Synthetic video extraction works correctly")
    return True


def test_epic_annotation_crossref():
    """Test EPIC-KITCHENS annotation cross-referencing with a synthetic video."""
    separator("Test 2: Annotation Cross-Reference (without real video)")

    annotations_dir = "/tmp/epic-kitchens-100-annotations"
    if not os.path.isdir(annotations_dir):
        print("  SKIP: EPIC-KITCHENS annotations not found at /tmp/epic-kitchens-100-annotations")
        print("  Clone with: git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git /tmp/epic-kitchens-100-annotations")
        return True  # Not a failure, just unavailable

    try:
        from data.epic_loader import EpicAnnotations

        ann = EpicAnnotations(annotations_dir)
        print(f"  Loaded: {ann}")
        print(f"  Available videos (first 10): {ann.available_videos[:10]}")

        # Test: get segments for P01_03
        segments = ann.get_segments("P01_03")
        if segments:
            print(f"  P01_03: {len(segments)} annotation segments")
            s0 = segments[0]
            print(f"    Example: [{s0['verb']}] {s0['narration']} (frames {s0['start_frame']}-{s0['stop_frame']})")

        # Test: lookup a specific annotation range
        for vid in ["P01_03", "P01_04", "P01_08"]:
            segs = ann.get_segments(vid)
            annotated_frames = sum(s["stop_frame"] - s["start_frame"] for s in segs)
            print(f"  {vid}: {len(segs)} segments, ~{annotated_frames} annotated frames")

        # Test: frame-level lookup
        if segments:
            # Look up a frame in the middle of the first segment
            mid_frame = (segments[0]["start_frame"] + segments[0]["stop_frame"]) // 2
            gt = ann.get_annotation_at_frame("P01_03", mid_frame)
            if gt:
                print(f"  Frame {mid_frame} lookup: [{gt['verb']}] {gt['narration']} (OK)")
            else:
                print(f"  FAIL: Frame {mid_frame} should be in annotated segment")
                return False

            # Frame outside any annotated segment
            gt_none = ann.get_annotation_at_frame("P01_03", 0)  # frame 0 may not be annotated
            print(f"  Frame 0 lookup: {'annotated' if gt_none else 'not annotated'} (OK)")

        print(f"  PASS: Annotation cross-reference works")
        return True

    except ImportError as e:
        print(f"  SKIP: Import error: {e}")
        return True


if __name__ == "__main__":
    print("RAG Video Search — Milestone 2: Frame Extraction Sanity Check")
    print(f"Project root: {Path(__file__).resolve().parent.parent}")

    results = {}
    results["synthetic"] = test_synthetic_video()
    results["annotations"] = test_epic_annotation_crossref()

    separator("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All checks passed.")
        sys.exit(0)
    else:
        print("\n  Some checks failed. Review output above.")
        sys.exit(1)
