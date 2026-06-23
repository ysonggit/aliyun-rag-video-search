#!/usr/bin/env python
"""
Download specific EPIC-KITCHENS-100 videos.

Uses the official download script from:
  https://github.com/epic-kitchens/epic-kitchens-download-scripts

Usage:
  python scripts/download_epic_videos.py

This downloads videos to storage/videos/ (gitignored).
"""

import os
import sys
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The downloader appends 'EPIC-KITCHENS/' to the base output path.
# So with base=storage, files go to storage/EPIC-KITCHENS/
VIDEO_OUTPUT = PROJECT_ROOT / "storage"

# 20 videos across 8 participants (P01-P08) for fine-tuning diversity
SELECTED_VIDEOS = [
    # P01 (7 videos, already downloaded)
    "P01_03", "P01_04", "P01_06", "P01_07", "P01_08", "P01_10", "P01_16",
    # P02 (2 videos, already downloaded)
    "P02_01", "P02_02",
    # P03 (2 new)
    "P03_107", "P03_123",
    # P04 (2 new)
    "P04_115", "P04_117",
    # P05 (2 new)
    "P05_02", "P05_06",
    # P06 (2 new)
    "P06_107", "P06_113",
    # P07 (2 new)
    "P07_110", "P07_115",
    # P08 (1 new)
    "P08_04",
]


def main():
    # Check if download script exists
    downloader_repo = Path("/tmp/epic-kitchens-download-scripts")
    if not (downloader_repo / "epic_downloader.py").exists():
        print("Cloning download scripts...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git",
             str(downloader_repo)],
            check=True,
        )

    VIDEO_OUTPUT.mkdir(parents=True, exist_ok=True)

    videos_str = ",".join(SELECTED_VIDEOS)
    cmd = [
        sys.executable,
        str(downloader_repo / "epic_downloader.py"),
        "--videos",
        "--specific-videos", videos_str,
        "--output-path", str(VIDEO_OUTPUT.parent),
    ]

    print(f"Downloading videos: {videos_str}")
    print(f"Output directory: {VIDEO_OUTPUT}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(downloader_repo))
    if result.returncode != 0:
        print("\nDownload failed. Check the error above.")
        print("Alternative: manually download from https://data.bris.ac.uk/")
        sys.exit(1)

    print(f"\nDone. Videos saved to: {VIDEO_OUTPUT}/EPIC-KITCHENS/")
    print(f"Expected files under: {VIDEO_OUTPUT}/EPIC-KITCHENS/P01/{', '.join(SELECTED_VIDEOS)}.MP4")


if __name__ == "__main__":
    main()
