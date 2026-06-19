"""
Scene captioning via Qwen-VL (MultiModalConversation) API.

Generates structured JSON scene descriptions for each frame:
  - objects: detected objects in the scene
  - actions: actions being performed
  - lighting: bright/dim/outdoor/mixed
  - occlusion: none/partial/heavy
  - is_anomaly: whether the scene is unusual

Usage:
    captioner = Captioner()
    frames = captioner.caption_frames(frames)  # frames: List[FrameMetadata]
    # Each FrameMetadata now has scene_desc, objects, actions, etc. populated
"""

import json
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import List, Optional, Dict, Any

import dashscope

from pipeline.frame_metadata import FrameMetadata


CAPTION_PROMPT = """You are analyzing frames from an egocentric (first-person) video captured by a head-mounted camera.

Analyze this image and output a JSON object with the following fields:
- "objects": list of strings, detectable objects in the scene (e.g., ["cup", "spoon", "sink", "counter", "hand"])
- "actions": list of strings, actions being performed (e.g., ["take", "pour", "wash", "open", "cut"])
- "lighting": string, one of: "bright", "dim", "outdoor", "mixed"
- "occlusion": string, one of: "none", "partial", "heavy" (how much of the action/scene is obscured)
- "is_anomaly": boolean, whether this frame shows something unusual or unexpected

Important rules:
1. Only include objects and actions that are clearly visible or happening in the image.
2. Use concise, single-word or short-phrase labels in English (e.g., "cutting_board" not "a wooden cutting board").
3. For actions, use verb forms (e.g., "take", "open", "wash").
4. Output ONLY the JSON object, no markdown fences, no explanations.
5. If uncertain, use defaults: [], [], "mixed", "none", false.

Example output:
{"objects":["cup","spoon","sink","tap","hand"],"actions":["wash","rinse"],"lighting":"bright","occlusion":"none","is_anomaly":false}"""


class Captioner:
    """
    Scene captioner using Qwen-VL model via DashScope MultiModalConversation API.

    Config:
        VL_MODEL (env): default "qwen-vl-max"
        DASHSCOPE_API_KEY (env): DashScope API key
    """

    # Default values when JSON parsing fails
    DEFAULTS: Dict[str, Any] = {
        "objects": [],
        "actions": [],
        "lighting": "mixed",
        "occlusion": "none",
        "is_anomaly": False,
    }

    def __init__(
        self,
        model: str = "qwen-vl-max",
        api_key: str = "",
        max_retries: int = 3,
        retry_delay: float = 3.0,
        max_workers: int = 8,
    ):
        """
        Args:
            model: VL model name (qwen-vl-max, qwen3-vl-plus, qwen3-vl-flash).
            api_key: DashScope API key.
            max_retries: Retry count on transient errors.
            retry_delay: Initial delay between retries (exponential backoff).
            max_workers: Concurrent caption API calls. Qwen-VL takes one image per
                         call (no batching), so throughput comes from concurrency.
                         Lower this if you hit DashScope rate limits.
        """
        self.model = model
        self.api_key = api_key or dashscope.api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers

    def caption_frames(self, frames: List[FrameMetadata]) -> List[FrameMetadata]:
        """
        Generate structured captions for a list of frames. Populates frame fields in-place.

        Args:
            frames: List of FrameMetadata with frame_path set.

        Returns:
            Same list with scene_desc, objects, actions, lighting, occlusion,
            is_anomaly populated on each frame.
        """
        total = len(frames)
        print(f"\n  Captioning {total} frames (model={self.model}, workers={self.max_workers})")
        progress_file = "/tmp/caption_progress.txt"

        lock = threading.Lock()
        counters = {"done": 0, "ok": 0, "fail": 0}

        def worker(f: FrameMetadata):
            ok = self._caption_frame(f)
            with lock:
                counters["done"] += 1
                counters["ok" if ok else "fail"] += 1
                done = counters["done"]
                if done % 20 == 0 or done == total:
                    msg = (f"Progress: {done}/{total} "
                           f"(ok={counters['ok']}, fail={counters['fail']})")
                    print(f"  {msg}")
                    # Write to file for external visibility (bypasses stdout buffering)
                    with open(progress_file, "w") as pf:
                        pf.write(f"{msg}\nLast frame: {f.frame_id}\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            list(pool.map(worker, frames))

        success, fail = counters["ok"], counters["fail"]
        print(f"  Done: {success} ok, {fail} failed")
        with open(progress_file, "w") as pf:
            pf.write(f"DONE: {success} ok, {fail} failed (total {total})\n")
        return frames

    def _caption_frame(self, f: FrameMetadata) -> bool:
        """Caption one frame, mutating it in place. Returns True on success."""
        try:
            caption = self._caption_single(f.frame_path)
            f.scene_desc_raw = json.dumps(caption, ensure_ascii=False)
            f.objects = caption.get("objects", [])
            f.actions = caption.get("actions", [])
            f.lighting = caption.get("lighting", "mixed")
            f.occlusion = caption.get("occlusion", "none")
            f.is_anomaly = caption.get("is_anomaly", False)
            f.category = "cooking"  # Default for EPIC-KITCHENS
            return True
        except Exception as e:
            print(f"  WARN: Frame {f.frame_id} caption failed: {e}")
            # Apply defaults
            f.scene_desc_raw = json.dumps(self.DEFAULTS)
            f.objects = self.DEFAULTS["objects"]
            f.actions = self.DEFAULTS["actions"]
            f.lighting = self.DEFAULTS["lighting"]
            f.occlusion = self.DEFAULTS["occlusion"]
            f.is_anomaly = self.DEFAULTS["is_anomaly"]
            f.category = "cooking"
            return False

    def _caption_single(self, image_path: str) -> Dict[str, Any]:
        """
        Generate a structured caption for a single image.

        Args:
            image_path: Path to the frame image file.

        Returns:
            Parsed JSON dict with objects, actions, lighting, occlusion, is_anomaly.

        Raises:
            RuntimeError: If API fails after all retries.
            ValueError: If JSON parsing fails after all attempts.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{image_path}"},
                    {"text": CAPTION_PROMPT},
                ],
            }
        ]

        for attempt in range(self.max_retries):
            resp = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
            )

            if resp.status_code == HTTPStatus.OK:
                text = self._extract_text(resp)
                return self._parse_json(text)
            else:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"    Retry {attempt+1}: {resp.code}: {resp.message}, waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Caption API failed after {self.max_retries} retries: "
                        f"{resp.code} - {resp.message}"
                    )

        raise RuntimeError("Unreachable")

    @staticmethod
    def _extract_text(resp) -> str:
        """Extract text content from MultiModalConversation response."""
        content = resp.output.choices[0].message.content
        if isinstance(content, list):
            # Response is a list of content blocks; find the text block
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    return block["text"]
            # Fallback: join all text values
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict)
            )
        return str(content)

    @classmethod
    def _parse_json(cls, text: str) -> Dict[str, Any]:
        """Parse JSON from model output, stripping markdown fences if present."""
        clean = text.strip()

        # Strip markdown code fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean = "\n".join(lines).strip()

        # Try to find JSON object in the text
        # Look for first { and last }
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end + 1]

        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            # Second attempt: try to fix common issues
            clean = cls._fix_json(clean)
            try:
                result = json.loads(clean)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse failed: {e}\nRaw text: {text[:200]}")

        # Validate required fields and apply defaults
        validated = {}
        validated["objects"] = result.get("objects", [])
        if not isinstance(validated["objects"], list):
            validated["objects"] = []

        validated["actions"] = result.get("actions", [])
        if not isinstance(validated["actions"], list):
            validated["actions"] = []

        validated["lighting"] = result.get("lighting", "mixed")
        if validated["lighting"] not in ("bright", "dim", "outdoor", "mixed"):
            validated["lighting"] = "mixed"

        validated["occlusion"] = result.get("occlusion", "none")
        if validated["occlusion"] not in ("none", "partial", "heavy"):
            validated["occlusion"] = "none"

        validated["is_anomaly"] = bool(result.get("is_anomaly", False))

        return validated

    @staticmethod
    def _fix_json(text: str) -> str:
        """Attempt to fix common JSON formatting issues from LLM output."""
        # Remove trailing commas
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        # Fix single quotes
        text = text.replace("'", '"')
        # Fix unquoted keys
        text = re.sub(r"(?<=\{|\,\s*)(\w+)(?=\s*:)", r'"\1"', text)
        return text
