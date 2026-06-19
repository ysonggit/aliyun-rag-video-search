"""
Milestone 5: Retrieval Evaluation (Improved)

Improvements over v1:
  1. Softer matching: same video + time window (±2s) of any EPIC segment
     instead of exact gt_narration match on the frame itself.
  2. New hybrid strategy: verb filter (action_label = expected EPIC verb)
  3. Per-strategy metrics: recall@k, MRR, mean rank

Usage:
  python tests/test_evaluation.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import dashscope_config, dashvector_config
from pipeline.retriever import Retriever, SearchResult
from data.epic_loader import EpicAnnotations


# ============================================================
# Query Set
# ============================================================

QUERIES = [
    # P01_03
    {"query": "open door",             "video": "P01_03", "verb": "open",    "noun": "door"},
    {"query": "open cupboard",         "video": "P01_03", "verb": "open",    "noun": "cupboard"},
    {"query": "take cup",              "video": "P01_03", "verb": "take",    "noun": "cup"},
    {"query": "put down cup",          "video": "P01_03", "verb": "put-down","noun": "cup"},
    {"query": "take soy milk",         "video": "P01_03", "verb": "take",    "noun": "soy milk"},
    {"query": "open fridge",           "video": "P01_03", "verb": "open",    "noun": "fridge"},
    {"query": "close fridge",          "video": "P01_03", "verb": "close",   "noun": "fridge"},
    {"query": "pour soy milk",         "video": "P01_03", "verb": "pour",    "noun": "soy milk"},
    {"query": "take spoon",            "video": "P01_03", "verb": "take",    "noun": "spoon"},
    {"query": "open drawer",           "video": "P01_03", "verb": "open",    "noun": "drawer"},
    # P01_04
    {"query": "wash cup",              "video": "P01_04", "verb": "wash",    "noun": "cup"},
    {"query": "rinse spoon",           "video": "P01_04", "verb": "rinse",   "noun": "spoon"},
    {"query": "open tap",              "video": "P01_04", "verb": "open",    "noun": "tap"},
    {"query": "dry hands",             "video": "P01_04", "verb": "dry",     "noun": "hands"},
    {"query": "drink water",           "video": "P01_04", "verb": "drink",   "noun": "water"},
    {"query": "pour water",            "video": "P01_04", "verb": "pour",    "noun": "water"},
    # P01_08 — cooking, cleaning (verified from actual annotations)
    {"query": "wash cup",              "video": "P01_08", "verb": "wash",    "noun": "cup"},
    {"query": "rinse spoon",           "video": "P01_08", "verb": "rinse",   "noun": "spoon"},
    {"query": "dry hands",             "video": "P01_08", "verb": "dry",     "noun": "hand"},
    {"query": "drink water",           "video": "P01_08", "verb": "drink",   "noun": "water"},
]

TIME_WINDOW = 2.0  # ±2 seconds tolerance for relevance


def build_annotation_index(annotations: EpicAnnotations) -> Dict[str, List[dict]]:
    """Build an index: (video_id, narration) → list of [start_ts, stop_ts]."""
    index = {}
    for vid in annotations.available_videos:
        for seg in annotations.get_segments(vid):
            key = (vid, seg["narration"].lower())
            if key not in index:
                index[key] = []
            index[key].append((seg["start_timestamp"], seg["stop_timestamp"]))
    return index


def is_relevant_soft(
    result: SearchResult,
    gt_query: dict,
    ann_index: Dict[tuple, List[tuple]],
    window: float = TIME_WINDOW,
) -> bool:
    """
    Soft relevance: result is relevant if its timestamp falls within
    the time window of any EPIC annotation segment matching the query.
    """
    if result.video_id != gt_query["video"]:
        return False

    key = (gt_query["video"], gt_query["query"].lower())
    segments = ann_index.get(key, [])

    r_ts = result.timestamp
    for start_ts, stop_ts in segments:
        if (start_ts - window) <= r_ts <= (stop_ts + window):
            return True
    return False


def evaluate(
    retriever: Retriever,
    queries: List[dict],
    ann_index: Dict[tuple, List[tuple]],
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Run multi-strategy evaluation."""
    results_per_query = []

    for i, q in enumerate(queries):
        qr = {"query": q["query"], "video": q["video"], "strategies": {}}
        print(f"  [{i+1}/{len(queries)}] \"{q['query']}\" ({q['video']})")

        # ---- Strategy A: Pure Vector ----
        r = retriever.search(q["query"], top_k=max(k_values))
        qr["strategies"]["pure_vector"] = compute_metrics(r, q, ann_index, k_values)

        # ---- Strategy B: Hybrid video filter ----
        r = retriever.search(q["query"], top_k=max(k_values), video_id=q["video"])
        qr["strategies"]["hybrid_video"] = compute_metrics(r, q, ann_index, k_values)

        # ---- Strategy C: Hybrid video + verb filter ----
        if q.get("verb"):
            r = retriever.search(
                q["query"], top_k=max(k_values),
                video_id=q["video"],
                episode_id="",  # clear
            )
            # Apply verb filter client-side since it's stored in action_label
            r = [x for x in r if x.frame_id.split("_")[0] + "_" + x.frame_id.split("_")[1] == q["video"] or True]
            # Actually use DashVector scalar filter on action_label
            from pipeline.filter_builder import FilterBuilder
            fb = FilterBuilder()
            fb.eq("video_id", q["video"])
            fb.eq("action_label", q["verb"])
            r = retriever.search_by_vector(
                retriever._embed_text(q["query"]),
                top_k=max(k_values),
                filter_str=fb.build(),
            )
            qr["strategies"]["hybrid_verb"] = compute_metrics(r, q, ann_index, k_values)

        print(f"    pure={hits_str(qr['strategies']['pure_vector'], max(k_values), k_values)}  "
              f"video={hits_str(qr['strategies']['hybrid_video'], max(k_values), k_values)}  "
              f"verb={hits_str(qr['strategies'].get('hybrid_verb', {}), max(k_values), k_values)}")

        results_per_query.append(qr)

    # Aggregate
    strategies = ["pure_vector", "hybrid_video", "hybrid_verb"]
    summary = {
        "total_queries": len(queries),
        "k_values": list(k_values),
        "time_window": TIME_WINDOW,
        "strategies": {},
        "per_query": results_per_query,
    }
    for s in strategies:
        metrics = [qr["strategies"].get(s, {}) for qr in results_per_query]
        metrics = [m for m in metrics if m]
        if metrics:
            summary["strategies"][s] = aggregate_metrics(metrics, k_values)

    return summary


def compute_metrics(
    results: List[SearchResult],
    gt: dict,
    ann_index: Dict[tuple, List[tuple]],
    k_values: Tuple[int, ...],
) -> dict:
    """Per-query metrics with soft matching."""
    m = {"total_retrieved": len(results)}

    for k in k_values:
        top_k = results[:k]
        hits = sum(1 for r in top_k if is_relevant_soft(r, gt, ann_index))
        m[f"hits@{k}"] = hits
        m[f"recall@{k}"] = 1.0 if hits > 0 else 0.0
        m[f"precision@{k}"] = hits / k if k > 0 else 0.0

    # MRR
    for rank, r in enumerate(results, 1):
        if is_relevant_soft(r, gt, ann_index):
            m["mrr"] = 1.0 / rank
            break
    else:
        m["mrr"] = 0.0

    return m


def hits_str(metrics: dict, max_k: int, k_values: tuple) -> str:
    if not metrics:
        return "N/A"
    parts = []
    for k in sorted(k_values):
        parts.append(f"{metrics.get(f'hits@{k}', '?')}/{k}")
    return "/".join(parts)


def aggregate_metrics(per_query: List[dict], k_values: Tuple[int, ...]) -> dict:
    n = len(per_query)
    agg = {"num_queries": n}
    for k in k_values:
        agg[f"recall@{k}"] = sum(m[f"recall@{k}"] for m in per_query) / n
    agg["mrr"] = sum(m["mrr"] for m in per_query) / n
    return agg


def print_report(summary: dict):
    k_vals = summary["k_values"]
    print(f"\n{'=' * 75}")
    print(f"  EVALUATION REPORT ({summary['total_queries']} queries, "
          f"time_window=±{summary['time_window']}s)")
    print(f"{'=' * 75}")

    # Header
    hdr = f"{'Strategy':<18s}"
    for k in k_vals:
        hdr += f"  Recall@{k:<2d}"
    hdr += f"    MRR"
    print(hdr)
    print("-" * 75)

    for name, m in summary["strategies"].items():
        line = f"{name:<18s}"
        for k in k_vals:
            line += f"    {m.get(f'recall@{k}', 0):.4f}"
        line += f"    {m.get('mrr', 0):.4f}"
        print(line)

    # Comparison with v1 (strict) equivalent
    print(f"\n  --- vs v1 (strict exact gt_narration match) ---")
    print(f"  {'Strategy':<18s}  Recall@5 (v2 soft)  Recall@5 (v1 strict)")
    print(f"  {'-'*55}")
    v1_results = {
        "pure_vector": 0.25,
        "hybrid_video": 0.30,
        "hybrid_verb": None,
    }
    for name in ["pure_vector", "hybrid_video", "hybrid_verb"]:
        m = summary["strategies"].get(name, {})
        v2 = m.get("recall@5", 0)
        v1 = v1_results.get(name)
        v1_str = f"{v1:.4f}" if v1 is not None else "N/A"
        print(f"  {name:<18s}  {v2:.4f}             {v1_str}")

    # Per-query detail
    print(f"\n{'=' * 75}")
    print(f"  PER-QUERY (hits@{max(k_vals)}, soft matching)")
    print(f"{'=' * 75}")
    for qr in summary["per_query"]:
        q = qr["query"]
        vid = qr["video"]
        p = qr["strategies"].get("pure_vector", {})
        h = qr["strategies"].get("hybrid_video", {})
        v = qr["strategies"].get("hybrid_verb", {})
        mk = max(k_vals)
        p_hit = p.get(f"hits@{mk}", "?")
        h_hit = h.get(f"hits@{mk}", "?")
        v_hit = v.get(f"hits@{mk}", "?")
        parts = [f"{q:<25s} ({vid})  pure={p_hit}/{mk}  video={h_hit}/{mk}"]
        if v:
            parts.append(f"  verb={v_hit}/{mk}")
        print("".join(parts))


def main():
    print("RAG Video Search — Milestone 5: Evaluation (v2 — soft matching)")
    print(f"Queries: {len(QUERIES)}, time window: ±{TIME_WINDOW}s")

    ann = EpicAnnotations(os.environ.get(
        "EPIC_ANNOTATIONS_DIR", "/tmp/epic-kitchens-100-annotations"
    ))
    ann_index = build_annotation_index(ann)

    retriever = Retriever(
        embedding_model=dashscope_config.embedding_model,
        dashvector_api_key=dashvector_config.api_key,
        dashvector_endpoint=dashvector_config.endpoint,
        collection_name=dashvector_config.collection_name,
    )

    print(f"\n[Eval] {len(QUERIES)} queries...")
    start = time.time()
    summary = evaluate(retriever, QUERIES, ann_index, k_values=(1, 3, 5, 10))
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.0f}s")

    print_report(summary)

    out_path = Path(__file__).resolve().parent.parent / "storage" / "eval_results_v2.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
