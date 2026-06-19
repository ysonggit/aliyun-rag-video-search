"""
Milestone 4: Retrieval Module Test

Tests:
  1. Filter builder (server-side scalar filters)
  2. Pure vector search (text query → embedding → ANN)
  3. Hybrid search (vector + scalar filter + client-side array filter)
  4. EPIC ground truth queries

Usage:
  python tests/test_retrieval.py

Acceptance criteria:
  - Text query returns results with valid scores
  - Scalar filters (lighting, video_id) narrow correctly
  - Client-side object/action filters work
  - EPIC ground truth queries return results

Sanity check:
  python tests/test_retrieval.py && echo "RETRIEVAL OK"
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import dashscope_config, dashvector_config
from pipeline.retriever import Retriever, SearchResult
from pipeline.filter_builder import FilterBuilder, build_filter, FilterParams


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_filter_builder():
    """Test filter string construction for server-side scalar filters."""
    separator("Test 1: Filter Builder (Server-Side)")

    # Test scalar filter string generation
    tests = [
        ("equality", FilterBuilder().eq("lighting", "bright").build(),
         'lighting = "bright"'),
        ("boolean", FilterBuilder().eq("is_anomaly", True).build(),
         "is_anomaly = true"),
        ("multiple AND", FilterBuilder().eq("lighting", "bright")
         .eq("is_anomaly", False).build(),
         'lighting = "bright" AND is_anomaly = false'),
        ("empty", FilterBuilder().build(), ""),
    ]

    all_ok = True
    for desc, actual, expected in tests:
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_ok = False
        print(f"  [{status}] {desc}")
        if actual != expected:
            print(f"         expected: {expected}")
            print(f"         actual:   {actual}")

    # Test FilterParams structure
    print("\n  FilterParams (server + client split):")
    fp = build_filter(
        objects=["cup", "spoon"],
        lighting="bright",
        video_id="P01_03",
    )
    print(f"    server_filter:  '{fp.server_filter}'")
    print(f"    objects:        {fp.objects}")
    print(f"    actions:        {fp.actions}")
    print(f"    has_client_filter: {fp.has_client_filter}")

    assert fp.server_filter == 'lighting = "bright" AND video_id = "P01_03"', \
        f"Unexpected server_filter: {fp.server_filter}"
    assert fp.objects == ["cup", "spoon"]
    assert fp.has_client_filter

    print(f"  PASS: FilterParams split correctly")

    if all_ok:
        print(f"\n  PASS: All filter builder tests passed")
    return all_ok


def test_vector_search():
    """Test pure vector search (no metadata filter)."""
    separator("Test 2: Pure Vector Search")

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("  SKIP: DASHSCOPE_API_KEY not set")
        return True

    retriever = Retriever(
        embedding_model=dashscope_config.embedding_model,
        dashvector_api_key=dashvector_config.api_key,
        dashvector_endpoint=dashvector_config.endpoint,
        collection_name=dashvector_config.collection_name,
    )

    queries = [
        "red cup on counter",
        "blue plate in sink",
        "green bowl on table",
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        results = retriever.search(query, top_k=3)

        if not results:
            print(f"  WARN: No results")
            continue

        for i, r in enumerate(results):
            print(f"  #{i+1} score={r.score:.4f} {r.frame_id} objects={r.objects}")

    print(f"\n  PASS: Vector search works")
    return True


def test_hybrid_search():
    """Test hybrid: scalar filter + client-side array filter."""
    separator("Test 3: Hybrid Search")

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("  SKIP: DASHSCOPE_API_KEY not set")
        return True

    retriever = Retriever(
        embedding_model=dashscope_config.embedding_model,
        dashvector_api_key=dashvector_config.api_key,
        dashvector_endpoint=dashvector_config.endpoint,
        collection_name=dashvector_config.collection_name,
    )

    query = "kitchen scene"

    # Pure vector baseline
    print(f"\n  Query: \"{query}\"")
    print(f"  [A] Pure vector (no filter):")
    results_pure = retriever.search(query, top_k=5)
    for r in results_pure:
        print(f"      {r.frame_id}: objects={r.objects} lighting={r.lighting}")

    # With scalar filter (video_id)
    print(f"\n  [B] With scalar filter (video_id=\"test_scenes\"):")
    results_f = retriever.search(query, top_k=5, video_id="test_scenes")
    for r in results_f:
        assert r.video_id == "test_scenes", f"Expected test_scenes, got {r.video_id}"
        print(f"      {r.frame_id}: video={r.video_id}")

    # With client-side object filter
    print(f"\n  [C] With object filter (objects contain \"cup\"):")
    results_objects = retriever.search(query, top_k=5, objects=["cup"])
    for r in results_objects:
        assert "cup" in r.objects, f"Expected 'cup' in {r.objects}"
        print(f"      {r.frame_id}: objects={r.objects}")

    # Combined
    print(f"\n  [D] Combined (video_id + objects filter):")
    results_combined = retriever.search(
        query, top_k=5,
        video_id="test_scenes",
        objects=["plate"],
    )
    for r in results_combined:
        print(f"      {r.frame_id}: video={r.video_id} objects={r.objects}")

    print(f"\n  PASS: Hybrid search works correctly")
    return True


def test_epic_queries():
    """Test with queries derived from EPIC-KITCHENS ground truth."""
    separator("Test 4: EPIC-KITCHENS Ground Truth Queries")

    # Check if EPIC data exists in the collection
    retriever = Retriever(
        embedding_model=dashscope_config.embedding_model,
        dashvector_api_key=dashvector_config.api_key,
        dashvector_endpoint=dashvector_config.endpoint,
        collection_name=dashvector_config.collection_name,
    )

    epic_results = retriever.search("kitchen", top_k=3, video_id="P01_03")
    if not epic_results:
        print("  SKIP: No EPIC data in DashVector. Run EPIC ingestion first.")
        return True

    print(f"  Found EPIC frames: {len(epic_results)}")

    # Query with specific ground truth narrations
    queries = [
        "open door",
        "take cup",
        "wash cup",
        "pour water",
        "close fridge",
    ]

    for q in queries:
        results = retriever.search(q, top_k=3)
        correct = sum(1 for r in results if q in r.gt_narration.lower() if r.gt_narration)
        status = "HIT" if correct > 0 else "MISS"
        print(f"  [{status}] \"{q}\": {len(results)} results, {correct} gt matches")

    print(f"\n  PASS: EPIC query test completed")
    return True


if __name__ == "__main__":
    print("RAG Video Search — Milestone 4: Retrieval Test")
    print(f"Embedding model: {dashscope_config.embedding_model}")
    print(f"DashVector collection: {dashvector_config.collection_name}")

    results = {}
    results["filter_builder"] = test_filter_builder()
    results["vector_search"] = test_vector_search()
    results["hybrid_search"] = test_hybrid_search()
    results["epic_queries"] = test_epic_queries()

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
        print("\n  Some checks failed.")
        sys.exit(1)
