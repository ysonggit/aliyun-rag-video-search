"""
Unit tests for the ingestion/query performance changes.

These run WITHOUT cloud credentials or heavy deps (cv2/dashscope/dashvector/oss2)
by stubbing those modules and loading the target pipeline modules directly,
bypassing pipeline/__init__ (which imports cv2).

Run:
    python tests/test_perf.py
"""

import sys
import types
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Stub heavy third-party deps before importing pipeline modules ──────────────

def _make_resp(ok=True, embeddings=None, output=None, usage=None):
    class _Resp:
        status_code = 200 if ok else 500
        code = "OK" if ok else "Throttling"
        message = "" if ok else "rate limited"
        def __init__(self):
            self.output = output if output is not None else {"embeddings": embeddings or []}
            self.usage = usage if usage is not None else {"total_tokens": 10}
        def __bool__(self):  # dashvector responses are checked with `if not rsp`
            return ok
    return _Resp()


_dashscope = types.ModuleType("dashscope")
_dashscope.api_key = ""
_dashscope.base_http_api_url = ""
_dashscope.MultiModalEmbedding = types.SimpleNamespace(call=lambda **k: _make_resp())
_dashscope.MultiModalConversation = types.SimpleNamespace(call=lambda **k: _make_resp())
sys.modules["dashscope"] = _dashscope

_dashvector = types.ModuleType("dashvector")
class _Doc:
    def __init__(self, id=None, vector=None, fields=None):
        self.id = id; self.vector = vector; self.fields = fields or {}
_dashvector.Doc = _Doc
_dashvector.Client = object  # replaced per-test
sys.modules["dashvector"] = _dashvector

# Synthetic 'pipeline' package mapped to the real dir, so submodule imports load
# the real source files but pipeline/__init__.py (cv2) never runs.
_pkg = types.ModuleType("pipeline")
_pkg.__path__ = [str(ROOT / "pipeline")]
sys.modules["pipeline"] = _pkg

from pipeline.frame_metadata import FrameMetadata
from pipeline import oss_uploader, embedder as embmod, indexer as idxmod, retriever as retmod


def _frames(n):
    return [
        FrameMetadata(
            video_id="P01_03", frame_id=f"P01_03_f{i:06d}",
            frame_path=f"/tmp/f{i}.jpg", timestamp=float(i), frame_number=i,
        )
        for i in range(n)
    ]


# ── Test 1: parallel OSS upload ────────────────────────────────────────────────

def test_upload_parallel():
    threads = set(); lock = threading.Lock()

    class FakeBucket:
        endpoint = "https://oss-ap-southeast-1.aliyuncs.com"
        bucket_name = "rag-videos-test"
        def put_object_from_file(self, key, path):
            with lock:
                threads.add(threading.current_thread().name)

    frames = _frames(20)
    oss_uploader.upload_frames(frames, FakeBucket(), prefix="frames", max_workers=4)

    assert all(f.frame_path.startswith("https://rag-videos-test.") for f in frames), "URLs rewritten"
    assert all(f.frame_id in f.frame_path for f in frames)
    assert len(threads) > 1, "should upload concurrently"
    print(f"  test_upload_parallel OK ({len(threads)} threads)")


# ── Test 2: concurrent embedder + failure propagation ──────────────────────────

def test_embedder_concurrent():
    threads = set(); lock = threading.Lock()

    def fake_call(**kwargs):
        with lock:
            threads.add(threading.current_thread().name)
        n = len(kwargs["input"])
        return _make_resp(embeddings=[{"embedding": [0.1] * 4} for _ in range(n)])

    _dashscope.MultiModalEmbedding.call = fake_call
    emb = embmod.Embedder(api_key="x", max_batch_size=8, max_workers=4)
    emb._encode_image = staticmethod(lambda p: "b64")  # avoid disk

    frames = _frames(25)  # 4 batches of 8/8/8/1
    emb.embed_frames(frames)
    assert all(f.embedding == [0.1] * 4 for f in frames), "all frames embedded"
    assert len(threads) > 1, "batches should run concurrently"

    # Failure path: non-OK response raises after retries (no backoff sleep)
    _dashscope.MultiModalEmbedding.call = lambda **k: _make_resp(ok=False)
    emb2 = embmod.Embedder(api_key="x", max_retries=1, retry_delay=0, max_workers=2)
    emb2._encode_image = staticmethod(lambda p: "b64")
    raised = False
    try:
        emb2.embed_frames(_frames(3))
    except RuntimeError:
        raised = True
    assert raised, "embedding failure should raise"
    print(f"  test_embedder_concurrent OK ({len(threads)} threads)")


# ── Test 3: embed + caption overlap writes disjoint fields ──────────────────────

def test_embed_caption_overlap():
    from concurrent.futures import ThreadPoolExecutor
    from pipeline import captions as capmod

    _dashscope.MultiModalEmbedding.call = lambda **k: _make_resp(
        embeddings=[{"embedding": [0.5] * 4} for _ in range(len(k["input"]))]
    )
    emb = embmod.Embedder(api_key="x", max_workers=3)
    emb._encode_image = staticmethod(lambda p: "b64")

    cap = capmod.Captioner(api_key="x", max_workers=3)
    cap._caption_single = lambda path: {
        "objects": ["cup"], "actions": ["take"],
        "lighting": "bright", "occlusion": "none", "is_anomaly": False,
    }

    frames = _frames(15)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fe = pool.submit(emb.embed_frames, frames)
        fc = pool.submit(cap.caption_frames, frames)
        fe.result(); fc.result()

    assert all(f.embedding == [0.5] * 4 for f in frames), "embed field set"
    assert all(f.objects == ["cup"] for f in frames), "caption field set"
    print("  test_embed_caption_overlap OK")


# ── Test 4: indexer batch checkpoint callback ──────────────────────────────────

def test_indexer_checkpoint():
    upserted = []
    class FakeCollection:
        def upsert(self, docs):
            return _make_resp()  # truthy
    idx = idxmod.Indexer(collection_name="c", dimension=4)
    idx._collection = FakeCollection()

    frames = _frames(120)
    for f in frames:
        f.embedding = [0.0] * 4

    seen = []
    n = idx.index(frames, batch_size=50, on_batch_indexed=lambda ids: seen.extend(ids))

    assert n == 120, "all upserted"
    assert seen == [f.frame_id for f in frames], "callback receives every indexed id in order"
    print(f"  test_indexer_checkpoint OK ({len(seen)} ids checkpointed)")


# ── Test 5: query embedding cache + adaptive backfill ──────────────────────────

def test_retriever_cache_and_backfill():
    calls = {"embed": 0, "query": 0}

    def fake_embed(**kwargs):
        calls["embed"] += 1
        return _make_resp(embeddings=[{"embedding": [0.2] * 4}])
    _dashscope.MultiModalEmbedding.call = fake_embed

    # Fake DashVector: first (small) query returns 6 docs all matching → after the
    # 'cup' filter, but only 2 contain 'cup'; saturated (raw==fetch_k) so it refetches
    # a bigger window that returns 12 docs, 10 with 'cup'.
    class Raw:
        def __init__(self, i, has_cup):
            self.id = f"f{i}"; self.score = 1.0 - i * 0.01
            self.fields = {"objects": ["cup"] if has_cup else ["plate"], "actions": []}
    class FakeCollection:
        def query(self, **kwargs):
            calls["query"] += 1
            topk = kwargs["topk"]
            if topk <= 30:   # first fetch: top_k(10)*3 = 30, saturated, few matches
                return _make_resp(output=[Raw(i, has_cup=(i < 2)) for i in range(topk)])
            return _make_resp(output=[Raw(i, has_cup=(i < 10)) for i in range(topk)])
    class FakeClient:
        def __init__(self, **k): pass
        def get(self, name): return FakeCollection()
    retmod.Client = FakeClient  # retriever bound `Client` at import time

    r = retmod.Retriever(embedding_model="m", collection_name="c")

    # Cache: same query text embeds once
    base = calls["embed"]
    r._embed_text("unique-query-string-A")
    r._embed_text("unique-query-string-A")
    assert calls["embed"] == base + 1, "repeat query text must hit the cache"

    # Backfill: hard object filter prunes below top_k → second query fires
    calls["query"] = 0
    results = r.search("scene", top_k=10, objects=["cup"])
    assert calls["query"] == 2, "should refetch with a larger window"
    assert len(results) == 10, "backfill fills the page"
    assert all("cup" in res.objects for res in results)
    print("  test_retriever_cache_and_backfill OK")


if __name__ == "__main__":
    test_upload_parallel()
    test_embedder_concurrent()
    test_embed_caption_overlap()
    test_indexer_checkpoint()
    test_retriever_cache_and_backfill()
    print("\nAll perf tests passed.")
