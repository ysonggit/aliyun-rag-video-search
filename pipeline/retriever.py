"""
Hybrid retrieval: vector recall + metadata keyword filtering.

Architecture:
  - DashVector handles: vector ANN search + scalar field filters (lighting, etc.)
  - Python handles: array field filters (objects, actions) — contain_any/contain_all
    not supported on Singapore DashVector cluster.

Usage:
    retriever = Retriever()
    results = retriever.search("take cup from cupboard", top_k=10)
    results = retriever.search("wash dishes", top_k=10, objects=["cup", "spoon"])
"""

from functools import lru_cache
from http import HTTPStatus
from typing import List, Optional

import dashscope
from dashvector import Client

from pipeline.filter_builder import build_filter, FilterParams


@lru_cache(maxsize=256)
def _embed_text_cached(model: str, text: str) -> tuple:
    """Embed query text, memoized by (model, text). Returns a hashable tuple.

    Streamlit reruns the script on every widget interaction, re-embedding the
    same query text; this cache avoids the redundant API calls.
    """
    resp = dashscope.MultiModalEmbedding.call(model=model, input=[{"text": text}])
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Text embedding failed: {resp.code} - {resp.message}")
    return tuple(resp.output["embeddings"][0]["embedding"])


class SearchResult:
    """A single retrieval result."""

    def __init__(self, raw_output):
        self.frame_id: str = raw_output.id
        self.score: float = getattr(raw_output, "score", 0.0)
        fields = getattr(raw_output, "fields", {})

        self.video_id: str = fields.get("video_id", "")
        self.timestamp: float = fields.get("timestamp", 0.0)
        self.frame_path: str = fields.get("oss_path", "")
        self.objects: List[str] = fields.get("objects", [])
        self.actions: List[str] = fields.get("actions", [])
        self.lighting: str = fields.get("lighting", "")
        self.occlusion: str = fields.get("occlusion", "")
        self.is_anomaly: bool = fields.get("is_anomaly", False)
        self.category: str = fields.get("category", "")
        self.gt_narration: str = fields.get("gt_narration", "")
        self.scene_desc: str = fields.get("scene_desc", "")

    def __repr__(self):
        return (
            f"SearchResult(id={self.frame_id}, score={self.score:.4f}, "
            f"objects={self.objects}, actions={self.actions})"
        )


class Retriever:
    """
    Hybrid retriever: text query → embedding → DashVector search → client-side filter.

    Config:
        DASHSCOPE_API_KEY, DASHVECTOR_API_KEY, DASHVECTOR_ENDPOINT (env)
    """

    OUTPUT_FIELDS = [
        "video_id", "timestamp", "oss_path",
        "objects", "actions", "lighting", "occlusion", "is_anomaly",
        "category", "gt_narration", "scene_desc",
    ]

    def __init__(
        self,
        embedding_model: str = "tongyi-embedding-vision-plus",
        dashscope_api_key: str = "",
        dashvector_api_key: str = "",
        dashvector_endpoint: str = "",
        collection_name: str = "scene_frames",
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self._dv_client = Client(
            api_key=dashvector_api_key,
            endpoint=dashvector_endpoint,
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._dv_client.get(self.collection_name)
            if self._collection is None:
                raise RuntimeError(
                    f"Collection '{self.collection_name}' not found. Run ingestion first."
                )
        return self._collection

    def search(
        self,
        query: str,
        top_k: int = 10,
        # --- Server-side filters (DashVector scalar fields) ---
        lighting: Optional[str] = None,
        occlusion: Optional[str] = None,
        is_anomaly: Optional[bool] = None,
        category: Optional[str] = None,
        video_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        view_type: Optional[str] = None,
        # --- Client-side filters (Python post-filter on arrays) ---
        objects: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        require_all_objects: bool = False,
        require_all_actions: bool = False,
        # --- Tuning ---
        vector_fetch_multiplier: int = 3,
    ) -> List[SearchResult]:
        """
        Search frames by text query with metadata filters.

        Scalar filters (lighting, occlusion, etc.) are applied server-side via DashVector.
        Array filters (objects, actions) are applied client-side in Python.

        Args:
            query: Natural language search query.
            top_k: Number of final results to return (after all filtering).
            lighting/occlusion/category/video_id/episode_id/view_type: Scalar filters.
            is_anomaly: Filter by anomaly status.
            objects: Client-side filter — frames must contain these objects.
            actions: Client-side filter — frames must contain these actions.
            require_all_objects/actions: If True, ALL specified values must be present.
            vector_fetch_multiplier: Fetch N*top_k from DashVector to allow room
                                     for client-side filtering.

        Returns:
            List of SearchResult, sorted by similarity score.
        """
        # Build filter params (split server vs client)
        fp = build_filter(
            objects=objects,
            actions=actions,
            lighting=lighting,
            occlusion=occlusion,
            is_anomaly=is_anomaly,
            category=category,
            video_id=video_id,
            episode_id=episode_id,
            view_type=view_type,
            require_all_objects=require_all_objects,
            require_all_actions=require_all_actions,
        )

        # Convert text query to embedding
        query_vec = self._embed_text(query)
        collection = self.collection

        def run_query(fetch_k: int):
            kwargs = {
                "vector": query_vec,
                "topk": fetch_k,
                "output_fields": self.OUTPUT_FIELDS,
            }
            if fp.server_filter:
                kwargs["filter"] = fp.server_filter
            rsp = collection.query(**kwargs)
            if not rsp:
                raise RuntimeError(f"Query failed: {rsp.code} - {rsp.message}")
            raw = rsp.output or []
            res = [SearchResult(r) for r in raw]
            if fp.has_client_filter:
                res = self._apply_array_filters(res, fp)
            return res, len(raw)

        # Over-fetch when array filters will prune client-side, then truncate.
        fetch_k = top_k * vector_fetch_multiplier if fp.has_client_filter else top_k
        results, raw_count = run_query(fetch_k)

        # Adaptive backfill: hard object/action filters can prune below top_k.
        # Refetch once with a larger window — but only if the first query came back
        # saturated (more candidates likely exist beyond what we fetched).
        if fp.has_client_filter and len(results) < top_k and raw_count >= fetch_k:
            bigger = min(fetch_k * 4, 200)
            if bigger > fetch_k:
                results, _ = run_query(bigger)

        return results[:top_k]

    def search_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        filter_str: str = "",
    ) -> List[SearchResult]:
        """Search with a pre-computed embedding vector."""
        collection = self.collection
        kwargs = {
            "vector": vector,
            "topk": top_k,
            "output_fields": self.OUTPUT_FIELDS,
        }
        if filter_str:
            kwargs["filter"] = filter_str

        rsp = collection.query(**kwargs)
        if not rsp:
            raise RuntimeError(f"Query failed: {rsp.code} - {rsp.message}")

        return [SearchResult(raw) for raw in (rsp.output or [])]

    def _embed_text(self, text: str) -> List[float]:
        """Embed a text query using the configured model (memoized)."""
        return list(_embed_text_cached(self.embedding_model, text))

    @staticmethod
    def _apply_array_filters(
        results: List[SearchResult],
        fp: FilterParams,
    ) -> List[SearchResult]:
        """Client-side filtering on array fields."""
        filtered = []

        for r in results:
            # Objects filter
            if fp.objects:
                if fp.require_all_objects:
                    if not all(obj in r.objects for obj in fp.objects):
                        continue
                else:
                    if not any(obj in r.objects for obj in fp.objects):
                        continue

            # Actions filter
            if fp.actions:
                if fp.require_all_actions:
                    if not all(act in r.actions for act in fp.actions):
                        continue
                else:
                    if not any(act in r.actions for act in fp.actions):
                        continue

            filtered.append(r)

        return filtered

    def format_results(self, results: List[SearchResult], max_display: int = 5) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."

        lines = [f"Top {min(len(results), max_display)} results:"]
        for i, r in enumerate(results[:max_display]):
            gt_info = f"  GT: [{r.gt_narration}]" if r.gt_narration else ""
            lines.append(
                f"  #{i+1}  score={r.score:.4f}  {r.frame_id}  "
                f"t={r.timestamp:.1f}s  video={r.video_id}"
            )
            lines.append(f"       objects={r.objects}  actions={r.actions}")
            lines.append(f"       lighting={r.lighting}  occlusion={r.occlusion}"
                         f"{gt_info}")

        return "\n".join(lines)
