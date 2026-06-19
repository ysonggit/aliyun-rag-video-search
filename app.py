"""
RAG Video Search — Streamlit UI

Usage:
  streamlit run app.py

Requires:
  pip install streamlit
  .env with DASHSCOPE_API_KEY, DASHVECTOR_API_KEY, DASHVECTOR_ENDPOINT
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load env
load_dotenv(Path(__file__).resolve().parent / ".env")

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import dashscope_config, dashvector_config
from pipeline.retriever import Retriever, SearchResult


# ── Page config ────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Video Search",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
  .frame-card {
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 10px;
    background: #161b22;
  }
  .frame-card img {
    border-radius: 6px;
    width: 100%;
    max-height: 180px;
    object-fit: cover;
  }
  .frame-score {
    color: #3fb950;
    font-weight: 600;
    font-size: 0.9em;
  }
  .frame-meta {
    color: #8b949e;
    font-size: 0.75em;
    line-height: 1.4;
  }
  .frame-video {
    color: #58a6ff;
    font-size: 0.7em;
    font-weight: 500;
  }
  .result-count {
    color: #8b949e;
    font-size: 0.85em;
    margin-bottom: 12px;
  }
  .detail-header {
    color: #79c0ff;
    font-weight: 600;
    margin-bottom: 4px;
  }
  @media (max-width: 768px) {
    .frame-card img { max-height: 140px; }
  }
</style>
""", unsafe_allow_html=True)


# ── Init retriever (cached) ────────────────────────────────

@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(
        embedding_model=dashscope_config.embedding_model,
        dashvector_api_key=dashvector_config.api_key,
        dashvector_endpoint=dashvector_config.endpoint,
        collection_name=dashvector_config.collection_name,
    )


# ── Helper: get available filter values ────────────────────

@st.cache_data(ttl=300)
def get_filter_options() -> dict:
    """Query DashVector to discover available filter values."""
    retriever = get_retriever()
    try:
        results = retriever.search("scene", top_k=50)
    except Exception:
        return {"videos": [], "lighting": [], "objects": [], "category": []}

    videos = sorted(set(r.video_id for r in results if r.video_id))
    lightings = sorted(set(r.lighting for r in results if r.lighting))
    objects = sorted(set(obj for r in results for obj in r.objects))
    categories = sorted(set(r.category for r in results if r.category))

    return {
        "videos": videos,
        "lighting": lightings,
        "objects": objects,
        "category": categories,
    }


# ── Layout ─────────────────────────────────────────────────

st.title("RAG Video Search")
st.caption(f"Collection: `{dashvector_config.collection_name}`  |  Model: `{dashscope_config.embedding_model}`")

# Sidebar
with st.sidebar:
    st.header("Filters")

    st.subheader("Semantic")
    top_k = st.slider("Top K", 3, 50, 10)

    st.subheader("Scalar Filters")
    options = get_filter_options()

    selected_video = st.selectbox(
        "Video ID",
        ["(any)"] + options["videos"],
        index=0,
    )
    selected_lighting = st.selectbox(
        "Lighting",
        ["(any)"] + options["lighting"],
        index=0,
    )
    selected_category = st.selectbox(
        "Category",
        ["(any)"] + options["category"],
        index=0,
    )
    is_anomaly = st.selectbox(
        "Anomaly",
        ["(any)", "true", "false"],
        index=0,
    )

    st.subheader("Object Filters")
    selected_objects = st.multiselect(
        "Objects (contain any)",
        options["objects"],
        default=[],
        help="Only show frames containing these objects",
    )
    require_all = st.checkbox("Require ALL objects", value=False)

    if st.button("Clear Filters", use_container_width=True):
        st.rerun()


# ── Search bar ─────────────────────────────────────────────

query = st.text_input(
    "Search query",
    placeholder='e.g. "take cup from cupboard", "wash dishes", "pour water" ...',
    key="search_input",
)

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col2:
    show_all = st.button("Show all frames", use_container_width=True)


# ── Search logic ───────────────────────────────────────────

if search_clicked and query.strip():
    with st.spinner(f"Searching: \"{query}\" ..."):
        try:
            retriever = get_retriever()

            results = retriever.search(
                query.strip(),
                top_k=top_k,
                video_id=selected_video if selected_video != "(any)" else None,
                lighting=selected_lighting if selected_lighting != "(any)" else None,
                category=selected_category if selected_category != "(any)" else None,
                is_anomaly=(
                    True if is_anomaly == "true"
                    else False if is_anomaly == "false"
                    else None
                ),
                objects=selected_objects if selected_objects else None,
                require_all_objects=require_all,
            )

            st.session_state["results"] = results
            st.session_state["query"] = query.strip()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.session_state["results"] = []

elif show_all:
    with st.spinner("Loading all frames..."):
        try:
            retriever = get_retriever()
            results = retriever.search("scene", top_k=top_k)
            st.session_state["results"] = results
            st.session_state["query"] = "(all frames)"
        except Exception as e:
            st.error(f"Failed: {e}")
            st.session_state["results"] = []


# ── Display results ────────────────────────────────────────

results = st.session_state.get("results", [])
saved_query = st.session_state.get("query", "")

if saved_query:
    st.markdown(f'<p class="result-count">Query: <strong>{saved_query}</strong> — {len(results)} results</p>', unsafe_allow_html=True)

if results:
    # Grid: 4 columns
    cols = st.columns(4)

    for i, r in enumerate(results):
        col = cols[i % 4]

        with col:
            # Build card
            img_exists = Path(r.frame_path).exists() if r.frame_path else False

            if img_exists:
                col.image(str(r.frame_path), use_container_width=True)
            else:
                col.markdown(
                    '<div style="height:180px;background:#1c2128;border-radius:6px;'
                    'display:flex;align-items:center;justify-content:center;'
                    'color:#484f58;font-size:0.85em;">No image</div>',
                    unsafe_allow_html=True,
                )

            col.markdown(
                f'<p class="frame-score">{r.score:.4f}</p>'
                f'<p class="frame-video">{r.frame_id}</p>'
                f'<p class="frame-meta">'
                f't={r.timestamp:.1f}s · {r.video_id}<br>'
                f'objects: {", ".join(r.objects[:4]) or "—"}<br>'
                f'actions: {", ".join(r.actions[:4]) or "—"}<br>'
                f'lighting: {r.lighting} · occlusion: {r.occlusion}'
                f'</p>',
                unsafe_allow_html=True,
            )

    # ── Detail expander ────────────────────────────────────
    if len(results) > 0:
        with st.expander("Details — click a frame to inspect", expanded=False):
            frame_ids = [r.frame_id for r in results]
            selected_frame = st.selectbox("Select frame", frame_ids)

            for r in results:
                if r.frame_id == selected_frame:
                    dcol1, dcol2 = st.columns([1, 2])

                    with dcol1:
                        if Path(r.frame_path).exists():
                            st.image(str(r.frame_path), use_container_width=True)

                    with dcol2:
                        st.markdown(f'<p class="detail-header">Frame: {r.frame_id}</p>', unsafe_allow_html=True)
                        st.json({
                            "score": r.score,
                            "video_id": r.video_id,
                            "timestamp": r.timestamp,
                            "objects": r.objects,
                            "actions": r.actions,
                            "lighting": r.lighting,
                            "occlusion": r.occlusion,
                            "is_anomaly": r.is_anomaly,
                            "category": r.category,
                            "gt_narration": r.gt_narration,
                        })

                    break

elif saved_query:
    st.info("No results found. Try a different query or adjust filters.")

else:
    # Welcome screen
    st.markdown("""
    ### Explore your EPIC-KITCHENS video index

    **Try these queries:**
    - `take cup from cupboard`
    - `wash dishes in sink`
    - `pour water into glass`
    - `open fridge door`
    - `close cupboard`

    Or click **"Show all frames"** to browse everything.

    ---
    **How it works:**
    1. Your text query → DashScope embedding (1152-dim vector)
    2. Vector similarity search in DashVector
    3. Metadata filters refine results
    4. Top frames displayed with their Qwen-VL scene captions
    """)
