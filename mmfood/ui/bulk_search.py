"""
Bulk Search UI tab: query the bulk Qdrant collection with minimal controls.
"""
from typing import Optional

import io
import streamlit as st
from PIL import Image

from mmfood.config import AppConfig
from mmfood.services import SearchService
from mmfood.ui.components import show_performance_metrics, display_search_results
from mmfood.qdrant.client import get_qdrant_client
from mmfood.database import MetricsDatabase

_MAX_SIDE = 1280
_MAX_PIXELS = 2_000_000


def _compute_downscale_dims(w: int, h: int, max_side: int = _MAX_SIDE, max_pixels: int = _MAX_PIXELS):
    if w <= 0 or h <= 0:
        return w, h
    scale_factors = []
    if max(w, h) > max_side:
        scale_factors.append(max_side / float(max(w, h)))
    if (w * h) > max_pixels:
        scale_factors.append((max_pixels / float(w * h)) ** 0.5)
    if not scale_factors:
        return w, h
    scale = min(scale_factors)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def render_bulk_search_tab(config: AppConfig, search_service: SearchService, session_id: str):
    st.subheader("Bulk Search")

    if not config.qdrant_bulk_collection_name:
        st.error("QDRANT_COLLECTION_NAME_BULK is not configured. Please set it in your .env file.")
        st.stop()

    # Query inputs
    mode = st.radio("Query type", ["Text", "Image"], horizontal=True, key="bulk_query_mode")

    query_text = ""
    query_image_bytes: Optional[bytes] = None
    query_image_filename: Optional[str] = None

    if mode == "Text":
        query_text = st.text_input("Search text", value="", key="bulk_search_text").strip()
    else:
        up = st.file_uploader("Upload query image", type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"], key="bulk_search_image")
        if up is not None:
            query_image_bytes = up.read()
            query_image_filename = up.name
            try:
                with Image.open(io.BytesIO(query_image_bytes)) as img:
                    w, h = img.size
                    new_w, new_h = _compute_downscale_dims(w, h)
                    if (new_w, new_h) != (w, h):
                        st.info(f"Query image will be downscaled from {w}x{h} to {new_w}x{new_h}.")
            except Exception:
                pass

    # Controls
    top_k = st.number_input("Results to return", min_value=1, max_value=20, value=8, step=1, key="bulk_top_k")
    score_threshold = st.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.01, key="bulk_score_threshold")

    if st.button("Run Bulk Search", type="primary", key="run_bulk_search"):
        if (mode == "Text" and not query_text) and (mode == "Image" and not query_image_bytes):
            st.error("Provide a search text or upload a query image.")
            st.stop()

        with st.spinner("Embedding query and searching bulk collection..."):
            # Execute search against bulk collection; reuse service
            result = search_service.execute_search(
                user_id=(config.bulk_user_id or "999999"),
                query_mode=("text" if mode == "Text" else "image"),
                query_text=query_text,
                query_image_bytes=query_image_bytes,
                query_image_filename=query_image_filename,
                date_range=None,
                meal_types=None,
                top_k=int(top_k),
                session_id=session_id,
                target_collection=config.qdrant_bulk_collection_name,
                score_threshold=float(score_threshold),
            )

            # Log bulk search request
            metrics_db = MetricsDatabase()
            if not result["success"]:
                metrics_db.log_bulk_search_request({
                    "query_type": ("text" if mode == "Text" else "image"),
                    "top_k": int(top_k),
                    "score_threshold": float(score_threshold),
                    "duration_ms_total": result.get("performance", {}).get("total_duration_ms", 0.0),
                    "duration_ms_embedding": result.get("performance", {}).get("embedding_duration_ms", 0.0),
                    "duration_ms_search": result.get("performance", {}).get("search_duration_ms", 0.0),
                    "results_count": 0,
                    "success": 0,
                    "error_message": result.get("error"),
                })
                st.error(f"Search failed: {result['error']}")
                return

            results = result["results"]
            s3_client = result["s3_client"]
            performance = result["performance"]

            metrics_db.log_bulk_search_request({
                "query_type": ("text" if mode == "Text" else "image"),
                "top_k": int(top_k),
                "score_threshold": float(score_threshold),
                "duration_ms_total": performance.get("total_duration_ms", 0.0),
                "duration_ms_embedding": performance.get("embedding_duration_ms", 0.0),
                "duration_ms_search": performance.get("search_duration_ms", 0.0),
                "results_count": len(results),
                "success": 1,
                "error_message": None,
            })

            if not results:
                st.info("No results.")
                return

            st.success(f"Found {len(results)} result(s)")
            show_performance_metrics(performance)

            qdrant_client = get_qdrant_client(
                config.qdrant_url,
                config.qdrant_api_key,
                config.qdrant_timeout,
            )
            display_search_results(
                results=results,
                s3_client=s3_client,
                qdrant_client=qdrant_client,
                qdrant_collection=config.qdrant_bulk_collection_name,
                debug_mode=False,
                columns=3,
                thumb_width=320,
            )
