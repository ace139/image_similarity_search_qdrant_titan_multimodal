"""
UI components for the search tab.
"""
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st

from mmfood.services import SearchService
from mmfood.ui.components import show_performance_metrics, display_search_results
from mmfood.qdrant.client import get_qdrant_client
from mmfood.config import AppConfig


def _env_bool(key: str, default: bool = False) -> bool:
    """Helper to read boolean from environment variable strings."""
    import os
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def render_search_tab(
    config: AppConfig, 
    search_service: SearchService,
    session_id: str
):
    """Render the search tab UI."""
    st.subheader("Search (User-specific)")
    
    # Debug mode checkbox
    debug_mode = st.checkbox(
        "Debug image fetch", 
        value=_env_bool("APP_DEBUG", False)
    )
    
    # Query input section
    query_text, query_image_bytes, query_image_filename = _render_query_inputs()
    
    # Filters section
    user_id_f, date_range, meal_types, top_k = _render_filters()
    
    # Search button and execution
    if st.button("Run Search", type="primary", key="run_search"):
        _execute_search(
            config, search_service, session_id, user_id_f, 
            query_text, query_image_bytes, query_image_filename,
            date_range, meal_types, top_k, debug_mode
        )


def _render_query_inputs():
    """Render query input section."""
    query_mode = st.radio(
        "Query type", ["Text", "Image"], horizontal=True, key="query_mode"
    )
    
    query_text = ""
    query_image_bytes: Optional[bytes] = None
    query_image_filename: Optional[str] = None
    
    if query_mode == "Text":
        query_text = st.text_input(
            "Search text", value="", key="search_text"
        ).strip()
    else:
        q_up = st.file_uploader(
            "Upload query image",
            type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
            key="search_image_uploader",
        )
        if q_up is not None:
            query_image_bytes = q_up.read()
            query_image_filename = q_up.name
            # No preview in Search tab by design
    
    return query_text, query_image_bytes, query_image_filename


def _render_filters():
    """Render search filters section."""
    st.markdown("**Filters**")
    
    user_id_f = st.text_input(
        "User ID (required)", value="", key="search_user_id"
    ).strip()
    
    # Date range defaults to last 7 days
    today = datetime.now().date()
    default_start = today - timedelta(days=6)
    date_range = st.date_input(
        "Date range", value=(default_start, today), key="date_range"
    )
    
    meal_types = st.multiselect(
        "Meal types",
        options=["breakfast", "lunch", "dinner", "snack", "other"],
        default=[],
        key="meal_types_filter",
    )
    
    top_k = st.number_input(
        "Results to return", min_value=1, max_value=20, value=5, step=1, key="top_k"
    )
    
    return user_id_f, date_range, meal_types, top_k


def _execute_search(
    config: AppConfig,
    search_service: SearchService,
    session_id: str,
    user_id_f: str,
    query_text: str,
    query_image_bytes: Optional[bytes],
    query_image_filename: Optional[str],
    date_range,
    meal_types,
    top_k: int,
    debug_mode: bool
):
    """Execute the search and display results."""
    # Validation
    if not user_id_f:
        st.error("Please provide a User ID for user-specific search.")
        st.stop()
    if not config.qdrant_url:
        st.error("Qdrant URL not configured. Please check your .env file.")
        st.stop()
    
    query_mode = "text" if query_text else "image"
    if (query_mode == "text" and not query_text) and (query_mode == "image" and not query_image_bytes):
        st.error("Provide a search text or upload a query image.")
        st.stop()

    with st.spinner("Embedding query and searching in Qdrant..."):
        # Execute search using service
        search_result = search_service.execute_search(
            user_id=user_id_f,
            query_mode=query_mode,
            query_text=query_text,
            query_image_bytes=query_image_bytes,
            query_image_filename=query_image_filename,
            date_range=date_range,
            meal_types=meal_types,
            top_k=top_k,
            session_id=session_id
        )
        
        if not search_result["success"]:
            st.error(f"Search failed: {search_result['error']}")
            return
        
        results = search_result["results"]
        s3_client = search_result["s3_client"]
        performance = search_result["performance"]
        
        if not results:
            st.info("No results.")
        else:
            st.success(f"Found {len(results)} result(s)")
            
            # Display performance metrics
            show_performance_metrics(performance)
            
            # Display results
            qdrant_client = get_qdrant_client(
                config.qdrant_url, 
                config.qdrant_api_key, 
                config.qdrant_timeout
            )
            
            display_search_results(
                results=results,
                s3_client=s3_client,
                qdrant_client=qdrant_client,
                qdrant_collection=config.qdrant_collection_name,
                debug_mode=debug_mode
            )
