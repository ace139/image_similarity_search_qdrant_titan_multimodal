"""
Shared UI components for the application.
"""
import io
from typing import Dict, Any, List, Optional
import streamlit as st
from PIL import Image
from botocore.exceptions import ClientError

from mmfood.aws.s3 import get_object_bytes_and_meta, presign_url
from mmfood.qdrant.operations import delete_vector


def show_performance_metrics(performance: Dict[str, float]):
    """Display performance metrics in an expandable section."""
    with st.expander("🔍 Search Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{performance.get('total_duration_ms', 0):.1f} ms")
        with col2:
            st.metric("Embedding Time", f"{performance.get('embedding_duration_ms', 0):.1f} ms")
        with col3:
            st.metric("Search Time", f"{performance.get('search_duration_ms', 0):.1f} ms")


def show_ingestion_performance(description_ms=None, embedding_ms=None, s3_image_upload_ms=None, s3_embedding_upload_ms=None, vector_index_ms=None):
    """Display ingestion performance metrics inline (non-expandable).

    Parameters are optional; metrics that are None will be omitted.
    """
    st.markdown("### ⚡ Ingestion Performance")
    # Build a list of (label, value) for non-null metrics in a sensible order
    items: List[tuple[str, Optional[float]]] = [
        ("Description Generation", description_ms),
        ("Embedding Generation", embedding_ms),
        ("S3 Image Upload", s3_image_upload_ms),
        ("S3 Embedding Upload", s3_embedding_upload_ms),
        ("Vector Indexing", vector_index_ms),
    ]
    # Filter out Nones
    items = [(label, val) for (label, val) in items if val is not None]
    if not items:
        st.write("No ingestion performance data available.")
        return
    # Render in rows of 3 for compactness
    for i in range(0, len(items), 3):
        row = items[i:i+3]
        cols = st.columns(len(row))
        for col, (label, val) in zip(cols, row):
            with col:
                st.metric(label, f"{float(val):.1f} ms")


def display_search_results(
    results: List[Dict[str, Any]], 
    s3_client: Any,
    qdrant_client: Any,
    qdrant_collection: str,
    debug_mode: bool = False,
    columns: int = 3,
):
    """Display search results in a responsive grid with per-item details.

    Parameters:
    - results: List of Qdrant search result items
    - s3_client, qdrant_client, qdrant_collection: dependencies for image and cleanup
    - debug_mode: when True, shows extra fetch logs
    - columns: number of cards per row
    """
    columns = max(1, int(columns or 3))
    cols = st.columns(columns)

    for idx, item in enumerate(results):
        vector_id = item.get("id")
        payload = item.get("payload", {})
        score = item.get("score")
        img_bucket = payload.get("s3_bucket")
        img_key = payload.get("s3_image_key")

        with cols[idx % columns]:
            # Image (with cleanup controls embedded by _display_result_image)
            if img_bucket and img_key:
                _display_result_image(
                    s3_client, img_bucket, img_key, vector_id,
                    payload, qdrant_client, qdrant_collection, debug_mode
                )
            else:
                st.write("(no image metadata)")

            # Basic metadata
            st.write(f"Key: {vector_id}")
            if score is not None:
                st.write(f"Similarity Score: {score:.4f}")

            # Full payload in an expander to save vertical space
            with st.expander("Details"):
                st.json(payload)


def _display_result_image(
    s3_client: Any,
    img_bucket: str,
    img_key: str,
    vector_id: str,
    payload: Dict[str, Any],
    qdrant_client: Any,
    qdrant_collection: str,
    debug_mode: bool
):
    """Display a single result image with error handling."""
    if debug_mode:
        st.write(f"Attempting image fetch: s3://{img_bucket}/{img_key}")
    
    # Prefer server-side fetch
    try:
        data, meta_info = get_object_bytes_and_meta(s3_client, img_bucket, img_key)
        if debug_mode:
            st.write({
                "len_bytes": len(data),
                **(meta_info or {})
            })
        
        # Try PIL decode first for reliability
        try:
            img_obj = Image.open(io.BytesIO(data))
            if debug_mode:
                st.write({"pil_format": img_obj.format, "size": img_obj.size})
            st.image(img_obj, caption=f"{vector_id}", use_container_width=True)
        except Exception as dec_err:
            if debug_mode:
                st.warning(f"PIL decode failed: {dec_err}")
            st.image(data, caption=f"{vector_id}", use_container_width=True)
            
    except Exception as fetch_err:
        if debug_mode:
            st.warning(f"Direct S3 fetch failed: {fetch_err}")
        
        # Check if the object is missing (404/NoSuchKey)
        missing = _is_missing_object_error(fetch_err)
        
        if missing:
            st.warning("Image object not found in S3. This looks like an **orphaned vector** (image deleted after indexing).")
            if st.button("🧹 Delete this vector & JSON", key=f"del_{vector_id}"):
                _cleanup_orphaned_vector(
                    vector_id, payload, qdrant_client, qdrant_collection, 
                    s3_client, img_bucket
                )
        else:
            # Fallback to a presigned URL if not a missing-object case
            _try_presigned_url_display(
                s3_client, img_bucket, img_key, vector_id, debug_mode
            )


def _is_missing_object_error(fetch_err: Exception) -> bool:
    """Check if the error indicates a missing S3 object."""
    missing = False
    try:
        if isinstance(fetch_err, ClientError):
            code = fetch_err.response.get("Error", {}).get("Code")
            status = fetch_err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            missing = (code in {"404", "NoSuchKey", "NotFound"}) or (status == 404)
    except Exception:
        pass
    return missing


def _cleanup_orphaned_vector(
    vector_id: str,
    payload: Dict[str, Any],
    qdrant_client: Any,
    qdrant_collection: str,
    s3_client: Any,
    img_bucket: str
):
    """Clean up an orphaned vector and its associated JSON."""
    try:
        if vector_id is None:
            st.error("Vector ID is missing, cannot delete.")
            return
        
        # Delete from Qdrant
        delete_success = delete_vector(qdrant_client, qdrant_collection, str(vector_id))
        
        # Delete embedding JSON if present
        emb_key = payload.get("s3_embedding_key")
        if emb_key:
            try:
                s3_client.delete_object(Bucket=img_bucket, Key=emb_key)
            except Exception as e:
                print(f"[cleanup] delete_object failed for s3://{img_bucket}/{emb_key}: {e}")
        
        if delete_success:
            st.success("Deleted vector and attempted to remove JSON artifact. Re-run the search.")
        else:
            st.error("Failed to delete vector from Qdrant")
            
    except Exception as del_err:
        st.error(f"Cleanup failed: {del_err}")


def _try_presigned_url_display(
    s3_client: Any,
    img_bucket: str,
    img_key: str,
    vector_id: str,
    debug_mode: bool
):
    """Try to display image using presigned URL as fallback."""
    try:
        url = presign_url(s3_client, img_bucket, img_key, expires_in=3600)
        if debug_mode:
            st.write({"presigned_url": url[:80] + "..."})
        st.image(url, caption=f"{vector_id}", use_container_width=True)
    except Exception as url_err:
        if debug_mode:
            st.error(f"Presigned URL display failed: {url_err}")
        st.write(f"Image: s3://{img_bucket}/{img_key}")


def display_upload_details(upload_details: Dict[str, Any]):
    """Display upload completion details."""
    st.success("✅ Upload complete!")
    st.write("**Upload Details:**")
    st.write(f"• S3 Image Key: `{upload_details['image_key']}`")
    st.write(f"• S3 Embedding Key: `{upload_details['vector_key']}`")
    st.write(f"• Qdrant Collection: `{upload_details['collection']}`")
    st.write(f"• Vector ID: `{upload_details['image_id']}`")
