"""
Bulk Ingest UI tab: sequential multi-file embedding + S3 upload, then single-shot Qdrant upsert.
"""
import io
from typing import List
from datetime import datetime

import streamlit as st
from PIL import Image

from mmfood.config import AppConfig
from mmfood.services import IngestService
from mmfood.database import MetricsDatabase, MetricsTimer
from mmfood.qdrant.client import get_qdrant_client, ensure_collection_exists, validate_collection_config
from mmfood.qdrant.operations import upsert_vectors_batch
from mmfood.ui.components import show_ingestion_performance


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


def render_bulk_ingest_tab(config: AppConfig, ingest_service: IngestService):
    st.subheader("Bulk Ingest")

    if not config.qdrant_bulk_collection_name:
        st.error("QDRANT_COLLECTION_NAME_BULK is not configured. Please set it in your .env file.")
        st.stop()

    st.caption("Upload multiple images; we will generate embeddings sequentially and upsert all points to Qdrant in one shot.")

    # Options
    fast_path = st.checkbox("Skip description (faster, image-only embedding)", value=False)

    uploads = st.file_uploader(
        "Select images for bulk ingest",
        type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
        accept_multiple_files=True,
        key="bulk_ingest_uploader",
    )

    if not uploads:
        st.info("Select one or more images to begin.")
        return

    if st.button("Start Bulk Ingest", type="primary"):
        _run_bulk_ingest(config, ingest_service, uploads, fast_path)


def _run_bulk_ingest(config: AppConfig, ingest_service: IngestService, uploads: List, fast_path: bool):
    metrics_db = MetricsDatabase()
    total_timer = MetricsTimer()

    # Prepare Qdrant client and ensure bulk collection exists
    qdrant = get_qdrant_client(config.qdrant_url, config.qdrant_api_key, config.qdrant_timeout)
    ensure_collection_exists(qdrant, config.qdrant_bulk_collection_name, config.output_dim)
    validate_collection_config(qdrant, config.qdrant_bulk_collection_name, config.output_dim)

    # Progress UI
    prog = st.progress(0)
    status = st.empty()

    points = []
    succeeded = 0
    failed = 0

    desc_times: List[float] = []
    embed_times: List[float] = []
    s3img_times: List[float] = []
    s3json_times: List[float] = []

    with total_timer:
        for idx, uploaded in enumerate(uploads, start=1):
            try:
                status.info(f"Processing {uploaded.name} ({idx}/{len(uploads)})")
                image_bytes = uploaded.read()

                # Optional preview and downscale notice
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    w, h = img.size
                    new_w, new_h = _compute_downscale_dims(w, h)
                    if (new_w, new_h) != (w, h):
                        st.info(f"{uploaded.name}: will be downscaled from {w}x{h} to {new_w}x{new_h} for Bedrock limits.")
                except Exception:
                    pass

                # Step 1: description + embedding
                if fast_path:
                    # Image-only embedding: pass empty text and image bytes
                    desc_ms = 0.0
                    display_text = ""
                    # Use service's Bedrock client via generate_mm_embedding
                    desc_text = ""
                    from mmfood.aws.session import get_bedrock_client
                    from mmfood.bedrock.ai import generate_mm_embedding
                    bedrock = get_bedrock_client(config.region, config.profile)
                    et = MetricsTimer()
                    with et:
                        embedding = generate_mm_embedding(
                            bedrock_client=bedrock,
                            model_id=config.model_id,
                            output_dim=config.output_dim,
                            input_text=desc_text,
                            input_image_bytes=image_bytes,
                        )
                    embed_ms = et.duration_ms
                else:
                    display_text, embedding, desc_ms, embed_ms = ingest_service.generate_description_and_embedding(
                        image_bytes, {"meal_type": "bulk"}
                    )
                desc_times.append(desc_ms)
                embed_times.append(embed_ms)

                # Step 2: S3 uploads + build PointStruct (no upsert yet)
                ok, details = ingest_service.upload_to_s3_and_prepare_point(
                    image_bytes=image_bytes,
                    image_filename=uploaded.name,
                    content_type=getattr(uploaded, "type", None),
                    embedding=embedding,
                    description=display_text,
                    user_id=config.bulk_user_id,
                    meal_datetime=datetime.utcnow(),
                    meal_type="bulk",
                )
                if not ok:
                    failed += 1
                    continue
                s3img_times.append(details.get("s3_image_upload_ms") or 0.0)
                s3json_times.append(details.get("s3_embedding_upload_ms") or 0.0)

                points.append(details["point"])  # PointStruct for batch
                succeeded += 1
            except Exception as e:
                failed += 1
                st.error(f"Failed {uploaded.name}: {e}")
            finally:
                prog.progress(int(idx * 100 / len(uploads)))

        # Final step: single-shot (or chunked) upsert to Qdrant
        upsert_timer = MetricsTimer()
        with upsert_timer:
            ok = upsert_vectors_batch(
                qdrant,
                config.qdrant_bulk_collection_name,
                points,
                wait=False,
                chunk_size=0 if len(points) <= 512 else 512,
            )

    # Persist bulk run summary
    try:
        metrics_db.log_bulk_ingest_run({
            "collection_name": config.qdrant_bulk_collection_name,
            "images_total": len(uploads),
            "succeeded": succeeded,
            "failed": failed,
            "duration_ms_total": total_timer.duration_ms,
            "duration_ms_qdrant_upsert": upsert_timer.duration_ms if 'upsert_timer' in locals() else None,
            "avg_description_ms": (sum(desc_times)/len(desc_times)) if desc_times else None,
            "avg_embedding_ms": (sum(embed_times)/len(embed_times)) if embed_times else None,
            "avg_s3_image_upload_ms": (sum(s3img_times)/len(s3img_times)) if s3img_times else None,
            "avg_s3_embedding_upload_ms": (sum(s3json_times)/len(s3json_times)) if s3json_times else None,
            "notes": "fast_path" if fast_path else None,
            "error_message": None if ok else "qdrant_upsert_failed",
        })
    except Exception as log_err:
        st.warning(f"Failed to log bulk ingest run: {log_err}")

    # UI summary
    st.success(f"Bulk ingest complete. Succeeded: {succeeded} / {len(uploads)}")
    if failed:
        st.warning(f"Failed: {failed}")

    # Display bulk performance using the same component as single-ingest
    # Averages across all successfully processed images
    avg_desc = (sum(desc_times)/len(desc_times)) if desc_times else None
    avg_embed = (sum(embed_times)/len(embed_times)) if embed_times else None
    avg_s3_img = (sum(s3img_times)/len(s3img_times)) if s3img_times else None
    avg_s3_json = (sum(s3json_times)/len(s3json_times)) if s3json_times else None

    st.markdown("### ⚡ Ingestion Performance (Averages)")
    show_ingestion_performance(
        description_ms=avg_desc,
        embedding_ms=avg_embed,
        s3_image_upload_ms=avg_s3_img,
        s3_embedding_upload_ms=avg_s3_json,
        vector_index_ms=(upsert_timer.duration_ms if 'upsert_timer' in locals() else None),
    )

    # Totals across all successfully processed images
    tot_desc = sum(desc_times) if desc_times else None
    tot_embed = sum(embed_times) if embed_times else None
    tot_s3_img = sum(s3img_times) if s3img_times else None
    tot_s3_json = sum(s3json_times) if s3json_times else None

    st.markdown("### ⚡ Ingestion Performance (Totals)")
    show_ingestion_performance(
        description_ms=tot_desc,
        embedding_ms=tot_embed,
        s3_image_upload_ms=tot_s3_img,
        s3_embedding_upload_ms=tot_s3_json,
        vector_index_ms=(upsert_timer.duration_ms if 'upsert_timer' in locals() else None),
    )

    # Totals summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Images", len(uploads))
    with c2:
        st.metric("Succeeded", succeeded)
    with c3:
        st.metric("Total Time", f"{total_timer.duration_ms:.1f} ms")
