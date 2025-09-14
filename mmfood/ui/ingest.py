"""
UI components for the image ingestion tab.
"""
import io
import uuid
from datetime import datetime
from typing import Optional

import streamlit as st
from PIL import Image
from botocore.exceptions import ClientError, BotoCoreError

from mmfood.utils.crypto import md5_hex
from mmfood.services import IngestService
from mmfood.ui.components import show_ingestion_performance, display_upload_details
from mmfood.config import AppConfig
from mmfood.database import MetricsDatabase, MetricsTimer


# Image constraints to align with Bedrock service safety caps (also enforced in mmfood/bedrock/ai.py)
_MAX_SIDE = 1280
_MAX_PIXELS = 2_000_000  # ~2.0 MP


def _compute_downscale_dims(w: int, h: int, max_side: int = _MAX_SIDE, max_pixels: int = _MAX_PIXELS):
    """Compute target dimensions if downscaling is needed; returns (new_w, new_h) or (w, h)."""
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


def render_ingest_tab(config: AppConfig, ingest_service: IngestService):
    """Render the ingestion tab UI."""
    st.subheader("Upload Food Image")

    # Inform users about model limits and app guardrails up front
    with st.expander("Model Limits and Downscaling"):
        st.markdown(
            """
            - Amazon Titan Multimodal Embeddings G1 accepts images up to 25 MB per image (compressed) with 128 tokens for Image Captions and about 256 tokens of text per request.
            - In addition to file size, service-side validation can reject very high-resolution images. To avoid errors and reduce latency, this app automatically downsizes large images before sending them to Bedrock:
              - Longest side ≤ 1280 px
              - Total pixels ≤ 2,000,000 (~2.0 MP)
            - Images are JPEG-encoded (quality 90) for the calls to Claude Vision and Titan Multimodal Embeddings.

            If you upload an image that exceeds these resolution limits, we will show an informational message and downscale it automatically.
            """
        )

    # Initialize session state for storing generated data
    _initialize_ingest_session_state()

    # Arrange uploader/preview and metadata side-by-side
    left, right = st.columns(2)

    with left:
        # File uploader
        uploaded = st.file_uploader(
            "Select a food plate image",
            type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
            key="ingest_uploader",
            help="Upload a single image of your food plate"
        )

        # Handle file upload
        if uploaded is not None:
            _handle_file_upload(uploaded)

    with right:
        # Metadata inputs
        user_id, meal_date, meal_time_val, meal_type = _render_metadata_inputs()

    # Single-step ingestion action
    st.markdown("---")
    st.markdown("### Ingest")
    if st.button(
        "🚀 Ingest Image",
        type="primary",
        key="ingest_all_button",
        disabled=(uploaded is None)
    ):
        _handle_full_ingest(
            config=config,
            ingest_service=ingest_service,
            user_id=user_id,
            meal_date=meal_date,
            meal_time_val=meal_time_val,
            meal_type=meal_type,
            uploaded=uploaded,
        )


def _initialize_ingest_session_state():
    """Initialize session state variables for ingestion."""
    session_vars = [
        'generated_description', 'generated_embedding', 'current_image_bytes',
        'current_image_name', 'last_image_hash', 'ingest_preview_bytes'
    ]

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None


def _handle_file_upload(uploaded):
    """Handle file upload and preview."""
    # Read bytes once and keep them; also display a preview
    image_bytes = uploaded.read()
    st.session_state.current_image_bytes = image_bytes
    st.session_state.current_image_name = uploaded.name
    st.session_state.ingest_preview_bytes = image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption=f"Preview: {uploaded.name}", use_container_width=True)

        # Warn if the image will be downscaled for Bedrock limits
        try:
            w, h = img.size
            new_w, new_h = _compute_downscale_dims(w, h)
            if (new_w, new_h) != (w, h):
                mp = (w * h) / 1_000_000.0
                new_mp = (new_w * new_h) / 1_000_000.0
                st.info(
                    "Your image exceeds the app's resolution guardrails and will be downscaled to meet model limits.\n\n"
                    f"• Original: {w}×{h} (~{mp:.2f} MP)\n"
                    f"• Downscaled: {new_w}×{new_h} (~{new_mp:.2f} MP)\n\n"
                    "Guardrails (applied to ensure reliable Bedrock calls):\n"
                    f"• Longest side ≤ {_MAX_SIDE}px\n"
                    f"• Total pixels ≤ {_MAX_PIXELS:,} (~{_MAX_PIXELS/1_000_000:.1f} MP)\n\n"
                    "Note: The image will be JPEG-encoded (quality 90) for the calls to Claude Vision and Titan Multimodal Embeddings."
                )
        except Exception:
            pass
    except Exception:
        st.info("Preview unavailable, proceeding with raw bytes.")

    # Reset generated data ONLY if a different image was uploaded
    new_hash = md5_hex(image_bytes)
    if (st.session_state.last_image_hash and
        st.session_state.last_image_hash != new_hash):
        st.session_state.generated_description = None
        st.session_state.generated_embedding = None
    st.session_state.last_image_hash = new_hash


def _render_metadata_inputs():
    """Render metadata input fields."""
    st.markdown("**Metadata**")
    user_id = st.text_input(
        "User ID (optional)", value="", key="ingest_user_id"
    ).strip()

    col1, col2 = st.columns(2)
    with col1:
        meal_date = st.date_input(
            "Meal date", value=datetime.now().date(), key="meal_date"
        )
    with col2:
        meal_time_val = st.time_input(
            "Meal time", value=datetime.now().time(), key="meal_time"
        )

    meal_type = st.selectbox(
        "Meal type",
        ["breakfast", "lunch", "dinner", "snack", "other"],
        index=1,
        key="meal_type",
    )

    return user_id, meal_date, meal_time_val, meal_type


def _handle_description_generation(ingest_service: IngestService, meal_type: str):
    """Handle description and embedding generation."""
    with st.spinner("Processing..."):
        try:
            if st.session_state.current_image_bytes is None:
                st.error("Image bytes are missing. Please upload an image first.")
                st.stop()

            # Prepare meal data for description generation
            meal_data = {
                'meal_type': meal_type,
                'tags': [],
                'protein_grams': 0,
            }

            # Generate description and embedding
            (display_text, embedding,
             description_duration, embedding_duration) = ingest_service.generate_description_and_embedding(
                st.session_state.current_image_bytes, meal_data
            )

            st.session_state.generated_description = display_text
            st.session_state.generated_embedding = embedding

            st.success("✅ Description and embedding generated successfully!")

        except (ClientError, BotoCoreError) as aws_err:
            st.error(f"AWS error: {aws_err}")
        except Exception as e:
            st.error(f"Failed to generate: {e}")


def _display_generated_content():
    """Display generated description and embedding info."""
    if st.session_state.generated_description:
        st.markdown("**Generated Description:**")
        st.info(st.session_state.generated_description)

        st.markdown("**Embedding Info:**")
        if st.session_state.generated_embedding is not None:
            st.write(f"• Dimension: {len(st.session_state.generated_embedding)}")
            st.write(f"• First 10 values: {st.session_state.generated_embedding[:10]}")


def _handle_s3_upload(
    config: AppConfig,
    ingest_service: IngestService,
    user_id: str,
    meal_date,
    meal_time_val,
    meal_type: str,
    uploaded
):
    """Handle S3 upload and Qdrant indexing."""
    # Validation
    if not config.bucket:
        st.error("S3 bucket not configured. Please check your .env file.")
        st.stop()
    if not config.qdrant_url:
        st.error("Qdrant URL not configured. Please check your .env file.")
        st.stop()
    if st.session_state.generated_embedding is None:
        st.error("No embedding available. Please run Step 1: Generate Description & Embedding.")
        st.stop()
    if st.session_state.current_image_bytes is None:
        st.error("Image bytes missing from session. Please re-upload the image and run Step 1 again.")
        st.stop()

    with st.spinner("Uploading to S3 and indexing in Qdrant..."):
        try:
            # Build meal datetime
            meal_dt = datetime.combine(meal_date, meal_time_val)
            content_type = getattr(uploaded, "type", None)

            # Upload and index
            success, upload_details = ingest_service.upload_to_s3_and_index(
                image_bytes=st.session_state.current_image_bytes,
                image_filename=st.session_state.current_image_name,
                content_type=content_type,
                embedding=st.session_state.generated_embedding,
                description=st.session_state.generated_description,
                user_id=user_id,
                meal_datetime=meal_dt,
                meal_type=meal_type
            )

            if success:
                display_upload_details(upload_details)

                # Show ingestion performance metrics (includes S3 upload timings when available)
                show_ingestion_performance(
                    s3_image_upload_ms=upload_details.get("s3_image_upload_ms"),
                    s3_embedding_upload_ms=upload_details.get("s3_embedding_upload_ms"),
                    vector_index_ms=upload_details.get("vector_duration_ms"),
                )

                # Clear session state after successful upload
                _clear_session_state()
            else:
                st.error("Failed to index vector in Qdrant")

        except (ClientError, BotoCoreError) as aws_err:
            st.error(f"AWS error: {aws_err}")
        except Exception as e:
            st.error(f"Failed to upload: {e}")


def _clear_session_state():
    """Clear session state after successful upload."""
    session_vars = [
        'generated_description', 'generated_embedding', 'current_image_bytes',
        'current_image_name', 'last_image_hash', 'ingest_preview_bytes'
    ]

    for var in session_vars:
        st.session_state[var] = None


def _handle_full_ingest(
    config: AppConfig,
    ingest_service: IngestService,
    user_id: str,
    meal_date,
    meal_time_val,
    meal_type: str,
    uploaded,
):
    """End-to-end ingestion: description + embedding + S3 uploads + Qdrant index, with DB logging."""
    # Validate config early
    if not config.bucket:
        st.error("S3 bucket not configured. Please check your .env file.")
        st.stop()
    if not config.qdrant_url:
        st.error("Qdrant URL not configured. Please check your .env file.")
        st.stop()
    if st.session_state.current_image_bytes is None:
        st.error("Image bytes are missing. Please upload an image first.")
        st.stop()

    # Setup metrics DB and total timer
    metrics_db = MetricsDatabase()
    total_timer = MetricsTimer()
    ingest_id = str(uuid.uuid4())

    # Compute image characteristics
    image_size_bytes = len(st.session_state.current_image_bytes)
    original_w = original_h = None
    resized_w = resized_h = None
    resized_applied = False
    try:
        img = Image.open(io.BytesIO(st.session_state.current_image_bytes))
        original_w, original_h = img.size
        # Compute would-be downscale based on the same guardrails used elsewhere
        new_w, new_h = _compute_downscale_dims(original_w, original_h)
        if (new_w, new_h) != (original_w, original_h):
            resized_applied = True
            resized_w, resized_h = new_w, new_h
    except Exception:
        pass

    error_step = None
    error_message = None
    description_ms = embedding_ms = None
    s3_img_ms = s3_emb_ms = vector_ms = None
    image_id = None
    emb_json_size = None

    with st.spinner("Ingesting image: describe → embed → upload → index..."):
        try:
            with total_timer:
                # 1) Generate description + embedding
                meal_data = {
                    'meal_type': meal_type,
                    'tags': [],
                    'protein_grams': 0,
                }
                try:
                    (display_text, embedding, desc_ms, embed_ms) = ingest_service.generate_description_and_embedding(
                        st.session_state.current_image_bytes, meal_data
                    )
                    description_ms, embedding_ms = desc_ms, embed_ms
                except Exception as e:
                    error_step = "describe_image" if "converse" in str(e).lower() else "generate_embedding"
                    error_message = str(e)
                    raise

                # 2) Upload to S3 and index in Qdrant
                meal_dt = datetime.combine(meal_date, meal_time_val)
                content_type = getattr(uploaded, "type", None)
                try:
                    success, upload_details = ingest_service.upload_to_s3_and_index(
                        image_bytes=st.session_state.current_image_bytes,
                        image_filename=st.session_state.current_image_name,
                        content_type=content_type,
                        embedding=embedding,
                        description=display_text,
                        user_id=(user_id or "demo"),
                        meal_datetime=meal_dt,
                        meal_type=meal_type,
                    )
                    image_id = upload_details.get("image_id")
                    vector_ms = upload_details.get("vector_duration_ms")
                    s3_img_ms = upload_details.get("s3_image_upload_ms")
                    s3_emb_ms = upload_details.get("s3_embedding_upload_ms")
                    emb_json_size = upload_details.get("embedding_json_size_bytes")
                    if not success:
                        error_step = "qdrant_upsert"
                        error_message = "Vector upsert reported failure"
                        raise RuntimeError(error_message)
                except Exception as e:
                    if error_step is None:
                        # try to infer whether S3 or Qdrant stage failed
                        msg = str(e).lower()
                        if "put_object" in msg or "s3" in msg:
                            error_step = "upload_s3_image" if s3_img_ms is None else "upload_s3_embedding"
                        elif "qdrant" in msg or "upsert" in msg:
                            error_step = "qdrant_upsert"
                        else:
                            error_step = "validate_index"
                        error_message = str(e)
                    raise

            # If we got here, success
            st.markdown("**Generated Description:**")
            st.info(display_text)
            display_upload_details(upload_details)
            show_ingestion_performance(
                description_ms=description_ms,
                embedding_ms=embedding_ms,
                s3_image_upload_ms=s3_img_ms,
                s3_embedding_upload_ms=s3_emb_ms,
                vector_index_ms=vector_ms,
            )
            _clear_session_state()
        except (ClientError, BotoCoreError) as aws_err:
            error_message = str(aws_err)
            st.error(f"AWS error: {aws_err}")
        except Exception as e:
            if error_message is None:
                error_message = str(e)
            st.error(f"Ingestion failed: {e}")
        finally:
            # Persist a single ingestion record
            try:
                metrics_db.log_ingest_record({
                    "id": ingest_id,
                    "image_id": image_id,
                    "content_type": getattr(uploaded, "type", None),
                    "model_id": config.model_id,
                    "output_dim": config.output_dim,
                    "qdrant_collection_name": config.qdrant_collection_name,
                    "s3_bucket": config.bucket,
                    "original_width": original_w,
                    "original_height": original_h,
                    "resized_width": resized_w,
                    "resized_height": resized_h,
                    "resized_applied": 1 if resized_applied else 0,
                    "image_size_bytes": image_size_bytes,
                    "embedding_json_size_bytes": emb_json_size,
                    "description_ms": description_ms,
                    "embedding_ms": embedding_ms,
                    "s3_image_upload_ms": s3_img_ms,
                    "s3_embedding_upload_ms": s3_emb_ms,
                    "qdrant_upsert_ms": vector_ms,
                    "total_duration_ms": total_timer.duration_ms,
                    "success": 1 if (error_message is None) else 0,
                    "error_step": error_step,
                    "error_message": error_message,
                })
            except Exception as log_err:
                # Avoid breaking the UX due to logging issues
                print(f"[ingest_metrics] failed to log ingestion record: {log_err}")
