"""
UI components for the image ingestion tab.
"""
import io
from datetime import datetime
from typing import Optional

import streamlit as st
from PIL import Image
from botocore.exceptions import ClientError, BotoCoreError

from mmfood.utils.crypto import md5_hex
from mmfood.services import IngestService
from mmfood.ui.components import show_ingestion_performance, display_upload_details
from mmfood.config import AppConfig


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
    """End-to-end ingestion: description + embedding + S3 uploads + Qdrant index."""
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

    with st.spinner("Ingesting image: describe → embed → upload → index..."):
        try:
            # 1) Generate description + embedding
            meal_data = {
                'meal_type': meal_type,
                'tags': [],
                'protein_grams': 0,
            }
            (display_text, embedding, desc_ms, embed_ms) = ingest_service.generate_description_and_embedding(
                st.session_state.current_image_bytes, meal_data
            )

            # 2) Upload to S3 and index in Qdrant
            meal_dt = datetime.combine(meal_date, meal_time_val)
            content_type = getattr(uploaded, "type", None)
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

            if success:
                # Display generated content summary
                st.markdown("**Generated Description:**")
                st.info(display_text)

                display_upload_details(upload_details)

                # Show ingestion performance metrics (now includes S3 upload timings)
                show_ingestion_performance(
                    description_ms=desc_ms,
                    embedding_ms=embed_ms,
                    s3_image_upload_ms=upload_details.get("s3_image_upload_ms"),
                    s3_embedding_upload_ms=upload_details.get("s3_embedding_upload_ms"),
                    vector_index_ms=upload_details.get("vector_duration_ms"),
                )

                # Clear session state
                _clear_session_state()
            else:
                st.error("Failed to index vector in Qdrant")

        except (ClientError, BotoCoreError) as aws_err:
            st.error(f"AWS error: {aws_err}")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
