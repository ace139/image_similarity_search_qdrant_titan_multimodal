"""
Multi-Modal Food Image Search Application - Refactored Version

A streamlined Streamlit app for food image search using:
- AWS Bedrock Titan Multi-Modal Embeddings
- Claude 4 Sonnet for AI-generated image descriptions
- Qdrant Vector Database for similarity search
- SQLite metrics logging for performance analytics

This is the refactored main application file with modular architecture.
"""
import uuid
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env before importing modules that read env
# This fixes cases where the app cannot see variables like APP_S3_BUCKET or QDRANT_URL
load_dotenv(find_dotenv(), override=True)

from mmfood.config import load_config
from mmfood.database import MetricsDatabase
from mmfood.services import IngestService, SearchService
from mmfood.ui import render_ingest_tab, render_search_tab, render_metrics_tab

APP_TITLE = "Multi-Modal Food Image Search with AWS AI Stack"


def _validate_required_env_vars(cfg):
    """Validate that all required environment variables are set."""
    missing_vars = cfg.missing_required()
    if missing_vars:
        st.error(
            "❌ **Missing Required Environment Variables**\n\n"
            "The following variables must be set in your `.env` file:\n\n" +
            "\n".join([f"• `{var}`" for var in missing_vars]) +
            "\n\nPlease check your `.env` file and restart the application."
        )
        st.stop()


def initialize_session_state():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def render_help_section():
    """Render the help and documentation section."""
    st.divider()
    with st.expander("Help & Notes"):
        st.markdown(
            """
            **Multi-Modal Image Similarity Search**

            - This app uses **Claude 4 Sonnet** to produce concise textual descriptions of your food images.
            - Descriptions are combined with images to create **multi-modal embeddings** with Amazon Titan Multimodal.
            - Qdrant is used as the vector database for fast, filtered similarity search.

            **Configuration (strict .env)**

            Set the following variables in your `.env` (see `.env.example`). These are strictly required by both the app and the sanity tests — no fallbacks are used:

            - `AWS_REGION`
            - `APP_S3_BUCKET`, `APP_IMAGES_PREFIX`, `APP_EMBEDDINGS_PREFIX`
            - `MODEL_ID`, `OUTPUT_EMBEDDING_LENGTH`
            - `QDRANT_URL`, `QDRANT_COLLECTION_NAME`, `QDRANT_TIMEOUT`, `QDRANT_API_KEY`
            - Optional: `AWS_PROFILE`

            Amazon Bedrock access requirements:
            - Titan Multimodal Embeddings model (`amazon.titan-embed-image-v1`)
            - **Claude Vision inference profile**. Set `CLAUDE_VISION_MODEL_ID` to the profile **ID** (e.g., `global.anthropic.claude-sonnet-4-20250514-v1:0`) or **ARN** (e.g. `arn:aws:bedrock:us-west-2:762035899142:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0`).
            - To find it: Bedrock Console → Inference and assessment → Cross-Region inference → select profile → copy ID/ARN.

            **Qdrant setup**
            - The app connects to `QDRANT_URL` and ensures the target collection’s payload indexes for filtering (`user_id`, `meal_type`, `ts`).
            - Vector size is validated against `OUTPUT_EMBEDDING_LENGTH` at runtime.

            **Stored objects in S3**
            - Images: `<images_prefix>/<uuid>.<ext>`
            - Embeddings JSON: `<embeddings_prefix>/<uuid>.json` (includes generated description and metadata)

            **Metrics**
            - All requests are logged to a local SQLite database (`rag_metrics.db`).
            - View latency and operation breakdowns in the **📊 Metrics** tab.

            **Environment sanity checks**
            - Use `tests/env_sanity.py` to validate credentials and connectivity:
              - Human-readable: `python tests/env_sanity.py`
              - JSON (for CI): `python tests/env_sanity.py --json`
              - Optional deeper checks: `--write-s3` and/or `--invoke-bedrock`
            - Exit codes: 0 (all OK), 1 (check failed), 2 (missing/invalid env).

            **Notes**
            - The app ensures required Qdrant payload indexes automatically at runtime.
            - Keep your `.env` in sync with `.env.example` when deploying or sharing the project.
            """
        )


def main():
    """Main application entry point."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Multi-Modal Food Search",
        page_icon="🍽️",
        layout="centered"
    )

    st.title(APP_TITLE)
    st.caption(
        "🔍 **Search food images using text or images** • Powered by Qdrant Vector Database, "
        "Bedrock Titan Multi-Modal Embeddings, and Claude 4 Sonnet for AI-generated image descriptions"
    )

    # Load configuration and validate
    config = load_config()
    _validate_required_env_vars(config)

    # Initialize components
    initialize_session_state()
    metrics_db = MetricsDatabase()

    # Initialize services
    ingest_service = IngestService(config, metrics_db)
    search_service = SearchService(config, metrics_db)

    # Create main tabs
    ingest_tab, search_tab, metrics_tab = st.tabs(["Ingest", "Search", "📊 Metrics"])

    # Render tabs using UI modules
    with ingest_tab:
        render_ingest_tab(config, ingest_service)

    with search_tab:
        render_search_tab(config, search_service, st.session_state.session_id)

    with metrics_tab:
        render_metrics_tab(metrics_db)

    # Render help section
    render_help_section()


if __name__ == "__main__":
    main()
