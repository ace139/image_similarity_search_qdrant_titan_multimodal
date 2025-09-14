"""
Multi-Modal Food Image Search Application - Refactored Version

A streamlined Streamlit app for food image search using:
- AWS Bedrock Titan Multi-Modal Embeddings
- Claude 3.5 Sonnet for AI-generated image descriptions  
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
            **Multi-Modal Image Similarity Search:**
            - This app uses **Claude Vision** to generate detailed textual descriptions of your food images.
            - These descriptions are combined with the images to create **multi-modal embeddings** using Titan Multimodal.
            - This approach significantly improves search accuracy by capturing both visual and semantic content.
            
            **Configuration:**
            - Ensure your AWS credentials are configured in your `.env` file or via environment variables.
            - Amazon Bedrock must be enabled with access to:
              - Titan Multimodal Embeddings model (`amazon.titan-embed-image-v1`)
              - Claude Vision via an **inference profile**. Set `CLAUDE_VISION_MODEL_ID` to the inference profile **ID** (e.g., `us.anthropic.claude-3-5-sonnet-20241022-v2:0`) or the **ARN**. The Converse API requires an inference profile for these Anthropic models.
              - To find the profile ID/ARN: Bedrock Console → Inference and assessment → **Cross-Region inference** → select the relevant profile (e.g., *US Anthropic Claude 3.5 Sonnet v2*) and copy its ID/ARN.
            - Qdrant vector database configuration is loaded from your `.env` file:
              - `QDRANT_URL`: URL of your Qdrant instance (e.g., `https://your-cluster.qdrant.tech:6333`)
              - `QDRANT_API_KEY`: API key for authentication (if required)
              - `QDRANT_COLLECTION_NAME`: Name of the collection to store vectors (default: `food_embeddings`)
            
            **Stored Objects:**
            - Images: `<images_prefix>/<uuid>.<ext>`
            - Embeddings JSON: `<embeddings_prefix>/<uuid>.json` (includes generated descriptions)
            - Qdrant Vectors: Multi-modal embeddings indexed by image UUID with rich metadata (no 10-key limit)
            
            **Metrics System:**
            - All search operations are automatically logged to a SQLite database (`rag_metrics.db`)
            - View performance analytics, usage patterns, and error tracking in the **📊 Metrics** tab
            - Metrics include request timing, success rates, query patterns, and user activity
            
            **Testing:**
            - Note: The existing test scripts may need updates to work with Qdrant instead of S3 Vectors.
            - Ensure your Qdrant instance is accessible and properly configured.
            - Run `python test_metrics.py` to verify the metrics logging system.
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
        "Bedrock Titan Multi-Modal Embeddings, and Claude 3.5 Sonnet for AI-generated image descriptions"
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
