"""
UI components for the metrics dashboard tab.
"""
import streamlit as st
from mmfood.database import MetricsDatabase


def render_metrics_tab(metrics_db: MetricsDatabase):
    """Render the metrics dashboard tab."""
    st.subheader("📊 RAG Retrieval Metrics Dashboard")
    
    # Metrics time period selection and cleanup
    days = _render_time_period_controls(metrics_db)
    
    # Get metrics data
    with st.spinner("Loading metrics..."):
        summary = metrics_db.get_metrics_summary(days=days)
        recent_errors = metrics_db.get_recent_errors(limit=5)
        ingest_summary = metrics_db.get_ingest_summary(days=days)
        ingest_errors = metrics_db.get_recent_ingest_errors(limit=5)
        ingest_rows = metrics_db.get_recent_ingest_rows(limit=10)
    
    # Display metrics or no data message
    if _has_metrics_data(summary):
        _display_metrics_dashboard(summary)
    else:
        st.info(f"No search metrics found for the last {days} day{'s' if days > 1 else ''}.")

    # Ingestion metrics section
    _display_ingestion_metrics(ingest_summary, ingest_errors)
    _display_ingestion_table(ingest_rows)
    
    # Display recent search errors if any
    _display_recent_errors(recent_errors)
    
    # Export option
    _render_export_section({"search": summary, "ingest": ingest_summary})


def _render_time_period_controls(metrics_db: MetricsDatabase):
    """Render time period selection and cleanup controls."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        days = st.selectbox(
            "Time Period", 
            [1, 7, 30], 
            index=1, 
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
        )
    
    with col2:
        if st.button(
            "🧹 Cleanup Old Records (30+ days)", 
            help="Remove metrics older than 30 days"
        ):
            with st.spinner("Cleaning up old records..."):
                metrics_db.cleanup_old_records(days_to_keep=30)
            st.success("Old records cleaned up!")
    
    return days


def _has_metrics_data(summary):
    """Check if there's meaningful metrics data to display."""
    return (summary.get("request_stats") and 
            summary["request_stats"].get("total_requests", 0) > 0)


def _display_metrics_dashboard(summary):
    """Display the main metrics dashboard."""
    # Performance Summary
    st.markdown("### 📈 Performance Summary")
    _display_performance_summary(summary["request_stats"])
    
    # Performance Breakdown
    st.markdown("### ⚡ Performance Breakdown")
    _display_performance_breakdown(summary["request_stats"])
    
    # Query type distribution
    if summary.get("query_types"):
        st.markdown("### 🔍 Query Types")
        _display_query_types(summary["query_types"])
    
    # Top users
    if summary.get("top_users"):
        st.markdown("### 👥 Top Users")
        _display_top_users(summary["top_users"])
    
    # Performance range
    if summary.get("performance"):
        st.markdown("### 📊 Performance Range")
        _display_performance_range(summary["performance"])


def _display_performance_summary(stats):
    """Display high-level performance metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", int(stats.get("total_requests", 0)))
    with col2:
        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
    with col3:
        st.metric("Avg Total Time", f"{stats.get('avg_total_duration', 0):.1f} ms")
    with col4:
        st.metric("Avg Results", f"{stats.get('avg_results_count', 0):.1f}")


def _display_performance_breakdown(stats):
    """Display detailed performance breakdown."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg Embedding Time", f"{stats.get('avg_embedding_duration', 0):.1f} ms")
    with col2:
        st.metric("Avg Search Time", f"{stats.get('avg_search_duration', 0):.1f} ms")


def _display_query_types(query_types):
    """Display query type distribution."""
    for item in query_types:
        st.write(f"• **{item['query_type'].title()}**: {item['count']} requests")


def _display_top_users(top_users):
    """Display top users by request count."""
    for i, user in enumerate(top_users[:5], 1):
        st.write(f"{i}. **{user['user_id']}**: {user['request_count']} requests")


def _display_performance_range(performance):
    """Display performance min/max range."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fastest Request", f"{performance.get('min_duration', 0):.1f} ms")
    with col2:
        st.metric("Slowest Request", f"{performance.get('max_duration', 0):.1f} ms")


def _display_ingestion_metrics(ingest_summary, ingest_errors):
    """Display ingestion metrics section."""
    st.markdown("## 🧪 Ingestion Metrics")
    stats = ingest_summary.get("ingest_stats") or {}
    perf = ingest_summary.get("ingest_performance") or {}

    if not stats or stats.get("total_ingests", 0) == 0:
        st.info("No ingestion metrics for the selected period.")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Ingests", int(stats.get("total_ingests", 0)))
    with col2:
        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
    with col3:
        st.metric("Avg Total Time", f"{stats.get('avg_total', 0):.1f} ms")

    # Step averages
    st.markdown("### Step Breakdown (averages)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Description", f"{stats.get('avg_description', 0):.1f} ms")
    with c2:
        st.metric("Embedding", f"{stats.get('avg_embedding', 0):.1f} ms")
    with c3:
        st.metric("S3 Image Upload", f"{stats.get('avg_s3_image', 0):.1f} ms")
    c4, c5 = st.columns(2)
    with c4:
        st.metric("S3 Embedding Upload", f"{stats.get('avg_s3_embedding', 0):.1f} ms")
    with c5:
        st.metric("Qdrant Upsert", f"{stats.get('avg_qdrant', 0):.1f} ms")

    # Range
    st.markdown("### Performance Range")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Fastest Ingest", f"{perf.get('min_total', 0):.1f} ms")
    with r2:
        st.metric("Slowest Ingest", f"{perf.get('max_total', 0):.1f} ms")

    # Recent errors
    if ingest_errors:
        st.markdown("### Recent Ingestion Errors")
        for err in ingest_errors:
            with st.expander(f"Error at {err['timestamp']} - Image: {err.get('image_id','?')}"):
                st.write(f"Step: {err.get('error_step','unknown')}")
                st.write(f"Error: {err.get('error_message','')}")


def _display_ingestion_table(rows):
    """Render a table with the latest ingestion rows."""
    st.markdown("### Latest Ingestions (10)")
    if not rows:
        st.write("No recent ingestions.")
        return
    # Shape rows for display
    display_rows = []
    for r in rows:
        display_rows.append({
            "timestamp": r.get("timestamp"),
            "image_id": r.get("image_id"),
            "content_type": r.get("content_type"),
            "orig_size": f"{r.get('original_width','?')}×{r.get('original_height','?')}",
            "resized": "yes" if r.get("resized_applied") else "no",
            "resized_size": (f"{r.get('resized_width')}×{r.get('resized_height')}" if r.get("resized_width") and r.get("resized_height") else "-"),
            "img_bytes": r.get("image_size_bytes"),
            "json_bytes": r.get("embedding_json_size_bytes"),
            "desc_ms": r.get("description_ms"),
            "embed_ms": r.get("embedding_ms"),
            "s3_img_ms": r.get("s3_image_upload_ms"),
            "s3_json_ms": r.get("s3_embedding_upload_ms"),
            "qdrant_ms": r.get("qdrant_upsert_ms"),
            "total_ms": r.get("total_duration_ms"),
            "ok": bool(r.get("success")),
            "err_step": r.get("error_step"),
        })
    st.dataframe(display_rows, use_container_width=True)


def _display_recent_errors(recent_errors):
    """Display recent errors section."""
    if recent_errors:
        st.markdown("### 🚨 Recent Errors")
        for error in recent_errors:
            with st.expander(f"Error at {error['timestamp']} - User: {error['user_id']}"):
                st.write(f"**Query Type:** {error['query_type']}")
                st.write(f"**Duration:** {error.get('total_duration_ms', 'N/A')} ms")
                st.write(f"**Error:** {error['error_message']}")


def _render_export_section(summary):
    """Render the export metrics section."""
    st.markdown("---")
    if st.button("📄 Export Metrics Summary"):
        st.json(summary)
