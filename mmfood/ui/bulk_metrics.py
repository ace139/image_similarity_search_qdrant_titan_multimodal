"""
Bulk Metrics UI: summary panels for bulk ingest and bulk search tables.
"""
import streamlit as st
from mmfood.database import MetricsDatabase


def render_bulk_metrics_tab(metrics_db: MetricsDatabase):
    st.subheader("📊 Bulk Metrics")

    # Time period selection (same UX as regular metrics)
    col1, _ = st.columns([1, 3])
    with col1:
        days = st.selectbox(
            "Time Period",
            [1, 7, 30],
            index=1,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
            key="bulk_metrics_days",
        )

    with st.spinner("Loading bulk metrics..."):
        ingest_summary = metrics_db.get_bulk_ingest_summary(days=days)
        ingest_runs = metrics_db.get_recent_bulk_ingest_runs(limit=10)
        search_summary = metrics_db.get_bulk_search_summary(days=days)
        search_errors = metrics_db.get_recent_bulk_search_errors(limit=10)

    st.markdown("### Ingest Summary")
    if ingest_summary:
        _kvs({
            "Total Runs": ingest_summary.get("total_runs"),
            "Total Images": ingest_summary.get("total_images"),
            "Succeeded": ingest_summary.get("total_succeeded"),
            "Failed": ingest_summary.get("total_failed"),
            "Avg Total (ms)": _fmt_ms(ingest_summary.get("avg_total_ms")),
            "Avg Qdrant Upsert (ms)": _fmt_ms(ingest_summary.get("avg_qdrant_ms")),
        })
    else:
        st.info("No bulk ingest data in the selected period.")

    st.markdown("### Latest Bulk Ingest Runs")
    if ingest_runs:
        st.dataframe(ingest_runs, use_container_width=True)
    else:
        st.write("No recent bulk ingest runs.")

    st.markdown("### Search Summary")
    if search_summary:
        _kvs({
            "Total Requests": search_summary.get("total_requests"),
            "Success Rate (%)": _fmt_pct(search_summary.get("success_rate")),
            "Avg Total (ms)": _fmt_ms(search_summary.get("avg_total_ms")),
            "Avg Embedding (ms)": _fmt_ms(search_summary.get("avg_embed_ms")),
            "Avg Search (ms)": _fmt_ms(search_summary.get("avg_search_ms")),
            "Avg Results": search_summary.get("avg_results"),
        })
    else:
        st.info("No bulk search data in the selected period.")

    if search_errors:
        st.markdown("### Recent Search Errors")
        for err in search_errors:
            with st.expander(f"Error at {err['timestamp']}"):
                st.write(f"Error: {err.get('error_message','')}")
                st.write(f"Duration: {_fmt_ms(err.get('duration_ms_total'))}")


def _fmt_ms(v):
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "-"


def _fmt_pct(v):
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "-"


def _kvs(d):
    cols = st.columns(min(4, len(d)))
    items = list(d.items())
    for i, (k, v) in enumerate(items):
        with cols[i % len(cols)]:
            st.metric(k, v)
