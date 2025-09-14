"""
Bulk Metrics UI: summary panels for bulk ingest and bulk search tables.
"""
import streamlit as st
from mmfood.database import MetricsDatabase


def render_bulk_metrics_tab(metrics_db: MetricsDatabase):
    """Render the metrics dashboard tab."""
    st.subheader("📊 Bulk Metrics")

    # Time period selection (same UX as regular metrics)
    col1, _ = st.columns([1, 3])
    with col1:
        days = st.selectbox(
            "Time Period",
            [1, 7, 30],
            index=1,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
        )

    # Load config to scope to bulk collection where relevant
    try:
        from mmfood.config import load_config
        cfg = load_config()
        _bulk_collection = cfg.qdrant_bulk_collection_name or cfg.qdrant_collection_name
    except Exception:
        _bulk_collection = None

    with st.spinner("Loading bulk metrics..."):
        ingest_summary = metrics_db.get_bulk_ingest_summary(days=days)
        ingest_runs = metrics_db.get_recent_bulk_ingest_runs(limit=10)
        search_summary = metrics_db.get_bulk_search_summary(days=days, collection_name=_bulk_collection)
        search_errors = metrics_db.get_recent_bulk_search_errors(limit=10)

    # Separate into Search vs Ingestion subtabs for clarity
    search_tab, ingest_tab = st.tabs(["🔎 Bulk Search Metrics", "🧪 Bulk Ingestion Metrics"])

    with ingest_tab:
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

    with search_tab:
        st.markdown("### Search Summary")
        if search_summary:
            st.markdown("### 📈 Performance Summary")
            _kvs({
                "Total Requests": search_summary.get("total_requests"),
                "Success Rate (%)": _fmt_pct(search_summary.get("success_rate")),
                "Avg Total (ms)": _fmt_ms(search_summary.get("avg_total_ms")),
                "Avg Results": search_summary.get("avg_results"),
            })

            st.markdown("### ⚡ Performance Breakdown")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Avg Embedding Time", _fmt_ms(search_summary.get("avg_embed_ms")) + " ms")
            with c2:
                st.metric("Avg Search Time", _fmt_ms(search_summary.get("avg_search_ms")) + " ms")

            st.markdown(f"### 🎯 Search Quality (last {days} days)")
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.metric("Avg Top-1 Score", _fmt_score(search_summary.get("avg_top1_score")))
            with q2:
                st.metric("Avg TopK Avg", _fmt_score(search_summary.get("avg_topk_avg_score")))
            with q3:
                st.metric("Avg Score Min", _fmt_score(search_summary.get("avg_score_min")))
            with q4:
                st.metric("Avg Score Max", _fmt_score(search_summary.get("avg_score_max")))
        else:
            st.info("No bulk search data in the selected period.")

        # Query type distribution (parity with normal metrics)
        counts = metrics_db.get_bulk_query_type_counts(days=days)
        if counts:
            st.markdown("### 🔍 Query Types")
            for item in counts:
                st.write(f"• **{item['query_type'].title()}**: {item['count']} requests")

        # Latest bulk searches table
        st.markdown("### Latest Bulk Searches (10)")
        try:
            rows = metrics_db.get_recent_bulk_search_rows(_bulk_collection, limit=10) if _bulk_collection else []
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.write("No recent bulk searches.")
        except Exception:
            pass

        # Performance range (parity with normal metrics)
        perf = metrics_db.get_bulk_search_performance_range(days=days)
        if perf:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Fastest Request", _fmt_ms(perf.get('min_total')) + " ms")
            with c2:
                st.metric("Slowest Request", _fmt_ms(perf.get('max_total')) + " ms")

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


def _fmt_score(v):
    try:
        return f"{float(v):.3f}"
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
