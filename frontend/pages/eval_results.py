"""Evaluation Results Browser.

Streamlit page for browsing evaluation harness results. Lists all completed
evaluation runs, renders markdown reports inline, and shows per-query
drill-down tables.
"""

import json
import logging
import os
from typing import Dict, List

import streamlit as st

logger = logging.getLogger(__name__)

RESULTS_DIR = "evaluation/results"


def _list_evaluation_runs() -> List[Dict]:
    """Scan the results directory and return metadata for each run."""
    runs = []
    if not os.path.exists(RESULTS_DIR):
        return runs

    for filename in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if filename.endswith("_results.json"):
            json_path = os.path.join(RESULTS_DIR, filename)
            md_path = json_path.replace("_results.json", "_report.md")
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                runs.append({
                    "filename": filename,
                    "json_path": json_path,
                    "md_path": md_path if os.path.exists(md_path) else None,
                    "dataset": data.get("dataset", "unknown"),
                    "timestamp": data.get("timestamp", "unknown"),
                    "total_queries": data.get("aggregates", {}).get("total_queries", 0),
                    "aggregates": data.get("aggregates", {}),
                    "config": data.get("config", {}),
                })
            except Exception as exc:
                logger.warning("Failed to load results file %s: %s", filename, exc)

    return runs


def render_eval_results():
    """Render the Evaluation Results Browser page."""
    st.title("Evaluation Results")
    st.markdown(
        "Browse results from automated evaluation harness runs. "
        "Each run benchmarks the pipeline against a standardized Q&A dataset."
    )

    runs = _list_evaluation_runs()

    if not runs:
        st.info(
            "No evaluation results found. Run the harness first:\n\n"
            "```\npython -m evaluation.harness --dataset nq --subset 50\n```"
        )
        return

    # Run selector
    run_labels = [
        f"{r['timestamp']} -- {r['dataset']} ({r['total_queries']} queries)"
        for r in runs
    ]
    selected_idx = st.selectbox(
        "Select an evaluation run:",
        range(len(runs)),
        format_func=lambda i: run_labels[i],
        key="eval_run_select",
    )

    selected_run = runs[selected_idx]

    # Summary metrics
    st.subheader("Summary Metrics")
    agg = selected_run.get("aggregates", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall@5", f"{agg.get('mean_recall_at_5', 0):.4f}")
    col2.metric("MRR", f"{agg.get('mean_mrr', 0):.4f}")
    col3.metric("Answer Accuracy", f"{agg.get('mean_answer_accuracy', 0):.4f}")
    col4.metric("Queries", agg.get("total_queries", 0))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Latency P50", f"{agg.get('latency_p50', 0):.3f}s")
    col6.metric("Latency P95", f"{agg.get('latency_p95', 0):.3f}s")
    col7.metric("Latency P99", f"{agg.get('latency_p99', 0):.3f}s")
    col8.metric("Avg Steps", f"{agg.get('mean_step_count', 0):.1f}")

    # Configuration
    with st.expander("Evaluation Configuration"):
        st.json(selected_run.get("config", {}))

    # Markdown report
    if selected_run.get("md_path"):
        st.subheader("Full Report")
        with open(selected_run["md_path"], "r") as f:
            report_content = f.read()
        st.markdown(report_content)

    # Per-query drill-down
    st.subheader("Per-Query Results")

    try:
        with open(selected_run["json_path"], "r") as f:
            full_data = json.load(f)
        per_query = full_data.get("per_query_results", [])

        if per_query:
            # Filter controls
            sort_by = st.selectbox(
                "Sort by:",
                ["answer_accuracy", "recall_at_5", "mrr", "latency", "step_count"],
                key="eval_sort",
            )
            ascending = st.checkbox("Ascending order", value=True, key="eval_asc")

            sorted_results = sorted(
                per_query,
                key=lambda r: r.get(sort_by, 0),
                reverse=not ascending,
            )

            # Display as a table
            table_data = []
            for r in sorted_results[:50]:
                table_data.append({
                    "Question": r.get("question", "")[:80],
                    "Gold Answer": r.get("gold_answer", "")[:40],
                    "Accuracy": r.get("answer_accuracy", 0.0),
                    "Recall@5": r.get("recall_at_5", 0.0),
                    "MRR": r.get("mrr", 0.0),
                    "Latency (s)": r.get("latency", 0.0),
                    "Steps": r.get("step_count", 0),
                })

            st.dataframe(table_data, use_container_width=True)

            # Detail view for selected query
            st.subheader("Query Detail View")
            query_labels = [r.get("question", "")[:100] for r in sorted_results[:50]]
            if query_labels:
                detail_idx = st.selectbox(
                    "Select a query for detail view:",
                    range(len(query_labels)),
                    format_func=lambda i: query_labels[i],
                    key="eval_detail",
                )
                detail = sorted_results[detail_idx]

                st.markdown(f"**Question:** {detail.get('question', '')}")
                st.markdown(f"**Gold Answer:** {detail.get('gold_answer', '')}")
                st.markdown(f"**Generated Answer:** {detail.get('generation', '')}")
                st.markdown(
                    f"**Metrics:** Accuracy={detail.get('answer_accuracy', 0):.2f}, "
                    f"Recall@5={detail.get('recall_at_5', 0):.2f}, "
                    f"MRR={detail.get('mrr', 0):.2f}, "
                    f"Latency={detail.get('latency', 0):.2f}s, "
                    f"Steps={detail.get('step_count', 0)}"
                )

    except Exception as exc:
        st.error(f"Failed to load per-query results: {exc}")
