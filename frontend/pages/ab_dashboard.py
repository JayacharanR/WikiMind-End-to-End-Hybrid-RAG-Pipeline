"""A/B Strategy Comparison Dashboard.

Streamlit page that runs the same query through multiple pipeline configurations
side-by-side and visualizes the differences in retrieval quality, answer text,
latency, and step count. Supports single-query comparison and batch CSV mode.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Predefined strategy configurations for comparison
STRATEGY_PRESETS = {
    "Baseline": {
        "multi_query": False,
        "hyde": False,
        "step_back": False,
        "decomposition": False,
    },
    "Multi-Query": {
        "multi_query": True,
        "hyde": False,
        "step_back": False,
        "decomposition": False,
    },
    "HyDE": {
        "multi_query": False,
        "hyde": True,
        "step_back": False,
        "decomposition": False,
    },
    "Step-Back": {
        "multi_query": False,
        "hyde": False,
        "step_back": True,
        "decomposition": False,
    },
    "Multi-Query + Step-Back": {
        "multi_query": True,
        "hyde": False,
        "step_back": True,
        "decomposition": False,
    },
    "Decomposition": {
        "multi_query": False,
        "hyde": False,
        "step_back": False,
        "decomposition": True,
    },
}


def _run_single_query(query: str, strategies: Dict[str, bool]) -> Dict[str, Any]:
    """Run a single query through the backend and collect the full response.

    Uses the /chat SSE endpoint, consuming all events and returning the
    final response payload with metadata.
    """
    import sseclient

    payload = {"query": query, "strategies": strategies}

    try:
        start = time.monotonic()
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=120,
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)
        result = {}

        for event in client.events():
            if event.event == "final":
                result = json.loads(event.data)
                break
            elif event.event == "error":
                result = {"answer": f"Error: {json.loads(event.data).get('detail', 'Unknown')}", "sources": [], "metadata": {}}
                break

        latency = time.monotonic() - start
        result["latency"] = round(latency, 3)
        return result

    except Exception as exc:
        return {
            "answer": f"Request failed: {exc}",
            "sources": [],
            "metadata": {},
            "latency": 0.0,
        }


def render_ab_dashboard():
    """Render the A/B Strategy Comparison Dashboard page."""
    st.title("A/B Strategy Comparison")
    st.markdown(
        "Compare how different query expansion strategies affect retrieval "
        "quality, answer accuracy, and latency for the same query."
    )

    # Input section
    st.subheader("Query Configuration")
    query = st.text_input(
        "Enter a test query:",
        value="What is the population of Tokyo?",
        key="ab_query",
    )

    selected_configs = st.multiselect(
        "Select configurations to compare (2-4 recommended):",
        options=list(STRATEGY_PRESETS.keys()),
        default=["Baseline", "Multi-Query"],
        key="ab_configs",
    )

    if len(selected_configs) < 2:
        st.warning("Select at least 2 configurations to compare.")
        return

    # Run comparison
    if st.button("Run Comparison", type="primary", key="ab_run"):
        results = {}

        progress = st.progress(0, text="Running comparisons...")
        for i, config_name in enumerate(selected_configs):
            progress.progress(
                (i) / len(selected_configs),
                text=f"Running: {config_name}...",
            )
            strategies = STRATEGY_PRESETS[config_name]
            results[config_name] = _run_single_query(query, strategies)

        progress.progress(1.0, text="Comparison complete.")

        # Display results in columns
        st.divider()
        st.subheader("Results Comparison")

        cols = st.columns(len(selected_configs))
        for col, config_name in zip(cols, selected_configs):
            result = results[config_name]
            metadata = result.get("metadata", {})
            sources = result.get("sources", [])

            with col:
                st.markdown(f"### {config_name}")

                # Metrics row
                latency = result.get("latency", 0.0)
                steps = metadata.get("agent_steps", 0)
                st.metric("Latency", f"{latency:.2f}s")
                st.metric("Agent Steps", steps)
                st.metric("Sources", len(sources))

                # Answer
                st.markdown("**Answer:**")
                st.markdown(result.get("answer", "No answer generated."))

                # Sources
                if sources:
                    with st.expander(f"Retrieved Sources ({len(sources)})"):
                        for j, src in enumerate(sources):
                            st.markdown(
                                f"**[{j+1}] {src.get('title', 'Unknown')}** "
                                f"(score: {src.get('score', 0):.3f})"
                            )
                            content = src.get("content", "")[:200]
                            st.caption(content)

                # Metadata
                with st.expander("Execution Metadata"):
                    st.json(metadata)

    # Batch mode section
    st.divider()
    st.subheader("Batch Comparison Mode")
    st.markdown(
        "Upload a CSV with a ``query`` column to run all selected configurations "
        "against multiple queries and see aggregate metrics."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV (must have a 'query' column):",
        type=["csv"],
        key="ab_csv",
    )

    if uploaded_file is not None and st.button("Run Batch", key="ab_batch_run"):
        import csv
        import io

        content = uploaded_file.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        queries = [row["query"] for row in reader if "query" in row]

        if not queries:
            st.error("CSV must have a 'query' column with at least one entry.")
            return

        st.info(f"Running {len(queries)} queries across {len(selected_configs)} configurations...")

        batch_results = {name: [] for name in selected_configs}
        progress = st.progress(0, text="Processing batch...")
        total = len(queries) * len(selected_configs)
        done = 0

        for q in queries:
            for config_name in selected_configs:
                strategies = STRATEGY_PRESETS[config_name]
                result = _run_single_query(q, strategies)
                result["query"] = q
                batch_results[config_name].append(result)
                done += 1
                progress.progress(done / total, text=f"Query {done}/{total}")

        progress.progress(1.0, text="Batch complete.")

        # Aggregate metrics table
        st.subheader("Aggregate Metrics")
        agg_data = []
        for config_name in selected_configs:
            results_list = batch_results[config_name]
            avg_latency = sum(r.get("latency", 0) for r in results_list) / len(results_list)
            avg_steps = sum(r.get("metadata", {}).get("agent_steps", 0) for r in results_list) / len(results_list)
            avg_sources = sum(len(r.get("sources", [])) for r in results_list) / len(results_list)
            agg_data.append({
                "Configuration": config_name,
                "Avg Latency (s)": round(avg_latency, 3),
                "Avg Steps": round(avg_steps, 1),
                "Avg Sources": round(avg_sources, 1),
            })

        st.table(agg_data)
