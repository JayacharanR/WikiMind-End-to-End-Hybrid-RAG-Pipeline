"""Evaluation Report Generator.

Produces markdown and JSON reports from evaluation harness results. Reports
include summary metric tables, per-query breakdowns of the worst-performing
queries, and the configuration used for the run.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

from evaluation.metrics import compute_aggregate_metrics

logger = logging.getLogger(__name__)

RESULTS_DIR = "evaluation/results"


def _ensure_results_dir() -> None:
    """Create the results directory if it does not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_report(
    per_query_results: List[Dict],
    config: Dict,
    dataset_name: str,
    output_prefix: str | None = None,
) -> str:
    """Generate a markdown evaluation report and save it alongside a JSON dump.

    Args:
        per_query_results: List of per-query result dicts from the harness.
        config: The evaluation configuration dict used for this run.
        dataset_name: Name of the dataset (e.g., ``nq``, ``triviaqa``).
        output_prefix: Optional filename prefix. Defaults to a timestamp.

    Returns:
        Absolute path to the generated markdown report.
    """
    _ensure_results_dir()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = output_prefix or timestamp
    md_path = os.path.join(RESULTS_DIR, f"{prefix}_report.md")
    json_path = os.path.join(RESULTS_DIR, f"{prefix}_results.json")

    # Compute aggregate metrics
    aggregates = compute_aggregate_metrics(per_query_results)

    # Build the markdown report
    lines = []
    lines.append("# WikiMind Evaluation Report")
    lines.append("")
    lines.append(f"**Dataset:** {dataset_name}")
    lines.append(f"**Timestamp:** {timestamp}")
    lines.append(f"**Total Queries:** {aggregates.get('total_queries', 0)}")
    lines.append("")

    # Summary metrics table
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Mean Recall@5 | {aggregates.get('mean_recall_at_5', 0.0):.4f} |")
    lines.append(f"| Mean MRR | {aggregates.get('mean_mrr', 0.0):.4f} |")
    lines.append(f"| Mean Answer Accuracy | {aggregates.get('mean_answer_accuracy', 0.0):.4f} |")
    lines.append(f"| Latency P50 | {aggregates.get('latency_p50', 0.0):.4f}s |")
    lines.append(f"| Latency P95 | {aggregates.get('latency_p95', 0.0):.4f}s |")
    lines.append(f"| Latency P99 | {aggregates.get('latency_p99', 0.0):.4f}s |")
    lines.append(f"| Mean Step Count | {aggregates.get('mean_step_count', 0.0):.2f} |")
    lines.append("")

    # Configuration used
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")

    # Per-query breakdown (worst 20 by answer accuracy, then by recall)
    lines.append("## Worst Performing Queries (Bottom 20)")
    lines.append("")

    sorted_results = sorted(
        per_query_results,
        key=lambda r: (r.get("answer_accuracy", 0.0), r.get("recall_at_5", 0.0)),
    )
    worst = sorted_results[:20]

    lines.append("| # | Question | Gold Answer | Accuracy | Recall@5 | MRR | Latency |")
    lines.append("|---|----------|-------------|----------|----------|-----|---------|")
    for i, result in enumerate(worst):
        question = result.get("question", "")[:80]
        gold = result.get("gold_answer", "")[:40]
        acc = result.get("answer_accuracy", 0.0)
        rec = result.get("recall_at_5", 0.0)
        mrr = result.get("mrr", 0.0)
        lat = result.get("latency", 0.0)
        lines.append(
            f"| {i + 1} | {question} | {gold} | {acc:.2f} | {rec:.2f} | {mrr:.2f} | {lat:.2f}s |"
        )

    lines.append("")

    # Write markdown report
    report_content = "\n".join(lines)
    with open(md_path, "w") as f:
        f.write(report_content)

    # Write JSON results
    json_output = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "config": config,
        "aggregates": aggregates,
        "per_query_results": per_query_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    logger.info("Report saved to: %s", md_path)
    logger.info("Results saved to: %s", json_path)
    return md_path
