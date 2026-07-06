"""Evaluation Harness -- Core Benchmark Runner.

Provides the main CLI entry point for running automated evaluations of the
WikiMind RAG pipeline against standardized Q&A datasets. Supports configurable
strategy presets, dataset selection, and subset sizing.

Usage::

    python -m evaluation.harness --dataset nq --subset 50 --config evaluation/configs/baseline.json
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

from evaluation.datasets import load_nq_subset, load_triviaqa_subset
from evaluation.metrics import answer_accuracy, mean_reciprocal_rank, recall_at_k
from evaluation.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load an evaluation configuration JSON file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return json.load(f)


async def _evaluate_single_query(
    query: str,
    gold_answer: str,
    strategies: Dict[str, bool],
) -> Dict[str, Any]:
    """Run the RAG pipeline for a single query and compute metrics.

    Invokes the LangGraph agent directly (not via HTTP) to avoid network
    overhead during benchmarking. Records the retrieved chunks, generated
    answer, latency, and step count.

    Args:
        query: The evaluation question.
        gold_answer: The expected answer string.
        strategies: Strategy toggles for the pipeline.

    Returns:
        Dict containing per-query metrics and raw outputs.
    """
    from backend.agent import agent_app, AgentState
    from backend.models import QueryStrategies

    active_strategies = QueryStrategies(**strategies)

    initial_state: AgentState = {
        "query": query,
        "expanded_queries": [],
        "target_articles": [],
        "documents": [],
        "web_snippets": [],
        "generation": "",
        "retrieval_grade": "",
        "hallucination_grade": "",
        "answer_grade": "",
        "steps": 0,
        "active_strategies": active_strategies,
        "hallucination_retries": 0,
        "answer_retries": 0,
    }

    start_time = time.monotonic()

    try:
        # Run the full agent graph
        final_state = await agent_app.ainvoke(initial_state)
        latency = time.monotonic() - start_time

        generation = final_state.get("generation", "")
        documents = final_state.get("documents", [])
        step_count = final_state.get("steps", 0)

        # Extract retrieved text for metric computation
        retrieved_texts = [
            doc.get("content", doc.get("page_content", ""))
            for doc in documents
        ]

        # Compute metrics
        rec_5 = recall_at_k(retrieved_texts, gold_answer, k=5)
        mrr = mean_reciprocal_rank(retrieved_texts, gold_answer)
        acc = answer_accuracy(generation, gold_answer)

        return {
            "question": query,
            "gold_answer": gold_answer,
            "generation": generation[:500],
            "recall_at_5": rec_5,
            "mrr": mrr,
            "answer_accuracy": acc,
            "latency": round(latency, 4),
            "step_count": step_count,
            "document_count": len(documents),
        }

    except Exception as exc:
        latency = time.monotonic() - start_time
        logger.error("Evaluation failed for query '%s': %s", query[:60], exc)
        return {
            "question": query,
            "gold_answer": gold_answer,
            "generation": f"ERROR: {exc}",
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "answer_accuracy": 0.0,
            "latency": round(latency, 4),
            "step_count": 0,
            "document_count": 0,
            "error": str(exc),
        }


async def run_evaluation(
    dataset_name: str,
    subset_size: int,
    config: Dict[str, Any],
) -> str:
    """Run the full evaluation harness.

    Loads the specified dataset, runs each query through the pipeline, computes
    per-query and aggregate metrics, and generates a report.

    Args:
        dataset_name: Dataset identifier (``nq`` or ``triviaqa``).
        subset_size: Number of queries to evaluate.
        config: Evaluation configuration dict specifying strategies and params.

    Returns:
        Path to the generated markdown report.
    """
    # Load dataset
    if dataset_name == "nq":
        samples = load_nq_subset(n=subset_size)
    elif dataset_name == "triviaqa":
        samples = load_triviaqa_subset(n=subset_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'nq' or 'triviaqa'.")

    if not samples:
        logger.error("No samples loaded. Exiting.")
        return ""

    strategies = config.get("strategies", {})
    logger.info(
        "Starting evaluation: dataset=%s, queries=%d, strategies=%s",
        dataset_name,
        len(samples),
        strategies,
    )

    # Run evaluation queries
    per_query_results = []
    for i, sample in enumerate(samples):
        question = sample["question"]
        gold_answer = sample["gold_answer"]

        logger.info("[%d/%d] Evaluating: %s", i + 1, len(samples), question[:80])

        result = await _evaluate_single_query(question, gold_answer, strategies)
        per_query_results.append(result)

        # Log progress every 10 queries
        if (i + 1) % 10 == 0:
            completed = per_query_results[-10:]
            avg_recall = sum(r["recall_at_5"] for r in completed) / len(completed)
            avg_latency = sum(r["latency"] for r in completed) / len(completed)
            logger.info(
                "  Progress: %d/%d | Last 10 avg recall@5=%.2f, avg latency=%.2fs",
                i + 1, len(samples), avg_recall, avg_latency,
            )

    # Generate report
    report_path = generate_report(
        per_query_results=per_query_results,
        config=config,
        dataset_name=dataset_name,
    )

    logger.info("Evaluation complete. Report: %s", report_path)
    return report_path


def main():
    """CLI entry point for the evaluation harness."""
    parser = argparse.ArgumentParser(
        description="WikiMind Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m evaluation.harness --dataset nq --subset 50\n"
            "  python -m evaluation.harness --dataset triviaqa --subset 100 "
            "--config evaluation/configs/with_expansion.json\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nq",
        choices=["nq", "triviaqa"],
        help="Evaluation dataset to use (default: nq).",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=50,
        help="Number of queries to evaluate (default: 50).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="evaluation/configs/baseline.json",
        help="Path to evaluation config JSON (default: baseline).",
    )

    args = parser.parse_args()

    config = _load_config(args.config)

    asyncio.run(run_evaluation(
        dataset_name=args.dataset,
        subset_size=args.subset,
        config=config,
    ))


if __name__ == "__main__":
    main()
