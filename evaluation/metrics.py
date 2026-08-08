"""Evaluation Metrics for RAG Pipeline Benchmarking.

Provides functions for computing retrieval quality (Recall@K, MRR), answer
accuracy (normalized substring match), and latency percentiles. All metric
functions operate on plain Python data structures and have no external
dependencies beyond the standard library.
"""

import re
import statistics
from typing import Dict, List

# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing, stripping articles,
    removing punctuation, and collapsing whitespace.

    This follows the standard normalization used in SQuAD and Natural Questions
    evaluation scripts.

    Args:
        text: Raw text string.

    Returns:
        Normalized text suitable for substring matching.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_texts: List[str], gold_answer: str, k: int = 5) -> float:
    """Compute Recall@K for a single query.

    Checks whether any of the top-K retrieved chunks contain the gold answer
    string (after normalization). Returns 1.0 if found, 0.0 otherwise.

    Args:
        retrieved_texts: List of retrieved chunk text strings, ordered by rank.
        gold_answer: The expected answer string.
        k: Number of top results to consider.

    Returns:
        1.0 if the gold answer appears in any of the top-K chunks, else 0.0.
    """
    normalized_answer = _normalize_text(gold_answer)
    if not normalized_answer:
        return 0.0

    for text in retrieved_texts[:k]:
        if normalized_answer in _normalize_text(text):
            return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved_texts: List[str], gold_answer: str) -> float:
    """Compute the Reciprocal Rank for a single query.

    Returns 1/rank where rank is the position (1-indexed) of the first
    retrieved chunk containing the gold answer. Returns 0.0 if no chunk
    contains the answer.

    Args:
        retrieved_texts: List of retrieved chunk text strings, ordered by rank.
        gold_answer: The expected answer string.

    Returns:
        Reciprocal rank (1/position) of the first relevant chunk, or 0.0.
    """
    normalized_answer = _normalize_text(gold_answer)
    if not normalized_answer:
        return 0.0

    for i, text in enumerate(retrieved_texts):
        if normalized_answer in _normalize_text(text):
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Answer Accuracy
# ---------------------------------------------------------------------------


def answer_accuracy(generation: str, gold_answer: str) -> float:
    """Compute answer accuracy via normalized substring match.

    After normalizing both the generated answer and the gold answer, checks
    whether the gold answer appears as a substring of the generation.

    Args:
        generation: The LLM-generated answer text.
        gold_answer: The expected answer string.

    Returns:
        1.0 if the normalized gold answer is a substring of the normalized
        generation, else 0.0.
    """
    normalized_gen = _normalize_text(generation)
    normalized_gold = _normalize_text(gold_answer)

    if not normalized_gold:
        return 0.0

    return 1.0 if normalized_gold in normalized_gen else 0.0


# ---------------------------------------------------------------------------
# Latency Metrics
# ---------------------------------------------------------------------------


def compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Compute P50, P95, and P99 latency percentiles.

    Args:
        latencies: List of latency measurements in seconds.

    Returns:
        Dictionary with keys ``p50``, ``p95``, ``p99`` mapping to latency
        values in seconds. Returns zeros if the input list is empty.
    """
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def _percentile(p: float) -> float:
        """Compute the p-th percentile using linear interpolation."""
        idx = (p / 100.0) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_latencies[lower] * (1 - frac) + sorted_latencies[upper] * frac

    return {
        "p50": round(_percentile(50), 4),
        "p95": round(_percentile(95), 4),
        "p99": round(_percentile(99), 4),
    }


# ---------------------------------------------------------------------------
# Aggregate Metrics
# ---------------------------------------------------------------------------


def compute_aggregate_metrics(
    per_query_results: List[Dict],
) -> Dict[str, float]:
    """Compute aggregate metrics across all evaluation queries.

    Args:
        per_query_results: List of per-query result dicts, each containing
            keys ``recall_at_5``, ``mrr``, ``answer_accuracy``, ``latency``,
            and ``step_count``.

    Returns:
        Dictionary of aggregate metrics including mean recall, MRR, accuracy,
        latency percentiles, and mean step count.
    """
    if not per_query_results:
        return {}

    recalls = [r.get("recall_at_5", 0.0) for r in per_query_results]
    mrrs = [r.get("mrr", 0.0) for r in per_query_results]
    accuracies = [r.get("answer_accuracy", 0.0) for r in per_query_results]
    latencies = [r.get("latency", 0.0) for r in per_query_results]
    steps = [r.get("step_count", 0) for r in per_query_results]

    latency_pcts = compute_latency_percentiles(latencies)

    return {
        "mean_recall_at_5": round(statistics.mean(recalls), 4),
        "mean_mrr": round(statistics.mean(mrrs), 4),
        "mean_answer_accuracy": round(statistics.mean(accuracies), 4),
        "latency_p50": latency_pcts["p50"],
        "latency_p95": latency_pcts["p95"],
        "latency_p99": latency_pcts["p99"],
        "mean_step_count": round(statistics.mean(steps), 2),
        "total_queries": len(per_query_results),
    }
