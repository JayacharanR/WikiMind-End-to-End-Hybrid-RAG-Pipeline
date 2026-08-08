"""Evaluation Dataset Loaders.

Provides functions for loading standardized Q&A evaluation datasets from
HuggingFace. Supports Natural Questions (NQ) and TriviaQA with configurable
subset sizes and local caching.
"""

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

CACHE_DIR = "data/eval_cache"


def _ensure_cache_dir() -> None:
    """Create the evaluation cache directory if it does not exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(dataset_name: str, subset_size: int) -> str:
    """Generate a deterministic cache file path for a dataset subset."""
    key = f"{dataset_name}_{subset_size}"
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_from_cache(cache_path: str) -> List[Dict] | None:
    """Load a cached dataset subset if available."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            logger.info("Loaded %d samples from cache: %s", len(data), cache_path)
            return data
        except Exception as exc:
            logger.warning("Cache load failed: %s", exc)
    return None


def _save_to_cache(cache_path: str, data: List[Dict]) -> None:
    """Save a dataset subset to the local cache."""
    _ensure_cache_dir()
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
        logger.info("Cached %d samples to: %s", len(data), cache_path)
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)


def load_nq_subset(n: int = 500) -> List[Dict]:
    """Load a subset of the Natural Questions dataset.

    Downloads from HuggingFace (``google-research-datasets/natural_questions``)
    on first use and extracts (question, short_answer, wikipedia_title) tuples.
    Results are cached locally for subsequent runs.

    Args:
        n: Number of question-answer pairs to load.

    Returns:
        List of dicts with keys: ``question``, ``gold_answer``, ``wikipedia_title``.
    """
    cache_path = _get_cache_path("nq", n)
    cached = _load_from_cache(cache_path)
    if cached is not None:
        return cached

    logger.info("Loading Natural Questions dataset from HuggingFace (subset=%d)...", n)

    from datasets import load_dataset

    # NQ validation split is smaller and faster to download
    dataset = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    samples = []
    for example in dataset:
        if len(samples) >= n:
            break

        # Extract short answers from the annotations
        # HuggingFace NQ format: annotations is a dict with nested lists
        # short_answers has start_token/end_token as lists of lists (per annotator)
        annotations = example.get("annotations", {})
        short_answers = annotations.get("short_answers", {})

        # short_answers is a dict with keys: start_token, end_token, start_byte, end_byte, text
        # Each value is a list of lists (one list per annotator)
        start_tokens_per_annotator = short_answers.get("start_token", [])
        end_tokens_per_annotator = short_answers.get("end_token", [])

        if not start_tokens_per_annotator or not end_tokens_per_annotator:
            continue

        # Find the first annotator with a non-empty answer span
        start_token = -1
        end_token = -1
        for ann_starts, ann_ends in zip(
            start_tokens_per_annotator, end_tokens_per_annotator, strict=False
        ):
            if isinstance(ann_starts, list):
                # Each annotator provides a list of spans; take the first
                if ann_starts and ann_ends:
                    start_token = ann_starts[0]
                    end_token = ann_ends[0]
                    break
            elif isinstance(ann_starts, int) and ann_starts >= 0:
                # Some formats provide ints directly
                start_token = ann_starts
                end_token = ann_ends
                break

        if start_token < 0 or end_token < 0:
            continue

        # Extract the answer text from the document tokens
        doc_tokens = example.get("document", {}).get("tokens", {})
        tokens = doc_tokens.get("token", [])
        is_html = doc_tokens.get("is_html", [])

        if not tokens or start_token >= len(tokens):
            continue

        # Reconstruct answer from non-HTML tokens in the answer span
        answer_tokens = []
        for idx in range(start_token, min(end_token, len(tokens))):
            if idx < len(is_html) and not is_html[idx]:
                answer_tokens.append(tokens[idx])

        answer_text = " ".join(answer_tokens).strip()
        if not answer_text:
            continue

        question = example.get("question", {}).get("text", "")
        title = example.get("document", {}).get("title", "")

        if question:
            samples.append(
                {
                    "question": question,
                    "gold_answer": answer_text,
                    "wikipedia_title": title,
                }
            )

    logger.info("Extracted %d NQ samples with valid short answers.", len(samples))
    _save_to_cache(cache_path, samples)
    return samples


def load_triviaqa_subset(n: int = 500) -> List[Dict]:
    """Load a subset of the TriviaQA dataset.

    Downloads from HuggingFace (``trivia_qa``) on first use and extracts
    (question, answer, evidence) tuples. Results are cached locally.

    Args:
        n: Number of question-answer pairs to load.

    Returns:
        List of dicts with keys: ``question``, ``gold_answer``, ``evidence``.
    """
    cache_path = _get_cache_path("triviaqa", n)
    cached = _load_from_cache(cache_path)
    if cached is not None:
        return cached

    logger.info("Loading TriviaQA dataset from HuggingFace (subset=%d)...", n)

    from datasets import load_dataset

    dataset = load_dataset(
        "trivia_qa",
        "rc",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    samples = []
    for example in dataset:
        if len(samples) >= n:
            break

        question = example.get("question", "")
        answer_data = example.get("answer", {})

        # TriviaQA provides normalized_value and aliases
        answer_text = answer_data.get("value", "")
        if not answer_text or not question:
            continue

        # Get evidence context if available
        search_results = example.get("search_results", {})
        search_contexts = search_results.get("search_context", [])
        evidence = search_contexts[0] if search_contexts else ""

        samples.append(
            {
                "question": question,
                "gold_answer": answer_text,
                "evidence": evidence[:1000] if evidence else "",
            }
        )

    logger.info("Extracted %d TriviaQA samples.", len(samples))
    _save_to_cache(cache_path, samples)
    return samples
