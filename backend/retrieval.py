"""Tri-Brid Retrieval Engine.

Implements the core retrieval logic including Qdrant Hybrid Search (Dense + Sparse)
and Reciprocal Rank Fusion (RRF) via the Qdrant Universal Query API. Supports
article-scoped filtering to restrict search to specific Wikipedia articles
identified by the web-search discovery step.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import unquote, urlparse

from qdrant_client.http import models

from backend.config import get_settings
from backend.qdrant_client import get_async_qdrant, get_sync_qdrant
from data_pipeline.ingest import get_dense_model, get_sparse_model

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

_ranker: Optional[Ranker] = None


def get_reranker() -> Ranker:
    """Lazy initialize the FlashRank cross-encoder model."""
    global _ranker
    if _ranker is None:
        settings = get_settings()
        logger.info("Initializing FlashRank reranker model: %s", settings.reranker_model)
        _ranker = Ranker(model_name=settings.reranker_model, cache_dir="data/flashrank_cache")
    return _ranker


# ---------------------------------------------------------------------------
# Article Title Extraction
# ---------------------------------------------------------------------------

def extract_title_from_wikipedia_url(url: str) -> Optional[str]:
    """Extract the article title from a Wikipedia URL.

    Handles standard Wikipedia URLs like:
        https://en.wikipedia.org/wiki/Tokyo
        https://en.wikipedia.org/wiki/Eiffel_Tower
        https://en.wikipedia.org/wiki/Machine_learning

    Args:
        url: A Wikipedia URL string.

    Returns:
        The decoded article title, or None if the URL is not a Wikipedia article.
    """
    try:
        parsed = urlparse(url)
        if "wikipedia.org" not in parsed.netloc:
            return None
        path = parsed.path
        if "/wiki/" in path:
            raw_title = path.split("/wiki/", 1)[1]
            # Remove any fragment or query
            raw_title = raw_title.split("#")[0].split("?")[0]
            # Decode URL encoding and replace underscores with spaces
            title = unquote(raw_title).replace("_", " ")
            return title if title else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Hybrid Search with Optional Article Scoping
# ---------------------------------------------------------------------------

async def hybrid_search(
    query: str,
    article_titles: Optional[List[str]] = None,
    as_of_date: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a hybrid dense+sparse search using Qdrant Prefetch API for RRF.

    When ``article_titles`` is provided, the search is scoped to only chunks
    whose ``title`` payload field matches one of the given titles. This
    dramatically reduces noise by restricting retrieval to the specific
    Wikipedia articles identified by the article discovery step.

    When ``as_of_date`` is provided (ISO format string), performs a time-travel
    query by filtering chunks ingested on or before that date, regardless of
    their ``is_current`` status. Otherwise, defaults to only returning chunks
    with ``is_current=true``.

    The cross-encoder reranker is always applied to ensure maximum precision.

    Args:
        query: The natural language search query.
        article_titles: Optional list of Wikipedia article titles to scope
            the search to. When None, searches the entire collection.
        as_of_date: Optional ISO date string for time-travel queries.

    Returns:
        Tuple of (List of candidate documents, Metadata dict).
    """
    settings = get_settings()

    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    logger.debug("Generating dual embeddings for query: %s", query)

    # Generate embeddings
    dense_vector = list(dense_model.embed([query]))[0].tolist()
    sparse_obj = list(sparse_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_obj.indices.tolist(),
        values=sparse_obj.values.tolist(),
    )

    # Build filter conditions
    filter_conditions = []

    # Article scope filter
    if article_titles:
        filter_conditions.append(
            models.FieldCondition(
                key="title",
                match=models.MatchAny(any=article_titles),
            )
        )
        logger.info(
            "Scoping hybrid search to %d article(s): %s",
            len(article_titles),
            ", ".join(article_titles[:5]),
        )

    # Temporal filter
    if as_of_date:
        # Time-travel mode: return chunks ingested on or before the given date
        filter_conditions.append(
            models.FieldCondition(
                key="ingested_at",
                range=models.Range(lte=as_of_date),
            )
        )
        logger.info("Time-travel mode: as_of_date=%s", as_of_date)
    else:
        # Default mode: only return current version of chunks
        filter_conditions.append(
            models.FieldCondition(
                key="is_current",
                match=models.MatchValue(value=True),
            )
        )

    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    # Qdrant Prefetch API for server-side Reciprocal Rank Fusion
    prefetch_dense = models.Prefetch(
        query=dense_vector,
        using="dense",
        limit=settings.rrf_k,
        filter=query_filter,
    )

    prefetch_sparse = models.Prefetch(
        query=sparse_vector,
        using="sparse",
        limit=settings.rrf_k,
        filter=query_filter,
    )

    logger.debug(
        "Executing Qdrant Universal Query API with RRF fusion (k=%d, scoped=%s)",
        settings.rrf_k,
        bool(article_titles),
    )

    metadata = {
        "rrf_candidates": 0,
        "reranker_applied": True,
        "article_scoped": bool(article_titles),
        "scoped_articles": article_titles or [],
    }

    try:
        qdrant_async = get_async_qdrant()
        if qdrant_async is not None:
            results = await qdrant_async.query_points(
                collection_name=settings.qdrant_collection,
                prefetch=[prefetch_dense, prefetch_sparse],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=settings.retrieval_top_k,
                with_payload=True,
            )
        else:
            # Embedded mode: use sync client via thread
            sync_client = get_sync_qdrant()
            results = await asyncio.to_thread(
                sync_client.query_points,
                collection_name=settings.qdrant_collection,
                prefetch=[prefetch_dense, prefetch_sparse],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=settings.retrieval_top_k,
                with_payload=True,
            )

        # Format results
        documents = []
        for point in results.points:
            documents.append({
                "id": str(point.id),
                "score": float(point.score),
                "title": point.payload.get("title", ""),
                "text": point.payload.get("page_content", ""),  # FlashRank expects "text" key
                "url": point.payload.get("url", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
            })

        metadata["rrf_candidates"] = len(documents)
        logger.info("Hybrid search returned %d RRF candidates.", len(documents))

        # Always apply cross-encoder reranker for maximum precision
        if documents:
            logger.debug("Applying FlashRank reranker...")
            ranker = get_reranker()
            rerank_request = RerankRequest(query=query, passages=documents)
            reranked_results = ranker.rerank(rerank_request)

            # FlashRank returns the list sorted by score (descending)
            top_documents = reranked_results[:settings.reranker_top_k]

            # Map "text" back to "content" for consistency
            for doc in top_documents:
                doc["content"] = doc.pop("text")

            logger.info("Reranked to top %d candidates.", len(top_documents))
            return top_documents, metadata
        else:
            return [], metadata

    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        return [], metadata
