"""Article-Level Index for Two-Stage Retrieval.

Manages a separate Qdrant collection (``wikimind_articles``) containing one
dense vector per Wikipedia article. The vector is generated from the
concatenation of the article title and the first two paragraphs of text,
providing a semantically rich summary embedding.

Stage 1 of the retrieval pipeline searches this collection to identify the
2-3 most relevant Wikipedia articles for a query. Stage 2 then scopes the
chunk-level hybrid search (in ``wikimind_hybrid``) to only those articles,
eliminating cross-article noise entirely.

This replaces the previous Tavily-based article discovery, making the entire
pipeline fully offline with no external API dependencies for retrieval.
"""

import hashlib
import logging
import uuid
import asyncio
from typing import List, Optional

from qdrant_client.http import models

from backend.config import get_settings
from backend.qdrant_client import get_async_qdrant, get_sync_qdrant
from data_pipeline.ingest import get_dense_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article Collection Management
# ---------------------------------------------------------------------------

def init_article_collection() -> None:
    """Initialize the article-level Qdrant collection.

    Creates the ``wikimind_articles`` collection if it does not already exist.
    This collection uses only dense vectors (no sparse/BM25) because the
    article-level text (title + first paragraphs) is short and semantically
    rich, making dense similarity sufficient for article identification.

    A payload index on ``title`` enables fast exact-match lookups for
    deduplication during incremental updates.
    """
    client = get_sync_qdrant()
    settings = get_settings()
    collection_name = settings.article_collection

    try:
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            logger.info("Article collection '%s' already exists.", collection_name)
            return

        logger.info(
            "Creating article collection '%s' (dense-only, %d dims)...",
            collection_name,
            settings.embedding_dim,
        )

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )

        # Payload index on title for fast deduplication lookups
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info("Article collection '%s' created successfully.", collection_name)
    except Exception as exc:
        logger.error("Failed to initialize article collection: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Article Search
# ---------------------------------------------------------------------------

async def search_articles(query: str, top_k: Optional[int] = None) -> List[str]:
    """Search the article-level index to identify relevant Wikipedia articles.

    Embeds the query with the same dense model used during ingestion and
    performs a cosine similarity search against the article-level collection.

    Args:
        query: The natural language search query.
        top_k: Number of article titles to return. Defaults to the
            ``article_search_top_k`` setting (typically 3).

    Returns:
        List of Wikipedia article titles, ordered by relevance.
    """
    settings = get_settings()
    effective_top_k = top_k or settings.article_search_top_k
    dense_model = get_dense_model()

    # Generate query embedding
    query_vector = list(dense_model.embed([query]))[0].tolist()

    try:
        qdrant_async = get_async_qdrant()
        if qdrant_async is not None:
            results = await qdrant_async.query_points(
                collection_name=settings.article_collection,
                query=query_vector,
                limit=effective_top_k,
                with_payload=True,
            )
        else:
            # Embedded mode: use sync client via thread
            sync_client = get_sync_qdrant()
            results = await asyncio.to_thread(
                sync_client.query_points,
                collection_name=settings.article_collection,
                query=query_vector,
                limit=effective_top_k,
                with_payload=True,
            )

        titles = []
        for point in results.points:
            title = point.payload.get("title", "")
            if title and title not in titles:
                titles.append(title)

        logger.info(
            "Article index identified %d article(s): %s",
            len(titles),
            ", ".join(titles),
        )
        return titles

    except Exception as exc:
        logger.error("Article index search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_article_id(title: str) -> str:
    """Generate a deterministic UUID for an article-level entry.

    Ensures idempotent upserts when re-indexing the same article.

    Args:
        title: The Wikipedia article title.

    Returns:
        A valid UUID string derived from the title.
    """
    key = f"article::{title}"
    hash_digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_digest))


def extract_article_summary(title: str, full_text: str, max_chars: int = 1500) -> str:
    """Extract a summary text for article-level embedding.

    Concatenates the article title with the first two paragraphs of the
    article text. The result is capped at ``max_chars`` to stay within
    the embedding model's optimal input length.

    Args:
        title: The article title.
        full_text: The full article text.
        max_chars: Maximum character length for the summary.

    Returns:
        A summary string suitable for embedding.
    """
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    first_paragraphs = "\n\n".join(paragraphs[:2]) if paragraphs else ""
    summary = f"{title}\n\n{first_paragraphs}"
    return summary[:max_chars]


async def upsert_article(title: str, summary_text: str, url: str = "") -> None:
    """Upsert a single article into the article-level index.

    Used by the live sync worker to refresh the Stage 1 discovery index
    whenever an article is re-ingested.

    Args:
        title: The Wikipedia article title.
        summary_text: Summary text to embed (title + first paragraphs).
        url: Optional article URL.
    """
    settings = get_settings()
    dense_model = get_dense_model()
    article_id = generate_article_id(title)

    # Generate embedding
    embedding = list(dense_model.embed([summary_text]))[0].tolist()

    point = models.PointStruct(
        id=article_id,
        vector=embedding,
        payload={
            "title": title,
            "url": url,
            "summary": summary_text[:500],
        },
    )

    qdrant = get_async_qdrant()
    if qdrant is not None:
        await qdrant.upsert(
            collection_name=settings.article_collection,
            points=[point],
        )
    else:
        sync_client = get_sync_qdrant()
        await asyncio.to_thread(
            sync_client.upsert,
            collection_name=settings.article_collection,
            points=[point],
        )
    logger.debug("Upserted article index entry for '%s'", title)
