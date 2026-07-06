"""Qdrant client and collection management.

Handles the initialization of the Qdrant vector database, including setting up
the hybrid collection schema with both dense and sparse (BM25) vector configurations.
Provides helper methods for ID generation to prevent race conditions during
parallel ingestion.
"""

import hashlib
import logging
import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models

from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qdrant Client Singletons
# ---------------------------------------------------------------------------

_async_client: Optional[AsyncQdrantClient] = None
_sync_client: Optional[QdrantClient] = None


def get_async_qdrant() -> AsyncQdrantClient:
    """Return a cached async Qdrant client instance."""
    global _async_client
    if _async_client is None:
        settings = get_settings()
        _async_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=10.0,
        )
    return _async_client


def get_sync_qdrant() -> QdrantClient:
    """Return a cached sync Qdrant client instance (useful for batch scripts)."""
    global _sync_client
    if _sync_client is None:
        settings = get_settings()
        _sync_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30.0,
        )
    return _sync_client


# ---------------------------------------------------------------------------
# Collection Schema
# ---------------------------------------------------------------------------

def init_collection() -> None:
    """Initialize the Qdrant collection with a hybrid vector schema.

    Creates the collection if it doesn't exist. The schema defines two named
    vectors:
    - ``dense``: Used for dense embeddings (e.g., BAAI/bge-small-en-v1.5)
    - ``sparse``: Used for sparse BM25 term frequency vectors.
    """
    client = get_sync_qdrant()
    settings = get_settings()
    collection_name = settings.qdrant_collection

    try:
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            logger.info("Qdrant collection '%s' already exists.", collection_name)
            return

        logger.info("Creating Qdrant collection '%s' with hybrid schema...", collection_name)
        
        # Define hybrid schema with both dense and sparse vectors
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        
        # Create a payload index on the 'title' field for faster exact-match filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Temporal versioning payload indices
        client.create_payload_index(
            collection_name=collection_name,
            field_name="is_current",
            field_schema=models.PayloadSchemaType.BOOL,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="ingested_at",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        logger.info("Collection '%s' created successfully.", collection_name)
    except Exception as exc:
        logger.error("Failed to initialize Qdrant collection: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_point_id(article_title: str, chunk_index: int) -> str:
    """Generate a deterministic UUID for a chunk to prevent duplication.

    Using a hash of the article title and chunk index ensures that if the
    same article is re-ingested (e.g., during an update or crash recovery),
    the upsert operation will overwrite the existing points idempotently
    rather than creating duplicates.

    Args:
        article_title: The title of the Wikipedia article.
        chunk_index: The sequential index of the chunk within the article.

    Returns:
        A valid UUID string.
    """
    key = f"{article_title}::chunk_{chunk_index}"
    hash_digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_digest))
