"""Qdrant client and collection management.

Handles the initialization of the Qdrant vector database, including setting up
the hybrid collection schema with both dense and sparse (BM25) vector configurations.
Provides helper methods for ID generation to prevent race conditions during
parallel ingestion.

Supports two modes:
- ``local`` (embedded): Stores data on disk via Qdrant's embedded mode.
  No Docker or external server required. Ideal for development and offline usage.
- ``remote``: Connects to a running Qdrant server (e.g., via Docker Compose).
"""

import asyncio
import hashlib
import logging
import os
import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models

from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qdrant Client Singletons
# ---------------------------------------------------------------------------

_async_client = None  # AsyncQdrantClient or LocalAsyncQdrantAdapter
_sync_client: Optional[QdrantClient] = None


class LocalAsyncQdrantAdapter:
    """Adapter that wraps a sync QdrantClient to provide an async-compatible
    interface by delegating all calls through ``asyncio.to_thread()``.

    This avoids the embedded storage folder lock conflict that occurs when
    both sync and async clients try to open the same embedded database, while
    still allowing async callers (wiki_updater, reconciler, page_index) to
    work transparently in local mode.
    """

    def __init__(self, sync_client: QdrantClient):
        self._sync = sync_client

    async def get_collection(self, **kwargs):
        return await asyncio.to_thread(self._sync.get_collection, **kwargs)

    async def scroll(self, **kwargs):
        return await asyncio.to_thread(self._sync.scroll, **kwargs)

    async def query_points(self, **kwargs):
        return await asyncio.to_thread(self._sync.query_points, **kwargs)

    async def upsert(self, **kwargs):
        return await asyncio.to_thread(self._sync.upsert, **kwargs)

    async def set_payload(self, **kwargs):
        return await asyncio.to_thread(self._sync.set_payload, **kwargs)

    async def delete(self, **kwargs):
        """Delete points through the embedded sync client.

        The local adapter is used by the live updater and reconciler.  Keeping
        deletion here is essential: without it, an article that shrinks after
        an edit retains its old surplus chunks forever.
        """
        return await asyncio.to_thread(self._sync.delete, **kwargs)

    async def search(self, **kwargs):
        return await asyncio.to_thread(self._sync.search, **kwargs)

    async def get_collections(self):
        return await asyncio.to_thread(self._sync.get_collections)


def _create_sync_client() -> QdrantClient:
    """Create a Qdrant sync client based on the configured mode."""
    settings = get_settings()

    if settings.qdrant_mode == "local":
        path = settings.qdrant_local_path
        os.makedirs(path, exist_ok=True)
        logger.info("Using Qdrant in EMBEDDED mode (path: %s)", path)
        return QdrantClient(path=path, timeout=60.0)
    else:
        logger.info("Using Qdrant in REMOTE mode (url: %s)", settings.qdrant_url)
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30.0,
        )


def _create_async_client():
    """Create a Qdrant async client based on the configured mode.

    In local (embedded) mode, returns a ``LocalAsyncQdrantAdapter`` that
    wraps the sync client through asyncio.to_thread() to avoid the storage
    folder lock conflict.
    """
    settings = get_settings()

    if settings.qdrant_mode == "local":
        # Wrap the sync client in an async adapter
        logger.info("Async Qdrant: using LocalAsyncQdrantAdapter in embedded mode")
        return LocalAsyncQdrantAdapter(get_sync_qdrant())
    else:
        logger.info("Using async Qdrant in REMOTE mode (url: %s)", settings.qdrant_url)
        return AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=10.0,
        )


def get_async_qdrant():
    """Return a cached async Qdrant client instance (or adapter in local mode)."""
    global _async_client
    if _async_client is None:
        _async_client = _create_async_client()
    return _async_client


def get_sync_qdrant() -> QdrantClient:
    """Return a cached sync Qdrant client instance (useful for batch scripts)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = _create_sync_client()
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
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

        # Create a payload index on the 'title' field for faster exact-match filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
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
