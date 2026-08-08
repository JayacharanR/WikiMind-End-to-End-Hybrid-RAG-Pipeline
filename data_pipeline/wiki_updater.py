"""Wikipedia EventStreams Listener.

Connects to the Wikimedia EventStreams SSE API to listen for live edits to
the English Wikipedia in real-time. Fetches the updated article content via
the MediaWiki API, chunks, embeds, and performs idempotent upserts into Qdrant.

Implements exponential backoff with a cap, a Dead Letter Queue (DLQ) for
persistently failed events, periodic DLQ retry, heartbeat tracking, and
graceful shutdown with DLQ persistence.
"""

import asyncio
import json
import logging
import signal
from typing import Any, Dict, Optional

import aiohttp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

from backend.config import get_settings
from backend.qdrant_client import generate_point_id, get_async_qdrant
from data_pipeline.ingest import get_dense_model, get_sparse_model
from data_pipeline.pipeline_health import PipelineHealthTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
MAX_BACKOFF = 300  # 5 minute cap on exponential backoff
BASE_BACKOFF = 2.0
DLQ_RETRY_INTERVAL = 100  # retry DLQ every N events processed

# Global health tracker
updater_health = PipelineHealthTracker(name="wiki-updater")


async def fetch_article_text(session: aiohttp.ClientSession, title: str) -> Optional[str]:
    """Fetch the raw markdown/text of a Wikipedia article using the MediaWiki API."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
    }
    try:
        req_timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(WIKI_API_URL, params=params, timeout=req_timeout) as response:
            if response.status == 403:
                logger.warning(
                    "Wikipedia API returned 403 for '%s'. "
                    "Ensure session has a proper User-Agent header.",
                    title,
                )
                return None
            if response.status != 200:
                logger.warning("Wikipedia API returned HTTP %d for '%s'.", response.status, title)
                return None
            data = await response.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    return None
                return page_data.get("extract", "")
    except Exception as exc:
        logger.warning("Error fetching article '%s': %s", title, exc)
    return None


def _title_filter(title: str) -> models.Filter:
    """Build the common Qdrant filter for all points belonging to a title."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="title",
                match=models.MatchValue(value=title),
            )
        ]
    )


async def _delete_all_article_chunks(qdrant, collection_name: str, title: str) -> None:
    """Delete every chunk for an article and fail if Qdrant rejects it."""
    await qdrant.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=_title_filter(title)),
    )


async def _remove_stale_article_chunks(
    qdrant,
    collection_name: str,
    title: str,
    keep_ids: set[str],
) -> None:
    """Remove old point IDs after the replacement points are safely upserted.

    Upserting first avoids a temporary data-loss window if embedding or
    upsert fails. Deleting only IDs that are not part of the new generation
    also handles articles that shrink and legacy non-contiguous chunk IDs.
    """
    stale_ids: list[str] = []
    offset = None
    while True:
        kwargs = {
            "collection_name": collection_name,
            "scroll_filter": _title_filter(title),
            "limit": 256,
            "with_payload": False,
            "with_vectors": False,
        }
        if offset is not None:
            kwargs["offset"] = offset
        points, next_offset = await qdrant.scroll(**kwargs)
        stale_ids.extend(str(point.id) for point in points if str(point.id) not in keep_ids)
        if next_offset is None:
            break
        offset = next_offset

    if stale_ids:
        await qdrant.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=stale_ids),
        )


async def process_event(event_data: Dict[str, Any], session: aiohttp.ClientSession) -> None:
    """Process a single Wikipedia edit event.

    Fetches the latest article text, chunks it, generates dual embeddings,
    and upserts the new chunks into Qdrant. Uses deterministic point IDs
    (based on title + chunk_index) so updates overwrite existing chunks
    for the same article.
    """
    title = event_data.get("title")
    meta = event_data.get("meta", {})
    uri = meta.get("uri", "")
    revision_id = str(event_data.get("revision", {}).get("new", ""))
    event_type = event_data.get("type", "edit")

    if not title:
        return

    logger.info("Processing update for: %s (revision %s)", title, revision_id)

    qdrant = get_async_qdrant()
    settings = get_settings()

    # Deletion events carry enough information to remove the current state
    # without making a second API request.
    if event_type == "delete":
        await _delete_all_article_chunks(qdrant, settings.qdrant_collection, title)
        from backend.article_index import delete_article

        await delete_article(title)
        from backend.cache import bump_cache_generation

        await bump_cache_generation()
        logger.info("Deleted all indexed state for '%s'", title)
        return

    # 1. Fetch updated content. None means the fetch failed; an empty string
    # is a confirmed empty page and must remove the previous indexed state.
    text = await fetch_article_text(session, title)
    if text is None:
        raise RuntimeError(f"Failed to fetch article text for '{title}'")
    if not text.strip():
        await _delete_all_article_chunks(qdrant, settings.qdrant_collection, title)
        from backend.article_index import delete_article

        await delete_article(title)
        from backend.cache import bump_cache_generation

        await bump_cache_generation()
        logger.info("Removed empty article '%s' from all indexes", title)
        return

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise RuntimeError(f"Article '{title}' produced no chunks")

    # 3. Embedding
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    dense_embeddings, sparse_embeddings = await asyncio.gather(
        asyncio.to_thread(lambda: list(dense_model.embed(chunks))),
        asyncio.to_thread(lambda: list(sparse_model.embed(chunks))),
    )

    # 4. Build points with deterministic IDs
    from datetime import datetime, timezone

    ingested_at = datetime.now(timezone.utc).isoformat()

    qdrant_points = []
    for i, chunk_text in enumerate(chunks):
        point_id = generate_point_id(title, i)
        sparse_obj = sparse_embeddings[i]

        qdrant_points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_obj.indices.tolist(),
                        values=sparse_obj.values.tolist(),
                    ),
                },
                payload={
                    "title": title,
                    "url": uri,
                    "page_content": chunk_text,
                    "chunk_index": i,
                    "revision_id": revision_id,
                    "revision_source": "live",
                    "ingested_at": ingested_at,
                },
            )
        )

    await qdrant.upsert(collection_name=settings.qdrant_collection, points=qdrant_points)

    # 5. Remove surplus/legacy chunks only after the new content is stored.
    keep_ids = {str(point.id) for point in qdrant_points}
    await _remove_stale_article_chunks(qdrant, settings.qdrant_collection, title, keep_ids)
    logger.debug("Upserted %d chunks for '%s' (revision %s)", len(chunks), title, revision_id)

    # 6. Update derived indexes and invalidate answer caches. Failures are
    # propagated so the event is retried through the DLQ.
    from backend.article_index import extract_article_summary, upsert_article

    await upsert_article(title, extract_article_summary(title, text), uri)
    from backend.cache import bump_cache_generation

    await bump_cache_generation()
    logger.debug("Updated article index and invalidated caches for '%s'", title)


async def listen_to_stream():
    """Main event loop listening to Wikimedia EventStreams.

    Self-healing features:
    - Exponential backoff with a 5-minute cap (never exits fatally)
    - DLQ for failed event processing with periodic retry
    - Heartbeat tracking for external health monitoring
    - Graceful shutdown on SIGTERM/SIGINT (flushes DLQ to disk)
    """
    settings = get_settings()
    stream_url = settings.wiki_stream_url

    # Initialize Qdrant collection if it doesn't exist
    from backend.qdrant_client import init_collection

    # Run sync init_collection in thread
    await asyncio.to_thread(init_collection)

    # Graceful shutdown handler
    shutdown_event = asyncio.Event()

    def _handle_shutdown(signum, frame):
        logger.info("Received signal %s. Initiating graceful shutdown...", signum)
        updater_health._save_dlq()
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    retry_count = 0
    events_since_dlq_retry = 0

    while not shutdown_event.is_set():
        try:
            logger.info("Connecting to Wikimedia EventStreams...")
            headers = {
                "User-Agent": "WikiMindBot/1.0 (https://github.com/JayacharanR/End-to-End-Hybrid-RAG-Pipeline)"
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(
                    stream_url, headers={"Accept": "text/event-stream"}
                ) as response:
                    if response.status != 200:
                        logger.error("Failed to connect: HTTP %d", response.status)
                        raise Exception(f"Connection failed: HTTP {response.status}")

                    retry_count = 0  # reset on successful connection
                    logger.info("Connected successfully. Listening for events...")

                    async for line in response.content:
                        if shutdown_event.is_set():
                            break

                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                event = json.loads(data_str)
                                # Filter: English Wikipedia, namespace 0 (Main
                                # articles), including edits, new pages, and
                                # deletions so latest-state semantics hold.
                                if (
                                    event.get("server_name") == "en.wikipedia.org"
                                    and event.get("namespace") == 0
                                    and event.get("type") in {"edit", "new", "delete"}
                                ):
                                    # Process event with DLQ error handling
                                    try:
                                        await process_event(event, session)
                                        updater_health.record_success(
                                            event_id=event.get("title", "")
                                        )
                                        events_since_dlq_retry += 1
                                    except Exception as e:
                                        updater_health.add_to_dlq(event=event, error_msg=str(e))

                                    # Periodic DLQ retry
                                    if events_since_dlq_retry >= DLQ_RETRY_INTERVAL:
                                        events_since_dlq_retry = 0
                                        if updater_health.dlq:
                                            logger.info(
                                                "Periodic DLQ retry (%d items)...",
                                                len(updater_health.dlq),
                                            )
                                            await updater_health.retry_dlq(
                                                process_fn=process_event,
                                                session=session,
                                            )

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            retry_count += 1
            # Capped exponential backoff (never exits fatally)
            sleep_time = min(BASE_BACKOFF**retry_count, MAX_BACKOFF)
            updater_health.record_failure(error=str(e))
            logger.warning(
                "Stream disconnected (attempt %d). Retrying in %.1fs... (%s)",
                retry_count,
                sleep_time,
                e,
            )
            await asyncio.sleep(sleep_time)

    # Final DLQ flush on shutdown
    updater_health._save_dlq()
    logger.info(
        "Wiki updater stopped. Processed: %d, Failed: %d, DLQ: %d",
        updater_health.events_processed,
        updater_health.events_failed,
        len(updater_health.dlq),
    )


def get_updater_health() -> Dict[str, Any]:
    """Return the health status of the wiki updater worker."""
    return updater_health.get_health_status()


if __name__ == "__main__":
    asyncio.run(listen_to_stream())
