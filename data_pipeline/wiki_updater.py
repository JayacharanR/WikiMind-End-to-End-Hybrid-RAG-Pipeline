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
from sse_starlette.sse import ServerSentEvent

from backend.config import get_settings
from backend.qdrant_client import generate_point_id, get_async_qdrant
from data_pipeline.ingest import get_dense_model, get_sparse_model
from data_pipeline.pipeline_health import PipelineHealthTracker
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


async def process_event(event_data: Dict[str, Any], session: aiohttp.ClientSession) -> None:
    """Process a single Wikipedia edit event with version-aware upserts.

    Before inserting new chunks, marks all existing chunks for the article
    as ``is_current=false`` so time-travel queries can distinguish versions.
    """
    title = event_data.get("title")
    meta = event_data.get("meta", {})
    uri = meta.get("uri", "")
    revision_id = str(event_data.get("revision", {}).get("new", ""))
    
    if not title:
        return

    logger.info("Processing update for: %s (revision %s)", title, revision_id)
    
    # 1. Fetch updated content
    text = await fetch_article_text(session, title)
    if not text:
        logger.warning("Could not fetch text for %s. Skipping.", title)
        return

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return

    # 3. Mark existing chunks as not current (version archival)
    qdrant = get_async_qdrant()
    settings = get_settings()
    
    try:
        await qdrant.set_payload(
            collection_name=settings.qdrant_collection,
            payload={"is_current": False},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="title",
                        match=models.MatchValue(value=title),
                    ),
                    models.FieldCondition(
                        key="is_current",
                        match=models.MatchValue(value=True),
                    ),
                ]
            ),
        )
        logger.debug("Archived existing chunks for: %s", title)
    except Exception as exc:
        logger.warning("Failed to archive old chunks for %s: %s", title, exc)

    # 4. Embedding
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    dense_embeddings = list(dense_model.embed(chunks))
    sparse_embeddings = list(sparse_model.embed(chunks))

    # 5. Build new versioned points
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
                    )
                },
                payload={
                    "title": title,
                    "url": uri,
                    "page_content": chunk_text,
                    "chunk_index": i,
                    "revision_id": revision_id,
                    "ingested_at": ingested_at,
                    "is_current": True,
                }
            )
        )

    await qdrant.upsert(
        collection_name=settings.qdrant_collection,
        points=qdrant_points
    )
    logger.debug("Successfully updated %s (%d chunks, revision %s)", title, len(chunks), revision_id)


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
            headers = {"User-Agent": "WikiMindBot/1.0 (https://github.com/JayacharanR/End-to-End-Hybrid-RAG-Pipeline)"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(stream_url, headers={"Accept": "text/event-stream"}) as response:
                    if response.status != 200:
                        logger.error("Failed to connect: HTTP %d", response.status)
                        raise Exception(f"Connection failed: HTTP {response.status}")
                        
                    retry_count = 0  # reset on successful connection
                    logger.info("Connected successfully. Listening for events...")
                    
                    async for line in response.content:
                        if shutdown_event.is_set():
                            break

                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                event = json.loads(data_str)
                                # Filter: English Wikipedia, namespace 0 (Main articles), type 'edit'
                                if (event.get("server_name") == "en.wikipedia.org" and
                                    event.get("namespace") == 0 and
                                    event.get("type") == "edit"):
                                    
                                    # Process event with DLQ error handling
                                    try:
                                        await process_event(event, session)
                                        updater_health.record_success(
                                            event_id=event.get("title", "")
                                        )
                                        events_since_dlq_retry += 1
                                    except Exception as e:
                                        updater_health.add_to_dlq(
                                            event=event, error_msg=str(e)
                                        )

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
            sleep_time = min(BASE_BACKOFF ** retry_count, MAX_BACKOFF)
            updater_health.record_failure(error=str(e))
            logger.warning(
                "Stream disconnected (attempt %d). Retrying in %.1fs... (%s)",
                retry_count, sleep_time, e,
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
