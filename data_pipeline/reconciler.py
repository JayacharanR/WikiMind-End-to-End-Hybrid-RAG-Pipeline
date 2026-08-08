"""State Reconciliation Worker.

Periodically samples a batch of articles from the Qdrant vector database,
fetches their latest revision timestamps from the live Wikipedia API, and
compares them. If drift is detected (the Qdrant article is stale), it triggers
a re-ingestion of that article to self-heal the knowledge base.

Drift metrics and reconciliation runs are logged to Langfuse for observability.
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List

import aiohttp

from backend.config import get_settings
from backend.llmops import get_langfuse_client
from backend.qdrant_client import get_async_qdrant
from data_pipeline.pipeline_health import PipelineHealthTracker
from data_pipeline.wiki_updater import process_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global health tracker
reconciler_health = PipelineHealthTracker(name="reconciler")

# Constants
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
SAMPLE_SIZE = 100


async def get_random_titles_from_qdrant(limit: int = SAMPLE_SIZE) -> List[str]:
    """Sample random article titles from the article-level index.
    
    Uses the ``wikimind_articles`` collection (one point per article) instead
    of the chunk collection, so each scroll result is a unique article.
    Applies a random offset to avoid always sampling the same first page.
    """
    qdrant = get_async_qdrant()
    settings = get_settings()
    # Prefer the article-level collection for sampling (1 point per article)
    collection = settings.article_collection
    
    try:
        # Get collection stats to find total articles
        col_info = await qdrant.get_collection(collection_name=collection)
        total_points = col_info.points_count
        
        if total_points == 0:
            # Fallback to chunk collection if article index is empty
            collection = settings.qdrant_collection
            col_info = await qdrant.get_collection(collection_name=collection)
            total_points = col_info.points_count
            if total_points == 0:
                return []
        
        # Use random offset for diverse sampling across the collection.
        # Generate a random UUID as a starting point for scroll so each
        # reconciliation cycle checks a different region of the collection.
        import uuid
        max_offset = max(0, total_points - limit)
        random_offset_id = str(uuid.UUID(int=random.randint(0, 2**128 - 1)))

        results, _ = await qdrant.scroll(
            collection_name=collection,
            limit=min(limit * 3, total_points),  # Over-fetch to account for dedup
            offset=random_offset_id,
            with_payload=["title"],
            with_vectors=False
        )

        # If random offset returned fewer results than desired, wrap around
        # with a second scroll from the start
        if len(results) < limit * 2:
            wrap_results, _ = await qdrant.scroll(
                collection_name=collection,
                limit=min(limit * 2, total_points),
                with_payload=["title"],
                with_vectors=False
            )
            results = list(results) + list(wrap_results)

        # Deduplicate titles and randomly sample from results
        all_titles = list(set(
            point.payload.get("title")
            for point in results
            if point.payload and point.payload.get("title")
        ))
        
        # Random sample if we have more than requested
        if len(all_titles) > limit:
            all_titles = random.sample(all_titles, limit)
        
        logger.info("Reconciler sampled %d unique titles from %d total points (offset=%s...)", 
                     len(all_titles), total_points, random_offset_id[:8])
        return all_titles
    except Exception as exc:
        logger.error("Error sampling from Qdrant: %s", exc)
        return []


async def check_live_revisions(session: aiohttp.ClientSession, titles: List[str]) -> List[tuple]:
    """Check live Wikipedia revisions and compare to stored revision_ids.

    Queries the MediaWiki API for the latest revision ID of each title,
    then scrolls the Qdrant collection to find the stored revision_id for
    each article. Articles where the stored revision_id differs from (or
    is older than) the live revision are flagged as stale.

    Args:
        session: aiohttp session for API requests.
        titles: List of article titles to check.

    Returns:
        List of (title, live_revision_id) tuples for articles that need re-ingestion.
    """
    if not titles:
        return []

    stale_titles = []

    # Batch fetch live revision IDs from MediaWiki API (up to 50 per request)
    for batch_start in range(0, len(titles), 50):
        batch = titles[batch_start:batch_start + 50]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "prop": "revisions",
            "rvprop": "ids",
            "rvlimit": "1",
        }

        try:
            async with session.get(WIKI_API_URL, params=params, timeout=15) as resp:
                if resp.status != 200:
                    logger.warning("MediaWiki API returned HTTP %d", resp.status)
                    continue

                data = await resp.json()
                pages = data.get("query", {}).get("pages", {})

                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        continue
                    page_title = page_data.get("title", "")
                    revisions = page_data.get("revisions", [])
                    if not revisions:
                        continue

                    live_revid = str(revisions[0].get("revid", ""))

                    # Fetch stored revision info from Qdrant
                    stored_revid, rev_source = await _get_stored_revision(page_title)

                    # Dataset-snapshot IDs are not MediaWiki revisions —
                    # always treat as needing refresh on first reconciliation
                    if rev_source == "dataset_snapshot":
                        logger.info(
                            "Refreshing '%s': batch-ingested (source_document_id=%s, not a revid)",
                            page_title, stored_revid,
                        )
                        stale_titles.append((page_title, live_revid))
                    elif stored_revid and stored_revid != live_revid:
                        logger.info(
                            "Drift detected for '%s': stored=%s, live=%s",
                            page_title, stored_revid, live_revid,
                        )
                        stale_titles.append((page_title, live_revid))
                    elif not stored_revid:
                        # No revision_id stored (legacy data), flag for refresh
                        stale_titles.append((page_title, live_revid))

        except Exception as exc:
            logger.warning("Error checking live revisions: %s", exc)

    return stale_titles


async def _get_stored_revision(title: str) -> tuple:
    """Retrieve the stored revision info for an article from Qdrant.

    Returns:
        Tuple of (revision_id_or_source_doc_id, revision_source).
        revision_source is 'live', 'dataset_snapshot', or '' if unknown.
    """
    qdrant = get_async_qdrant()
    settings = get_settings()

    try:
        from qdrant_client.http import models as qmodels
        results, _ = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="title",
                        match=qmodels.MatchValue(value=title),
                    ),
                ]
            ),
            limit=1,
            with_payload=["revision_id", "source_document_id", "revision_source"],
            with_vectors=False,
        )
        if results:
            payload = results[0].payload
            rev_source = payload.get("revision_source", "")
            # Live-sync data uses revision_id; batch data uses source_document_id
            stored_id = payload.get("revision_id", "") or payload.get("source_document_id", "")
            return stored_id, rev_source
    except Exception as exc:
        logger.warning("Error reading stored revision for '%s': %s", title, exc)

    return "", ""


async def run_reconciliation_cycle():
    """Run a single reconciliation cycle with error isolation and drift metrics."""
    logger.info("Beginning reconciliation cycle...")
    cycle_start = time.time()
    drift_count = 0
    success_count = 0
    failed_count = 0

    try:
        titles = await get_random_titles_from_qdrant(limit=SAMPLE_SIZE)
        if not titles:
            logger.info("Qdrant collection empty. Sleeping...")
            return
            
        logger.info("Sampled %d unique articles for reconciliation.", len(titles))
        
        headers = {"User-Agent": "WikiMindBot/1.0 (https://github.com/JayacharanR/End-to-End-Hybrid-RAG-Pipeline)"}
        async with aiohttp.ClientSession(headers=headers) as session:
            stale_titles = await check_live_revisions(session, titles)
            drift_count = len(stale_titles)
            
            if stale_titles:
                logger.warning("Detected drift in %d articles. Re-ingesting...", len(stale_titles))
                for title, live_rev in stale_titles:
                    fake_event = {
                        "title": title,
                        "meta": {"uri": f"https://en.wikipedia.org/wiki/{title}"},
                        "revision": {"new": live_rev},
                    }
                    try:
                        await process_event(fake_event, session)
                        success_count += 1
                        reconciler_health.record_success(event_id=title)
                        logger.info("Re-ingestion succeeded for '%s'", title)
                    except Exception as exc:
                        failed_count += 1
                        reconciler_health.add_to_dlq(
                            event=fake_event, error_msg=str(exc)
                        )
                        logger.warning("Re-ingestion failed for '%s': %s", title, exc)
            else:
                logger.info("No drift detected in sample.")

        # Retry any DLQ items from previous cycles
        if reconciler_health.dlq:
            logger.info("Retrying %d DLQ items from previous cycles...", len(reconciler_health.dlq))
            async with aiohttp.ClientSession(headers=headers) as retry_session:
                await reconciler_health.retry_dlq(
                    process_fn=process_event,
                    session=retry_session,
                )

    except Exception as exc:
        reconciler_health.record_failure(error=str(exc))
        logger.error("Reconciliation cycle failed: %s", exc)

    # Record cycle metrics
    elapsed = time.time() - cycle_start
    reconciler_health.record_cycle_stats(
        drift_count=drift_count,
        success_count=success_count,
        failed_count=failed_count,
        elapsed_sec=elapsed,
    )
    logger.info(
        "Reconciliation cycle complete: drift=%d, success=%d, failed=%d, elapsed=%.1fs",
        drift_count, success_count, failed_count, elapsed,
    )


async def reconcile_loop():
    """Main reconciliation loop."""
    settings = get_settings()
    interval_hours = settings.wiki_reconcile_interval_hours
    interval_seconds = interval_hours * 3600
    
    logger.info("Starting State Reconciliation Worker. Interval: %dh", interval_hours)
    
    while True:
        start_time = time.time()
        
        await run_reconciliation_cycle()
                
        # Flush traces
        langfuse = get_langfuse_client()
        if langfuse:
            langfuse.flush()
            
        elapsed = time.time() - start_time
        sleep_time = max(0, interval_seconds - elapsed)
        logger.info("Reconciliation cycle complete (took %.1fs). Sleeping for %.1fh...", elapsed, sleep_time / 3600)
        await asyncio.sleep(sleep_time)


def get_reconciler_health() -> Dict[str, Any]:
    """Return the health status of the reconciler worker."""
    return reconciler_health.get_health_status()


if __name__ == "__main__":
    asyncio.run(reconcile_loop())
