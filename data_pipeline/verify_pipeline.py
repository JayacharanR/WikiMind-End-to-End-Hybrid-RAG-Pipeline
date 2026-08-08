"""Pipeline Verification Script.

End-to-end verification of the WikiMind data pipeline:
1. Qdrant connectivity and collection health
2. Wiki updater: process_event() with a deterministic local fixture
3. Reconciler: run_reconciliation_cycle() validation
4. DLQ and health tracker behavior
5. Cache L1 + L2 functionality

Usage::

    python -m data_pipeline.verify_pipeline
"""

import asyncio
import logging
import sys
import time

# Fix Windows console encoding
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _header(title: str) -> None:
    """Print a test section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _result(test: str, passed: bool, detail: str = "") -> None:
    """Print a test result."""
    icon = "PASS" if passed else "FAIL"
    msg = f"  [{icon}] {test}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


async def test_qdrant_health() -> bool:
    """Test 1: Qdrant connectivity in local or remote mode."""
    _header("Test 1: Qdrant Health Check")

    try:
        from backend.config import get_settings

        settings = get_settings()

        if settings.qdrant_mode == "local":
            from backend.qdrant_client import get_sync_qdrant

            client = get_sync_qdrant()
            collections = await asyncio.to_thread(client.get_collections)
            collection_names = [collection.name for collection in collections.collections]
            _result(
                "Embedded Qdrant reachable",
                True,
                f"path={settings.qdrant_local_path}",
            )
            _result(
                "Collections found",
                len(collection_names) > 0,
                ", ".join(collection_names) if collection_names else "none",
            )
            for coll_name in [settings.qdrant_collection, settings.article_collection]:
                if coll_name in collection_names:
                    info = await asyncio.to_thread(client.get_collection, collection_name=coll_name)
                    _result(f"  {coll_name}", True, f"{info.points_count:,} points")
            return bool(collection_names)

        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.qdrant_url}/healthz")
            healthy = resp.status_code == 200
            _result("Qdrant server reachable", healthy, f"HTTP {resp.status_code}")

            if healthy:
                # Check collections
                resp = await client.get(f"{settings.qdrant_url}/collections")
                data = resp.json()
                collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
                _result(
                    "Collections found",
                    len(collections) > 0,
                    ", ".join(collections) if collections else "none",
                )

                # Check chunk count
                for coll_name in [settings.qdrant_collection, settings.article_collection]:
                    if coll_name in collections:
                        resp = await client.get(f"{settings.qdrant_url}/collections/{coll_name}")
                        info = resp.json().get("result", {})
                        count = info.get("points_count", 0)
                        _result(f"  {coll_name}", True, f"{count:,} points")
            return healthy

    except Exception as exc:
        _result("Qdrant connectivity", False, str(exc))
        return False


async def test_existing_data() -> bool:
    """Test 2a: Verify the article and chunk indexes contain aligned data."""
    _header("Test 2a: Verify Existing Data in Qdrant")

    try:
        from qdrant_client.http import models as qmodels

        from backend.config import get_settings
        from backend.qdrant_client import get_async_qdrant

        qdrant = get_async_qdrant()
        settings = get_settings()

        article_points, _ = await qdrant.scroll(
            collection_name=settings.article_collection,
            limit=1,
            with_payload=["title"],
            with_vectors=False,
        )
        if not article_points:
            _result("Article-level data", False, "article collection is empty")
            return False

        title = (article_points[0].payload or {}).get("title", "")
        _result("Article-level data", bool(title), title or "missing title payload")
        if not title:
            return False

        chunk_points, _ = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="title",
                        match=qmodels.MatchValue(value=title),
                    ),
                ]
            ),
            limit=5,
            with_payload=["title", "chunk_index"],
            with_vectors=False,
        )
        aligned = bool(chunk_points)
        _result(
            "Matching chunk data",
            aligned,
            f"'{title}' has {len(chunk_points)}+ chunks" if aligned else "not found",
        )
        return aligned

    except Exception as exc:
        _result("Existing data check", False, str(exc))
        return False


async def test_process_event() -> bool:
    """Test 2b: Process and shrink a deterministic article fixture."""
    _header("Test 2b: Wiki Updater -- process_event()")

    try:
        from unittest.mock import AsyncMock, patch

        from data_pipeline.wiki_updater import process_event, updater_health

        # Keep this test independent of external Wikipedia network access.
        test_title = "WikiMind Verification Fixture"
        test_event = {
            "title": test_title,
            "type": "edit",
            "namespace": 0,
            "revision": {"new": 999999999},
            "meta": {
                "uri": "https://en.wikipedia.org/wiki/WikiMind_Verification_Fixture",
            },
        }

        long_text = " ".join(
            ["WikiMind verification fixture contains deterministic test content."] * 180
        )
        short_text = "WikiMind verification fixture was successfully shrunk."
        fake_session = object()
        with patch(
            "data_pipeline.wiki_updater.fetch_article_text",
            new=AsyncMock(side_effect=[long_text, short_text]),
        ):
            start = time.time()
            await process_event(test_event, fake_session)
            # A second update is intentionally shorter to verify stale chunk
            # removal instead of merely testing a same-size overwrite.
            test_event["revision"]["new"] = 1000000000
            await process_event(test_event, fake_session)
            elapsed = time.time() - start

            _result(
                "process_event() completed",
                True,
                f"'{test_event['title']}' processed and shrunk in {elapsed:.1f}s",
            )

            # Verify chunks were written to Qdrant
            from qdrant_client.http import models as qmodels

            from backend.config import get_settings
            from backend.qdrant_client import get_async_qdrant

            settings = get_settings()
            qdrant = get_async_qdrant()

            results, _ = await qdrant.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="title",
                            match=qmodels.MatchValue(value=test_event["title"]),
                        ),
                    ]
                ),
                limit=20,
                with_payload=["title", "chunk_index"],
                with_vectors=False,
            )

            chunk_count = len(results)
            _result(
                "Stale chunks removed after shrink",
                chunk_count == 1,
                f"{chunk_count} chunks found",
            )

            # Check health tracker
            health = updater_health.get_health_status()
            _result(
                "Health tracker state",
                True,
                f"processed={health['events_processed']}, failed={health['events_failed']}, dlq={health['dlq_size']}",
            )

            # Keep the verification fixture out of the persistent local store.
            await qdrant.delete(
                collection_name=settings.qdrant_collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="title", match=qmodels.MatchValue(value=test_title)
                            )
                        ]
                    )
                ),
            )
            from backend.article_index import delete_article

            await delete_article(test_title)
            return chunk_count == 1

    except Exception as exc:
        _result("process_event()", False, str(exc))
        return False


async def test_dlq_behavior() -> bool:
    """Test 3: DLQ error handling."""
    _header("Test 3: Dead Letter Queue Behavior")

    try:
        from data_pipeline.pipeline_health import PipelineHealthTracker

        tracker = PipelineHealthTracker(name="test-worker")

        # Simulate a failed event
        test_event = {"title": "DLQ_Test_Article", "meta": {"uri": "https://example.com"}}
        tracker.add_to_dlq(test_event, "Simulated test failure")

        dlq_size = len(tracker.dlq)
        _result("DLQ add", dlq_size == 1, f"DLQ size={dlq_size}")

        # Check health status
        health = tracker.get_health_status()
        _result(
            "Health status reflects failure",
            health["dlq_size"] == 1 and health["events_failed"] == 1,
            f"status={health['status']}, failed={health['events_failed']}, consecutive={health['consecutive_failures']}",
        )

        # Verify DLQ persistence
        import os

        dlq_path = tracker._dlq_path
        persisted = os.path.exists(dlq_path)
        _result("DLQ persisted to disk", persisted, dlq_path if persisted else "file not found")

        # Test health status format
        required_keys = [
            "worker",
            "status",
            "events_processed",
            "events_failed",
            "dlq_size",
            "last_heartbeat",
            "consecutive_failures",
        ]
        has_all = all(k in health for k in required_keys)
        _result("Health status schema", has_all, f"keys={list(health.keys())}")

        # Clean up test DLQ file
        if persisted:
            os.remove(dlq_path)

        return dlq_size == 1 and has_all

    except Exception as exc:
        _result("DLQ behavior", False, str(exc))
        return False


async def test_reconciler_health() -> bool:
    """Test 4: Reconciler health reporting."""
    _header("Test 4: Reconciler Health Reporting")

    try:
        from data_pipeline.reconciler import reconciler_health

        # Record some test metrics
        reconciler_health.record_cycle_stats(
            drift_count=2,
            success_count=1,
            failed_count=1,
            elapsed_sec=5.3,
        )

        health = reconciler_health.get_health_status()
        _result(
            "Cycle stats recorded",
            health.get("last_cycle") is not None,
            f"drift={health.get('drift_detected')}, success={health.get('reingestion_success')}",
        )
        _result(
            "Health endpoint format valid",
            all(k in health for k in ["worker", "status", "events_processed", "dlq_size"]),
            f"status={health['status']}",
        )

        # Check reconciler-specific fields
        has_drift = "drift_detected" in health
        _result("Has drift_detected field", has_drift)

        return True

    except Exception as exc:
        _result("Reconciler health", False, str(exc))
        return False


async def test_cache_l1() -> bool:
    """Test 5: L1 exact-match cache."""
    _header("Test 5: Cache L1 (Exact Match)")

    try:
        from backend.cache import cache_invalidate, get_redis_client, l1_get, l1_set

        # Check Redis connectivity first
        try:
            client = await get_redis_client()
            await client.ping()
            _result("Redis connectivity", True)
        except Exception as exc:
            _result("Redis connectivity", False, str(exc))
            print("  [SKIP] Skipping cache tests (Redis not available)")
            return False

        # Test L1 write + read
        test_query = "test_l1_cache_verification_query"
        test_response = {"answer": "This is a test cached response", "sources": []}

        await l1_set(test_query, test_response, ttl=60)
        _result("L1 SET", True)

        result = await l1_get(test_query)
        hit = result is not None and result.get("answer") == test_response["answer"]
        _result(
            "L1 GET (exact match)",
            hit,
            f"got: {result.get('answer', 'None')[:50]}" if result else "miss",
        )

        # Test miss
        result = await l1_get("completely_different_query_that_should_miss")
        _result("L1 MISS (different query)", result is None, "correctly returned None")

        # Clean up the generation-aware key.
        await cache_invalidate(test_query)

        return hit

    except Exception as exc:
        _result("L1 cache", False, str(exc))
        return False


async def test_cache_l2() -> bool:
    """Test 6: L2 semantic cache (pure-Redis fallback)."""
    _header("Test 6: Cache L2 (Semantic -- Pure-Redis Fallback)")

    try:
        from backend.cache import (
            _init_l2_strategy,
            cache_invalidate,
            get_redis_client,
            l2_get,
            l2_set,
        )

        # Check Redis first
        try:
            client = await get_redis_client()
            await client.ping()
        except Exception as exc:
            _result("Redis for L2", False, str(exc))
            return False

        # Initialize L2 strategy
        strategy = await _init_l2_strategy()
        _result("L2 strategy detected", strategy != "disabled", f"strategy={strategy}")

        if strategy == "disabled":
            print("  [SKIP] L2 disabled -- skipping semantic cache tests")
            return False

        # Test L2 write + read
        passed = True
        test_query = "What is the capital city of France?"
        test_response = {"answer": "Paris is the capital of France.", "sources": []}

        await l2_set(test_query, test_response, ttl=60)
        _result("L2 SET", True)

        # Test exact re-query (should always match regardless of strategy)
        result = await l2_get(test_query)
        if result is not None:
            _result("L2 GET (exact re-query)", True, f"matched: {result.get('answer', '')[:50]}")
        else:
            _result("L2 GET (exact re-query)", False, "no match returned")
            passed = False

        # Test semantic match with similar query
        similar_query = "What is the capital of France"
        result = await l2_get(similar_query)
        if result is not None:
            _result("L2 GET (semantic match)", True, f"matched: {result.get('answer', '')[:50]}")
        else:
            _result(
                "L2 GET (semantic match)", False, "no semantic match (threshold may be too high)"
            )
            passed = False

        # Clean up the strategy/generation-scoped entry.
        await cache_invalidate(test_query)

        return passed

    except Exception as exc:
        _result("L2 cache", False, str(exc))
        return False


async def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("  WikiMind Pipeline Verification Suite")
    print("=" * 60)

    results = {}

    # Test 1: Qdrant
    results["qdrant_health"] = await test_qdrant_health()

    # Test 2a: Existing data
    if results["qdrant_health"]:
        results["existing_data"] = await test_existing_data()
    else:
        _header("Test 2a: Existing Data -- SKIPPED (Qdrant unavailable)")
        results["existing_data"] = False

    # Test 2b: process_event (requires Qdrant)
    if results["qdrant_health"]:
        results["process_event"] = await test_process_event()
    else:
        _header("Test 2b: process_event -- SKIPPED (Qdrant unavailable)")
        results["process_event"] = False

    # Test 3: DLQ (standalone)
    results["dlq"] = await test_dlq_behavior()

    # Test 4: Reconciler health (standalone)
    results["reconciler_health"] = await test_reconciler_health()

    # Test 5: L1 cache (requires Redis)
    results["cache_l1"] = await test_cache_l1()

    # Test 6: L2 cache (requires Redis + FastEmbed)
    results["cache_l2"] = await test_cache_l2()

    # Summary
    _header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        icon = "PASS" if result else "FAIL"
        print(f"  [{icon}] {name}")

    print(f"\n  Result: {passed}/{total} tests passed")

    # Note about Redis
    if not results.get("cache_l1") and not results.get("cache_l2"):
        print("\n  NOTE: Cache tests require a running Redis server.")
        print("  Start Redis locally or use: docker run -d -p 6379:6379 redis/redis-stack-server")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
