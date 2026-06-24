"""Dual-layer semantic caching module for WikiMind.

Implements a two-tier caching strategy to minimize redundant LLM invocations:

- **L1 (Exact Match)**: SHA-256 hash of the normalized query string stored as
  a Redis string key. Provides sub-millisecond lookups for identical queries.
- **L2 (Semantic Similarity)**: RedisVL ``SemanticCache`` backed by a HNSW
  vector index. Matches semantically equivalent queries that differ in surface
  form using cosine similarity with a configurable threshold (default 0.92).

Cache hits bypass the entire LangGraph pipeline, dramatically reducing latency
and compute costs for repeated or near-duplicate queries.
"""

import hashlib
import json
import logging
import time
from typing import Optional

import redis.asyncio as aioredis

from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection singleton
# ---------------------------------------------------------------------------

_redis_client: Optional[aioredis.Redis] = None


async def get_redis_client() -> aioredis.Redis:
    """Return a cached async Redis client instance.

    Creates the connection on first call and reuses it for the application
    lifetime. Connection parameters are read from the centralized settings.

    Returns:
        aioredis.Redis: Connected Redis client.
    """
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    """Close the Redis connection during application shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed.")


# ---------------------------------------------------------------------------
# L1 Exact-Match Cache
# ---------------------------------------------------------------------------

def _normalize_query(query: str) -> str:
    """Normalize a query string for consistent cache key generation.

    Strips whitespace, lowercases, and removes trailing punctuation to
    ensure that trivially different formulations produce the same hash.

    Args:
        query: Raw user query string.

    Returns:
        Normalized query string.
    """
    return query.strip().lower().rstrip("?.!")


def _hash_query(query: str) -> str:
    """Generate a SHA-256 hash key for a normalized query.

    Args:
        query: Normalized query string.

    Returns:
        Hex-encoded SHA-256 hash prefixed with ``wikimind:cache:l1:``.
    """
    normalized = _normalize_query(query)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"wikimind:cache:l1:{digest}"


async def l1_get(query: str) -> Optional[dict]:
    """Look up an exact-match cached response for the given query.

    Args:
        query: The user's query string.

    Returns:
        Cached response dict if found, None otherwise.
    """
    client = await get_redis_client()
    key = _hash_query(query)

    try:
        cached = await client.get(key)
        if cached is not None:
            logger.info("L1 cache HIT for query hash %s", key[-12:])
            return json.loads(cached)
    except Exception as exc:
        logger.warning("L1 cache lookup failed: %s", exc)

    return None


async def l1_set(query: str, response: dict, ttl: Optional[int] = None) -> None:
    """Store a response in the L1 exact-match cache.

    Args:
        query: The user's query string (will be normalized and hashed).
        response: The response dict to cache.
        ttl: Time-to-live in seconds. Defaults to the static TTL from settings.
    """
    settings = get_settings()
    client = await get_redis_client()
    key = _hash_query(query)
    effective_ttl = ttl or settings.cache_ttl_static

    try:
        await client.setex(key, effective_ttl, json.dumps(response))
        logger.debug("L1 cache SET for query hash %s (TTL=%ds)", key[-12:], effective_ttl)
    except Exception as exc:
        logger.warning("L1 cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# L2 Semantic Cache (RedisVL)
# ---------------------------------------------------------------------------
# The L2 semantic cache uses RedisVL's SemanticCache class which requires
# a Redis instance with the RediSearch module (provided by redis-stack).
# Initialization is deferred to avoid import-time failures when Redis is
# not available (e.g., during testing or local development without Docker).

_semantic_cache = None


async def _get_semantic_cache():
    """Lazy-initialize the RedisVL SemanticCache.

    The cache is initialized on first access. If RedisVL or the Redis
    Search module is unavailable, returns None and logs a warning.

    Returns:
        SemanticCache instance or None.
    """
    global _semantic_cache
    if _semantic_cache is not None:
        return _semantic_cache

    try:
        from redisvl.extensions.llmcache import SemanticCache

        settings = get_settings()
        _semantic_cache = SemanticCache(
            name="wikimind_l2_cache",
            redis_url=settings.redis_url,
            distance_threshold=1.0 - settings.cache_similarity_threshold,
        )
        logger.info(
            "L2 semantic cache initialized (threshold=%.2f).",
            settings.cache_similarity_threshold,
        )
        return _semantic_cache
    except ImportError:
        logger.warning("RedisVL not available. L2 semantic cache disabled.")
        return None
    except Exception as exc:
        logger.warning("L2 semantic cache initialization failed: %s", exc)
        return None


async def l2_get(query: str) -> Optional[dict]:
    """Look up a semantically similar cached response.

    Uses RedisVL's HNSW vector index to find cached responses whose
    query embedding is within the configured cosine similarity threshold.

    Args:
        query: The user's query string.

    Returns:
        Cached response dict if a semantic match is found, None otherwise.
    """
    cache = await _get_semantic_cache()
    if cache is None:
        return None

    try:
        results = cache.check(prompt=query)
        if results:
            best = results[0]
            logger.info(
                "L2 cache HIT (similarity=%.4f) for query: %s",
                best.get("vector_distance", 0.0),
                query[:60],
            )
            response_str = best.get("response", "")
            if response_str:
                return json.loads(response_str)
    except Exception as exc:
        logger.warning("L2 cache lookup failed: %s", exc)

    return None


async def l2_set(query: str, response: dict, ttl: Optional[int] = None) -> None:
    """Store a response in the L2 semantic cache.

    Args:
        query: The user's query string (will be embedded for similarity matching).
        response: The response dict to cache.
        ttl: Time-to-live in seconds. Defaults to the static TTL from settings.
    """
    cache = await _get_semantic_cache()
    if cache is None:
        return

    settings = get_settings()
    effective_ttl = ttl or settings.cache_ttl_static

    try:
        cache.store(
            prompt=query,
            response=json.dumps(response),
            metadata={"ttl": effective_ttl},
        )
        logger.debug("L2 cache SET for query: %s", query[:60])
    except Exception as exc:
        logger.warning("L2 cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Unified Cache Interface
# ---------------------------------------------------------------------------

async def cache_lookup(query: str) -> tuple[Optional[dict], Optional[str]]:
    """Perform a tiered cache lookup (L1 first, then L2).

    The caller should check the returned cache level to log appropriate
    metrics (exact-match vs semantic-match).

    Args:
        query: The user's query string.

    Returns:
        A tuple of (cached_response, cache_level) where cache_level is
        ``"l1"``, ``"l2"``, or None if no cache hit.
    """
    start = time.monotonic()

    # L1: exact match (fastest)
    result = await l1_get(query)
    if result is not None:
        elapsed = (time.monotonic() - start) * 1000
        logger.info("Cache resolved via L1 in %.1fms", elapsed)
        return result, "l1"

    # L2: semantic similarity
    result = await l2_get(query)
    if result is not None:
        elapsed = (time.monotonic() - start) * 1000
        logger.info("Cache resolved via L2 in %.1fms", elapsed)
        return result, "l2"

    elapsed = (time.monotonic() - start) * 1000
    logger.debug("Cache MISS (checked L1+L2 in %.1fms)", elapsed)
    return None, None


async def cache_store(query: str, response: dict, ttl: Optional[int] = None) -> None:
    """Store a response in both L1 and L2 caches.

    Writing to both layers ensures that identical queries are served from
    L1 (sub-millisecond) while semantically similar queries benefit from
    L2 vector matching.

    Args:
        query: The user's query string.
        response: The response dict to cache.
        ttl: Time-to-live in seconds. Defaults to the static TTL from settings.
    """
    await l1_set(query, response, ttl=ttl)
    await l2_set(query, response, ttl=ttl)


async def cache_invalidate(query: str) -> None:
    """Invalidate cached entries for a specific query.

    Removes the L1 exact-match entry. L2 entries will expire via TTL
    since RedisVL does not support targeted deletion by prompt.

    Args:
        query: The query string whose cache entries should be invalidated.
    """
    client = await get_redis_client()
    key = _hash_query(query)
    try:
        await client.delete(key)
        logger.debug("L1 cache invalidated for query hash %s", key[-12:])
    except Exception as exc:
        logger.warning("L1 cache invalidation failed: %s", exc)
