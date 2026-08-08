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


_CACHE_GENERATION_KEY = "wikimind:cache:knowledge_generation"


def _strategy_scope(strategies: dict = None, generation: str = "0") -> str:
    """Return a stable cache namespace for strategy flags and KB generation."""
    active = sorted(k for k, value in (strategies or {}).items() if value)
    strategy_name = ",".join(active) or "default"
    return f"generation={generation};strategies={strategy_name}"


async def get_cache_generation() -> str:
    """Read the shared knowledge-base generation used to invalidate answers."""
    try:
        client = await get_redis_client()
        generation = await client.get(_CACHE_GENERATION_KEY)
        if generation is None:
            await client.set(_CACHE_GENERATION_KEY, "0", nx=True)
            return "0"
        return str(generation)
    except Exception as exc:
        logger.warning("Cache generation lookup failed: %s", exc)
        return "0"


async def bump_cache_generation() -> str:
    """Invalidate all answer-cache namespaces after a source update."""
    try:
        client = await get_redis_client()
        generation = await client.incr(_CACHE_GENERATION_KEY)
        logger.info("Advanced knowledge-base cache generation to %s", generation)
        return str(generation)
    except Exception as exc:
        logger.warning("Cache generation invalidation failed: %s", exc)
        return "0"


def _hash_query(query: str, strategies: dict = None, generation: str = "0") -> str:
    """Generate a SHA-256 hash key for a normalized query + strategies.

    Includes active strategies in the hash so the same query with different
    strategy configurations produces distinct cache keys.

    Args:
        query: Raw query string.
        strategies: Optional dict of strategy toggles (e.g., {"multi_query": True}).

    Returns:
        Hex-encoded SHA-256 hash prefixed with ``wikimind:cache:l1:``.
    """
    normalized = _normalize_query(query)
    normalized += "|" + _strategy_scope(strategies, generation)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"wikimind:cache:l1:{digest}"


async def l1_get(query: str, strategies: dict = None) -> Optional[dict]:
    """Look up an exact-match cached response for the given query.

    Args:
        query: The user's query string.
        strategies: Optional strategy toggles to include in cache key.

    Returns:
        Cached response dict if found, None otherwise.
    """
    try:
        generation = await get_cache_generation()
        key = _hash_query(query, strategies, generation)
        client = await get_redis_client()
        cached = await client.get(key)
        if cached is not None:
            logger.info("L1 cache HIT for query hash %s", key[-12:])
            return json.loads(cached)
    except Exception as exc:
        logger.warning("L1 cache lookup failed: %s", exc)

    return None


async def l1_set(
    query: str, response: dict, ttl: Optional[int] = None, strategies: dict = None
) -> None:
    """Store a response in the L1 exact-match cache.

    Args:
        query: The user's query string (will be normalized and hashed).
        response: The response dict to cache.
        ttl: Time-to-live in seconds. Defaults to the static TTL from settings.
        strategies: Optional strategy toggles to include in cache key.
    """
    try:
        settings = get_settings()
        generation = await get_cache_generation()
        client = await get_redis_client()
        key = _hash_query(query, strategies, generation)
        effective_ttl = ttl or settings.cache_ttl_static
        await client.setex(key, effective_ttl, json.dumps(response))
        logger.debug("L1 cache SET for query hash %s (TTL=%ds)", key[-12:], effective_ttl)
    except Exception as exc:
        logger.warning("L1 cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# L2 Semantic Cache (dual-strategy: RedisVL primary, pure-Redis fallback)
# ---------------------------------------------------------------------------
# Strategy 1 (Primary): RedisVL SemanticCache with HNSW index (requires Redis Stack)
# Strategy 2 (Fallback): Pure-Redis with FastEmbed embeddings + brute-force cosine
# Strategy 3 (Degraded): L2 disabled entirely (standard logging)

_semantic_caches = {}
_l2_strategy: Optional[str] = None  # "redisvl", "pure_redis", or None
_l2_embed_model = None

# Pool size cap for pure-Redis L2 fallback
_L2_POOL_PREFIX = "wikimind:cache:l2:pool"
_L2_MAX_POOL_SIZE = 500


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_l2_embed_model():
    """Lazy-init the FastEmbed model for pure-Redis L2."""
    global _l2_embed_model
    if _l2_embed_model is None:
        try:
            from fastembed import TextEmbedding

            settings = get_settings()
            _l2_embed_model = TextEmbedding(model_name=settings.embedding_model)
            logger.debug("L2 FastEmbed model loaded: %s", settings.embedding_model)
        except Exception as exc:
            logger.warning("Failed to load FastEmbed for L2: %s", exc)
    return _l2_embed_model


async def _init_l2_strategy():
    """Determine and initialize the best available L2 strategy."""
    global _l2_strategy

    if _l2_strategy is not None:
        return _l2_strategy

    # Strategy 1: Try RedisVL (requires Redis Stack with RediSearch)
    try:
        from redisvl.extensions.llmcache import SemanticCache

        settings = get_settings()
        _semantic_caches[_strategy_scope()] = SemanticCache(
            name="wikimind_l2_cache_default",
            redis_url=settings.redis_url,
            distance_threshold=1.0 - settings.cache_similarity_threshold,
        )
        _l2_strategy = "redisvl"
        logger.info(
            "L2 semantic cache: RedisVL strategy (threshold=%.2f).",
            settings.cache_similarity_threshold,
        )
        return _l2_strategy
    except ImportError:
        logger.info("RedisVL not installed. Trying pure-Redis L2 fallback...")
    except Exception as exc:
        logger.info("RedisVL init failed (%s). Trying pure-Redis L2 fallback...", exc)

    # Strategy 2: Pure-Redis with FastEmbed
    try:
        model = _get_l2_embed_model()
        if model is not None:
            client = await get_redis_client()
            await client.ping()
            _l2_strategy = "pure_redis"
            logger.info("L2 semantic cache: pure-Redis fallback strategy active.")
            return _l2_strategy
    except Exception as exc:
        logger.info("Pure-Redis L2 fallback failed: %s", exc)

    # Strategy 3: Disabled
    _l2_strategy = "disabled"
    logger.warning("L2 semantic cache disabled (no Redis Stack or FastEmbed available).")
    return _l2_strategy


def _get_redisvl_cache(scope: str):
    """Return a RedisVL cache isolated to one strategy/generation scope."""
    if scope in _semantic_caches:
        return _semantic_caches[scope]

    from redisvl.extensions.llmcache import SemanticCache

    settings = get_settings()
    scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    cache = SemanticCache(
        name=f"wikimind_l2_cache_{scope_hash}",
        redis_url=settings.redis_url,
        distance_threshold=1.0 - settings.cache_similarity_threshold,
    )
    _semantic_caches[scope] = cache
    return cache


async def l2_get(query: str, strategies: dict = None) -> Optional[dict]:
    """Look up a semantically similar cached response.

    Tries RedisVL first (HNSW index), falls back to pure-Redis
    brute-force cosine similarity if Redis Stack is unavailable.

    Args:
        query: The user's query string.

    Returns:
        Cached response dict if a semantic match is found, None otherwise.
    """
    strategy = await _init_l2_strategy()

    if strategy == "disabled":
        return None

    if strategy == "redisvl":
        try:
            scope = _strategy_scope(strategies, await get_cache_generation())
            semantic_cache = _get_redisvl_cache(scope)
            results = semantic_cache.check(prompt=query)
            if results:
                best = results[0]
                logger.info(
                    "L2 cache HIT [redisvl] (dist=%.4f) for: %s",
                    best.get("vector_distance", 0.0),
                    query[:60],
                )
                response_str = best.get("response", "")
                if response_str:
                    return json.loads(response_str)
        except Exception as exc:
            logger.warning("L2 [redisvl] lookup failed: %s", exc)
        return None

    # Pure-Redis fallback
    try:
        model = _get_l2_embed_model()
        if model is None:
            return None

        generation = await get_cache_generation()
        scope = _strategy_scope(strategies, generation)
        query_embedding = list(model.embed([query]))[0].tolist()
        client = await get_redis_client()
        settings = get_settings()
        threshold = settings.cache_similarity_threshold

        # Scan pool keys
        pool_keys = []
        async for key in client.scan_iter(match=f"{_L2_POOL_PREFIX}:*", count=100):
            pool_keys.append(key)

        if not pool_keys:
            return None

        best_score = 0.0
        best_response = None

        # Batch get all entries
        for key in pool_keys:
            try:
                data = await client.get(key)
                if data is None:
                    continue
                entry = json.loads(data)
                if entry.get("scope") != scope:
                    continue
                stored_embedding = entry.get("embedding", [])
                similarity = _cosine_similarity(query_embedding, stored_embedding)
                if similarity > best_score:
                    best_score = similarity
                    best_response = entry.get("response")
            except Exception:
                continue

        if best_score >= threshold and best_response:
            logger.info(
                "L2 cache HIT [pure_redis] (sim=%.4f) for: %s",
                best_score,
                query[:60],
            )
            return json.loads(best_response) if isinstance(best_response, str) else best_response

    except Exception as exc:
        logger.warning("L2 [pure_redis] lookup failed: %s", exc)

    return None


async def l2_set(
    query: str,
    response: dict,
    ttl: Optional[int] = None,
    strategies: dict = None,
) -> None:
    """Store a response in the L2 semantic cache.

    Uses RedisVL if available, otherwise falls back to pure-Redis
    with FastEmbed embeddings.

    Args:
        query: The user's query string.
        response: The response dict to cache.
        ttl: Time-to-live in seconds.
    """
    strategy = await _init_l2_strategy()

    if strategy == "disabled":
        return

    settings = get_settings()
    effective_ttl = ttl or settings.cache_ttl_static

    if strategy == "redisvl":
        try:
            scope = _strategy_scope(strategies, await get_cache_generation())
            semantic_cache = _get_redisvl_cache(scope)
            semantic_cache.store(
                prompt=query,
                response=json.dumps(response),
                metadata={"ttl": effective_ttl, "scope": scope},
            )
            logger.debug("L2 [redisvl] SET for: %s", query[:60])
        except Exception as exc:
            logger.warning("L2 [redisvl] write failed: %s", exc)
        return

    # Pure-Redis fallback
    try:
        model = _get_l2_embed_model()
        if model is None:
            return

        generation = await get_cache_generation()
        scope = _strategy_scope(strategies, generation)
        query_embedding = list(model.embed([query]))[0].tolist()
        client = await get_redis_client()

        # Generate a unique key for this entry
        entry_hash = hashlib.md5(f"{scope}|{query}".encode()).hexdigest()
        entry_key = f"{_L2_POOL_PREFIX}:{entry_hash}"
        entry_data = json.dumps(
            {
                "embedding": query_embedding,
                "response": json.dumps(response),
                "query": query[:200],
                "scope": scope,
            }
        )

        await client.setex(entry_key, effective_ttl, entry_data)

        # Enforce pool size cap (evict oldest if over limit)
        pool_size = 0
        async for _ in client.scan_iter(match=f"{_L2_POOL_PREFIX}:*", count=100):
            pool_size += 1
            if pool_size > _L2_MAX_POOL_SIZE:
                break

        if pool_size > _L2_MAX_POOL_SIZE:
            # Evict random entries to stay under cap
            evict_count = pool_size - _L2_MAX_POOL_SIZE + 10
            evicted = 0
            async for key in client.scan_iter(match=f"{_L2_POOL_PREFIX}:*", count=100):
                if evicted >= evict_count:
                    break
                await client.delete(key)
                evicted += 1

        logger.debug("L2 [pure_redis] SET for: %s", query[:60])
    except Exception as exc:
        logger.warning("L2 [pure_redis] write failed: %s", exc)


# ---------------------------------------------------------------------------
# Unified Cache Interface
# ---------------------------------------------------------------------------


async def cache_lookup(query: str, strategies: dict = None) -> tuple[Optional[dict], Optional[str]]:
    """Perform a tiered cache lookup (L1 first, then L2).

    The caller should check the returned cache level to log appropriate
    metrics (exact-match vs semantic-match).

    Args:
        query: The user's query string.
        strategies: Optional strategy toggles for cache key partitioning.

    Returns:
        A tuple of (cached_response, cache_level) where cache_level is
        ``"l1"``, ``"l2"``, or None if no cache hit.
    """
    start = time.monotonic()

    # L1: exact match (fastest)
    result = await l1_get(query, strategies)
    if result is not None:
        elapsed = (time.monotonic() - start) * 1000
        logger.info("Cache resolved via L1 in %.1fms", elapsed)
        return result, "l1"

    # L2: semantic similarity (strategy-blind — uses raw query only)
    result = await l2_get(query, strategies)
    if result is not None:
        elapsed = (time.monotonic() - start) * 1000
        logger.info("Cache resolved via L2 in %.1fms", elapsed)
        return result, "l2"

    elapsed = (time.monotonic() - start) * 1000
    logger.debug("Cache MISS (checked L1+L2 in %.1fms)", elapsed)
    return None, None


async def cache_store(
    query: str, response: dict, ttl: Optional[int] = None, strategies: dict = None
) -> None:
    """Store a response in both L1 and L2 caches.

    Writing to both layers ensures that identical queries are served from
    L1 (sub-millisecond) while semantically similar queries benefit from
    L2 vector matching.

    Args:
        query: The user's query string.
        response: The response dict to cache.
        ttl: Time-to-live in seconds. Defaults to the static TTL from settings.
        strategies: Optional strategy toggles for cache key partitioning.
    """
    await l1_set(query, response, ttl=ttl, strategies=strategies)
    await l2_set(query, response, ttl=ttl, strategies=strategies)


async def cache_invalidate(query: str, strategies: dict = None) -> None:
    """Invalidate cached entries for a specific query.

    Removes the L1 exact-match entry. L2 entries will expire via TTL
    since RedisVL does not support targeted deletion by prompt.

    Args:
        query: The query string whose cache entries should be invalidated.
        strategies: Optional strategy toggles for cache key partitioning.
    """
    try:
        generation = await get_cache_generation()
        client = await get_redis_client()
        key = _hash_query(query, strategies, generation)
        await client.delete(key)
        logger.debug("L1 cache invalidated for query hash %s", key[-12:])
        if _l2_strategy == "pure_redis":
            scope = _strategy_scope(strategies, generation)
            entry_hash = hashlib.md5(f"{scope}|{query}".encode()).hexdigest()
            await client.delete(f"{_L2_POOL_PREFIX}:{entry_hash}")
    except Exception as exc:
        logger.warning("L1 cache invalidation failed: %s", exc)
