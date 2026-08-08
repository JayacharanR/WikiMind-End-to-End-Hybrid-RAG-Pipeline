"""WikiMind FastAPI Backend Application.

Entry point for the WikiMind RAG API server. Configures the FastAPI application
with lifespan-managed resource initialization, CORS middleware, Prometheus
metrics instrumentation, and cache-first query routing.
"""

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from backend.cache import cache_lookup, cache_store, close_redis, get_redis_client
from backend.config import get_settings
from backend.llmops import get_langfuse_client, init_observability
from backend.models import (
    ChatRequest,
    CompareRequest,
    HealthResponse,
    ServiceStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-Memory Trace Log (ring buffer of last 500 query traces)
# ---------------------------------------------------------------------------
_MAX_TRACES = 500
_trace_log: deque = deque(maxlen=_MAX_TRACES)
_rate_limit_windows: dict[str, deque] = {}


def _check_request_access(request: Request):
    """Apply optional API-key authentication and a bounded local rate limit."""
    settings = get_settings()
    if settings.api_key:
        supplied = request.headers.get("x-api-key", "")
        if supplied != settings.api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

    limit = max(0, settings.rate_limit_per_minute)
    if limit == 0:
        return None

    now = time.monotonic()
    client_host = request.client.host if request.client else "unknown"
    window = _rate_limit_windows.setdefault(client_host, deque())
    while window and now - window[0] >= 60:
        window.popleft()
    if len(window) >= limit:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": "60"},
        )
    window.append(now)
    return None


def _record_trace(
    query: str,
    final_state: dict,
    latency_ms: float,
    cache_hit: bool = False,
    strategies: list = None,
) -> None:
    """Record a query trace to the in-memory log for dashboard consumption."""
    import datetime

    _trace_log.append(
        {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "query": query,
            "generation": final_state.get("generation", "")[:300],
            "latency_ms": round(latency_ms),
            "steps": final_state.get("steps", 0),
            "tokens": {
                "prompt": 0,  # populated by Langfuse if available
                "completion": 0,
            },
            "provenance_score": final_state.get("provenance_score", 0.0),
            "attribution": final_state.get("attribution", "unknown"),
            "guardrails_applied": final_state.get("guardrails_applied", False),
            "retrieval_grade": final_state.get("retrieval_grade", ""),
            "hallucination_grade": final_state.get("hallucination_grade", ""),
            "answer_grade": final_state.get("answer_grade", ""),
            "hallucination_retries": final_state.get("hallucination_retries", 0),
            "answer_retries": final_state.get("answer_retries", 0),
            "document_count": len(final_state.get("documents", [])),
            "retrieval_status": final_state.get("retrieval_status", "unknown"),
            "article_discovery_status": final_state.get("article_discovery_status", "unknown"),
            "expanded_queries": final_state.get("expanded_queries", []),
            "strategies": strategies or [],
            "cache_hit": cache_hit,
            "citation_map": final_state.get("citation_map", {}),
        }
    )


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    On startup: validates Langfuse connection, initializes Redis client,
    and logs readiness status.
    On shutdown: closes Redis connection and flushes any pending state.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    logger.info("WikiMind Backend starting up...")
    init_observability()

    # Pre-warm Redis connection
    try:
        redis_client = await get_redis_client()
        await redis_client.ping()
        logger.info("Redis connection established at %s", settings.redis_url)
    except Exception as exc:
        logger.warning("Redis connection failed during startup: %s", exc)

    # Pre-initialize Guardrails (non-fatal if unavailable)
    try:
        from backend.llmops import get_guardrails

        rails = get_guardrails()
        if rails:
            logger.info("NeMo Guardrails loaded successfully.")
        else:
            logger.info("NeMo Guardrails not available — running without safety rails.")
    except Exception as exc:
        logger.warning("Guardrails initialization failed (non-fatal): %s", exc)

    # Initialize Qdrant collections (chunk-level + article-level)
    import asyncio

    from backend.article_index import init_article_collection
    from backend.qdrant_client import init_collection

    await asyncio.to_thread(init_collection)
    await asyncio.to_thread(init_article_collection)

    logger.info("WikiMind Backend ready on %s:%d", settings.app_host, settings.app_port)

    yield

    # Shutdown
    logger.info("WikiMind Backend shutting down...")
    await close_redis()

    langfuse = get_langfuse_client()
    if langfuse is not None:
        try:
            langfuse.flush()
        except Exception:
            pass

    logger.info("WikiMind Backend shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WikiMind RAG API",
    description="Production-grade Tri-Brid Hybrid Agentic RAG Pipeline backed by Wikipedia",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — restrict to known frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:8080",  # Dashboard
        "http://127.0.0.1:8080",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Prometheus metrics instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    logger.info("Prometheus metrics instrumentation enabled.")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed. Metrics disabled.")

# Mount dashboard static files
_dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if _dashboard_dir.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")
    logger.info("Dashboard mounted at /dashboard")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with per-component status.

    Pings Qdrant, Redis, and Langfuse to report connectivity and latency
    for each infrastructure dependency.
    """
    components = []

    # Redis health
    try:
        redis_client = await get_redis_client()
        start = time.monotonic()
        await redis_client.ping()
        latency = (time.monotonic() - start) * 1000
        components.append(ServiceStatus(name="redis", healthy=True, latency_ms=round(latency, 2)))
    except Exception as exc:
        components.append(ServiceStatus(name="redis", healthy=False, detail=str(exc)))

    # Qdrant health — branch on local vs remote mode
    settings = get_settings()
    if settings.qdrant_mode == "local":
        try:
            from backend.qdrant_client import get_sync_qdrant

            start = time.monotonic()
            client = get_sync_qdrant()
            collections = await asyncio.to_thread(client.get_collections)
            cols = collections.collections
            latency = (time.monotonic() - start) * 1000
            components.append(
                ServiceStatus(
                    name="qdrant",
                    healthy=True,
                    latency_ms=round(latency, 2),
                    detail=f"local, {len(cols)} collection(s)",
                )
            )
        except Exception as exc:
            components.append(ServiceStatus(name="qdrant", healthy=False, detail=str(exc)))
    else:
        try:
            import httpx

            start = time.monotonic()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.qdrant_url}/healthz")
                latency = (time.monotonic() - start) * 1000
                healthy = resp.status_code == 200
                components.append(
                    ServiceStatus(name="qdrant", healthy=healthy, latency_ms=round(latency, 2))
                )
        except Exception as exc:
            components.append(ServiceStatus(name="qdrant", healthy=False, detail=str(exc)))

    # Langfuse health
    langfuse = get_langfuse_client()
    if langfuse is not None:
        try:
            start = time.monotonic()
            auth_ok = await asyncio.to_thread(langfuse.auth_check)
            latency = (time.monotonic() - start) * 1000
            components.append(
                ServiceStatus(name="langfuse", healthy=auth_ok, latency_ms=round(latency, 2))
            )
        except Exception as exc:
            components.append(ServiceStatus(name="langfuse", healthy=False, detail=str(exc)))
    else:
        components.append(ServiceStatus(name="langfuse", healthy=False, detail="Not configured"))

    overall = "healthy" if all(c.healthy for c in components) else "degraded"
    return HealthResponse(status=overall, components=components)


@app.post("/chat")
async def chat_endpoint(http_request: Request, request: ChatRequest):
    """Primary RAG chat endpoint with cache-first routing.

    Checks the dual-layer cache (L1 exact-match, then L2 semantic) before
    invoking the LangGraph agent pipeline. Cache hits are returned immediately.
    Cache misses invoke the CRAG/Self-RAG state machine and stream the response
    via SSE.
    """
    access_error = _check_request_access(http_request)
    if access_error is not None:
        return access_error

    query = request.query

    # Cache-first: check L1 and L2 before running the agent
    strategy_flags = request.strategies.model_dump()
    try:
        cached_response, cache_level = await cache_lookup(query, strategies=strategy_flags)
    except Exception as exc:
        logger.warning("Cache lookup infrastructure failure (degraded mode): %s", exc)
        cached_response, cache_level = None, None

    if cached_response is not None:
        logger.info("Serving cached response (level=%s) for: %s", cache_level, query[:60])
        # Record trace for cache hits so dashboard metrics are accurate
        _record_trace(
            query=query,
            final_state={
                "generation": cached_response.get("answer", ""),
                **cached_response.get("metadata", {}),
            },
            latency_ms=0,
            cache_hit=True,
            strategies=[k for k, v in strategy_flags.items() if v],
        )
        return JSONResponse(
            content={
                "answer": cached_response.get("answer", ""),
                "sources": cached_response.get("sources", []),
                "metadata": {
                    "cache_hit": True,
                    "cache_level": cache_level,
                    "strategies_used": [k for k, v in strategy_flags.items() if v],
                    "retrieval_status": cached_response.get("metadata", {}).get(
                        "retrieval_status", "unknown"
                    ),
                    "article_discovery_status": cached_response.get("metadata", {}).get(
                        "article_discovery_status", "unknown"
                    ),
                    "agent_steps": 0,
                },
            }
        )

    # Cache miss: Stream from LangGraph
    async def sse_generator():
        from backend.agent import AgentState, agent_app
        from backend.llmops import get_langfuse_handler

        _sse_start_time = time.time()

        initial_state: AgentState = {
            "query": query,
            "expanded_queries": [],
            "target_articles": [],
            "documents": [],
            "web_snippets": [],
            "generation": "",
            "retrieval_grade": "",
            "hallucination_grade": "",
            "answer_grade": "",
            "steps": 0,
            "active_strategies": request.strategies,
            "hallucination_retries": 0,
            "answer_retries": 0,
            "article_discovery_failed": False,
            "article_discovery_status": "unknown",
            "retrieval_status": "unknown",
            "citation_map": {},
            "provenance_score": 0.0,
            "attribution": "unknown",
            "guardrails_applied": False,
        }

        # Setup Langfuse callbacks for per-query tracing
        config = {}
        handler = get_langfuse_handler(
            trace_name="wikimind_query",
            session_id=f"session_{int(time.time())}",
            metadata={
                "query": query[:100],
                "strategies": list(k for k, v in request.strategies.model_dump().items() if v),
            },
        )
        if handler:
            config = {"callbacks": [handler]}

        logger.info("Invoking LangGraph agent pipeline for: %s", query[:60])

        try:
            current_state = dict(initial_state)

            # Stream the state updates from LangGraph
            async def bounded_state_stream():
                async with asyncio.timeout(get_settings().request_timeout_seconds):
                    async for output in agent_app.astream(
                        initial_state, config=config, stream_mode="updates"
                    ):
                        yield output

            async for output in bounded_state_stream():
                # output is a dict keyed by the node name
                for node_name, state_update in output.items():
                    current_state.update(state_update)
                    event_data = {
                        "node": node_name,
                        "steps": current_state.get("steps", 0),
                        "status": f"Completed node: {node_name}",
                    }

                    if node_name == "retrieve":
                        docs = current_state.get("documents", [])
                        event_data["document_count"] = len(docs)

                    yield {"event": "update", "data": json.dumps(event_data)}

            # Once graph completes, yield the final answer and cache it
            if "generation" in current_state:
                answer = current_state["generation"]
                sources = current_state.get("documents", [])

                # Format sources for response
                formatted_sources = [
                    {
                        "title": d.get("title", ""),
                        "content": d.get("content", ""),
                        "score": float(d.get("score", 0.0)),
                        "url": d.get("url"),
                    }
                    for d in sources
                ]

                final_response = {
                    "answer": answer,
                    "sources": formatted_sources,
                    "metadata": {
                        "cache_hit": False,
                        "strategies_used": list(
                            k for k, v in request.strategies.model_dump().items() if v
                        ),
                        "agent_steps": current_state.get("steps", 0),
                        "expanded_queries": current_state.get("expanded_queries", []),
                        "provenance_score": current_state.get("provenance_score", 0.0),
                        "attribution": current_state.get("attribution", "unknown"),
                        "guardrails_applied": current_state.get("guardrails_applied", False),
                        "retrieval_status": current_state.get("retrieval_status", "unknown"),
                        "article_discovery_status": current_state.get(
                            "article_discovery_status", "unknown"
                        ),
                    },
                    "citation_map": current_state.get("citation_map", {}),
                }

                # Write to cache — but skip abstention/error responses
                attribution = current_state.get("attribution", "unknown")
                answer_text = final_response.get("answer", "")
                is_cacheable = (
                    attribution not in {"abstention", "retrieval_unavailable"}
                    and "Error generating response" not in answer_text
                )
                if is_cacheable:
                    asyncio.create_task(
                        cache_store(query, final_response, strategies=strategy_flags)
                    )

                # Record trace for dashboard
                trace_latency = (time.time() - _sse_start_time) * 1000
                _record_trace(
                    query=query,
                    final_state=current_state,
                    latency_ms=trace_latency,
                    strategies=list(k for k, v in request.strategies.model_dump().items() if v),
                )

                yield {"event": "final", "data": json.dumps(final_response)}

        except Exception as exc:
            import uuid as _uuid

            error_id = str(_uuid.uuid4())[:8]
            logger.error("Error during LangGraph streaming [%s]: %s", error_id, exc)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": f"An internal error occurred. Reference: {error_id}"}
                ),
            }

    return EventSourceResponse(sse_generator())


@app.post("/chat/compare")
async def chat_compare(request: Request, compare_req: CompareRequest):
    """Run a query through multiple strategy configurations for A/B comparison.

    Accepts a query and a list of named strategy configs, executes the pipeline
    sequentially for each, and returns all results in a single response.
    """
    access_error = _check_request_access(request)
    if access_error is not None:
        return access_error

    from backend.llmops import get_langfuse_handler
    from backend.models import QueryStrategies

    results = []
    for config in compare_req.configs:
        config_name = config.get("name", "unknown")
        strategies = QueryStrategies(
            multi_query=config.get("multi_query", False),
            hyde=config.get("hyde", False),
            step_back=config.get("step_back", False),
            decomposition=config.get("decomposition", False),
            page_index=config.get("page_index", False),
            knowledge_graph=config.get("knowledge_graph", False),
        )

        initial_state = {
            "query": compare_req.query,
            "expanded_queries": [],
            "target_articles": [],
            "documents": [],
            "web_snippets": [],
            "generation": "",
            "retrieval_grade": "",
            "hallucination_grade": "",
            "answer_grade": "",
            "steps": 0,
            "active_strategies": strategies,
            "hallucination_retries": 0,
            "answer_retries": 0,
            "citation_map": {},
            "provenance_score": 0.0,
            "attribution": "unknown",
            "guardrails_applied": False,
            "article_discovery_failed": False,
            "article_discovery_status": "unknown",
            "retrieval_status": "unknown",
        }

        import time as time_mod

        from backend.agent import agent_app

        # Langfuse trace per config
        handler = get_langfuse_handler(
            trace_name=f"compare_{config_name}",
            session_id=f"compare_{int(time_mod.time())}",
            metadata={"query": compare_req.query[:100], "config": config_name},
        )
        invoke_config = {"callbacks": [handler]} if handler else {}

        start = time_mod.monotonic()
        try:
            final_state = await asyncio.wait_for(
                agent_app.ainvoke(initial_state, config=invoke_config),
                timeout=get_settings().request_timeout_seconds,
            )
            latency = time_mod.monotonic() - start

            sources = [
                {
                    "title": doc.get("title", ""),
                    "content": doc.get("content", doc.get("page_content", "")),
                    "score": doc.get("score", 0.0),
                    "url": doc.get("url", ""),
                }
                for doc in final_state.get("documents", [])
            ]

            results.append(
                {
                    "config_name": config_name,
                    "answer": final_state.get("generation", ""),
                    "sources": sources,
                    "metadata": {
                        "agent_steps": final_state.get("steps", 0),
                        "hallucination_retries": final_state.get("hallucination_retries", 0),
                        "retrieval_grade": final_state.get("retrieval_grade", ""),
                        "answer_grade": final_state.get("answer_grade", ""),
                        "retrieval_status": final_state.get("retrieval_status", "unknown"),
                        "article_discovery_status": final_state.get(
                            "article_discovery_status", "unknown"
                        ),
                    },
                    "latency": round(latency, 3),
                }
            )
        except Exception as exc:
            latency = time_mod.monotonic() - start
            results.append(
                {
                    "config_name": config_name,
                    "answer": f"Error: {exc}",
                    "sources": [],
                    "metadata": {},
                    "latency": round(latency, 3),
                    "error": str(exc),
                }
            )

    return {"query": compare_req.query, "results": results}


# ---------------------------------------------------------------------------
# Dashboard API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/metrics")
async def api_metrics():
    """Aggregated dashboard metrics from the in-memory trace log."""
    traces = list(_trace_log)
    total = len(traces)

    if total == 0:
        return {
            "summary": {
                "total_queries": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "avg_steps": 0,
                "avg_provenance_score": 0.0,
                "cache_hit_rate": 0.0,
                "attribution_breakdown": {"rag_grounded": 0, "parametric_risk": 0, "unknown": 0},
                "guardrails_stats": {"applied": 0, "bypassed": 0},
                "grade_breakdown": {"grounded": 0, "hallucinated": 0, "useful": 0, "not_useful": 0},
            },
            "recent_traces": [],
        }

    latencies = sorted(t["latency_ms"] for t in traces)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    attr_breakdown = {"rag_grounded": 0, "parametric_risk": 0, "unknown": 0}
    guard_stats = {"applied": 0, "bypassed": 0}
    grade_breakdown = {"grounded": 0, "hallucinated": 0, "useful": 0, "not_useful": 0}
    total_prov = 0.0
    cache_hits = 0

    for t in traces:
        attr = t.get("attribution", "unknown")
        attr_breakdown[attr] = attr_breakdown.get(attr, 0) + 1
        if t.get("guardrails_applied"):
            guard_stats["applied"] += 1
        else:
            guard_stats["bypassed"] += 1
        total_prov += t.get("provenance_score", 0.0)
        if t.get("cache_hit"):
            cache_hits += 1
        # Grade counts
        hg = t.get("hallucination_grade", "")
        if hg in grade_breakdown:
            grade_breakdown[hg] += 1
        ag = t.get("answer_grade", "")
        if ag in grade_breakdown:
            grade_breakdown[ag] += 1

    return {
        "summary": {
            "total_queries": total,
            "p50_latency_ms": round(p50),
            "p95_latency_ms": round(p95),
            "avg_steps": round(sum(t["steps"] for t in traces) / total, 1),
            "avg_provenance_score": round(total_prov / total, 3),
            "cache_hit_rate": round(cache_hits / total, 3),
            "attribution_breakdown": attr_breakdown,
            "guardrails_stats": guard_stats,
            "grade_breakdown": grade_breakdown,
        },
        "recent_traces": list(reversed(traces[-50:])),
    }


@app.get("/api/traces")
async def api_traces(limit: int = 100):
    """Return the last N query traces for the dashboard."""
    limit = max(1, min(limit, 500))  # Cap to prevent abuse
    traces = list(_trace_log)
    return {"traces": list(reversed(traces[-limit:]))}


@app.get("/api/eval-results")
async def api_eval_results():
    """Return evaluation benchmark results if available.

    Normalizes the JSON shape to match the dashboard's expected format:
    - ``aggregate`` (dashboard) ← ``aggregates`` (harness output)
    - ``per_query`` (dashboard) ← ``per_query_results`` (harness output)
    """
    results_dir = Path(__file__).resolve().parent.parent / "evaluation" / "results"
    if not results_dir.is_dir():
        return {"results": []}

    import json as json_mod

    results = []
    for f in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            data = json_mod.loads(f.read_text(encoding="utf-8"))
            # Normalize keys for dashboard compatibility
            agg = data.get("aggregates", data.get("aggregate", {}))
            # Normalize metric names: evaluator uses 'mean_answer_accuracy'
            # but dashboard reads 'mean_accuracy'; same for step_count
            if "mean_answer_accuracy" in agg and "mean_accuracy" not in agg:
                agg["mean_accuracy"] = agg["mean_answer_accuracy"]
            if "mean_step_count" in agg and "mean_steps" not in agg:
                agg["mean_steps"] = agg["mean_step_count"]
            normalized = {
                "filename": f.name,
                "aggregate": agg,
                "per_query": data.get("per_query_results", data.get("per_query", [])),
                "dataset": data.get("dataset", ""),
                "timestamp": data.get("timestamp", ""),
                "config": data.get("config", {}),
            }
            results.append(normalized)
        except Exception:
            continue
    return {"results": results[:10]}  # last 10 benchmark runs


@app.get("/api/pipeline-health")
async def api_pipeline_health():
    """Return health status of data pipeline workers (updater + reconciler).

    Exposes DLQ sizes, heartbeat timestamps, event counts, drift metrics,
    and consecutive failure counts for monitoring self-healing behavior.
    """
    workers = []

    try:
        from data_pipeline.wiki_updater import get_updater_health

        workers.append(get_updater_health())
    except Exception as exc:
        workers.append({"worker": "wiki-updater", "status": "unavailable", "error": str(exc)})

    try:
        from data_pipeline.reconciler import get_reconciler_health

        workers.append(get_reconciler_health())
    except Exception as exc:
        workers.append({"worker": "reconciler", "status": "unavailable", "error": str(exc)})

    overall = "healthy"
    for w in workers:
        if w.get("status") == "degraded":
            overall = "degraded"
            break

    return {"status": overall, "workers": workers}
