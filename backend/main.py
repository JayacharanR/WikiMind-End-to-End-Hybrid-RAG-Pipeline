"""WikiMind FastAPI Backend Application.

Entry point for the WikiMind RAG API server. Configures the FastAPI application
with lifespan-managed resource initialization, CORS middleware, Prometheus
metrics instrumentation, and cache-first query routing.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.cache import cache_lookup, cache_store, close_redis, get_redis_client
from backend.config import get_settings
from backend.llmops import get_langfuse_client, init_observability
from backend.models import ChatRequest, ChatResponse, HealthResponse, RetrievalMetadata, ServiceStatus

logger = logging.getLogger(__name__)


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
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("WikiMind Backend starting up...")
    init_observability()

    # Pre-warm Redis connection
    try:
        redis_client = await get_redis_client()
        await redis_client.ping()
        logger.info("Redis connection established at %s", settings.redis_url)
    except Exception as exc:
        logger.warning("Redis connection failed during startup: %s", exc)

    # Pre-initialize Guardrails
    from backend.llmops import get_guardrails
    get_guardrails()
    
    # Initialize Qdrant collections (chunk-level + article-level)
    from backend.qdrant_client import init_collection
    from backend.article_index import init_article_collection
    import asyncio
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

# CORS middleware for Streamlit frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
    logger.info("Prometheus metrics instrumentation enabled.")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed. Metrics disabled.")


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

    # Qdrant health
    settings = get_settings()
    try:
        import httpx
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.qdrant_url}/healthz")
            latency = (time.monotonic() - start) * 1000
            healthy = resp.status_code == 200
            components.append(ServiceStatus(name="qdrant", healthy=healthy, latency_ms=round(latency, 2)))
    except Exception as exc:
        components.append(ServiceStatus(name="qdrant", healthy=False, detail=str(exc)))

    # Langfuse health
    langfuse = get_langfuse_client()
    if langfuse is not None:
        try:
            start = time.monotonic()
            auth_ok = langfuse.auth_check()
            latency = (time.monotonic() - start) * 1000
            components.append(ServiceStatus(name="langfuse", healthy=auth_ok, latency_ms=round(latency, 2)))
        except Exception as exc:
            components.append(ServiceStatus(name="langfuse", healthy=False, detail=str(exc)))
    else:
        components.append(ServiceStatus(name="langfuse", healthy=False, detail="Not configured"))

    overall = "healthy" if all(c.healthy for c in components) else "degraded"
    return HealthResponse(status=overall, components=components)


import json
import asyncio
from sse_starlette.sse import EventSourceResponse

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Primary RAG chat endpoint with cache-first routing.

    Checks the dual-layer cache (L1 exact-match, then L2 semantic) before
    invoking the LangGraph agent pipeline. Cache hits are returned immediately.
    Cache misses invoke the CRAG/Self-RAG state machine and stream the response
    via SSE.
    """
    query = request.query

    # Cache-first: check L1 and L2 before running the agent
    cached_response, cache_level = await cache_lookup(query)
    if cached_response is not None:
        logger.info("Serving cached response (level=%s) for: %s", cache_level, query[:60])
        return JSONResponse(content={
            "answer": cached_response.get("answer", ""),
            "sources": cached_response.get("sources", []),
            "metadata": {
                "cache_hit": True,
                "cache_level": cache_level,
                "strategies_used": [],
                "agent_steps": 0,
            },
        })

    # Cache miss: Stream from LangGraph
    async def sse_generator():
        from backend.agent import agent_app, AgentState
        from backend.llmops import get_langfuse_handler
        
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
        }
        
        # Setup Langfuse callbacks
        config = {}
        handler = get_langfuse_handler()
        if handler:
            config = {"callbacks": [handler]}
            
        logger.info("Invoking LangGraph agent pipeline for: %s", query[:60])
        
        try:
            # Stream the state updates from LangGraph
            async for output in agent_app.astream(initial_state, config=config, stream_mode="updates"):
                # output is a dict keyed by the node name
                for node_name, state_update in output.items():
                    event_data = {
                        "node": node_name,
                        "steps": state_update.get("steps", 0),
                        "status": f"Completed node: {node_name}"
                    }
                    
                    if node_name == "retrieve":
                        docs = state_update.get("documents", [])
                        event_data["document_count"] = len(docs)
                        
                    yield {
                        "event": "update",
                        "data": json.dumps(event_data)
                    }
                    
                    # Store final state to yield at the end
                    final_state_update = state_update
                    
            # Once graph completes, yield the final answer and cache it
            if 'generation' in final_state_update:
                answer = final_state_update['generation']
                sources = final_state_update.get('documents', [])
                
                # Format sources for response
                formatted_sources = [
                    {
                        "title": d.get("title", ""),
                        "content": d.get("content", ""),
                        "score": d.get("score", 0.0),
                        "url": d.get("url")
                    } for d in sources
                ]
                
                final_response = {
                    "answer": answer,
                    "sources": formatted_sources,
                    "metadata": {
                        "cache_hit": False,
                        "strategies_used": list(
                            k for k, v in request.strategies.model_dump().items() if v
                        ),
                        "agent_steps": final_state_update.get("steps", 0),
                    }
                }
                
                # Write to semantic cache
                asyncio.create_task(cache_store(query, final_response))
                
                yield {
                    "event": "final",
                    "data": json.dumps(final_response)
                }
                
        except Exception as exc:
            logger.error("Error during LangGraph streaming: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(exc)})
            }
            
    return EventSourceResponse(sse_generator())


@app.post("/chat/compare")
async def chat_compare(request: Request):
    """Run a query through multiple strategy configurations for A/B comparison.

    Accepts a query and a list of named strategy configs, executes the pipeline
    sequentially for each, and returns all results in a single response.
    """
    from backend.models import CompareRequest, QueryStrategies

    body = await request.json()
    compare_req = CompareRequest(**body)

    results = []
    for config in compare_req.configs:
        config_name = config.get("name", "unknown")
        strategies = QueryStrategies(
            multi_query=config.get("multi_query", False),
            hyde=config.get("hyde", False),
            step_back=config.get("step_back", False),
            decomposition=config.get("decomposition", False),
            page_index=config.get("page_index", False),
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
        }

        import time as time_mod
        start = time_mod.monotonic()
        try:
            final_state = await agent_app.ainvoke(initial_state)
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

            results.append({
                "config_name": config_name,
                "answer": final_state.get("generation", ""),
                "sources": sources,
                "metadata": {
                    "agent_steps": final_state.get("steps", 0),
                    "hallucination_retries": final_state.get("hallucination_retries", 0),
                    "retrieval_grade": final_state.get("retrieval_grade", ""),
                    "answer_grade": final_state.get("answer_grade", ""),
                },
                "latency": round(latency, 3),
            })
        except Exception as exc:
            latency = time_mod.monotonic() - start
            results.append({
                "config_name": config_name,
                "answer": f"Error: {exc}",
                "sources": [],
                "metadata": {},
                "latency": round(latency, 3),
                "error": str(exc),
            })

    return {"query": compare_req.query, "results": results}

