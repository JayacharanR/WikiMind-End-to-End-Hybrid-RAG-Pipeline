"""Pydantic request and response schemas for the WikiMind API.

Defines the data contracts for the FastAPI endpoints, ensuring strict
validation of incoming requests and consistent serialization of outgoing
responses. All schemas use Pydantic V2 model conventions.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class QueryStrategies(BaseModel):
    """User-configurable retrieval strategy toggles.

    Each boolean flag controls whether a specific query expansion or
    retrieval strategy is active for the current request.
    """

    multi_query: bool = Field(
        default=False,
        description="Generate 3 semantically diverse query reformulations.",
    )
    hyde: bool = Field(
        default=False,
        description="Generate a hypothetical answer and use its embedding for retrieval.",
    )
    step_back: bool = Field(
        default=False,
        description="Generate an abstract foundational query for broader context.",
    )
    decomposition: bool = Field(
        default=False,
        description="Break multi-part questions into atomic sub-questions.",
    )
    page_index: bool = Field(
        default=False,
        description="Enable vectorless PageIndex tree navigation for deep extraction.",
    )
    knowledge_graph: bool = Field(
        default=False,
        description="Enable knowledge graph traversal for multi-hop entity queries.",
    )


class ChatRequest(BaseModel):
    """Request body for the ``POST /chat`` endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural language query.",
    )
    strategies: QueryStrategies = Field(
        default_factory=QueryStrategies,
        description="Optional retrieval strategy toggles.",
    )

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        """Reject whitespace-only queries while preserving user wording."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace text")
        return normalized


class CompareRequest(BaseModel):
    """Request body for the ``POST /chat/compare`` endpoint.

    Accepts a single query and a list of named strategy configurations to
    run in sequence for side-by-side comparison.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural language query.",
    )
    configs: list[dict] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of strategy config dicts, each with a 'name' key and strategy booleans. Max 5.",
    )

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace text")
        return normalized


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """A single retrieved source document included in the response."""

    title: str = Field(description="Wikipedia article title.")
    content: str = Field(description="Extracted text content from the article.")
    score: float = Field(description="Combined retrieval score (post-reranking).")
    url: Optional[str] = Field(
        default=None,
        description="URL to the original Wikipedia article.",
    )


class RetrievalMetadata(BaseModel):
    """Diagnostic metadata about the retrieval pipeline execution."""

    cache_hit: bool = Field(
        default=False,
        description="Whether the response was served from the semantic cache.",
    )
    cache_similarity: Optional[float] = Field(
        default=None,
        description="Cosine similarity score if a cache hit occurred.",
    )
    retrieval_count: int = Field(
        default=0,
        description="Number of documents retrieved before reranking.",
    )
    rrf_candidates: int = Field(
        default=0,
        description="Number of candidates produced by RRF fusion.",
    )
    reranker_applied: bool = Field(
        default=False,
        description="Whether the cross-encoder reranker was applied.",
    )
    strategies_used: list[str] = Field(
        default_factory=list,
        description="List of query expansion strategies that were active.",
    )
    agent_steps: int = Field(
        default=0,
        description="Number of LangGraph state transitions in this request.",
    )
    hallucination_retries: int = Field(
        default=0,
        description="Number of hallucination check retries triggered.",
    )

    retrieval_status: str = Field(
        default="unknown",
        description="Retrieval outcome: ok, no_results, or unavailable.",
    )
    article_discovery_status: str = Field(
        default="unknown",
        description="Article discovery outcome: ok, no_match, or unavailable.",
    )


class ChatResponse(BaseModel):
    """Response body for the ``POST /chat`` endpoint (non-streaming fallback)."""

    answer: str = Field(description="The generated answer text.")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Retrieved source documents used for generation.",
    )
    metadata: RetrievalMetadata = Field(
        default_factory=RetrievalMetadata,
        description="Diagnostic retrieval pipeline metadata.",
    )


class ServiceStatus(BaseModel):
    """Health status of a single infrastructure component."""

    name: str = Field(description="Service name (e.g., 'qdrant', 'redis').")
    healthy: bool = Field(description="Whether the service is reachable.")
    latency_ms: Optional[float] = Field(
        default=None,
        description="Round-trip latency to the service in milliseconds.",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional diagnostic information.",
    )


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint."""

    status: str = Field(description="Overall system status: 'healthy' or 'degraded'.")
    service: str = Field(default="WikiMind Backend")
    components: list[ServiceStatus] = Field(
        default_factory=list,
        description="Per-component health status.",
    )
