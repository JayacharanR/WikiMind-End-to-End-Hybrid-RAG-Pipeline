"""LLM Observability and Safety module.

Provides lazy-initialized Langfuse client, LangChain CallbackHandler factory
for LangGraph trace instrumentation, and NeMo Guardrails initialization with
proper error handling and fallback behavior.

All module-level side effects have been eliminated; initialization is deferred
to explicit function calls during the FastAPI lifespan startup.
"""

import logging
from functools import lru_cache
from typing import Optional

from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Langfuse Observability
# ---------------------------------------------------------------------------


def _create_langfuse_client() -> Optional[Langfuse]:
    """Create a Langfuse client instance if credentials are available.

    Returns:
        Langfuse client if credentials are configured, None otherwise.
    """
    from backend.config import get_settings

    settings = get_settings()

    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.warning("Langfuse credentials missing. Observability is disabled.")
        return None

    client = Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )
    return client


@lru_cache
def get_langfuse_client() -> Optional[Langfuse]:
    """Return a cached singleton Langfuse client.

    The client is created on first call and reused for the application
    lifetime. Returns None when credentials are not configured.

    Returns:
        Optional[Langfuse]: Langfuse client or None.
    """
    return _create_langfuse_client()


def init_observability() -> None:
    """Validate the Langfuse connection during application startup.

    Called from the FastAPI lifespan context manager. Logs the connection
    status without raising exceptions to allow graceful degradation.
    """
    client = get_langfuse_client()
    if client is None:
        return

    try:
        if client.auth_check():
            logger.info("Langfuse observability initialized successfully.")
        else:
            logger.warning("Langfuse authentication failed. Check API keys.")
    except Exception as exc:
        logger.warning("Langfuse connection check failed: %s", exc)


def get_langfuse_handler(**kwargs) -> Optional[LangfuseCallbackHandler]:
    """Create a LangfuseCallbackHandler for LangChain/LangGraph tracing.

    Returns a handler that instruments all LLM calls with Langfuse traces.
    Returns None when Langfuse credentials are not configured.

    Args:
        **kwargs: Extra arguments passed to LangfuseCallbackHandler
                  (e.g., session_id, user_id, trace_name).

    Returns:
        Optional[LangfuseCallbackHandler]: Handler or None.
    """
    from backend.config import get_settings

    settings = get_settings()

    if (
        not settings.langfuse_secret_key
        or not settings.langfuse_public_key
        or "placeholder" in settings.langfuse_secret_key
    ):
        return None

    try:
        return LangfuseCallbackHandler(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
            **kwargs,
        )
    except Exception as exc:
        logger.warning("Failed to create Langfuse handler: %s", exc)
        return None


def push_eval_scores(
    trace_name: str,
    scores: dict,
    metadata: dict = None,
) -> None:
    """Push evaluation scores to Langfuse for a completed trace.

    Used by the evaluation harness to record benchmark results
    (recall, accuracy, latency) as Langfuse scores.

    Args:
        trace_name: Name identifying the evaluation run.
        scores: Dict of metric_name -> float value.
        metadata: Optional metadata dict for the trace.
    """
    client = get_langfuse_client()
    if client is None:
        return

    try:
        trace = client.trace(name=trace_name, metadata=metadata or {})
        for metric_name, value in scores.items():
            trace.score(name=metric_name, value=value)
        client.flush()
        logger.info("Pushed %d eval scores to Langfuse trace '%s'.", len(scores), trace_name)
    except Exception as exc:
        logger.warning("Failed to push eval scores to Langfuse: %s", exc)


# ---------------------------------------------------------------------------
# NeMo Guardrails
# ---------------------------------------------------------------------------


@lru_cache
def get_guardrails() -> Optional[LLMRails]:
    """Initialize and cache the NeMo Guardrails application.

    Uses ``@lru_cache`` to ensure the configuration is loaded only once.
    Returns None if initialization fails, allowing the application to
    operate without guardrails in development environments.

    Returns:
        Optional[LLMRails]: Initialized guardrails application, or None.
    """
    try:
        import os

        config_path = os.path.join(os.path.dirname(__file__), "guardrails_config")
        from backend.agent import _get_llm

        config = RailsConfig.from_path(config_path)
        rails = LLMRails(config, llm=_get_llm())
        logger.info("NeMo Guardrails initialized from %s", config_path)
        return rails
    except Exception as exc:
        logger.warning("NeMo Guardrails unavailable (non-fatal): %s", exc)
        return None


@observe(as_type="generation")
async def safe_generate(query: str, context: str = "") -> Optional[str]:
    """Apply NeMo Guardrails to the outgoing generation request.

    Evaluates the query against predefined safety and topical rails, then
    generates a response. The ``@observe`` decorator logs the full trace
    to Langfuse automatically.

    Returns the generated response string, or None if guardrails are
    unavailable or generation fails. Callers should fall back to direct
    LLM generation when None is returned.

    Args:
        query: The user's natural language query.
        context: Retrieved context chunks to ground the generation.

    Returns:
        The generated response string, or None on failure.
    """
    rails_app = get_guardrails()
    if rails_app is None:
        logger.warning("Guardrails not available. Returning None for fallback.")
        return None

    # Build prompt with context and explicit sentence-level citation rules
    context_block = f"\n\nContext:\n{context}\n\n" if context else ""
    strict_query = (
        f"Question: {query}\n"
        f"{context_block}"
        "IMPORTANT INSTRUCTIONS:\n"
        "1. Answer using ONLY the provided numbered context chunks above. Do not output JSON or tool calls. Answer directly in plain text.\n"
        "2. Every factual claim MUST end with its inline source citation in brackets, e.g. 'Paris is the capital of France [1].'\n"
        "3. If the context does not contain the answer, state that you cannot answer based on the retrieved context.\n"
        "4. Keep the answer direct and concise (1-3 sentences). Do not add meta-commentary, conversational filler, or disclaimers."
    )

    messages = [
        {"role": "context", "content": {"relevant_chunks": context}},
        {"role": "user", "content": strict_query},
    ]

    try:
        response = await rails_app.generate_async(messages=messages)
        content = None
        if isinstance(response, dict):
            content = response.get("content")
        elif response:
            content = str(response)

        if not content:
            return None

        # If Guardrails output a tool call JSON or error, fall back to direct LLM
        trimmed = content.strip()
        if (
            (trimmed.startswith("{") and trimmed.endswith("}"))
            or "Error:" in trimmed
            or "not initialized" in trimmed
        ):
            logger.warning("Guardrails returned invalid or tool-call response. Falling back to direct LLM.")
            return None

        return content
    except Exception as exc:
        logger.warning("NeMo Guardrails generation failed: %s", exc)
        return None
