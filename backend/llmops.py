"""LLM Observability and Safety module.

Provides lazy-initialized Langfuse client, LangChain CallbackHandler factory
for LangGraph trace instrumentation, and NeMo Guardrails initialization with
proper error handling and fallback behavior.

All module-level side effects have been eliminated; initialization is deferred
to explicit function calls during the FastAPI lifespan startup.
"""

import logging
import os
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
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")

    if not secret_key or not public_key:
        logger.warning("Langfuse credentials missing. Observability is disabled.")
        return None

    client = Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
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


def get_langfuse_handler() -> Optional[LangfuseCallbackHandler]:
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")

    if not secret_key or not public_key or "placeholder" in secret_key or "placeholder" in public_key:
        return None

    return LangfuseCallbackHandler()


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
    config_path = os.path.join(os.path.dirname(__file__), "guardrails_config")
    try:
        from backend.agent import _get_llm
        config = RailsConfig.from_path(config_path)
        rails = LLMRails(config, llm=_get_llm())
        logger.info("NeMo Guardrails initialized from %s", config_path)
        return rails
    except Exception as exc:
        logger.warning("NeMo Guardrails initialization failed: %s", exc)
        return None


@observe(as_type="generation")
async def safe_generate(query: str, context: str = "") -> str:
    """Apply NeMo Guardrails to the outgoing generation request.

    Evaluates the query against predefined safety and topical rails, then
    generates a response. The ``@observe`` decorator logs the full trace
    to Langfuse automatically.

    If guardrails are not available (initialization failure or missing
    config), falls back to returning an error message rather than raising.

    Args:
        query: The user's natural language query.
        context: Retrieved context chunks to ground the generation.

    Returns:
        The generated response string.
    """
    rails_app = get_guardrails()
    if rails_app is None:
        logger.error("Guardrails not available. Cannot generate safely.")
        return "Error: Safety guardrails are not initialized."

    # Append strict anti-tool-call instruction to the query for the local Llama model
    strict_query = (
        f"{query}\n\n"
        "IMPORTANT: Do not output JSON. Do not output tool calls. Answer the question directly in plain text using the context. "
        "Always list the URLs of the Wikipedia articles you used as References at the very end of your answer."
    )

    messages = [
        {"role": "context", "content": {"relevant_chunks": context}},
        {"role": "user", "content": strict_query},
    ]

    response = await rails_app.generate_async(messages=messages)

    if isinstance(response, dict):
        return response.get("content", "Error generating response.")
    return str(response)
