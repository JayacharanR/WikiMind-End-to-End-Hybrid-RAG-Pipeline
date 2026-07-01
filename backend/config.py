"""Centralized configuration module using Pydantic BaseSettings.

All environment variables are validated at startup with type coercion and
sensible defaults. Configuration is organized into logical sections covering
every external service the application depends on.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    Pydantic BaseSettings reads values from environment variables (case-
    insensitive) and the ``.env`` file in the project root. Every field
    documents its purpose, accepted values, and default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenRouter -------------------------------------------------------
    openrouter_api_key: str = ""
    openrouter_model: str = "gpt-oss-20b:free"

    # --- Embedding Model --------------------------------------------------
    # Dense embedding model used for semantic vector generation.
    # Default uses the local FastEmbed BAAI/bge-small-en-v1.5 (384 dims).
    # Set to "text-embedding-3-small" to use OpenAI embeddings (1536 dims).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Qdrant Vector Database -------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "wikimind_hybrid"

    # --- Redis / Semantic Cache -------------------------------------------
    redis_url: str = "redis://localhost:6379"
    cache_similarity_threshold: float = 0.92
    cache_ttl_static: int = 86400  # 24 hours in seconds
    cache_ttl_dynamic: int = 3600  # 1 hour in seconds

    # --- Langfuse Observability -------------------------------------------
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Tavily Web Search ------------------------------------------------
    tavily_api_key: str = ""

    # --- Retrieval Parameters ---------------------------------------------
    retrieval_top_k: int = 50
    rrf_k: int = 60
    reranker_top_k: int = 5
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"

    # --- Application Server -----------------------------------------------
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- Wikipedia Sync ---------------------------------------------------
    wiki_stream_url: str = "https://stream.wikimedia.org/v2/stream/recentchange"
    wiki_reconcile_interval_hours: int = 24


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings.

    Using ``@lru_cache`` ensures the ``.env`` file is read only once and the
    same ``Settings`` object is reused across the entire application lifetime.
    This function is intended to be used with FastAPI's ``Depends()`` for
    dependency injection.

    Returns:
        Settings: Validated application settings.
    """
    return Settings()
