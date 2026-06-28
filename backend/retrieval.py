"""Tri-Brid Retrieval Engine.

Implements the core retrieval logic including Qdrant Hybrid Search (Dense + Sparse)
and Reciprocal Rank Fusion (RRF) via the Qdrant Universal Query API.
"""

import logging
from typing import List, Dict, Any, Optional

from qdrant_client.http import models

from backend.config import get_settings
from backend.qdrant_client import get_async_qdrant
from data_pipeline.ingest import get_dense_model, get_sparse_model

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

_ranker: Optional[Ranker] = None


def get_reranker() -> Ranker:
    """Lazy initialize the FlashRank cross-encoder model."""
    global _ranker
    if _ranker is None:
        settings = get_settings()
        logger.info("Initializing FlashRank reranker model: %s", settings.reranker_model)
        _ranker = Ranker(model_name=settings.reranker_model, cache_dir="data/flashrank_cache")
    return _ranker


async def hybrid_search(query: str, apply_reranker: bool = True) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute a hybrid dense+sparse search using Qdrant Prefetch API for RRF.
    
    Args:
        query: The natural language search query.
        apply_reranker: Whether to apply the cross-encoder reranker.
        
    Returns:
        Tuple of (List of candidate documents, Metadata dict).
    """
    settings = get_settings()
    qdrant = get_async_qdrant()
    
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    
    logger.debug("Generating dual embeddings for query: %s", query)
    
    # Generate embeddings
    dense_vector = list(dense_model.embed([query]))[0].tolist()
    sparse_obj = list(sparse_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_obj.indices.tolist(),
        values=sparse_obj.values.tolist(),
    )
    
    # Qdrant Prefetch API for server-side Reciprocal Rank Fusion
    prefetch_dense = models.Prefetch(
        query=dense_vector,
        using="dense",
        limit=settings.rrf_k,
    )
    
    prefetch_sparse = models.Prefetch(
        query=sparse_vector,
        using="sparse",
        limit=settings.rrf_k,
    )
    
    logger.debug("Executing Qdrant Universal Query API with RRF fusion (k=%d)", settings.rrf_k)
    
    metadata = {
        "rrf_candidates": 0,
        "reranker_applied": apply_reranker,
    }
    
    try:
        # The query method with multiple prefetches performs RRF by default
        results = await qdrant.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=settings.retrieval_top_k,
            with_payload=True,
        )
        
        # Format results
        documents = []
        for point in results.points:
            documents.append({
                "id": str(point.id),
                "score": float(point.score),
                "title": point.payload.get("title", ""),
                "text": point.payload.get("page_content", ""), # FlashRank expects "text" key
                "url": point.payload.get("url", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
            })
            
        metadata["rrf_candidates"] = len(documents)
        logger.info("Hybrid search returned %d RRF candidates.", len(documents))
        
        if apply_reranker and documents:
            logger.debug("Applying FlashRank reranker...")
            ranker = get_reranker()
            rerank_request = RerankRequest(query=query, passages=documents)
            reranked_results = ranker.rerank(rerank_request)
            
            # FlashRank returns the list sorted by score (descending)
            # and modifies the dicts in place (or returns new ones) with a "score" key
            top_documents = reranked_results[:settings.reranker_top_k]
            
            # Map "text" back to "content" for consistency with other parts of the app
            for doc in top_documents:
                doc["content"] = doc.pop("text")
                
            logger.info("Reranked to top %d candidates.", len(top_documents))
            return top_documents, metadata
        else:
            # Re-map "text" to "content" if not reranking
            for doc in documents:
                doc["content"] = doc.pop("text")
            return documents, metadata
        
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        return [], metadata


# ---------------------------------------------------------------------------
# Query Router
# ---------------------------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert query router. Your task is to analyze a user's question and classify it as either 'simple' or 'complex'.\n"
               "A 'simple' query asks for a direct fact, definition, or a specific entity's property (e.g., 'When was Python created?', 'Capital of France?').\n"
               "A 'complex' query requires multi-hop reasoning, synthesis across multiple topics, deep conceptual explanation, or structural extraction (e.g., 'Compare the economic impacts of X and Y', 'What are the main arguments in Z?').\n"
               "Respond with EXACTLY ONE WORD: either 'simple' or 'complex'."),
    ("user", "Query: {query}\nClassification:")
])


async def route_query(query: str) -> str:
    """Classify a query as 'simple' or 'complex' to determine the retrieval path.
    
    Simple queries can be routed directly to hybrid_search to save latency and cost.
    Complex queries should be routed through the full CRAG pipeline with Query 
    Expansion and PageIndex structural extraction.
    
    Args:
        query: The user's search query.
        
    Returns:
        String: 'simple' or 'complex'. Defaults to 'complex' on failure to be safe.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
        max_tokens=10
    )
    
    chain = ROUTER_PROMPT | llm
    
    try:
        res = await chain.ainvoke({"query": query})
        content = res.content if hasattr(res, "content") else str(res)
        classification = content.strip().lower()
        
        if "simple" in classification:
            logger.info("Query routed as 'simple': %s", query[:60])
            return "simple"
        else:
            logger.info("Query routed as 'complex': %s", query[:60])
            return "complex"
            
    except Exception as exc:
        logger.warning("Query routing failed, defaulting to 'complex': %s", exc)
        return "complex"

