"""Query Expansion Module.

Implements four distinct query expansion strategies to improve retrieval recall:
1. Multi-Query: Generates semantic reformulations of the original query.
2. HyDE (Hypothetical Document Embeddings): Generates a hypothetical answer paragraph.
3. Step-Back: Abstracts the query to a higher-level foundational question.
4. Decomposition: Breaks a complex multi-part query into atomic sub-questions.
"""

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.models import QueryStrategies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI language model assistant. Your task is to generate 3 different versions of the given user query to retrieve relevant documents from a vector database. By generating multiple perspectives on the user query, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines. Do not number them."),
    ("user", "Original query: {query}")
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Wikipedia author. Write a short, factual, hypothetical Wikipedia paragraph that directly answers the user's query. Do not include any introductory or concluding remarks, just the factual text."),
    ("user", "Query: {query}")
])

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:\nOriginal: Which team did the person who scored the most points in the NBA finals play for?\nStep-back: Who scored the most points in the NBA finals?\nOriginal: Est_Ovest is located in which country?\nStep-back: What is Est_Ovest?"),
    ("user", "Original question: {query}\nStep-back question:")
])

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that breaks down complex questions into simpler, atomic sub-questions. Generate between 2 and 4 sub-questions that, when answered together, would provide a complete answer to the original question. Output one question per line, with no numbering."),
    ("user", "Complex question: {query}")
])


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

async def _generate_multi_queries(query: str, llm: ChatOpenAI) -> List[str]:
    """Generate multiple semantic reformulations of the query."""
    chain = MULTI_QUERY_PROMPT | llm
    try:
        res = await chain.ainvoke({"query": query})
        content = res.content if hasattr(res, "content") else str(res)
        queries = [q.strip() for q in content.split("\n") if q.strip()]
        return queries[:3]
    except Exception as exc:
        logger.warning("Multi-query expansion failed: %s", exc)
        return []


async def _generate_hyde(query: str, llm: ChatOpenAI) -> List[str]:
    """Generate a hypothetical document embedding string."""
    chain = HYDE_PROMPT | llm
    try:
        res = await chain.ainvoke({"query": query})
        content = res.content if hasattr(res, "content") else str(res)
        # HyDE returns the generated text as the search query
        return [content.strip()] if content.strip() else []
    except Exception as exc:
        logger.warning("HyDE expansion failed: %s", exc)
        return []


async def _generate_step_back(query: str, llm: ChatOpenAI) -> List[str]:
    """Generate a foundational step-back query."""
    chain = STEP_BACK_PROMPT | llm
    try:
        res = await chain.ainvoke({"query": query})
        content = res.content if hasattr(res, "content") else str(res)
        return [content.strip()] if content.strip() else []
    except Exception as exc:
        logger.warning("Step-back expansion failed: %s", exc)
        return []


async def _generate_decomposition(query: str, llm: ChatOpenAI) -> List[str]:
    """Decompose a complex query into atomic sub-questions."""
    chain = DECOMPOSITION_PROMPT | llm
    try:
        res = await chain.ainvoke({"query": query})
        content = res.content if hasattr(res, "content") else str(res)
        queries = [q.strip() for q in content.split("\n") if q.strip()]
        return queries
    except Exception as exc:
        logger.warning("Decomposition expansion failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

async def expand_query(query: str, strategies: QueryStrategies) -> List[str]:
    """Apply the active query expansion strategies to generate additional search queries.
    
    Args:
        query: The original user query.
        strategies: Pydantic model defining which strategies are toggled on.
        
    Returns:
        A list of generated queries, plus the original query as the first item.
    """
    expanded_queries = [query]
    
    # If no strategies are active, return just the original query
    if not (strategies.multi_query or strategies.hyde or strategies.step_back or strategies.decomposition):
        return expanded_queries
        
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.0
    )
    
    logger.info("Applying active query expansion strategies...")
    
    # Run active strategies sequentially (or concurrently in a production environment)
    # For MVP, we use sequential await to simplify error handling
    if strategies.multi_query:
        logger.debug("Applying Multi-Query expansion...")
        mq = await _generate_multi_queries(query, llm)
        expanded_queries.extend(mq)
        
    if strategies.hyde:
        logger.debug("Applying HyDE expansion...")
        hyde = await _generate_hyde(query, llm)
        expanded_queries.extend(hyde)
        
    if strategies.step_back:
        logger.debug("Applying Step-Back expansion...")
        sb = await _generate_step_back(query, llm)
        expanded_queries.extend(sb)
        
    if strategies.decomposition:
        logger.debug("Applying Decomposition expansion...")
        decomp = await _generate_decomposition(query, llm)
        expanded_queries.extend(decomp)
        
    # Deduplicate queries while preserving order
    seen = set()
    unique_queries = []
    for q in expanded_queries:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)
            
    logger.info("Query expansion generated %d unique queries.", len(unique_queries))
    return unique_queries
