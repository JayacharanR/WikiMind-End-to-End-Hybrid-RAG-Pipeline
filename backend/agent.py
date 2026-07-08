"""Agentic Orchestration (Search-Scoped CRAG/Self-RAG).

Defines the LangGraph state machine orchestrating the Search-Scoped Hybrid RAG
pipeline. Uses a local article-level Qdrant index (``wikimind_articles``) to
identify relevant Wikipedia articles, then performs article-scoped hybrid
retrieval in Qdrant. Implements batched document grading, hallucination
checking, and answer quality loops with separate retry counters and a hard
step budget.
"""

import json
import logging
from typing import Dict, List, Literal, TypedDict
from urllib.parse import unquote, urlparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.article_index import search_articles
from backend.config import get_settings
from backend.llmops import get_langfuse_handler, safe_generate
from backend.models import QueryStrategies
from backend.query_expansion import expand_query
from backend.retrieval import extract_title_from_wikipedia_url, hybrid_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The state dictionary for the LangGraph agent."""
    query: str
    expanded_queries: List[str]
    target_articles: List[str]
    documents: List[Dict]
    web_snippets: List[Dict]
    generation: str
    retrieval_grade: str
    hallucination_grade: str
    answer_grade: str
    steps: int
    active_strategies: QueryStrategies
    hallucination_retries: int
    answer_retries: int
    as_of_date: str


# ---------------------------------------------------------------------------
# Helper: Get LLM Instance
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.0, max_tokens: int = 256) -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for OpenRouter."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Node Definitions
# ---------------------------------------------------------------------------

async def node_expand_query(state: AgentState) -> Dict:
    """Node: Expand the original query using active strategies."""
    query = state["query"]
    strategies = state["active_strategies"]
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: EXPAND QUERY (step %d) ---", steps)
    expanded_queries = await expand_query(query, strategies)

    return {"expanded_queries": expanded_queries, "steps": steps}


async def node_identify_articles(state: AgentState) -> Dict:
    """Node: Search the local article-level Qdrant index to identify the
    most relevant Wikipedia article(s) for the query.

    Uses the ``wikimind_articles`` collection (dense-only, one vector per
    article built from title + first two paragraphs) to find the top-K
    articles. This replaces the previous Tavily web search, making the
    pipeline fully offline with no external API dependencies for retrieval.
    """
    query = state["query"]
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: IDENTIFY ARTICLES (step %d) ---", steps)

    try:
        target_articles = await search_articles(query)

        logger.info(
            "Local article index identified %d article(s): %s",
            len(target_articles),
            ", ".join(target_articles),
        )

        return {
            "target_articles": target_articles,
            "web_snippets": [],
            "steps": steps,
        }

    except Exception as exc:
        logger.error("Article index search failed: %s", exc)
        return {"target_articles": [], "web_snippets": [], "steps": steps}


async def node_graph_search(state: AgentState) -> Dict:
    """Node: Enrich retrieval with knowledge graph traversal.

    Uses spaCy NER to extract entities from the query, then traverses the
    co-occurrence knowledge graph (stored in Redis) up to 2 hops to find
    related entities and their source articles. These are merged into the
    target_articles list to broaden article-scoped retrieval.

    Only runs when the ``knowledge_graph`` strategy is enabled.
    """
    query = state["query"]
    target_articles = state.get("target_articles", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: GRAPH SEARCH (step %d) ---", steps)

    try:
        from backend.knowledge_graph import graph_search

        graph_results = await graph_search(query, max_hops=2, max_results=10)

        # Extract unique source titles from graph traversal
        graph_titles = set()
        for result in graph_results:
            source = result.get("source_title", "")
            if source and source not in target_articles:
                graph_titles.add(source)

        # Merge graph-discovered articles with existing targets
        merged = list(target_articles) + sorted(graph_titles)

        logger.info(
            "Graph search added %d article(s): %s",
            len(graph_titles),
            ", ".join(sorted(graph_titles)) if graph_titles else "(none)",
        )

        return {"target_articles": merged, "steps": steps}

    except Exception as exc:
        logger.warning("Graph search failed (non-fatal): %s", exc)
        return {"steps": steps}


async def node_retrieve(state: AgentState) -> Dict:
    """Node: Retrieve documents using article-scoped hybrid search.

    Uses the article titles identified by the article-level index to filter
    the Qdrant search. If no articles were identified, falls back to unscoped
    search across the entire collection.
    """
    queries_to_search = state.get("expanded_queries", [state["query"]])
    target_articles = state.get("target_articles", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: RETRIEVE (step %d) ---", steps)

    settings = get_settings()
    all_documents = []
    seen_ids = set()

    for q in queries_to_search:
        docs, _ = await hybrid_search(
            q,
            article_titles=target_articles if target_articles else None,
            as_of_date=state.get("as_of_date")
        )
        for doc in docs:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_documents.append(doc)

    # Sort by reranker score descending
    all_documents.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Cap at max_generation_docs to keep the context focused
    final_docs = all_documents[:settings.max_generation_docs]

    logger.info(
        "Retrieve produced %d unique chunks (capped to %d).",
        len(all_documents),
        len(final_docs),
    )

    # If scoped search returned nothing but we have web snippets, use those
    if not final_docs and state.get("web_snippets"):
        logger.info("Scoped search returned no results. Using Tavily web snippets as fallback.")
        final_docs = state["web_snippets"][:settings.max_generation_docs]

    return {"documents": final_docs, "steps": steps}


async def node_grade_documents(state: AgentState) -> Dict:
    """Node: Evaluate document relevance using a single batched LLM call.

    Instead of making N serial LLM calls (one per document), concatenates
    all documents with numbered indices and asks the LLM to return a
    comma-separated list of relevant document numbers.
    """
    query = state["query"]
    documents = state.get("documents", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: GRADE DOCUMENTS (step %d) ---", steps)

    if not documents:
        return {"documents": [], "retrieval_grade": "irrelevant", "steps": steps}

    llm = _get_llm(temperature=0.0, max_tokens=100)

    # Build a numbered list of document snippets for batched grading
    doc_summaries = []
    for i, doc in enumerate(documents):
        content = doc.get("content", "")[:300]
        doc_summaries.append(f"[{i}] Title: {doc.get('title', 'Unknown')}\n{content}")

    docs_text = "\n\n".join(doc_summaries)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a relevance grader. Given a user question and a numbered list of "
         "retrieved document snippets, identify which documents are relevant to "
         "answering the question.\n"
         "Return ONLY a comma-separated list of the relevant document numbers "
         "(e.g., '0,2,4'). If none are relevant, return 'NONE'. "
         "Do not include any other text."),
        ("user",
         "User question: {query}\n\n"
         "Documents:\n{documents}\n\n"
         "Relevant document numbers:"),
    ])

    chain = prompt | llm

    try:
        res = await chain.ainvoke({"query": query, "documents": docs_text})
        content = (res.content if hasattr(res, "content") else str(res)).strip()

        if "none" in content.lower():
            logger.info("Batched grading: no documents deemed relevant.")
            return {"documents": [], "retrieval_grade": "irrelevant", "steps": steps}

        # Parse the comma-separated indices
        relevant_indices = set()
        for part in content.replace(" ", "").split(","):
            try:
                idx = int(part)
                if 0 <= idx < len(documents):
                    relevant_indices.add(idx)
            except ValueError:
                continue

        filtered_docs = [documents[i] for i in sorted(relevant_indices)]

        grade = "relevant" if filtered_docs else "irrelevant"
        logger.info(
            "Batched grading result: %s (%d kept out of %d)",
            grade, len(filtered_docs), len(documents),
        )

        return {"documents": filtered_docs, "retrieval_grade": grade, "steps": steps}

    except Exception as exc:
        logger.warning("Batched grading failed: %s. Keeping all documents.", exc)
        return {"documents": documents, "retrieval_grade": "relevant", "steps": steps}


async def node_generate_from_web(state: AgentState) -> Dict:
    """Node: Generate a response using fallback context.

    Called when scoped retrieval returns no relevant documents. Uses any
    available web snippets or generates a response acknowledging that no
    relevant context was found.
    """
    query = state["query"]
    web_snippets = state.get("web_snippets", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: GENERATE FALLBACK (step %d) ---", steps)

    if web_snippets:
        context = "\n\n".join(
            f"Title: {d.get('title')}\nURL: {d.get('url', 'N/A')}\nContent: {d.get('content')}"
            for d in web_snippets
        )
    else:
        context = "No relevant context was found."

    generation = await safe_generate(query=query, context=context)
    return {"generation": generation, "documents": web_snippets, "steps": steps}


async def node_generate(state: AgentState) -> Dict:
    """Node: Generate response using NeMo Guardrails with retrieved context."""
    query = state["query"]
    documents = state.get("documents", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: GENERATE (step %d) ---", steps)

    context = "\n\n".join(
        f"Title: {d.get('title')}\nURL: {d.get('url', 'N/A')}\nContent: {d.get('content')}"
        for d in documents
    )

    generation = await safe_generate(query=query, context=context)
    return {"generation": generation, "steps": steps}


async def node_check_hallucination(state: AgentState) -> Dict:
    """Node: Evaluate if the generation is grounded in the retrieved documents."""
    documents = state.get("documents", [])
    generation = state["generation"]
    steps = state.get("steps", 0) + 1
    retries = state.get("hallucination_retries", 0)

    logger.info("--- NODE: CHECK HALLUCINATION (step %d) ---", steps)

    llm = _get_llm(temperature=0.0, max_tokens=10)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader assessing whether an LLM generation is grounded in "
         "a set of retrieved facts. Give a binary score 'yes' or 'no'. "
         "'Yes' means that the answer is grounded in the facts."),
        ("user",
         "Set of facts:\n\n{documents}\n\n"
         "LLM generation: {generation}\n\n"
         "Score (yes/no):"),
    ])

    context = "\n\n".join(
        f"Title: {d.get('title')}\nContent: {d.get('content')}"
        for d in documents
    )
    chain = prompt | llm

    try:
        res = await chain.ainvoke({"documents": context, "generation": generation})
        grade = (res.content if hasattr(res, "content") else str(res)).strip().lower()

        if "yes" in grade:
            logger.info("Hallucination check passed (grounded).")
            return {"hallucination_grade": "grounded", "steps": steps}
        else:
            logger.warning("Hallucination check failed (not grounded). Retry %d.", retries + 1)
            return {
                "hallucination_grade": "hallucinated",
                "steps": steps,
                "hallucination_retries": retries + 1,
            }
    except Exception as exc:
        logger.warning("Hallucination check failed with error: %s. Passing through.", exc)
        return {"hallucination_grade": "grounded", "steps": steps}


async def node_check_answer_quality(state: AgentState) -> Dict:
    """Node: Evaluate if the generation answers the original query."""
    query = state["query"]
    generation = state["generation"]
    steps = state.get("steps", 0) + 1
    retries = state.get("answer_retries", 0)

    logger.info("--- NODE: CHECK ANSWER QUALITY (step %d) ---", steps)

    llm = _get_llm(temperature=0.0, max_tokens=10)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader assessing whether an answer addresses a question. "
         "Give a binary score 'yes' or 'no'. 'Yes' means the answer resolves the question."),
        ("user",
         "User question:\n\n{query}\n\n"
         "LLM generation: {generation}\n\n"
         "Score (yes/no):"),
    ])

    chain = prompt | llm

    try:
        res = await chain.ainvoke({"query": query, "generation": generation})
        grade = (res.content if hasattr(res, "content") else str(res)).strip().lower()

        if "yes" in grade:
            logger.info("Answer quality check passed (useful).")
            return {"answer_grade": "useful", "steps": steps}
        else:
            logger.warning("Answer quality check failed (not useful). Retry %d.", retries + 1)
            return {
                "answer_grade": "not_useful",
                "steps": steps,
                "answer_retries": retries + 1,
            }
    except Exception as exc:
        logger.warning("Answer quality check failed with error: %s. Passing through.", exc)
        return {"answer_grade": "useful", "steps": steps}


# ---------------------------------------------------------------------------
# Conditional Edges (with step budget enforcement)
# ---------------------------------------------------------------------------

def _is_over_budget(state: AgentState) -> bool:
    """Check if the graph has exceeded its step budget."""
    settings = get_settings()
    return state.get("steps", 0) >= settings.max_graph_steps


def route_after_grading(state: AgentState) -> Literal["generate_from_web", "generate"]:
    """Route based on document relevance after grading.

    If no relevant documents remain, fall back to generating a response
    without grounded context.
    """
    if state.get("retrieval_grade") == "irrelevant":
        return "generate_from_web"
    return "generate"


def route_after_hallucination(
    state: AgentState,
) -> Literal["generate", "check_answer_quality"]:
    """Route based on grounding. Retry generation if hallucinated (max 2 retries).

    Does not loop back to retrieve -- the documents are already scoped
    and reranked. Instead, retries the generation step with the same context.
    """
    if _is_over_budget(state):
        logger.warning("Step budget exhausted. Forcing answer quality check.")
        return "check_answer_quality"

    if (
        state.get("hallucination_grade") == "hallucinated"
        and state.get("hallucination_retries", 0) < 2
    ):
        return "generate"
    return "check_answer_quality"


def route_after_answer_quality(state: AgentState) -> Literal["expand_query", "__end__"]:
    """Route based on answer usefulness. Expand query if not useful (max 2 retries)."""
    if _is_over_budget(state):
        logger.warning("Step budget exhausted. Terminating graph.")
        return END

    if (
        state.get("answer_grade") == "not_useful"
        and state.get("answer_retries", 0) < 2
    ):
        return "expand_query"
    return END


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------

def compile_agent_graph():
    """Compile the LangGraph state machine workflow.

    Graph topology:
        expand_query -> identify_articles
            -> [if knowledge_graph enabled] -> graph_search -> retrieve
            -> [if knowledge_graph disabled] -> retrieve
        retrieve -> grade_documents
            -> [if irrelevant] -> generate_from_web -> END
            -> [if relevant]   -> generate -> check_hallucination
                -> [if hallucinated, retries < 2] -> generate (retry)
                -> [if grounded] -> check_answer_quality
                    -> [if not useful, retries < 2] -> expand_query (re-expand and re-retrieve)
                    -> [if useful or budget exhausted] -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("expand_query", node_expand_query)
    workflow.add_node("identify_articles", node_identify_articles)
    workflow.add_node("graph_search", node_graph_search)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade_documents", node_grade_documents)
    workflow.add_node("generate_from_web", node_generate_from_web)
    workflow.add_node("generate", node_generate)
    workflow.add_node("check_hallucination", node_check_hallucination)
    workflow.add_node("check_answer_quality", node_check_answer_quality)

    # Set entry point
    workflow.set_entry_point("expand_query")

    # Standard edges
    workflow.add_edge("expand_query", "identify_articles")
    workflow.add_edge("graph_search", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("generate_from_web", END)
    workflow.add_edge("generate", "check_hallucination")

    # Conditional: after identify_articles, optionally run graph_search
    def route_after_identify(state: AgentState) -> Literal["graph_search", "retrieve"]:
        """Route to graph_search if the knowledge_graph strategy is active."""
        strategies = state.get("active_strategies")
        if strategies and getattr(strategies, "knowledge_graph", False):
            return "graph_search"
        return "retrieve"

    workflow.add_conditional_edges(
        "identify_articles",
        route_after_identify,
        {
            "graph_search": "graph_search",
            "retrieve": "retrieve",
        },
    )

    # Conditional edges
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_from_web": "generate_from_web",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "generate": "generate",
            "check_answer_quality": "check_answer_quality",
        },
    )

    workflow.add_conditional_edges(
        "check_answer_quality",
        route_after_answer_quality,
        {
            "expand_query": "expand_query",
            END: END,
        },
    )

    app = workflow.compile()
    logger.info("Two-Stage CRAG/Self-RAG workflow compiled successfully.")
    return app


# Singleton graph instance
agent_app = compile_agent_graph()
