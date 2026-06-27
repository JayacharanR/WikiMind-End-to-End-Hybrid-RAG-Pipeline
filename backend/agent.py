"""Agentic Orchestration (LangGraph CRAG/Self-RAG).

Defines the LangGraph state machine orchestrating the Tri-Brid Hybrid RAG pipeline.
Implements nodes for query expansion, retrieval, document grading (CRAG), web search
fallback, and safe generation with hallucination and answer quality loops (Self-RAG).
"""

import logging
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient

from backend.config import get_settings
from backend.llmops import get_langfuse_handler, safe_generate
from backend.models import QueryStrategies
from backend.page_index import navigate_article
from backend.query_expansion import expand_query
from backend.retrieval import hybrid_search, route_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The state dictionary for the LangGraph agent."""
    query: str
    expanded_queries: List[str]
    documents: List[Dict]
    generation: str
    web_results: List[Dict]
    retrieval_grade: str
    hallucination_grade: str
    answer_grade: str
    steps: int
    active_strategies: QueryStrategies
    retry_count: int


# ---------------------------------------------------------------------------
# Node Definitions
# ---------------------------------------------------------------------------

async def node_expand_query(state: AgentState) -> Dict:
    """Node: Expand the original query using active strategies."""
    query = state["query"]
    strategies = state["active_strategies"]
    steps = state.get("steps", 0) + 1
    
    logger.info("--- NODE: EXPAND QUERY ---")
    expanded_queries = await expand_query(query, strategies)
    
    return {"expanded_queries": expanded_queries, "steps": steps}


async def node_retrieve(state: AgentState) -> Dict:
    """Node: Retrieve documents using the Tri-Brid engine."""
    queries_to_search = state.get("expanded_queries", [state["query"]])
    steps = state.get("steps", 0) + 1
    
    logger.info("--- NODE: RETRIEVE ---")
    all_documents = []
    seen_ids = set()
    
    # 1. Execute hybrid search for each query (deduplicating by chunk ID)
    for q in queries_to_search:
        # Simple queries bypass reranker for speed, complex ones use it
        route = await route_query(q)
        apply_reranker = (route == "complex")
        
        docs, _ = await hybrid_search(q, apply_reranker=apply_reranker)
        for doc in docs:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_documents.append(doc)
                
    # 2. PageIndex extraction (L3) if enabled
    strategies = state["active_strategies"]
    if strategies.page_index and all_documents:
        logger.info("Executing PageIndex extraction on top documents...")
        # For MVP, apply PageIndex to the top 2 unique articles
        processed_titles = set()
        for doc in all_documents:
            title = doc.get("title")
            if title and title not in processed_titles:
                processed_titles.add(title)
                # In a real system, you'd fetch the full markdown from Qdrant payload or Wikipedia API here.
                # For this MVP, we assume `full_text` was stored in the first chunk or fetched dynamically.
                # We will pass dummy text here since we didn't store full text in Qdrant payload yet.
                dummy_text = f"== {title} ==\nThis is a placeholder for the full article text."
                extracted = await navigate_article(state["query"], title, dummy_text)
                if extracted:
                    all_documents.append({
                        "id": f"pageindex_{title}",
                        "title": title,
                        "content": extracted,
                        "score": 1.0, # High confidence for structural extraction
                    })
            if len(processed_titles) >= 2:
                break
                
    # Sort by score descending
    all_documents.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    # Limit final context window
    settings = get_settings()
    final_docs = all_documents[:settings.retrieval_top_k]
    
    return {"documents": final_docs, "steps": steps}


async def node_grade_documents(state: AgentState) -> Dict:
    """Node: Evaluate document relevance to the query (CRAG)."""
    query = state["query"]
    documents = state.get("documents", [])
    steps = state.get("steps", 0) + 1
    
    logger.info("--- NODE: GRADE DOCUMENTS ---")
    
    if not documents:
        return {"documents": [], "retrieval_grade": "irrelevant", "steps": steps}
        
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. "
                   "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. "
                   "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."),
        ("user", "Retrieved document: \n\n {document} \n\n User question: {query} \n\n Score (yes/no):")
    ])
    
    chain = prompt | llm
    
    filtered_docs = []
    for doc in documents:
        res = await chain.ainvoke({"query": query, "document": doc.get("content", "")})
        grade = (res.content if hasattr(res, "content") else str(res)).strip().lower()
        if "yes" in grade:
            filtered_docs.append(doc)
            
    # If we kept at least one document, consider the retrieval a success
    grade = "relevant" if filtered_docs else "irrelevant"
    logger.info("Document grading result: %s (%d kept out of %d)", grade, len(filtered_docs), len(documents))
    
    return {"documents": filtered_docs, "retrieval_grade": grade, "steps": steps}


async def node_web_search(state: AgentState) -> Dict:
    """Node: Fallback to Tavily web search if Wikipedia retrieval fails."""
    query = state["query"]
    steps = state.get("steps", 0) + 1
    
    logger.info("--- NODE: WEB SEARCH FALLBACK ---")
    
    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("Tavily API key not configured. Skipping web search.")
        return {"documents": state.get("documents", []), "steps": steps}
        
    client = TavilyClient(api_key=settings.tavily_api_key)
    
    try:
        # Run sync client in thread (using asyncio.to_thread in prod)
        response = client.search(query=query, max_results=3)
        web_results = []
        for r in response.get("results", []):
            web_results.append({
                "id": r.get("url"),
                "title": r.get("title", "Web Result"),
                "content": r.get("content", ""),
                "url": r.get("url", ""),
                "score": r.get("score", 0.0),
            })
            
        # Append web results to existing documents
        documents = state.get("documents", [])
        documents.extend(web_results)
        return {"documents": documents, "web_results": web_results, "steps": steps}
        
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return {"steps": steps}


async def node_generate(state: AgentState) -> Dict:
    """Node: Generate response using NeMo Guardrails."""
    query = state["query"]
    documents = state.get("documents", [])
    steps = state.get("steps", 0) + 1
    
    logger.info("--- NODE: GENERATE ---")
    
    context = "\n\n".join(f"Title: {d.get('title')}\nContent: {d.get('content')}" for d in documents)
    
    # safe_generate is wrapped with @observe and NeMo Guardrails
    generation = await safe_generate(query=query, context=context)
    
    return {"generation": generation, "steps": steps}


async def node_check_hallucination(state: AgentState) -> Dict:
    """Node: Evaluate if the generation is grounded in the retrieved documents."""
    query = state["query"]
    documents = state.get("documents", [])
    generation = state["generation"]
    steps = state.get("steps", 0) + 1
    retry_count = state.get("retry_count", 0)
    
    logger.info("--- NODE: CHECK HALLUCINATION ---")
    
    # We invoke the guardrails output check. 
    # For a pure LangGraph implementation without Guardrails output rails, 
    # we would use an LLM chain here. Since we have NeMo Guardrails, we can use an LLM chain
    # mirroring the hallucination check for explicit routing control in LangGraph.
    
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n"
                   "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."),
        ("user", "Set of facts: \n\n {documents} \n\n LLM generation: {generation} \n\n Score (yes/no):")
    ])
    
    context = "\n\n".join(f"Title: {d.get('title')}\nContent: {d.get('content')}" for d in documents)
    chain = prompt | llm
    
    res = await chain.ainvoke({"documents": context, "generation": generation})
    grade = (res.content if hasattr(res, "content") else str(res)).strip().lower()
    
    if "yes" in grade:
        logger.info("Hallucination check passed (grounded).")
        return {"hallucination_grade": "grounded", "steps": steps}
    else:
        logger.warning("Hallucination check failed (not grounded).")
        return {"hallucination_grade": "hallucinated", "steps": steps, "retry_count": retry_count + 1}


async def node_check_answer_quality(state: AgentState) -> Dict:
    """Node: Evaluate if the generation answers the original query."""
    query = state["query"]
    generation = state["generation"]
    steps = state.get("steps", 0) + 1
    retry_count = state.get("retry_count", 0)
    
    logger.info("--- NODE: CHECK ANSWER QUALITY ---")
    
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing whether an answer addresses / resolves a question. \n"
                   "Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."),
        ("user", "User question: \n\n {query} \n\n LLM generation: {generation} \n\n Score (yes/no):")
    ])
    
    chain = prompt | llm
    
    res = await chain.ainvoke({"query": query, "generation": generation})
    grade = (res.content if hasattr(res, "content") else str(res)).strip().lower()
    
    if "yes" in grade:
        logger.info("Answer quality check passed (useful).")
        return {"answer_grade": "useful", "steps": steps}
    else:
        logger.warning("Answer quality check failed (not useful).")
        return {"answer_grade": "not_useful", "steps": steps, "retry_count": retry_count + 1}


# ---------------------------------------------------------------------------
# Conditional Edges
# ---------------------------------------------------------------------------

def route_after_grading(state: AgentState) -> Literal["web_search", "generate"]:
    """Route based on document relevance."""
    if state.get("retrieval_grade") == "irrelevant":
        return "web_search"
    return "generate"


def route_after_hallucination(state: AgentState) -> Literal["retrieve", "check_answer_quality"]:
    """Route based on grounding. Retry retrieval if hallucinated (max 2 retries)."""
    if state.get("hallucination_grade") == "hallucinated" and state.get("retry_count", 0) < 2:
        return "retrieve"
    return "check_answer_quality"


def route_after_answer_quality(state: AgentState) -> Literal["expand_query", END]:
    """Route based on answer usefulness. Expand query if not useful (max 2 retries)."""
    if state.get("answer_grade") == "not_useful" and state.get("retry_count", 0) < 2:
        return "expand_query"
    return END


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------

def compile_agent_graph():
    """Compile the LangGraph state machine workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("expand_query", node_expand_query)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade_documents", node_grade_documents)
    workflow.add_node("web_search", node_web_search)
    workflow.add_node("generate", node_generate)
    workflow.add_node("check_hallucination", node_check_hallucination)
    workflow.add_node("check_answer_quality", node_check_answer_quality)
    
    # Set entry point
    workflow.set_entry_point("expand_query")
    
    # Add standard edges
    workflow.add_edge("expand_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_hallucination")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "web_search": "web_search",
            "generate": "generate",
        }
    )
    
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "retrieve": "retrieve",
            "check_answer_quality": "check_answer_quality",
        }
    )
    
    workflow.add_conditional_edges(
        "check_answer_quality",
        route_after_answer_quality,
        {
            "expand_query": "expand_query",
            END: END,
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    logger.info("LangGraph CRAG/Self-RAG workflow compiled successfully.")
    return app


# Singleton graph instance
agent_app = compile_agent_graph()
