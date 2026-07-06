"""WikiMind Streamlit Frontend (Multi-Page).

Provides an interactive chat interface for the Two-Stage Agentic RAG pipeline
as the default page, with additional pages for A/B strategy comparison and
evaluation results browsing. Includes sidebar toggles for configuring query
expansion and retrieval strategies, displays real-time SSE streaming from the
FastAPI backend, and renders retrieved sources and execution metadata.
"""

import json
import logging
import os
import requests
from typing import Dict, Any

import sseclient
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")


def configure_sidebar() -> Dict[str, Any]:
    """Render the configuration sidebar and return the selected strategies."""
    st.sidebar.title("WikiMind Configuration")
    
    st.sidebar.subheader("Retrieval Architecture")
    st.sidebar.markdown(
        "WikiMind uses **Two-Stage Hybrid RAG**: a local article-level index "
        "identifies the relevant Wikipedia articles, then Qdrant hybrid search "
        "(Dense + Sparse + RRF + Reranker) extracts precise chunks from those articles."
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Query Expansion")
    st.sidebar.markdown("Toggle parallel expansion strategies.")
    
    multi_query = st.sidebar.toggle(
        "Multi-Query Reformulation", 
        value=False,
        help="Generate semantic alternatives to the original query."
    )
    
    hyde = st.sidebar.toggle(
        "HyDE (Hypothetical Document Embeddings)", 
        value=False,
        help="Generate a hypothetical answer to embed for semantic search."
    )
    
    step_back = st.sidebar.toggle(
        "Step-Back Abstraction", 
        value=False,
        help="Abstract the query to a higher-level foundational question."
    )
    
    decomposition = st.sidebar.toggle(
        "Query Decomposition", 
        value=False,
        help="Break complex queries into atomic sub-questions."
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Time-Travel Mode")
    time_travel = st.sidebar.toggle(
        "Enable Time-Travel",
        value=False,
        help="Query the knowledge base as it existed on a specific date.",
    )
    as_of_date = None
    if time_travel:
        from datetime import date
        selected_date = st.sidebar.date_input(
            "As of date:",
            value=date.today(),
            help="Retrieve articles ingested on or before this date.",
        )
        as_of_date = selected_date.isoformat() + "T23:59:59Z"
        st.sidebar.caption(f"Querying as of: {as_of_date}")

    st.sidebar.divider()
    
    st.sidebar.subheader("System Health")
    if st.sidebar.button("Check Backend Health"):
        try:
            res = requests.get(f"{API_URL}/health", timeout=5)
            if res.status_code == 200:
                data = res.json()
                status = data.get("status")
                st.sidebar.success(f"Status: {status.upper()}")
                for comp in data.get("components", []):
                    icon = "OK" if comp.get("healthy") else "FAIL"
                    latency = comp.get("latency_ms", "N/A")
                    st.sidebar.text(f"{icon} {comp.get('name')}: {latency}ms")
            else:
                st.sidebar.error("Backend returned an error.")
        except Exception as exc:
            st.sidebar.error(f"Cannot reach backend: {exc}")
            
    return {
        "multi_query": multi_query,
        "hyde": hyde,
        "step_back": step_back,
        "decomposition": decomposition,
        "as_of_date": as_of_date,
    }


def stream_chat_response(query: str, strategies: Dict[str, bool]):
    """Stream the response from the FastAPI SSE endpoint."""
    as_of_date = strategies.pop("as_of_date", None)
    payload = {
        "query": query,
        "strategies": strategies,
    }
    if as_of_date:
        payload["as_of_date"] = as_of_date
    
    try:
        # Use requests to get the SSE stream
        response = requests.post(
            f"{API_URL}/chat", 
            json=payload,
            stream=True,
            headers={'Accept': 'text/event-stream'}
        )
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        for event in client.events():
            if event.event == "update":
                data = json.loads(event.data)
                node = data.get("node")
                status = data.get("status")
                status_placeholder.info(f"Agent working: {status} ({node})")
                
            elif event.event == "final":
                data = json.loads(event.data)
                
                # Clear status
                status_placeholder.empty()
                
                # Render answer
                answer = data.get("answer", "")
                answer_placeholder.markdown(answer)
                
                # Render sources
                sources = data.get("sources", [])
                if sources:
                    with sources_placeholder.expander(f"View {len(sources)} Retrieved Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**[{i+1}] {source.get('title')}** (Score: {source.get('score', 0):.2f})")
                            st.markdown(f"> {source.get('content')[:300]}...")
                            if source.get('url'):
                                st.markdown(f"[Read on Wikipedia]({source.get('url')})")
                            st.divider()
                            
                # Render metadata
                metadata = data.get("metadata", {})
                with st.expander("Execution Metadata"):
                    st.json(metadata)
                    
                # Save to session state
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "metadata": metadata})
                break
                
            elif event.event == "error":
                data = json.loads(event.data)
                st.error(f"Backend error: {data.get('detail')}")
                break
                
    except Exception as exc:
        st.error(f"Failed to connect to backend: {exc}")


def chat_page():
    """Render the main chat interface page."""
    st.title("WikiMind RAG Pipeline")
    st.markdown("Ask complex questions. The agent uses Two-Stage Hybrid RAG: a local article index identifies Wikipedia articles, then Qdrant hybrid search extracts precise answers.")
    
    # Sidebar config
    strategies = configure_sidebar()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("sources"):
                    with st.expander(f"View {len(message['sources'])} Retrieved Sources"):
                        for i, source in enumerate(message['sources']):
                            st.markdown(f"**[{i+1}] {source.get('title')}** (Score: {source.get('score', 0):.2f})")
                            st.markdown(f"> {source.get('content')[:300]}...")
                            if source.get('url'):
                                st.markdown(f"[Read on Wikipedia]({source.get('url')})")
                            st.divider()
                if message.get("metadata"):
                    with st.expander("Execution Metadata"):
                        st.json(message["metadata"])
                        
    # Chat input
    if query := st.chat_input("Ask Wikipedia something complex..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
            
        # Add to session state
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display assistant response stream
        with st.chat_message("assistant"):
            stream_chat_response(query, strategies)


def main():
    """Multi-page Streamlit application entry point."""
    st.set_page_config(
        page_title="WikiMind | Hybrid RAG Pipeline",
        page_icon="W",
        layout="wide",
    )

    from frontend.pages.ab_dashboard import render_ab_dashboard
    from frontend.pages.eval_results import render_eval_results

    pages = {
        "Chat": st.Page(chat_page, title="Chat", icon=":material/chat:"),
        "A/B Dashboard": st.Page(render_ab_dashboard, title="A/B Dashboard", icon=":material/compare_arrows:"),
        "Eval Results": st.Page(render_eval_results, title="Eval Results", icon=":material/analytics:"),
    }

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()

