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
import time
from typing import Dict, Any

import sseclient
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")


def inject_llama_css():
    st.markdown(
        """
        <style>
        /* Open WebUI / Llama WebUI Theme */
        
        /* General Backgrounds */
        .stApp, .stAppViewBlockContainer {
            background-color: #111111 !important;
            color: #E5E5E5 !important;
            font-family: system-ui, -apple-system, sans-serif !important;
        }
        [data-testid="stSidebar"] {
            background-color: #0A0A0A !important;
            border-right: none !important;
        }
        
        /* Chat Input Container */
        .stChatInputContainer {
            background-color: transparent !important;
            border: none !important;
            padding-bottom: 2rem !important;
        }
        .stChatInputContainer > div {
            background-color: #2F2F2F !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 4px 10px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        }
        .stChatInputContainer textarea {
            color: #FFFFFF !important;
        }
        
        /* Default Streamlit Chat Bubble overrides */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0.5rem !important;
            animation: slideUp 0.3s ease-out forwards;
            opacity: 0;
            transform: translateY(10px);
        }
        
        /* Flexbox overrides for User vs Assistant */
        
        /* Assistant Message */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            display: flex !important;
            flex-direction: row !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
            color: #E5E5E5 !important;
        }
        
        /* User Message */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            display: flex !important;
            flex-direction: row-reverse !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
            background-color: #2F2F2F !important;
            padding: 12px 18px !important;
            border-radius: 18px 18px 0px 18px !important;
            display: inline-block !important;
            max-width: 80% !important;
            color: #FFFFFF !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageAvatar {
            display: none !important; /* Hide user avatar */
        }
        
        /* Clean Headers */
        [data-testid="stHeader"] {
            display: none !important;
        }
        footer {
            display: none !important;
        }
        
        /* Empty State Headers */
        .empty-state-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 600;
            margin-top: 15vh;
            color: #FFFFFF;
        }
        .empty-state-subtitle {
            text-align: center;
            font-size: 1rem;
            color: #888888;
            margin-bottom: 2rem;
        }
        
        /* Animations */
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


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
        start_time = time.time()
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
                total_time = time.time() - start_time
                data = json.loads(event.data)
                
                # Clear status
                status_placeholder.empty()
                
                # Render answer
                answer = data.get("answer", "")
                
                # If answer is a JSON tool call, parse it out
                try:
                    ans_json = json.loads(answer)
                    if isinstance(ans_json, dict) and ans_json.get("name") == "generate_text":
                        params = ans_json.get("parameters", {})
                        if isinstance(params, str):
                            params = json.loads(params)
                        answer = params.get("input", answer)
                except Exception:
                    pass
                
                answer_placeholder.markdown(answer)
                
                # Render metadata and add total time taken
                metadata = data.get("metadata", {})
                metadata["total_time_seconds"] = round(total_time, 2)
                
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
                            
                # Render expanded queries if any
                expanded_queries = metadata.get("expanded_queries", [])
                strategies_used = metadata.get("strategies_used", [])
                if len(expanded_queries) > 1:
                    if "hyde" in strategies_used:
                        with st.expander("View Hypothetical Document (HyDE)"):
                            for q in expanded_queries[1:]:
                                st.markdown(f"> {q}")
                    else:
                        with st.expander("View Expanded Queries"):
                            for q in expanded_queries[1:]:
                                st.markdown(f"- {q}")
                            
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
    
    # Sidebar config
    strategies = configure_sidebar()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if len(st.session_state.messages) == 0:
        st.markdown('<div class="empty-state-title">Hello there</div>', unsafe_allow_html=True)
        st.markdown('<div class="empty-state-subtitle">Type a message to get started with WikiMind</div>', unsafe_allow_html=True)
        
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
                    expanded_queries = message["metadata"].get("expanded_queries", [])
                    strategies_used = message["metadata"].get("strategies_used", [])
                    if len(expanded_queries) > 1:
                        if "hyde" in strategies_used:
                            with st.expander("View Hypothetical Document (HyDE)"):
                                for q in expanded_queries[1:]:
                                    st.markdown(f"> {q}")
                        else:
                            with st.expander("View Expanded Queries"):
                                for q in expanded_queries[1:]:
                                    st.markdown(f"- {q}")
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
        layout="wide",
    )
    
    inject_llama_css()

    from frontend.pages.ab_dashboard import render_ab_dashboard
    from frontend.pages.eval_results import render_eval_results

    pages = [
        st.Page(chat_page, title="Chat", icon=":material/terminal:"),
        st.Page(render_ab_dashboard, title="A/B Dashboard", icon=":material/radar:"),
        st.Page(render_eval_results, title="Eval Results", icon=":material/troubleshoot:"),
    ]

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()

