"""WikiMind Streamlit Frontend.

Provides an interactive chat interface for the Tri-Brid Agentic RAG pipeline.
Includes sidebar toggles for configuring query expansion and retrieval strategies,
displays real-time SSE streaming from the FastAPI backend, and renders retrieved
sources and execution metadata in expandable sections.
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
    st.sidebar.title("⚙️ WikiMind Configuration")
    
    st.sidebar.subheader("Retrieval Architecture")
    st.sidebar.markdown("Configure the active Tri-Brid components.")
    
    use_page_index = st.sidebar.toggle(
        "Enable PageIndex (L3)", 
        value=True, 
        help="Use LLM structural navigation to extract full sections from Wikipedia."
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Query Expansion (L2)")
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
    
    st.sidebar.subheader("System Health")
    if st.sidebar.button("Check Backend Health"):
        try:
            res = requests.get(f"{API_URL}/health", timeout=5)
            if res.status_code == 200:
                data = res.json()
                status = data.get("status")
                st.sidebar.success(f"Status: {status.upper()}")
                for comp in data.get("components", []):
                    icon = "✅" if comp.get("healthy") else "❌"
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
        "page_index": use_page_index,
    }


def stream_chat_response(query: str, strategies: Dict[str, bool]):
    """Stream the response from the FastAPI SSE endpoint."""
    payload = {
        "query": query,
        "strategies": strategies
    }
    
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
                status_placeholder.info(f"🔄 Agent working: {status} ({node})")
                
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
                    with sources_placeholder.expander(f"📚 View {len(sources)} Retrieved Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**[{i+1}] {source.get('title')}** (Score: {source.get('score', 0):.2f})")
                            st.markdown(f"> {source.get('content')[:300]}...")
                            if source.get('url'):
                                st.markdown(f"[Read on Wikipedia]({source.get('url')})")
                            st.divider()
                            
                # Render metadata
                metadata = data.get("metadata", {})
                with st.expander("🛠️ Execution Metadata"):
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


def main():
    st.set_page_config(
        page_title="WikiMind | Tri-Brid Hybrid RAG",
        page_icon="🧠",
        layout="wide",
    )
    
    st.title("🧠 WikiMind RAG Pipeline")
    st.markdown("Ask complex questions. The agent will orchestrate Hybrid Search, RRF, and PageIndex structural extraction to answer them.")
    
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
                    with st.expander(f"📚 View {len(message['sources'])} Retrieved Sources"):
                        for i, source in enumerate(message['sources']):
                            st.markdown(f"**[{i+1}] {source.get('title')}** (Score: {source.get('score', 0):.2f})")
                            st.markdown(f"> {source.get('content')[:300]}...")
                            if source.get('url'):
                                st.markdown(f"[Read on Wikipedia]({source.get('url')})")
                            st.divider()
                if message.get("metadata"):
                    with st.expander("🛠️ Execution Metadata"):
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


if __name__ == "__main__":
    main()
