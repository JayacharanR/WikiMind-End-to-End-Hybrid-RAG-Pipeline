# WikiMind: Hybrid Agentic RAG Pipeline

WikiMind is a Tri-Modal Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, and accurately synthesize answers from the Wikipedia dataset. It is structured as a microservices architecture to ensure high availability and robust performance.

## Key Features

*   **Tri-Modal Retrieval Architecture**: Incorporates Dense Search (Semantic), Sparse Search (BM25), and Vectorless Tree Navigation (PageIndex).
*   **Semantic Caching**: Utilizes RedisVL to intercept queries and return cached responses at low latency, bypassing expensive LLM computation.
*   **Agentic Orchestration**: Deploys LangGraph for a Corrective RAG (CRAG) loop with web search fallbacks via Tavily.
*   **Continuous Synchronization**: Listens to Wikimedia EventStreams API to continuously synchronize the Qdrant vector database with live Wikipedia edits.
*   **Observability and Guardrails**: Monitored strictly via Langfuse, with interaction safety enforced by NeMo Guardrails.

## Tech Stack

*   **Backend**: Python, FastAPI, LangGraph, NeMo Guardrails
*   **Frontend**: Streamlit
*   **Data Pipeline**: aiohttp, LangChain, text-embedding-3-small
*   **Databases**: Qdrant (Vector & Sparse Storage), Redis (Semantic Caching via RedisVL), MongoDB/Local (Document Storage)
*   **Observability**: Langfuse
*   **Deployment**: Docker Compose

## Documentation Requirements
Per project guidelines, all documentation, code, and commit messages must maintain a strict professional and technical tone.

---
*Note: The project architecture has recently been revised to support containerized microservices. The codebase is currently undergoing transition.*
