# WikiMind -- Complete Architecture and Pipeline Explanation

This document explains every layer of WikiMind in plain language: what it does,
how it works, and why each design choice was made. It is written for someone
encountering the project for the first time.

---

## Table of Contents

1. [What is WikiMind?](#what-is-wikimind)
2. [The Core Problem](#the-core-problem)
3. [How the Data Gets Into the System](#how-the-data-gets-into-the-system)
4. [How a Query Flows Through the System](#how-a-query-flows-through-the-system)
5. [Detailed Component Breakdown](#detailed-component-breakdown)
6. [Knowledge Graph Layer](#knowledge-graph-layer)
7. [Temporal Versioning and Time-Travel](#temporal-versioning-and-time-travel)
8. [Evaluation Harness](#evaluation-harness)
9. [Observability (Langfuse)](#observability-langfuse)
10. [Custom Observability Dashboard](#custom-observability-dashboard)
11. [Architecture Diagrams](#architecture-diagrams)
12. [Benchmark Results](#benchmark-results)
13. [Key Technical Decisions](#key-technical-decisions)

---

## What is WikiMind?

WikiMind is a question-answering system backed by the English Wikipedia. You ask
it a question in natural language, and it:

1. Figures out which Wikipedia articles are relevant.
2. Extracts the precise text passages that contain the answer.
3. Generates a grounded, verified answer using an LLM.
4. Checks its own answer for hallucinations and quality before returning it.

It is NOT a chatbot wrapper around ChatGPT. It is a full retrieval-augmented
generation (RAG) pipeline where the LLM never makes up facts -- it only
synthesizes answers from text chunks that were actually retrieved from Wikipedia.

---

## The Core Problem

Wikipedia has over 6.8 million English articles. When you ingest and chunk those
articles into a vector database, you get tens of millions of text fragments.

If a user asks "What is the population of Tokyo?", a naive vector search across
all those chunks returns results from Tokyo, Osaka, Japan demographics, List of
largest cities, Metropolitan areas, and dozens of tangentially related articles.
The LLM then receives a context window full of noisy, loosely related text and
either hallucinates, loops, or gives a vague answer.

WikiMind solves this with a **two-stage retrieval architecture**: first identify
the right articles, then search only within those articles for the exact answer.

---

## How the Data Gets Into the System

This is the most important thing to understand: WikiMind does NOT query
Wikipedia live when you ask a question. The entire Wikipedia corpus (or a subset
of it) is pre-processed and stored locally. Here is exactly what happens:

### Step 1: Download the Wikipedia Dataset

The ingestion script (`data_pipeline/ingest.py`) streams the complete English
Wikipedia dataset from HuggingFace (`wikimedia/wikipedia`, the November 2023
snapshot). This dataset contains the full text of every English Wikipedia
article -- approximately 6.8 million articles totaling around 21 GB of raw
text.

The script uses HuggingFace's streaming mode, which means it downloads articles
one at a time without requiring the full 21 GB in memory or on disk upfront.

### Step 2: Chunk Each Article

Each article's text is split into overlapping chunks of 512 characters with
64-character overlap, using LangChain's RecursiveCharacterTextSplitter. A single
long Wikipedia article (say, 20,000 characters) becomes roughly 40 chunks. The
chunking preserves paragraph boundaries when possible.

### Step 3: Embed Each Chunk (Dual Embeddings)

Every chunk gets two different embeddings generated locally (no API calls):

1. **Dense embedding** -- Uses `BAAI/bge-small-en-v1.5` via FastEmbed. This is
   a 384-dimensional vector that captures the semantic meaning of the text.
   "Capital of France" and "Paris is the capital city" would have high cosine
   similarity even though they share few words.

2. **Sparse embedding** -- Uses `Qdrant/bm25` via FastEmbed. This is a
   sparse vector (like traditional BM25/TF-IDF) that captures exact keyword
   matches. "Population of Tokyo" scores high on chunks containing those exact
   words.

Both embeddings are generated entirely on your local machine using CPU-based
models. No OpenAI or cloud API is called for embedding.

### Step 4: Extract Named Entities

Each chunk is also run through spaCy NER (Named Entity Recognition) to extract
entities like person names, organizations, locations, and events. These entities
are stored in the chunk's metadata and later used to build the knowledge graph.

### Step 5: Store in Qdrant (Two Collections)

The data is stored in Qdrant (a vector database running locally in Docker) in
two separate collections:

1. **`wikimind_hybrid`** -- Contains all the individual chunks with their dense
   and sparse embeddings, plus metadata (title, URL, chunk index, entities,
   revision ID, ingested timestamp, is_current flag). This is where the actual
   retrieval happens.

2. **`wikimind_articles`** -- Contains one entry per article (not per chunk).
   Each entry stores a single dense embedding of the article's title +
   first two paragraphs. This is used for Stage 1 article discovery.

### Step 6: Build the Knowledge Graph (Optional)

A separate script (`data_pipeline/graph_builder.py`) scrolls through all
chunks in Qdrant, reads the pre-extracted entities, and builds a NetworkX
co-occurrence graph. If "Albert Einstein" and "Theory of Relativity" appear in
the same chunk, they get connected with an edge.

The graph uses **dual persistence** with automatic fallback:

1. **Primary**: Serialized to a local JSON file at `data/knowledge_graph.json`.
2. **Secondary**: Saved to Redis for fast runtime access (if available).

At load time, the system tries Redis first, then falls back to the local JSON
file. This means the knowledge graph works without Redis running, which is
important for local development and environments where Docker is unavailable.

### What You Actually Need to Run

By default, the ingestion script processes 1,000 articles:

```bash
python -m data_pipeline.ingest --max 1000
```

For the full Wikipedia, set max to 0 (unlimited):

```bash
python -m data_pipeline.ingest --max 0
```

Processing the full 6.8M articles takes significant time and disk space
(approximately 50-100 GB of Qdrant storage for all embeddings). For development
and demonstration purposes, 1,000-10,000 articles covers a broad enough set of
topics to answer most common factual questions.

---

## How a Query Flows Through the System

When a user types a question into the Streamlit UI, here is the exact sequence
of operations, step by step:

### 1. Cache Check (L1 + L2)

Before any computation, the system checks its dual-layer cache:

- **L1 (Exact Match)**: Is this exact query string already cached in Redis?
  If yes, return the cached answer immediately. Latency: ~1ms.

- **L2 (Semantic Similarity)**: Is there a semantically similar query already
  cached? The query is embedded and compared against cached query embeddings
  using cosine similarity. If similarity exceeds 0.92, the cached answer is
  returned. This means "What is Tokyo's population?" hits the cache for
  "Population of Tokyo?" even though the strings are different.

If neither cache hits, proceed to the pipeline.

### 2. Query Expansion (Optional)

If the user enabled any expansion strategies in the sidebar, the system
generates alternative versions of the query to improve retrieval coverage:

- **Multi-Query**: Generates 3 semantically diverse reformulations.
  "Population of Tokyo" might become "How many people live in Tokyo?",
  "Tokyo metropolitan area residents", "Japan capital city population count".

- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer
  paragraph and uses its embedding for search. The LLM writes something like
  "The population of Tokyo is approximately 14 million as of the latest census"
  and this fake-but-plausible text is used as the search query.

- **Step-Back Abstraction**: Generates a broader question.
  "What year did Einstein publish the theory of relativity?" becomes
  "What were Einstein's major scientific contributions?".

- **Decomposition**: Breaks multi-part questions into sub-questions.
  "Compare the populations of Tokyo and New York" becomes
  "What is the population of Tokyo?" + "What is the population of New York?".

### 3. Article Discovery (Stage 1)

The expanded queries are searched against the `wikimind_articles` collection.
This collection contains one dense embedding per article (title + first
paragraphs). The search returns the top 3 most relevant article titles.

For "What is the population of Tokyo?", Stage 1 might return:
- Tokyo
- Demographics of Japan
- Greater Tokyo Area

This narrows millions of chunks down to a few hundred (the chunks belonging to
these 3 articles).

### 4. Knowledge Graph Traversal (Optional)

If the `knowledge_graph` strategy is enabled, the system also:

1. Runs spaCy NER on the query to extract entities ("Tokyo").
2. Looks up "Tokyo" in the co-occurrence knowledge graph stored in Redis.
3. Traverses up to 2 hops to find related entities and their source articles.
4. Merges any newly discovered article titles into the target list.

This helps with multi-hop questions. "Who was the first emperor of the dynasty
that built the Great Wall?" would discover the "Great Wall of China" article,
traverse to "Qin dynasty", and then to "Qin Shi Huang".

### 5. Article-Scoped Hybrid Search (Stage 2)

Now the system searches the `wikimind_hybrid` collection, but ONLY within the
chunks belonging to the identified articles. This is done using Qdrant's
payload filter on the `title` field.

The search runs two parallel tracks:

- **Dense search**: Computes cosine similarity between the query embedding and
  all chunk embeddings within the target articles.
- **Sparse search**: Computes BM25 keyword match scores for the same chunks.

Both result sets are fused using **Reciprocal Rank Fusion (RRF)**, which
combines rankings without needing to normalize scores across different metrics.
If a chunk ranks #1 in dense search and #3 in sparse search, RRF assigns it a
combined score that reflects both signals.

### 6. Cross-Encoder Reranking

The top RRF candidates (typically 20) are passed through a FlashRank
cross-encoder reranker. Unlike the embedding models which encode query and
document separately, the cross-encoder processes the query and document together
as a single input, producing a much more accurate relevance score.

The top 5 chunks after reranking become the generation context.

### 7. Document Grading

A single batched LLM call evaluates all 5 chunks at once, asking: "Which of
these documents are relevant to the query?" The LLM returns a comma-separated
list of relevant document indices (e.g., "1,3,5"). This replaces the old design
of making 5 separate LLM calls, reducing latency by ~80%.

If no documents pass grading, the pipeline falls back to generating from
whatever web snippets are available or returns a "no relevant information found"
response.

### 8. LLM Generation with Guardrails + Inline Citations

The relevant chunks are formatted with **numbered citation labels** (`[1]`,
`[2]`, etc.) and passed to the LLM with a generation prompt that requires
inline citations after each factual claim.

Generation uses a **guardrails-first** approach:

1. **Try NeMo Guardrails** via `safe_generate()` — routes the prompt through
   jailbreak detection, topic filtering, and output safety checks.
2. **Fallback to direct LLM** — if guardrails aren't initialized (missing
   config, NeMo not installed, or local LLM not running), the system
   transparently falls back to a direct LLM call with the same citation
   prompt. No errors are raised.

The guardrails enforce:
- Input safety rails (block harmful/adversarial queries).
- Output safety rails (block toxic or personally identifiable content).
- Topic rails (keep responses grounded in the retrieved context).

A `guardrails_applied` flag in the response metadata indicates which path
was used.

### 9. Hallucination Check + Citation Verification

After generation, a **two-phase** check runs:

**Phase 1 — LLM grounding check** (relaxed criteria for local LLMs):

- **Paraphrasing allowed**: The answer doesn't need to quote the context
  verbatim. Reasonable paraphrasing and summarization are accepted.
- **"Cannot answer" shortcut**: If the generation explicitly says it cannot
  answer, the hallucination check is skipped entirely (it's grounded by
  definition).
- **Tolerant parsing**: The checker extracts the first word of the LLM response
  for robust yes/no detection.
- **Single retry**: If flagged as hallucinated, the system retries once.

**Phase 2 — Citation verification** (provenance proof):

- Parses all `[N]` inline citations from the generation text.
- For each cited sentence, extracts key terms (words > 3 chars, excluding
  stopwords) and checks if ≥40% of them appear in the referenced chunk.
- Computes a **`provenance_score`** (0.0–1.0): the fraction of cited claims
  whose key terms were verified against the source chunk.
- A score of 0.0 means no citations were found (the LLM didn't cite).
- A score of 1.0 means every citation checked out.

This is the project's **provenance proof mechanism** — it provides a
quantitative measure of how well the LLM's claims are traceable to specific
retrieved chunks.

### 10. Answer Quality Check + Attribution Detection

If the answer passes the hallucination check, two things happen:

**Quality check** (with heuristic bypass):
- If the generation is non-trivial (≥10 words) and doesn't refuse to answer,
  the LLM quality check is skipped entirely.
- Short or ambiguous answers go through the LLM quality checker.

**Attribution detection** (context-ablation):
- A separate LLM call asks: _"Can you answer this question WITHOUT any external
  context, from your training data alone?"_
- If the LLM says "yes" → the answer may come from **parametric knowledge**
  rather than RAG → `attribution = "parametric_risk"`
- If the LLM says "no" → the answer likely required the RAG context →
  `attribution = "rag_grounded"`

This is a heuristic, not a guarantee — but it provides a signal that helps
evaluate whether the RAG pipeline is actually contributing to the answer or
whether the LLM could have answered alone.

These tuning changes reduced the **mean step count from 16 to 7** and **P50
latency from 8.9s to 4.7s** without sacrificing accuracy.

### 11. Cache and Return

Once the answer passes both checks, it is stored in the Redis cache (both L1
and L2) and returned to the user via SSE (Server-Sent Events) streaming. The
Streamlit UI displays the answer along with the retrieved source documents and
execution metadata.

---

## Detailed Component Breakdown

### LangGraph State Machine

The entire query flow is implemented as a LangGraph state machine with 10 nodes
and conditional edges. The state dictionary carries all intermediate results
between nodes:

```
State = {
    query:                The original user question
    expanded_queries:     List of reformulated queries
    target_articles:      Article titles from Stage 1
    documents:            Retrieved chunks from Stage 2
    generation:           The LLM's answer text
    retrieval_grade:      "relevant" or "irrelevant"
    hallucination_grade:  "grounded" or "hallucinated"
    answer_grade:         "useful" or "not_useful"
    steps:                Total node transitions (hard budget: 15)
    hallucination_retries: Independent counter (max 1)
    answer_retries:       Independent counter (max 1)
    citation_map:         {1: chunk_dict, 2: chunk_dict, ...}
    provenance_score:     0.0–1.0, fraction of cited claims verified
    attribution:          "rag_grounded", "parametric_risk", or "unknown"
    guardrails_applied:   Whether NeMo Guardrails were used for generation
}
```

The graph looks like this:

```
expand_query -> identify_articles -> [knowledge_graph?] -> retrieve
    -> grade_documents
        -> [irrelevant] -> generate_from_web -> END
        -> [relevant] -> generate -> check_hallucination
            -> [hallucinated, retries < 1] -> generate (retry)
            -> [grounded] -> check_answer_quality
                -> [not useful, retries < 1] -> expand_query (loop back)
                -> [useful or budget exhausted] -> END
```

### Dual-Layer Semantic Cache

The cache has two tiers stored in Redis:

- **L1**: Hash-based exact match. O(1) lookup. Catches repeated identical
  queries.
- **L2**: Vector similarity search. The query embedding is compared against
  a RedisVL vector index of previously cached query embeddings. Catches
  semantically equivalent reformulations of the same question.

Cache entries include a TTL (time-to-live) that distinguishes between static
facts (24-hour TTL) and dynamic facts (1-hour TTL based on the freshness of
the underlying article).

### Self-Healing Knowledge Base

Three components keep the data fresh:

1. **Wiki Updater** (`data_pipeline/wiki_updater.py`): Connects to Wikimedia
   EventStreams, a live SSE feed of every edit to every Wikipedia article in
   real-time. When an article is edited, the updater fetches the new text,
   re-chunks, re-embeds, and upserts to Qdrant with version tracking.

2. **State Reconciler** (`data_pipeline/reconciler.py`): Runs on a configurable
   schedule (default: every 6 hours). Samples 100 random articles from Qdrant,
   fetches their current revision ID from the MediaWiki API, and compares it to
   the stored `revision_id`. If drift is detected, triggers re-ingestion.

3. **Version-Aware Upserts**: When an article is updated, all existing chunks
   for that article are marked `is_current=false` before the new chunks are
   inserted with `is_current=true`. Old versions are preserved, enabling
   time-travel queries.

---

## Knowledge Graph Layer

The knowledge graph adds an entity-relationship dimension to retrieval. Here is
how it works:

### Building the Graph

1. During ingestion, every chunk is processed with spaCy NER to extract named
   entities (PERSON, ORG, GPE, LOC, EVENT, WORK_OF_ART).

2. The graph builder script scrolls through all chunks and creates a NetworkX
   directed graph where:
   - Each **node** is a unique entity (e.g., "Albert Einstein", "Germany").
   - Each **edge** connects two entities that co-occur in the same chunk, with
     metadata recording the source article title and chunk index.
   - Edge **weights** increase when two entities co-occur in multiple chunks.

3. The graph is serialized to a local JSON file (`data/knowledge_graph.json`)
   and optionally mirrored to Redis for fast runtime access.

### Using the Graph at Query Time

When a user asks a multi-hop question like "What university did the inventor
of the telephone attend?":

1. spaCy extracts "telephone" from the query.
2. The graph is searched for nodes matching "telephone".
3. BFS traversal (up to 2 hops) discovers connected entities like
   "Alexander Graham Bell" and source articles like "Invention of the
   telephone" and "Alexander Graham Bell".
4. These article titles are merged with the Stage 1 results, broadening the
   retrieval scope to include articles the vector search alone might miss.

---

## Temporal Versioning and Time-Travel

Every chunk in Qdrant stores three versioning fields:

- `revision_id`: The Wikipedia revision ID at the time of ingestion.
- `ingested_at`: The UTC timestamp when the chunk was embedded and stored.
- `is_current`: Boolean flag indicating whether this is the latest version.

### Default Behavior

Normal queries automatically filter to `is_current=true`, so you always get
the latest version of every article.

### Time-Travel Mode

When the user enables Time-Travel in the sidebar and selects a date, the query
filter switches from `is_current=true` to `ingested_at <= selected_date`. This
retrieves the version of the article as it existed on that date.

Use case: "What did the Wikipedia article about COVID-19 say in March 2020?"

---

## Evaluation Harness

The evaluation system benchmarks the pipeline against standardized Q&A datasets.

### Datasets

- **Natural Questions (NQ)**: Real Google search queries with short answers
  extracted from Wikipedia by human annotators.
- **TriviaQA**: Trivia questions with evidence documents.

Both are loaded from HuggingFace with local disk caching.

### Metrics

For each query, the harness measures:

- **Recall@K**: Did any of the top-K retrieved chunks contain the gold answer?
  Uses normalized substring matching (lowercase, strip articles, remove
  punctuation).
- **MRR (Mean Reciprocal Rank)**: At what position was the first relevant chunk?
  1/rank of the first chunk containing the answer.
- **Answer Accuracy**: Does the LLM's generated answer contain the gold answer?
  Normalized substring match.
- **Latency Percentiles**: P50, P95, P99 of end-to-end query time.
- **Step Count**: How many LangGraph state transitions did the query require?

### Running an Evaluation

```bash
# Baseline (no expansion strategies)
python -m evaluation.harness --dataset nq --subset 50

# With multi-query + step-back expansion
python -m evaluation.harness --dataset triviaqa --subset 100 \
    --config evaluation/configs/with_expansion.json
```

Results are saved as both a markdown report and a JSON dump in
`evaluation/results/`. If Langfuse is running, aggregate scores (recall, MRR,
accuracy, latency, step count) are also pushed as Langfuse evaluation traces
for dashboard tracking across benchmark runs.

---

## Observability (Langfuse)

WikiMind uses a **Langfuse-first** observability strategy, purpose-built for
LLM/RAG pipeline monitoring. This replaces the original Prometheus + Grafana
stack, which was designed for generic infrastructure metrics and added
maintenance overhead without capturing what matters for a RAG pipeline.

### Why Langfuse Instead of Prometheus + Grafana?

| Concern | Langfuse | Prometheus + Grafana |
|---------|----------|---------------------|
| **LLM trace waterfalls** | ✅ Built-in | ❌ Not possible |
| **Token cost tracking** | ✅ Per-trace | ❌ Not possible |
| **Evaluation scores** | ✅ Native integration | ❌ Requires custom exporters |
| **Prompt versioning** | ✅ Built-in | ❌ Not possible |
| **HTTP request rates** | ❌ Use Prometheus `/metrics` | ✅ Built-in |
| **Dashboards to maintain** | 1 (auto-generated) | 2+ (custom build required) |

For a RAG portfolio project, Langfuse captures what interviewers want to see:
the full trace of how a query flows through expand → retrieve → grade →
generate → hallucination check, with per-node latency and token counts.

### Self-Hosted Deployment

Langfuse runs via `docker-compose.yml` alongside the existing services:

- **`langfuse`**: The Langfuse server (langfuse/langfuse:2), accessible at
  `http://localhost:3000`.
- **`langfuse-db`**: PostgreSQL 15 Alpine for Langfuse's metadata storage.

The docker-compose auto-seeds a project with pre-configured API keys via
`LANGFUSE_INIT_*` environment variables, so no manual setup is needed.

Login: `admin@wikimind.local` / `wikimind-admin`

### What Gets Traced

Every query through the `/chat` or `/chat/compare` endpoint creates a Langfuse
trace with:

```
Trace: wikimind_query
├─ expand_query        (12ms, 0 tokens)
├─ identify_articles   (268ms, 0 tokens)
├─ retrieve            (830ms, 0 tokens)
├─ grade_documents     (1,427ms, 142 tokens)
├─ generate            (937ms, 312 tokens)
├─ check_hallucination (1,113ms, 48 tokens)
└─ check_answer_quality (0ms, heuristic bypass)

Total: 4.7s | 502 tokens | Score: grounded ✓
```

The evaluation harness also pushes benchmark aggregate scores to Langfuse,
enabling comparison across runs.

### Prometheus (Retained, No Dashboards)

The `prometheus-fastapi-instrumentator` is kept (2 lines of code in `main.py`).
It exposes standard HTTP metrics at `/metrics` for production environments that
require Grafana/Datadog integration. No Grafana dashboards are bundled since
Langfuse handles all RAG-specific observability.

---

## Custom Observability Dashboard

While Langfuse provides deep LLM trace analysis, WikiMind also includes a
custom-built observability dashboard served directly from the FastAPI backend.
This provides an at-a-glance operational view without requiring Langfuse access.

### Architecture

The dashboard is a multi-file vanilla HTML/CSS/JS application (no build tools)
served as static files at `/dashboard` via FastAPI's `StaticFiles` mount:

```
dashboard/
├── index.html          # Entry point with 5-tab layout
├── css/
│   └── dashboard.css   # Design system (dark/light mode, glassmorphism)
└── js/
    ├── app.js          # Tab routing, state management, auto-refresh
    ├── api.js          # Fetch wrappers for backend endpoints
    ├── charts.js       # Chart.js factories (line, doughnut, bar)
    ├── kpi.js          # KPI card rendering (6 metrics)
    ├── traces.js       # Trace table with expandable detail rows
    ├── guardrails.js   # Guardrails monitoring panel
    └── evaluation.js   # Benchmark results viewer
```

### Backend Data Layer

The dashboard reads from three backend API endpoints:

- **`GET /api/metrics`** — Aggregated KPIs computed from an in-memory ring
  buffer of the last 500 query traces. Returns latency percentiles (P50, P95),
  attribution breakdown, guardrails stats, grade counts, and cache hit rate.

- **`GET /api/traces`** — Raw trace objects for the trace explorer. Each trace
  contains the query, generation, latency, step count, provenance score,
  attribution, grade results, expanded queries, and citation map.

- **`GET /api/eval-results`** — Evaluation benchmark results read from
  `evaluation/results/*.json` on disk.

### Dashboard Tabs

1. **Overview** — 6 KPI cards (total queries, P50 latency, avg steps,
   provenance score, RAG grounded %, guardrails applied). Latency timeline
   chart (Chart.js line), attribution donut chart, recent queries table.
   Auto-refreshes every 30 seconds.

2. **Traces** — Full query trace table with expandable rows. Each expanded row
   shows: query text, generation text, metadata grid (latency, steps, documents,
   provenance), attribution and grade badges, expanded queries list, and
   citation map. Color-coded latency (green < 3s, amber < 8s, red > 8s).

3. **Guardrails** — Applied vs bypassed donut chart, quality grade breakdown
   (grounded/hallucinated/useful/not useful as horizontal bars), and a safety
   event log table listing guardrail-related or quality-flagged queries.

4. **Evaluation** — Benchmark results viewer. Shows aggregate metrics
   (Recall@5, MRR, Accuracy, Latency P50, Mean Steps) and a per-query table
   for each evaluation run.

5. **System** — Component health from the `/health` endpoint. Displays
   Qdrant, Redis, LLM, and Langfuse status with latency indicators.

### Design System

- **Dark mode default** with light mode toggle (matches the existing Streamlit
  chat UI aesthetic).
- **Typography**: Inter (body) + JetBrains Mono (metrics/code) via Google Fonts.
- **Visualization**: Chart.js 4.x for all charts.
- **Icons**: Lucide icon library.
- **No build tools**: Pure ES modules, served as static files.

### Why a Custom Dashboard Instead of Grafana?

Grafana is designed for infrastructure metrics (CPU, memory, request rates).
A custom dashboard can display RAG-specific information that Grafana can't:
provenance scores, attribution breakdown, citation maps, expanded queries,
and quality grades. For a portfolio project, a purpose-built dashboard
demonstrates architectural understanding far better than generic charts.

---

## Architecture Diagrams

### End-to-End Pipeline

```
+==============================================================+
|                      DATA PIPELINE                            |
|                                                               |
|  HuggingFace Wikipedia  ---stream--->  Batch Ingestor         |
|  (6.8M articles)                       |                      |
|                                        +-> Chunk (512 chars)  |
|                                        +-> Dense Embed (384d) |
|                                        +-> Sparse Embed (BM25)|
|                                        +-> Entity Extract     |
|                                        +-> Upsert to Qdrant   |
|                                        +-> Article-level Index|
|                                                               |
|  Wikimedia EventStreams ---live SSE--> Wiki Updater            |
|  (real-time edits)                    +-> Version-aware upsert|
|                                       +-> Archive old chunks  |
|                                                               |
|  State Reconciler ---periodic--> Compare stored vs live revid |
|                                  +-> Re-ingest stale articles |
+==============================================================+

+==============================================================+
|                     QUERY PIPELINE                            |
|                                                               |
|  User Query                                                   |
|       |                                                       |
|       v                                                       |
|  [L1/L2 Cache Check] ---hit---> Return cached answer          |
|       |                                                       |
|       v (miss)                                                |
|  [Query Expansion] (optional: Multi-Query/HyDE/StepBack)     |
|       |                                                       |
|       v                                                       |
|  [Stage 1: Article Discovery]                                 |
|  Search wikimind_articles -> top 3 article titles             |
|       |                                                       |
|       v (optional)                                            |
|  [Knowledge Graph Traversal]                                  |
|  Extract entities -> BFS 2-hop -> merge article titles        |
|       |                                                       |
|       v                                                       |
|  [Stage 2: Article-Scoped Hybrid Search]                      |
|  Dense + Sparse + RRF (filtered to target articles)           |
|       |                                                       |
|       v                                                       |
|  [Cross-Encoder Reranking] -> top 5 chunks                    |
|       |                                                       |
|       v                                                       |
|  [Batched Document Grading] (1 LLM call for all 5)            |
|       |                                                       |
|       v                                                       |
|  [LLM Generation + NeMo Guardrails]                           |
|       |                                                       |
|       v                                                       |
|  [Hallucination Check] ---fail (max 1)--> retry generation    |
|       |                 (relaxed grounding + tolerant parsing) |
|       v (pass)                                                |
|  [Answer Quality Check] ---fail (max 1)--> loop to expansion  |
|       |                 (heuristic bypass for non-trivial)     |
|       v (pass)                                                |
|  [Cache Store + Return Answer via SSE]                        |
|  [Langfuse Trace Recorded]                                    |
+==============================================================+
```

### LangGraph State Machine Nodes

```
+----------------+     +--------------------+     +---------------+
| expand_query   |---->| identify_articles  |---->| graph_search  |
+----------------+     +--------------------+     | (conditional) |
       ^                                          +-------+-------+
       |                                                  |
       |                                          +-------v-------+
       |                                          |   retrieve    |
       |                                          +-------+-------+
       |                                                  |
       |                                          +-------v-------+
       |                                          | grade_docs    |
       |                                          +---+-------+---+
       |                                              |       |
       |                                     irrelevant     relevant
       |                                              |       |
       |                                    +---------v-+  +--v----------+
       |                                    | gen_web   |  |  generate   |
       |                                    +-----+-----+  +--+----------+
       |                                          |           |
       |                                         END   +------v------+
       |                                               | check_hall  |
       |                                               +--+------+--+
       |                                                  |      |
       |                                           hallucinated  grounded
       |                                                  |      |
       |                                            (retry, <=1) |
       |                                                  |      |
       |                                          +-------v------v--+
       |                                          | check_answer    |
       |                                          | (heuristic      |
       |                                          |  bypass if >=10 |
       |                                          |  words)         |
       |                                          +--+----------+---+
       |                                             |          |
       |                                        not_useful    useful
       |                                             |          |
       |                                             |         END
       +---------(loop back, retries <= 1)-----------+   |
```

---

## Benchmark Results

The evaluation harness has been executed against a 10-question curated Wikipedia
Q&A dataset. Two configurations were tested: **baseline** (no expansion) and
**with_expansion** (multi-query + step-back).

### Baseline vs Expansion (Before Hallucination Tuning)

| Metric | Baseline | With Expansion | Delta |
|--------|----------|---------------|-------|
| Mean Recall@5 | 0.9000 | 0.9000 | — |
| Mean MRR | 0.6833 | **0.7250** | **+6.1%** |
| Mean Answer Accuracy | 0.8000 | 0.8000 | — |
| Latency P50 | **8.94s** | 17.92s | +100% |
| Mean Step Count | 16.0 | 16.0 | — |

Expansion improved MRR by 6.1% (better ranking) at the cost of doubled latency.
Both configs exhausted the 16-step budget on every query due to the overly
aggressive hallucination checker.

### Impact of Hallucination Checker Tuning

| Metric | Before Tuning | After Tuning | Improvement |
|--------|:---:|:---:|:---:|
| Mean Recall@5 | 0.90 | 0.90 | — |
| Mean Answer Accuracy | 0.80 | 0.80 | — |
| **Latency P50** | 8.94s | **4.72s** | **↓ 47%** |
| **Mean Step Count** | 16.0 | **7.0** | **↓ 56%** |

Quality stayed identical, but latency nearly halved and step count dropped from
16 to 7. The three changes that had the biggest impact:

1. **Heuristic answer quality bypass** — saves 1 LLM call per query.
2. **"Cannot answer" shortcut** — no wasted retries on refusal responses.
3. **Retry limit reduction (2→1)** — worst case is now 7 steps, not 16.

### Knowledge Graph Statistics

Built from 5,000 Qdrant chunks in 167.7 seconds:

| Metric | Value |
|--------|-------|
| Nodes (entities) | 18,010 |
| Edges (co-occurrences) | 71,491 |
| Top entity by degree | "the United States" (425 connections) |
| Storage | `data/knowledge_graph.json` |

---

## Key Technical Decisions

### Why Two-Stage Retrieval Instead of Full-Corpus Search?

With millions of chunks, a single vector search returns cross-article noise.
Stage 1 narrows the scope to 2-3 articles using a lightweight article-level
index. Stage 2 then performs precise hybrid search within only those articles.
This reduces noise dramatically and keeps the reranker working with high-quality
candidates.

The original design used Tavily (a web search API) for Stage 1. This was
replaced with a fully local article index (`wikimind_articles` Qdrant
collection) to eliminate the external dependency, enable offline operation,
and remove per-query API costs.

### Why Dense + Sparse Instead of Dense Only?

Dense embeddings capture semantic meaning but miss exact keyword matches.
Sparse (BM25) embeddings capture exact keywords but miss semantic similarity.
Combining both with Reciprocal Rank Fusion gives the best of both worlds:
a chunk that is both semantically relevant AND contains the exact keywords
scores highest.

### Why Cross-Encoder Reranking?

Bi-encoder embeddings (like the dense embeddings) encode query and document
independently. This is fast but loses fine-grained interaction between query
and document tokens. A cross-encoder processes them jointly, attending to
every word in both, producing much more accurate relevance scores at the cost
of higher latency. We apply it only to the top 20 candidates (already filtered
by RRF) to keep latency acceptable.

### Why Batched Document Grading?

The original design made one LLM call per document to check relevance. With 5
documents, that was 5 serial LLM calls adding ~3 seconds of latency. The
batched approach concatenates all documents with numbered indices and asks the
LLM to return a comma-separated list of relevant document numbers in a single
call. Same quality, ~80% less latency.

### Why Separate Retry Counters?

The hallucination check and answer quality check serve different purposes. A
hallucination retry regenerates from the same context (the answer was factually
wrong). An answer quality retry re-expands and re-retrieves (the answer was
factually correct but did not address the question). Using a shared counter
caused one check type to consume the other's retry budget.

### Why Reduced Retry Limits (2 → 1)?

Benchmarking revealed that with a local 8B-parameter LLM, the hallucination and
answer quality checkers were producing false negatives on most queries, causing
the pipeline to exhaust its 16-step budget on every single query. Reducing from
2 retries to 1, combined with relaxed grounding criteria and heuristic bypasses,
cut step count from 16 to 7 without any loss in answer accuracy.

### Why Langfuse Instead of Prometheus + Grafana?

Prometheus + Grafana is designed for infrastructure metrics (request rates,
error counts, CPU usage). Langfuse is purpose-built for LLM observability
(trace waterfalls, token costs, evaluation scores, prompt versioning).
Maintaining three dashboards for what one tool does better was unnecessary
complexity. The Prometheus `/metrics` endpoint is retained (2 lines of code)
for production environments that need infrastructure monitoring.

### Why Citation-Based Provenance Instead of Chunk ID Injection?

Alternative approaches like injecting chunk UUIDs into the prompt or
post-processing with embedding similarity are either fragile (UUIDs get
hallucinated) or expensive (recomputing embeddings per sentence). The inline
`[N]` citation approach works because:

1. **LLMs are trained on academic text** — they understand `[N]` citation
   conventions and follow the instruction reliably.
2. **Verification is cheap** — parsing `[N]` with regex and checking key term
   overlap against chunks requires zero LLM calls.
3. **User-visible** — citations appear in the response, so users can click
   through to the source chunk. This doubles as a UX feature.

The `provenance_score` (0.0–1.0) provides a quantitative trust signal per
response, not just per-query accuracy.

### Why Context-Ablation for Attribution?

The fundamental question is: "Did the RAG pipeline contribute to this answer,
or did the LLM already know the answer?" There are three approaches:

1. **Use a weak model** — deliberately use an LLM that can't answer without
   context. But this sacrifices generation quality.
2. **Constrained decoding** — force the LLM to only output tokens from the
   context. But this breaks fluency and summarization.
3. **Context-ablation** — ask the LLM if it *could* answer without context.

Option 3 adds ~1 LLM call per query but doesn't sacrifice quality. The
`attribution` field is informational (`"rag_grounded"` vs `"parametric_risk"`)
rather than a blocker — it flags answers that might need human review.

### Why Store is_current Instead of Deleting Old Chunks?

Deleting old chunks is irreversible. By marking them `is_current=false`, we
preserve the full history and enable time-travel queries. The `is_current`
payload index ensures that default queries still only hit the latest version
with no performance penalty.

---

## Running WikiMind Fully Local

WikiMind runs entirely on your local machine. No cloud LLM embedding APIs are
needed for retrieval -- FastEmbed generates all embeddings locally on CPU.

The only external API requirement is an LLM for the generation, grading, and
expansion steps. This uses the OpenAI-compatible API configured via the
`OPENROUTER_API_KEY` environment variable (or a local Llama server on port
8080).

### Ingestion Commands

```bash
# Ingest 1,000 articles (fast, for development)
python -m data_pipeline.ingest --max 1000

# Ingest 10,000 articles (broader coverage, ~30 min)
python -m data_pipeline.ingest --max 10000

# Ingest ALL of Wikipedia (6.8M articles, takes hours, ~50-100GB storage)
python -m data_pipeline.ingest --max 0

# Rebuild only the article-level index (no chunk re-embedding)
python -m data_pipeline.ingest --articles-only

# Build the knowledge graph from existing chunks
python -m data_pipeline.graph_builder --scroll-limit 5000
```

### Infrastructure

All infrastructure runs in Docker:

```bash
docker compose up -d  # Starts Qdrant, Redis, Langfuse, Backend, Frontend
```

Or individually:

```bash
docker compose up qdrant redis langfuse langfuse-db -d  # Databases + observability
uvicorn backend.main:app --reload                       # Backend
streamlit run frontend/app.py                           # Frontend
```

### Access Points

| Service | URL | Notes |
|---------|-----|-------|
| **Frontend** | http://localhost:8501 | Streamlit chat UI |
| **Backend API** | http://localhost:8000 | FastAPI + Swagger at `/docs` |
| **Observability Dashboard** | http://localhost:8000/dashboard/ | Custom 5-tab dashboard |
| **Langfuse** | http://localhost:3000 | LLM traces and eval dashboard |
| **Qdrant** | http://localhost:6333 | Vector DB dashboard |
| **Prometheus Metrics** | http://localhost:8000/metrics | Raw endpoint, no dashboard |
