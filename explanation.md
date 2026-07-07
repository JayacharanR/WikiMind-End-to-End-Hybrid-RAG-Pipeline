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
9. [Architecture Diagrams](#architecture-diagrams)
10. [Key Technical Decisions](#key-technical-decisions)

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
the same chunk, they get connected with an edge. This graph is serialized and
stored in Redis.

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

### 8. LLM Generation with Guardrails

The relevant chunks are formatted into a context block and passed to the LLM
with a generation prompt. NeMo Guardrails wraps this call to enforce:

- Input safety rails (block harmful/adversarial queries).
- Output safety rails (block toxic or personally identifiable content).
- Topic rails (keep responses grounded in the retrieved context).

### 9. Hallucination Check

After generation, a separate LLM call checks: "Is this answer supported by the
retrieved documents, or did the model make up facts?" If the answer is flagged
as hallucinated, the system retries generation with the same context (up to 2
retries). Each retry uses a different temperature to encourage a different
phrasing.

### 10. Answer Quality Check

If the answer passes the hallucination check, another LLM call evaluates:
"Does this answer actually address the user's question?" If the answer is
deemed off-topic or incomplete, the system loops all the way back to Step 2
(query expansion) and tries a different expansion strategy (up to 2 retries).

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
    hallucination_retries: Independent counter (max 2)
    answer_retries:       Independent counter (max 2)
}
```

The graph looks like this:

```
expand_query -> identify_articles -> [knowledge_graph?] -> retrieve
    -> grade_documents
        -> [irrelevant] -> generate_from_web -> END
        -> [relevant] -> generate -> check_hallucination
            -> [hallucinated, retries < 2] -> generate (retry)
            -> [grounded] -> check_answer_quality
                -> [not useful, retries < 2] -> expand_query (loop back)
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

3. The graph is serialized to JSON and stored in Redis for fast access.

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
`evaluation/results/`.

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
|  [Hallucination Check] ---fail (max 2)--> retry generation    |
|       |                                                       |
|       v (pass)                                                |
|  [Answer Quality Check] ---fail (max 2)--> loop to expansion  |
|       |                                                       |
|       v (pass)                                                |
|  [Cache Store + Return Answer via SSE]                        |
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
       |                                            (retry, <=2) |
       |                                                  |      |
       |                                          +-------v------v--+
       |                                          | check_answer    |
       |                                          +--+----------+---+
       |                                             |          |
       |                                        not_useful    useful
       |                                             |          |
       +---------(loop back, retries <= 2)-----------+         END
```

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
`OPENROUTER_API_KEY` environment variable.

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
python -m data_pipeline.graph_builder
```

### Infrastructure

All infrastructure runs in Docker:

```bash
make dev   # Starts Qdrant, Redis, Prometheus, Grafana, Backend, Frontend
```

Or individually:

```bash
docker compose up qdrant redis -d       # Just the databases
uvicorn backend.main:app --reload       # Backend
streamlit run frontend/app.py           # Frontend
```
