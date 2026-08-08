# WikiMind — Final Architecture Explanation

This document describes the final implemented state of WikiMind. WikiMind is an
English Wikipedia question-answering system built as a complete retrieval-
augmented generation pipeline.

The system has one source-of-truth policy: for every article, Qdrant should
contain the latest successfully synchronized article state. Historical article
contents are deliberately not stored.

## 1. System objective

LLMs can produce fluent but unsupported answers. WikiMind reduces that risk by:

1. discovering relevant Wikipedia articles before chunk retrieval;
2. combining dense semantic and sparse lexical retrieval;
3. reranking and grading retrieved chunks;
4. asking the model to cite claims using numbered source chunks;
5. checking citations and grounding before returning an answer;
6. abstaining when evidence is missing or infrastructure is unavailable.

The project is a portfolio implementation. It has production-minded controls,
but external service deployment, corpus completeness, model quality, and
security configuration still affect its final behavior.

## 2. Runtime architecture

The backend is a FastAPI application with a compiled LangGraph workflow.

```text
POST /chat
  |
  +-- API key and rate-limit checks
  +-- L1 exact cache
  +-- L2 semantic cache, partitioned by strategy and KB generation
  |
  +-- expand_query
  +-- identify_articles
  |     +-- local article-level dense search
  |     +-- typed status: ok / no_match / unavailable
  |
  +-- optional graph_search
  +-- retrieve
  |     +-- article-scoped dense + sparse Qdrant search
  |     +-- server-side RRF
  |     +-- FlashRank reranking
  |     +-- typed status: ok / no_results / unavailable
  |
  +-- optional page_index_enrich
  +-- grade_documents
  +-- generate
  |     +-- NeMo Guardrails first
  |     +-- direct LLM fallback when Guardrails is unavailable
  |     +-- numbered citations
  |
  +-- check_hallucination
  |     +-- LLM grounding check
  |     +-- citation range and evidence verification
  |     +-- retry or deterministic abstention
  |
  +-- check_answer_quality
  +-- attribution check
  +-- final SSE response and cache write
```

The graph has a bounded step budget and separate hallucination/quality retry
budgets. A request deadline prevents a slow external model from holding an SSE
connection indefinitely.

## 3. Data ingestion

### 3.1 Batch snapshot ingestion

`data_pipeline/ingest.py` streams the English Wikipedia snapshot from
Hugging Face:

1. read articles in streaming mode;
2. split text into 512-character chunks with 64-character overlap;
3. create dense FastEmbed vectors;
4. create sparse BM25 vectors;
5. optionally extract spaCy entities;
6. upsert chunk points and article-level points;
7. save a checkpoint only after Qdrant acknowledges the upload.

Chunk payloads contain:

- `title`
- `url`
- `page_content`
- `chunk_index`
- `entities`
- `source_document_id`
- `revision_source = "dataset_snapshot"`
- `ingested_at`

The Hugging Face article ID is explicitly treated as a dataset document
identifier, not as a MediaWiki revision ID.

The article collection contains one dense point per title. Its vector is made
from the title and first two paragraphs. The chunk collection contains the
hybrid dense/sparse points used for answer retrieval.

### 3.2 Live synchronization

`data_pipeline/wiki_updater.py` listens to Wikimedia EventStreams for
English namespace-0 edits, new pages, and deletes.

For an edit or new page:

1. fetch the current article text from the MediaWiki API;
2. reject fetch failures so the event enters the DLQ;
3. treat a confirmed empty page as a deletion;
4. create new embeddings off the async event loop;
5. upsert the new deterministic chunk IDs;
6. scroll the title and delete every old point not in the new ID set;
7. update the article-level index;
8. advance the shared Redis knowledge-base cache generation.

This order matters. New content is written before surplus content is removed,
so an embedding or upsert failure does not erase the previously available
content. A stale-delete failure is still raised and retried rather than
silently ignored.

For a delete or confirmed empty page, the updater removes both the chunk-level
and article-level entries, then invalidates answer caches.

### 3.3 Reconciliation

`data_pipeline/reconciler.py` periodically samples indexed titles, checks
their latest MediaWiki revision IDs, and replays stale articles through the
same updater path.

Sampling uses valid Qdrant scroll cursors and reservoir sampling. It does not
invent random point IDs. The reconciler distinguishes:

- live data, where `revision_id` is a MediaWiki revision;
- snapshot data, where `source_document_id` is not comparable to a live
  revision;
- legacy data with no source metadata, which is eligible for refresh.

Every cycle records health statistics, including empty cycles and failures.

## 4. Retrieval

### 4.1 Article discovery

The article-level index reduces cross-article contamination. A successful
search can return titles or a clean no-match result. Qdrant/model failures
raise an `ArticleIndexUnavailable` error, which the graph records as
`article_discovery_status = "unavailable"`.

An empty result is not treated as an infrastructure failure.

### 4.2 Hybrid chunk search

For each query variant, `backend/retrieval.py`:

- embeds the query densely and sparsely;
- optionally applies a title MatchAny filter;
- executes dense and sparse Prefetch queries;
- fuses candidates with Qdrant RRF;
- reranks the candidates with FlashRank;
- returns documents plus `retrieval_status`.

The status values are:

- `ok`: documents were retrieved and reranked;
- `no_results`: the service worked but produced no documents;
- `unavailable`: an infrastructure or model call failed.

The graph preserves this status in AgentState and in API metadata. Retrieval
unavailability is not represented as if Wikipedia simply had no answer.

### 4.3 PageIndex and knowledge graph

PageIndex reconstructs article text from chunks, parses structural headings,
and asks an LLM to select relevant sections. The knowledge graph uses NER
entities and co-occurrence edges to broaden article discovery for multi-hop
questions.

These are optional strategies. Their failures are isolated and logged, while
the primary article/chunk retrieval path remains available.

## 5. Generation and provenance

The generation prompt numbers each retrieved document and wraps it in a
`<document>` block. Document text is explicitly labelled untrusted evidence;
instructions inside it must not be followed.

The assistant is instructed to:

- answer only from the supplied evidence;
- append valid `[N]` citations to factual claims;
- say it cannot answer when evidence is missing;
- avoid JSON and tool-call output.

The citation checker evaluates sentences independently. It:

1. extracts references attached to that sentence;
2. rejects references outside the document range;
3. token-matches key terms against the referenced chunk;
4. computes a provenance score;
5. applies a minimum provenance gate.

The grounding LLM call fails closed. If the answer remains ungrounded after
the configured retry budget, the graph returns a deterministic abstention
instead of sending the unsupported answer through quality checking.

The response reports:

- `provenance_score`
- `attribution`
- `citation_map`
- `retrieval_status`
- `article_discovery_status`
- retry counts
- `guardrails_applied`

The checker is evidence verification, not a formal proof of factual truth. It
should be treated as a strong application-level control rather than a
mathematical guarantee.

## 6. Caching

WikiMind has two cache layers:

- L1: exact normalized query hash;
- L2: semantic similarity through RedisVL or a bounded pure-Redis fallback.

Both layers include:

- active strategy names;
- the Redis knowledge-base generation.

Therefore, the same question under different retrieval strategies cannot
cross-hit, and a successful live article update invalidates old answer
namespaces by advancing the generation.

The pure-Redis fallback scans a bounded pool and is O(n). RedisVL is preferred
for larger deployments. Cache failures are degraded gracefully and never
prevent a fresh query from running.

## 7. API and operations

The API exposes:

- `POST /chat`
- `POST /chat/compare`
- `GET /health`
- `GET /api/metrics`
- `GET /api/traces`
- `GET /api/eval-results`
- `GET /api/pipeline-health`

The chat routes support:

- optional `X-API-Key` authentication;
- per-client in-process rate limiting;
- configurable request timeout;
- bounded trace history;
- safe SSE error references instead of raw exception details.

Set `API_KEY` for a deployed environment. The default empty value is
convenient for local development and should not be used as a production
security posture.

The health endpoint checks local Qdrant through the synchronous client in a
worker thread and checks remote Qdrant through HTTP. Langfuse authentication
checks are also moved off the async event loop.

Local and remote Qdrant modes point to different stores. The embedded path is
useful for development and offline tests, while a Qdrant service is the
recommended mode for a large indexed corpus. A mode switch alone does not
move or expose data between those stores.

## 8. Observability

Langfuse traces are optional and failure-tolerant. Prometheus instrumentation
is enabled when installed. The dashboard displays recent in-process traces,
latency, steps, provenance, attribution, Guardrails activity, and evaluation
results.

The in-memory trace ring is intentionally limited to 500 entries. It resets on
process restart and is not shared between multiple workers. Durable production
observability should use Langfuse and Prometheus.

## 9. Evaluation

The evaluation harness loads Natural Questions or TriviaQA subsets and records:

- Recall@K
- MRR
- answer accuracy
- latency percentiles
- graph step counts
- raw per-query outputs

Benchmark results are environment-dependent. A fair comparison must keep the
dataset subset, corpus, model, provider, and strategy configuration fixed.

## 10. Testing

The repository includes regression tests for:

- local async Qdrant deletion;
- batch provenance payloads;
- citation validation and sentence-level verification;
- hallucination retry exhaustion;
- strategy-partitioned cache keys;
- whitespace-only request rejection.

Run:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall -q backend data_pipeline evaluation frontend tests
```

`data_pipeline.verify_pipeline` remains a service-backed integration suite.
It can validate local embedded Qdrant or remote Qdrant, but must be run with
the selected Qdrant mode and Redis available to validate the cache layer. The
ingestion checkpoint is progress metadata only; actual collection counts are
the source of truth for a deployment.

## 11. Deployment boundaries

The project is ready to present as a carefully engineered portfolio project.
The following boundaries should remain visible in a resume/demo discussion:

- no historical article storage;
- no distributed transaction across Qdrant and derived indexes;
- dashboard traces are process-local;
- pure-Redis semantic search is bounded;
- embedded Qdrant is not the recommended mode for large collections;
- API authentication is optional and must be enabled for deployment;
- Docker/service integration requires a machine with Docker and the required
  secrets.

The final project claim is therefore: **an end-to-end, latest-state, hybrid RAG
portfolio system with explicit grounding checks, repair paths, and automated
regression coverage**.
