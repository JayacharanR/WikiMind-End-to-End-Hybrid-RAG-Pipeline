# WikiMind Code Review Report

**Review date:** 2026-08-07
**Repository state:** main at 9933d5e (clean before this report)
**Scope:** Backend, retrieval/orchestration, ingestion and live sync, cache, evaluation, frontend/dashboard, Docker/configuration, and repository claims.

## Executive verdict

WikiMind is an ambitious and technically interesting RAG prototype with a good component decomposition, useful observability surfaces, and a credible portfolio story. The Python modules compile and the LangGraph graph imports successfully.

It is not yet safe to describe the current repository as “production-grade,” “two-stage article-scoped retrieval,” “temporal versioned,” or “self-healing” without qualification. Several important paths either bypass the feature they claim to implement or fail under the default/local deployment mode. The most serious issues affect answer grounding, cache correctness, time travel, live synchronization, Docker deployment, and the primary Streamlit UI.

### Release recommendation

- **For a college portfolio demo:** acceptable after fixing the P0 items below and rewriting the claims to describe a validated prototype.
- **For a production-ready claim:** not ready. Authentication, isolation, timeouts, tests, deployment validation, data/version semantics, and multi-process state still need work.

## Verification performed

| Check | Result |
|---|---|
| Python compilation of backend, data_pipeline, evaluation, frontend | Passed |
| Import and compilation of backend.agent.agent_app | Passed; CompiledStateGraph created |
| QDRANT_MODE=local async client check | Confirmed get_async_qdrant() returns None |
| Redis failure behavior | Confirmed cache.l1_get() raises if get_redis_client() raises |
| Qdrant time-range model check | Confirmed models.Range(lte="2024-06-15T00:00:00Z") raises a validation error; DatetimeRange is compatible |
| Hybrid retrieval against an invalid Qdrant endpoint | Returns an empty result and logs the failure instead of propagating a service error |
| Unit test collection | Not available: no tests/ directory and pytest is not installed locally |
| Ruff/lint check | Not available: Ruff is not installed locally |
| Docker end-to-end run | Not performed; external Docker services/data are not part of this review environment |

The local ignored runtime artifacts contain Qdrant/data files, but those artifacts are not part of the Git revision. Therefore the reported 3M+ chunk state cannot be reproduced from a clean clone without a separate data/rebuild procedure.

## Priority summary

| Priority | Meaning | Count | Recommended action |
|---|---|---:|---|
| P0 | Correctness, data integrity, or deployment blocker | 10 | Fix before presenting as finished |
| P1 | High-risk reliability, security, or evaluation weakness | 15 | Fix before calling it production-ready |
| P2 | Quality, maintainability, and documentation improvements | 12 | Fix during final polish |

## P0 findings — fix before release

### P0-1 — Article-level discovery is completely bypassed

**Evidence:** backend/agent.py:185-190 calls hybrid_search(..., article_titles=None) even though target_articles was produced in the previous node. backend/retrieval.py:126-137 contains the intended article filter, but the agent never supplies it.

**Impact:** The advertised two-stage retrieval architecture is not operating. Stage 1 still consumes embedding/search work, but Stage 2 searches the complete chunk collection. This increases noise and latency and invalidates any benchmark or explanation that attributes quality to article scoping.

**Fix:** Pass article_titles=target_articles for every expanded query. Keep an explicit, measured global fallback only when article discovery fails, and expose that fallback in metadata. Use a separate article_discovery_failed state rather than silently treating an empty index and a valid “no articles found” result the same way.

### P0-2 — Cache initialization errors can fail chat requests

**Evidence:** backend/main.py:222-224 performs cache_lookup() before invoking the graph. backend/cache.py:95-105 calls get_redis_client() before entering its try block. The same pattern exists in l1_set() at backend/cache.py:126-128.

**Reproduced:** Monkey-patching get_redis_client() to raise RuntimeError("redis unavailable") caused l1_get() to raise the same exception.

**Impact:** Normal Redis command failures are caught, but failures while creating/configuring the client are not. This makes the graceful-degradation guarantee incomplete: an invalid Redis URL, client initialization problem, or future factory change can prevent the agent from running. Fire-and-forget cache writes can also create unobserved task failures.

**Fix:** Wrap client acquisition and operation in the same exception boundary, or make a single safe_cache_lookup() wrapper that always returns (None, None) on infrastructure failure. Await or supervise cache writes and log task failures. Add tests for both a refused Redis connection and an invalid Redis URL.

### P0-3 — Local Qdrant mode breaks PageIndex and all async workers

**Evidence:** backend/qdrant_client.py:57-75 deliberately returns None for the async client in local mode. backend/retrieval.py and backend/article_index.py correctly contain sync fallbacks, but backend/agent.py:252-267, data_pipeline/wiki_updater.py:107-126, data_pipeline/reconciler.py:43-61, and data_pipeline/reconciler.py:148-170 call methods on the async client without a local fallback.

**Reproduced:** With QDRANT_MODE=local, get_async_qdrant() returned None.

**Impact:** The default configuration says local embedded mode is supported, but PageIndex, live updates, reconciliation, and the verification script cannot operate in that mode.

**Fix:** Provide shared helpers for sync Qdrant calls via asyncio.to_thread(), and use them consistently. Alternatively, make local mode explicitly single-process/development-only and fail startup with a clear configuration error when a worker requiring async remote Qdrant is enabled.

### P0-4 — Time-travel filtering is implemented with incompatible types

**Evidence:** backend/qdrant_client.py:134-158 creates the ingested_at payload index as KEYWORD. backend/retrieval.py:141-147 uses models.Range(lte=as_of_date), where the installed Qdrant client expects numeric values. backend/models.py:64-69 accepts any string rather than validating an ISO datetime.

**Reproduced:** models.Range(lte="2024-06-15T00:00:00Z") raises a Pydantic validation error; models.DatetimeRange accepts the value.

**Impact:** A request using as_of_date can fail before the retrieval exception handler. Even if the filter were accepted, a keyword index is not the correct schema for datetime range queries. The feature described as time travel is therefore not reliable.

**Fix:** Parse the request into a timezone-aware datetime. Use a Qdrant DATETIME payload index with DatetimeRange, or store epoch seconds as an INTEGER and use numeric Range. Add tests for current mode, before-first-ingest, between revisions, malformed dates, and timezone offsets.

### P0-5 — Live updates do not preserve temporal versions

**Evidence:** backend/qdrant_client.py:160-177 generates IDs from only article_title and chunk_index. data_pipeline/wiki_updater.py:106-127 marks current points false, then data_pipeline/wiki_updater.py:141-171 upserts new points with those same IDs.

**Impact:** New chunks overwrite the old points. Old versions are not retained for overlapping chunk positions, so historical content cannot be retrieved. Only chunks that disappear from a later article may remain as archived tombstones. The temporal versioning claim is false for most edits.

**Fix:** Include revision ID/version in the point ID, for example (title, revision_id, chunk_index). Write the new revision first, verify it, then atomically or transactionally switch a current-revision marker. Store a revision manifest per article so time-travel retrieval can select the newest revision at or before the requested date rather than returning arbitrary old chunks.

### P0-6 — Reconciliation stores an empty revision ID

**Evidence:** data_pipeline/reconciler.py:200-205 creates a fake event containing only title and meta.uri; data_pipeline/wiki_updater.py:80-84 reads event_data["revision"]["new"], which is absent in the fake event.

**Impact:** Re-ingestion triggered by reconciliation writes revision_id="". The next reconciliation sees no stored revision and flags the same article again. The worker can report successful re-ingestion while failing to repair drift state.

**Fix:** Have the reconciliation API request return the current revision ID and pass it into process_event, or change process_event to accept an explicit revision_id. Test two consecutive reconciliation cycles and assert that the second cycle does not report the same unchanged article as stale.

### P0-7 — Docker workers do not share the backend’s Qdrant

**Evidence:** docker-compose.yml:14 sets QDRANT_MODE=remote only for backend. docker-compose.yml:64-76 and :86-99 set QDRANT_URL for the updater and reconciler but omit QDRANT_MODE=remote. The config default is local (backend/config.py:38-46).

**Impact:** In Docker, the backend connects to qdrant:6333, while the workers default to embedded Qdrant and ignore QDRANT_URL. They therefore operate on isolated/non-persistent stores rather than the shared collection.

**Fix:** Set QDRANT_MODE=remote for every service, mount only the required data/DLQ paths, and add a startup log/test asserting the resolved mode and endpoint. Do not rely on an environment variable that is present only in the backend service.

### P0-8 — A clean Docker deployment starts with empty collections

**Evidence:** docker-compose.yml:109-116 creates a fresh named Qdrant volume. There is no ingestion/import service or seed job. The checked-in Qdrant files are ignored by Git, and Dockerfile.backend only copies source code (:39-43).

**Impact:** docker compose up -d can create empty collections, after which article discovery returns no targets and the fallback generator can answer from model memory. The documented full stack is not a reproducible answering system.

**Fix:** Add an explicit data bootstrap flow: a documented remote ingestion command, a small committed demo snapshot, or a versioned Qdrant backup/import procedure. Make the backend readiness endpoint reject an empty index when demo data is required. Do not imply that the full Wikipedia data ships with the repository.

### P0-9 — No-context fallback can generate ungrounded answers and skips validation

**Evidence:** backend/agent.py:461-487 calls direct LLM generation even when there are no snippets, using the string “No relevant context was found.” backend/agent.py:767-775 routes irrelevant retrieval directly to generate_from_web, and backend/agent.py:852 ends that path immediately.

**Impact:** The pipeline can answer from parametric memory when retrieval fails, without hallucination grading, answer grading, attribution detection, or a reliable provenance result. This conflicts with the central “answer only from verified Wikipedia context” promise. The UI explicitly acknowledges this at ui/script.js:308, which confirms the behavior is intentional but not aligned with the product claim.

**Fix:** Return a deterministic abstention when no verified context exists, or make web/parametric fallback a clearly labeled and separately evaluated mode. Route fallback output through the same safety/quality/provenance policy and never cache an ungrounded answer as if it were a normal RAG response.

### P0-10 — The primary Streamlit frontend cannot handle cache-hit responses

**Evidence:** The backend returns JSON immediately on cache hit at backend/main.py:223-235, but frontend/app.py:221-242 always posts as an SSE stream and constructs sseclient.SSEClient(response) without checking Content-Type.

**Impact:** The first query may work, while an exact or semantic repeat can render no final answer in the Streamlit UI. The JavaScript ui/ client does implement a JSON/SSE split, but ui/ is not the Docker frontend and appears to be a separate/stale surface.

**Fix:** Handle application/json in Streamlit before constructing the SSE client, or make /chat return one stable response protocol. Add an API contract test for both cache miss and cache hit.

## P1 findings — high-risk issues

### P1-1 — Retry counts and the step budget do not match the documentation

backend/agent.py:790-795 permits only one hallucination retry, and :804-809 permits only one answer-quality retry, despite the guide describing larger budgets. _is_over_budget() is checked only in selected routing functions (:761-765, :786-807), not before every node. A quality retry can run expansion, discovery, retrieval, grading, generation, and checking beyond the configured step count.

Additionally, hallucination retries regenerate the same query and same context with temperature zero (:503-516), so there is no correction mechanism. Answer-quality retries can also repeat the same pipeline when no expansion strategy is enabled.

**Fix:** Define retry budgets in settings, pass recursion_limit to LangGraph, check budget before each node, and make retries change something measurable: retrieve more candidates, alter the prompt, use a different query, or abstain.

### P1-2 — Citation verification is not a reliable verifier and is not enforced

backend/agent.py:526-584 does not use citation_map to validate references. It uses substring matching, so terms such as art can match unrelated words; it does not reject out-of-range citations; and generation.find(segment) can associate citations from later parts of the answer with an earlier segment. Most importantly, no-citation output gets a score of 0.0, but node_check_hallucination() can still mark the answer grounded based only on the LLM result (:643-649).

The committed evaluation results contain answers without inline citations, despite the citation prompt. This is direct evidence that the citation contract is not enforced.

**Fix:** Parse claims and citation spans deterministically, validate every reference against the current document list, require citations for factual sentences, and make low provenance fail or abstain. Add tests for multiple citations, invalid references, uncited claims, repeated text, negation, and paraphrases. Treat the score as a diagnostic unless it is backed by a stronger claim-level verifier.

### P1-3 — The answer quality check is mostly bypassed

backend/agent.py:706-710 automatically passes every generation of at least ten words unless it contains “cannot answer.” This makes the LLM quality grader irrelevant for most answers, including long off-topic or error messages. Exception paths in both grading checks default to pass (:399-401, :733-735).

**Fix:** Remove the word-count bypass. Use structured output with a strict schema, explicit uncertainty handling, and a fail-closed policy for malformed checker output. Keep a bounded fallback only when the service is deliberately in degraded mode.

### P1-4 — Cache keys can return the wrong answer for the same query

backend/cache.py:81-92 keys only on normalized query text. backend/main.py:220-224 does not include retrieval strategies, as_of_date, index/data version, or any response policy in the cache lookup.

**Impact:** A current answer can be returned for a historical query; a baseline answer can be returned after enabling PageIndex or graph search; an answer from an old Wikipedia revision can survive an update.

**Fix:** Include a canonical request signature containing query, strategies, date, model/prompt version, and knowledge-base revision. Invalidate affected keys on article updates, or use a version namespace. cache_invalidate() only removes L1 (backend/cache.py:434-449) and does not remove the L2 entry.

### P1-5 — Retrieval hides infrastructure failures as “no documents”

backend/retrieval.py:245-247 catches every exception and returns an empty document list. This converts Qdrant outages, embedding failures, reranker failures, schema errors, and malformed filters into the ungrounded fallback path.

**Fix:** Distinguish no_results from retrieval_unavailable. Return a typed error/status to the graph, expose it in metadata, and abstain or return a service-unavailable response when the knowledge base cannot be queried.

### P1-6 — Synchronous embedding and reranking block the async server

backend/retrieval.py:110-121 performs embeddings synchronously, and :226-231 performs FlashRank reranking synchronously inside an async function. Article search and graph/NER work have similar behavior.

**Impact:** One CPU/GPU-heavy request can block the event loop, reducing concurrency and making latency measurements optimistic under sequential benchmarks.

**Fix:** Move CPU/model operations to controlled worker threads/processes, bound concurrency, and measure queue time separately from model time. Do not share a mutable model instance across uncontrolled ingestion threads without testing thread safety.

### P1-7 — Live updater silently treats fetch failures as successes

data_pipeline/wiki_updater.py:90-94 returns normally when the article fetch fails. The listener then calls record_success() at :233-238. HTTP errors and transient network failures therefore bypass the DLQ.

**Fix:** Raise a typed retryable exception for fetch failures, distinguish deleted page from temporary failure, and add the event to the DLQ only when retryable processing fails.

### P1-8 — Reconciliation samples the first page, not a random population

data_pipeline/reconciler.py:36-69 calculates total_points but never uses it and calls scroll() without a random offset. It repeatedly examines the first chunk page, which is not representative at millions of points.

**Fix:** Sample current article IDs from a dedicated article collection, use a reproducible random strategy, or maintain a reconciliation queue/partition cursor. Report sample coverage and age.

### P1-9 — Ingestion is not portable or safely resumable

data_pipeline/ingest.py:26-39 hardcodes Windows-specific directories and changes process-wide cache/temp environment variables at import time. This breaks Linux Docker portability and contaminates backend imports because backend/retrieval.py imports the ingestion module.

The article/chunk writes use wait=False (:204-210, :357-364) but checkpoints are saved after the submitted batch is considered complete (:588-603). A crash can checkpoint work before Qdrant has durably accepted it. The three-thread executor (:584-649) also shares global embedding models and a Qdrant client without a demonstrated concurrency contract.

**Fix:** Move paths to settings and resolve them relative to an explicit application/data root. Keep ingestion-only environment setup inside the CLI. Use wait=True or an explicit completion/acknowledgement before checkpointing, and use bounded, tested worker isolation. Checkpoint article IDs/titles plus dataset revision, not only a row count.

### P1-10 — Batch ingestion likely stores a page ID as a revision ID

data_pipeline/ingest.py:289-310 maps the dataset field id directly to revision_id. The ingestion code does not establish that this field is a MediaWiki revision ID, while the live/reconciliation path compares it with live revision IDs.

**Impact:** If the Hugging Face Wikipedia dataset id is a page/document ID, reconciliation will mark every batch-ingested article stale.

**Fix:** Store the source dataset/document ID under its own field and store a verified MediaWiki revision ID separately. If the snapshot has no revision ID, mark it as snapshot data and exclude it from revision equality checks until enriched.

### P1-11 — Article index and caches are not refreshed by live sync

data_pipeline/wiki_updater.py:131-171 updates only the chunk collection. It does not update the article-level vector in wikimind_articles, PageIndex trees, graph data, or cached answers.

**Impact:** Stage 1 can continue selecting stale summaries or miss newly created articles. PageIndex can serve a stale tree for 24 hours, and L1/L2 answer caches can serve stale content after a Wikipedia edit.

**Fix:** Treat an article update as a consistency workflow: write chunk revision, article summary revision, graph delta/rebuild, invalidate PageIndex and answer-cache namespaces, then publish the new current revision.

### P1-12 — Knowledge graph incremental merge is not actually additive or current-only

data_pipeline/graph_builder.py:56-93 scrolls all points without filtering is_current=true. :192-198 uses nx.compose, which can replace node/edge attributes rather than summing all historical weights and source provenance. The live updater never updates the graph.

**Fix:** Build from current revision chunks only, use a graph storage model that supports per-edge source/revision records, and make incremental updates idempotent. Avoid loading the full NetworkX graph from Redis/file on every query (backend/knowledge_graph.py:252-256).

### P1-13 — PageIndex has stale-cache and structural correctness problems

backend/page_index.py:137-167 keys the cached tree only by article title, not revision. parse_article_tree() stores children in a dictionary keyed by title (:82-92), so duplicate section titles overwrite each other. LLM-selected section titles must exactly match _find_section() (:112-122), making normal LLM formatting variations fail silently.

**Fix:** Key trees by (title, revision_id), preserve duplicate sections as ordered lists with stable IDs, validate/normalize selected section identifiers, and add parser tests for nested, duplicate, malformed, and short articles.

### P1-14 — Docker backend runs four processes with process-local state

Dockerfile.backend:50 starts Uvicorn with four workers. Traces (backend/main.py:30-31), reranker/model singletons, cache strategy state, and Qdrant client state are process-local.

**Impact:** /api/metrics reports only one worker’s ring buffer; each worker loads heavy models; embedded Qdrant mode can lock or corrupt under multiple processes; singleton compilation happens once per worker. The observed memory estimate is therefore multiplied by worker count.

**Fix:** Start with one worker for the model-heavy service, or move shared state to Redis/Prometheus/Langfuse and use a separate model-serving strategy. Explicitly test multi-worker behavior before enabling it.

### P1-15 — API security and resource controls are absent

backend/main.py:130-137 allows all origins and credentials, and no endpoint has authentication, authorization, rate limiting, request quotas, or per-node timeouts. /api/traces exposes queries and generated text. SSE errors return raw exception strings (:351-356). CompareRequest accepts an unbounded list of arbitrary config dictionaries (backend/models.py:76-90).

**Fix:** For a public deployment add API-key/JWT auth, origin allowlisting, rate limits, body/compare limits, request IDs, timeouts, safe error messages, and redaction/access control for traces. For a local demo, state explicitly that the API is unauthenticated and bind only to localhost.

## P2 findings — final polish and maintainability

1. **Configuration is inconsistent.** backend/config.py:27-29 uses OpenRouter fields, while .env.example:8-9 and the README describe OPENAI_API_KEY/OPENAI_MODEL. llmops.py:33-44 and :93-104 read os.getenv() directly rather than the Pydantic settings object, so values present only in .env may be invisible to Langfuse locally. Choose one provider naming scheme and use get_settings() everywhere.

2. **Dependency ownership is incomplete.** data_pipeline/ingest.py directly imports langchain_text_splitters, but pyproject.toml does not list it as a direct dependency. Direct imports should be declared directly. The project uses broad >= constraints and Docker images latest; use a locked, tested release policy and pin infrastructure images.

3. **Packaging configuration appears stale.** pyproject.toml lists mobile in Hatch packages, but there is no tracked mobile/ package. Remove it or add the package and run a clean wheel build in CI.

4. **The Makefile does not match the project toolchain.** Makefile:14-20 invokes Poetry, but the repository provides pyproject.toml/uv.lock and no Poetry configuration. make test also assumes a non-existent tests/ directory. Standardize on uv (or add Poetry and dev dependencies) and make make test pass on a clean clone.

5. **There is no real unit-test suite or CI.** Add tests for pure functions first, then mocked Qdrant/Redis/LLM graph tests, then service-level tests. Add GitHub Actions for formatting, type checks, unit tests, build, and a smoke test. Keep live Wikipedia tests opt-in and isolated.

6. **The verification script is not a reliable test runner.** data_pipeline/verify_pipeline.py:146-171 writes a fake revision for Guido van Rossum into the real collection and does not clean it up. The test does not verify archival semantics. test_cache_l2() returns True unconditionally at :367-394, even if both semantic checks fail. It should use a temporary collection/namespace, assert every sub-check, and return structured results.

7. **Evaluation metrics are too weak for the claims.** evaluation/metrics.py:45-66 calls answer-string containment Recall@K; :69-90 computes MRR over the entire list and has no K parameter; :97-117 uses normalized substring accuracy. These are useful smoke metrics but should be labeled heuristic exact-match metrics. Add dataset passage/title hit rate, citation correctness, abstention precision/recall, groundedness, error rate, and confidence intervals.

8. **The recorded benchmark is small and not fully reproducible.** The committed reports use 10 NQ examples, with mean step count 16 in older runs, while the current graph’s normal successful path is 7 steps. The results should state code commit, model/provider, prompt version, database snapshot, retrieval configuration, and whether caching was disabled. Do not present 0.90 recall/0.80 accuracy as general system quality.

9. **The dashboard evaluation tab consumes the wrong JSON shape.** Backend /api/eval-results returns saved report keys aggregates and per_query_results (backend/main.py:533-549). dashboard/js/evaluation.js:24-29 expects aggregate and per_query, and later expects mean_accuracy/mean_steps. The tab will show empty/zero data. Normalize the API response or update the dashboard fields.

10. **Dashboard and UI output are not escaped.** dashboard/js/traces.js:63-111 and :143-154, dashboard/js/app.js:125-141, and ui/script.js:298-308 interpolate queries, generations, titles, and URLs into innerHTML. The model/retrieved text is untrusted. Use text nodes/DOM APIs or a sanitizer and validate URL schemes. Pin CDN assets and add a CSP for the standalone UI.

11. **The standalone ui/ tree is a second, stale frontend.** It hardcodes http://localhost:8000 (ui/script.js:1), is not copied by Docker, and exposes controls that do not include all backend strategies. Either remove it, serve it intentionally, or label it as an experimental client. Avoid documenting two competing frontends.

12. **SSE is progress streaming, not answer-token streaming.** backend/main.py:281-349 emits node-completion events and one final answer; it does not stream model tokens. Update the README/UI wording or implement token streaming with cancellation and a robust SSE parser. The standalone JavaScript parser (ui/script.js:180-215) also assumes line boundaries align with network chunks, which is not guaranteed.

## Additional correctness and design notes

- node_identify_articles() searches only the original query (backend/agent.py:99-105); expanded queries are used only later. If expansion is intended to improve Stage 1 discovery, pass the expanded candidates or fuse article-level results.
- node_generate_from_web() is now a misleading name because the current article index replaced Tavily and web_snippets is initialized empty. Rename it to an abstention/parametric fallback node and expose its source mode.
- node_check_hallucination() treats any answer containing “cannot answer” or “no relevant” as grounded with provenance 1.0 (backend/agent.py:603-606). This is not evidence of grounding; it is an abstention signal and should be represented separately.
- _check_attribution() (backend/agent.py:660-689) asks the same LLM whether it knows the answer from training data. That is a self-report, not a context-ablation experiment and not a reliable source attribution measurement. Compare answer quality with and without context using a separate controlled evaluation if this feature is kept.
- query_expansion.py:141-161 runs expansion strategies sequentially despite documentation calling them parallel. Add concurrency only with a bounded semaphore and per-call timeout; otherwise document the latency trade-off.
- There are no per-node LLM timeouts. max_graph_steps cannot stop a node that is currently waiting on a network call. Add provider/client timeouts and a request deadline propagated through all nodes.
- get_reranker() and model initialization use module globals without locks. Concurrent first requests can initialize duplicate models. Use application lifespan initialization, an async lock, or a dependency container.
- PipelineHealthTracker._save_dlq() (data_pipeline/pipeline_health.py:280-287) writes directly to the final JSON file without an atomic temp-file replace or lock. A process interruption can corrupt it. The health status also depends mostly on consecutive failures (:233-247) and does not mark a worker stale when heartbeat stops.
- The updater’s shutdown handler can wait through the full backoff sleep (data_pipeline/wiki_updater.py:261-270). Use an interruptible event wait.
- qdrant_client.init_collection() returns immediately when a collection exists (backend/qdrant_client.py:104-111) and never validates vector dimensions, sparse schema, or required payload indexes. Add schema/version checks and migrations.
- health_check() uses HTTP QDRANT_URL regardless of local/remote mode (backend/main.py:177-188) and performs synchronous Langfuse authentication in an async endpoint (:190-200). Separate liveness/readiness, use the configured client, and avoid blocking calls on the event loop.
- CORS with allow_origins=["*"] and allow_credentials=True is unsafe and not a useful production policy (backend/main.py:130-137).
- The cache stores final responses even when generation returned "Error generating response." (backend/agent.py:446-458, backend/main.py:300-333). Do not cache failed or degraded responses unless that behavior is explicit and short-lived.

## Recommended implementation order

### Phase 1 — Make the current demo truthful and safe

1. Fix cache exception handling and cache-key dimensions.
2. Fix article scoping in node_retrieve().
3. Replace ungrounded no-context generation with explicit abstention.
4. Fix local/remote Qdrant behavior and time-travel types.
5. Fix Streamlit cache-hit handling.
6. Fix Docker worker Qdrant mode and document/validate data bootstrap.
7. Remove or rewrite claims that are not yet implemented.

### Phase 2 — Repair data integrity and evaluation

1. Implement revision-aware point IDs/manifests and update article index/cache invalidation.
2. Fix reconciliation revision propagation and sampling.
3. Replace the verification script with isolated tests that do not mutate the production collection.
4. Add unit tests for cache, filters, citation parsing, graph routing, checkpointing, and SSE/API contracts.
5. Re-run benchmarks from a documented commit/snapshot with cache disabled and report error/abstention rates.

### Phase 3 — Production hardening

1. Add authentication, rate limits, request deadlines, structured logs, redaction, and safe error responses.
2. Move shared metrics/state out of process-local globals or run a single worker intentionally.
3. Add CI, a clean Docker build/smoke test, dependency/image pinning, and schema migration checks.
4. Sanitize all browser-rendered model/user content and remove the duplicate UI surface.

## Recommended portfolio wording after Phase 1

Use wording such as:

> “Built and evaluated a Wikipedia-based hybrid RAG prototype with dense+sparse retrieval, LangGraph orchestration, citation-aware generation, live-sync workers, caching, and observability. Validated the core query path and ingestion components locally; documented remaining production hardening work including authentication, automated tests, revision consistency, and deployment data bootstrapping.”

Avoid claiming “production-grade,” “true temporal versioning,” “fully article-scoped retrieval,” “self-healing,” or “every claim is verified” until the corresponding P0/P1 findings are fixed and covered by tests.

## Final acceptance checklist

- [ ] Clean clone can install with the documented toolchain.
- [ ] pytest exists, tests run, and CI is green.
- [ ] Redis-offline chat still reaches the agent or returns a controlled degraded response.
- [ ] Local and remote Qdrant modes are each tested, or local mode is explicitly unsupported for workers.
- [ ] Article discovery actually scopes chunk retrieval.
- [ ] Historical queries use a valid datetime schema and return a deterministic revision.
- [ ] Live update keeps old revisions, updates article index, and invalidates stale caches.
- [ ] Reconciliation writes and verifies real revision IDs.
- [ ] No-context behavior abstains or is explicitly labeled as parametric/web fallback.
- [ ] Cache miss and cache hit both render in Streamlit.
- [ ] Docker workers share the intended Qdrant and a documented dataset is available.
- [ ] Citation and provenance tests reject missing/invalid citations.
- [ ] API has the intended authentication, rate, timeout, and trace-access policy.
- [ ] Dashboard evaluation data matches the backend JSON schema.
- [ ] Model/user text is escaped or sanitized in every browser surface.
