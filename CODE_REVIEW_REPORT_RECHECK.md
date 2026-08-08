# WikiMind Code Review Re-check

**Review date:** 2026-08-08  
**Reviewed commit:** `13f36e7` (`origin/main`)  
**Compared against:** the previous review at `9933d5e` and the supplied remediation walkthrough  
**Scope:** backend API and agent, retrieval, Qdrant lifecycle, ingestion and live sync, reconciliation, cache, evaluation, frontend/dashboard, documentation, Docker, and release claims.

## Executive verdict

The remediation work fixed several real defects: Redis acquisition is now protected, local Qdrant has an async adapter, temporal fields were removed from Python application code, reconciliation passes the live revision, Docker workers use remote Qdrant, no-context output abstains deterministically, Streamlit handles JSON cache hits, retry settings are configurable, and the backend is down to one worker.

However, the repository is not yet ready to be described as production-grade or fully verified. The current commit has one startup-blocking defect, two data/grounding correctness defects, and several residual gaps in freshness, caching, observability, security, testing, and documentation.

### Release recommendation

- **College portfolio/demo:** close after fixing the P0 items, updating the stale claims, and adding a small focused test suite. The architecture is strong enough to present as an evaluated RAG prototype.
- **Production deployment:** not ready. Do not expose the API publicly until authentication, rate limits, deadlines, data bootstrap, data consistency, cache invalidation, automated tests, and deployment smoke tests exist.

The most accurate current description is: **a substantial Wikipedia-based hybrid RAG prototype with a partially implemented live-sync pipeline and observability surfaces, validated by compilation and selected integration checks but not yet production-hardened.**

## Verification performed

| Check | Result |
|---|---|
| Python `compileall` for backend, data pipeline, evaluation, frontend | Passed |
| Imports for models, settings, Qdrant client, and LLMOps | Passed |
| `get_guardrails()` runtime check | Failed with `NameError: name 'os' is not defined` |
| JavaScript syntax checks for dashboard modules | Passed with Node |
| Temporal references in Python application code | None found |
| Temporal references in repository | Still present in `ui/`, `README.md`, and `explanation.md` |
| Unit-test suite | Not present; no `tests/` directory |
| Pytest/Ruff dependency configuration | Not present |
| Docker Compose build/run | Not performed; Docker is unavailable in this environment |
| Qdrant/Redis/live Wikipedia end-to-end re-test | Not performed in this re-check |

The local Qdrant/runtime data is ignored by Git. Therefore, the reported multi-million-chunk database cannot be reproduced from a clean clone without a documented data bootstrap procedure.

## Priority summary

| Priority | Count | Meaning |
|---|---:|---|
| P0 | 3 | Fix before calling the project finished; affects startup, factual freshness, or grounding claims |
| P1 | 14 | Fix before production claims; high-risk correctness, reliability, security, or reproducibility issues |
| P2 | 10 | Final quality, maintainability, evaluation, and documentation work |

## P0 findings

### P0-1 - Backend startup crashes in the FastAPI lifespan

**Evidence:** `backend/llmops.py:11-17` does not import `os`, but `get_guardrails()` calls `os.path.join()` at `backend/llmops.py:154` before its `try` block. `backend/main.py:90-92` calls `get_guardrails()` during startup without a surrounding exception boundary. The runtime check reproduced:

```text
NameError: name 'os' is not defined
```

**Impact:** `uvicorn backend.main:app` can fail during startup before serving requests. The documented graceful Guardrails fallback is not reached. This is a release blocker.

**Required change:** import `os`, move path construction inside the guarded block, and add a startup test that exercises the lifespan with Guardrails unavailable. More robustly, make the Guardrails initialization failure non-fatal and expose `guardrails_available=false` in health metadata.

### P0-2 - Live article updates leave stale chunks when the new article is shorter

**Evidence:** `data_pipeline/wiki_updater.py:115-149` upserts only the new chunk IDs. IDs are deterministic from `(title, chunk_index)` via `backend/qdrant_client.py:184-200`, but there is no delete operation for old chunk indexes and no article manifest.

**Impact:** if an article changes from ten chunks to six, chunks `6..9` remain in Qdrant with the previous text. Retrieval filters only by title, so stale content can be returned alongside the new revision. Removing temporal versioning makes deletion of obsolete current data mandatory; otherwise the knowledge base can answer from facts that no longer exist in Wikipedia.

**Required change:** make article replacement atomic at the application level: write the new revision to a staging namespace or temporary collection, verify the complete set, delete all old points for that title, then publish the new set. At minimum, delete all points for the title before/after upsert with a recovery strategy, and wait for Qdrant acknowledgement. Update the article-level index in the same workflow. Add a regression test where the second version has fewer chunks and assert that no old chunk remains.

### P0-3 - Citation/provenance verification is still advisory, not a gate

**Evidence:** `backend/agent.py:636-650` calculates `provenance_score`, but routes to the useful-answer path whenever the LLM grounding response is positive. A response with no citations receives score `0.0` at `backend/agent.py:535-537` and can still be graded `grounded`. Invalid references are not rejected. The sentence-to-citation association at `backend/agent.py:547-550` searches from the first matching sentence through the rest of the entire answer, so later citations can be assigned to earlier claims.

**Impact:** the project still cannot truthfully claim that every factual claim is cited and verified. A model can omit citations or cite an out-of-range source while passing the grounding node.

**Required change:** parse answer sentences and citation spans deterministically, reject references outside the current citation map, require a citation for every factual sentence, and make a score below the configured threshold route to retry or abstention. Treat LLM grading as an additional signal, not the final authority. Add tests for no citations, invalid citations, multiple citations, repeated text, and uncited claims.

## P1 findings

### P1-1 - Article discovery failure still silently becomes global retrieval

**Evidence:** `backend/agent.py:104-121` returns an empty list both for a valid empty search and for an exception. Although `article_discovery_failed` exists in the state schema, it is never set. `backend/agent.py:185-189` passes `article_titles=None` whenever the list is empty, causing a full-collection search.

**Impact:** the advertised article-scoped Stage 2 is bypassed exactly when Stage 1 is unavailable or returns no article. This is especially misleading in a clean deployment where the article index is empty. It also prevents operators from distinguishing “no matching article” from “article index outage.”

**Required change:** set explicit states such as `article_discovery_failed`, `article_discovery_empty`, and `article_discovery_ok`. Decide the policy explicitly: fail closed/abstain when discovery is required, or expose a deliberately labeled global fallback in response metadata. Do not silently convert both conditions to `None`.

### P1-2 - Strategy-aware cache keys are implemented but not wired

**Evidence:** `backend/cache.py:81-100` accepts a `strategies` argument, but `l1_get`, `l1_set`, `cache_lookup`, `cache_store`, and `cache_invalidate` all call `_hash_query(query)` without strategies. `backend/main.py:230` and `backend/main.py:351` also pass only the query.

**Impact:** the same question can receive a cached Baseline answer after the user enables HyDE, PageIndex, or knowledge-graph retrieval. L2 is also strategy-blind because the semantic prompt is only the raw query. This defeats the stated Phase 2 cache-safety fix.

**Required change:** define a canonical request signature containing normalized query, all strategy flags, model/prompt version, and a knowledge-base generation/version. Pass it through every cache operation and test different strategy combinations. Use a version namespace or explicit invalidation when an article changes.

### P1-3 - Batch snapshot IDs are still treated as MediaWiki revision IDs

**Evidence:** `data_pipeline/ingest.py:287-307` sets `revision_id = str(article.get("id", ""))`. Reconciliation compares that value with MediaWiki `revid` values at `data_pipeline/reconciler.py:138-151`.

**Impact:** the Hugging Face snapshot `id` is not established to be the current MediaWiki revision ID; for the Wikipedia dataset it is generally a page/document identifier. The reconciler can therefore mark ordinary batch-ingested articles stale on the first sample and repeatedly re-ingest them.

**Required change:** store the dataset identifier as `source_document_id`. Store a verified MediaWiki revision separately, or mark snapshot records as `revision_source="dataset_snapshot"` and exclude them from equality comparison until enriched. Add a test with a known page ID and known revision ID.

### P1-4 - Live sync does not maintain all derived indexes or invalidate stale data

**Evidence:** `process_event()` updates only the chunk collection (`data_pipeline/wiki_updater.py:107-149`). It does not update `wikimind_articles`, invalidate PageIndex trees, update the knowledge graph, or invalidate answer caches. PageIndex cache keys are title-only at `backend/page_index.py:137-165`.

**Impact:** Stage 1 can select a stale article summary; new articles may never be discoverable; PageIndex can use a previous tree for up to 24 hours; and cached answers can survive an article edit. The live worker is therefore not a complete consistency workflow.

**Required change:** use an article update coordinator that updates chunks, article summary, and derived indexes, then advances a knowledge-base generation. Namespace PageIndex and answer caches by revision/generation. Handle new and deleted pages explicitly; the current listener processes only namespace-0 `edit` events.

### P1-5 - Retrieval hides infrastructure failures as empty results

**Evidence:** `backend/retrieval.py:219-221` catches every exception, including embedding, Qdrant schema, connection, and FlashRank failures, and returns `([], metadata)`.

**Impact:** an outage is indistinguishable from “no relevant Wikipedia content.” The graph may abstain normally, which hides operational failure from the caller and makes evaluation results misleading.

**Required change:** return a typed retrieval status such as `ok`, `no_results`, or `unavailable`; expose it in response metadata; and return a controlled 503/degraded result when the knowledge base cannot be queried. Log and measure each failure class separately.

### P1-6 - Local-mode health is incorrect

**Evidence:** `backend/main.py:183-194` always checks `settings.qdrant_url/healthz`, even when `QDRANT_MODE=local`. The default is local at `backend/config.py:40-47`.

**Impact:** the documented no-Docker local setup can be functional while `/health` reports Qdrant unhealthy because no server is running on port 6333. Docker health checks and portfolio screenshots can therefore show a false degraded state.

**Required change:** use the configured Qdrant client for embedded mode, or report local storage readiness separately. Distinguish liveness from readiness and include collection existence/counts without treating an empty collection as a network failure.

### P1-7 - Ingestion checkpoints can get ahead of durable Qdrant writes

**Evidence:** `data_pipeline/ingest.py:202-208` and `:354-360` call `upload_points(..., wait=False)`, while the runner checkpoints the batch after the worker future completes (`:585-600`). The executor also shares global embedding models and a Qdrant client across three worker threads (`:581-645`).

**Impact:** a crash or delayed Qdrant write can leave a checkpoint claiming work is complete when the data is incomplete. Shared model/client concurrency has not been demonstrated safe, especially for GPU-backed FastEmbed.

**Required change:** wait for acknowledgement before checkpointing, or persist an idempotent batch manifest and verify counts. Use bounded concurrency supported by the embedding library, or isolate model workers. Add interruption/resume tests.

### P1-8 - Reconciler sampling is not random despite its name and comments

**Evidence:** `data_pipeline/reconciler.py:61-83` calculates `max_offset` but never uses it. It scrolls the first page, then randomly samples only from that page.

**Impact:** the same leading portion of the article collection is repeatedly checked; the sample is not representative of 254K+ articles. Drift elsewhere can remain undetected indefinitely.

**Required change:** use a Qdrant-supported random/sample strategy, a reproducible hash partition/cursor, or a maintained reconciliation queue. Report sample coverage and age. Record health stats even when the collection is empty or sampling fails.

### P1-9 - LLM and reranker operations can still block the async service

**Evidence:** query embedding was moved to `asyncio.to_thread()` in `backend/retrieval.py:108-114`, but FlashRank reranking remains synchronous at `:200-205`; article-level embedding at `backend/article_index.py:104-107`, ingestion/live embeddings, spaCy NER, and several LLM/cache library calls remain synchronous or unbounded.

**Impact:** a model-heavy request can occupy the event loop and reduce concurrency. The 4.72s benchmark is a sequential benchmark, not a throughput result.

**Required change:** offload CPU/model work through a bounded executor, initialize models once under a lock, add queue-time metrics, and run a concurrent-load test. Keep the explicit trade-off documented if the project is intentionally single-user/demo-only.

### P1-10 - Retry and deadline handling is still incomplete

**Evidence:** settings now contain retry budgets, but `_is_over_budget()` is only evaluated in selected routing functions (`backend/agent.py:751-805`). There is no per-node or overall request deadline. Hallucination retries regenerate the same context with the same temperature (`backend/agent.py:441-516`). Answer-quality checker exceptions default to useful at `backend/agent.py:717-725`.

**Impact:** a slow provider can hang a request despite the graph step budget; retries may repeat the same failure; and checker outages fail open. Configurable counters are an improvement but are not equivalent to robust retry control.

**Required change:** propagate an overall deadline, set provider/Qdrant/Redis timeouts, check budget before every node, use LangGraph recursion/deadline limits, and make checker failure produce `unknown`/degraded output rather than an automatic pass. Retries must change retrieval, prompt, or policy, or abstain.

### P1-11 - API security remains incomplete

**Evidence:** CORS is now allowlisted in `backend/main.py:130-143`, and comparison configs are capped at five, but there is still no authentication, authorization, rate limiting, request quota, or trace access control. `/api/traces` returns user queries and generated text. SSE errors return raw exception text at `backend/main.py:369-374`.

**Impact:** any reachable caller can consume LLM/vector resources and read sensitive query/answer traces. CORS is a browser policy, not authentication.

**Required change:** for public deployment add API-key/JWT auth, rate limits, body and timeout limits, safe error IDs, and redacted/authenticated trace endpoints. For a local portfolio demo, bind to localhost or explicitly document that the service is unauthenticated and private.

### P1-12 - Clean Docker still has no data bootstrap path

**Evidence:** Compose creates a fresh named Qdrant volume (`docker-compose.yml:111-129`) and the Dockerfiles copy source but no Qdrant snapshot or ingestion job. The deterministic abstention makes the empty deployment safer, but it still cannot answer until a separate multi-hour ingestion step is performed.

**Impact:** `docker compose up -d` does not produce a usable demo system from a clean clone. The README’s “start the entire stack” flow is incomplete.

**Required change:** provide one documented bootstrap option: a small committed demo snapshot, a versioned Qdrant backup/import, or a compose profile/job that ingests a bounded dataset. Add readiness checks for collection existence and minimum document counts.

### P1-13 - Verification is still not a trustworthy automated test suite

**Evidence:** there is no `tests/` directory, no CI workflow, and no pytest/Ruff dev dependency configuration. `data_pipeline/verify_pipeline.py:344-398` reports L2 sub-check failures but returns `True` unconditionally after the strategy is enabled. The verification process also writes a real test article into the configured collection (`:137-209`).

**Impact:** a green verification run can hide a failed semantic-cache assertion and can mutate the user’s knowledge base. The Makefile advertises `make test` and `make lint`, but a clean environment does not have the required tools declared.

**Required change:** add `pytest`, `pytest-asyncio`, Ruff, and optionally mypy to a dev dependency group; create isolated unit tests; make integration tests opt-in and use a temporary collection/namespace; assert every sub-check; add GitHub Actions for tests, lint, package build, and a Docker smoke test.

### P1-14 - Observability reports are incomplete or inaccurate

**Evidence:** cache-hit responses return at `backend/main.py:235-246` without calling `_record_trace()`, so `/api/metrics` cannot count those hits even though it has a cache-hit metric. The trace ring buffer is process-local. The API uses synchronous Langfuse authentication inside the async health endpoint (`backend/main.py:197-205`).

**Impact:** dashboard cache-hit rate is understated, traces disappear across restarts/worker boundaries, and health checks can block the event loop.

**Required change:** record cache hits with safe metadata, move durable metrics to Prometheus/Langfuse, bound trace payloads and redact sensitive text, and run synchronous health calls off-loop or use an async client.

## P2 findings

### P2-1 - Temporal versioning was removed from Python code, but repository claims and a legacy UI still advertise it

**Evidence:** `ui/script.js:16-73` still defines the Time-Travel control and `as_of_date`; `ui/index.html:67-74` still renders it. `README.md:3`, `README.md:58`, `README.md:79`, and `README.md:182` still describe temporal versioning and an `as_of_date` request. `explanation.md` contains a complete old temporal-versioning section.

**Impact:** a reviewer or user can follow stale documentation and send a request the current API does not model. The project appears internally inconsistent, and an old client can silently lose its intended feature.

**Required change:** remove the legacy time-travel UI, or label it archived; update README, explanation, API examples, diagrams, and feature tables; state clearly that the current system stores the latest article state only. Existing Qdrant data should be rebuilt or cleaned so payloads from the old design do not create ambiguity.

### P2-2 - Dashboard evaluation normalization is incomplete

**Evidence:** `backend/main.py:569-576` normalizes `aggregates` to `aggregate` and `per_query_results` to `per_query`, but `dashboard/js/evaluation.js:50-54` reads `mean_accuracy` and `mean_steps`. The evaluator emits `mean_answer_accuracy` and `mean_step_count` (`evaluation/metrics.py:184-192`).

**Impact:** the evaluation tab can show missing values for accuracy and average steps even after the JSON shape fix.

**Required change:** normalize metric names in one API schema or change the dashboard to consume the evaluator’s canonical names. Add a fixture-based browser/API contract test.

### P2-3 - Dashboard XSS hardening was only partial

**Evidence:** `dashboard/js/evaluation.js` escapes filename and question values, but `dashboard/js/traces.js:108-111`, `:146`, and `:153`, plus `dashboard/js/app.js:125-141`, interpolate query, generation, and expanded-query values into `innerHTML` without escaping.

**Impact:** user queries and model output are untrusted browser content. A malicious query or generated string can execute script in the dashboard origin.

**Required change:** escape every interpolated value, use DOM text nodes, sanitize only where rich Markdown is explicitly intended, validate URL schemes, and add a CSP. Pin external CDN assets.

### P2-4 - `.env.example`, README, and settings use different provider names and defaults

**Evidence:** `backend/config.py:28-32` expects OpenRouter settings, while `.env.example:8-10` and `README.md:120-123` describe OpenAI settings. `.env.example` also omits `QDRANT_MODE`, retry settings, and local path settings.

**Impact:** a clean setup can appear configured while the backend still has an empty LLM API key. The documented install path is not reliable.

**Required change:** generate `.env.example` from the actual settings model, choose one provider naming convention, document required versus optional values, and validate required production settings at startup.

### P2-5 - Dependency and image version policy is still broad

**Evidence:** `pyproject.toml:8-45` uses broad `>=` constraints; Compose uses `qdrant/qdrant:latest` and `redis/redis-stack-server:latest`. `uv.lock` provides a lock for one environment, but source metadata and infrastructure tags still permit unreviewed upgrades.

**Impact:** a fresh install or Docker pull can change behavior without a code change.

**Required change:** use a documented lock/update policy, pin infrastructure image tags or digests, and test upgrades in CI.

### P2-6 - Collection initialization has no schema migration or validation

**Evidence:** `backend/qdrant_client.py:143-147` returns as soon as a collection exists. It does not validate vector dimensions, named vectors, sparse configuration, or required payload indexes.

**Impact:** a stale or incompatible collection can survive code changes and fail later during retrieval, while startup reports success.

**Required change:** add a schema version, inspect collection metadata, fail with an actionable error on mismatch, and provide explicit migrations/rebuild commands.

### P2-7 - Evaluation metrics remain heuristic and the benchmark is too small for broad claims

**Evidence:** `evaluation/metrics.py:45-117` uses answer substring containment for Recall@K, MRR, and accuracy; the recorded runs use ten questions. The harness evaluates final post-grading documents rather than clearly separating candidate retrieval, grading, generation, and abstention errors.

**Impact:** `Recall@5=0.90` and `Accuracy=0.80` are useful smoke-test numbers, not general evidence of Wikipedia-scale quality.

**Required change:** report dataset/title hit rate, citation precision/recall, abstention precision/recall, groundedness, error rate, confidence intervals, cache state, data snapshot, code commit, model, prompt version, and test size. Add a larger fixed evaluation set.

### P2-8 - Prompt and policy versions are not recorded

**Evidence:** prompts remain inline in `backend/agent.py`, `backend/query_expansion.py`, `backend/page_index.py`, and `backend/llmops.py`. Trace metadata records strategy names but not prompt/model/data versions.

**Impact:** benchmark changes cannot be attributed reproducibly to code, prompts, model, or data.

**Required change:** centralize or version prompts, include model/prompt/index versions in trace and evaluation metadata, and record the Git commit in reports.

### P2-9 - Several API contracts are still misleading

**Evidence:** `ChatRequest.session_id` is declared in `backend/models.py:63-66` but is not used; the chat trace creates a timestamp session in `backend/main.py:277-280`. `/api/traces` accepts an unbounded/negative `limit` (`:544-548`). Compare configuration construction omits `knowledge_graph` (`:394-400`).

**Impact:** clients can believe conversation continuity exists when it does not, and A/B results may not represent all available strategy flags.

**Required change:** implement or remove `session_id`, validate and cap trace limits, and build compare strategies from the full `QueryStrategies` schema.

### P2-10 - Global singleton state and file persistence remain weakly isolated

**Evidence:** Qdrant clients, embedding models, reranker, cache strategy, and graph state use module globals. `PipelineHealthTracker._save_dlq()` writes directly to the final JSON path at `data_pipeline/pipeline_health.py:280-287` without a lock or atomic replace.

**Impact:** concurrent first requests can initialize duplicate heavy models; multiple processes have inconsistent state; interruption during DLQ writes can corrupt recovery data.

**Required change:** initialize resources in lifespan/dependency containers with locks, keep the one-worker constraint explicit, use atomic temp-file replacement and a file lock for DLQ persistence, and make worker health stale when heartbeats stop.

## What the remediation successfully fixed

These changes are real improvements and should remain in the final portfolio narrative:

1. Redis client acquisition is inside the cache error boundary.
2. Local Qdrant async calls are adapted through `asyncio.to_thread()`.
3. Python application code no longer contains temporal request filters or `is_current` filtering.
4. Reconciler fake events carry the live revision ID.
5. Compose updater/reconciler services explicitly use remote Qdrant.
6. No-context retrieval returns deterministic abstention and is excluded from normal response caching.
7. Streamlit checks `Content-Type` before parsing JSON versus SSE.
8. Embedding work in hybrid retrieval is offloaded from the event loop.
9. Fetch failures now raise and can enter the DLQ.
10. Docker uses one backend worker, which is appropriate for process-local models and traces.
11. Retry budgets and compare request count are bounded/configurable.
12. The previous answer-quality word-count bypass was removed.
13. Direct `langchain-text-splitters` dependency and the uv-based Makefile were added.
14. Obsolete mobile packaging was removed.

These are fixes, not proof of complete production readiness; each still needs automated regression coverage.

## Temporal-versioning removal assessment

Removing temporal versioning is a defensible simplification for a placement portfolio. It reduces storage complexity, avoids a gimmicky time-travel surface, and makes the product promise clearer: WikiMind answers from the latest synchronized Wikipedia state.

The removal is incomplete at the repository level and creates a required data-integrity obligation:

- Remove old UI controls and all stale documentation/API examples.
- Decide whether the existing Qdrant data is disposable. If it is, document a clean rebuild. If it is not, run a migration that removes obsolete version payloads and stale surplus chunks.
- Use an explicit latest-state replacement workflow so deleted/shortened content does not remain retrievable.
- Keep `revision_id` only as freshness metadata, not as a historical retrieval mechanism.
- Add a “latest-state only” invariant test for update, shrink, delete, and re-ingest cases.

## Recommended implementation order

### Before final portfolio submission

1. Fix the missing `os` import and add a startup smoke test.
2. Fix article replacement deletion and add a shortened-article regression test.
3. Enforce citations/provenance or change the README claim to “citation-aware generation with diagnostic verification.”
4. Remove temporal UI/documentation and perform a clean latest-state data rebuild.
5. Add focused unit tests for cache keys, article scoping, citation parsing, Qdrant update replacement, reconciliation revision handling, and JSON/SSE API behavior.
6. Correct evaluation dashboard metric names and escape all dashboard trace content.

### Before any production claim

1. Add authentication, rate limits, deadlines, safe errors, and trace redaction.
2. Add a reproducible demo-data/bootstrap path and Docker build/smoke CI.
3. Fix revision-source semantics, article-index refresh, PageIndex invalidation, graph freshness, and cache invalidation.
4. Add Qdrant schema validation/migrations and durable metrics.
5. Run concurrent-load, live-sync, reconciliation, Redis-offline, and remote/local Qdrant tests.

## Portfolio wording to use now

> Built and evaluated a Wikipedia-based hybrid RAG prototype with dense+sparse retrieval, article-level discovery, LangGraph orchestration, citation-aware generation, live synchronization workers, caching, and observability. Validated the core query path and selected ingestion components locally; documented remaining hardening work around startup testing, latest-state replacement, citation enforcement, authentication, cache invalidation, and reproducible deployment.

Avoid the unqualified phrases **“production-grade,” “every claim is verified,” “fully self-healing,”** and **“fully article-scoped”** until the findings above are fixed and covered by tests.

## Final acceptance checklist

- [ ] Backend lifespan starts when Guardrails is unavailable.
- [ ] Updating an article to fewer chunks removes all obsolete chunks.
- [ ] Citation absence/invalid references cannot pass as grounded.
- [ ] Article discovery failure and empty-result behavior are distinct and visible.
- [ ] Strategy flags and knowledge-base generation are part of cache identity.
- [ ] Batch snapshot IDs are not mislabeled as MediaWiki revisions.
- [ ] Live updates refresh article index and invalidate derived caches.
- [ ] Local `/health` checks embedded Qdrant correctly.
- [ ] Reconciler samples the full population and records empty/failure cycles.
- [ ] Docker has a reproducible data bootstrap path.
- [ ] `pytest`, lint, and CI run on a clean clone.
- [ ] Dashboard metrics use the evaluator’s actual field names.
- [ ] All browser-rendered query/model content is escaped or sanitized.
- [ ] README, explanation, and legacy UI no longer mention removed temporal versioning.
- [ ] Public API security and timeout policy is explicit and implemented.
