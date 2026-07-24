# Observability Architecture

WikiMind uses a **Langfuse-first** observability strategy with a custom-built dashboard.

## Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **LLM Traces** | Langfuse (self-hosted) | Per-node trace waterfalls, token costs, latency, eval scores |
| **Custom Dashboard** | `dashboard/` (HTML/JS/CSS) | Real-time KPIs, trace explorer, guardrails monitor, eval viewer |
| **HTTP Metrics** | Prometheus (endpoint only) | Basic request/error rate via `/metrics` endpoint |

## Custom Dashboard

Accessible at `http://localhost:8000/dashboard/` when the backend is running.

### Tabs:
1. **Overview** — KPI cards (queries, latency, provenance, attribution), latency timeline, attribution donut
2. **Traces** — Full query trace table with expandable details (metadata, citations, generation)
3. **Guardrails** — Applied vs bypassed donut, quality grade breakdown, safety event log
4. **Evaluation** — Benchmark results from the evaluation harness (recall, MRR, accuracy)
5. **System** — Component health (Qdrant, Redis, LLM, Langfuse)

### Backend API:
- `GET /api/metrics` — Aggregated dashboard metrics from in-memory trace log
- `GET /api/traces` — Last N query traces with full detail
- `GET /api/eval-results` — Evaluation benchmark results from `evaluation/results/`

## Langfuse (Primary)

Self-hosted via `docker-compose.yml` (Langfuse server + PostgreSQL).

- **Dashboard**: http://localhost:3000
- **Login**: admin@wikimind.local / wikimind-admin
- **Traces**: Every agent invocation → expand → retrieve → grade → generate → hallucination check
- **Scores**: Evaluation harness pushes recall, MRR, accuracy after each benchmark run

## Prometheus (Secondary)

The FastAPI app exposes Prometheus metrics at `/metrics` via `prometheus-fastapi-instrumentator`.
This endpoint is production-ready for infrastructure monitoring but **no dashboards are bundled** —
the custom dashboard and Langfuse handle all RAG-specific observability.
