# Observability Architecture

WikiMind uses a **Langfuse-first** observability strategy, purpose-built for LLM/RAG pipeline monitoring.

## Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **LLM Traces** | Langfuse (self-hosted) | Per-node trace waterfalls, token costs, latency, eval scores |
| **HTTP Metrics** | Prometheus (endpoint only) | Basic request/error rate via `/metrics` endpoint |

## Langfuse (Primary)

Self-hosted via `docker-compose.yml` (Langfuse server + PostgreSQL).

- **Dashboard**: http://localhost:3000
- **Login**: admin@wikimind.local / wikimind-admin
- **Traces**: Every agent invocation → expand → retrieve → grade → generate → hallucination check
- **Scores**: Evaluation harness pushes recall, MRR, accuracy after each benchmark run

## Prometheus (Secondary)

The FastAPI app exposes Prometheus metrics at `/metrics` via `prometheus-fastapi-instrumentator`.
This endpoint is production-ready for Grafana/Datadog integration but **no dashboards are bundled** — 
Langfuse handles all RAG-specific observability.

The `prometheus.yml` config file is retained for reference if you want to add a Prometheus scraper.

## Legacy

The `grafana/dashboards/wikimind.json` is a reference Grafana dashboard definition retained for
production deployments that require infrastructure-level monitoring alongside LLM observability.
