.PHONY: dev ingest test lint clean logs stop help verify

help:
	@echo "WikiMind Tri-Brid RAG Pipeline Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  dev      - Start the full stack using Docker Compose (builds images)"
	@echo "  stop     - Stop the Docker Compose stack"
	@echo "  ingest   - Run the Wikipedia batch ingestion script locally"
	@echo "  test     - Run the pytest test suite"
	@echo "  lint     - Run ruff for code linting and formatting"
	@echo "  verify   - Run the pipeline verification suite"
	@echo "  clean    - Remove Docker volumes, caches, and orphaned containers"
	@echo "  logs     - Tail the logs for all Docker Compose services"

dev:
	docker compose up --build -d
	@echo "WikiMind stack is starting in the background. Run 'make logs' to view output."

stop:
	docker compose down

ingest:
	python -m data_pipeline.ingest

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check .
	python -m ruff format --check .

verify:
	python -m data_pipeline.verify_pipeline

clean:
	docker compose down -v --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf data/flashrank_cache
	@echo "Cleaned up volumes and caches."

logs:
	docker compose logs -f
