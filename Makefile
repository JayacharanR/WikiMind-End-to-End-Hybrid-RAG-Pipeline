.PHONY: dev ingest test lint clean logs stop help verify

help:
	@echo "WikiMind Tri-Brid RAG Pipeline Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  dev      - Start the full stack using docker-compose (builds images)"
	@echo "  stop     - Stop the docker-compose stack"
	@echo "  ingest   - Run the Wikipedia batch ingestion script locally"
	@echo "  test     - Run the pytest test suite"
	@echo "  lint     - Run ruff for code linting and formatting"
	@echo "  verify   - Run the pipeline verification suite"
	@echo "  clean    - Remove docker volumes, caches, and orphaned containers"
	@echo "  logs     - Tail the logs for all docker-compose services"

dev:
	docker-compose up --build -d
	@echo "WikiMind stack is starting in the background. Run 'make logs' to view output."

stop:
	docker-compose down

ingest:
	uv run python -m data_pipeline.ingest

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .

verify:
	uv run python -m data_pipeline.verify_pipeline

clean:
	docker-compose down -v --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf data/flashrank_cache
	@echo "Cleaned up volumes and caches."

logs:
	docker-compose logs -f
