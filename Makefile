.PHONY: dev ingest test lint clean logs stop help

help:
	@echo "WikiMind Tri-Brid RAG Pipeline Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  dev      - Start the full stack using docker-compose (builds images)"
	@echo "  stop     - Stop the docker-compose stack"
	@echo "  ingest   - Run the Wikipedia batch ingestion script locally"
	@echo "  test     - Run the pytest test suite"
	@echo "  lint     - Run ruff for code linting and formatting"
	@echo "  clean    - Remove docker volumes, caches, and orphaned containers"
	@echo "  logs     - Tail the logs for all docker-compose services"

dev:
	docker-compose up --build -d
	@echo "WikiMind stack is starting in the background. Run 'make logs' to view output."

stop:
	docker-compose down

ingest:
	poetry run python -m data_pipeline.ingest

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check .
	poetry run ruff format --check .

clean:
	docker-compose down -v --remove-orphans
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf data/flashrank_cache
	@echo "Cleaned up volumes and caches."

logs:
	docker-compose logs -f
