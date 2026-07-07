"""Wikipedia Batch Ingestion Script.

Streams the English Wikipedia dataset from Hugging Face, chunks the text,
generates dual embeddings (dense and sparse BM25), and upserts the data
into Qdrant. Also generates article-level embeddings for the two-stage
retrieval architecture (``wikimind_articles`` collection).

Uses checkpointing to resume from the last processed article in case of
failure.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

from backend.config import get_settings
from backend.qdrant_client import generate_point_id, get_sync_qdrant, init_collection

# Configure logging for the batch script
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "data/ingest_checkpoint.json"

# ---------------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------------

_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_dense_model() -> TextEmbedding:
    """Lazy initialize the dense embedding model."""
    global _dense_model
    if _dense_model is None:
        settings = get_settings()
        # For this MVP, we use the local fastembed model. 
        # In a real scenario, this could switch based on config (e.g. OpenAI).
        logger.info("Initializing dense embedding model: %s", settings.embedding_model)
        _dense_model = TextEmbedding(model_name=settings.embedding_model)
    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy initialize the sparse BM25 embedding model."""
    global _sparse_model
    if _sparse_model is None:
        logger.info("Initializing sparse embedding model: Qdrant/bm25")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint() -> int:
    """Load the number of articles processed so far."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
                return data.get("articles_processed", 0)
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s. Starting from 0.", exc)
    return 0


def save_checkpoint(articles_processed: int) -> None:
    """Save the progress checkpoint."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"articles_processed": articles_processed}, f)
    except Exception as exc:
        logger.warning("Failed to save checkpoint: %s", exc)


# ---------------------------------------------------------------------------
# Article-Level Indexing
# ---------------------------------------------------------------------------

def _upsert_article_entries(articles: List[Dict[str, Any]]) -> None:
    """Generate article-level embeddings and upsert to wikimind_articles.

    For each article, concatenates the title with the first two paragraphs
    (capped at 1500 chars) to produce a single dense embedding. This enables
    Stage 1 of the two-stage retrieval architecture.
    """
    from backend.article_index import extract_article_summary, generate_article_id

    settings = get_settings()
    client = get_sync_qdrant()
    dense_model = get_dense_model()

    summaries = []
    article_data = []

    for article in articles:
        title = article.get("title", "Unknown")
        text = article.get("text", "")
        url = article.get("url", "")

        summary_text = extract_article_summary(title, text)
        summaries.append(summary_text)
        article_data.append({
            "id": generate_article_id(title),
            "title": title,
            "url": url,
            "paragraph_preview": summary_text[:500],
        })

    if not summaries:
        return

    # Embed all article summaries in one batch
    embeddings = list(dense_model.embed(summaries))

    # Assemble Qdrant points
    points = []
    for i, data in enumerate(article_data):
        points.append(
            models.PointStruct(
                id=data["id"],
                vector=embeddings[i].tolist(),
                payload={
                    "title": data["title"],
                    "url": data["url"],
                    "paragraph_preview": data["paragraph_preview"],
                },
            )
        )

    client.upsert(
        collection_name=settings.article_collection,
        points=points,
    )
    logger.debug("Upserted %d article-level entries.", len(points))


# ---------------------------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------------------------

def process_batch(articles: List[Dict[str, Any]], articles_only: bool = False) -> None:
    """Process a batch of Wikipedia articles.

    Chunks the text, embeds it, and upserts to Qdrant. Also generates
    article-level embeddings for the two-stage retrieval index.

    Args:
        articles: List of article dicts from the HuggingFace dataset.
        articles_only: If True, only upsert article-level entries (skip chunks).
    """
    if not articles:
        return

    # Always upsert article-level entries
    _upsert_article_entries(articles)

    if articles_only:
        return

    settings = get_settings()
    client = get_sync_qdrant()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", " ", ""],
    )

    points = []
    
    # 1. Chunking + Entity Extraction
    try:
        from backend.knowledge_graph import extract_entities
        do_ner = True
    except Exception:
        logger.debug("spaCy NER unavailable; skipping entity extraction.")
        do_ner = False

    for article in articles:
        title = article.get("title", "Unknown")
        text = article.get("text", "")
        url = article.get("url", "")
        revision_id = str(article.get("id", ""))
        ingested_at = datetime.now(timezone.utc).isoformat()
        
        chunks = splitter.split_text(text)
        
        for i, chunk_text in enumerate(chunks):
            point_id = generate_point_id(title, i)
            entities = extract_entities(chunk_text) if do_ner else []
            points.append({
                "id": point_id,
                "text": chunk_text,
                "title": title,
                "url": url,
                "chunk_index": i,
                "entities": entities,
                "revision_id": revision_id,
                "ingested_at": ingested_at,
            })

    if not points:
        return

    # 2. Embedding
    texts_to_embed = [p["text"] for p in points]
    
    # Dense embeddings
    dense_embeddings = list(dense_model.embed(texts_to_embed))
    
    # Sparse embeddings
    sparse_embeddings = list(sparse_model.embed(texts_to_embed))

    # 3. Assemble Qdrant Points
    qdrant_points = []
    for i, point_data in enumerate(points):
        # Sparse embedding is an object with indices and values
        sparse_obj = sparse_embeddings[i]
        
        vector_dict = {
            "dense": dense_embeddings[i].tolist(),
            "sparse": models.SparseVector(
                indices=sparse_obj.indices.tolist(),
                values=sparse_obj.values.tolist(),
            )
        }
        
        payload = {
            "title": point_data["title"],
            "url": point_data["url"],
            "page_content": point_data["text"],
            "chunk_index": point_data["chunk_index"],
            "entities": point_data.get("entities", []),
            "revision_id": point_data.get("revision_id", ""),
            "ingested_at": point_data.get("ingested_at", ""),
            "is_current": True,
        }
        
        qdrant_points.append(
            models.PointStruct(
                id=point_data["id"],
                vector=vector_dict,
                payload=payload,
            )
        )

    # 4. Upsert
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=qdrant_points
    )
    
    logger.debug("Upserted %d chunks from %d articles.", len(qdrant_points), len(articles))


def run_ingestion(max_articles: int = 1000, batch_size: int = 50, articles_only: bool = False) -> None:
    """Run the ingestion pipeline.

    Streams the Wikipedia dataset, processing it in batches. Resumes from the
    last saved checkpoint.

    Args:
        max_articles: Maximum number of articles to process (0 for unlimited).
        batch_size: Number of articles per batch.
        articles_only: If True, only build the article-level index (skip chunks).
    """
    from backend.article_index import init_article_collection

    init_collection()
    init_article_collection()
    
    processed_count = load_checkpoint()
    logger.info("Starting ingestion. Resuming from %d articles.", processed_count)
    
    # Load wikipedia dataset in streaming mode
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    # Skip already processed
    if processed_count > 0:
        logger.info("Skipping first %d articles...", processed_count)
        # Note: dataset.skip() can be slow for large numbers on streaming datasets
        # A more robust implementation would use a sharded dataset or date-based filtering
        dataset = dataset.skip(processed_count)

    batch = []
    start_time = time.monotonic()
    
    try:
        for article in dataset:
            batch.append(article)
            
            if len(batch) >= batch_size:
                process_batch(batch, articles_only=articles_only)
                processed_count += len(batch)
                save_checkpoint(processed_count)
                
                elapsed = time.monotonic() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                logger.info("Processed %d articles. Rate: %.2f articles/sec", processed_count, rate)
                
                batch = []
                
                if max_articles > 0 and processed_count >= max_articles:
                    logger.info("Reached maximum requested articles (%d). Stopping.", max_articles)
                    break
                    
        # Process any remaining
        if batch and (max_articles <= 0 or processed_count < max_articles):
            process_batch(batch, articles_only=articles_only)
            processed_count += len(batch)
            save_checkpoint(processed_count)
            logger.info("Processed final batch. Total: %d articles.", processed_count)
            
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user. Saved checkpoint at %d.", processed_count)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WikiMind Wikipedia Ingestion Script")
    parser.add_argument("--max", type=int, default=1000, help="Maximum number of articles to process (0 for unlimited)")
    parser.add_argument("--batch", type=int, default=50, help="Number of articles per batch")
    parser.add_argument("--articles-only", action="store_true", help="Only build the article-level index (skip chunk embedding)")
    args = parser.parse_args()
    
    run_ingestion(max_articles=args.max, batch_size=args.batch, articles_only=args.articles_only)
