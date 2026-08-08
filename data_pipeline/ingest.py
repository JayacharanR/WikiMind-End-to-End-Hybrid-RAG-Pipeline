"""Wikipedia Batch Ingestion Script.

Streams the English Wikipedia dataset from Hugging Face, chunks the text,
generates dual embeddings (dense and sparse BM25), and upserts the data
into Qdrant. Also generates article-level embeddings for the two-stage
retrieval architecture (``wikimind_articles`` collection).

Uses checkpointing to resume from the last processed article in case of
failure.
"""

# Imports intentionally occur after environment/cache setup below so FastEmbed
# and Hugging Face read the project-local cache locations.
# ruff: noqa: E402

import argparse
import json
import logging
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import queue
import threading

# Optional: inject cuDNN DLLs for ONNX Runtime GPU acceleration on Windows
if os.name == "nt":
    cudnn_path = os.environ.get("CUDNN_PATH", "")
    if cudnn_path and os.path.exists(cudnn_path):
        os.environ["PATH"] = cudnn_path + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(cudnn_path)

# Set up cache directories (portable: uses env var or project-relative default)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.environ.get("WIKIMIND_CACHE_DIR", os.path.join(_project_root, "data", "cache"))
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))
os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(cache_dir, "fastembed"))

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from datasets import load_dataset
from fastembed import SparseTextEmbedding, TextEmbedding
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
        logger.info("Initializing dense embedding model: %s", settings.embedding_model)
        try:
            _dense_model = TextEmbedding(
                model_name=settings.embedding_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as exc:
            logger.warning("CUDA embedding provider unavailable (%s); using CPU.", exc)
            _dense_model = TextEmbedding(
                model_name=settings.embedding_model,
                providers=["CPUExecutionProvider"],
            )
    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy initialize the sparse BM25 embedding model."""
    global _sparse_model
    if _sparse_model is None:
        logger.info("Initializing sparse embedding model: Qdrant/bm25")
        try:
            _sparse_model = SparseTextEmbedding(
                model_name="Qdrant/bm25",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as exc:
            logger.warning("CUDA sparse provider unavailable (%s); using CPU.", exc)
            _sparse_model = SparseTextEmbedding(
                model_name="Qdrant/bm25", providers=["CPUExecutionProvider"]
            )
    return _sparse_model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def load_checkpoint() -> dict:
    """Load the checkpoint with full session metadata."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
                # Backwards compatibility: old checkpoints only had articles_processed
                if isinstance(data.get("articles_processed"), int) and "total_chunks" not in data:
                    data["total_chunks"] = 0
                    data["last_article_title"] = ""
                    data["sessions"] = []
                return data
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s. Starting from 0.", exc)
    return {
        "articles_processed": 0,
        "total_chunks": 0,
        "last_article_title": "",
        "sessions": [],
    }


def save_checkpoint(
    articles_processed: int,
    total_chunks: int = 0,
    last_title: str = "",
    session_info: dict | None = None,
) -> None:
    """Save the progress checkpoint with detailed metadata."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    try:
        # Load existing to preserve session history
        existing = load_checkpoint()
        existing["articles_processed"] = articles_processed
        existing["total_chunks"] = total_chunks
        existing["last_article_title"] = last_title
        existing["last_updated"] = datetime.now(timezone.utc).isoformat()

        if session_info:
            sessions = existing.get("sessions", [])
            sessions.append(session_info)
            # Keep last 20 sessions
            existing["sessions"] = sessions[-20:]

        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(existing, f, indent=2)
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
        article_data.append(
            {
                "id": generate_article_id(title),
                "title": title,
                "url": url,
                "paragraph_preview": summary_text[:500],
            }
        )

    if not summaries:
        return

    # Embed all article summaries in one batch
    embeddings = list(dense_model.embed(summaries, batch_size=256))

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

    client.upload_points(
        collection_name=settings.article_collection,
        points=points,
        # Checkpointing occurs after process_batch returns, so the upload must
        # be acknowledged before the batch is considered resumable.
        wait=True,
        batch_size=256,
        max_retries=3,
    )
    logger.debug("Upserted %d article-level entries.", len(points))


def prefetch_generator(generator, max_prefetch=5000):
    """Run a generator in a background thread to prevent I/O blocking."""
    q = queue.Queue(maxsize=max_prefetch)

    def worker():
        try:
            for item in generator:
                q.put(item)
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
        finally:
            q.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item


# ---------------------------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------------------------


def process_batch(
    articles: List[Dict[str, Any]],
    articles_only: bool = False,
    skip_ner: bool = False,
) -> int:
    """Process a batch of Wikipedia articles.

    Chunks the text, embeds it, and upserts to Qdrant. Also generates
    article-level embeddings for the two-stage retrieval index.

    Args:
        articles: List of article dicts from the HuggingFace dataset.
        articles_only: If True, only upsert article-level entries (skip chunks).
        skip_ner: If True, skip spaCy NER entity extraction (faster ingestion).

    Returns:
        Number of chunks processed in this batch.
    """
    if not articles:
        return 0

    # Always upsert article-level entries
    _upsert_article_entries(articles)

    if articles_only:
        return 0

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
    do_ner = False
    if not skip_ner:
        try:
            from backend.knowledge_graph import extract_entities

            do_ner = True
        except Exception:
            logger.debug("spaCy NER unavailable; skipping entity extraction.")

    for article in articles:
        title = article.get("title", "Unknown")
        text = article.get("text", "")
        url = article.get("url", "")
        revision_id = str(article.get("id", ""))
        # HF dataset 'id' is a document identifier, NOT a MediaWiki revision.
        # Mark it as such so the reconciler doesn't false-alarm.
        revision_source = "dataset_snapshot"
        ingested_at = datetime.now(timezone.utc).isoformat()

        chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(chunks):
            point_id = generate_point_id(title, i)
            entities = extract_entities(chunk_text) if do_ner else []
            points.append(
                {
                    "id": point_id,
                    "text": chunk_text,
                    "title": title,
                    "url": url,
                    "chunk_index": i,
                    "entities": entities,
                    "source_document_id": revision_id,
                    "revision_source": revision_source,
                    "ingested_at": ingested_at,
                }
            )

    if not points:
        return 0

    # 2. Embedding
    texts_to_embed = [p["text"] for p in points]

    # Dense embeddings
    dense_embeddings = list(dense_model.embed(texts_to_embed, batch_size=256))

    # Sparse embeddings
    sparse_embeddings = list(sparse_model.embed(texts_to_embed, batch_size=256))

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
            ),
        }

        payload = build_chunk_payload(point_data)

        qdrant_points.append(
            models.PointStruct(
                id=point_data["id"],
                vector=vector_dict,
                payload=payload,
            )
        )

    # 4. Upsert (upload_points automatically handles batching and multithreading!)
    client.upload_points(
        collection_name=settings.qdrant_collection,
        points=qdrant_points,
        # Do not advance the ingestion checkpoint before Qdrant acknowledges
        # the batch.
        wait=True,
        batch_size=256,
        max_retries=3,
    )

    logger.debug("Upserted %d chunks from %d articles.", len(qdrant_points), len(articles))
    return len(qdrant_points)


def build_chunk_payload(point_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the persisted payload for one batch-ingested chunk."""
    return {
        "title": point_data["title"],
        "url": point_data["url"],
        "page_content": point_data["text"],
        "chunk_index": point_data["chunk_index"],
        "entities": point_data.get("entities", []),
        # Hugging Face's id is a dataset document id, not a MediaWiki
        # revision. Preserve its provenance explicitly for reconciliation.
        "source_document_id": point_data.get("source_document_id", ""),
        "revision_source": point_data.get("revision_source", "dataset_snapshot"),
        "ingested_at": point_data.get("ingested_at", ""),
    }


# ---------------------------------------------------------------------------
# Live Progress Display
# ---------------------------------------------------------------------------


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def _get_dir_size_mb(path: str) -> float:
    """Get the size of a directory in MB."""
    total = 0
    if os.path.exists(path):
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total / (1024 * 1024)


def _print_progress(
    articles_done: int,
    articles_target: int,
    chunks_done: int,
    articles_per_sec: float,
    chunks_per_sec: float,
    elapsed: float,
    eta_seconds: float,
    last_title: str,
    qdrant_path: str,
) -> None:
    """Print a live progress dashboard to the terminal."""
    # Progress bar
    if articles_target > 0:
        pct = min(articles_done / articles_target * 100, 100)
        bar_width = 40
        filled = int(bar_width * articles_done / articles_target)
        bar = "█" * filled + "░" * (bar_width - filled)
    else:
        pct = 0
        bar = "~" * 40

    # Qdrant storage size
    storage_mb = _get_dir_size_mb(qdrant_path)
    if storage_mb > 1024:
        storage_str = f"{storage_mb / 1024:.1f} GB"
    else:
        storage_str = f"{storage_mb:.0f} MB"

    # Build output
    lines = [
        "",
        "  ┌─────────────────────────────────────────────────────┐",
        "  │  📚 WikiMind Offline Embedding Progress             │",
        "  ├─────────────────────────────────────────────────────┤",
        f"  │  [{bar}] {pct:5.1f}%  │",
        "  │                                                     │",
        f"  │  Articles:  {articles_done:>8,} / {articles_target:>8,}                │",
        f"  │  Chunks:    {chunks_done:>8,}                              │",
        f"  │  Speed:     {articles_per_sec:>6.1f} articles/sec | {chunks_per_sec:>6.0f} chunks/sec │",
        f"  │  Elapsed:   {_format_time(elapsed):<12s}                         │",
        f"  │  ETA:       {_format_time(eta_seconds) if eta_seconds > 0 else 'calculating...':<12s}                         │",
        f"  │  Storage:   {storage_str:<12s}                         │",
        "  │                                                     │",
        f"  │  Last: {last_title[:45]:<45s} │",
        "  └─────────────────────────────────────────────────────┘",
    ]

    # Move cursor up and overwrite
    output = "\r" + "\n".join(lines)
    sys.stdout.write(f"\033[{len(lines)}A" + output)
    sys.stdout.flush()


def _print_header(
    max_articles: int,
    batch_size: int,
    skip_ner: bool,
    checkpoint_articles: int,
    qdrant_mode: str,
) -> None:
    """Print the startup header."""
    print("\n" + "=" * 57)
    print("  🧠 WikiMind — Offline Wikipedia Embedding Pipeline")
    print("=" * 57)
    print(f"  Target articles:   {max_articles if max_articles > 0 else 'unlimited':>10}")
    print(f"  Batch size:        {batch_size:>10}")
    print(f"  NER extraction:    {'enabled' if not skip_ner else 'disabled':>10}")
    print(f"  Qdrant mode:       {qdrant_mode:>10}")
    print(f"  Resuming from:     {checkpoint_articles:>10} articles")
    if checkpoint_articles > 0:
        print(f"  ✅ Found checkpoint — will skip first {checkpoint_articles} articles")
    print("=" * 57)
    print()
    # Print blank lines that the progress display will overwrite
    for _ in range(16):
        print()


def _print_final_report(
    articles_this_session: int,
    chunks_this_session: int,
    total_articles: int,
    elapsed: float,
    qdrant_path: str,
    interrupted: bool = False,
) -> None:
    """Print the final summary report."""
    storage_mb = _get_dir_size_mb(qdrant_path)
    if storage_mb > 1024:
        storage_str = f"{storage_mb / 1024:.1f} GB"
    else:
        storage_str = f"{storage_mb:.0f} MB"

    rate = articles_this_session / elapsed if elapsed > 0 else 0

    print("\n\n")
    print("=" * 57)
    if interrupted:
        print("  ⚠️  Ingestion PAUSED (checkpoint saved)")
    else:
        print("  ✅  Ingestion COMPLETE")
    print("=" * 57)
    print(f"  Articles this session:  {articles_this_session:>10,}")
    print(f"  Chunks this session:    {chunks_this_session:>10,}")
    print(f"  Total articles (all):   {total_articles:>10,}")
    print(f"  Avg speed:              {rate:>10.1f} articles/sec")
    print(f"  Elapsed time:           {_format_time(elapsed):>10}")
    print(f"  Qdrant storage:         {storage_str:>10}")
    print(f"  Checkpoint:             {CHECKPOINT_FILE}")
    print("=" * 57)
    if interrupted:
        print("\n  To resume, run the same command again.")
        print("  The script will automatically skip already-processed articles.\n")


# ---------------------------------------------------------------------------
# Main Ingestion Runner
# ---------------------------------------------------------------------------


def run_ingestion(
    max_articles: int = 1000,
    batch_size: int = 50,
    articles_only: bool = False,
    skip_ner: bool = False,
) -> None:
    """Run the ingestion pipeline.

    Streams the Wikipedia dataset, processing it in batches. Resumes from the
    last saved checkpoint.

    Args:
        max_articles: Maximum number of articles to process (0 for unlimited).
        batch_size: Number of articles per batch.
        articles_only: If True, only build the article-level index (skip chunks).
        skip_ner: If True, skip spaCy NER entity extraction (faster ingestion).
    """
    from backend.article_index import init_article_collection

    settings = get_settings()

    init_collection()
    init_article_collection()

    checkpoint = load_checkpoint()
    processed_count = checkpoint.get("articles_processed", 0)
    total_chunks = checkpoint.get("total_chunks", 0)

    # Print startup header
    _print_header(
        max_articles=max_articles,
        batch_size=batch_size,
        skip_ner=skip_ner,
        checkpoint_articles=processed_count,
        qdrant_mode=settings.qdrant_mode,
    )

    target_label = str(max_articles) if max_articles > 0 else "unlimited"
    logger.info(
        "Starting ingestion. Target: %s articles. Resuming from %d. NER: %s.",
        target_label,
        processed_count,
        "off" if skip_ner else "on",
    )

    # Load wikipedia dataset in streaming mode
    logger.info("Loading Wikipedia dataset from HuggingFace (streaming mode)...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    # Skip already processed
    if processed_count > 0:
        logger.info("Skipping first %d articles (resuming from checkpoint)...", processed_count)
        dataset = dataset.skip(processed_count)

    # Wrap the dataset in a background prefetch thread to prevent network I/O from starving the GPU
    dataset = prefetch_generator(dataset, max_prefetch=2000)

    batch = []
    start_time = time.monotonic()
    articles_this_session = 0
    chunks_this_session = 0
    last_title = checkpoint.get("last_article_title", "")

    try:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures_queue = []

            def process_completed(f_done, batch_done):
                nonlocal processed_count, articles_this_session, chunks_this_session
                nonlocal total_chunks, last_title, start_time

                batch_chunks = f_done.result()
                processed_count += len(batch_done)
                articles_this_session += len(batch_done)
                chunks_this_session += batch_chunks
                total_chunks += batch_chunks
                last_title = batch_done[-1].get("title", "")

                save_checkpoint(
                    articles_processed=processed_count,
                    total_chunks=total_chunks,
                    last_title=last_title,
                )

                elapsed = time.monotonic() - start_time
                rate = articles_this_session / elapsed if elapsed > 0 else 0
                chunk_rate = chunks_this_session / elapsed if elapsed > 0 else 0

                if max_articles > 0 and rate > 0:
                    remaining = max_articles - processed_count
                    eta_seconds = remaining / rate
                else:
                    eta_seconds = -1

                _print_progress(
                    articles_done=processed_count,
                    articles_target=max_articles if max_articles > 0 else processed_count,
                    chunks_done=total_chunks,
                    articles_per_sec=rate,
                    chunks_per_sec=chunk_rate,
                    elapsed=elapsed,
                    eta_seconds=eta_seconds,
                    last_title=last_title,
                    qdrant_path=settings.qdrant_local_path,
                )

                if max_articles > 0:
                    logger.info(
                        "[%d/%s] %.1f articles/sec | %d chunks | ETA: %s",
                        processed_count,
                        target_label,
                        rate,
                        total_chunks,
                        _format_time(eta_seconds) if eta_seconds > 0 else "N/A",
                    )
                else:
                    logger.info(
                        "[%d/%s] %.1f articles/sec | %d chunks",
                        processed_count,
                        target_label,
                        rate,
                        total_chunks,
                    )

            for article in dataset:
                batch.append(article)

                if len(batch) >= batch_size:
                    f = executor.submit(process_batch, list(batch), articles_only, skip_ner)
                    futures_queue.append((f, list(batch)))
                    batch = []

                    if len(futures_queue) >= 4:
                        f_done, batch_done = futures_queue.pop(0)
                        process_completed(f_done, batch_done)

                        if max_articles > 0 and processed_count >= max_articles:
                            logger.info("Reached target (%d articles). Done.", max_articles)
                            break

            # Flush remaining inflight futures
            for f_done, batch_done in futures_queue:
                if max_articles > 0 and processed_count >= max_articles:
                    break
                process_completed(f_done, batch_done)

        # Process any remaining
        if batch and (max_articles <= 0 or processed_count < max_articles):
            batch_chunks = process_batch(batch, articles_only=articles_only, skip_ner=skip_ner)
            processed_count += len(batch)
            articles_this_session += len(batch)
            chunks_this_session += batch_chunks
            total_chunks += batch_chunks
            last_title = batch[-1].get("title", "") if batch else last_title

            save_checkpoint(
                articles_processed=processed_count,
                total_chunks=total_chunks,
                last_title=last_title,
                session_info={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "articles": articles_this_session,
                    "chunks": chunks_this_session,
                    "duration_sec": round(time.monotonic() - start_time, 1),
                },
            )
            logger.info("Processed final batch. Total: %d articles.", processed_count)

    except KeyboardInterrupt:
        # Save checkpoint on Ctrl+C
        save_checkpoint(
            articles_processed=processed_count,
            total_chunks=total_chunks,
            last_title=last_title,
            session_info={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "articles": articles_this_session,
                "chunks": chunks_this_session,
                "duration_sec": round(time.monotonic() - start_time, 1),
                "interrupted": True,
            },
        )
        elapsed = time.monotonic() - start_time
        _print_final_report(
            articles_this_session=articles_this_session,
            chunks_this_session=chunks_this_session,
            total_articles=processed_count,
            elapsed=elapsed,
            qdrant_path=settings.qdrant_local_path,
            interrupted=True,
        )
        return

    except Exception as exc:
        save_checkpoint(
            articles_processed=processed_count,
            total_chunks=total_chunks,
            last_title=last_title,
        )
        logger.error("Ingestion failed at article %d: %s", processed_count, exc)
        raise

    elapsed = time.monotonic() - start_time

    # Save final session info
    save_checkpoint(
        articles_processed=processed_count,
        total_chunks=total_chunks,
        last_title=last_title,
        session_info={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "articles": articles_this_session,
            "chunks": chunks_this_session,
            "duration_sec": round(elapsed, 1),
            "completed": True,
        },
    )

    _print_final_report(
        articles_this_session=articles_this_session,
        chunks_this_session=chunks_this_session,
        total_articles=processed_count,
        elapsed=elapsed,
        qdrant_path=settings.qdrant_local_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WikiMind Wikipedia Ingestion Script")
    parser.add_argument(
        "--max",
        type=int,
        default=1000,
        help="Maximum number of articles to process (0 for unlimited)",
    )
    parser.add_argument("--batch", type=int, default=50, help="Number of articles per batch")
    parser.add_argument(
        "--articles-only",
        action="store_true",
        help="Only build the article-level index (skip chunk embedding)",
    )
    parser.add_argument(
        "--skip-ner",
        action="store_true",
        help="Skip spaCy NER entity extraction (faster ingestion)",
    )
    args = parser.parse_args()

    run_ingestion(
        max_articles=args.max,
        batch_size=args.batch,
        articles_only=args.articles_only,
        skip_ner=args.skip_ner,
    )
