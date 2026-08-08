"""Knowledge Graph Builder.

Batch script that scrolls all chunks from the ``wikimind_hybrid`` Qdrant
collection, extracts named entities via spaCy NER, builds a co-occurrence
graph using NetworkX, and persists the result in Redis.

Supports incremental updates by merging new edges into an existing graph.

Usage::

    python -m data_pipeline.graph_builder
    python -m data_pipeline.graph_builder --scroll-limit 10000
"""

import argparse
import asyncio
import logging
import time
from typing import Any, Dict, List

import networkx as nx

from backend.config import get_settings
from backend.knowledge_graph import (
    build_co_occurrence_edges,
    extract_entities,
    load_graph_from_redis,
    save_graph_to_redis,
)
from backend.qdrant_client import get_sync_qdrant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _scroll_all_chunks(limit: int = 0) -> List[Dict[str, Any]]:
    """Scroll through all chunk points in the wikimind_hybrid collection.

    Args:
        limit: Maximum number of points to retrieve. 0 for unlimited.

    Returns:
        List of dicts with ``title``, ``page_content``, ``chunk_index``.
    """
    settings = get_settings()
    client = get_sync_qdrant()

    chunks = []
    offset = None
    batch_size = 100
    total = 0

    while True:
        result = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = result

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            chunks.append(
                {
                    "title": payload.get("title", ""),
                    "page_content": payload.get("page_content", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                }
            )
            total += 1

            if limit > 0 and total >= limit:
                break

        if limit > 0 and total >= limit:
            break

        offset = next_offset
        if offset is None:
            break

        if total % 1000 == 0:
            logger.info("Scrolled %d chunks...", total)

    logger.info("Total chunks retrieved: %d", total)
    return chunks


def _build_graph_from_chunks(chunks: List[Dict[str, Any]]) -> nx.DiGraph:
    """Process chunks and build a co-occurrence knowledge graph.

    For each chunk, extracts entities using spaCy NER, then generates
    co-occurrence edges between all entity pairs in the same chunk.

    Args:
        chunks: List of chunk dicts from Qdrant.

    Returns:
        NetworkX DiGraph with entities as nodes and co-occurrence edges.
    """
    graph = nx.DiGraph()
    total_entities = 0
    total_edges = 0

    for i, chunk in enumerate(chunks):
        text = chunk.get("page_content", "")
        title = chunk.get("title", "")
        chunk_index = chunk.get("chunk_index", 0)

        if not text:
            continue

        # Extract entities
        entities = extract_entities(text)

        if not entities:
            continue

        total_entities += len(entities)

        # Add entity nodes with metadata
        for entity in entities:
            if not graph.has_node(entity):
                graph.add_node(entity, count=0, sources=[])
            graph.nodes[entity]["count"] += 1
            # Track sources (limit to avoid memory bloat)
            sources = graph.nodes[entity]["sources"]
            if len(sources) < 10:
                sources.append(title)

        # Build co-occurrence edges
        edges = build_co_occurrence_edges(entities, title, chunk_index)
        for src, dst, data in edges:
            if graph.has_edge(src, dst):
                # Increment weight for existing edges
                graph.edges[src, dst]["weight"] += 1.0
            else:
                graph.add_edge(src, dst, **data)
            total_edges += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "Processed %d/%d chunks | %d entities | %d edges",
                i + 1,
                len(chunks),
                total_entities,
                total_edges,
            )

    logger.info(
        "Graph build complete: %d nodes, %d edges (from %d entity extractions).",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        total_entities,
    )
    return graph


async def build_knowledge_graph(
    scroll_limit: int = 0,
    incremental: bool = True,
) -> nx.DiGraph:
    """Build the knowledge graph from Qdrant chunks.

    If ``incremental`` is True, loads the existing graph from Redis and
    merges new edges into it. Otherwise, builds from scratch.

    Args:
        scroll_limit: Maximum chunks to process (0 for all).
        incremental: Whether to merge into existing graph.

    Returns:
        The completed knowledge graph.
    """
    start = time.monotonic()

    # Scroll chunks
    logger.info("Scrolling chunks from Qdrant (limit=%d)...", scroll_limit)
    chunks = _scroll_all_chunks(limit=scroll_limit)

    if not chunks:
        logger.warning("No chunks found in Qdrant. Build the chunk index first.")
        return nx.DiGraph()

    # Build new graph from chunks
    new_graph = _build_graph_from_chunks(chunks)

    # Incremental merge
    if incremental:
        existing = await load_graph_from_redis()
        if existing is not None:
            logger.info("Merging new graph into existing (%d nodes)...", existing.number_of_nodes())
            existing = nx.compose(existing, new_graph)
            new_graph = existing

    # Save to Redis
    await save_graph_to_redis(new_graph)

    elapsed = time.monotonic() - start
    logger.info(
        "Knowledge graph saved: %d nodes, %d edges in %.1fs.",
        new_graph.number_of_nodes(),
        new_graph.number_of_edges(),
        elapsed,
    )
    return new_graph


def main():
    """CLI entry point for the graph builder."""
    parser = argparse.ArgumentParser(
        description="WikiMind Knowledge Graph Builder",
    )
    parser.add_argument(
        "--scroll-limit",
        type=int,
        default=0,
        help="Maximum number of chunks to process (0 for all).",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Build from scratch instead of merging with existing graph.",
    )

    args = parser.parse_args()

    asyncio.run(
        build_knowledge_graph(
            scroll_limit=args.scroll_limit,
            incremental=not args.no_incremental,
        )
    )


if __name__ == "__main__":
    main()
