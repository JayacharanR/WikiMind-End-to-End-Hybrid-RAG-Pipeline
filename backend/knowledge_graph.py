"""Knowledge Graph Layer.

Builds and queries a co-occurrence knowledge graph from named entities extracted
from Wikipedia chunks. Uses spaCy for NER and NetworkX for graph representation.
The graph is serialized to Redis for persistence and fast access.

Entities are connected with ``appears_with`` edges when they co-occur in the
same chunk. Graph traversal enables multi-hop reasoning for complex questions
that span multiple articles.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import spacy

from backend.cache import get_redis_client

logger = logging.getLogger(__name__)

# Key for storing the serialized graph in Redis
GRAPH_REDIS_KEY = "wikimind:knowledge_graph"

# Local file fallback for graph persistence (works without Redis)
GRAPH_LOCAL_PATH = os.path.join("data", "knowledge_graph.json")

# spaCy NER entity types to extract
TARGET_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"}

# Lazy-loaded spaCy model
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    """Lazy-load the spaCy language model.

    Uses ``en_core_web_sm`` for speed. In production, consider
    ``en_core_web_trf`` for higher accuracy.
    """
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model: en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


# ---------------------------------------------------------------------------
# Entity Extraction
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> List[str]:
    """Extract named entities from a text chunk.

    Filters to the target entity types defined in ``TARGET_ENTITY_TYPES``
    and returns a deduplicated, sorted list of entity labels.

    Args:
        text: Input text to process.

    Returns:
        Sorted list of unique entity strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    entities: Set[str] = set()
    for ent in doc.ents:
        if ent.label_ in TARGET_ENTITY_TYPES:
            # Normalize: strip, title-case, deduplicate
            normalized = ent.text.strip()
            if len(normalized) > 1:
                entities.add(normalized)

    return sorted(entities)


def build_co_occurrence_edges(
    entities: List[str],
    source_title: str,
    chunk_index: int,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Generate co-occurrence edges between entities found in the same chunk.

    For each pair of entities, creates a weighted edge with provenance
    metadata (source article title and chunk index).

    Args:
        entities: List of entity strings from a single chunk.
        source_title: The Wikipedia article title containing these entities.
        chunk_index: The chunk index within the article.

    Returns:
        List of (entity_a, entity_b, edge_data) tuples.
    """
    edges = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            edge_data = {
                "relation": "appears_with",
                "source_title": source_title,
                "chunk_index": chunk_index,
                "weight": 1.0,
            }
            edges.append((entities[i], entities[j], edge_data))
    return edges


# ---------------------------------------------------------------------------
# Graph Persistence (Redis + Local File Fallback)
# ---------------------------------------------------------------------------


def save_graph_to_file(graph: nx.DiGraph) -> None:
    """Save the knowledge graph to a local JSON file as a fallback.

    This ensures the graph persists even when Redis is unavailable.

    Args:
        graph: The NetworkX directed graph to store.
    """
    try:
        os.makedirs(os.path.dirname(GRAPH_LOCAL_PATH), exist_ok=True)
        data = nx.node_link_data(graph)
        with open(GRAPH_LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(
            "Saved knowledge graph to local file: %s (%d nodes, %d edges).",
            GRAPH_LOCAL_PATH,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
    except Exception as exc:
        logger.error("Failed to save knowledge graph to file: %s", exc)


def load_graph_from_file() -> Optional[nx.DiGraph]:
    """Load the knowledge graph from a local JSON file.

    Returns:
        The deserialized NetworkX DiGraph, or None if the file is missing.
    """
    if not os.path.exists(GRAPH_LOCAL_PATH):
        logger.info("No local knowledge graph file found at %s.", GRAPH_LOCAL_PATH)
        return None

    try:
        with open(GRAPH_LOCAL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data)
        logger.info(
            "Loaded knowledge graph from local file: %d nodes, %d edges.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph
    except Exception as exc:
        logger.error("Failed to load knowledge graph from file: %s", exc)
        return None


async def save_graph_to_redis(graph: nx.DiGraph) -> None:
    """Serialize and store the knowledge graph in Redis AND local file.

    Uses NetworkX's node-link JSON format for compact serialization.
    Always saves to a local JSON file as a fallback. Attempts Redis
    persistence when available.

    Args:
        graph: The NetworkX directed graph to store.
    """
    # Always save to local file first (guaranteed to work)
    save_graph_to_file(graph)

    # Then try Redis
    try:
        client = await get_redis_client()
        data = nx.node_link_data(graph)
        serialized = json.dumps(data)
        await client.set(GRAPH_REDIS_KEY, serialized)
        logger.info(
            "Saved knowledge graph to Redis: %d nodes, %d edges.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
    except Exception as exc:
        logger.warning("Redis save failed (graph is persisted to local file): %s", exc)


async def load_graph_from_redis() -> Optional[nx.DiGraph]:
    """Load the knowledge graph from Redis, with local file fallback.

    Tries Redis first for fastest access. If Redis is unavailable or the
    key doesn't exist, falls back to loading from the local JSON file.

    Returns:
        The deserialized NetworkX DiGraph, or None if not found anywhere.
    """
    # Try Redis first
    try:
        client = await get_redis_client()
        data = await client.get(GRAPH_REDIS_KEY)
        if data is not None:
            graph = nx.node_link_graph(json.loads(data))
            logger.info(
                "Loaded knowledge graph from Redis: %d nodes, %d edges.",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return graph
        logger.info("No knowledge graph found in Redis. Trying local file...")
    except Exception as exc:
        logger.warning("Redis load failed, falling back to local file: %s", exc)

    # Fall back to local file
    return load_graph_from_file()


# ---------------------------------------------------------------------------
# Graph Search
# ---------------------------------------------------------------------------


async def graph_search(
    query: str,
    max_hops: int = 2,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Search the knowledge graph for entities related to the query.

    Extracts entities from the query, then traverses the graph up to
    ``max_hops`` edges away, collecting connected entities and their
    source provenance.

    Args:
        query: The user's search query.
        max_hops: Maximum number of edge hops from seed entities.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with ``entity``, ``connected_to``, ``relation``,
        ``source_title``, and ``hop_distance`` fields.
    """
    # Extract seed entities from the query
    seed_entities = extract_entities(query)

    if not seed_entities:
        logger.debug("No entities extracted from query for graph search.")
        return []

    # Load graph
    graph = await load_graph_from_redis()
    if graph is None:
        logger.warning("Knowledge graph not available. Run graph_builder first.")
        return []

    # BFS traversal from seed entities
    results = []
    visited: Set[str] = set()

    # Match seed entities to graph nodes (case-insensitive)
    node_map = {n.lower(): n for n in graph.nodes()}
    starting_nodes = []
    for entity in seed_entities:
        matched = node_map.get(entity.lower())
        if matched:
            starting_nodes.append(matched)

    if not starting_nodes:
        logger.debug("No seed entities found in the knowledge graph.")
        return []

    # Multi-hop BFS
    frontier = [(node, 0) for node in starting_nodes]
    visited.update(starting_nodes)

    while frontier and len(results) < max_results:
        current_node, distance = frontier.pop(0)

        if distance > 0:
            # Collect edge data from predecessors
            for pred in graph.predecessors(current_node):
                edge_data = graph.get_edge_data(pred, current_node, default={})
                results.append(
                    {
                        "entity": current_node,
                        "connected_to": pred,
                        "relation": edge_data.get("relation", "appears_with"),
                        "source_title": edge_data.get("source_title", ""),
                        "hop_distance": distance,
                    }
                )
                if len(results) >= max_results:
                    break

            for succ in graph.successors(current_node):
                edge_data = graph.get_edge_data(current_node, succ, default={})
                results.append(
                    {
                        "entity": current_node,
                        "connected_to": succ,
                        "relation": edge_data.get("relation", "appears_with"),
                        "source_title": edge_data.get("source_title", ""),
                        "hop_distance": distance,
                    }
                )
                if len(results) >= max_results:
                    break

        if distance < max_hops:
            for neighbor in list(graph.successors(current_node)) + list(
                graph.predecessors(current_node)
            ):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, distance + 1))

    logger.info(
        "Graph search from %s returned %d results (max_hops=%d).",
        seed_entities,
        len(results),
        max_hops,
    )
    return results
