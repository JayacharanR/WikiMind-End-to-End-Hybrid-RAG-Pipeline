import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend import main as backend_main
from backend.agent import (
    _validate_citation_refs,
    _verify_citations,
    route_after_hallucination,
)
from backend.cache import _hash_query
from backend.models import ChatRequest, CompareRequest
from backend.qdrant_client import LocalAsyncQdrantAdapter
from data_pipeline.ingest import build_chunk_payload
from data_pipeline.wiki_updater import _remove_stale_article_chunks


class FakeSyncQdrant:
    def __init__(self):
        self.deleted = []

    def delete(self, **kwargs):
        self.deleted.append(kwargs)
        return "completed"


class FakePoint:
    def __init__(self, point_id):
        self.id = point_id


class FakeUpdaterQdrant:
    def __init__(self):
        self.deleted = []

    async def scroll(self, **kwargs):
        return [FakePoint("chunk-0"), FakePoint("chunk-1"), FakePoint("chunk-2")], None

    async def delete(self, **kwargs):
        self.deleted.append(kwargs)


class FakeRedis:
    async def ping(self):
        return True


class FakeCollections:
    collections = ["wikimind_hybrid", "wikimind_articles"]


class FakeSyncHealthQdrant:
    def get_collections(self):
        return FakeCollections()


def test_local_async_adapter_forwards_delete():
    sync_client = FakeSyncQdrant()
    result = asyncio.run(
        LocalAsyncQdrantAdapter(sync_client).delete(
            collection_name="wikimind_hybrid",
            points_selector={"points": ["point-1"]},
        )
    )

    assert result == "completed"
    assert sync_client.deleted[0]["collection_name"] == "wikimind_hybrid"


def test_batch_payload_preserves_dataset_provenance():
    payload = build_chunk_payload(
        {
            "title": "Example",
            "url": "https://en.wikipedia.org/wiki/Example",
            "text": "A chunk",
            "chunk_index": 0,
            "source_document_id": "dataset-123",
            "revision_source": "dataset_snapshot",
        }
    )

    assert payload["source_document_id"] == "dataset-123"
    assert payload["revision_source"] == "dataset_snapshot"
    assert "revision_id" not in payload


def test_latest_state_removes_surplus_chunk_ids_after_upsert():
    qdrant = FakeUpdaterQdrant()
    asyncio.run(
        _remove_stale_article_chunks(
            qdrant,
            "wikimind_hybrid",
            "Example",
            {"chunk-0", "chunk-1"},
        )
    )

    selector = qdrant.deleted[0]["points_selector"]
    assert selector.points == ["chunk-2"]


def test_citation_validation_rejects_out_of_range_references():
    assert _validate_citation_refs("Fact [1]", 1) == []
    assert _validate_citation_refs("Fact [1][99]", 1) == [99]
    assert (
        _verify_citations(
            "Paris is the capital of France [1].",
            {},
            [{"content": "Paris is the capital of France."}],
        )
        == 1.0
    )
    assert (
        _verify_citations(
            "Paris is the capital of France [1]. It has a river.",
            {},
            [{"content": "Paris is the capital of France."}],
        )
        < 1.0
    )


def test_hallucinated_answer_exhaustion_routes_to_abstention():
    state = {
        "hallucination_grade": "hallucinated",
        "hallucination_retries": 99,
        "steps": 1,
    }
    assert route_after_hallucination(state) == "generate_from_web"


def test_cache_keys_are_partitioned_by_strategy():
    default_key = _hash_query("What is Paris?", {})
    expanded_key = _hash_query("What is Paris?", {"multi_query": True})
    assert default_key != expanded_key


def test_request_models_reject_whitespace_queries():
    with pytest.raises(ValidationError):
        ChatRequest(query="   ")
    with pytest.raises(ValidationError):
        CompareRequest(query="\n\t", configs=[{"name": "a"}, {"name": "b"}])


def test_local_health_check_uses_sync_qdrant_adapter(monkeypatch):
    async def fake_redis_client():
        return FakeRedis()

    monkeypatch.setattr(backend_main, "get_redis_client", fake_redis_client)
    monkeypatch.setattr(backend_main, "get_langfuse_client", lambda: None)
    monkeypatch.setattr(
        backend_main,
        "get_settings",
        lambda: SimpleNamespace(qdrant_mode="local"),
    )

    import backend.qdrant_client as qdrant_client

    monkeypatch.setattr(qdrant_client, "get_sync_qdrant", lambda: FakeSyncHealthQdrant())
    response = asyncio.run(backend_main.health_check())

    qdrant_status = next(
        component for component in response.components if component.name == "qdrant"
    )
    assert qdrant_status.healthy is True
    assert "2 collection" in qdrant_status.detail
