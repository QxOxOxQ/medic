from __future__ import annotations

from typing import Any

from qdrant_client.http import models

from rag.measurement.uploader import Uploader, CollectionInfo


class FakeClient:
    def __init__(self) -> None:
        self.models = models
        self._collections: dict[str, Any] = {}
        self.last_upsert: dict[str, Any] | None = None

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, **kwargs: Any) -> None:
        name = kwargs["collection_name"]
        self._collections[name] = kwargs

    def upsert(self, **kwargs: Any) -> None:
        self.last_upsert = kwargs


def test_list_collections_returns_six_choices() -> None:
    up = Uploader(client=FakeClient())
    choices = up.list_collections()
    assert len(choices) == 6
    assert all("name" in c and "slug" in c and "source_url" in c for c in choices)


def test_ensure_collection_creates_with_expected_vector_params() -> None:
    client = FakeClient()
    up = Uploader(client=client)

    up.ensure_collection("ag_news_small", vector_size=8)

    assert client.collection_exists("ag_news_small")
    created = client._collections["ag_news_small"]
    vp = created["vectors_config"]
    assert isinstance(vp, models.VectorParams)
    assert vp.size == 8
    assert vp.distance == models.Distance.COSINE


def test_ensure_collection_creates_multivector_params() -> None:
    client = FakeClient()
    up = Uploader(client=client)

    up.ensure_collection("colbert_collection", vector_size=128, is_multivector=True)

    created = client._collections["colbert_collection"]
    vp = created["vectors_config"]
    assert vp.size == 128
    assert vp.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM


def test_upsert_texts_uses_injected_embed_fn_and_calls_upsert() -> None:
    client = FakeClient()
    # Make sure collection exists to isolate upsert behavior
    up = Uploader(client=client)
    up.ensure_collection("dbpedia_test", vector_size=3)

    def tiny_embed(text: str) -> list[float]:
        # deterministic 3-dim embedding for tests
        return [float(len(text) % 3 == i) for i in range(3)]

    up.embed_fn = tiny_embed
    items = [
        {"text": "alpha", "label": "A"},
        {"text": "beta", "label": "B"},
    ]
    saved = up.upsert_texts(collection_name="dbpedia_test", items=items)

    assert saved == 2
    assert client.last_upsert is not None
    pts = client.last_upsert["points"]
    assert len(pts) == 2
    assert all(hasattr(p, "payload") and hasattr(p, "vector") for p in pts)
    assert all(len(p.vector) == 3 for p in pts)
    assert pts[0].payload == {"label": "A"}


def test_create_and_upsert_returns_collection_info_and_saves_it() -> None:
    client = FakeClient()

    def tiny_embed(text: str) -> list[float]:
        return [1.0, 0.0, 1.0, 0.0]

    up = Uploader(client=client, embed_fn=tiny_embed)
    items = [{"text": "some text", "source": "x"}, {"text": "other", "source": "y"}]

    info = up.create_and_upsert(dataset_name="trec_small", items=items)

    assert isinstance(info, CollectionInfo)
    assert info.collection_name == "trec_small"
    assert info.vector_size == 4
    assert info.count == 2
    assert info.distance == models.Distance.COSINE
    all_infos = up.created_collections_info()
    assert len(all_infos) == 1
    assert all_infos[0] == info


def test_create_and_upsert_handles_1536_dimensions() -> None:
    client = FakeClient()

    def openai_like_embed(text: str) -> list[float]:
        return [0.1] * 1536

    up = Uploader(client=client, embed_fn=openai_like_embed)
    items = [{"text": "heavy text"}]

    info = up.create_and_upsert(dataset_name="msmarco_1536", items=items)

    assert info.vector_size == 1536
    created = client._collections["msmarco_1536"]
    assert created["vectors_config"].size == 1536


def test_create_and_upsert_supports_multivector_embeddings() -> None:
    client = FakeClient()

    def colbert_like_embed(text: str) -> list[list[float]]:
        return [[0.1, 0.2], [0.3, 0.4]]

    up = Uploader(client=client, embed_fn=colbert_like_embed)
    items = [{"text": "clinical text"}]

    info = up.create_and_upsert(dataset_name="colbert_collection", items=items)

    created = client._collections["colbert_collection"]
    point = client.last_upsert["points"][0]
    assert info.vector_size == 2
    assert created["vectors_config"].size == 2
    assert created["vectors_config"].multivector_config.comparator == (
        models.MultiVectorComparator.MAX_SIM
    )
    assert point.vector == [[0.1, 0.2], [0.3, 0.4]]
