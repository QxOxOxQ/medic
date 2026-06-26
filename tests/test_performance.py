from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from rag.measurement.performance import DEFAULT_TEST_QUERIES, Performance, run_benchmark
from rag.measurement.quantization_config import quantization_configs
from rag.config import SETTINGS

DENSE_VECTOR_NAME = SETTINGS["qdrant"]["dense_vector"]["name"]


@dataclass
class Point:
    id: int
    vector: Any
    payload: dict[str, Any]


@dataclass
class QueryResult:
    points: list[Point]


class RecordingClient:
    models = models

    def __init__(
        self,
        collection_infos: dict[str, Any],
        points_by_collection: dict[str, list[Point]] | None = None,
        existing_collections: set[str] | None = None,
        query_results: list[QueryResult] | None = None,
    ) -> None:
        self.collection_infos = collection_infos
        self.points_by_collection = points_by_collection or {}
        self.existing_collections = existing_collections or set()
        self.query_results = query_results or []
        self.query_calls: list[dict[str, Any]] = []
        self.create_collection_calls: list[dict[str, Any]] = []
        self.upload_points_calls: list[dict[str, Any]] = []
        self.delete_collection_calls: list[str] = []
        self.scroll_calls: list[dict[str, Any]] = []

    def query_points(self, **kwargs: Any) -> QueryResult:
        self.query_calls.append(kwargs)
        if self.query_results:
            return self.query_results.pop(0)
        return QueryResult(points=[])

    def create_collection(self, **kwargs: Any) -> None:
        self.create_collection_calls.append(kwargs)

    def upload_points(self, **kwargs: Any) -> None:
        self.upload_points_calls.append(kwargs)

    def get_collection(self, collection_name: str) -> Any:
        return self.collection_infos[collection_name]

    def scroll(self, **kwargs: Any) -> tuple[list[Point], None]:
        self.scroll_calls.append(kwargs)
        return list(self.points_by_collection.get(kwargs["collection_name"], [])), None

    def delete_collection(self, collection_name: str) -> None:
        self.delete_collection_calls.append(collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.existing_collections


def test_performance_defaults_come_from_settings_hash() -> None:
    performance_settings = SETTINGS["measurement"]["performance"]

    assert Performance.scroll_batch_size == performance_settings["scroll_batch_size"]
    assert Performance.upload_batch_size == performance_settings["upload_batch_size"]
    assert Performance.upload_parallel == performance_settings["upload_parallel"]
    assert Performance.search_limit == performance_settings["search_limit"]
    assert set(quantization_configs) == set(
        SETTINGS["measurement"]["quantization_configs"]
    )


def test_default_test_queries_cover_medical_domains() -> None:
    joined_queries = " ".join(DEFAULT_TEST_QUERIES).lower()

    assert len(DEFAULT_TEST_QUERIES) == 20
    assert "badania krwi" in joined_queries
    assert "cholesterol" in joined_queries
    assert "kolana" in joined_queries
    assert "neurolog" in joined_queries


def test_measure_search_performance_uses_named_dense_vector() -> None:
    client = RecordingClient(
        collection_infos={
            "documents": collection_info(
                vectors_config={DENSE_VECTOR_NAME: vector_params(size=3)},
            )
        }
    )
    performance = Performance(
        client=client,
        collection_name="documents",
        embedding_function=embedding_function,
    )

    result = performance.measure_search_performance(["first", "second"], label="Test")

    assert set(result) == {"avg", "p95"}
    assert len(client.query_calls) == 3
    assert all(call["collection_name"] == "documents" for call in client.query_calls)
    assert all(call["using"] == DENSE_VECTOR_NAME for call in client.query_calls)


def test_measure_search_performance_omits_using_for_unnamed_vector() -> None:
    client = RecordingClient(
        collection_infos={
            "documents": collection_info(vectors_config=vector_params(size=3))
        }
    )
    performance = Performance(
        client=client,
        collection_name="documents",
        embedding_function=embedding_function,
    )

    performance.measure_search_performance(["first", "second"], label="Test")

    assert len(client.query_calls) == 3
    assert all("using" not in call for call in client.query_calls)


def test_prepare_quantized_collections_preserves_unnamed_vectors() -> None:
    source_point = Point(id=1, vector=[0.1, 0.2, 0.3], payload={"title": "source"})
    client = RecordingClient(
        collection_infos={
            "documents": collection_info(
                points_count=1,
                vectors_config=vector_params(size=3),
            )
        },
        points_by_collection={"documents": [source_point]},
    )
    performance = Performance(
        client=client,
        collection_name="documents",
        embedding_function=embedding_function,
    )

    performance.prepare_quantized_collections()

    assert len(client.create_collection_calls) == len(quantization_configs)
    assert len(client.upload_points_calls) == len(quantization_configs)
    assert all(
        isinstance(call["vectors_config"], models.VectorParams)
        for call in client.create_collection_calls
    )
    for method_name, create_call in zip(
        quantization_configs,
        client.create_collection_calls,
        strict=True,
    ):
        vectors_config = create_call["vectors_config"]
        assert vectors_config.on_disk is True
        assert (
            vectors_config.quantization_config
            == quantization_configs[method_name]["config"]
        )
    assert all(
        upload_call["points"][0].vector == source_point.vector
        for upload_call in client.upload_points_calls
    )
    assert all(
        upload_call["wait"] is True
        for upload_call in client.upload_points_calls
    )


def test_prepare_quantized_collections_skips_matching_targets() -> None:
    target_names = quantized_collection_names()
    shared_point = Point(
        id=1,
        vector={DENSE_VECTOR_NAME: [0.1, 0.2, 0.3]},
        payload={"id": 1},
    )
    client = RecordingClient(
        collection_infos={
            "documents": collection_info(
                points_count=1,
                vectors_config={DENSE_VECTOR_NAME: vector_params(size=3)},
            ),
            **quantized_collection_infos(),
        },
        existing_collections=set(target_names),
        points_by_collection={
            "documents": [shared_point],
            **{target_name: [shared_point] for target_name in target_names},
        },
    )
    performance = Performance(
        client=client,
        collection_name="documents",
        embedding_function=embedding_function,
    )

    performance.prepare_quantized_collections()

    assert client.create_collection_calls == []
    assert client.upload_points_calls == []
    assert client.delete_collection_calls == []


def test_prepare_quantized_collections_recreates_matching_targets_without_quantization() -> None:
    target_names = quantized_collection_names()
    shared_point = Point(
        id=1,
        vector={DENSE_VECTOR_NAME: [0.1, 0.2, 0.3]},
        payload={"id": 1},
    )
    client = RecordingClient(
        collection_infos={
            "documents": collection_info(
                points_count=1,
                vectors_config={DENSE_VECTOR_NAME: vector_params(size=3)},
            ),
            **{
                target_name: collection_info(
                    points_count=1,
                    vectors_config={DENSE_VECTOR_NAME: vector_params(size=3)},
                )
                for target_name in target_names
            },
        },
        existing_collections=set(target_names),
        points_by_collection={
            "documents": [shared_point],
            **{target_name: [shared_point] for target_name in target_names},
        },
    )
    performance = Performance(
        client=client,
        collection_name="documents",
        embedding_function=embedding_function,
    )

    performance.prepare_quantized_collections()

    assert client.delete_collection_calls == target_names
    assert len(client.create_collection_calls) == len(quantization_configs)
    assert len(client.upload_points_calls) == len(quantization_configs)
    assert all(
        upload_call["points"][0].payload == shared_point.payload
        for upload_call in client.upload_points_calls
    )


def test_prepare_quantized_collections_recreates_mismatching_targets() -> None:
    target_names = quantized_collection_names()
    source_point = Point(
        id=1,
        vector={DENSE_VECTOR_NAME: [0.1, 0.2, 0.3]},
        payload={"id": 1},
    )
    target_point = Point(
        id=2,
        vector={DENSE_VECTOR_NAME: [0.3, 0.2, 0.1]},
        payload={"id": 2},
    )
    client = RecordingClient(
        collection_infos={
            "documents": collection_info(
                points_count=1,
                vectors_config={DENSE_VECTOR_NAME: vector_params(size=3)},
            ),
            **quantized_collection_infos(),
        },
        existing_collections=set(target_names),
        points_by_collection={
            "documents": [source_point],
            **{target_name: [target_point] for target_name in target_names},
        },
    )
    performance = Performance(
        client=client,
        collection_name="documents",
        embedding_function=embedding_function,
    )

    performance.prepare_quantized_collections()

    assert client.delete_collection_calls == target_names
    assert len(client.create_collection_calls) == len(quantization_configs)
    assert len(client.upload_points_calls) == len(quantization_configs)


def test_prepare_quantized_collections_recreates_when_samples_are_empty() -> None:
    target_names = quantized_collection_names()
    client = RecordingClient(
        collection_infos={
            "documents": collection_info(
                points_count=1,
                vectors_config={DENSE_VECTOR_NAME: vector_params(size=3)},
            ),
            **quantized_collection_infos(),
        },
        existing_collections=set(target_names),
    )
    performance = Performance(
        client=client,
        collection_name="documents",
        embedding_function=embedding_function,
    )

    performance.prepare_quantized_collections()

    assert client.delete_collection_calls == target_names
    assert len(client.create_collection_calls) == len(quantization_configs)


def test_measure_accuracy_uses_rescore_params_and_detected_vector_names() -> None:
    client = RecordingClient(
        collection_infos={
            "baseline": collection_info(vectors_config=vector_params(size=3)),
            "quantized": collection_info(
                vectors_config={DENSE_VECTOR_NAME: vector_params(size=3)}
            ),
        },
        query_results=[
            QueryResult(
                points=[
                    Point(id=1, vector=[], payload={}),
                    Point(id=2, vector=[], payload={}),
                ]
            ),
            QueryResult(points=[Point(id=1, vector=[], payload={})]),
        ],
    )
    performance = Performance(
        client=client,
        collection_name="baseline",
        embedding_function=embedding_function,
    )

    accuracy = performance.measure_accuracy(["question"], "quantized", oversampling=2.0)

    assert accuracy == pytest.approx(0.5)
    baseline_call, quantized_call = client.query_calls
    assert "using" not in baseline_call
    assert quantized_call["using"] == DENSE_VECTOR_NAME
    assert quantized_call["search_params"].quantization.oversampling == 2.0


def test_run_benchmark_writes_report_and_prints_summary(tmp_path, capsys) -> None:
    report_path = tmp_path / "benchmark_report.md"
    benchmark = DeterministicBenchmark()

    report = run_benchmark(
        collection_name="documents",
        test_queries=["first", "second"],
        report_path=report_path,
        performance=benchmark,
    )

    output = capsys.readouterr().out
    report_text = report_path.read_text(encoding="utf-8")

    assert benchmark.prepared_collections
    assert report.report_path == report_path
    assert report.query_count == 2
    assert "Recommended quantized method: `scalar`" in report_text
    assert "| Method | Avg | P95 | Avg speedup | P95 speedup |" in report_text
    assert "| scalar | 50.00ms | 90.00ms | 2.00x | 2.22x | 95.00% | - | 1.900 | 4x |" in report_text
    assert "--- Final Benchmark Summary ---" in output
    assert f"Report saved to: {report_path}" in output
    assert "Recommended quantized method: scalar" in output
    assert "Fastest: binary_2bit" in output


def test_run_benchmark_uses_performance_default_collection(tmp_path) -> None:
    report_path = tmp_path / "benchmark_report.md"
    benchmark = DeterministicBenchmark()

    report = run_benchmark(
        test_queries=["first"],
        report_path=report_path,
        performance=benchmark,
    )

    report_text = report_path.read_text(encoding="utf-8")

    assert report.collection_name == benchmark.collection_name
    assert report.baseline.collection_name == benchmark.collection_name
    assert f"Source collection: `{benchmark.collection_name}`" in report_text


def collection_info(
    vectors_config: Any,
    points_count: int = 0,
    sparse_vectors_config: Any | None = None,
) -> Any:
    return SimpleNamespace(
        points_count=points_count,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=vectors_config,
                sparse_vectors=sparse_vectors_config or {},
            )
        ),
    )


def vector_params(
    size: int,
    on_disk: bool | None = None,
    quantization_config: Any | None = None,
) -> models.VectorParams:
    params = {"size": size, "distance": models.Distance.COSINE}
    if on_disk is not None:
        params["on_disk"] = on_disk
    if quantization_config is not None:
        params["quantization_config"] = quantization_config
    return models.VectorParams(**params)


def embedding_function(texts: list[str]) -> list[list[float]]:
    return [[float(index), 0.0, 0.0] for index, _ in enumerate(texts, start=1)]


def quantized_collection_names() -> list[str]:
    return [f"quantized_{method_name}" for method_name in quantization_configs]


def quantized_collection_infos(points_count: int = 1) -> dict[str, Any]:
    return {
        f"quantized_{method_name}": collection_info(
            points_count=points_count,
            vectors_config={
                DENSE_VECTOR_NAME: vector_params(
                    size=3,
                    on_disk=True,
                    quantization_config=config_info["config"],
                )
            },
        )
        for method_name, config_info in quantization_configs.items()
    }


class DeterministicBenchmark:
    def __init__(self) -> None:
        self.collection_name = "default_documents"
        self.prepared_collections = False

    def measure_search_performance(
        self,
        test_queries,
        label="Baseline",
        using=None,
        collection_name=None,
    ) -> dict[str, float]:
        del test_queries, label, using
        if collection_name is None:
            return {"avg": 100.0, "p95": 200.0}

        return {
            "quantized_scalar": {"avg": 50.0, "p95": 90.0},
            "quantized_binary": {"avg": 70.0, "p95": 100.0},
            "quantized_binary_2bit": {"avg": 40.0, "p95": 85.0},
        }[collection_name]

    def prepare_quantized_collections(self) -> None:
        self.prepared_collections = True

    def measure_accuracy(
        self,
        test_queries,
        quantized_collection,
        oversampling=1.0,
    ) -> float:
        del test_queries
        accuracies = {
            ("quantized_scalar", 1.0): 0.95,
            ("quantized_binary", 1.0): 0.30,
            ("quantized_binary", 2.0): 0.40,
            ("quantized_binary_2bit", 1.0): 0.20,
            ("quantized_binary_2bit", 2.0): 0.25,
        }
        return accuracies[(quantized_collection, oversampling)]
