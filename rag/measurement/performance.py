from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.embedding.embedder import embed_texts
from rag.measurement.quantization_config import quantization_configs
from rag.qdrant import Qdrant
from rag.config import SETTINGS, get_qdrant_settings


EmbeddingFunction = Callable[[list[str]], list[Any]]
PERFORMANCE_SETTINGS = SETTINGS["measurement"]["performance"]

DEFAULT_TEST_QUERIES = [
    "Interpretacja morfologii krwi z obniżoną hemoglobiną",
    "Co oznacza podwyższone CRP w badaniach krwi?",
    "Ferrytyna i żelazo w diagnostyce niedokrwistości",
    "HbA1c and fasting glucose in diabetes risk assessment",
    "Kiedy powtórzyć badania krwi po nieprawidłowych leukocytach?",
    "Interpretacja lipidogramu: cholesterol całkowity LDL HDL",
    "Jak obniżyć wysoki cholesterol LDL?",
    "Trójglicerydy powyżej normy a ryzyko sercowo-naczyniowe",
    "Cholesterol HDL jako czynnik ochronny w lipidogramie",
    "Kiedy rozważyć leczenie statyną przy wysokim cholesterolu?",
    "Obrzęk kolana po rekonstrukcji ACL i powrót do biegania",
    "Ból przedniego kolana podczas rehabilitacji po urazie",
    "Kiedy wykonać MRI kolana przy nawracającym wysięku?",
    "Kryteria progresji ćwiczeń po operacji więzadła krzyżowego",
    "Ograniczony zakres ruchu kolana i osłabienie mięśnia czworogłowego",
    "Migrena z aurą i objawy wymagające konsultacji neurologicznej",
    "Silny nagły ból głowy jako wskazanie do pilnej diagnostyki",
    "Zawroty głowy i zaburzenia równowagi w neurologii",
    "Ból głowy z drętwieniem kończyn i zaburzeniami mowy",
    "Przewlekłe bóle głowy: kiedy skierować pacjenta do neurologa?",
]


@dataclass(frozen=True)
class BenchmarkMethodResult:
    method_name: str
    collection_name: str
    avg_latency_ms: float
    p95_latency_ms: float
    accuracy: float | None = None
    rescored_accuracy: float | None = None
    expected_speedup: str = "-"
    expected_compression: str = "-"

    @property
    def best_accuracy(self) -> float | None:
        scores = [
            score
            for score in [self.accuracy, self.rescored_accuracy]
            if score is not None
        ]
        if not scores:
            return None
        return max(scores)


@dataclass(frozen=True)
class BenchmarkReport:
    collection_name: str
    query_count: int
    created_at: datetime
    report_path: Path
    baseline: BenchmarkMethodResult
    method_results: list[BenchmarkMethodResult]


class VectorDBClient(Protocol):
    @property
    def models(self) -> Any:
        ...

    def query_points(self, **kwargs: Any) -> Any:
        ...

    def create_collection(self, **kwargs: Any) -> Any:
        ...

    def upload_points(self, **kwargs: Any) -> Any:
        ...

    def get_collection(self, collection_name: str) -> Any:
        ...

    def scroll(self, **kwargs: Any) -> Any:
        ...

    def delete_collection(self, collection_name: str) -> Any:
        ...

    def collection_exists(self, collection_name: str) -> bool:
        ...


class Performance:
    scroll_batch_size = PERFORMANCE_SETTINGS["scroll_batch_size"]
    upload_batch_size = PERFORMANCE_SETTINGS["upload_batch_size"]
    upload_parallel = PERFORMANCE_SETTINGS["upload_parallel"]
    search_limit = PERFORMANCE_SETTINGS["search_limit"]

    def __init__(
        self,
        client: VectorDBClient | None = None,
        collection_name: str | None = None,
        embedding_function: EmbeddingFunction = embed_texts,
    ) -> None:
        self.client = client or Qdrant()
        self.collection_name = (
            collection_name or get_qdrant_settings().qdrant_collection_name
        )
        self.embedding_function = embedding_function
        self.vector_name_cache: dict[str, str | None] = {}

    def measure_search_performance(
        self,
        test_queries: Sequence[str],
        label: str = "Baseline",
        using: str | None = None,
        collection_name: str | None = None,
    ) -> dict[str, float]:
        benchmark_collection = collection_name or self.collection_name
        vector_name = using or self._get_dense_vector_name(benchmark_collection)
        query_vectors = self._embed_queries(test_queries, label)

        print(f"[{label}] Warm up search...")
        self._query_collection(benchmark_collection, query_vectors[0], vector_name)

        print(f"[{label}] Running benchmark...")
        latencies = self._measure_query_latencies(
            benchmark_collection,
            query_vectors,
            vector_name,
            label,
        )

        result = {
            "avg": sum(latencies) / len(latencies),
            "p95": _percentile(latencies, 95),
        }
        self._print_search_summary(label, result)
        return result

    def prepare_quantized_collections(self) -> None:
        print(f"Fetching source collection info: {self.collection_name}")
        source_info = self.client.get_collection(self.collection_name)
        source_points_count = source_info.points_count or 0
        print(
            f"Source collection {self.collection_name} "
            f"has {source_points_count} points."
        )

        source_vectors_config = source_info.config.params.vectors
        source_sparse_config = source_info.config.params.sparse_vectors
        source_vector_name = self._get_dense_vector_name(self.collection_name)
        source_dense_params = self._get_dense_vector_params(source_vectors_config)

        if source_dense_params is None:
            raise ValueError("Could not find dense vector configuration.")

        for method_name, config_info in quantization_configs.items():
            self._prepare_quantized_collection(
                method_name=method_name,
                quantization_config=config_info["config"],
                source_points_count=source_points_count,
                source_dense_params=source_dense_params,
                source_sparse_config=source_sparse_config,
                source_vector_name=source_vector_name,
            )

    def measure_accuracy(
        self,
        test_queries: Sequence[str],
        quantized_collection: str,
        oversampling: float = 1.0,
    ) -> float:
        label = quantized_collection
        query_vectors = self._embed_queries(test_queries, label)
        base_vector_name = self._get_dense_vector_name(self.collection_name)
        quantized_vector_name = self._get_dense_vector_name(quantized_collection)
        search_params = self._build_quantization_search_params(oversampling)

        print(f"[{label}] Measuring accuracy (oversampling={oversampling})...")
        overlaps = []
        for index, vector in enumerate(query_vectors, start=1):
            baseline_ids = self._result_ids(
                self._query_collection(
                    self.collection_name,
                    vector,
                    base_vector_name,
                )
            )
            quantized_ids = self._result_ids(
                self._query_collection(
                    quantized_collection,
                    vector,
                    quantized_vector_name,
                    search_params=search_params,
                )
            )

            overlaps.append(_overlap_ratio(baseline_ids, quantized_ids))
            self._print_progress(label, index, len(query_vectors), "Accuracy check")

        average_accuracy = sum(overlaps) / len(overlaps)
        print(
            f"Accuracy for {quantized_collection} "
            f"(oversampling {oversampling}): {average_accuracy:.2%}"
        )
        return average_accuracy

    def _prepare_quantized_collection(
        self,
        method_name: str,
        quantization_config: Any,
        source_points_count: int,
        source_dense_params: Any,
        source_sparse_config: Any,
        source_vector_name: str | None,
    ) -> None:
        target_name = _quantized_collection_name(method_name)

        if self._target_collection_is_current(
            target_name,
            source_points_count,
            source_dense_params,
            quantization_config,
            source_vector_name,
        ):
            print(
                f"Collection {target_name} already exists and matches source. "
                "Skipping migration."
            )
            return

        if self.client.collection_exists(target_name):
            print(
                f"Collection {target_name} exists but does not match source. "
                "Recreating..."
            )
            self.client.delete_collection(target_name)
        else:
            print(f"Creating collection: {target_name}...")

        self.client.create_collection(
            collection_name=target_name,
            vectors_config=self._build_quantized_vectors_config(
                source_dense_params,
                quantization_config,
                source_vector_name,
            ),
            sparse_vectors_config=source_sparse_config,
        )
        self._migrate_points(target_name, source_points_count, source_vector_name)

    def _target_collection_is_current(
        self,
        target_name: str,
        source_points_count: int,
        source_dense_params: Any,
        quantization_config: Any,
        source_vector_name: str | None,
    ) -> bool:
        if not self.client.collection_exists(target_name):
            return False

        target_info = self.client.get_collection(target_name)
        if not self._target_vectors_match(
            target_info.config.params.vectors,
            source_dense_params,
            quantization_config,
            source_vector_name,
        ):
            return False

        target_points_count = target_info.points_count or 0
        if target_points_count != source_points_count:
            return False

        if source_points_count == 0:
            return True

        return self._sample_payloads_match(target_name)

    def _target_vectors_match(
        self,
        target_vectors_config: Any,
        source_dense_params: Any,
        quantization_config: Any,
        source_vector_name: str | None,
    ) -> bool:
        if _select_dense_vector_name(target_vectors_config) != source_vector_name:
            return False

        target_dense_params = self._get_dense_vector_params(target_vectors_config)
        return (
            getattr(target_dense_params, "size", None) == source_dense_params.size
            and getattr(target_dense_params, "distance", None)
            == source_dense_params.distance
            and getattr(target_dense_params, "on_disk", None) is True
            and getattr(target_dense_params, "quantization_config", None)
            == quantization_config
        )

    def _sample_payloads_match(self, target_name: str) -> bool:
        source_sample, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=5,
            with_vectors=False,
            with_payload=True,
        )
        target_sample, _ = self.client.scroll(
            collection_name=target_name,
            limit=5,
            with_vectors=False,
            with_payload=True,
        )
        source_payloads = _payload_by_id(source_sample)
        if not source_payloads:
            return False
        return source_payloads == _payload_by_id(target_sample)

    def _build_quantized_vectors_config(
        self,
        source_dense_params: Any,
        quantization_config: Any,
        source_vector_name: str | None,
    ) -> Any:
        vector_params_kwargs = {
            "size": source_dense_params.size,
            "distance": source_dense_params.distance,
            "on_disk": PERFORMANCE_SETTINGS["quantized_vectors_on_disk"],
            "quantization_config": quantization_config,
        }
        if getattr(source_dense_params, "multivector_config", None) is not None:
            vector_params_kwargs["multivector_config"] = (
                source_dense_params.multivector_config
            )
        vector_params = self.client.models.VectorParams(**vector_params_kwargs)
        if source_vector_name is None:
            return vector_params
        return {source_vector_name: vector_params}

    def _migrate_points(
        self,
        target_name: str,
        source_points_count: int,
        target_vector_name: str | None,
    ) -> None:
        offset = None
        migrated_count = 0

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=self.scroll_batch_size,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
            if not points:
                break

            self.client.upload_points(
                collection_name=target_name,
                points=[
                    self._build_point_struct(point, target_vector_name)
                    for point in points
                ],
                batch_size=self.upload_batch_size,
                parallel=self.upload_parallel,
                wait=True,
            )
            migrated_count += len(points)
            print(
                f"  [{target_name}] Migrated "
                f"{migrated_count}/{source_points_count} points..."
            )

            if next_offset is None:
                break
            offset = next_offset

        print(f"Zmigrowano {migrated_count} punktów do: {target_name}")

    def _build_point_struct(self, point: Any, target_vector_name: str | None) -> Any:
        return self.client.models.PointStruct(
            id=point.id,
            vector=_vector_for_target(point.vector, target_vector_name),
            payload=point.payload,
        )

    def _get_dense_vector_name(self, collection_name: str) -> str | None:
        if collection_name not in self.vector_name_cache:
            collection_info = self.client.get_collection(collection_name)
            vector_name = _select_dense_vector_name(collection_info.config.params.vectors)
            self.vector_name_cache[collection_name] = vector_name
        return self.vector_name_cache[collection_name]

    def _get_dense_vector_params(self, vectors_config: Any) -> Any:
        vector_name = _select_dense_vector_name(vectors_config)
        if vector_name is None:
            return vectors_config
        return vectors_config[vector_name]

    def _embed_queries(self, test_queries: Sequence[str], label: str) -> list[Any]:
        queries = list(test_queries)
        if not queries:
            raise ValueError("test_queries must not be empty")

        print(f"[{label}] Embedding {len(queries)} test queries...")
        return self.embedding_function(queries)

    def _measure_query_latencies(
        self,
        collection_name: str,
        query_vectors: Sequence[Any],
        vector_name: str | None,
        label: str,
    ) -> list[float]:
        latencies = []
        for index, vector in enumerate(query_vectors, start=1):
            started_at = time.perf_counter()
            self._query_collection(collection_name, vector, vector_name)
            latencies.append((time.perf_counter() - started_at) * 1000)
            self._print_progress(label, index, len(query_vectors), "Progress")
        return latencies

    def _query_collection(
        self,
        collection_name: str,
        vector: Any,
        vector_name: str | None,
        search_params: Any | None = None,
    ) -> Any:
        query_args: dict[str, Any] = {
            "collection_name": collection_name,
            "query": vector,
            "limit": self.search_limit,
        }
        if vector_name:
            query_args["using"] = vector_name
        if search_params is not None:
            query_args["search_params"] = search_params
        return self.client.query_points(**query_args)

    def _build_quantization_search_params(self, oversampling: float) -> Any:
        return self.client.models.SearchParams(
            quantization=self.client.models.QuantizationSearchParams(
                rescore=True,
                oversampling=oversampling,
            )
        )

    def _result_ids(self, result: Any) -> set[Any]:
        return {point.id for point in result.points}

    def _print_search_summary(self, label: str, result: dict[str, float]) -> None:
        print(f"{label}:")
        print(f"  Average latency: {result['avg']:.2f}ms")
        print(f"  P95 latency: {result['p95']:.2f}ms")
        print("  Memory usage: Check Qdrant Cloud dashboard")

    def _print_progress(
        self,
        label: str,
        current: int,
        total: int,
        prefix: str,
    ) -> None:
        if current % 5 == 0 or current == total:
            print(f"  [{label}] {prefix}: {current}/{total} queries processed")


def run_benchmark(
    collection_name: str | None = None,
    test_queries: Sequence[str] = DEFAULT_TEST_QUERIES,
    report_path: str | Path = "benchmark_report.md",
    performance: Performance | None = None,
) -> BenchmarkReport:
    performance = performance or Performance(collection_name=collection_name)
    benchmark_collection_name = collection_name or performance.collection_name
    queries = list(test_queries)

    print("--- 1. Baseline Search Performance (Float32) ---")
    baseline_metrics = performance.measure_search_performance(queries, label="Baseline")
    baseline_result = BenchmarkMethodResult(
        method_name="baseline",
        collection_name=benchmark_collection_name,
        avg_latency_ms=baseline_metrics["avg"],
        p95_latency_ms=baseline_metrics["p95"],
        accuracy=1.0,
    )

    print("\n--- 2. Preparing Quantized Collections (Migration) ---")
    performance.prepare_quantized_collections()

    print("\n--- 3. Benchmarking Quantized Collections ---")
    method_results = []
    for method_name in quantization_configs:
        config_info = quantization_configs[method_name]
        target_name = _quantized_collection_name(method_name)
        print(f"\n>> Method: {method_name}")
        performance_metrics = performance.measure_search_performance(
            test_queries,
            label=f"Performance ({method_name})",
            collection_name=target_name,
        )
        accuracy = performance.measure_accuracy(test_queries, target_name)

        rescored_accuracy = None
        if "binary" in method_name:
            rescored_accuracy = performance.measure_accuracy(
                test_queries,
                target_name,
                oversampling=2.0,
            )

        method_results.append(
            BenchmarkMethodResult(
                method_name=method_name,
                collection_name=target_name,
                avg_latency_ms=performance_metrics["avg"],
                p95_latency_ms=performance_metrics["p95"],
                accuracy=accuracy,
                rescored_accuracy=rescored_accuracy,
                expected_speedup=config_info["expected_speedup"],
                expected_compression=config_info["expected_compression"],
            )
        )

    report = BenchmarkReport(
        collection_name=benchmark_collection_name,
        query_count=len(queries),
        created_at=datetime.now().astimezone(),
        report_path=Path(report_path),
        baseline=baseline_result,
        method_results=method_results,
    )
    _write_benchmark_report(report)
    _print_final_benchmark_summary(report)
    return report


def _write_benchmark_report(report: BenchmarkReport) -> None:
    report.report_path.parent.mkdir(parents=True, exist_ok=True)
    report.report_path.write_text(_build_benchmark_report(report), encoding="utf-8")


def _build_benchmark_report(report: BenchmarkReport) -> str:
    recommended = _recommended_method(report)
    fastest = _fastest_method(report)
    most_accurate = _most_accurate_method(report)
    lines = [
        "# Qdrant Benchmark Report",
        "",
        f"- Generated at: `{report.created_at.isoformat(timespec='seconds')}`",
        f"- Source collection: `{report.collection_name}`",
        f"- Test queries: `{report.query_count}`",
        "",
        "## Recommendation",
        "",
        (
            f"- Recommended quantized method: `{recommended.method_name}` "
            f"with score `{_format_score(_method_score(recommended, report.baseline))}`."
        ),
        (
            f"- Fastest average latency: `{fastest.method_name}` "
            f"({_format_ms(fastest.avg_latency_ms)})."
        ),
        (
            f"- Highest measured accuracy: `{most_accurate.method_name}` "
            f"({_format_optional_percent(most_accurate.best_accuracy)})."
        ),
        "",
        "## Comparison",
        "",
        _markdown_result_table(report),
        "",
        "## How To Read This",
        "",
        "- Lower latency is better.",
        "- Higher speedup is better; `1.00x` means equal to baseline.",
        "- Accuracy is overlap with baseline search results.",
        "- Score is `average speedup x best measured accuracy`; use it as a balanced starting point, not as an absolute truth.",
        "",
    ]
    return "\n".join(lines)


def _markdown_result_table(report: BenchmarkReport) -> str:
    rows = [_baseline_table_row(report.baseline)]
    rows.extend(_method_table_row(result, report.baseline) for result in report.method_results)
    headers = [
        "Method",
        "Avg",
        "P95",
        "Avg speedup",
        "P95 speedup",
        "Accuracy",
        "Rescore acc.",
        "Score",
        "Compression",
    ]
    return _markdown_table(headers, rows)


def _baseline_table_row(result: BenchmarkMethodResult) -> list[str]:
    return [
        result.method_name,
        _format_ms(result.avg_latency_ms),
        _format_ms(result.p95_latency_ms),
        "1.00x",
        "1.00x",
        _format_optional_percent(result.accuracy),
        "-",
        "1.000",
        "-",
    ]


def _method_table_row(
    result: BenchmarkMethodResult,
    baseline: BenchmarkMethodResult,
) -> list[str]:
    return [
        result.method_name,
        _format_ms(result.avg_latency_ms),
        _format_ms(result.p95_latency_ms),
        _format_speedup(baseline.avg_latency_ms, result.avg_latency_ms),
        _format_speedup(baseline.p95_latency_ms, result.p95_latency_ms),
        _format_optional_percent(result.accuracy),
        _format_optional_percent(result.rescored_accuracy),
        _format_score(_method_score(result, baseline)),
        result.expected_compression,
    ]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def _print_final_benchmark_summary(report: BenchmarkReport) -> None:
    print("\n--- Final Benchmark Summary ---")
    print(f"Report saved to: {report.report_path}")
    print(_console_result_table(report))
    recommended = _recommended_method(report)
    fastest = _fastest_method(report)
    most_accurate = _most_accurate_method(report)
    print(
        "\nRecommended quantized method: "
        f"{recommended.method_name} "
        f"(score={_format_score(_method_score(recommended, report.baseline))}, "
        f"avg={_format_ms(recommended.avg_latency_ms)}, "
        f"accuracy={_format_optional_percent(recommended.best_accuracy)})"
    )
    print(
        "Fastest: "
        f"{fastest.method_name} ({_format_ms(fastest.avg_latency_ms)} avg), "
        "highest accuracy: "
        f"{most_accurate.method_name} "
        f"({_format_optional_percent(most_accurate.best_accuracy)})"
    )


def _console_result_table(report: BenchmarkReport) -> str:
    headers = [
        "method",
        "avg",
        "p95",
        "avg x",
        "p95 x",
        "acc",
        "rescore",
        "score",
    ]
    rows = [
        _baseline_table_row(report.baseline)[:8],
        *[
            _method_table_row(result, report.baseline)[:8]
            for result in report.method_results
        ],
    ]
    column_widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    output_lines = [
        _console_table_row(headers, column_widths),
        _console_table_separator(column_widths),
    ]
    output_lines.extend(
        _console_table_row(row, column_widths)
        for row in rows
    )
    return "\n".join(output_lines)


def _console_table_row(row: Sequence[str], column_widths: Sequence[int]) -> str:
    cells = [
        value.ljust(column_widths[index])
        for index, value in enumerate(row)
    ]
    return " | ".join(cells)


def _console_table_separator(column_widths: Sequence[int]) -> str:
    return "-+-".join("-" * width for width in column_widths)


def _recommended_method(report: BenchmarkReport) -> BenchmarkMethodResult:
    return max(
        report.method_results,
        key=lambda result: _method_score(result, report.baseline),
    )


def _fastest_method(report: BenchmarkReport) -> BenchmarkMethodResult:
    return min(report.method_results, key=lambda result: result.avg_latency_ms)


def _most_accurate_method(report: BenchmarkReport) -> BenchmarkMethodResult:
    return max(report.method_results, key=lambda result: result.best_accuracy or 0.0)


def _method_score(
    result: BenchmarkMethodResult,
    baseline: BenchmarkMethodResult,
) -> float:
    accuracy = result.best_accuracy or 0.0
    return _speedup(baseline.avg_latency_ms, result.avg_latency_ms) * accuracy


def _format_ms(value: float) -> str:
    return f"{value:.2f}ms"


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2%}"


def _format_speedup(baseline_value: float, compared_value: float) -> str:
    return f"{_speedup(baseline_value, compared_value):.2f}x"


def _format_score(value: float) -> str:
    return f"{value:.3f}"


def _speedup(baseline_value: float, compared_value: float) -> float:
    if baseline_value <= 0 or compared_value <= 0:
        return 0.0
    return baseline_value / compared_value


def _select_dense_vector_name(vectors_config: Any) -> str | None:
    if not isinstance(vectors_config, dict):
        return None

    configured_name = SETTINGS["qdrant"]["dense_vector"]["name"]
    dense_vector_name = configured_name if isinstance(configured_name, str) else None
    if dense_vector_name is not None and dense_vector_name in vectors_config:
        return dense_vector_name

    for vector_name in vectors_config:
        return vector_name if isinstance(vector_name, str) and vector_name else None

    return None


def _vector_for_target(vector: Any, target_vector_name: str | None) -> Any:
    if isinstance(vector, list) and target_vector_name:
        return {target_vector_name: vector}

    if isinstance(vector, dict) and target_vector_name is None and len(vector) == 1:
        return next(iter(vector.values()))

    return vector


def _payload_by_id(points: Sequence[Any]) -> dict[Any, Any]:
    return {point.id: point.payload for point in points}


def _overlap_ratio(baseline_ids: set[Any], quantized_ids: set[Any]) -> float:
    if not baseline_ids:
        return 0.0
    return len(baseline_ids & quantized_ids) / len(baseline_ids)


def _percentile(values: Sequence[float], percentile: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("values must not be empty")

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile / 100
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (rank - lower_index)


def _quantized_collection_name(method_name: str) -> str:
    return f"quantized_{method_name}"


if __name__ == "__main__":
    run_benchmark()
