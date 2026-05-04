from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
import httpx
from langfuse import Langfuse
from langfuse.api.core import ApiError
from langfuse.experiment import ExperimentItemResult, ExperimentResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from sqlalchemy.orm import sessionmaker

from clients.chat_models import ChatModelSettings
from evaluation.application.errors import (
    CorpusIsolationError,
    EvaluationConfigurationError,
    EvaluationDatasetError,
    MetricEvaluationError,
    EvaluationPublishingError,
)
from evaluation.application.run import EvaluationRunSummarizer
from evaluation.application.scoring import ScoreAggregator
from evaluation.config import EvaluationSettings
from evaluation.domain.report import CaseResult, MetricResult
from evaluation.domain.samples import AnswerEvaluationSample, RetrievalEvaluationSample
from evaluation.domain.suite import EvaluationCase, EvaluationProfile
from evaluation.domain.values import MetricName, Score, Threshold
from evaluation.infrastructure.corpus.guard import EvaluationCollectionGuard
from evaluation.infrastructure.corpus.inspector import EvaluationCollectionInspector
from evaluation.infrastructure.corpus.seeder import EvaluationDocumentSeeder
from evaluation.infrastructure.fingerprints import (
    JudgeFingerprintCalculator,
    ProfileFingerprintCalculator,
    SystemFingerprintCalculator,
)
from evaluation.infrastructure.langfuse_bootstrap import LangfuseDatasetBootstrapper
from evaluation.infrastructure.langfuse_gateway import LangfuseExperimentGateway
from evaluation.infrastructure.profile_json import JsonProfileRepository
from evaluation.infrastructure import ragas_adapter
from rag.database.migrations import upgrade_database
from rag.database.repositories import DocumentRepository, UserRepository
from rag.database.session import create_database_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_collection_guard_rejects_production_and_unmarked_names() -> None:
    guard = EvaluationCollectionGuard("medic_eval_production")

    with pytest.raises(CorpusIsolationError):
        guard.validate("medic_eval_production")
    with pytest.raises(CorpusIsolationError):
        guard.validate("production_documents")

    guard.validate("medic_eval_medical_demo_v1_abcdef")


def test_seeder_versions_document_paths_by_corpus_fingerprint(tmp_path: Path) -> None:
    factory, owner_id = _session_factory_with_user(tmp_path)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"synthetic pdf")
    profile = _profile(document_path="source.pdf")
    seeder = EvaluationDocumentSeeder(
        project_root=tmp_path,
        raw_documents_dir=tmp_path / "raw",
        session_factory=factory,
    )

    first = seeder.seed(profile, corpus_fingerprint="a" * 64, owner_user_id=owner_id)
    second = seeder.seed(profile, corpus_fingerprint="b" * 64, owner_user_id=owner_id)

    assert first.document_ids.isdisjoint(second.document_ids)
    assert first.relative_raw_paths == frozenset(
        {f"profile/{'a' * 64}/source.pdf"}
    )
    assert second.relative_raw_paths == frozenset(
        {f"profile/{'b' * 64}/source.pdf"}
    )


def test_collection_inspector_rejects_partial_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, owner_id = _session_factory_with_user(tmp_path)
    content_hash = "a" * 64
    with factory() as session:
        document = DocumentRepository(session).create_uploaded_document(
            owner_user_id=owner_id,
            original_filename="document.pdf",
            relative_raw_path="profile/fingerprint/document.pdf",
            byte_size=1,
        )
        document.content_hash = content_hash
        session.commit()
        document_id = UUID(str(document.id))
    fake = FakeQdrant()
    monkeypatch.setattr(
        "evaluation.infrastructure.corpus.inspector.Qdrant",
        lambda **_: fake,
    )
    inspector = EvaluationCollectionInspector(factory)

    assert inspector.is_ready(
        collection_name="medic_eval_test", document_ids=frozenset({document_id})
    ) is False
    fake.content_hashes.add(content_hash)
    assert inspector.is_ready(
        collection_name="medic_eval_test", document_ids=frozenset({document_id})
    ) is True


def test_fingerprints_change_with_profile_system_and_judge_configuration(
    tmp_path: Path,
) -> None:
    profile = _profile()
    stricter = EvaluationProfile(
        **{
            **profile.__dict__,
            "thresholds": (Threshold(MetricName.FAITHFULNESS, Score(0.95)),),
        }
    )
    profile_calculator = ProfileFingerprintCalculator()
    chat = ChatModelSettings("openrouter", "deepseek/model", 0.2, 3, 1, {})
    changed_chat = ChatModelSettings("openrouter", "deepseek/new", 0.2, 3, 1, {})

    assert profile_calculator.calculate(profile) != profile_calculator.calculate(stricter)
    assert SystemFingerprintCalculator(chat, agent_prompt_version="agents-v1").calculate(
        profile, corpus_fingerprint="a" * 64
    ) != SystemFingerprintCalculator(
        changed_chat,
        agent_prompt_version="agents-v1",
    ).calculate(
        profile, corpus_fingerprint="a" * 64
    )
    first_settings = _settings(tmp_path, judge_model="judge-a")
    second_settings = _settings(tmp_path, judge_model="judge-b")
    assert JudgeFingerprintCalculator(first_settings).calculate() != (
        JudgeFingerprintCalculator(second_settings).calculate()
    )


def test_langfuse_bootstrap_is_idempotent_and_repairs_partial_import() -> None:
    client = FakeLangfuse()
    gateway = LangfuseDatasetBootstrapper(cast(Langfuse, client))
    profile = JsonProfileRepository(PROJECT_ROOT / "evaluation" / "profiles").get(
        "medical-demo-v1"
    )
    manifest = str(PROJECT_ROOT / "evaluation" / "suites" / "medical_demo_v1.json")

    first = gateway.bootstrap(profile=profile, manifest_path=manifest)
    second = gateway.bootstrap(profile=profile, manifest_path=manifest)
    client.items.pop()
    repaired = gateway.bootstrap(profile=profile, manifest_path=manifest)

    assert first.created_items == 24
    assert second.verified_items == 24
    assert repaired.created_items == 1


def test_langfuse_bootstrap_detects_item_drift() -> None:
    client = FakeLangfuse()
    gateway = LangfuseDatasetBootstrapper(cast(Langfuse, client))
    profile = JsonProfileRepository(PROJECT_ROOT / "evaluation" / "profiles").get(
        "medical-demo-v1"
    )
    manifest = str(PROJECT_ROOT / "evaluation" / "suites" / "medical_demo_v1.json")
    gateway.bootstrap(profile=profile, manifest_path=manifest)
    client.items[0].input["question"] = "drift"

    with pytest.raises(EvaluationDatasetError):
        gateway.bootstrap(profile=profile, manifest_path=manifest)


def test_langfuse_gateway_maps_results_scores_and_gate() -> None:
    client = FakeLangfuse(items=[_dataset_item()])
    gateway = LangfuseExperimentGateway(
        cast(Langfuse, client), confirmation_timeout_seconds=1
    )
    profile = _profile()

    execution = gateway.execute(
        profile=profile,
        dataset_version=datetime.now(UTC),
        run_name="run",
        metadata={"synthetic_only": "true", "thresholds": "[]"},
        task=lambda case: _case_result(case.id),
        summarize=EvaluationRunSummarizer(profile, ScoreAggregator()),
    )
    gateway.flush_and_verify(execution)

    assert execution.dataset_run_id == "dataset-run"
    assert execution.summary.decision.passed is True
    assert client.run_evaluations[0].name == "faithfulness"
    assert any(evaluation.name == "gate_pass" for evaluation in client.run_evaluations)
    assert "thresholds" in client.last_metadata


def test_langfuse_gateway_runs_sync_case_task_outside_sdk_event_loop() -> None:
    case_results: dict[str, CaseResult] = {}

    def execute_case(case: EvaluationCase) -> CaseResult:
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return _case_result(case.id)

    task = LangfuseExperimentGateway._task(execute_case, case_results)
    output = asyncio.run(task(item=_dataset_item()))

    assert output["case_id"] == "case"
    assert tuple(case_results) == ("case",)


def test_langfuse_gateway_fails_closed_for_auth_and_unconfirmed_run() -> None:
    unauthenticated = FakeLangfuse(authenticated=False)
    gateway = LangfuseExperimentGateway(
        cast(Langfuse, unauthenticated), confirmation_timeout_seconds=0
    )
    with pytest.raises(EvaluationConfigurationError):
        gateway.authenticate()

    client = FakeLangfuse(items=[_dataset_item()])
    gateway = LangfuseExperimentGateway(
        cast(Langfuse, client), confirmation_timeout_seconds=0
    )
    execution = gateway.execute(
        profile=_profile(),
        dataset_version=datetime.now(UTC),
        run_name="run",
        metadata={},
        task=lambda case: _case_result(case.id),
        summarize=EvaluationRunSummarizer(_profile(), ScoreAggregator()),
    )
    with pytest.raises(EvaluationPublishingError):
        gateway.flush_and_verify(execution)


def test_langfuse_gateway_requires_remote_item_and_run_scores() -> None:
    client = FakeLangfuse(items=[_dataset_item()], publish_scores=False)
    gateway = LangfuseExperimentGateway(
        cast(Langfuse, client), confirmation_timeout_seconds=0
    )
    execution = gateway.execute(
        profile=_profile(),
        dataset_version=datetime.now(UTC),
        run_name="run",
        metadata={},
        task=lambda case: _case_result(case.id),
        summarize=EvaluationRunSummarizer(_profile(), ScoreAggregator()),
    )

    assert gateway._run_is_complete(execution) is False
    client.publish_scores = True
    assert gateway._run_is_complete(execution) is True
    client.score_value_offset = 0.1
    assert gateway._run_is_complete(execution) is False


def test_ragas_openai_client_records_generation_model_and_usage() -> None:
    async def exercise_client() -> tuple[object, ...]:
        def respond(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "judge-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        exporter = InMemorySpanExporter()
        langfuse = Langfuse(
            public_key="pk-ragas-instrumentation-test",
            secret_key="sk-test",
            base_url="https://langfuse.test",
            span_exporter=exporter,
        )
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        client = ragas_adapter.AsyncOpenAI(
            api_key="test",
            base_url="https://openrouter.test/v1",
            http_client=http_client,
        )
        try:
            with langfuse.start_as_current_observation(name="scoring"):
                await client.chat.completions.create(
                    model="judge-model",
                    messages=[{"role": "user", "content": "question"}],
                )
            langfuse.flush()
            return tuple(exporter.get_finished_spans())
        finally:
            await http_client.aclose()
            langfuse.shutdown()

    spans = asyncio.run(exercise_client())
    generation = next(span for span in spans if span.name == "OpenAI-generation")
    scoring = next(span for span in spans if span.name == "scoring")

    assert generation.parent is not None
    assert generation.parent.span_id == scoring.context.span_id
    assert generation.attributes["langfuse.observation.type"] == "generation"
    assert generation.attributes["langfuse.observation.model.name"] == "judge-model"
    assert '"total_tokens": 5' in generation.attributes[
        "langfuse.observation.usage_details"
    ]


def test_ragas_metric_retries_once_and_reports_exhausted_metric() -> None:
    class FlakyMetric:
        def __init__(self, failures: int) -> None:
            self.failures = failures
            self.calls = 0

        def score(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls <= self.failures:
                raise RuntimeError("transient judge failure")
            return SimpleNamespace(value=0.75, reason="ok", traces=[])

    sample = _case_result("case").answer
    flaky = FlakyMetric(failures=1)
    evaluator = object.__new__(ragas_adapter.RagasMetricEvaluator)
    evaluator._metrics = (
        (MetricName.FAITHFULNESS, flaky, lambda _: {}),
    )

    results = evaluator.evaluate(sample)

    assert results[0].score == Score(0.75)
    assert flaky.calls == 2

    exhausted = FlakyMetric(failures=2)
    evaluator._metrics = (
        (MetricName.ANSWER_CORRECTNESS, exhausted, lambda _: {}),
    )
    with pytest.raises(MetricEvaluationError, match="answer_correctness.*case"):
        evaluator.evaluate(sample)


class FakeDatasetsApi:
    def __init__(self, parent: "FakeLangfuse") -> None:
        self._parent = parent

    def get(self, *, dataset_name: str):
        if not self._parent.dataset_created:
            raise ApiError(status_code=404)
        return SimpleNamespace(name=dataset_name)

    def get_run(self, *, dataset_name: str, run_name: str):
        return SimpleNamespace(
            id="dataset-run",
            dataset_run_items=[object() for _ in self._parent.last_run_item_ids],
        )


class FakeScoresApi:
    def __init__(self, parent: "FakeLangfuse") -> None:
        self._parent = parent

    def get_many(
        self,
        *,
        page: int,
        limit: int,
        trace_id: str | None,
        dataset_run_id: str | None,
    ):
        del page, limit
        evaluations = []
        if self._parent.publish_scores and trace_id is not None:
            evaluations = self._parent.trace_evaluations.get(trace_id, [])
        if self._parent.publish_scores and dataset_run_id is not None:
            evaluations = self._parent.run_evaluations
        scores = [
            SimpleNamespace(
                name=evaluation.name,
                value=float(evaluation.value) + self._parent.score_value_offset,
                data_type=evaluation.data_type or "NUMERIC",
            )
            for evaluation in evaluations
        ]
        return SimpleNamespace(
            data=scores,
            meta=SimpleNamespace(total_pages=1),
        )


class FakeApi:
    def __init__(self, parent: "FakeLangfuse") -> None:
        self.datasets = FakeDatasetsApi(parent)
        self.scores = FakeScoresApi(parent)


class FakeLangfuse:
    def __init__(
        self,
        *,
        items=None,
        authenticated: bool = True,
        publish_scores: bool = True,
    ) -> None:
        self.items = list(items or [])
        self.authenticated = authenticated
        self.publish_scores = publish_scores
        self.score_value_offset = 0.0
        self.dataset_created = bool(items)
        self.api = FakeApi(self)
        self.last_run_item_ids: list[str] = []
        self.trace_evaluations = {}
        self.run_evaluations = []
        self.last_metadata = {}

    def auth_check(self) -> bool:
        return self.authenticated

    def create_dataset(self, **kwargs):
        self.dataset_created = True
        return SimpleNamespace(**kwargs)

    def get_dataset(self, name: str, *, version=None):
        if not self.dataset_created:
            raise ApiError(status_code=404)
        return SimpleNamespace(name=name, items=self.items, version=version)

    def create_dataset_item(self, **kwargs):
        item = SimpleNamespace(
            id=kwargs["id"],
            input=kwargs["input"],
            expected_output=kwargs["expected_output"],
            metadata=kwargs["metadata"],
        )
        self.items.append(item)
        return item

    def run_experiment(
        self,
        *,
        run_name,
        data,
        task,
        evaluators,
        run_evaluators,
        **kwargs,
    ):
        item_results = []
        for item in data:
            trace_id = f"trace-{item.id}"
            output = task(item=item)
            if inspect.isawaitable(output):
                output = asyncio.run(output)
            evaluations = []
            for evaluator in evaluators:
                evaluations.extend(evaluator(output=output))
            self.trace_evaluations[trace_id] = evaluations
            item_results.append(
                ExperimentItemResult(
                    item=item,
                    output=output,
                    evaluations=evaluations,
                    trace_id=trace_id,
                    dataset_run_id="dataset-run",
                )
            )
        self.run_evaluations = run_evaluators[0](item_results=item_results)
        self.last_metadata = kwargs.get("metadata", {})
        self.last_run_item_ids = [item.id for item in data]
        return ExperimentResult(
            name=kwargs["name"],
            run_name=run_name,
            description=kwargs.get("description"),
            item_results=item_results,
            run_evaluations=self.run_evaluations,
            experiment_id="experiment",
            dataset_run_id="dataset-run",
            dataset_run_url="https://cloud.langfuse.com/run",
        )

    def flush(self) -> None:
        return None


class FakeQdrant:
    def __init__(self) -> None:
        self.content_hashes: set[str] = set()

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def scroll(self, **kwargs):
        condition = kwargs["scroll_filter"].must[0]
        value = condition.match.value
        return ([object()] if value in self.content_hashes else [], None)


def _dataset_item():
    return SimpleNamespace(
        id="item",
        input={
            "id": "case",
            "question": "Question?",
            "expected_source_keys": ["document.pdf"],
            "answerable": True,
            "requested_agent": None,
            "tags": [],
        },
        expected_output={"reference_answer": "Answer."},
        metadata={},
    )


def _case_result(case_id: str) -> CaseResult:
    retrieval = RetrievalEvaluationSample(case_id, "Question?", (), ())
    answer = AnswerEvaluationSample(
        case_id, "Question?", "Answer.", "Answer.", (), False, True, 1
    )
    return CaseResult(
        case_id,
        retrieval,
        answer,
        (MetricResult(MetricName.FAITHFULNESS, Score(1.0), case_id),),
    )


def _profile(*, document_path: str = "document.pdf") -> EvaluationProfile:
    return EvaluationProfile(
        id="profile",
        version="1",
        dataset_name="medic/test",
        document_paths=(document_path,),
        thresholds=(Threshold(MetricName.FAITHFULNESS, Score(0.9)),),
        gate_version="gate-v1",
        agent_prompt_version="agents-v1",
    )


def _settings(tmp_path: Path, *, judge_model: str) -> EvaluationSettings:
    return EvaluationSettings(
        collection_prefix="medic_eval",
        profile_directory=tmp_path,
        bootstrap_directory=tmp_path,
        calibration_path=tmp_path / "calibration.json",
        raw_documents_dir=tmp_path / "raw",
        parsed_markdown_dir=tmp_path / "parsed",
        judge_model=judge_model,
        judge_provider="OpenAI",
        judge_prompt_version="v1",
        embedding_model="embedding",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_base_url="https://cloud.langfuse.com",
        langfuse_environment="evaluation",
        confirmation_timeout_seconds=1,
    )


def _session_factory_with_user(tmp_path: Path) -> tuple[sessionmaker, UUID]:
    database_url = f"sqlite:///{tmp_path / 'evaluation.db'}"
    upgrade_database(database_url)
    engine = create_database_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        user = UserRepository(session).create_user(username="eval", password="secret")
        session.commit()
        return factory, UUID(str(user.id))
