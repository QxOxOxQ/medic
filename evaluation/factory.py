from __future__ import annotations

from dataclasses import dataclass

from langfuse import Langfuse
from sqlalchemy.orm import Session, sessionmaker

from agents.graph import AGENT_PROMPT_VERSION
from clients.chat_models import get_chat_model_settings
from clients.openrouter import get_openrouter_settings
from evaluation.application.bootstrap import BootstrapEvaluationDataset
from evaluation.application.case_runner import EvaluationCaseRunner
from evaluation.application.corpus import EnsureEvaluationCorpus
from evaluation.application.guard import SyntheticBoundaryGuard
from evaluation.application.errors import (
    EvaluationApplicationError,
    EvaluationConfigurationError,
)
from evaluation.application.ports import JudgeCalibration
from evaluation.application.run import RunEvaluation
from evaluation.application.scoring import (
    AnswerScoringPipeline,
    RetrievalScorer,
    SampleScoringPipeline,
    ScoreAggregator,
)
from evaluation.config import EvaluationSettings, get_evaluation_settings
from evaluation.infrastructure.corpus import (
    CorpusFingerprintCalculator,
    EvaluationCollectionGuard,
    EvaluationCollectionInspector,
    EvaluationDocumentSeeder,
    EvaluationIndexRebuilder,
    EvaluationTenantProvisioner,
)
from evaluation.infrastructure.fingerprints import (
    JudgeFingerprintCalculator,
    ProfileFingerprintCalculator,
    SystemFingerprintCalculator,
)
from evaluation.infrastructure.langfuse_bootstrap import LangfuseDatasetBootstrapper
from evaluation.infrastructure.langfuse_gateway import LangfuseExperimentGateway
from evaluation.infrastructure.langfuse_tracing import (
    TracingAnswerSystemUnderTest,
    TracingRagasEvaluator,
    TracingRetrieverUnderTest,
)
from evaluation.infrastructure.medic_answer_system import MedicAnswerSystemUnderTest
from evaluation.infrastructure.medic_retriever import MedicRetrieverUnderTest
from evaluation.infrastructure.profile_json import JsonProfileRepository
from evaluation.infrastructure.ragas_adapter import (
    RagasJudgeCalibration,
    RagasMetricEvaluator,
)
from observability.langfuse import build_nested_agent_observability
from rag.config import PROJECT_ROOT, get_qdrant_settings


@dataclass(frozen=True)
class EvaluationServices:
    run: RunEvaluation
    calibration: JudgeCalibration
    settings: EvaluationSettings


def build_evaluation_services(
    *,
    session_factory: sessionmaker[Session],
    settings: EvaluationSettings | None = None,
) -> EvaluationServices:
    try:
        return _build_evaluation_services(
            session_factory=session_factory,
            settings=settings,
        )
    except EvaluationApplicationError:
        raise
    except Exception as error:
        raise EvaluationConfigurationError(
            "Evaluation services could not be configured"
        ) from error


def build_dataset_bootstrap_service(
    *,
    settings: EvaluationSettings | None = None,
) -> BootstrapEvaluationDataset:
    try:
        resolved = settings or get_evaluation_settings()
        profiles = JsonProfileRepository(resolved.profile_directory)
        bootstrapper = LangfuseDatasetBootstrapper(_langfuse_client(resolved))
        return BootstrapEvaluationDataset(
            profiles=profiles,
            experiments=bootstrapper,
            manifest_path_for=lambda profile_id: str(
                resolved.bootstrap_path(profile_id)
            ),
        )
    except EvaluationApplicationError:
        raise
    except Exception as error:
        raise EvaluationConfigurationError(
            "Evaluation bootstrap could not be configured"
        ) from error


def _build_evaluation_services(
    *,
    session_factory: sessionmaker[Session],
    settings: EvaluationSettings | None,
) -> EvaluationServices:
    resolved = settings or get_evaluation_settings()
    profiles = JsonProfileRepository(resolved.profile_directory)
    langfuse = _langfuse_client(resolved)
    experiments = _experiment_gateway(resolved, client=langfuse)
    openrouter = get_openrouter_settings()
    ragas = RagasMetricEvaluator(
        api_key=openrouter.api_key,
        base_url=openrouter.base_url,
        judge_model=resolved.judge_model,
        embedding_model=resolved.embedding_model,
    )
    traced_ragas = TracingRagasEvaluator(ragas, langfuse)
    calibration = RagasJudgeCalibration(traced_ragas, resolved.calibration_path)
    agent_observability = build_nested_agent_observability(
        client=langfuse,
        public_key=resolved.langfuse_public_key,
        prompt_version=AGENT_PROMPT_VERSION,
        capture_content=True,
    )
    chat_settings = get_chat_model_settings()
    corpus = EnsureEvaluationCorpus(
        collection_prefix=resolved.collection_prefix,
        fingerprint_provider=CorpusFingerprintCalculator(PROJECT_ROOT),
        collection_guard=EvaluationCollectionGuard(
            get_qdrant_settings().qdrant_collection_name
        ),
        tenant_provisioner=EvaluationTenantProvisioner(session_factory),
        document_seeder=EvaluationDocumentSeeder(
            project_root=PROJECT_ROOT,
            raw_documents_dir=resolved.raw_documents_dir,
            session_factory=session_factory,
        ),
        index_rebuilder=EvaluationIndexRebuilder(
            raw_documents_dir=resolved.raw_documents_dir,
            parsed_markdown_dir=resolved.parsed_markdown_dir,
            session_factory=session_factory,
        ),
        collection_inspector=EvaluationCollectionInspector(session_factory),
    )
    run = RunEvaluation(
        profiles=profiles,
        corpus=corpus,
        case_runner=EvaluationCaseRunner(
            retriever=TracingRetrieverUnderTest(
                MedicRetrieverUnderTest(session_factory),
                langfuse,
            ),
            answer_system=TracingAnswerSystemUnderTest(
                MedicAnswerSystemUnderTest(session_factory, agent_observability),
                langfuse,
            ),
        ),
        scoring=SampleScoringPipeline(
            retrieval_scorer=RetrievalScorer(),
            answer_scorer=AnswerScoringPipeline(ragas_evaluator=traced_ragas),
        ),
        aggregator=ScoreAggregator(),
        guard=SyntheticBoundaryGuard(),
        experiments=experiments,
        calibration=calibration,
        profile_fingerprint=ProfileFingerprintCalculator(),
        system_fingerprint=SystemFingerprintCalculator(
            chat_settings,
            agent_prompt_version=AGENT_PROMPT_VERSION,
        ),
        judge_fingerprint=JudgeFingerprintCalculator(resolved),
        configuration_metadata={
            "answer_provider": chat_settings.provider,
            "answer_model": chat_settings.model,
            "answer_temperature": str(chat_settings.temperature),
            "max_tool_iterations": str(chat_settings.max_tool_iterations),
            "max_review_rounds": str(chat_settings.max_review_rounds),
            "judge_provider": resolved.judge_provider,
            "judge_model": resolved.judge_model,
            "judge_prompt_version": resolved.judge_prompt_version,
            "embedding_model": resolved.embedding_model,
            "synthetic_only": "true",
        },
    )
    return EvaluationServices(
        run=run,
        calibration=calibration,
        settings=resolved,
    )


def _langfuse_client(settings: EvaluationSettings) -> Langfuse:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise EvaluationConfigurationError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required"
        )
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
    )


def _experiment_gateway(
    settings: EvaluationSettings,
    *,
    client: Langfuse | None = None,
) -> LangfuseExperimentGateway:
    return LangfuseExperimentGateway(
        client or _langfuse_client(settings),
        confirmation_timeout_seconds=settings.confirmation_timeout_seconds,
    )
