from __future__ import annotations

from uuid import UUID
from uuid import uuid4

import pytest

from agents.models import AgentAnswer, AgentExecutionError, AgentRequest, AgentSource
from backend.use_cases import AnswerQuestionUseCase, EmptyQuestionError


class RecordingAgentRunner:
    def __init__(self, answer: AgentAnswer | None = None) -> None:
        self._answer = answer
        self.requests: list[AgentRequest] = []

    def answer(self, request: AgentRequest) -> AgentAnswer:
        self.requests.append(request)
        return self._answer or AgentAnswer(
            answer="Agent answer [S1].",
            agents=("cardiometabolic_internist",),
            sources=(
                AgentSource(
                    id="S1",
                    source="report.md",
                    document_name="Clinical Report",
                    content_hash="hash",
                    score=0.82,
                    excerpt="LDL cholesterol is elevated.",
                ),
            ),
            insufficient_context=False,
        )


class RecordingAgentRunnerFactory:
    def __init__(self, runner: RecordingAgentRunner) -> None:
        self._runner = runner
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
    ) -> RecordingAgentRunner:
        self.calls.append(
            {
                "owner_user_id": owner_user_id,
                "retrieval_limit": retrieval_limit,
            }
        )
        return self._runner


def test_answer_question_use_case_returns_agent_answer_with_sources() -> None:
    owner_user_id = uuid4()
    runner = RecordingAgentRunner()
    runner_factory = RecordingAgentRunnerFactory(runner)
    use_case = AnswerQuestionUseCase(agent_runner_factory=runner_factory)

    answer = use_case.execute(
        question=" Is the lipid panel concerning? ",
        limit=3,
        owner_user_id=owner_user_id,
        requested_agent="internist",
    )

    assert answer.answer == "Agent answer [S1]."
    assert answer.agents == ("cardiometabolic_internist",)
    assert answer.sources[0].document_name == "Clinical Report"
    assert answer.insufficient_context is False
    assert runner.requests[0].question == "Is the lipid panel concerning?"
    assert runner.requests[0].requested_agent == "internist"
    assert runner.requests[0].user_id == owner_user_id
    assert runner_factory.calls == [
        {"owner_user_id": owner_user_id, "retrieval_limit": 3}
    ]


def test_answer_question_use_case_returns_insufficient_context_answer() -> None:
    runner = RecordingAgentRunner(
        AgentAnswer(
            answer="I could not find enough context.",
            agents=("cardiometabolic_internist",),
            sources=(),
            insufficient_context=True,
        )
    )
    use_case = AnswerQuestionUseCase(
        agent_runner_factory=RecordingAgentRunnerFactory(runner),
    )

    answer = use_case.execute(
        question="What next?",
        limit=3,
        owner_user_id=uuid4(),
    )

    assert answer.insufficient_context is True
    assert answer.sources == ()
    assert runner.requests[0].question == "What next?"


def test_answer_question_use_case_rejects_empty_question() -> None:
    use_case = AnswerQuestionUseCase(
        agent_runner_factory=RecordingAgentRunnerFactory(RecordingAgentRunner()),
    )

    with pytest.raises(EmptyQuestionError, match="Question is required"):
        use_case.execute(question="   ", limit=3, owner_user_id=uuid4())


def test_answer_question_use_case_wraps_agent_initialization_failure() -> None:
    def failing_agent_runner(
        *,
        owner_user_id: UUID,
        retrieval_limit: int,
    ) -> RecordingAgentRunner:
        raise RuntimeError("missing client configuration")

    use_case = AnswerQuestionUseCase(agent_runner_factory=failing_agent_runner)

    with pytest.raises(AgentExecutionError, match="Agent execution failed"):
        use_case.execute(
            question="What does the result mean?",
            limit=3,
            owner_user_id=uuid4(),
        )
