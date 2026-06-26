from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from agents.models import AgentRequest, AgentSource
from agents.professor import MedicalContextCollector, ProfessorSourceExpander
from agents.structured_output import DocumentExpansionPayload
from agents.trace import AgentTraceRecorder
from backend.full_document_reader import ParsedMarkdownDocumentReader
from rag.config import DocumentPreparationSettings


def _source(source_id: str) -> AgentSource:
    return AgentSource(
        id=source_id,
        source=f"{source_id}.md",
        content_hash=f"hash-{source_id}",
        document_name=f"Doc {source_id}",
        score=0.5,
        excerpt=f"excerpt {source_id}",
        document_id=uuid4(),
        relative_raw_path=f"raw/{source_id}.pdf",
    )


class _FakeSearchPort:
    def __init__(self, sources: Iterable[AgentSource]) -> None:
        self._sources = list(sources)

    def search_sources(self, *, query: str) -> tuple[AgentSource, ...]:
        del query
        return tuple(self._sources)

    def sources(self) -> tuple[AgentSource, ...]:
        return tuple(self._sources)

    def attach_full_content(self, *, source_id: str, full_content: str) -> None:
        self._sources = [
            replace(source, full_content=full_content)
            if source.id == source_id
            else source
            for source in self._sources
        ]


class _FakeExpansionGateway:
    def __init__(self, selected: Iterable[str]) -> None:
        self._selected = tuple(selected)
        self.calls: list[dict[str, object]] = []

    def select_full_documents(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        valid_source_ids: set[str],
        max_documents: int,
        agent_name: str,
        phase: str,
    ) -> tuple[str, ...]:
        del system_prompt
        self.calls.append(
            {
                "valid": set(valid_source_ids),
                "max": max_documents,
                "agent": agent_name,
                "phase": phase,
                "prompt": user_prompt,
            }
        )
        return self._selected


class _FakeReader:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = dict(mapping)

    def read(self, source: AgentSource) -> str | None:
        return self._mapping.get(source.id)


def _expander(
    *,
    sources: Iterable[AgentSource],
    selected: Iterable[str],
    full_text: dict[str, str],
    recorder: AgentTraceRecorder,
    max_documents: int = 3,
) -> tuple[ProfessorSourceExpander, MedicalContextCollector, _FakeExpansionGateway]:
    collector = MedicalContextCollector(
        search_port=_FakeSearchPort(sources),
        trace_recorder=recorder,
        max_queries=6,
    )
    gateway = _FakeExpansionGateway(selected)
    expander = ProfessorSourceExpander(
        model_gateway=gateway,  # type: ignore[arg-type]
        professor_prompt="professor",
        full_document_reader=_FakeReader(full_text),
        context_collector=collector,
        trace_recorder=recorder,
        max_documents=max_documents,
    )
    return expander, collector, gateway


def test_document_expansion_payload_filters_caps_and_dedupes() -> None:
    payload = DocumentExpansionPayload(source_ids=[" S1 ", "S1", "S9", "S2", "S3"])

    selected = payload.to_domain(
        valid_source_ids={"S1", "S2", "S3"},
        max_documents=2,
    )

    assert selected == ("S1", "S2")


def test_document_expansion_payload_drops_unknown_and_empty_ids() -> None:
    payload = DocumentExpansionPayload(source_ids=["S9", "", "  "])

    assert payload.to_domain(valid_source_ids={"S1"}, max_documents=3) == ()


def test_source_expander_attaches_full_content_to_selected_sources() -> None:
    recorder = AgentTraceRecorder()
    expander, collector, gateway = _expander(
        sources=(_source("S1"), _source("S2")),
        selected=("S1", "S2"),
        full_text={"S1": "FULL ONE"},
        recorder=recorder,
    )

    expanded = expander.expand(AgentRequest(question="What does it mean?"))

    assert expanded == ("S1",)
    by_id = {source.id: source for source in collector.sources()}
    assert by_id["S1"].full_content == "FULL ONE"
    assert "content_type: full_document" in by_id["S1"].prompt_block()
    assert "FULL ONE" in by_id["S1"].prompt_block()
    assert by_id["S2"].full_content is None
    assert "content_type: excerpt" in by_id["S2"].prompt_block()
    event = next(
        event
        for event in recorder.events()
        if event.event_type == "source_expansion"
    )
    assert event.payload["selected_source_ids"] == ["S1", "S2"]
    assert event.payload["expanded_source_ids"] == ["S1"]
    assert gateway.calls[0]["valid"] == {"S1", "S2"}
    assert gateway.calls[0]["max"] == 3


def test_prompt_block_uses_excerpt_when_full_disabled() -> None:
    source = replace(_source("S1"), full_content="THE WHOLE DOCUMENT")

    assert "content_type: full_document" in source.prompt_block()
    assert "THE WHOLE DOCUMENT" in source.prompt_block()
    lean = source.prompt_block(full=False)
    assert "content_type: excerpt" in lean
    assert "THE WHOLE DOCUMENT" not in lean
    assert "excerpt S1" in lean


def test_source_expander_noops_without_sources() -> None:
    recorder = AgentTraceRecorder()
    expander, _, gateway = _expander(
        sources=(),
        selected=("S1",),
        full_text={"S1": "FULL"},
        recorder=recorder,
    )

    assert expander.expand(AgentRequest(question="q")) == ()
    assert gateway.calls == []
    assert not [
        event
        for event in recorder.events()
        if event.event_type == "source_expansion"
    ]


class _FakeDocument:
    def __init__(self, parsed_markdown_path: str | None) -> None:
        self.parsed_markdown_path = parsed_markdown_path


class _FakeSession:
    def __init__(self, document: _FakeDocument | None) -> None:
        self._document = document

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def scalar(self, *args: object, **kwargs: object) -> _FakeDocument | None:
        del args, kwargs
        return self._document


def _session_factory(document: _FakeDocument | None):  # type: ignore[no-untyped-def]
    def factory() -> _FakeSession:
        return _FakeSession(document)

    return factory


def test_full_document_reader_reads_file_and_caps(tmp_path: Path) -> None:
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat" / "doc.md").write_text("0123456789ABCDEF", encoding="utf-8")
    settings = DocumentPreparationSettings(
        raw_documents_dir=tmp_path,
        parsed_markdown_dir=tmp_path,
    )
    reader = ParsedMarkdownDocumentReader(
        database_session_factory=_session_factory(_FakeDocument("cat/doc.md")),
        owner_user_id=uuid4(),
        settings=settings,
        max_chars=10,
    )

    text = reader.read(_source("S1"))

    assert text is not None
    assert text.startswith("0123456789")
    assert "[document truncated]" in text


def test_full_document_reader_returns_none_for_unknown_document(
    tmp_path: Path,
) -> None:
    settings = DocumentPreparationSettings(
        raw_documents_dir=tmp_path,
        parsed_markdown_dir=tmp_path,
    )
    reader = ParsedMarkdownDocumentReader(
        database_session_factory=_session_factory(None),
        owner_user_id=uuid4(),
        settings=settings,
    )

    assert reader.read(_source("S1")) is None
