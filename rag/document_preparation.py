from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import logging
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

import pymupdf
import pymupdf4llm
from sqlalchemy.orm import Session, sessionmaker

from rag.config import DocumentPreparationSettings, get_document_preparation_settings
from rag.database.repositories import DocumentRepository
from rag.database.session import get_session_factory
from rag.document_paths import (
    delete_file_if_exists,
    parsed_markdown_relative_path,
    relative_path_key,
    safe_relative_pdf_path,
)
from rag.progress import ProgressCallback, ProgressEmitter

logger = logging.getLogger(__name__)

_REPLACEMENT_CHARACTER = "\ufffd"
_BAD_EXTRACTION_RATIO_THRESHOLD = 0.2
_MIN_RECOVERED_WORDS_FOR_NOISY_EXTRACTION = 20
_MIN_RECOVERED_ALNUM_FOR_NOISY_EXTRACTION = 80
_WORD_PATTERN = re.compile(r"[^\W_]{2,}", re.UNICODE)
_PICTURE_OMITTED_PATTERN = re.compile(
    r"\*\*==> picture \[[^\]]+\] intentionally omitted <==\*\*(?:<br>)?"
)
_PICTURE_TEXT_BOUNDARY_PATTERN = re.compile(
    r"(?:<br>)?\*\*----- (?:Start|End) of picture text -----\*\*(?:<br>)?"
)
_OCR_DPI = 220
_MIN_OCR_CONFIDENCE = 0.3


@dataclass(frozen=True)
class _OcrLine:
    text: str
    confidence: float
    x: float
    y: float


class _Pixmap(Protocol):
    def tobytes(self, output: str) -> bytes:
        ...


class _PdfPage(Protocol):
    def get_pixmap(self, *, dpi: int, alpha: bool) -> _Pixmap:
        ...


class _PdfDocument(Protocol):
    def __iter__(self) -> Iterator[_PdfPage]:
        ...

    def close(self) -> None:
        ...


@dataclass
class PreparationSummary:
    scanned: int = 0
    prepared: int = 0
    reprepared: int = 0
    pruned: int = 0
    skipped: int = 0
    duplicates_removed: int = 0
    failed: int = 0
    failed_paths: list[str] = field(default_factory=list)

    def as_report_line(self) -> str:
        return (
            f"scanned={self.scanned} "
            f"prepared={self.prepared} "
            f"reprepared={self.reprepared} "
            f"pruned={self.pruned} "
            f"skipped={self.skipped} "
            f"duplicates_removed={self.duplicates_removed} "
            f"failed={self.failed}"
        )


def discover_raw_documents(raw_documents_dir: Path) -> list[Path]:
    if not raw_documents_dir.exists():
        return []

    return sorted(
        path
        for path in raw_documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def calculate_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_pdf_to_markdown(pdf_path: Path) -> str:
    markdown = _to_markdown_with_ocr_fallback(pdf_path)
    if not isinstance(markdown, str):
        raise TypeError("Expected markdown parser to return a string")
    cleaned_markdown = _clean_extracted_markdown(markdown)

    if _is_low_quality_extraction(markdown, cleaned_markdown):
        markdown, cleaned_markdown = _recover_low_quality_extraction(
            pdf_path=pdf_path,
            markdown=markdown,
            cleaned_markdown=cleaned_markdown,
        )

    _validate_extraction_quality(
        pdf_path=pdf_path,
        raw_markdown=markdown,
        cleaned_markdown=cleaned_markdown,
    )
    return cleaned_markdown


def _recover_low_quality_extraction(
    *,
    pdf_path: Path,
    markdown: str,
    cleaned_markdown: str,
) -> tuple[str, str]:
    if not _has_ocr_support():
        return markdown, cleaned_markdown

    pymupdf_ocr_markdown = _to_pymupdf_ocr_markdown(pdf_path)
    if pymupdf_ocr_markdown is not None:
        markdown, cleaned_markdown = _select_better_markdown(
            candidate_markdown=pymupdf_ocr_markdown,
            current_markdown=markdown,
            current_cleaned_markdown=cleaned_markdown,
        )

    if not _is_low_quality_extraction(markdown, cleaned_markdown):
        return markdown, cleaned_markdown

    rapidocr_markdown = _to_markdown_with_rapidocr(pdf_path)
    if rapidocr_markdown is None:
        return markdown, cleaned_markdown

    return _select_better_markdown(
        candidate_markdown=rapidocr_markdown,
        current_markdown=markdown,
        current_cleaned_markdown=cleaned_markdown,
    )


def _to_pymupdf_ocr_markdown(pdf_path: Path) -> str | None:
    try:
        ocr_markdown = _to_markdown_with_muted_mupdf_diagnostics(
            pdf_path,
            force_ocr=True,
        )
    except RuntimeError as error:
        if "Tesseract is not installed" in str(error):
            return None
        raise

    if not isinstance(ocr_markdown, str):
        raise TypeError("Expected markdown parser to return a string")
    return ocr_markdown


def _select_better_markdown(
    *,
    candidate_markdown: str,
    current_markdown: str,
    current_cleaned_markdown: str,
) -> tuple[str, str]:
    candidate_cleaned_markdown = _clean_extracted_markdown(candidate_markdown)
    if _is_better_extraction(candidate_cleaned_markdown, current_cleaned_markdown):
        return candidate_markdown, candidate_cleaned_markdown
    return current_markdown, current_cleaned_markdown


def _to_markdown_with_ocr_fallback(pdf_path: Path) -> str:
    try:
        return _to_markdown_with_muted_mupdf_diagnostics(pdf_path)
    except RuntimeError as error:
        if "Tesseract is not installed" not in str(error):
            raise
        return _to_markdown_with_muted_mupdf_diagnostics(pdf_path, use_ocr=False)


def _to_markdown_with_muted_mupdf_diagnostics(
    pdf_path: Path,
    *,
    use_ocr: bool = True,
    force_ocr: bool = False,
) -> str:
    previous_errors = _mupdf_errors_enabled()
    previous_warnings = _mupdf_warnings_enabled()
    _set_mupdf_errors(False)
    _set_mupdf_warnings(False)

    try:
        with _suppress_output_file_descriptors():
            markdown = pymupdf4llm.to_markdown(
                str(pdf_path),
                use_ocr=use_ocr,
                force_ocr=force_ocr,
                show_progress=False,
                embed_images=False,
                write_images=False,
            )
            if not isinstance(markdown, str):
                raise TypeError("Expected markdown parser to return a string")
            return markdown
    finally:
        _set_mupdf_errors(previous_errors)
        _set_mupdf_warnings(previous_warnings)


def _mupdf_errors_enabled() -> bool:
    return bool(pymupdf.TOOLS.mupdf_display_errors())  # type: ignore[no-untyped-call]


def _mupdf_warnings_enabled() -> bool:
    return bool(pymupdf.TOOLS.mupdf_display_warnings())  # type: ignore[no-untyped-call]


def _set_mupdf_errors(enabled: bool) -> None:
    pymupdf.TOOLS.mupdf_display_errors(enabled)  # type: ignore[no-untyped-call]


def _set_mupdf_warnings(enabled: bool) -> None:
    pymupdf.TOOLS.mupdf_display_warnings(enabled)  # type: ignore[no-untyped-call]


def _to_markdown_with_rapidocr(pdf_path: Path) -> str | None:
    rapidocr_engine = _rapidocr_engine()
    if rapidocr_engine is None:
        return None

    ocr_engine = rapidocr_engine()
    pages: list[str] = []
    document = cast(
        _PdfDocument,
        pymupdf.open(pdf_path),  # type: ignore[no-untyped-call]
    )
    try:
        for page in document:
            lines = _rapidocr_page_lines(ocr_engine, page)
            if lines:
                pages.append("\n".join(line.text for line in lines))
    finally:
        document.close()

    if not pages:
        return None
    return "\n\n".join(pages)


def _rapidocr_engine() -> type[Any] | None:
    if importlib.util.find_spec("rapidocr_onnxruntime") is None:
        return None

    try:
        from rapidocr_onnxruntime import RapidOCR
    except (ImportError, OSError) as error:
        logger.warning("RapidOCR is unavailable: %s", error)
        return None

    return cast(type[Any], RapidOCR)


def _rapidocr_page_lines(ocr_engine: Any, page: _PdfPage) -> list[_OcrLine]:
    pixmap = page.get_pixmap(dpi=_OCR_DPI, alpha=False)
    result, _elapsed = ocr_engine(pixmap.tobytes("png"))
    lines = [_ocr_line(item) for item in result or []]
    return sorted(
        [line for line in lines if line is not None],
        key=lambda line: (line.y, line.x),
    )


def _ocr_line(item: Any) -> _OcrLine | None:
    if not isinstance(item, list) or len(item) < 3:
        return None

    text = item[1]
    confidence = item[2]
    if not isinstance(text, str):
        return None
    if not isinstance(confidence, Real):
        return None
    if confidence < _MIN_OCR_CONFIDENCE:
        return None
    stripped_text = text.strip()
    if not stripped_text:
        return None

    x, y = _ocr_line_origin(item[0])
    return _OcrLine(text=stripped_text, confidence=float(confidence), x=x, y=y)


def _ocr_line_origin(points: Any) -> tuple[float, float]:
    if not isinstance(points, list) or not points:
        return 0.0, 0.0

    x_coordinates = []
    y_coordinates = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        x_coordinate = _real_value(point[0])
        y_coordinate = _real_value(point[1])
        if x_coordinate is None or y_coordinate is None:
            continue
        x_coordinates.append(x_coordinate)
        y_coordinates.append(y_coordinate)

    if not x_coordinates or not y_coordinates:
        return 0.0, 0.0
    return min(x_coordinates), min(y_coordinates)


def _real_value(value: Any) -> float | None:
    if not isinstance(value, Real):
        return None
    return float(value)


@contextlib.contextmanager
def _suppress_output_file_descriptors() -> Iterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)

    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(stdout_fd)
        os.close(devnull_fd)


def _clean_extracted_markdown(markdown: str) -> str:
    cleaned = _PICTURE_OMITTED_PATTERN.sub("", markdown)
    cleaned = _PICTURE_TEXT_BOUNDARY_PATTERN.sub("", cleaned)
    cleaned_lines = []

    for line in cleaned.splitlines():
        if _is_garbage_extraction_line(line):
            continue
        cleaned_lines.append(line.replace(_REPLACEMENT_CHARACTER, ""))

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def _is_garbage_extraction_line(line: str) -> bool:
    visible_characters = [character for character in line if not character.isspace()]
    if not visible_characters:
        return False

    replacement_ratio = line.count(_REPLACEMENT_CHARACTER) / len(visible_characters)
    control_ratio = _control_character_count(line) / len(visible_characters)
    return replacement_ratio >= 0.25 or control_ratio >= 0.1


def _validate_extraction_quality(
    *,
    pdf_path: Path,
    raw_markdown: str,
    cleaned_markdown: str,
) -> None:
    if not _is_low_quality_extraction(raw_markdown, cleaned_markdown):
        return

    raise ValueError(
        f"Parsed markdown quality is too low for {pdf_path.name}; OCR is required"
    )


def _is_low_quality_extraction(raw_markdown: str, cleaned_markdown: str) -> bool:
    raw_visible_count = _visible_character_count(raw_markdown)
    if raw_visible_count == 0:
        return False

    bad_character_ratio = (
        raw_markdown.count(_REPLACEMENT_CHARACTER)
        + _control_character_count(raw_markdown)
    ) / raw_visible_count
    if bad_character_ratio < _BAD_EXTRACTION_RATIO_THRESHOLD:
        return False

    return (
        _recovered_word_count(cleaned_markdown)
        < _MIN_RECOVERED_WORDS_FOR_NOISY_EXTRACTION
        or _alnum_count(cleaned_markdown)
        < _MIN_RECOVERED_ALNUM_FOR_NOISY_EXTRACTION
    )


def _visible_character_count(text: str) -> int:
    return sum(not character.isspace() for character in text)


def _control_character_count(text: str) -> int:
    return sum(
        ord(character) < 32 and character not in "\n\r\t" for character in text
    )


def _recovered_word_count(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _alnum_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _is_better_extraction(candidate_markdown: str, current_markdown: str) -> bool:
    if _recovered_word_count(candidate_markdown) > _recovered_word_count(
        current_markdown
    ):
        return True

    return _alnum_count(candidate_markdown) > _alnum_count(current_markdown)


def _has_ocr_support() -> bool:
    if _rapidocr_engine() is not None:
        return True

    try:
        tessdata = pymupdf.get_tessdata()  # type: ignore[no-untyped-call]
        return tessdata is not None
    except RuntimeError:
        return False


def normalize_markdown(markdown: str) -> str:
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    if not normalized:
        raise ValueError("Parsed markdown is empty")

    return f"{normalized}\n"


def _write_markdown(markdown_path: Path, markdown: str) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = markdown_path.with_suffix(f"{markdown_path.suffix}.tmp")
    temp_path.write_text(markdown, encoding="utf-8")
    temp_path.replace(markdown_path)


def _delete_file_if_exists(file_path: Path) -> bool:
    return delete_file_if_exists(file_path)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _summary_payload(summary: PreparationSummary) -> dict[str, Any]:
    return asdict(summary)


def _processing_error_message(error: BaseException) -> str:
    message = str(error).strip()
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message}"


def prepare_documents(
    *,
    parser: Callable[[Path], str] = parse_pdf_to_markdown,
    now: Callable[[], datetime] = _utc_now,
    settings: DocumentPreparationSettings | None = None,
    progress_callback: ProgressCallback | None = None,
    selected_raw_paths: Iterable[str] | None = None,
    database_session_factory: sessionmaker[Session] | None = None,
    owner_user_id: UUID | None = None,
) -> PreparationSummary:
    settings = settings or get_document_preparation_settings()
    database_session_factory = database_session_factory or _optional_session_factory()
    selected_raw_keys = _selected_raw_keys(selected_raw_paths)
    progress = ProgressEmitter(progress_callback)
    logger.info("Discovering raw documents: directory=%s", settings.raw_documents_dir)
    progress.emit(
        step="discover",
        status="running",
        message="Discovering raw PDF documents",
    )
    raw_documents_by_key = _raw_documents_by_key(settings)
    work_items = _work_items(
        settings=settings,
        raw_documents_by_key=raw_documents_by_key,
        selected_raw_keys=selected_raw_keys,
        database_session_factory=database_session_factory,
        owner_user_id=owner_user_id,
    )
    logger.info("Discovered raw documents: files=%d", len(work_items))
    progress.emit(
        step="discover",
        status="succeeded",
        message="Discovered raw PDF documents",
        counters={"files": len(work_items)},
    )
    summary = PreparationSummary(scanned=len(work_items))
    if (
        selected_raw_keys is None
        and owner_user_id is not None
        and database_session_factory is not None
    ):
        summary.pruned = _mark_stale_missing_documents(
            database_session_factory=database_session_factory,
            owner_user_id=owner_user_id,
            existing_relative_raw_paths=set(raw_documents_by_key),
        )
    if summary.pruned:
        logger.info("Marked stale missing raw documents: count=%d", summary.pruned)
        progress.emit(
            step="prepare",
            status="succeeded",
            message="Marked stale missing raw documents",
            counters={"pruned": summary.pruned},
        )

    for index, item in enumerate(work_items, start=1):
        relative_raw_key = item.relative_raw_path
        raw_document_path = item.raw_path
        parsed_relative_path = parsed_markdown_relative_path(Path(relative_raw_key))
        parsed_markdown_path = settings.parsed_markdown_dir / parsed_relative_path
        logger.info(
            "Preparing raw document %d/%d: %s",
            index,
            len(work_items),
            relative_raw_key,
        )
        progress.emit(
            step="prepare",
            status="running",
            message=f"Preparing {relative_raw_key}",
            counters={"index": index, "total": len(work_items)},
        )

        try:
            markdown = normalize_markdown(parser(raw_document_path))
        except Exception as error:
            error_message = _processing_error_message(error)
            logger.exception("Failed to prepare raw document: %s", relative_raw_key)
            _delete_file_if_exists(parsed_markdown_path)
            if item.previous_parsed_markdown_path:
                _delete_file_if_exists(
                    settings.parsed_markdown_dir / item.previous_parsed_markdown_path
                )
            summary.failed += 1
            summary.failed_paths.append(relative_raw_key)
            _mark_failed_document(
                database_session_factory=database_session_factory,
                owner_user_id=owner_user_id,
                relative_raw_path=relative_raw_key,
                original_filename=item.original_filename,
                byte_size=_file_size(raw_document_path),
                failed_at=now(),
                processing_error=error_message,
            )
            progress.emit(
                step="prepare",
                status="failed",
                message=f"Failed to prepare {relative_raw_key}",
                counters={"index": index, "total": len(work_items)},
                result={"path": relative_raw_key, "error": error_message},
            )
            continue

        content_hash = calculate_text_sha256(markdown)
        if item.previous_content_hash == content_hash and parsed_markdown_path.exists():
            _mark_prepared_document(
                database_session_factory=database_session_factory,
                owner_user_id=owner_user_id,
                relative_raw_path=relative_raw_key,
                original_filename=item.original_filename,
                parsed_markdown_path=parsed_relative_path.as_posix(),
                content_hash=content_hash,
                byte_size=_file_size(raw_document_path),
                processed_at=item.processed_at,
            )
            summary.skipped += 1
            logger.info("Skipped unchanged raw document: %s", relative_raw_key)
            progress.emit(
                step="prepare",
                status="skipped",
                message=f"Skipped unchanged {relative_raw_key}",
                counters={"index": index, "total": len(work_items)},
                result={"path": relative_raw_key},
            )
            continue

        duplicate_of = _duplicate_original_filename(
            database_session_factory=database_session_factory,
            owner_user_id=owner_user_id,
            content_hash=content_hash,
            relative_raw_path=relative_raw_key,
            is_fresh_document=item.previous_content_hash is None,
        )
        if duplicate_of is not None:
            _remove_duplicate_document(
                database_session_factory=database_session_factory,
                relative_raw_path=relative_raw_key,
                raw_document_path=raw_document_path,
                parsed_markdown_path=parsed_markdown_path,
                raw_documents_dir=settings.raw_documents_dir,
            )
            summary.duplicates_removed += 1
            logger.info(
                "Removed duplicate raw document: %s duplicate_of=%s",
                relative_raw_key,
                duplicate_of,
            )
            progress.emit(
                step="prepare",
                status="skipped",
                message=(
                    f"Removed duplicate {relative_raw_key} "
                    f"(same content as {duplicate_of})"
                ),
                counters={"index": index, "total": len(work_items)},
                result={"path": relative_raw_key, "duplicate_of": duplicate_of},
            )
            continue

        _write_markdown(parsed_markdown_path, markdown)
        _mark_prepared_document(
            database_session_factory=database_session_factory,
            owner_user_id=owner_user_id,
            relative_raw_path=relative_raw_key,
            original_filename=item.original_filename,
            parsed_markdown_path=parsed_relative_path.as_posix(),
            content_hash=content_hash,
            byte_size=_file_size(raw_document_path),
            processed_at=now(),
        )

        if item.previous_content_hash is None:
            summary.prepared += 1
            logger.info(
                "Prepared raw document: %s parsed=%s",
                relative_raw_key,
                parsed_relative_path.as_posix(),
            )
            progress.emit(
                step="prepare",
                status="succeeded",
                message=f"Prepared {relative_raw_key}",
                counters={"index": index, "total": len(work_items)},
                result={
                    "path": relative_raw_key,
                    "parsed": parsed_relative_path.as_posix(),
                },
            )
        else:
            summary.reprepared += 1
            logger.info(
                "Reprepared raw document: %s parsed=%s",
                relative_raw_key,
                parsed_relative_path.as_posix(),
            )
            progress.emit(
                step="prepare",
                status="succeeded",
                message=f"Reprepared {relative_raw_key}",
                counters={"index": index, "total": len(work_items)},
                result={
                    "path": relative_raw_key,
                    "parsed": parsed_relative_path.as_posix(),
                },
            )

    logger.info("Finished document preparation: %s", summary.as_report_line())
    progress.emit(
        step="prepare",
        status="failed" if summary.failed else "succeeded",
        message="Finished document preparation",
        result=_summary_payload(summary),
    )
    return summary


def _selected_raw_keys(selected_raw_paths: Iterable[str] | None) -> set[str] | None:
    if selected_raw_paths is None:
        return None
    return {
        relative_path_key(safe_relative_pdf_path(relative_raw_path))
        for relative_raw_path in selected_raw_paths
    }


@dataclass(frozen=True)
class _WorkItem:
    relative_raw_path: str
    raw_path: Path
    original_filename: str
    previous_content_hash: str | None
    previous_parsed_markdown_path: str | None
    processed_at: datetime | None


def _raw_documents_by_key(settings: DocumentPreparationSettings) -> dict[str, Path]:
    return {
        relative_path_key(path.relative_to(settings.raw_documents_dir)): path
        for path in discover_raw_documents(settings.raw_documents_dir)
    }


def _work_items(
    *,
    settings: DocumentPreparationSettings,
    raw_documents_by_key: dict[str, Path],
    selected_raw_keys: set[str] | None,
    database_session_factory: sessionmaker[Session] | None,
    owner_user_id: UUID | None,
) -> list[_WorkItem]:
    if owner_user_id is not None and database_session_factory is not None:
        return _database_work_items(
            raw_documents_by_key=raw_documents_by_key,
            selected_raw_keys=selected_raw_keys,
            database_session_factory=database_session_factory,
            owner_user_id=owner_user_id,
        )

    selected_keys = selected_raw_keys or set(raw_documents_by_key)
    missing = sorted(selected_keys - set(raw_documents_by_key))
    if missing:
        raise ValueError(f"Selected raw documents do not exist: {', '.join(missing)}")
    return [
        _WorkItem(
            relative_raw_path=key,
            raw_path=raw_documents_by_key[key],
            original_filename=Path(key).name,
            previous_content_hash=_existing_markdown_hash(
                settings.parsed_markdown_dir / parsed_markdown_relative_path(Path(key))
            ),
            previous_parsed_markdown_path=parsed_markdown_relative_path(Path(key)).as_posix(),
            processed_at=None,
        )
        for key in sorted(selected_keys)
    ]


def _database_work_items(
    *,
    raw_documents_by_key: dict[str, Path],
    selected_raw_keys: set[str] | None,
    database_session_factory: sessionmaker[Session],
    owner_user_id: UUID,
) -> list[_WorkItem]:
    with database_session_factory() as session:
        documents = DocumentRepository(session).list_for_owner(owner_user_id)
        documents_by_path = {document.relative_raw_path: document for document in documents}
        if selected_raw_keys is None:
            selected_keys = set(documents_by_path) & set(raw_documents_by_key)
        else:
            selected_keys = selected_raw_keys
            missing = sorted(selected_keys - set(raw_documents_by_key))
            if missing:
                raise ValueError(
                    f"Selected raw documents do not exist: {', '.join(missing)}"
                )
        unowned = sorted(selected_keys - set(documents_by_path))
        if unowned:
            raise ValueError(f"Selected raw documents are not owned: {', '.join(unowned)}")
        return [
            _WorkItem(
                relative_raw_path=key,
                raw_path=raw_documents_by_key[key],
                original_filename=documents_by_path[key].original_filename,
                previous_content_hash=documents_by_path[key].content_hash,
                previous_parsed_markdown_path=documents_by_path[
                    key
                ].parsed_markdown_path,
                processed_at=documents_by_path[key].processed_at,
            )
            for key in sorted(selected_keys)
        ]


def _duplicate_original_filename(
    *,
    database_session_factory: sessionmaker[Session] | None,
    owner_user_id: UUID | None,
    content_hash: str,
    relative_raw_path: str,
    is_fresh_document: bool,
) -> str | None:
    if not is_fresh_document:
        return None
    if database_session_factory is None or owner_user_id is None:
        return None
    with database_session_factory() as session:
        original = DocumentRepository(session).get_duplicate_by_content_hash(
            owner_user_id=owner_user_id,
            content_hash=content_hash,
            exclude_relative_raw_path=relative_raw_path,
        )
        return None if original is None else original.original_filename


def _remove_duplicate_document(
    *,
    database_session_factory: sessionmaker[Session] | None,
    relative_raw_path: str,
    raw_document_path: Path,
    parsed_markdown_path: Path,
    raw_documents_dir: Path,
) -> None:
    _delete_file_if_exists(parsed_markdown_path)
    _delete_file_if_exists(raw_document_path)
    _remove_empty_parent(raw_document_path, stop_at=raw_documents_dir)
    if database_session_factory is None:
        return
    with database_session_factory() as session:
        repository = DocumentRepository(session)
        document = repository.get_by_relative_raw_path(relative_raw_path)
        if document is not None:
            repository.delete_document(document)
        session.commit()


def _remove_empty_parent(path: Path, *, stop_at: Path) -> None:
    parent = path.parent
    try:
        if parent != stop_at:
            parent.rmdir()
    except OSError:
        return


def _existing_markdown_hash(markdown_path: Path) -> str | None:
    if not markdown_path.exists():
        return None
    return calculate_text_sha256(markdown_path.read_text(encoding="utf-8"))


def _file_size(file_path: Path) -> int | None:
    return file_path.stat().st_size if file_path.exists() else None


def _mark_prepared_document(
    *,
    database_session_factory: sessionmaker[Session] | None,
    owner_user_id: UUID | None,
    relative_raw_path: str,
    original_filename: str,
    parsed_markdown_path: str,
    content_hash: str,
    byte_size: int | None,
    processed_at: datetime | None,
) -> None:
    if database_session_factory is None or owner_user_id is None:
        return
    with database_session_factory() as session:
        repository = DocumentRepository(session)
        repository.upsert_prepared_document(
            owner_user_id=owner_user_id,
            relative_raw_path=relative_raw_path,
            original_filename=original_filename,
            parsed_markdown_path=parsed_markdown_path,
            content_hash=content_hash,
            byte_size=byte_size,
            processed_at=processed_at,
            status="prepared",
        )
        session.commit()


def _mark_failed_document(
    *,
    database_session_factory: sessionmaker[Session] | None,
    owner_user_id: UUID | None,
    relative_raw_path: str,
    original_filename: str,
    byte_size: int | None,
    failed_at: datetime,
    processing_error: str,
) -> None:
    if database_session_factory is None or owner_user_id is None:
        return
    with database_session_factory() as session:
        DocumentRepository(session).mark_processing_failed(
            owner_user_id=owner_user_id,
            relative_raw_path=relative_raw_path,
            original_filename=original_filename,
            byte_size=byte_size,
            processed_at=failed_at,
            processing_error=processing_error,
        )
        session.commit()


def _mark_stale_missing_documents(
    *,
    database_session_factory: sessionmaker[Session],
    owner_user_id: UUID,
    existing_relative_raw_paths: set[str],
) -> int:
    with database_session_factory() as session:
        count = DocumentRepository(session).mark_stale_for_missing_raw_paths(
            owner_user_id=owner_user_id,
            existing_relative_raw_paths=existing_relative_raw_paths,
        )
        session.commit()
        return count


def _optional_session_factory() -> sessionmaker[Session] | None:
    try:
        return get_session_factory()
    except ValueError:
        return None
