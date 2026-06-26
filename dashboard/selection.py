from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException, Request, status

from rag.config import DocumentPreparationSettings
from rag.document_paths import relative_path_key, safe_relative_pdf_path


async def json_payload(request: Request) -> Mapping[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        return {}

    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        ) from error

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Expected JSON object",
        )
    return payload


def selected_relative_raw_paths(
    payload: Mapping[str, Any],
    *,
    required: bool,
) -> list[str] | None:
    raw_paths = payload.get("relative_raw_paths")
    if raw_paths is None:
        if required:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Select at least one document",
            )
        return None
    if not isinstance(raw_paths, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Expected relative_raw_paths list",
        )

    selected_paths = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Expected relative_raw_paths list of strings",
            )
        try:
            selected_paths.append(relative_path_key(safe_relative_pdf_path(raw_path)))
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    selected_paths = list(dict.fromkeys(selected_paths))
    if not selected_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Select at least one document",
        )
    return selected_paths


def ensure_raw_documents_exist(
    relative_raw_paths: list[str],
    *,
    settings: DocumentPreparationSettings,
) -> None:
    missing = []
    for relative_raw_path in relative_raw_paths:
        raw_path = settings.raw_documents_dir / safe_relative_pdf_path(
            relative_raw_path
        )
        if not raw_path.is_file():
            missing.append(relative_raw_path)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Selected PDF not found: {', '.join(missing)}",
        )
