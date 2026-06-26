from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from dashboard.api_models import (
    DocumentChunksResponse,
    DocumentDeleteRequest,
    DocumentIndexResponse,
    DocumentMarkdownResponse,
    DocumentPageResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from dashboard.auth import verify_csrf
from dashboard.dependencies import (
    auth_settings,
    current_user,
    document_catalog,
    document_settings,
    document_storage,
    process_detail_service,
)
from dashboard.selection import json_payload, selected_relative_raw_paths
from dashboard.services.document_storage import (
    DocumentOperationError,
    DocumentPermissionError,
)


router = APIRouter(prefix="/api")


@router.get("/status")
def api_status(request: Request) -> dict[str, Any]:
    user = current_user(request)
    return document_catalog(request).dashboard_status(
        document_settings(request),
        owner_user_id=user.id,
    ).as_dict()


@router.get("/documents", response_model=DocumentPageResponse)
def api_documents(
    request: Request,
    page: int = 1,
    page_size: int = 25,
    query: str | None = None,
    status_filter: str | None = Query(default=None, alias="status"),
    sort: str = "updated_at",
    direction: str = "desc",
) -> dict[str, Any]:
    user = current_user(request)
    normalized_page = max(1, page)
    normalized_page_size = max(1, min(page_size, 100))
    records, total, status_counts, qdrant_error = (
        document_catalog(request).paginated_records(
            document_settings(request),
            owner_user_id=user.id,
            page=normalized_page,
            page_size=normalized_page_size,
            query=query,
            status=status_filter,
            sort=sort,
            direction="asc" if direction == "asc" else "desc",
        )
    )
    return {
        "ok": True,
        "documents": [record.as_dict() for record in records],
        "page": normalized_page,
        "page_size": normalized_page_size,
        "total": total,
        "pages": max(1, (total + normalized_page_size - 1) // normalized_page_size),
        "status_counts": status_counts,
        "qdrant_error": qdrant_error,
    }


@router.get("/documents/process")
def api_document_process(
    request: Request,
    relative_raw_path: str,
) -> dict[str, Any]:
    user = current_user(request)
    try:
        return process_detail_service(request).document_process_detail(
            relative_raw_path,
            settings=document_settings(request),
            owner_user_id=user.id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def api_document(request: Request, document_id: UUID) -> dict[str, Any]:
    user = current_user(request)
    record = document_catalog(request).record_by_id(
        document_settings(request),
        owner_user_id=user.id,
        document_id=document_id,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return {"ok": True, "document": record.as_dict()}


@router.get(
    "/documents/{document_id}/markdown",
    response_model=DocumentMarkdownResponse,
)
def api_document_markdown(request: Request, document_id: UUID) -> dict[str, Any]:
    user = current_user(request)
    try:
        record, markdown = process_detail_service(request).markdown_by_id(
            document_id,
            settings=document_settings(request),
            owner_user_id=user.id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return {
        "ok": True,
        "document": record.as_dict(),
        "markdown": markdown,
    }


@router.get(
    "/documents/{document_id}/chunks",
    response_model=DocumentChunksResponse,
)
def api_document_chunks(
    request: Request,
    document_id: UUID,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    user = current_user(request)
    try:
        record, chunks = process_detail_service(request).chunks_by_id(
            document_id,
            settings=document_settings(request),
            owner_user_id=user.id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    normalized_page = max(1, page)
    normalized_page_size = max(1, min(page_size, 100))
    start = (normalized_page - 1) * normalized_page_size
    return {
        "ok": True,
        "document": record.as_dict(),
        "chunks": chunks[start : start + normalized_page_size],
        "page": normalized_page,
        "page_size": normalized_page_size,
        "total": len(chunks),
    }


@router.get(
    "/documents/{document_id}/index-points",
    response_model=DocumentIndexResponse,
)
def api_document_index_points(
    request: Request,
    document_id: UUID,
) -> dict[str, Any]:
    user = current_user(request)
    try:
        record, index = process_detail_service(request).index_points_by_id(
            document_id,
            settings=document_settings(request),
            owner_user_id=user.id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return {
        "ok": True,
        "document": record.as_dict(),
        "index": index,
    }


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    files: list[UploadFile] = File(..., alias="file"),
    csrf_token: str = Form(""),
) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request), token=csrf_token)
    uploads: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for uploaded_file in files:
        file_name = uploaded_file.filename or "<unnamed>"
        try:
            upload = document_storage(request).save_uploaded_pdf(
                file_name=uploaded_file.filename,
                content=await uploaded_file.read(),
                owner=user,
                settings=document_settings(request),
            )
            uploads.append(upload)
            results.append({"file_name": file_name, "status": "uploaded", **upload})
        except DocumentOperationError as error:
            results.append(
                {
                    "file_name": file_name,
                    "status": "failed",
                    "error": str(error),
                }
            )
    failed_count = len(results) - len(uploads)
    if not uploads and failed_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(results[0]["error"]),
        )
    return JSONResponse(
        {
            "ok": failed_count == 0,
            "uploads": uploads,
            "results": results,
            "uploaded_count": len(uploads),
            "failed_count": failed_count,
        },
        status_code=(
            status.HTTP_207_MULTI_STATUS
            if failed_count
            else status.HTTP_200_OK
        ),
    )


@router.post("/documents/delete")
def delete_document_endpoint(
    request: Request,
    relative_raw_path: str = Form(""),
    csrf_token: str = Form(""),
) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request), token=csrf_token)
    try:
        result = document_storage(request).delete_document(
            relative_raw_path,
            owner=user,
            settings=document_settings(request),
        )
    except DocumentPermissionError as error:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return JSONResponse({"ok": True, **result})


@router.post("/documents/delete-selected")
async def delete_selected_documents_endpoint(request: Request) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request))
    selected_paths = selected_relative_raw_paths(
        await json_payload(request),
        required=True,
    )
    if selected_paths is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents selected",
        )
    try:
        result = document_storage(request).delete_documents(
            selected_paths,
            owner=user,
            settings=document_settings(request),
        )
    except DocumentPermissionError as error:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return JSONResponse({"ok": True, **result})


@router.post("/documents/delete-by-id")
def delete_documents_by_id_endpoint(
    request: Request,
    payload: DocumentDeleteRequest,
) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request))
    try:
        result = document_storage(request).delete_documents_by_id(
            payload.document_ids,
            owner=user,
            settings=document_settings(request),
        )
    except DocumentPermissionError as error:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(error),
        ) from error
    except (DocumentOperationError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return JSONResponse({"ok": True, **result})
