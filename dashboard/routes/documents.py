from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

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


@router.get("/documents")
def api_documents(request: Request) -> dict[str, Any]:
    user = current_user(request)
    records, qdrant_error = document_catalog(request).list_records(
        document_settings(request),
        owner_user_id=user.id,
    )
    return {
        "documents": [record.as_dict() for record in records],
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


@router.post("/documents/upload")
async def upload_document(
    request: Request,
    files: list[UploadFile] = File(..., alias="file"),
    csrf_token: str = Form(""),
) -> JSONResponse:
    user = current_user(request)
    verify_csrf(request, auth_settings(request), token=csrf_token)
    try:
        uploads = document_storage(request).save_uploaded_pdfs(
            [
                (uploaded_file.filename, await uploaded_file.read())
                for uploaded_file in files
            ],
            owner=user,
            settings=document_settings(request),
        )
    except DocumentOperationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return JSONResponse(
        {"ok": True, "uploads": uploads, "uploaded_count": len(uploads)}
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
