from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from dashboard.auth import (
    clear_session_cookie,
    read_session,
    set_session_cookie,
    verify_csrf,
)
from dashboard.assets import AssetManifest
from dashboard.dependencies import (
    auth_settings,
    current_user,
    database_session_factory,
    templates,
)
from dashboard.http import redirect, template_response
from rag.database import UserRepository


router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> Response:
    if _active_session_exists(request):
        return redirect("/")
    return template_response(
        templates(request),
        request,
        "login.html",
        {"error": None},
    )


@router.post("/login", response_class=HTMLResponse)
def login(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
) -> Response:
    settings = auth_settings(request)
    with database_session_factory(request)() as session:
        user = UserRepository(session).authenticate(
            username=username,
            password=password,
        )

    if user is None:
        return template_response(
            templates(request),
            request,
            "login.html",
            {"error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    response = redirect("/")
    set_session_cookie(
        response,
        settings,
        user_id=user.id,
        username=user.username,
    )
    return response


@router.post("/logout")
def logout(request: Request, csrf_token: str = Form("")) -> RedirectResponse:
    settings = auth_settings(request)
    verify_csrf(request, settings, token=csrf_token)
    response = redirect("/login")
    clear_session_cookie(response, settings)
    return response


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    return _application_page(request)


@router.get("/overview", response_class=HTMLResponse)
@router.get("/documents", response_class=HTMLResponse)
@router.get("/pipeline", response_class=HTMLResponse)
@router.get("/assistant", response_class=HTMLResponse)
@router.get("/retrieval", response_class=HTMLResponse)
@router.get("/llm-providers", response_class=HTMLResponse)
def application_route(request: Request) -> Response:
    return _application_page(request)


def _application_page(request: Request) -> Response:
    session = read_session(request, auth_settings(request))
    if session is None:
        return redirect("/login")
    user = current_user(request)
    assets = AssetManifest(
        manifest_path=request.app.state.frontend_manifest_path,
    ).frontend()
    return template_response(
        templates(request),
        request,
        "app.html",
        {
            "username": user.username,
            "is_admin": user.is_admin,
            "csrf_token": session.csrf_token,
            "frontend_assets": assets,
        },
    )


def _active_session_exists(request: Request) -> bool:
    if read_session(request, auth_settings(request)) is None:
        return False
    try:
        current_user(request)
    except HTTPException:
        return False
    return True
