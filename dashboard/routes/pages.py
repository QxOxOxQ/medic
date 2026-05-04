from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from dashboard.auth import (
    clear_session_cookie,
    read_session,
    set_session_cookie,
    verify_csrf,
)
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
    session = read_session(request, auth_settings(request))
    if session is None:
        return redirect("/login")
    user = current_user(request)

    return template_response(
        templates(request),
        request,
        "index.html",
        {
            "username": user.username,
            "csrf_token": session.csrf_token,
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
