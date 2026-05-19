"""FastAPI application factory for the OMRChecker Web UI."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from webui.api import router as api_router
from webui.settings import get_settings
from webui.views import router as views_router

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_app() -> FastAPI:
    """Build the FastAPI app exposing both the JSON API and the HTML UI."""
    settings = get_settings()
    settings.ensure_storage()

    app = FastAPI(
        title="OMRChecker Web UI",
        description=(
            "Thin FastAPI wrapper around the OMRChecker engine. "
            "Manage batches of scanned sheets, upload images, set "
            "template/config/evaluation JSON, and run OMR. "
            "All UI actions are available as JSON via /api/v1."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next) -> Response:
        """Inject security and correlation headers on every response."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'"
        )
        return response

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    app.include_router(api_router)
    app.include_router(views_router)

    return app


app = create_app()
