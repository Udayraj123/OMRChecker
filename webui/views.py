"""Jinja2 HTML views for the OMRChecker Web UI.

These handlers are intentionally thin: they read from the service layer
and render templates. Every mutation is performed by the browser via
``fetch`` against ``/api/v1/*``.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from webui.services import batches as batches_service
from webui.services import omr as omr_service
from webui.services.batches import BatchNotFound
from webui.settings import Settings, get_settings

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _compute_static_version() -> str:
    """Return a stable cache-busting token for ``/static`` assets.

    Uses the most recent mtime found under ``STATIC_DIR`` so that the
    token bumps automatically whenever any CSS/JS file changes, but stays
    stable across server restarts when nothing has changed (good for HTTP
    caching). Falls back to the server start time when the directory is
    missing or unreadable (development edge cases).
    """
    try:
        latest = 0.0
        for path in STATIC_DIR.rglob("*"):
            if path.is_file():
                mtime = path.stat().st_mtime
                if mtime > latest:
                    latest = mtime
        if latest > 0:
            return str(int(latest))
    except OSError:
        pass
    return str(int(time.time()))


STATIC_VERSION = _compute_static_version()

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["STATIC_VERSION"] = STATIC_VERSION

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request, settings: Settings = Depends(get_settings)
) -> HTMLResponse:
    all_batches = batches_service.list_batches(settings)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "batches": all_batches,
            "allow_directory_import": settings.allow_directory_import,
        },
    )


@router.get("/batches/{batch_id}", response_class=HTMLResponse)
async def batch_detail(
    batch_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    try:
        batch = batches_service.get_batch(batch_id, settings)
    except BatchNotFound:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    files = batches_service.list_files(batch_id, settings)
    template_doc = batches_service.get_json_document(batch_id, "template", settings)
    config_doc = batches_service.get_json_document(batch_id, "config", settings)
    evaluation_doc = batches_service.get_json_document(batch_id, "evaluation", settings)
    template_assets = batches_service.list_template_assets(batch_id, settings)
    results = omr_service.read_results(batch_id, settings)
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    preprocess_failures = list(metadata.get("preprocess_failures", []))

    return templates.TemplateResponse(
        request,
        "batch_detail.html",
        {
            "batch": batch,
            "files": files,
            "template_doc": template_doc,
            "config_doc": config_doc,
            "evaluation_doc": evaluation_doc,
            "template_assets": template_assets,
            "results": results,
            "preprocess_failures": preprocess_failures,
            "allow_directory_import": settings.allow_directory_import,
        },
    )


@router.get("/prefill", response_class=HTMLResponse)
async def prefill_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "prefill.html", {})


@router.get("/generate-csv", response_class=HTMLResponse)
async def generate_csv_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "generate_csv.html", {})


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "settings.html", {})
