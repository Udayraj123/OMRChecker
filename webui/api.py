"""JSON API router for the OMRChecker Web UI.

Exposed under ``/api/v1``. All mutations live here so that the HTML UI
and any third-party API consumer go through identical codepaths.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import secrets
import threading
import time as _time_mod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import csv
import io
import os
import tempfile

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from webui.schemas import (
    Batch,
    BatchCreate,
    BatchRotationUpdate,
    BatchStatus,
    BatchStatusResponse,
    DirectoryImportRequest,
    FileRef,
    ImportResult,
    ProcessAccepted,
    ResultsPayload,
    TemplateAssetRef,
)
from webui.services import batches as batches_service
from webui.services import omr as omr_service
from webui.services import prefill as prefill_service
from webui.services import presets as presets_service
from webui.services.scan_simulation import normalize_realism_preset
from webui import log_stream
from webui.schemas_settings import (
    RuntimeSettingsResponse,
    RuntimeSettingsUpdate,
    SettingsMetaResponse,
    build_meta_response,
)
from webui.services.batches import BatchNotFound, InvalidBatchRequest
from webui.settings import (
    RUNTIME_MUTABLE_SETTINGS,
    Settings,
    _load_overrides,
    get_settings,
    reload_settings,
    write_overrides,
)

router = APIRouter(prefix="/api/v1", tags=["omr"])


# Built-in template/config for sheets produced by the Prefill page.
#
# The generated sheets include four ArUco corner markers, but the OMR engine
# still needs a template to map the cropped page into fields and bubbles.
# Auto-attaching these documents keeps the generated-sheet workflow one-click:
# generate/download prefilled sheets -> upload/process without manually adding
# template.json/config.json.
_PREFILLED_25Q_TEMPLATE: dict[str, Any] = {
    "pageDimensions": [666, 515],
    "bubbleDimensions": [10, 10],
    "customLabels": {"CandidateNumber": ["cand1..10"]},
    "outputColumns": ["CandidateNumber", "q1..25"],
    "fieldBlocks": {
        "CandidateNumber": {
            "origin": [430, 103],
            "bubblesGap": 10.0,
            "labelsGap": 21.5,
            "fieldLabels": ["cand1..10"],
            "fieldType": "QTYPE_INT",
        },
        "q01block": {
            "origin": [53, 257],
            "bubblesGap": 20.0,
            "labelsGap": 42.0,
            "fieldLabels": ["q1..5"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q06block": {
            "origin": [181, 257],
            "bubblesGap": 20.0,
            "labelsGap": 42.0,
            "fieldLabels": ["q6..10"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q11block": {
            "origin": [310, 257],
            "bubblesGap": 19.5,
            "labelsGap": 42.0,
            "fieldLabels": ["q11..15"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q16block": {
            "origin": [435, 257],
            "bubblesGap": 20.3,
            "labelsGap": 42.0,
            "fieldLabels": ["q16..20"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
        "q21block": {
            "origin": [566, 257],
            "bubblesGap": 19.7,
            "labelsGap": 42.0,
            "fieldLabels": ["q21..25"],
            "emptyValue": "NR",
            "fieldType": "QTYPE_MCQ4",
        },
    },
    "preProcessors": [
        {
            "name": "CropOnMarkers",
            "options": {
                "type": "aruco",
                "arucoDictionary": "DICT_4X4_50",
                "arucoCornerIds": [0, 1, 2, 3],
                "preserveFullImage": True,
                "referenceMarkerCenters": [
                    [13.5, 13.2],
                    [651.5, 13.2],
                    [13.5, 499.0],
                    [651.5, 499.0],
                ],
            },
        }
    ],
}

_PREFILLED_25Q_CONFIG: dict[str, Any] = {
    "dimensions": {
        "display_height": 515,
        "display_width": 666,
        "processing_height": 515,
        "processing_width": 666,
    },
    "outputs": {"show_image_level": 0},
}

# ---------------------------------------------------------------------------
# Prefill backpressure: cap concurrent batch jobs server-side. A 5k-row PDF
# crashed the server in stress testing because each concurrent batch spawns
# its own pool of (cpu_count - 1) workers — N concurrent batches × W workers
# easily exceeds RAM. The semaphore enforces "at most this many heavy
# prefill batches at once" and excess requests get a fast HTTP 429.
_PREFILL_BATCH_LIMIT = max(1, int(os.environ.get("OMR_WEBUI_PREFILL_CONCURRENCY", "2")))
_PREFILL_BATCH_SEM = threading.BoundedSemaphore(_PREFILL_BATCH_LIMIT)

# Single-sheet endpoint also forks a process pool for PNG/PDF rendering, so
# uncapped concurrency (e.g. 50 simultaneous requests) can wedge the host.
# Allow a higher limit than batches but still bounded.
_PREFILL_SINGLE_LIMIT = max(2, int(os.environ.get("OMR_WEBUI_PREFILL_SINGLE_CONCURRENCY", "8")))
_PREFILL_SINGLE_SEM = threading.BoundedSemaphore(_PREFILL_SINGLE_LIMIT)

# /prefill/sample is fired in parallel by the in-page comparison gallery
# (4 concurrent requests on page load). Cap server-side concurrency so a
# misbehaving client (or a tight retry loop) can't pile up dozens of PNG
# renders simultaneously.
_PREFILL_SAMPLE_LIMIT = max(2, int(os.environ.get("OMR_WEBUI_PREFILL_SAMPLE_CONCURRENCY", "6")))
_PREFILL_SAMPLE_SEM = threading.BoundedSemaphore(_PREFILL_SAMPLE_LIMIT)

# Hard caps on prefill batch sizes. PDF assembly is heavier than ZIP because
# each page incurs PyMuPDF parsing overhead; ZIP just stores PNG bytes verbatim.
_PREFILL_PDF_MAX_ROWS = int(os.environ.get("OMR_WEBUI_PREFILL_PDF_MAX_ROWS", "5000"))
_PREFILL_ZIP_MAX_ROWS = int(os.environ.get("OMR_WEBUI_PREFILL_ZIP_MAX_ROWS", "10000"))
_PREFILL_CSV_MAX_BYTES = int(os.environ.get("OMR_WEBUI_PREFILL_CSV_MAX_BYTES", str(50 * 1024 * 1024)))

# Download token store: maps token -> (tmp_path, media_type, filename, expires_at)
# Tokens are single-use and expire after 10 minutes so orphaned files are cleaned up.
_DOWNLOAD_STORE: dict[str, tuple[Path, str, str, float]] = {}
_DOWNLOAD_STORE_LOCK = threading.Lock()

def _register_download(tmp_path: Path, media_type: str, filename: str) -> str:
    """Store a completed batch file and return a one-time download token."""
    token = secrets.token_urlsafe(24)
    expires_at = _time_mod.monotonic() + 600  # 10 minutes
    with _DOWNLOAD_STORE_LOCK:
        # Evict any expired tokens first
        expired = [k for k, (_, _, _, exp) in _DOWNLOAD_STORE.items() if _time_mod.monotonic() > exp]
        for k in expired:
            try:
                _DOWNLOAD_STORE[k][0].unlink(missing_ok=True)
            except OSError:
                pass
            del _DOWNLOAD_STORE[k]
        _DOWNLOAD_STORE[token] = (tmp_path, media_type, filename, expires_at)
    return token


def _is_prefilled_sheet_upload(filename: str | None) -> bool:
    """Return true for files produced by this app's Prefill page."""
    name = (filename or "").lower()
    return "prefilled_sheet" in name or "prefilled_sheets" in name


def _attach_prefilled_25q_defaults(
    batch_id: str,
    settings: Settings,
    *,
    reason: str,
) -> None:
    """Attach the built-in 25Q prefill template/config if absent.

    We never overwrite user-supplied documents. This only fills the common
    gap where the upload was generated by the app's Prefill page and therefore
    has ArUco markers but no explicit batch template yet.
    """
    attached: list[str] = []
    if batches_service.get_json_document(batch_id, "template", settings) is None:
        batches_service.save_json_document(
            batch_id,
            "template",
            copy.deepcopy(_PREFILLED_25Q_TEMPLATE),
            settings,
        )
        attached.append("template.json")
    if batches_service.get_json_document(batch_id, "config", settings) is None:
        batches_service.save_json_document(
            batch_id,
            "config",
            copy.deepcopy(_PREFILLED_25Q_CONFIG),
            settings,
        )
        attached.append("config.json")

    batches_service.update_batch_metadata(
        batch_id,
        {
            "input_profile": "prefilled_25q",
            "auto_attached_template": True,
            "auto_attached_template_reason": reason,
        },
        settings,
    )
    if attached:
        logger.info(
            "Auto-attached prefilled 25Q defaults | batch=%s | files=%s | reason=%s",
            batch_id,
            ", ".join(attached),
            reason,
        )


def _maybe_attach_prefilled_25q_defaults(
    batch_id: str,
    settings: Settings,
    *,
    reason: str,
) -> bool:
    """Attach defaults for known prefilled-sheet batches.

    Returns ``True`` when the batch is or has been marked as a prefilled 25Q
    batch. This is used by the process guard to repair existing batches that
    were uploaded before the template was auto-attached.
    """
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    if metadata.get("input_profile") != "prefilled_25q":
        return False
    _attach_prefilled_25q_defaults(batch_id, settings, reason=reason)
    return True


# ---------------------------------------------------------------------------
# Log streaming
# ---------------------------------------------------------------------------


@router.get("/logs/poll")
async def logs_poll(since: int = -1) -> dict:
    """Return log entries with sequence number > ``since``.

    Used by the front-end log panel which polls once per second over plain
    HTTP. WebView2 has known SSE buffering issues so this is the preferred
    transport in the desktop wrapper.
    """
    return log_stream.poll(since=since)


@router.get("/logs/stream")
async def logs_stream() -> StreamingResponse:
    """Server-Sent Events stream of log lines (non-WebView2 clients)."""
    return StreamingResponse(
        log_stream.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


@router.get("/system/info")
async def system_info() -> dict:
    """Return static system capabilities: GPU status and default worker count."""
    import os
    from src.utils.gpu import gpu_status, is_gpu_available
    from webui.services.omr import _default_max_workers
    return {
        "gpu_available": is_gpu_available(),
        "gpu_status": gpu_status(),
        "cpu_count": os.cpu_count(),
        "default_max_workers": _default_max_workers(),
    }


# ---------------------------------------------------------------------------
# Runtime-mutable settings (used by the /settings UI)
# ---------------------------------------------------------------------------


def _settings_response(settings: Settings) -> RuntimeSettingsResponse:
    """Project the live :class:`Settings` instance into the API response."""
    return RuntimeSettingsResponse(
        **{key: getattr(settings, key) for key in RUNTIME_MUTABLE_SETTINGS}
    )


@router.get("/settings", response_model=RuntimeSettingsResponse)
async def get_runtime_settings(
    settings: Settings = Depends(get_settings),
) -> RuntimeSettingsResponse:
    """Return the current value of every runtime-mutable setting."""
    return _settings_response(settings)


@router.put("/settings", response_model=RuntimeSettingsResponse)
async def update_runtime_settings(
    payload: RuntimeSettingsUpdate,
    settings: Settings = Depends(get_settings),
) -> RuntimeSettingsResponse:
    """Persist runtime overrides, reload settings, and return the new state.

    Only fields explicitly present in the request body are written. This
    lets the UI send PATCH-style partial updates (toggle a single switch)
    without round-tripping every setting on every save. Unknown keys are
    rejected by ``RuntimeSettingsUpdate`` (``extra=\"forbid\"``) so the
    allowlist is enforced at the schema layer, not by post-hoc filtering.
    """
    new_values = payload.model_dump(exclude_unset=True)
    if not new_values:
        # No-op PUT: don't touch the overrides file, just echo current state.
        return _settings_response(settings)

    # Storage_root is the SEP-friendly default location for the overrides
    # file (see ``Settings.overrides_path``). ``_load_overrides`` and
    # ``write_overrides`` both accept either storage_root or cache_root
    # and resolve the actual path through the live Settings instance.
    overrides_root = settings.storage_root
    merged = _load_overrides(overrides_root)
    # Log every actual change before writing so a failed disk write still
    # produces an audit trail of what the operator attempted.
    for key, new_val in new_values.items():
        old_val = getattr(settings, key)
        if old_val != new_val:
            logger.info(
                "Settings updated | key=%s | old=%s | new=%s",
                key, old_val, new_val,
            )
        merged[key] = new_val

    write_overrides(merged, overrides_root)
    fresh = reload_settings()
    return _settings_response(fresh)


@router.get("/settings/meta", response_model=SettingsMetaResponse)
async def get_runtime_settings_meta() -> SettingsMetaResponse:
    """Return descriptions + defaults for every mutable setting.

    The ``/settings`` UI uses this to render field labels, helper text
    under each input, and a per-field \"Reset to default\" button.
    """
    return build_meta_response()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@router.get("/presets")
async def list_presets(
    settings: Settings = Depends(get_settings),
) -> list[str]:
    """Return the names of all available presets."""
    return presets_service.list_presets(settings)


@router.get("/presets/{preset_name}")
async def get_preset(
    preset_name: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return all JSON documents (template/config/evaluation) for a preset."""
    try:
        docs = presets_service.get_preset_documents(preset_name, settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not docs:
        raise HTTPException(status_code=404, detail=f"Preset {preset_name!r} not found.")
    return docs


@router.post("/batches/{batch_id}/preset")
async def apply_preset(
    batch_id: str,
    preset_name: str = Body(..., embed=True),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Copy all files from a preset (template, config, assets) into a batch."""
    batch_root = settings.ensure_storage() / batch_id
    if not batch_root.is_dir():
        raise HTTPException(status_code=404, detail=f"Batch {batch_id!r} not found.")
    try:
        presets_service.apply_preset_to_batch(batch_root, preset_name, settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "ok", "preset": preset_name}


def _handle_errors(func):
    """Wrap service calls so our custom exceptions map to HTTP status codes."""
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except BatchNotFound as exc:
            raise HTTPException(status_code=404, detail=f"Batch not found: {exc}")
        except InvalidBatchRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return wrapper


@router.post("/batches", response_model=Batch, status_code=status.HTTP_201_CREATED)
@_handle_errors
async def create_batch(
    payload: BatchCreate,
    settings: Settings = Depends(get_settings),
) -> Batch:
    return batches_service.create_batch(payload.name, settings)


@router.get("/batches", response_model=list[Batch])
@_handle_errors
async def list_batches(settings: Settings = Depends(get_settings)) -> list[Batch]:
    return batches_service.list_batches(settings)


@router.get("/batches/{batch_id}", response_model=Batch)
@_handle_errors
async def get_batch(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> Batch:
    return batches_service.get_batch(batch_id, settings)


@router.delete("/batches/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
@_handle_errors
async def delete_batch(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> None:
    batches_service.delete_batch(batch_id, settings)


@router.put("/batches/{batch_id}/rotation", response_model=Batch)
@_handle_errors
async def update_batch_rotation(
    batch_id: str,
    payload: BatchRotationUpdate,
    settings: Settings = Depends(get_settings),
) -> Batch:
    return batches_service.set_rotation(batch_id, payload.rotation_degrees, settings)


@router.get("/batches/{batch_id}/files", response_model=list[FileRef])
@_handle_errors
async def list_files(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> list[FileRef]:
    return batches_service.list_files(batch_id, settings)


def _is_pdf_upload(upload: UploadFile) -> bool:
    """Return True when the uploaded file is a PDF by name or content-type."""
    name = (upload.filename or "").lower()
    if name.endswith(".pdf"):
        return True
    content_type = (upload.content_type or "").lower()
    return content_type == "application/pdf"


@router.post("/batches/{batch_id}/files")
@_handle_errors
async def upload_files(
    batch_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
):
    """Accept image + PDF uploads.

    Image uploads (PNG / JPG / JPEG) are written synchronously and the
    endpoint returns ``201`` with the resulting :class:`FileRef` list.

    PDF uploads are scheduled as a background task because a 5000-page PDF
    can take minutes to render; the endpoint returns ``202`` with
    ``{"processing": True, "files": [...image refs already saved...]}``.
    The frontend polls ``/batches/{batch_id}/status`` for the split
    progress and refreshes its file list when ``pdf_split_total`` returns
    to zero (i.e. the background task finished).
    """
    from fastapi.responses import JSONResponse

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    # Read all upload bytes up front: we need to know each file's size for
    # the size check anyway, and the UploadFile stream is consumed once.
    image_refs: list[FileRef] = []
    pdf_jobs: list[tuple[str, bytes]] = []
    inferred_preset: str | None = None
    has_prefilled_sheet_upload = False
    for upload in files:
        data = await upload.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File {upload.filename!r} exceeds max_upload_bytes "
                    f"({settings.max_upload_bytes} bytes)"
                ),
            )
        # Try to infer the realism preset from the filename so we have an
        # audit trail when a prefill-generated file is later debugged. This
        # is best-effort only; legitimate user uploads with these names are
        # rare. ``adversarial`` matches first because both ``moderate`` and
        # ``adversarial`` contain ``a``.
        name_lower = (upload.filename or "").lower()
        if _is_prefilled_sheet_upload(upload.filename):
            has_prefilled_sheet_upload = True
        for candidate in ("adversarial", "moderate", "subtle"):
            if candidate in name_lower:
                if inferred_preset is None:
                    inferred_preset = candidate
                break
        if _is_pdf_upload(upload):
            pdf_jobs.append((upload.filename or "upload.pdf", data))
            continue
        # Synchronous image save (fast, no rendering).
        refs = await asyncio.to_thread(
            batches_service.save_uploaded_file,
            batch_id,
            upload.filename or "upload",
            data,
            settings,
        )
        image_refs.extend(refs)

    if inferred_preset is not None:
        batches_service.update_batch_metadata(
            batch_id,
            {"inferred_realism_preset": inferred_preset},
            settings,
        )
        logger.info(
            "Inferred realism preset from upload filename | batch=%s | preset=%s",
            batch_id,
            inferred_preset,
        )

    if has_prefilled_sheet_upload:
        _attach_prefilled_25q_defaults(
            batch_id,
            settings,
            reason="upload filename matched prefilled_sheet(s)",
        )

    if pdf_jobs:
        # Schedule PDF rendering as background work. BackgroundTasks runs
        # after the response is sent in production; in TestClient it runs
        # synchronously, which is exactly what the test harness expects.
        #
        # Each task is wrapped in a closure so that any exception (corrupt
        # PDF, disk full, etc.) is caught and persisted into batch metadata
        # rather than silently swallowed by the BackgroundTasks runner.
        for filename, data in pdf_jobs:
            stem = Path(filename).stem

            def _run_pdf_split(fn=filename, d=data, s=stem):
                try:
                    batches_service.save_uploaded_file(batch_id, fn, d, settings)
                except Exception as exc:  # noqa: BLE001
                    error_msg = f"{s}: {type(exc).__name__}: {exc}"
                    logger.exception(
                        "Background PDF split failed | batch=%s | file=%s",
                        batch_id, fn,
                    )
                    batches_service._record_pdf_split_error(batch_id, settings, error_msg)

            background_tasks.add_task(_run_pdf_split)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "processing": True,
                "files": [ref.model_dump() for ref in image_refs],
                "pdf_count": len(pdf_jobs),
            },
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=[ref.model_dump() for ref in image_refs],
    )


@router.post(
    "/batches/{batch_id}/files/import",
    response_model=ImportResult,
    status_code=status.HTTP_201_CREATED,
)
@_handle_errors
async def import_from_directory(
    batch_id: str,
    payload: DirectoryImportRequest,
    settings: Settings = Depends(get_settings),
) -> ImportResult:
    imported, skipped = batches_service.import_directory(
        batch_id, payload.source_dir, payload.copy_files, settings
    )
    return ImportResult(imported=imported, skipped=skipped)


@router.delete(
    "/batches/{batch_id}/files/{filename}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@_handle_errors
async def delete_file(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> None:
    batches_service.delete_file(batch_id, filename, settings)


@router.get("/batches/{batch_id}/files/{filename}/preview")
@_handle_errors
async def preview_file(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    resolved = batches_service.resolve_input_file(batch_id, filename, settings)
    return FileResponse(resolved, filename=resolved.name)


@router.get(
    "/batches/{batch_id}/assets",
    response_model=list[TemplateAssetRef],
)
@_handle_errors
async def list_template_assets(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> list[TemplateAssetRef]:
    return batches_service.list_template_assets(batch_id, settings)


@router.post(
    "/batches/{batch_id}/assets",
    response_model=list[TemplateAssetRef],
    status_code=status.HTTP_201_CREATED,
)
@_handle_errors
async def upload_template_assets(
    batch_id: str,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> list[TemplateAssetRef]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    stored: list[TemplateAssetRef] = []
    for upload in files:
        data = await upload.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File {upload.filename!r} exceeds max_upload_bytes "
                    f"({settings.max_upload_bytes} bytes)"
                ),
            )
        stored.append(
            batches_service.save_template_asset(
                batch_id,
                upload.filename or "asset",
                data,
                settings,
            )
        )
    return stored


@router.delete(
    "/batches/{batch_id}/assets/{filename}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@_handle_errors
async def delete_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> None:
    batches_service.delete_template_asset(batch_id, filename, settings)


@router.get("/batches/{batch_id}/assets/{filename}/preview")
@_handle_errors
async def preview_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    resolved = batches_service.resolve_template_asset(batch_id, filename, settings)
    return FileResponse(resolved, filename=resolved.name)


def _make_json_endpoints(doc_name: str) -> None:
    """Attach GET/PUT routes for each optional JSON document."""

    @router.get(f"/batches/{{batch_id}}/{doc_name}", name=f"get_{doc_name}")
    @_handle_errors
    async def get_doc(
        batch_id: str, settings: Settings = Depends(get_settings)
    ) -> dict[str, Any] | None:
        return batches_service.get_json_document(batch_id, doc_name, settings)

    @router.put(f"/batches/{{batch_id}}/{doc_name}", name=f"put_{doc_name}")
    @_handle_errors
    async def put_doc(
        batch_id: str,
        content: dict[str, Any] | None = Body(
            default=None,
            description=f"Full JSON body for {doc_name}.json (null to delete)",
        ),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, str]:
        batches_service.save_json_document(batch_id, doc_name, content, settings)
        return {"status": "saved" if content is not None else "deleted"}


for _doc in ("template", "config", "evaluation"):
    _make_json_endpoints(_doc)


def _assert_batch_ready_to_run(batch: Batch, settings: Settings) -> None:
    """Fail fast with a clear message if anything would block a run."""
    if batch.file_count == 0:
        raise HTTPException(status_code=400, detail="Batch has no input images.")
    if not batch.has_template:
        if _maybe_attach_prefilled_25q_defaults(
            batch.id,
            settings,
            reason="process requested for prefilled_25q batch without template",
        ):
            # The process guard is called with a Batch snapshot that was
            # loaded before this repair, so do not inspect batch.has_template
            # again here. The following missing-asset check reads from disk.
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Batch is missing template.json. If this batch came from "
                    "the Prefill page, upload the original file whose name "
                    "starts with 'prefilled_sheet' so the built-in 25Q "
                    "template can be attached automatically."
                ),
            )
    missing = batches_service.missing_template_assets(batch.id, settings)
    if missing:
        names = ", ".join(missing)
        raise HTTPException(
            status_code=400,
            detail=(
                f"template.json references missing asset(s): {names}. "
                "Upload them under Template assets before running."
            ),
        )


@router.post(
    "/batches/{batch_id}/process",
    response_model=ProcessAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@_handle_errors
async def process_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> ProcessAccepted:
    batch = batches_service.get_batch(batch_id, settings)
    _assert_batch_ready_to_run(batch, settings)
    omr_service.queue_run(batch_id, settings)
    background_tasks.add_task(omr_service.run_batch_sync, batch_id, settings)
    return ProcessAccepted(batch_id=batch_id, status=BatchStatus.queued)


@router.post(
    "/batches/{batch_id}/cancel",
    response_model=ProcessAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@_handle_errors
async def cancel_batch(
    batch_id: str,
    settings: Settings = Depends(get_settings),
) -> ProcessAccepted:
    next_status = omr_service.request_cancel(batch_id, settings)
    return ProcessAccepted(batch_id=batch_id, status=next_status)


@router.post(
    "/batches/{batch_id}/restart",
    response_model=ProcessAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@_handle_errors
async def restart_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> ProcessAccepted:
    batch = batches_service.get_batch(batch_id, settings)
    if batch.status in {BatchStatus.queued, BatchStatus.running}:
        raise HTTPException(
            status_code=409,
            detail="Stop the current run before restarting this batch.",
        )
    _assert_batch_ready_to_run(batch, settings)

    batches_service.reset_batch_runtime_state(batch_id, settings)
    omr_service.queue_run(batch_id, settings)
    background_tasks.add_task(omr_service.run_batch_sync, batch_id, settings)
    return ProcessAccepted(batch_id=batch_id, status=BatchStatus.queued)


@router.get(
    "/batches/{batch_id}/status",
    response_model=BatchStatusResponse,
)
@_handle_errors
async def batch_status(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> BatchStatusResponse:
    import time as _time
    batch = batches_service.get_batch(batch_id, settings)
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    processed = int(metadata.get("processed_files", 0))
    total = int(metadata.get("total_files", batch.file_count))
    elapsed_s: float | None = None
    rate_per_min: float | None = None
    eta_s: float | None = None
    run_started_at = metadata.get("run_started_at")
    run_elapsed = metadata.get("run_elapsed_s")
    if run_elapsed is not None:
        # Prefer the stored final value — avoids the timer ticking on after the
        # batch finishes while the status write is still in-flight.
        elapsed_s = float(run_elapsed)
    elif batch.status.value == "running" and run_started_at is not None:
        # No checkpoint written yet; derive from wall-clock start time.
        elapsed_s = round(_time.time() - float(run_started_at), 1)
    if elapsed_s and elapsed_s > 0 and processed > 0:
        rate_per_min = round(processed / elapsed_s * 60, 1)
        if batch.status.value == "running":
            remaining = total - processed
            if rate_per_min > 0:
                eta_s = round(remaining / (processed / elapsed_s))
    return BatchStatusResponse(
        id=batch.id,
        status=batch.status,
        last_error=batch.last_error,
        file_count=batch.file_count,
        updated_at=batch.updated_at,
        processed_files=processed,
        total_files=total,
        latest_processed_file=metadata.get("latest_processed_file"),
        latest_dynamic_dimensions=metadata.get("latest_dynamic_dimensions"),
        cancel_requested=bool(metadata.get("cancel_requested", False)),
        preprocess_failures=list(metadata.get("preprocess_failures", [])),
        elapsed_s=elapsed_s,
        rate_per_min=rate_per_min,
        eta_s=eta_s,
        pdf_split_pages=int(metadata.get("pdf_split_pages", 0)),
        pdf_split_total=int(metadata.get("pdf_split_total", 0)),
        pdf_split_error=metadata.get("pdf_split_error") or None,
        pipelined_run=bool(metadata.get("pipelined_run", False)),
        # Mirror runtime-mutable auto-start settings so the frontend
        # poller can decide whether to auto-fire /process without
        # making a separate /api/v1/settings request per tick.
        auto_start_omr_with_split=settings.auto_start_omr_with_split,
        auto_start_omr_min_pages=settings.auto_start_omr_min_pages,
        auto_start_omr_require_config=settings.auto_start_omr_require_config,
        has_template=batch.has_template,
        has_config=batch.has_config,
    )


@router.get("/batches/{batch_id}/results", response_model=ResultsPayload)
@_handle_errors
async def batch_results(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> ResultsPayload:
    batches_service.get_batch(batch_id, settings)
    return omr_service.read_results(batch_id, settings)


@router.get("/batches/{batch_id}/results/download")
@_handle_errors
async def download_results(
    batch_id: str, settings: Settings = Depends(get_settings)
) -> FileResponse:
    batches_service.get_batch(batch_id, settings)
    path = omr_service.results_csv_path(batch_id, settings)
    if path is None:
        raise HTTPException(
            status_code=404, detail="No results CSV yet. Run the batch first."
        )
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{batch_id}_{path.name}",
    )


@router.get("/batches/{batch_id}/outputs/{file_path:path}")
@_handle_errors
async def download_output_file(
    batch_id: str,
    file_path: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    resolved = batches_service.resolve_output_file(batch_id, file_path, settings)
    return FileResponse(resolved, filename=resolved.name)


@router.get("/batches/{batch_id}/results/{filename}/checked")
@_handle_errors
async def get_checked_output_image(
    batch_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve the OMR-annotated output image for a given input filename.

    Searches CheckedOMRs first, then MultiMarkedFiles, then ErrorFiles so a
    single URL works regardless of which output subdirectory the engine used.
    """
    safe_name = Path(filename).name  # strip any path components
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    batch_root = batches_service.get_batch_root(batch_id, settings)
    outputs_dir = batch_root / "outputs"
    for subdir in ("CheckedOMRs", "Manual/MultiMarkedFiles", "Manual/ErrorFiles"):
        candidate = (outputs_dir / subdir / safe_name).resolve()
        try:
            candidate.relative_to(outputs_dir.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return FileResponse(candidate, filename=safe_name)
    raise HTTPException(status_code=404, detail=f"No checked output image found for {filename!r}.")


# ---------------------------------------------------------------------------
# Prefill endpoints
# ---------------------------------------------------------------------------

@router.get("/prefill/sample")
async def prefill_sample(
    preset: str = "none",
    candidate_number: str = "9010690012",
    student_name: str = "Jane Doe",
    school_name: str = "Sample School",
    exam_name: str = "Sample Exam",
) -> StreamingResponse:
    """Return an inline PNG preview of a single preset.

    Used by the in-page "Compare presets" gallery so users can see exactly
    what each realism preset produces without downloading anything. Response
    is marked ``Cache-Control: no-store`` so WebView2 / browser caches cannot
    serve a stale version after the simulator code changes.
    """
    try:
        preset_norm = normalize_realism_preset(preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not _PREFILL_SAMPLE_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Sample renderer busy; try again shortly "
                f"(max {_PREFILL_SAMPLE_LIMIT} concurrent previews)."
            ),
        )
    try:
        try:
            data = await asyncio.to_thread(
                prefill_service.generate_single_png,
                student_name, school_name, exam_name, candidate_number, preset_norm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Sample render failed: {type(exc).__name__}: {exc}",
            )
    finally:
        _PREFILL_SAMPLE_SEM.release()
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="preview_{preset_norm}.png"',
            "Cache-Control": "no-store, max-age=0",
        },
    )


@router.post("/prefill/single")
async def prefill_single(
    student_name: str = Form(...),
    school_name: str = Form(...),
    exam_name: str = Form(...),
    candidate_number: str = Form(...),
    output_format: str = Form("png"),
    realism_preset: str = Form("none"),
) -> StreamingResponse:
    """Generate a single pre-filled answer sheet and stream it as a download."""
    output_format = (output_format or "").strip().lower()
    if output_format not in {"png", "pdf"}:
        raise HTTPException(
            status_code=422,
            detail="output_format must be 'png' or 'pdf'.",
        )
    try:
        realism_preset = normalize_realism_preset(realism_preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Backpressure: bounded concurrency so a flood of requests cannot exhaust
    # the threadpool / RAM. Excess requests get a fast 429 with Retry-After.
    if not _PREFILL_SINGLE_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server busy: max {_PREFILL_SINGLE_LIMIT} concurrent single "
                "prefill requests in flight. Please retry shortly."
            ),
            headers={"Retry-After": "2"},
        )
    try:
        # Offload CPU-heavy rendering off the event loop so a flood of
        # /prefill/single requests cannot block other endpoints (e.g. health).
        # Suffix the filename with the preset (when not "none") so users can
        # immediately tell which realism preset produced a given download.
        preset_suffix = "" if realism_preset == "none" else f"_{realism_preset}"
        if output_format == "pdf":
            data = await asyncio.to_thread(
                prefill_service.generate_single_pdf,
                student_name, school_name, exam_name, candidate_number, realism_preset,
            )
            media_type = "application/pdf"
            filename = f"prefilled_sheet{preset_suffix}.pdf"
        else:
            data = await asyncio.to_thread(
                prefill_service.generate_single_png,
                student_name, school_name, exam_name, candidate_number, realism_preset,
            )
            media_type = "image/png"
            filename = f"prefilled_sheet{preset_suffix}.png"
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - never leak stack traces
        raise HTTPException(
            status_code=500,
            detail=f"Single prefill failed: {type(exc).__name__}: {exc}",
        )
    finally:
        _PREFILL_SINGLE_SEM.release()

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/prefill/batch")
async def prefill_batch(
    background_tasks: BackgroundTasks,
    csv_text: str | None = Form(default=None),
    csv_file: UploadFile | None = File(default=None),
    output_mode: str = Form("pdf"),
    realism_preset: str = Form("none"),
) -> dict:
    """Generate pre-filled answer sheets for multiple students.

    Streams generation to a server-side temp file then returns a one-time
    download token as JSON. The client uses window.location.href on the
    download URL so large files stream directly to disk (no browser buffering).
    """
    # 1) Validate output_mode early — reject unknown values explicitly.
    output_mode = (output_mode or "").strip().lower()
    if output_mode not in {"pdf", "zip"}:
        raise HTTPException(
            status_code=422,
            detail="output_mode must be 'pdf' or 'zip'.",
        )
    try:
        realism_preset = normalize_realism_preset(realism_preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not csv_text and (csv_file is None or not csv_file.filename):
        raise HTTPException(
            status_code=422,
            detail="Provide either csv_text or a csv_file.",
        )

    # 2) Bound the CSV body size BEFORE materialising it. For uploads we read
    # in chunks so a hostile client can't blow up RAM by sending a multi-GB file.
    max_bytes = _PREFILL_CSV_MAX_BYTES
    if csv_text and csv_text.strip():
        encoded = csv_text.strip().encode("utf-8")
        if len(encoded) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"CSV text exceeds the {max_bytes // (1024*1024)} MiB limit.",
            )
        raw = encoded.decode("utf-8-sig")
    else:
        # csv_text was empty/blank, so csv_file must be present (validated above).
        assert csv_file is not None
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await csv_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {csv_file.filename!r} exceeds the "
                           f"{max_bytes // (1024*1024)} MiB limit.",
                )
            chunks.append(chunk)
        try:
            raw = b"".join(chunks).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"CSV must be UTF-8 encoded: {exc}",
            )

    # 3) Parse CSV defensively. csv.DictReader raises for some malformed inputs.
    try:
        reader = csv.DictReader(io.StringIO(raw))
        rows = [row for row in reader if row]
    except csv.Error as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {exc}")
    except Exception as exc:  # noqa: BLE001 - normalise to 422
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {exc}")

    if not rows:
        raise HTTPException(status_code=422, detail="CSV contains no data rows.")

    # 4) Required column check happens once on the first row.
    required_cols = {"student_name", "school_name", "exam_name", "candidate_number"}
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {', '.join(sorted(missing_cols))}",
        )

    # 5) Row-count cap so a runaway batch can't dominate the server.
    row_cap = _PREFILL_PDF_MAX_ROWS if output_mode == "pdf" else _PREFILL_ZIP_MAX_ROWS
    if len(rows) > row_cap:
        raise HTTPException(
            status_code=422,
            detail=(
                f"CSV has {len(rows)} rows but the per-batch limit for "
                f"{output_mode.upper()} output is {row_cap}. Split the file into "
                "smaller batches or set the OMR_WEBUI_PREFILL_*_MAX_ROWS env var."
            ),
        )

    # 6) Backpressure heavy jobs server-wide.
    if not _PREFILL_BATCH_SEM.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server is already running {_PREFILL_BATCH_LIMIT} prefill batch "
                "job(s). Try again shortly."
            ),
            headers={"Retry-After": "10"},
        )

    suffix = ".pdf" if output_mode == "pdf" else ".zip"
    media_type = "application/pdf" if output_mode == "pdf" else "application/zip"
    preset_suffix = "" if realism_preset == "none" else f"_{realism_preset}"
    filename = f"prefilled_sheets{preset_suffix}{suffix}"

    # 7) Write to a temp file the response will stream from. The file is
    # deleted after the response finishes via background_tasks.
    fd, tmp_path_str = tempfile.mkstemp(prefix="prefill_", suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    def _cleanup() -> None:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        _PREFILL_BATCH_SEM.release()

    def _release_sem() -> None:
        _PREFILL_BATCH_SEM.release()

    try:
        # Offload to a thread so a long-running batch cannot block the event
        # loop and stall every other request (incl. health checks).
        if output_mode == "zip":
            meta = await asyncio.to_thread(
                prefill_service.generate_batch_zip_to_file, rows, tmp_path, realism_preset,
            )
        else:
            meta = await asyncio.to_thread(
                prefill_service.generate_batch_pdf_to_file, rows, tmp_path, realism_preset,
            )
    except ValueError as exc:
        _cleanup()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - never leak stack traces over HTTP
        _cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Prefill batch generation failed: {type(exc).__name__}: {exc}",
        )

    # If literally every row failed, surface that as a 422 instead of returning
    # an empty PDF/zip the user has to inspect to discover.
    if meta["successes"] == 0:
        _cleanup()
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All rows failed to generate.",
                "errors": meta["errors"],
            },
        )

    # Release the semaphore now — generation is done, file is held for download.
    _release_sem()
    # Register the file for a one-time token-based GET download instead of
    # streaming directly. This allows the frontend to use window.location.href
    # which bypasses browser PDF viewer buffering for large files.
    token = _register_download(tmp_path, media_type, filename)
    return {
        "download_url": f"/api/v1/prefill/batch/download/{token}",
        "filename": filename,
        "count": meta["count"],
        "successes": meta["successes"],
        "errors": meta["errors"],
        "elapsed_s": meta["elapsed_s"],
        "size_bytes": meta["size_bytes"],
    }


@router.get("/prefill/batch/download/{token}")
async def prefill_batch_download(token: str, background_tasks: BackgroundTasks):
    """One-time token download endpoint. Returns the generated file and deletes it."""
    with _DOWNLOAD_STORE_LOCK:
        entry = _DOWNLOAD_STORE.pop(token, None)

    if entry is None:
        raise HTTPException(status_code=404, detail="Download link not found or already used.")

    tmp_path, media_type, filename, expires_at = entry
    if not tmp_path.exists():
        raise HTTPException(status_code=410, detail="File no longer available.")
    if _time_mod.monotonic() > expires_at:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=410, detail="Download link has expired.")

    def _cleanup():
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    background_tasks.add_task(_cleanup)
    return FileResponse(
        path=str(tmp_path),
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        background=background_tasks,
    )
