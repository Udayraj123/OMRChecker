"""JSON API router for the OMRChecker Web UI.

Exposed under ``/api/v1``. All mutations live here so that the HTML UI
and any third-party API consumer go through identical codepaths.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time as _time_mod
from pathlib import Path
from typing import Any

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
from webui.services.batches import BatchNotFound, InvalidBatchRequest
from webui.settings import Settings, get_settings

router = APIRouter(prefix="/api/v1", tags=["omr"])

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

    if pdf_jobs:
        # Schedule PDF rendering as background work. BackgroundTasks runs
        # after the response is sent in production; in TestClient it runs
        # synchronously, which is exactly what the test harness expects.
        for filename, data in pdf_jobs:
            background_tasks.add_task(
                batches_service.save_uploaded_file,
                batch_id,
                filename,
                data,
                settings,
            )
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
        raise HTTPException(
            status_code=400,
            detail="Batch is missing template.json; upload one before processing.",
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

@router.post("/prefill/single")
async def prefill_single(
    student_name: str = Form(...),
    school_name: str = Form(...),
    exam_name: str = Form(...),
    candidate_number: str = Form(...),
    output_format: str = Form("png"),
) -> StreamingResponse:
    """Generate a single pre-filled answer sheet and stream it as a download."""
    output_format = (output_format or "").strip().lower()
    if output_format not in {"png", "pdf"}:
        raise HTTPException(
            status_code=422,
            detail="output_format must be 'png' or 'pdf'.",
        )
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
        if output_format == "pdf":
            data = await asyncio.to_thread(
                prefill_service.generate_single_pdf,
                student_name, school_name, exam_name, candidate_number,
            )
            media_type = "application/pdf"
            filename = "prefilled_sheet.pdf"
        else:
            data = await asyncio.to_thread(
                prefill_service.generate_single_png,
                student_name, school_name, exam_name, candidate_number,
            )
            media_type = "image/png"
            filename = "prefilled_sheet.png"
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
    filename = "prefilled_sheets" + suffix

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
                prefill_service.generate_batch_zip_to_file, rows, tmp_path,
            )
        else:
            meta = await asyncio.to_thread(
                prefill_service.generate_batch_pdf_to_file, rows, tmp_path,
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
