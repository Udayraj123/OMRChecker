"""Filesystem-backed batch management service.

A batch is a self-contained directory that mirrors what the OMRChecker
engine already expects::

    <storage_root>/<batch_id>/
        metadata.json
        inputs/
        outputs/
        template.json      (optional)
        config.json        (optional)
        evaluation.json    (optional)

Using the engine's native layout means we can process a batch via the
existing ``entry_point_for_args`` without modifying any engine code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _atomic_replace(tmp: Path, dest: Path) -> None:
    """Rename *tmp* over *dest* atomically, retrying on Windows PermissionError.

    Windows Defender (and some AV products) briefly locks a newly-written file
    before scanning completes, causing ``os.replace`` to raise
    ``PermissionError [WinError 5]``.  Three quick retries with a short sleep
    cover the vast majority of cases; if all retries fail we fall back to a
    non-atomic copy-then-delete so the write still succeeds.
    """
    for attempt in range(3):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    # Last-resort fallback: not atomic, but avoids a hard crash.
    shutil.copy2(tmp, dest)
    tmp.unlink(missing_ok=True)

from webui.schemas import (
    Batch,
    BatchStatus,
    FileRef,
    SourceMode,
    TemplateAssetRef,
)
from webui.settings import Settings, get_settings

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}
UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS
TEMPLATE_FILENAME = "template.json"
CONFIG_FILENAME = "config.json"
EVALUATION_FILENAME = "evaluation.json"
METADATA_FILENAME = "metadata.json"
ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
RESERVED_BATCH_FILES = {
    TEMPLATE_FILENAME,
    CONFIG_FILENAME,
    EVALUATION_FILENAME,
    METADATA_FILENAME,
}

JSON_DOC_NAMES = {
    "template": TEMPLATE_FILENAME,
    "config": CONFIG_FILENAME,
    "evaluation": EVALUATION_FILENAME,
}

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
VALID_ROTATIONS = {0, 90, 180, 270}


class BatchNotFound(Exception):
    """Raised when a batch id does not map to an existing directory."""


class InvalidBatchRequest(Exception):
    """Raised on bad input (e.g. unsafe filenames, missing directory)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sanitise_filename(name: str) -> str:
    """Return a filesystem-safe version of a user-supplied filename."""
    stem = Path(name).name
    cleaned = _SAFE_FILENAME.sub("_", stem).strip("._")
    if not cleaned:
        raise InvalidBatchRequest(f"Invalid filename: {name!r}")
    return cleaned


def _serialise(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _batch_root(settings: Settings, batch_id: str) -> Path:
    return settings.ensure_storage() / batch_id


def _metadata_path(settings: Settings, batch_id: str) -> Path:
    return _batch_root(settings, batch_id) / METADATA_FILENAME


def _inputs_dir(settings: Settings, batch_id: str) -> Path:
    return _batch_root(settings, batch_id) / "inputs"


def _outputs_dir(settings: Settings, batch_id: str) -> Path:
    return _batch_root(settings, batch_id) / "outputs"


def _load_metadata(settings: Settings, batch_id: str) -> dict[str, Any]:
    meta_path = _metadata_path(settings, batch_id)
    if not meta_path.exists():
        raise BatchNotFound(batch_id)
    with meta_path.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            # Transient: another thread is mid-write. Return an empty dict so
            # callers degrade gracefully; the next read will see the full file.
            return {}


def _save_metadata(settings: Settings, batch_id: str, data: dict[str, Any]) -> None:
    meta_path = _metadata_path(settings, batch_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: _serialise(v) for k, v in data.items()}
    # Write to a sibling temp file then atomically rename so readers never
    # see a truncated or partially-written file (os.replace is atomic on
    # both POSIX and Windows NT).
    tmp = meta_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    _atomic_replace(tmp, meta_path)


def _file_count(batch_dir: Path) -> int:
    inputs = batch_dir / "inputs"
    if not inputs.exists():
        return 0
    return sum(
        1
        for child in inputs.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def _next_available_path(directory: Path, filename: str) -> Path:
    target = directory / filename
    if not target.exists():
        return target
    stem, ext = target.stem, target.suffix
    counter = 1
    while (directory / f"{stem}_{counter}{ext}").exists():
        counter += 1
    return directory / f"{stem}_{counter}{ext}"


def _remove_generated_pdf_pages(inputs: Path, stem: str) -> None:
    """Remove page images previously generated from the same PDF stem.

    Accepts both the current default extension (``.jpg``) and the legacy
    ``.png`` extension so that a re-upload cleans up files left over from
    older versions of the pipeline. Optional ``_<n>`` collision suffix is
    also matched.
    """
    page_pattern = re.compile(
        rf"^{re.escape(stem)}_page_\d{{4}}(?:_\d+)?\.(?:jpg|jpeg|png)$",
        re.IGNORECASE,
    )
    for child in inputs.iterdir():
        if child.is_file() and page_pattern.match(child.name):
            try:
                child.unlink()
            except PermissionError:
                pass


def _to_batch(settings: Settings, batch_id: str, meta: dict[str, Any]) -> Batch:
    batch_dir = _batch_root(settings, batch_id)
    rotation_degrees = int(meta.get("rotation_degrees", 0))
    if rotation_degrees not in VALID_ROTATIONS:
        rotation_degrees = 0
    return Batch(
        id=batch_id,
        name=meta.get("name", batch_id),
        status=BatchStatus(meta.get("status", BatchStatus.created.value)),
        created_at=meta.get("created_at") or _now().isoformat(),
        updated_at=meta.get("updated_at") or meta.get("created_at") or _now().isoformat(),
        source_mode=(
            SourceMode(meta["source_mode"]) if meta.get("source_mode") else None
        ),
        source_dir=meta.get("source_dir"),
        last_error=meta.get("last_error"),
        file_count=_file_count(batch_dir),
        has_template=(batch_dir / TEMPLATE_FILENAME).exists(),
        has_config=(batch_dir / CONFIG_FILENAME).exists(),
        has_evaluation=(batch_dir / EVALUATION_FILENAME).exists(),
        rotation_degrees=rotation_degrees,
    )


def create_batch(name: str, settings: Settings | None = None) -> Batch:
    """Create a new empty batch and return its metadata."""
    settings = settings or get_settings()
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = _batch_root(settings, batch_id)
    (batch_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (batch_dir / "outputs").mkdir(parents=True, exist_ok=True)
    now = _now()
    meta = {
        "id": batch_id,
        "name": name.strip() or batch_id,
        "status": BatchStatus.created.value,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "source_mode": None,
        "source_dir": None,
        "last_error": None,
        "rotation_degrees": 0,
    }
    _save_metadata(settings, batch_id, meta)

    # Auto-apply the default preset (best-effort — batch is valid without it)
    if settings.default_preset:
        try:
            from webui.services.presets import apply_preset_to_batch
            apply_preset_to_batch(batch_dir, settings.default_preset, settings)
        except Exception:
            pass

    return _to_batch(settings, batch_id, meta)


def list_batches(settings: Settings | None = None) -> list[Batch]:
    """Return all known batches, newest first."""
    settings = settings or get_settings()
    root = settings.ensure_storage()
    batches: list[Batch] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / METADATA_FILENAME
        if not meta_path.exists():
            continue
        try:
            meta = _load_metadata(settings, child.name)
        except (OSError, json.JSONDecodeError):
            continue
        batches.append(_to_batch(settings, child.name, meta))
    batches.sort(key=lambda b: b.created_at, reverse=True)
    return batches


def get_batch(batch_id: str, settings: Settings | None = None) -> Batch:
    """Return a single batch or raise ``BatchNotFound``."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    return _to_batch(settings, batch_id, meta)


def delete_batch(batch_id: str, settings: Settings | None = None) -> None:
    """Delete the batch directory from disk."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    shutil.rmtree(batch_dir)


def update_status(
    batch_id: str,
    status: BatchStatus,
    last_error: str | None = None,
    settings: Settings | None = None,
) -> Batch:
    """Persist a status transition for a batch."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    meta["status"] = status.value
    meta["updated_at"] = _now().isoformat()
    if last_error is not None or status == BatchStatus.failed:
        meta["last_error"] = last_error
    elif status in {BatchStatus.queued, BatchStatus.running, BatchStatus.done}:
        meta["last_error"] = None
    _save_metadata(settings, batch_id, meta)
    return _to_batch(settings, batch_id, meta)


def set_rotation(
    batch_id: str,
    degrees: int,
    settings: Settings | None = None,
) -> Batch:
    """Persist a per-batch image rotation used during runtime staging."""
    settings = settings or get_settings()
    if degrees not in VALID_ROTATIONS:
        raise InvalidBatchRequest(
            f"Invalid rotation {degrees!r}; allowed: {sorted(VALID_ROTATIONS)}"
        )
    meta = _load_metadata(settings, batch_id)
    meta["rotation_degrees"] = degrees
    meta["updated_at"] = _now().isoformat()
    _save_metadata(settings, batch_id, meta)
    return _to_batch(settings, batch_id, meta)


def set_source(
    batch_id: str,
    source_mode: SourceMode,
    source_dir: str | None,
    settings: Settings | None = None,
) -> None:
    """Record the last source used to add files to a batch."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    existing = meta.get("source_mode")
    if existing and existing != source_mode.value:
        meta["source_mode"] = SourceMode.mixed.value
    else:
        meta["source_mode"] = source_mode.value
    if source_dir is not None:
        meta["source_dir"] = source_dir
    meta["updated_at"] = _now().isoformat()
    _save_metadata(settings, batch_id, meta)


def list_files(batch_id: str, settings: Settings | None = None) -> list[FileRef]:
    """Return the image files currently attached to a batch."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    # os.scandir() caches DirEntry.stat() from the OS directory listing,
    # avoiding a separate stat() syscall per file — measurably faster on
    # Windows for large directories (hundreds of PDF-split pages).
    refs: list[FileRef] = []
    try:
        with os.scandir(inputs) as it:
            for entry in it:
                if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        refs.append(FileRef(name=entry.name, size_bytes=entry.stat().st_size))
                    except OSError:
                        pass
    except OSError:
        pass
    refs.sort(key=lambda r: r.name)
    return refs


def clear_stale_pdf_split_progress(settings: Settings | None = None) -> None:
    """Reset pdf_split progress on any batch whose split was interrupted.

    Called at server startup so the UI is never stuck showing a stale
    "X/5000 pages" progress bar from a previous run that was killed.
    """
    settings = settings or get_settings()
    storage = settings.storage_root
    if not storage.exists():
        return
    cleared = 0
    for meta_file in storage.glob("*/metadata.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if int(meta.get("pdf_split_total", 0)) > 0:
                meta["pdf_split_pages"] = 0
                meta["pdf_split_total"] = 0
                meta_file.write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                cleared += 1
        except Exception:  # noqa: BLE001
            pass
    if cleared:
        logger.info(
            "Cleared stale pdf_split_progress for %d batch(es) on startup",
            cleared,
        )


def list_input_image_paths(
    batch_id: str, settings: Settings | None = None
) -> list[Path]:
    """Return on-disk paths for input images in a batch."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    return [
        child
        for child in sorted(inputs.iterdir())
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _write_pdf_split_progress(
    batch_id: str | None,
    settings: Settings | None,
    pages_done: int,
    total: int,
) -> None:
    """Best-effort metadata write for live PDF split progress (never raises)."""
    if batch_id is None or settings is None:
        return
    try:
        meta = _load_metadata(settings, batch_id)
        meta["pdf_split_pages"] = pages_done
        meta["pdf_split_total"] = total
        meta["updated_at"] = _now().isoformat()
        _save_metadata(settings, batch_id, meta)
    except Exception:  # noqa: BLE001
        pass  # progress write failure must never abort the split


def save_uploaded_file(
    batch_id: str,
    filename: str,
    data: bytes,
    settings: Settings | None = None,
) -> list[FileRef]:
    """Write an uploaded image or split an uploaded PDF into page images."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    safe = _sanitise_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise InvalidBatchRequest(
            f"Unsupported file type {suffix!r}; allowed: {sorted(UPLOAD_EXTENSIONS)}"
        )
    if suffix in PDF_EXTENSIONS:
        stored = _save_pdf_pages_as_images(
            inputs, safe, data,
            dpi=settings.pdf_render_dpi,
            grayscale=settings.pdf_render_grayscale,
            page_format=settings.pdf_page_format,
            jpeg_quality=settings.pdf_jpeg_quality,
            batch_id=batch_id,
            settings=settings,
        )
        set_source(batch_id, SourceMode.upload, None, settings)
        return stored
    target = _next_available_path(inputs, safe)
    target.write_bytes(data)
    set_source(batch_id, SourceMode.upload, None, settings)
    return [FileRef(name=target.name, size_bytes=target.stat().st_size)]


# Per-worker cache: each ProcessPoolExecutor worker keeps the parsed PDF
# document open across all tasks it processes, paying the parse cost ONCE
# per worker instead of once per page. The cache is keyed by absolute PDF
# path so multiple PDFs in the same worker would coexist (rare in practice).
_PDF_WORKER_CACHE: dict[str, Any] = {}


def _resolve_pdf_format(page_format: str) -> tuple[str, str]:
    """Map a configured page_format to (canonical_name, file_extension)."""
    fmt = (page_format or "jpeg").strip().lower()
    if fmt in {"jpg", "jpeg"}:
        return "jpeg", ".jpg"
    if fmt == "png":
        return "png", ".png"
    raise InvalidBatchRequest(
        f"Unsupported pdf_page_format {page_format!r}; expected 'jpeg' or 'png'."
    )


def _render_pdf_page_to_disk(
    pdf_path: str,
    page_index: int,
    output_dir: str,
    stem: str,
    *,
    dpi: int,
    grayscale: bool,
    ext: str,
    jpeg_quality: int,
) -> tuple[int, str | None, int, str | None]:
    """Worker entry point: render ONE page of ``pdf_path`` and save to disk.

    Returns ``(page_index, filename_or_None, size_bytes, error_or_None)``.

    Defined at module scope so ``ProcessPoolExecutor`` can pickle it. The
    worker process caches the opened ``fitz.Document`` between tasks via
    :data:`_PDF_WORKER_CACHE`, so the PDF is parsed only once per worker.
    """
    import fitz

    doc = _PDF_WORKER_CACHE.get(pdf_path)
    if doc is None:
        doc = fitz.open(pdf_path)
        _PDF_WORKER_CACHE[pdf_path] = doc

    try:
        page = doc.load_page(page_index - 1)
        colorspace = fitz.csGRAY if grayscale else fitz.csRGB
        pixmap = page.get_pixmap(dpi=dpi, alpha=False, colorspace=colorspace)
        target = Path(output_dir) / f"{stem}_page_{page_index:04d}{ext}"
        if ext == ".jpg":
            try:
                pixmap.save(str(target), jpg_quality=jpeg_quality)
            except TypeError:
                pixmap.save(str(target))
        else:
            try:
                pixmap.save(str(target), compress_level=1)
            except TypeError:
                pixmap.save(str(target))
        size = target.stat().st_size
        del pixmap
        return (page_index, target.name, size, None)
    except Exception as exc:  # noqa: BLE001 — worker boundary
        return (page_index, None, 0, f"{type(exc).__name__}: {exc}")


def _default_pdf_split_workers(settings: Settings | None) -> int:
    """Auto-detect a safe default worker count for PDF rendering.

    PDF render + JPEG encode is CPU-bound, so we aim for half the cores
    (leaving the rest for the OS, the FastAPI event loop, and any
    concurrent OMR processing) capped at 8 to keep per-worker spawn
    overhead worth it for batches of any size.
    """
    if settings is not None and getattr(settings, "pdf_split_workers", 0):
        return int(settings.pdf_split_workers)
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu // 2))


def _save_pdf_pages_as_images(
    inputs: Path,
    safe_filename: str,
    data: bytes,
    *,
    dpi: int = 150,
    grayscale: bool = True,
    page_format: str = "jpeg",
    jpeg_quality: int = 92,
    batch_id: str | None = None,
    settings: Settings | None = None,
) -> list[FileRef]:
    """Render every PDF page into an image in ``inputs``.

    Default output is **grayscale JPEG at quality 92**, which is ~5x smaller
    on disk than the equivalent PNG yet visually lossless for OMR bubble
    detection.

    The render loop runs in a :class:`ProcessPoolExecutor` for PDFs with
    more than ``settings.pdf_split_min_pages_for_parallel`` pages — render
    + JPEG encode is CPU-bound and embarrassingly parallel, so this gives
    a near-linear speedup up to ``pdf_split_workers`` workers. Smaller
    PDFs use the legacy single-threaded loop where process-pool spawn
    overhead would dominate.
    """
    try:
        import fitz
    except ImportError as exc:
        raise InvalidBatchRequest(
            "PDF uploads require PyMuPDF. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    _fmt_name, ext = _resolve_pdf_format(page_format)
    quality = max(60, min(100, int(jpeg_quality)))

    # Determine page count cheaply (one parse) before deciding strategy.
    try:
        with fitz.open(stream=data, filetype="pdf") as probe:
            page_count = probe.page_count
    except InvalidBatchRequest:
        raise
    except Exception as exc:
        raise InvalidBatchRequest(
            f"Could not convert PDF {safe_filename!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if page_count == 0:
        raise InvalidBatchRequest(f"PDF has no pages: {safe_filename}")

    stem = Path(safe_filename).stem
    _remove_generated_pdf_pages(inputs, stem)
    _write_pdf_split_progress(batch_id, settings, 0, page_count)

    min_parallel = (
        getattr(settings, "pdf_split_min_pages_for_parallel", 16)
        if settings is not None else 16
    )
    requested_workers = _default_pdf_split_workers(settings)
    use_parallel = (
        requested_workers > 1
        and page_count >= max(1, int(min_parallel))
    )

    logger.info(
        "PDF split started | file=%s | pages=%d | dpi=%d | grayscale=%s | "
        "fmt=%s | mode=%s | workers=%d",
        safe_filename, page_count, dpi, grayscale, ext.lstrip("."),
        "parallel" if use_parallel else "serial",
        requested_workers if use_parallel else 1,
    )

    try:
        if use_parallel:
            stored, failed_pages = _save_pdf_pages_parallel(
                inputs=inputs,
                safe_filename=safe_filename,
                data=data,
                stem=stem,
                page_count=page_count,
                dpi=dpi,
                grayscale=grayscale,
                ext=ext,
                jpeg_quality=quality,
                workers=requested_workers,
                batch_id=batch_id,
                settings=settings,
            )
        else:
            stored, failed_pages = _save_pdf_pages_serial(
                inputs=inputs,
                safe_filename=safe_filename,
                data=data,
                stem=stem,
                page_count=page_count,
                dpi=dpi,
                grayscale=grayscale,
                ext=ext,
                jpeg_quality=quality,
                batch_id=batch_id,
                settings=settings,
            )
    except InvalidBatchRequest:
        raise
    except Exception as exc:
        raise InvalidBatchRequest(
            f"Could not convert PDF {safe_filename!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not stored:
        _write_pdf_split_progress(batch_id, settings, 0, 0)  # clear progress
        raise InvalidBatchRequest(
            f"PDF {safe_filename!r}: all {len(failed_pages)} page(s) failed to render."
        )
    # Clear progress fields so the UI does not show stale split data
    _write_pdf_split_progress(batch_id, settings, 0, 0)
    if failed_pages:
        logger.warning(
            "PDF split finished with errors | file=%s | ok=%d | failed=%d | "
            "first_failed_pages=%s",
            safe_filename, len(stored), len(failed_pages), failed_pages[:20],
        )
    else:
        logger.info(
            "PDF split complete | file=%s | pages=%d | dpi=%d | grayscale=%s",
            safe_filename, len(stored), dpi, grayscale,
        )
    return stored


def _save_pdf_pages_serial(
    *,
    inputs: Path,
    safe_filename: str,
    data: bytes,
    stem: str,
    page_count: int,
    dpi: int,
    grayscale: bool,
    ext: str,
    jpeg_quality: int,
    batch_id: str | None,
    settings: Settings | None,
) -> tuple[list[FileRef], list[int]]:
    """Single-threaded PDF -> image render. Used for small PDFs."""
    import fitz

    stored: list[FileRef] = []
    failed_pages: list[int] = []
    colorspace = fitz.csGRAY if grayscale else fitz.csRGB

    with fitz.open(stream=data, filetype="pdf") as pdf:
        for page_index, page in enumerate(pdf, start=1):
            try:
                pixmap = page.get_pixmap(
                    dpi=dpi, alpha=False, colorspace=colorspace
                )
                target = inputs / f"{stem}_page_{page_index:04d}{ext}"
                if ext == ".jpg":
                    try:
                        pixmap.save(str(target), jpg_quality=jpeg_quality)
                    except TypeError:
                        pixmap.save(str(target))
                else:
                    try:
                        pixmap.save(str(target), compress_level=1)
                    except TypeError:
                        pixmap.save(str(target))
                del pixmap
                stored.append(
                    FileRef(name=target.name, size_bytes=target.stat().st_size)
                )
            except Exception as page_exc:  # noqa: BLE001
                failed_pages.append(page_index)
                logger.warning(
                    "PDF page render failed | file=%s | page=%d/%d | %s: %s",
                    safe_filename, page_index, page_count,
                    type(page_exc).__name__, page_exc,
                )
            if page_index % 10 == 0 or page_index == page_count:
                _write_pdf_split_progress(
                    batch_id, settings, len(stored), page_count
                )
            if page_index % 50 == 0 or page_index == page_count:
                logger.info(
                    "PDF split progress | file=%s | %d/%d pages saved | failed=%d",
                    safe_filename, len(stored), page_count, len(failed_pages),
                )
    return stored, failed_pages


def _save_pdf_pages_parallel(
    *,
    inputs: Path,
    safe_filename: str,
    data: bytes,
    stem: str,
    page_count: int,
    dpi: int,
    grayscale: bool,
    ext: str,
    jpeg_quality: int,
    workers: int,
    batch_id: str | None,
    settings: Settings | None,
) -> tuple[list[FileRef], list[int]]:
    """Parallel PDF -> image render via :class:`ProcessPoolExecutor`.

    Each worker process opens the PDF once (cached in
    :data:`_PDF_WORKER_CACHE`) and renders the pages routed to it. The PDF
    bytes are written to a single temporary file under the scratch cache
    root and each worker opens it by path — that avoids pickling the full
    PDF bytes (potentially 100+ MB) once per worker process, and keeps the
    temp file inside the SEP-excluded directory.
    """
    # Stage the PDF as a single file under the scratch cache. Using the
    # cache root means the temp file lands inside the AV exclusion (one
    # place, not %TEMP%).
    if settings is not None:
        try:
            scratch_root = settings.ensure_cache_root() / "pdf_split_inputs"
        except Exception:  # noqa: BLE001
            scratch_root = inputs
    else:
        scratch_root = inputs
    scratch_root.mkdir(parents=True, exist_ok=True)
    pdf_tmp_name = f"{uuid.uuid4().hex}_{Path(safe_filename).name}"
    pdf_tmp_path = scratch_root / pdf_tmp_name
    pdf_tmp_path.write_bytes(data)
    pdf_path_str = str(pdf_tmp_path)

    stored_by_index: dict[int, FileRef] = {}
    failed_pages: list[int] = []

    progress_step = max(1, page_count // 50)

    try:
        # Submit one task per page. Each task is tiny to pickle (a path
        # plus a few ints) and the worker reuses the cached fitz.Document
        # for every page it handles.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _render_pdf_page_to_disk,
                    pdf_path_str,
                    page_index,
                    str(inputs),
                    stem,
                    dpi=dpi,
                    grayscale=grayscale,
                    ext=ext,
                    jpeg_quality=jpeg_quality,
                ): page_index
                for page_index in range(1, page_count + 1)
            }
            completed = 0
            for fut in as_completed(futures):
                page_index = futures[fut]
                try:
                    res_page, res_name, res_size, res_err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res_page = page_index
                    res_name = None
                    res_size = 0
                    res_err = f"{type(exc).__name__}: {exc}"

                if res_err is not None or res_name is None:
                    failed_pages.append(res_page)
                    logger.warning(
                        "PDF page render failed | file=%s | page=%d/%d | %s",
                        safe_filename, res_page, page_count, res_err,
                    )
                else:
                    stored_by_index[res_page] = FileRef(
                        name=res_name, size_bytes=res_size
                    )

                completed += 1
                if (
                    completed % progress_step == 0
                    or completed == page_count
                ):
                    _write_pdf_split_progress(
                        batch_id, settings,
                        len(stored_by_index), page_count,
                    )
                if completed % 50 == 0 or completed == page_count:
                    logger.info(
                        "PDF split progress | file=%s | %d/%d pages saved | failed=%d",
                        safe_filename, len(stored_by_index),
                        page_count, len(failed_pages),
                    )
    finally:
        # Always clean up the staged PDF — workers have already closed
        # their handles when the executor shut down.
        try:
            pdf_tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    failed_pages.sort()
    # Return refs in page order so the UI list is naturally sorted by page.
    stored = [stored_by_index[i] for i in sorted(stored_by_index)]
    return stored, failed_pages


def delete_file(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> None:
    """Remove a single file from a batch's inputs directory."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    safe = _sanitise_filename(filename)
    target = inputs / safe
    if not target.exists() or not target.is_file():
        raise InvalidBatchRequest(f"File not found: {safe}")
    target.unlink()


def resolve_input_file(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> Path:
    """Return a safely resolved input image path for preview/download."""
    settings = settings or get_settings()
    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)
    safe = _sanitise_filename(filename)
    target = (inputs / safe).resolve()
    try:
        target.relative_to(inputs.resolve())
    except ValueError as exc:
        raise InvalidBatchRequest("Path traversal not allowed") from exc
    if (
        not target.exists()
        or not target.is_file()
        or target.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        raise InvalidBatchRequest(f"Input image not found: {safe}")
    return target


def import_directory(
    batch_id: str,
    source_dir: str,
    copy: bool = True,
    settings: Settings | None = None,
) -> tuple[list[FileRef], list[str]]:
    """Copy (or link) all supported images from ``source_dir`` into the batch."""
    settings = settings or get_settings()
    if not settings.allow_directory_import:
        raise InvalidBatchRequest(
            "Directory import is disabled on this server (ALLOW_DIRECTORY_IMPORT=false)."
        )
    src = Path(source_dir).expanduser()
    if not src.exists() or not src.is_dir():
        raise InvalidBatchRequest(f"Source directory does not exist: {source_dir}")

    inputs = _inputs_dir(settings, batch_id)
    if not inputs.exists():
        raise BatchNotFound(batch_id)

    imported: list[FileRef] = []
    skipped: list[str] = []
    for child in sorted(src.iterdir()):
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix not in UPLOAD_EXTENSIONS:
            skipped.append(child.name)
            continue
        safe = _sanitise_filename(child.name)
        if suffix in PDF_EXTENSIONS:
            imported.extend(
                _save_pdf_pages_as_images(
                    inputs, safe, child.read_bytes(),
                    dpi=settings.pdf_render_dpi,
                    grayscale=settings.pdf_render_grayscale,
                    batch_id=batch_id,
                    settings=settings,
                )
            )
            continue
        target = _next_available_path(inputs, safe)
        if not copy:
            try:
                target.symlink_to(child.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(child, target)
        else:
            shutil.copy2(child, target)
        imported.append(FileRef(name=target.name, size_bytes=target.stat().st_size))

    if imported:
        set_source(batch_id, SourceMode.directory, str(src), settings)
    return imported, skipped


def get_json_document(
    batch_id: str,
    doc_name: str,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    """Return a parsed template/config/evaluation JSON or ``None`` if absent."""
    settings = settings or get_settings()
    if doc_name not in JSON_DOC_NAMES:
        raise InvalidBatchRequest(f"Unknown document {doc_name!r}")
    path = _batch_root(settings, batch_id) / JSON_DOC_NAMES[doc_name]
    if not _batch_root(settings, batch_id).exists():
        raise BatchNotFound(batch_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json_document(
    batch_id: str,
    doc_name: str,
    content: dict[str, Any] | None,
    settings: Settings | None = None,
) -> None:
    """Write or delete one of the optional template/config/evaluation files."""
    settings = settings or get_settings()
    if doc_name not in JSON_DOC_NAMES:
        raise InvalidBatchRequest(f"Unknown document {doc_name!r}")
    root = _batch_root(settings, batch_id)
    if not root.exists():
        raise BatchNotFound(batch_id)
    path = root / JSON_DOC_NAMES[doc_name]
    if content is None:
        if path.exists():
            path.unlink()
        return
    if not isinstance(content, dict):
        raise InvalidBatchRequest(f"{doc_name}.json must be a JSON object")
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, sort_keys=True)
    _atomic_replace(tmp, path)
    meta = _load_metadata(settings, batch_id)
    meta["updated_at"] = _now().isoformat()
    _save_metadata(settings, batch_id, meta)


def get_batch_metadata(
    batch_id: str, settings: Settings | None = None
) -> dict[str, Any]:
    """Return raw metadata.json content for a batch."""
    settings = settings or get_settings()
    return _load_metadata(settings, batch_id)


def update_batch_metadata(
    batch_id: str,
    updates: dict[str, Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Merge arbitrary keys into metadata.json and return the new payload."""
    settings = settings or get_settings()
    meta = _load_metadata(settings, batch_id)
    meta.update(updates)
    meta["updated_at"] = _now().isoformat()
    _save_metadata(settings, batch_id, meta)
    return meta


def find_results_csv(batch_id: str, settings: Settings | None = None) -> Path | None:
    """Locate the most recent ``Results_*.csv`` produced by the engine."""
    settings = settings or get_settings()
    outputs = _outputs_dir(settings, batch_id)
    if not outputs.exists():
        return None
    candidates: list[Path] = []
    for results_dir in outputs.rglob("Results"):
        if not results_dir.is_dir():
            continue
        candidates.extend(results_dir.glob("Results_*.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return candidates[0]


def iter_output_files(batch_id: str, settings: Settings | None = None) -> Iterable[Path]:
    """Yield every file produced in the batch's ``outputs/`` tree."""
    settings = settings or get_settings()
    outputs = _outputs_dir(settings, batch_id)
    if not outputs.exists():
        return []
    return (p for p in outputs.rglob("*") if p.is_file())


def resolve_output_file(
    batch_id: str,
    relative: str,
    settings: Settings | None = None,
) -> Path:
    """Resolve ``relative`` safely against the batch's ``outputs/`` folder."""
    settings = settings or get_settings()
    outputs = _outputs_dir(settings, batch_id).resolve()
    if not outputs.exists():
        raise BatchNotFound(batch_id)
    candidate = (outputs / relative).resolve()
    try:
        candidate.relative_to(outputs)
    except ValueError as exc:
        raise InvalidBatchRequest("Path traversal not allowed") from exc
    if not candidate.exists() or not candidate.is_file():
        raise InvalidBatchRequest(f"Output file not found: {relative}")
    return candidate


def get_batch_root(batch_id: str, settings: Settings | None = None) -> Path:
    """Return the on-disk directory for a batch (for engine invocation)."""
    settings = settings or get_settings()
    root = _batch_root(settings, batch_id)
    if not root.exists():
        raise BatchNotFound(batch_id)
    return root


def _collect_template_relative_paths(value: Any) -> list[str]:
    """Walk a parsed template.json payload and collect ``relativePath`` values."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "relativePath" and isinstance(item, str):
                found.append(item)
            else:
                found.extend(_collect_template_relative_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_template_relative_paths(item))
    return found


def _template_required_asset_names(
    batch_id: str, settings: Settings
) -> list[str]:
    """Return deduplicated asset filenames referenced by template.json."""
    template_doc = get_json_document(batch_id, "template", settings)
    if not template_doc:
        return []
    raw_paths = _collect_template_relative_paths(
        template_doc.get("preProcessors", [])
    )
    seen: list[str] = []
    for rel in raw_paths:
        name = Path(rel).name
        if name and name not in seen:
            seen.append(name)
    return seen


def _asset_is_present(batch_dir: Path, name: str) -> Path | None:
    """Return the path to an asset if it exists in root or inputs/, else None."""
    candidates = (batch_dir / name, batch_dir / "inputs" / name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def list_template_assets(
    batch_id: str, settings: Settings | None = None
) -> list[TemplateAssetRef]:
    """Return template-referenced assets and whether each is present on disk."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)

    required = _template_required_asset_names(batch_id, settings)
    refs: list[TemplateAssetRef] = []
    for name in required:
        resolved = _asset_is_present(batch_dir, name)
        refs.append(
            TemplateAssetRef(
                name=name,
                required=True,
                present=resolved is not None,
                size_bytes=resolved.stat().st_size if resolved else None,
            )
        )
    return refs


def missing_template_assets(
    batch_id: str, settings: Settings | None = None
) -> list[str]:
    """Return names of template-referenced assets that are not yet on disk."""
    return [asset.name for asset in list_template_assets(batch_id, settings) if not asset.present]


def _validate_asset_filename(name: str) -> str:
    """Sanitize and validate a user-supplied asset filename."""
    safe = _sanitise_filename(name)
    if safe in RESERVED_BATCH_FILES:
        raise InvalidBatchRequest(
            f"{safe!r} is reserved for the batch itself; use a different filename."
        )
    suffix = Path(safe).suffix.lower()
    if suffix not in ASSET_EXTENSIONS:
        raise InvalidBatchRequest(
            f"Unsupported asset type {suffix!r}; allowed: {sorted(ASSET_EXTENSIONS)}"
        )
    return safe


def save_template_asset(
    batch_id: str,
    filename: str,
    data: bytes,
    settings: Settings | None = None,
) -> TemplateAssetRef:
    """Write a template asset (e.g. ``omr_marker.jpg``) into the batch root."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    safe = _validate_asset_filename(filename)
    target = batch_dir / safe
    target.write_bytes(data)

    required = _template_required_asset_names(batch_id, settings)
    return TemplateAssetRef(
        name=safe,
        required=safe in required,
        present=True,
        size_bytes=target.stat().st_size,
    )


def delete_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> None:
    """Remove a previously uploaded template asset from the batch root."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    safe = _validate_asset_filename(filename)
    target = batch_dir / safe
    if not target.exists() or not target.is_file():
        raise InvalidBatchRequest(f"Asset not found: {safe}")
    target.unlink()


def resolve_template_asset(
    batch_id: str,
    filename: str,
    settings: Settings | None = None,
) -> Path:
    """Return a safely resolved template asset image path for preview/download."""
    settings = settings or get_settings()
    batch_dir = _batch_root(settings, batch_id)
    if not batch_dir.exists():
        raise BatchNotFound(batch_id)
    safe = _validate_asset_filename(filename)
    resolved = _asset_is_present(batch_dir, safe)
    if resolved is None:
        raise InvalidBatchRequest(f"Asset not found: {safe}")
    return resolved.resolve()


def reset_batch_runtime_state(
    batch_id: str, settings: Settings | None = None
) -> None:
    """Remove generated runtime/output artifacts and reset run metadata."""
    settings = settings or get_settings()
    root = get_batch_root(batch_id, settings)

    for name in ("outputs", "_runtime"):
        target = root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    (root / "outputs").mkdir(parents=True, exist_ok=True)

    meta = _load_metadata(settings, batch_id)
    meta["status"] = BatchStatus.created.value
    meta["last_error"] = None
    meta["updated_at"] = _now().isoformat()
    for key in (
        "processed_files",
        "total_files",
        "latest_processed_file",
        "latest_dynamic_dimensions",
        "dynamic_dimensions_by_file",
        "cancel_requested",
    ):
        meta.pop(key, None)
    _save_metadata(settings, batch_id, meta)
