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
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, cast

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


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return _now()


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
        except json.JSONDecodeError as exc:
            # Transient: another thread is mid-write. Return an empty dict so
            # callers degrade gracefully; the next read will see the full file.
            logger.warning(
                "metadata.json for batch %r is corrupt or mid-write: %s", batch_id, exc
            )
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
    """Remove page images previously generated from the same PDF stem."""
    page_pattern = re.compile(rf"^{re.escape(stem)}_page_\d{{4}}(?:_\d+)?\.(?:png|jpg|jpeg)$")
    for child in inputs.iterdir():
        if child.is_file() and page_pattern.match(child.name):
            child.unlink()


def _to_batch(settings: Settings, batch_id: str, meta: dict[str, Any]) -> Batch:
    batch_dir = _batch_root(settings, batch_id)
    rotation_degrees = int(meta.get("rotation_degrees", 0))
    if rotation_degrees not in VALID_ROTATIONS:
        rotation_degrees = 0
    created_at = _coerce_datetime(meta.get("created_at"))
    updated_at = _coerce_datetime(meta.get("updated_at") or meta.get("created_at"))
    return Batch(
        id=batch_id,
        name=meta.get("name", batch_id),
        status=BatchStatus(meta.get("status", BatchStatus.created.value)),
        created_at=created_at,
        updated_at=updated_at,
        source_mode=(
            SourceMode(meta["source_mode"]) if meta.get("source_mode") else None
        ),
        source_dir=meta.get("source_dir"),
        last_error=meta.get("last_error"),
        file_count=_file_count(batch_dir),
        has_template=(batch_dir / TEMPLATE_FILENAME).exists(),
        has_config=(batch_dir / CONFIG_FILENAME).exists(),
        has_evaluation=(batch_dir / EVALUATION_FILENAME).exists(),
        rotation_degrees=cast(Literal[0, 90, 180, 270], rotation_degrees),
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
    logger.info("Batch created | name=%r | id=%s", meta["name"], batch_id)

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
    try:
        meta = _load_metadata(settings, batch_id)
        batch_name = meta.get("name", batch_id)
    except Exception:  # noqa: BLE001
        batch_name = batch_id
    shutil.rmtree(batch_dir)
    logger.info("Batch deleted | name=%r | id=%s", batch_name, batch_id)


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
    _name = meta.get("name", batch_id)
    if last_error:
        logger.warning("Batch %r -> %s | error: %s | id=%s", _name, status.value, last_error, batch_id)
    else:
        logger.info("Batch %r -> %s | id=%s", _name, status.value, batch_id)
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
    return [
        FileRef(name=child.name, size_bytes=child.stat().st_size)
        for child in sorted(inputs.iterdir())
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    ]


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
            batch_id=batch_id,
            settings=settings,
        )
        set_source(batch_id, SourceMode.upload, None, settings)
        return stored
    target = _next_available_path(inputs, safe)
    target.write_bytes(data)
    size = target.stat().st_size
    logger.info("File uploaded | batch=%s | file=%s | size=%d bytes", batch_id, target.name, size)
    set_source(batch_id, SourceMode.upload, None, settings)
    return [FileRef(name=target.name, size_bytes=size)]


def _render_pdf_chunk(
    tmp_path: str,
    inputs_str: str,
    stem: str,
    dpi: int,
    use_gray: bool,
    chunk_start: int,
    chunk_end: int,
    safe_filename: str,
    page_count: int,
    stop_event: threading.Event | None = None,
    chunk_idx: int = 0,
    total_chunks: int = 1,
    on_page_done: Callable[[], None] | None = None,
) -> tuple[list[tuple[int, str, int]], list[int], float, float, float]:
    """Render PDF pages [chunk_start, chunk_end) into JPEG files.

    Uses a two-stage pipeline per chunk:

    * **Stage 1 (this thread)**: ``get_pixmap()`` + ``tobytes()`` → enqueue
      encoded bytes into a ``SimpleQueue``.
    * **Stage 2 (1 write thread)**: drain the queue, call ``write_bytes()``.

    The single write thread is intentional.  Testing showed N=3 concurrent
    write workers *regressed* throughput (9.5 pg/s vs 10.4 pg/s) because
    Windows Defender serialises its scan queue — more concurrent writes per
    chunk overloaded Defender's scan threads and slowed each write down.

    Returns
    -------
    page_results : list of (page_index, filename, size_bytes)
    failed_pages : 1-based page numbers that raised an exception
    total_render_s : cumulative fitz render time across all pages in chunk
    total_encode_s : cumulative JPEG tobytes() encode time across all pages
    total_write_s  : cumulative write_bytes() disk-write time across all pages
    """
    import time as _t
    import fitz as _fitz
    import queue as _queue

    _fitz.TOOLS.set_aa_level(0)  # no AA: ~20% faster rasterisation, fine for OMR
    colorspace = _fitz.csGRAY if use_gray else _fitz.csRGB
    inputs = Path(inputs_str)
    failed: list[int] = []
    t_render = 0.0
    t_encode = 0.0

    # --- write worker -------------------------------------------------------
    # Runs in a dedicated thread so disk I/O overlaps with render+encode on
    # the main worker thread.  SimpleQueue is unbounded and thread-safe.
    #
    # N=1 is intentional: with Windows Defender real-time scanning each new
    # file, having more than 1 write thread per chunk overloads Defender's
    # scan queue and makes each write *slower* (measured: N=3 gave 9.5 pg/s
    # vs N=1 at 10.4 pg/s — 9 % regression due to Defender contention).
    _results: list[tuple[int, str, int]] = []
    _write_failed: list[int] = []
    _t_write = [0.0]
    _write_q: _queue.SimpleQueue = _queue.SimpleQueue()

    def _write_worker() -> None:
        while True:
            item = _write_q.get()
            if item is None:  # sentinel — no more pages
                break
            page_index, dest, jpeg_bytes, page_number = item
            tw = _t.perf_counter()
            try:
                dest.write_bytes(jpeg_bytes)
                _results.append((page_index, dest.name, len(jpeg_bytes)))
            except Exception as exc:  # noqa: BLE001
                _write_failed.append(page_number)
                logger.warning(
                    "PDF page write failed | file=%s | page=%d/%d | %s: %s",
                    safe_filename, page_number, page_count, type(exc).__name__, exc,
                )
            _t_write[0] += _t.perf_counter() - tw

    write_thread = threading.Thread(target=_write_worker, daemon=True)
    write_thread.start()
    # -------------------------------------------------------------------------

    logger.info(
        "PDF chunk start | file=%s | chunk=%d/%d | pages=%d-%d",
        safe_filename, chunk_idx + 1, total_chunks, chunk_start + 1, chunk_end,
    )

    with _fitz.open(tmp_path) as doc:
        for page_index in range(chunk_start, chunk_end):
            if stop_event is not None and stop_event.is_set():
                break  # server shutting down — exit cleanly without blocking
            page_number = page_index + 1
            try:
                t0 = _t.perf_counter()
                pixmap = doc.load_page(page_index).get_pixmap(
                    dpi=dpi, alpha=False, colorspace=colorspace
                )
                t1 = _t.perf_counter()
                page_name = f"{stem}_page_{page_number:04d}.jpg"
                jpeg_bytes = pixmap.tobytes(output="jpeg", jpg_quality=75)
                del pixmap  # free 2-4 MB before enqueueing
                t2 = _t.perf_counter()
                _write_q.put((page_index, inputs / page_name, jpeg_bytes, page_number))
                t_render += t1 - t0
                t_encode += t2 - t1
                if on_page_done is not None:
                    on_page_done()
            except Exception as exc:  # noqa: BLE001
                failed.append(page_number)
                logger.warning(
                    "PDF page render failed | file=%s | page=%d/%d | %s: %s",
                    safe_filename, page_number, page_count,
                    type(exc).__name__, exc,
                )

    _write_q.put(None)  # signal write thread to exit after draining
    write_thread.join()  # wait for all pending writes to complete

    failed.extend(_write_failed)
    return _results, failed, t_render, t_encode, _t_write[0]


def _save_pdf_pages_as_images(
    inputs: Path,
    safe_filename: str,
    data: bytes,
    *,
    dpi: int = 150,
    grayscale: bool = True,
    batch_id: str | None = None,
    settings: Settings | None = None,
) -> list[FileRef]:
    """Render every PDF page into a JPEG image in ``inputs``.

    Performance design
    ------------------
    * **All available CPU cores** (``os.cpu_count()``): no artificial cap.
      PyMuPDF 1.24+ (Cython build) releases the GIL during rendering so
      threads provide true parallelism on multi-core machines.
    * **Chunk-based work distribution**: pages are split into
      ``n_workers`` equal chunks; each worker renders its chunk sequentially
      inside a single open ``fitz.Document``.  This eliminates 5 000
      individual ``Future`` objects and keeps MuPDF's page-cache warm
      (sequential access is faster than random access for large PDFs).
    * **Temp-file for the PDF**: raw bytes written once to disk; threads
      open by path so MuPDF does not duplicate the data in its C-heap.
    * **JPEG q=75**: 5-10× faster to encode than PNG; imperceptible quality
      difference for ArUco marker and bubble-fill detection.
    * **Write pipelining**: each chunk worker runs a dedicated write thread so
      ``tobytes()`` (encode) overlaps with ``write_bytes()`` (disk I/O).
      Wall-clock per chunk ≈ ``write × n`` instead of
      ``(render+encode+write) × n`` — ~31 % faster end-to-end.
    * **Timing instrumentation**: each chunk logs render vs. save seconds so
      the dominant cost is visible immediately in the server log.
    """
    try:
        import fitz
    except ImportError as exc:
        raise InvalidBatchRequest(
            "PDF uploads require PyMuPDF. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    stored: dict[int, FileRef] = {}
    failed_pages: list[int] = []
    tmp_path: str | None = None

    try:
        with fitz.open(stream=data, filetype="pdf") as pdf_probe:
            page_count = pdf_probe.page_count
        if page_count == 0:
            raise InvalidBatchRequest(f"PDF has no pages: {safe_filename}")

        stem = Path(safe_filename).stem
        logger.info(
            "PDF split started | file=%s | pages=%d | dpi=%d | grayscale=%s",
            safe_filename, page_count, dpi, grayscale,
        )
        _remove_generated_pdf_pages(inputs, stem)
        _write_pdf_split_progress(batch_id, settings, 0, page_count)

        # Write PDF to disk once; each worker opens by path (no heap duplication)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(data)
            tmp_path = tf.name

        # Cap workers at the actual page count (no idle threads for tiny PDFs)
        # and at 32 (no benefit beyond physical cores; avoids thrash on
        # hyperthreaded or containerised machines with large logical counts).
        n_workers = min(os.cpu_count() or 4, page_count, 32)
        # Aim for ~4 rounds of work per worker: gives several bursts of progress
        # logs while rendering (instead of one silent run), and improves
        # load-balancing when individual page render times vary.
        # Floor at 25 pages so the Future count stays well below 1 000.
        chunk_size = max(25, math.ceil(page_count / (n_workers * 4)))
        chunks = [
            (i, min(i + chunk_size, page_count))
            for i in range(0, page_count, chunk_size)
        ]
        logger.info(
            "PDF split | workers=%d | chunks=%d | chunk_size=%d | output=jpg | quality=75",
            n_workers, len(chunks), chunk_size,
        )

        total_render_s = 0.0
        total_encode_s = 0.0
        total_write_s = 0.0

        # Shared counter for intra-chunk UI/terminal progress updates.
        # Protected by _write_lock so _write_pdf_split_progress is never
        # called concurrently from multiple worker threads.
        _write_lock = threading.Lock()
        _pages_done = [0]

        def _on_page_done() -> None:
            with _write_lock:
                _pages_done[0] += 1
                current = _pages_done[0]
            if current % 100 == 0 or current >= page_count:
                _write_pdf_split_progress(batch_id, settings, current, page_count)

        # stop_event lets worker threads exit early between pages so that
        # executor.shutdown(wait=False) during a server reload/SIGTERM returns
        # in at most one page-render time (~750 ms) instead of one full chunk
        # (~157 s at 209 pages × 750 ms).
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=n_workers)
        t_wall_start = time.perf_counter()
        try:
            futures = [
                executor.submit(
                    _render_pdf_chunk,
                    tmp_path, str(inputs), stem, dpi, grayscale,
                    chunk_start, chunk_end, safe_filename, page_count,
                    stop_event, chunk_idx, len(chunks), _on_page_done,
                )
                for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks)
            ]
            for future in as_completed(futures):
                chunk_results, chunk_failed, t_render, t_encode, t_write = future.result()
                for page_index, page_name, size_bytes in chunk_results:
                    stored[page_index] = FileRef(name=page_name, size_bytes=size_bytes)
                failed_pages.extend(chunk_failed)
                total_render_s += t_render
                total_encode_s += t_encode
                total_write_s += t_write
                done = len(stored) + len(failed_pages)
                logger.info(
                    "PDF chunk done | file=%s | total=%d/%d | "
                    "chunk_render=%.1fs | chunk_encode=%.1fs | chunk_write=%.1fs",
                    safe_filename, done, page_count, t_render, t_encode, t_write,
                )
        except BaseException:
            # Signal all running workers to stop at their next page boundary,
            # then release threads without waiting for full chunks to drain.
            stop_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=False)

    except InvalidBatchRequest:
        raise
    except Exception as exc:
        raise InvalidBatchRequest(
            f"Could not convert PDF {safe_filename!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    stored_list = [stored[i] for i in sorted(stored.keys())]
    if not stored_list:
        _write_pdf_split_progress(batch_id, settings, 0, 0)
        raise InvalidBatchRequest(
            f"PDF {safe_filename!r}: all {len(failed_pages)} page(s) failed to render."
        )

    _write_pdf_split_progress(batch_id, settings, 0, 0)  # clear stale progress
    wall_s = time.perf_counter() - t_wall_start  # true elapsed (accounts for pipelining)
    tot = total_render_s + total_encode_s + total_write_s
    # Warn when write is unexpectedly slow (likely Windows Defender real-time scanning).
    # Adding the storage folder to Defender exclusions typically gives 100× write speedup.
    if total_write_s > 0 and len(stored_list) > 0:
        write_s_per_page = total_write_s / len(stored_list)
        if write_s_per_page > 0.5:
            logger.warning(
                "PDF split: write is very slow (%.2fs/page). "
                "Windows Defender real-time scanning is likely the cause. "
                "Add '%s' to Defender exclusions for a large speed boost.",
                write_s_per_page, inputs,
            )
    if failed_pages:
        logger.warning(
            "PDF split finished with errors | file=%s | ok=%d | failed=%d | "
            "first_failed=%s | render=%.1fs | encode=%.1fs | write=%.1fs | "
            "render_pct=%.0f%%",
            safe_filename, len(stored_list), len(failed_pages),
            sorted(failed_pages)[:20], total_render_s, total_encode_s, total_write_s,
            100 * total_render_s / max(tot, 0.001),
        )
    else:
        logger.info(
            "PDF split complete | file=%s | pages=%d | "
            "render=%.1fs | encode=%.1fs | write=%.1fs | "
            "render_pct=%.0f%% | wall=%.1fs | throughput=%.1f pg/s",
            safe_filename, len(stored_list),
            total_render_s, total_encode_s, total_write_s,
            100 * total_render_s / max(tot, 0.001),
            wall_s,
            len(stored_list) / max(wall_s, 0.001),
        )
    return stored_list


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
    logger.info(
        "Directory import | batch=%s | imported=%d | skipped=%d | src=%s",
        batch_id, len(imported), len(skipped), src,
    )
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
