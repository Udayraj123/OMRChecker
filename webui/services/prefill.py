"""Service layer for the prefill answer-sheet feature.

Wraps ``prefill_only_package.prefill_answer_sheet_final``. Batch outputs are
streamed directly to a temp file on disk so peak memory is bounded regardless
of row count (a 5k-row PDF must not OOM the server).
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import tempfile
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from webui.services.scan_simulation import (
    BubbleGeometry,
    MarkerBox,
    apply_scan_simulation,
    normalize_realism_preset,
)

# Built-in blank template shipped with the package.
# When running as a PyInstaller frozen bundle sys._MEIPASS is the _internal/
# directory where data files are extracted; fall back to the source-tree path.
if getattr(sys, "frozen", False):
    _PKG_ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    _PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = _PKG_ROOT / "prefill_only_package" / "blank_template_reference.png"

logger = logging.getLogger(__name__)

# Per-field clamps. Names that overflow drawable area produced runtime
# overflow / pillow ValueError in earlier stress runs; clamp at the source.
_MAX_FIELD_LEN = 200
# Strip control characters except common whitespace.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_field(value: Any, *, max_len: int = _MAX_FIELD_LEN) -> str:
    """Strip control chars & clamp length so renderer never sees pathological input."""
    text = "" if value is None else str(value)
    text = _CONTROL_RE.sub("", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


def _import_prefill():
    """Lazy import to avoid loading PIL at module-level if not needed."""
    from prefill_only_package.prefill_answer_sheet_final import prefill_sheet

    return prefill_sheet


def _import_prefill_module():
    """Lazy import of the full prefill module (needed for batch helpers)."""
    import prefill_only_package.prefill_answer_sheet_final as m

    return m


def _images_to_pdf_bytes(images) -> bytes:
    """Convert a list of PIL Images to a PDF byte string in memory."""
    imgs = [im.convert("RGB") for im in images]
    if not imgs:
        raise ValueError("No images to convert.")
    buf = io.BytesIO()
    imgs[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=imgs[1:],
        resolution=300.0,
    )
    return buf.getvalue()


def _images_to_pdf_bytes_fast(png_bytes_list: list[bytes]) -> bytes:
    """Assemble PNG bytes into PDF using PyMuPDF (faster, lower peak RAM)."""
    import fitz

    doc = fitz.open()
    for png_bytes in png_bytes_list:
        img_doc = fitz.open("png", png_bytes)
        pdf_bytes = img_doc.convert_to_pdf()
        img_doc.close()
        src = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(src)
        src.close()
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def _validate_candidate_number(candidate_number: str) -> None:
    if len(candidate_number) != 10 or not candidate_number.isdigit():
        raise ValueError("Candidate number must be exactly 10 digits.")


def _build_payload(stamped_bytes: bytes, row: dict[str, Any]) -> dict[str, Any]:
    """Validate + sanitise a row into a legacy worker payload (includes stamped_bytes)."""
    candidate_number = _clean_field(row.get("candidate_number"), max_len=10)
    _validate_candidate_number(candidate_number)
    return {
        "stamped_bytes": stamped_bytes,
        "student_name": _clean_field(row.get("student_name")),
        "school_name": _clean_field(row.get("school_name")),
        "exam_name": _clean_field(row.get("exam_name")),
        "candidate_number": candidate_number,
    }


def _build_fast_payload(
    row: dict[str, Any],
    output_format: str = "png",
    realism_preset: str = "none",
) -> dict[str, Any]:
    """Validate + sanitise a row into a fast worker payload (no stamped_bytes)."""
    candidate_number = _clean_field(row.get("candidate_number"), max_len=10)
    _validate_candidate_number(candidate_number)
    return {
        "student_name": _clean_field(row.get("student_name")),
        "school_name": _clean_field(row.get("school_name")),
        "exam_name": _clean_field(row.get("exam_name")),
        "candidate_number": candidate_number,
        "output_format": output_format,
        "realism_preset": normalize_realism_preset(realism_preset),
    }


# ---------------------------------------------------------------------------
# Module-level stamped template cache.
# The stamped PIL image (ArUco corners drawn) is constant for the lifetime of
# the process.  Caching it here avoids ~50 ms of disk-read + ArUco work on
# every single-sheet HTTP request.
# ---------------------------------------------------------------------------
_STAMPED_IMG_CACHE: 'Image.Image | None' = None  # type: ignore[name-defined]  # noqa: F821
_STAMPED_ARR_CACHE: 'np.ndarray | None' = None  # type: ignore[name-defined]  # noqa: F821


def _get_stamped_img():
    """Return the stamped PIL Image, building and caching it on first call."""
    global _STAMPED_IMG_CACHE, _STAMPED_ARR_CACHE
    if _STAMPED_IMG_CACHE is None:
        m = _import_prefill_module()
        t = time.perf_counter()
        _STAMPED_IMG_CACHE = m.load_stamped_template(DEFAULT_TEMPLATE)
        _STAMPED_ARR_CACHE = np.array(_STAMPED_IMG_CACHE)
        logger.info("Stamped template cached | %.1fms", (time.perf_counter() - t) * 1000)
    return _STAMPED_IMG_CACHE, _STAMPED_ARR_CACHE


def _stamp_template_once() -> bytes:
    """Render the ArUco-stamped template once, returning its PNG bytes."""
    stamped_img, _ = _get_stamped_img()
    assert stamped_img is not None
    stamped_buf = io.BytesIO()
    stamped_img.save(stamped_buf, format="PNG", compress_level=1)
    return stamped_buf.getvalue()


def _max_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _simulate_scan_if_needed(
    image,
    prefill_module,
    *,
    candidate_number: str,
    realism_preset: str = "none",
):
    """Apply optional deterministic scan simulation after content drawing."""
    preset = normalize_realism_preset(realism_preset)
    if preset == "none":
        return image

    w, h = image.size
    bubbles = [
        BubbleGeometry(
            column=int(item["column"]),
            digit=int(item["digit"]),
            cx=int(item["cx"]),
            cy=int(item["cy"]),
            radius=int(item["radius"]),
            filled=bool(item.get("filled", False)),
        )
        for item in prefill_module.candidate_bubble_geometry(w, h, candidate_number)
    ]
    markers = [
        MarkerBox(
            corner=int(item["corner"]),
            x0=int(item["x0"]),
            y0=int(item["y0"]),
            x1=int(item["x1"]),
            y1=int(item["y1"]),
        )
        for item in prefill_module.aruco_marker_boxes(w, h)
    ]
    return apply_scan_simulation(
        image,
        preset=preset,
        candidate_number=candidate_number,
        bubbles=bubbles,
        markers=markers,
    )


def _thread_render(payload: dict) -> bytes:
    """Thread worker: renders one prefill sheet using the shared in-process template cache.

    Threads share the stamped PIL Image already held in ``_STAMPED_IMG_CACHE``;
    no IPC or pickling is required.  PIL image operations release the GIL so
    multiple threads make real progress in parallel.
    """
    m = _import_prefill_module()
    stamped_img, _ = _get_stamped_img()
    assert stamped_img is not None
    img = stamped_img.copy()
    img = m._draw_sheet_content(
        img,
        payload['student_name'],
        payload['school_name'],
        payload['exam_name'],
        payload['candidate_number'],
    )
    img = _simulate_scan_if_needed(
        img,
        m,
        candidate_number=payload['candidate_number'],
        realism_preset=payload.get('realism_preset', 'none'),
    )
    buf = io.BytesIO()
    fmt = payload.get('output_format', 'png').lower()
    if fmt == 'jpeg':
        img.save(buf, format='JPEG', quality=payload.get('jpeg_quality', 75))
    else:
        img.save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def _iter_pngs_fast(payloads: list[dict], *, preserve_order: bool = True):
    """Yield ``(index, png_bytes_or_None, error_or_None)`` using a thread pool.

    Uses ``ThreadPoolExecutor`` (not ``ProcessPoolExecutor``) so spawned threads
    never inherit the server's listening socket — which was the root cause of
    orphaned workers stealing connections and hanging the server.

    PIL image operations release the GIL, so threads achieve real parallelism
    for the CPU-bound rendering work.  Falls back to serial on any executor
    error.
    """
    n = len(payloads)
    max_workers = _max_workers()
    window_size = max_workers * 4
    # If workers produce nothing for this many seconds, abort rather than
    # looping forever.  300 s (5 min) is generous even for very large batches.
    _MAX_IDLE_S = 300

    yielded_indexes: set[int] = set()
    ex = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures: dict = {}
        completed: dict[int, tuple[bytes | None, str | None]] = {}
        next_submit = [0]  # list so the inner closure can mutate it
        next_yield = 0
        last_progress = time.perf_counter()

        def submit_until_window() -> None:
            while next_submit[0] < n and len(futures) < window_size:
                idx = next_submit[0]
                future = ex.submit(_thread_render, payloads[idx])
                futures[future] = idx
                next_submit[0] += 1

        submit_until_window()
        while futures:
            done, _ = wait(
                list(futures.keys()), timeout=30, return_when=FIRST_COMPLETED
            )
            if not done:
                idle_secs = time.perf_counter() - last_progress
                logger.warning(
                    "Prefill worker pool idle for %.0fs | pending=%d/%d",
                    idle_secs,
                    len(futures),
                    n,
                )
                if idle_secs > _MAX_IDLE_S:
                    logger.error(
                        "Aborting prefill pool — no progress for %.0fs (limit=%ds)",
                        idle_secs,
                        _MAX_IDLE_S,
                    )
                    break
                continue
            for future in done:
                idx = futures.pop(future)
                last_progress = time.perf_counter()
                try:
                    result: tuple[bytes | None, str | None] = (future.result(), None)
                except Exception as exc:  # noqa: BLE001
                    result = (None, f"{type(exc).__name__}: {exc}")

                if preserve_order:
                    completed[idx] = result
                else:
                    yielded_indexes.add(idx)
                    yield idx, result[0], result[1]

            if preserve_order:
                while next_yield in completed:
                    result = completed.pop(next_yield)
                    yielded_indexes.add(next_yield)
                    yield next_yield, result[0], result[1]
                    next_yield += 1

            submit_until_window()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Thread worker pool error after %d/%d rows; falling back to serial: %s",
            len(yielded_indexes), n, exc,
        )
    finally:
        # Non-blocking shutdown: don't wait for stuck threads.
        # cancel_futures=True cancels any not-yet-started submissions.
        ex.shutdown(wait=False, cancel_futures=True)

    # Serial fallback for remaining.
    for idx in range(n):
        if idx in yielded_indexes:
            continue
        try:
            # Re-use stamped img directly in-process to avoid another pool.
            stamped_img, _ = _get_stamped_img()
            assert stamped_img is not None
            img = stamped_img.copy()
            m2 = _import_prefill_module()
            img = m2._draw_sheet_content(
                img,
                payloads[idx]['student_name'],
                payloads[idx]['school_name'],
                payloads[idx]['exam_name'],
                payloads[idx]['candidate_number'],
            )
            img = _simulate_scan_if_needed(
                img,
                m2,
                candidate_number=payloads[idx]['candidate_number'],
                realism_preset=payloads[idx].get('realism_preset', 'none'),
            )
            buf = io.BytesIO()
            img.save(buf, format='PNG', compress_level=1)
            yield idx, buf.getvalue(), None
        except Exception as inner:  # noqa: BLE001
            yield idx, None, f"{type(inner).__name__}: {inner}"


def _iter_pngs_with_fallback(payloads: list[dict]):
    """Legacy path kept for backward compatibility. Delegates to fast path."""
    fast_payloads = [
        {k: v for k, v in p.items() if k != "stamped_bytes"} for p in payloads
    ]
    yield from _iter_pngs_fast(fast_payloads)


def generate_single_png(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
    realism_preset: str = "none",
) -> bytes:
    candidate_number = _clean_field(candidate_number, max_len=10)
    _validate_candidate_number(candidate_number)
    realism_preset = normalize_realism_preset(realism_preset)
    m = _import_prefill_module()
    stamped_img, _ = _get_stamped_img()
    assert stamped_img is not None
    image = m._draw_sheet_content(
        stamped_img.copy(),
        _clean_field(student_name),
        _clean_field(school_name),
        _clean_field(exam_name),
        candidate_number,
    )
    image = _simulate_scan_if_needed(
        image,
        m,
        candidate_number=candidate_number,
        realism_preset=realism_preset,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def generate_single_pdf(
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
    realism_preset: str = "none",
) -> bytes:
    import fitz
    import struct
    candidate_number = _clean_field(candidate_number, max_len=10)
    _validate_candidate_number(candidate_number)
    png_bytes = generate_single_png(
        student_name,
        school_name,
        exam_name,
        candidate_number,
        realism_preset=realism_preset,
    )
    w, h = struct.unpack('>II', png_bytes[16:24])
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=png_bytes)
    buf = io.BytesIO()
    doc.save(buf, garbage=0)
    doc.close()
    return buf.getvalue()


def generate_batch_pdf_to_file(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str = "none",
) -> dict:
    """Stream PDF generation directly to ``dst_path``.

    Workers output JPEG bytes; JPEG is stored natively in PDF as DCT so no
    re-encoding or deflate pass is needed.  Progress is logged every 500 sheets.
    Returns a metadata dict: ``{count, successes, errors, elapsed_s, size_bytes}``.
    """
    import fitz  # PyMuPDF

    count = len(rows)
    logger.info("Prefill batch PDF started | count=%d", count)
    t_batch = time.perf_counter()

    realism_preset = normalize_realism_preset(realism_preset)
    payloads = [
        _build_fast_payload(row, output_format="jpeg", realism_preset=realism_preset)
        for row in rows
    ]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    successes = 0
    errors: list[str] = []
    last_log = time.perf_counter()

    # Log progress at ~10% intervals (min every 100 rows, max every 500).
    _progress_step = max(100, min(500, count // 10 or 1))
    try:
        for idx, img_bytes, err in _iter_pngs_fast(payloads):
            if err or img_bytes is None:
                errors.append(f"row {idx}: {err or 'empty result'}")
                continue
            try:
                # JPEG bytes: read dimensions via fitz (avoids struct parsing JPEG SOF)
                tmp = fitz.open("jpeg", img_bytes)
                w, h = tmp[0].rect.width, tmp[0].rect.height
                tmp.close()
                page = doc.new_page(width=int(w), height=int(h))
                page.insert_image(page.rect, stream=img_bytes)
                successes += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {idx}: {type(exc).__name__}: {exc}")
            # Progress log at ~10% intervals or every 30s
            now = time.perf_counter()
            if successes % _progress_step == 0 and successes > 0 or now - last_log > 30:
                rate = successes / (now - t_batch) * 60
                logger.info(
                    "Prefill PDF progress | %d/%d (%.0f/min) | err=%d",
                    successes, count, rate, len(errors),
                )
                last_log = now
        save_start = time.perf_counter()
        logger.info(
            "Prefill PDF final save started | pages=%d | target=%s",
            successes,
            dst_path,
        )
        # Full garbage collection (garbage=4) is very expensive on thousands
        # of image-only pages and looks like a hang after rendering finishes.
        # These files are newly built, so a plain save is enough and much faster.
        doc.save(str(dst_path), garbage=0, deflate=False)
        logger.info(
            "Prefill PDF final save complete | pages=%d | elapsed=%.1fs",
            successes,
            time.perf_counter() - save_start,
        )
    finally:
        doc.close()

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    rate = (successes / elapsed) * 60 if elapsed > 0 else 0
    logger.info(
        "Prefill batch PDF complete | count=%d | ok=%d | err=%d | elapsed=%.1fs | "
        "rate=%.0f/min | size_mb=%.1f",
        count, successes, len(errors), elapsed, rate, size_bytes / (1024 * 1024),
    )
    return {
        "count": count,
        "successes": successes,
        "errors": errors[:50],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
    }


def generate_batch_zip_to_file(
    rows: list[dict[str, Any]],
    dst_path: Path,
    realism_preset: str = "none",
) -> dict:
    """Stream ZIP generation directly to ``dst_path``. Bounded memory."""
    count = len(rows)
    logger.info("Prefill batch ZIP started | count=%d", count)
    t_batch = time.perf_counter()

    payloads: list[dict] = []
    filenames: list[str] = []
    for i, row in enumerate(rows, start=1):
        payloads.append(_build_fast_payload(row, realism_preset=realism_preset))
        filename = Path(_clean_field(row.get("output_file", "")) or "").name \
            or f"sheet_{i:03d}.png"
        if not filename.lower().endswith(".png"):
            filename += ".png"
        filenames.append(filename)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    errors: list[str] = []
    with zipfile.ZipFile(
        dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zf:
        for idx, png_bytes, err in _iter_pngs_fast(payloads, preserve_order=False):
            if err or png_bytes is None:
                errors.append(f"row {idx} ({filenames[idx]}): {err or 'empty result'}")
                continue
            try:
                zf.writestr(filenames[idx], png_bytes)
                successes += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {idx}: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t_batch
    size_bytes = dst_path.stat().st_size if dst_path.exists() else 0
    rate = (successes / elapsed) * 60 if elapsed > 0 else 0
    logger.info(
        "Prefill batch ZIP complete | count=%d | ok=%d | err=%d | elapsed=%.1fs | "
        "rate=%.0f/min | size_mb=%.1f",
        count, successes, len(errors), elapsed, rate, size_bytes / (1024 * 1024),
    )
    return {
        "count": count,
        "successes": successes,
        "errors": errors[:50],
        "elapsed_s": round(elapsed, 2),
        "size_bytes": size_bytes,
    }


# Backwards-compatible in-memory wrappers (still used by older callers / tests).
# These now stream to a temp file first then read it back, so peak memory matches
# the streaming path even when the caller wants raw bytes.
def generate_batch_pdf(rows: list[dict[str, Any]], realism_preset: str = "none") -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        generate_batch_pdf_to_file(rows, tmp_path, realism_preset=realism_preset)
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def generate_batch_zip(rows: list[dict[str, Any]], realism_preset: str = "none") -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        generate_batch_zip_to_file(rows, tmp_path, realism_preset=realism_preset)
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
