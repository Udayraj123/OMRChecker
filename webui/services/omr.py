"""Wrapper that drives the OMRChecker engine for a given batch.

The engine reads ``template.json``, ``config.json`` and ``evaluation.json``
from the *same* directory as the images. To keep the engine untouched, we
copy any batch-level JSON files into ``inputs/`` just before processing
and tidy them up afterwards.
"""

from __future__ import annotations

import csv
import copy
import json
import logging
import os
import re
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Literal

import cv2
from PIL import Image as _PILImage

from src.defaults.config import CONFIG_DEFAULTS
from webui.schemas import (
    BatchStatus,
    ResultsPayload,
    ResultsRow,
)
from webui.services import batches as batches_service
from webui.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_COPIED_JSON_FILES = ("template.json", "config.json", "evaluation.json")
_RUNTIME_DIR_NAME = "_runtime"
_RUNTIME_BASE_DIR_NAME = "_base"
_DISPLAY_KEYS = ("display_height", "display_width", "processing_height", "processing_width")

_batch_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

_QC_Q_COL = re.compile(r"^q\d+", re.IGNORECASE)


def _lock_for(batch_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _batch_locks.get(batch_id)
        if lock is None:
            lock = threading.Lock()
            _batch_locks[batch_id] = lock
        return lock


def release_batch_lock(batch_id: str) -> None:
    """Remove the per-batch processing lock when the batch is deleted.

    Safe to call even if no lock exists for the batch.
    """
    with _locks_guard:
        _batch_locks.pop(batch_id, None)


def _is_cancel_requested(batch_id: str, settings: Settings) -> bool:
    metadata = batches_service.get_batch_metadata(batch_id, settings)
    return bool(metadata.get("cancel_requested", False))


def _image_ended_up_in_errors(outputs_dir: Path, file_name: str) -> bool:
    """Return True when the OMR engine moved ``file_name`` into the
    ``Manual/ErrorFiles`` directory (which happens when a pre-processor
    like ``CropOnMarkers`` could not locate its markers)."""
    error_dir = outputs_dir / "Manual" / "ErrorFiles"
    return (error_dir / file_name).exists()


def _mark_cancelled(
    batch_id: str,
    reason: str,
    settings: Settings,
) -> None:
    batches_service.update_status(
        batch_id,
        BatchStatus.cancelled,
        last_error=reason,
        settings=settings,
    )


def _discover_input_images(batch_id: str, settings: Settings) -> list[Path]:
    """Return all processable images for a batch."""
    batch_root = batches_service.get_batch_root(batch_id, settings)
    asset_names = {Path(asset).name for asset in _get_template_relative_assets(batch_root)}
    return [
        path
        for path in batches_service.list_input_image_paths(batch_id, settings)
        if path.name not in asset_names
    ]


def _scale_dimensions(
    source_width: int,
    source_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    """Scale dimensions down to fit within a bounding box without upscaling."""
    scale = min(max_width / source_width, max_height / source_height, 1.0)
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return width, height


def _template_page_dimensions(template_payload: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(template_payload, dict):
        return None
    page_dimensions = template_payload.get("pageDimensions")
    if (
        not isinstance(page_dimensions, list)
        or len(page_dimensions) != 2
        or not all(isinstance(value, int | float) for value in page_dimensions)
    ):
        return None
    width, height = (int(page_dimensions[0]), int(page_dimensions[1]))
    if width <= 0 or height <= 0:
        return None
    return width, height


def _uses_marker_template_space(template_payload: dict[str, Any] | None) -> bool:
    if not isinstance(template_payload, dict):
        return False
    preprocessors = template_payload.get("preProcessors")
    if not isinstance(preprocessors, list):
        return False
    return any(
        isinstance(processor, dict) and processor.get("name") == "CropOnMarkers"
        for processor in preprocessors
    )


def _compute_dynamic_dimensions(
    image_path: Path,
    template_payload: dict[str, Any] | None = None,
    rotation_degrees: int = 0,
) -> dict[str, int]:
    """Compute per-image display and processing dimensions.

    Uses PIL header-only decoding to read image size without decompressing
    pixel data, which is significantly faster than cv2.imread for this purpose.
    """
    try:
        with _PILImage.open(image_path) as _pil:
            # PIL returns (width, height); no pixel decoding happens here.
            source_width, source_height = _pil.size
    except Exception as exc:
        raise ValueError(f"Could not read image: {image_path.as_posix()}") from exc
    if rotation_degrees in {90, 270}:
        source_width, source_height = source_height, source_width
    display_width, display_height = _scale_dimensions(
        source_width,
        source_height,
        int(CONFIG_DEFAULTS.dimensions.display_width),
        int(CONFIG_DEFAULTS.dimensions.display_height),
    )
    template_dimensions = (
        _template_page_dimensions(template_payload)
        if _uses_marker_template_space(template_payload)
        else None
    )
    if template_dimensions is None:
        processing_width, processing_height = _scale_dimensions(
            source_width,
            source_height,
            int(CONFIG_DEFAULTS.dimensions.processing_width),
            int(CONFIG_DEFAULTS.dimensions.processing_height),
        )
    else:
        processing_width, processing_height = template_dimensions

    return {
        "source_height": int(source_height),
        "source_width": int(source_width),
        "display_height": int(display_height),
        "display_width": int(display_width),
        "processing_height": int(processing_height),
        "processing_width": int(processing_width),
    }


def _merge_dimensions_into_config(
    config: dict | None, dynamic_dimensions: dict[str, int]
) -> dict:
    """Merge computed dimensions into an existing config payload."""
    merged = copy.deepcopy(config) if isinstance(config, dict) else {}
    dimensions = merged.get("dimensions")
    if not isinstance(dimensions, dict):
        dimensions = {}
    for key in _DISPLAY_KEYS:
        dimensions[key] = int(dynamic_dimensions[key])
    merged["dimensions"] = dimensions
    return merged


def _write_runtime_config(config: dict, dst: Path) -> None:
    """Write staged config while forcing non-interactive web execution."""
    staged = copy.deepcopy(config)
    outputs = staged.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    outputs["show_image_level"] = 0
    staged["outputs"] = outputs
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(staged, fh, indent=2, sort_keys=True)


def _load_template_payload(batch_root: Path) -> dict[str, Any]:
    template_path = batch_root / "template.json"
    if not template_path.exists():
        return {}
    with template_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _with_runtime_template_defaults(template_payload: dict[str, Any]) -> dict[str, Any]:
    """Apply safe runtime defaults without mutating the user's saved template."""
    if not isinstance(template_payload, dict):
        return {}
    runtime_template = copy.deepcopy(template_payload)
    preprocessors = runtime_template.get("preProcessors")
    if not isinstance(preprocessors, list):
        return runtime_template
    for processor in preprocessors:
        if not isinstance(processor, dict) or processor.get("name") != "CropOnMarkers":
            continue
        options = processor.get("options")
        if not isinstance(options, dict):
            options = {}
            processor["options"] = options
        if "markerCorners" in options:
            options.setdefault("markerSearchPadding", 20)
            options.setdefault("fallbackToExpandedMarkerCorners", True)
            # Heavy-bubble or slightly tilted scans legitimately produce
            # corner-score spreads ~0.4-0.7. Keep older saved templates
            # tolerant by injecting the safer default at runtime only when
            # the user hasn't pinned a stricter value of their own.
            options.setdefault("max_matching_variation", 0.5)
    return runtime_template


def _write_runtime_template(template_payload: dict[str, Any], dst: Path) -> None:
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(_with_runtime_template_defaults(template_payload), fh, indent=2, sort_keys=True)


def _collect_relative_paths(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "relativePath" and isinstance(item, str):
                found.append(item)
            else:
                found.extend(_collect_relative_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_relative_paths(item))
    return found


def _get_template_relative_assets(batch_root: Path) -> list[str]:
    """Return relative asset paths referenced by the template."""
    template_payload = _load_template_payload(batch_root)
    return _collect_relative_paths(template_payload.get("preProcessors", []))


def _resolve_batch_asset(batch_root: Path, relative_path: str) -> Path:
    """Resolve a template-referenced relative asset from batch root or inputs."""
    direct = (batch_root / relative_path).resolve()
    if direct.exists():
        return direct

    inputs_candidate = (batch_root / "inputs" / Path(relative_path).name).resolve()
    if inputs_candidate.exists():
        return inputs_candidate

    raise FileNotFoundError(
        f"Missing template asset '{relative_path}'. Place it in the batch root or inputs folder."
    )


def _copy_runtime_assets(batch_root: Path, runtime_root: Path) -> None:
    """Copy any template-referenced relative assets into the runtime folder."""
    for relative_path in _get_template_relative_assets(batch_root):
        src = _resolve_batch_asset(batch_root, relative_path)
        dst = runtime_root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_runtime_base(batch_root: Path) -> Path:
    runtime_root = batch_root / _RUNTIME_DIR_NAME
    base_root = runtime_root / _RUNTIME_BASE_DIR_NAME
    if base_root.exists():
        shutil.rmtree(base_root)
    base_root.mkdir(parents=True, exist_ok=True)

    template_payload = _load_template_payload(batch_root)
    if template_payload:
        _write_runtime_template(template_payload, base_root / "template.json")
    evaluation_src = batch_root / "evaluation.json"
    if evaluation_src.exists():
        _link_or_copy(evaluation_src, base_root / "evaluation.json")

    _copy_runtime_assets(batch_root, base_root)
    return base_root


def _rotate_image_for_runtime(
    src: Path,
    dst: Path,
    rotation_degrees: int,
    pre_resize_to: tuple[int, int] | None = None,
) -> None:
    """Copy *src* into *dst*, applying rotation and optional downscale.

    ``pre_resize_to`` is ``(width, height)`` and is only applied when the
    operation would make the image *smaller* (never upscales).
    """
    if rotation_degrees == 0 and pre_resize_to is None:
        shutil.copy2(src, dst)
        return

    # Build a cache key that encodes both rotation and resize so the cached
    # file stays valid across different processing-dimension configs.
    resize_tag = f"_s{pre_resize_to[0]}x{pre_resize_to[1]}" if pre_resize_to else ""
    rot_tag = f"_rot{rotation_degrees}" if rotation_degrees else ""
    cache_key = f"{src.stem}{rot_tag}{resize_tag}{src.suffix.lower()}"

    rotated_dir = dst.parent.parent / "_rotated"
    rotated_dir.mkdir(parents=True, exist_ok=True)
    cached = rotated_dir / cache_key
    try:
        if cached.exists() and cached.stat().st_mtime >= src.stat().st_mtime:
            shutil.copy2(cached, dst)
            return
    except OSError:
        pass

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {src.as_posix()}")

    if rotation_degrees:
        rotate_codes = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        rotate_code = rotate_codes.get(rotation_degrees)
        if rotate_code is None:
            raise ValueError(f"Unsupported rotation: {rotation_degrees}")
        image = cv2.rotate(image, rotate_code)

    if pre_resize_to is not None:
        proc_w, proc_h = pre_resize_to
        h, w = image.shape[:2]
        if w > proc_w or h > proc_h:
            image = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    if not cv2.imwrite(str(dst), image):
        raise ValueError(f"Could not write runtime image: {dst.as_posix()}")
    try:
        cv2.imwrite(str(cached), image)
    except Exception:
        pass


def _prepare_runtime_dir(
    batch_root: Path,
    image_path: Path,
    runtime_config: dict,
    index: int,
    rotation_degrees: int = 0,
    base_root: Path | None = None,
    pre_resize_to: tuple[int, int] | None = None,
) -> Path:
    """Build an isolated single-image runtime directory for engine execution."""
    runtime_root = batch_root / _RUNTIME_DIR_NAME / f"{index:04d}_{image_path.stem}"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    _rotate_image_for_runtime(
        image_path,
        runtime_root / image_path.name,
        rotation_degrees,
        pre_resize_to=pre_resize_to,
    )
    if base_root is None:
        base_root = _prepare_runtime_base(batch_root)
    for name in ("template.json", "evaluation.json"):
        src = base_root / name
        if src.exists():
            _link_or_copy(src, runtime_root / name)
    for path in base_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {"template.json", "evaluation.json"}:
            continue
        rel = path.relative_to(base_root)
        _link_or_copy(path, runtime_root / rel)
    _write_runtime_config(runtime_config, runtime_root / "config.json")
    return runtime_root


def _cleanup_runtime_dir(runtime_dir: Path | None) -> None:
    if runtime_dir and runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _default_max_workers() -> int:
    # Allow explicit override via environment variable.
    env_val = os.environ.get("OMR_WEBUI_MAX_WORKERS")
    if env_val:
        try:
            return max(1, min(32, int(env_val)))
        except (TypeError, ValueError):
            pass
    cpu = os.cpu_count() or 2
    # Leave 1 core free for the main process; cap at 16 for modern many-core
    # machines. Override via max_workers in config.json (accepts 1-32).
    return max(1, min(cpu - 1, 16))


def _coerce_max_workers(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _default_max_workers()
    return max(1, min(32, parsed))


def _process_one_image(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint to process a single image in its own runtime dir."""
    # Each worker process should use exactly 1 OpenCV thread so that the
    # OS-level ProcessPoolExecutor provides the parallelism without internal
    # thread-pool contention (N workers × M OpenCV threads = N*M threads
    # competing for N cores).
    cv2.setNumThreads(1)

    from main import entry_point_for_args

    batch_root = Path(payload["batch_root"])
    outputs_dir = Path(payload["outputs_dir"])
    image_path = Path(payload["image_path"])
    base_config = payload.get("base_config") or {}
    template_payload = payload.get("template_payload") or {}
    base_root = Path(payload["base_root"])
    rotation_degrees = int(payload.get("rotation_degrees", 0))
    index = int(payload.get("index", 1))

    runtime_dir: Path | None = None
    logger.debug("Worker | index=%d | file=%s", index, image_path.name)
    t_start = time.perf_counter()
    try:
        if not base_root.exists():
            base_root = _prepare_runtime_base(batch_root)
        dynamic_dimensions = _compute_dynamic_dimensions(
            image_path, template_payload, rotation_degrees
        )
        runtime_config = _merge_dimensions_into_config(base_config, dynamic_dimensions)

        # Pre-resize: if the source image is meaningfully larger than the
        # processing dimensions, shrink it before writing to the runtime dir.
        # The engine will then read a smaller file and its own resize becomes
        # a near-no-op, saving both I/O and per-pixel work in ArUco/bubble steps.
        src_w = dynamic_dimensions["source_width"]
        src_h = dynamic_dimensions["source_height"]
        proc_w = dynamic_dimensions["processing_width"]
        proc_h = dynamic_dimensions["processing_height"]
        pre_resize_to: tuple[int, int] | None = None
        if src_w > proc_w * 1.1 or src_h > proc_h * 1.1:
            pre_resize_to = (proc_w, proc_h)

        runtime_dir = _prepare_runtime_dir(
            batch_root,
            image_path,
            runtime_config,
            index,
            rotation_degrees,
            base_root=base_root,
            pre_resize_to=pre_resize_to,
        )
        worker_outputs_dir = outputs_dir / "_workers" / f"{index:04d}_{image_path.stem}"
        worker_outputs_dir.mkdir(parents=True, exist_ok=True)
        args = {
            "input_paths": [str(runtime_dir)],
            "output_dir": str(worker_outputs_dir),
            "debug": False,
            "autoAlign": False,
            "setLayout": False,
        }
        entry_point_for_args(args)
        ended_in_errors = _image_ended_up_in_errors(worker_outputs_dir, image_path.name)

        def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
            if not path.exists():
                return [], []
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                raw = [row for row in reader if row]
            if not raw:
                return [], []
            return raw[0], raw[1:]

        results_csv = next((worker_outputs_dir / "Results").glob("Results_*.csv"), None)
        mm_csv = worker_outputs_dir / "Manual" / "MultiMarkedFiles.csv"
        err_csv = worker_outputs_dir / "Manual" / "ErrorFiles.csv"

        results_header, results_rows = read_rows(results_csv) if results_csv else ([], [])
        mm_header, mm_rows = read_rows(mm_csv)
        err_header, err_rows = read_rows(err_csv)

        checked_image = worker_outputs_dir / "CheckedOMRs" / image_path.name
        error_image = worker_outputs_dir / "Manual" / "ErrorFiles" / image_path.name
        mm_image = worker_outputs_dir / "Manual" / "MultiMarkedFiles" / image_path.name

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("Worker done | index=%d | file=%s | elapsed_ms=%.0f | error=%s", index, image_path.name, elapsed_ms, None)
        return {
            "file_name": image_path.name,
            "index": index,
            "dynamic_dimensions": dynamic_dimensions,
            "runtime_config": runtime_config,
            "ended_in_errors": ended_in_errors,
            "worker_outputs_dir": str(worker_outputs_dir),
            "results_header": results_header,
            "results_rows": results_rows,
            "mm_header": mm_header,
            "mm_rows": mm_rows,
            "err_header": err_header,
            "err_rows": err_rows,
            "checked_image": str(checked_image) if checked_image.exists() else None,
            "error_image": str(error_image) if error_image.exists() else None,
            "mm_image": str(mm_image) if mm_image.exists() else None,
            "error": None,
        }
    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("Worker done | index=%d | file=%s | elapsed_ms=%.0f | error=%s", index, image_path.name, elapsed_ms, f"{type(exc).__name__}: {exc}")
        return {
            "file_name": image_path.name,
            "index": index,
            "dynamic_dimensions": None,
            "runtime_config": None,
            "ended_in_errors": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        _cleanup_runtime_dir(runtime_dir)


def _write_non_interactive_config(src: Path, dst: Path) -> None:
    """Copy config while forcing non-interactive processing in web runs.

    The core engine uses ``outputs.show_image_level`` to decide whether to open
    OpenCV windows and block on ``waitKey``. In API/background usage this can
    hang processing, so we always override it to 0 for staged run config.
    """
    with src.open("r", encoding="utf-8") as fh:
        content = json.load(fh)
    outputs = content.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    outputs["show_image_level"] = 0
    content["outputs"] = outputs
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, sort_keys=True)

def run_batch_sync(batch_id: str, settings: Settings | None = None) -> None:
    """Run OMR processing for a single batch synchronously.

    Updates the batch status throughout (``running`` → ``done``/``failed``).
    Safe to call from a ``BackgroundTasks`` task or directly from tests.
    """
    settings = settings or get_settings()
    lock = _lock_for(batch_id)
    if not lock.acquire(blocking=False):
        logger.info("Batch %s is already processing; skipping duplicate run", batch_id)
        return
    logger.info("OMR run starting | batch_id=%s", batch_id)

    runtime_dir: Path | None = None
    run_started_at = time.monotonic()  # initialised early so except block can always reference it
    try:
        batch_root = batches_service.get_batch_root(batch_id, settings)
        outputs_dir = batch_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Remove any stale worker scratch dirs left by a previous interrupted
        # run before discovering images, so they can't interfere with new
        # worker paths and won't be double-counted at end-of-run cleanup.
        stale_workers = outputs_dir / "_workers"
        if stale_workers.exists():
            logger.info("OMR run: removing stale _workers/ from prior interrupted run | batch_id=%s", batch_id)
            shutil.rmtree(stale_workers, ignore_errors=True)

        input_images = _discover_input_images(batch_id, settings)

        if _is_cancel_requested(batch_id, settings):
            _mark_cancelled(batch_id, "Cancelled before processing started.", settings)
            return

        if not input_images:
            batches_service.update_status(
                batch_id,
                BatchStatus.failed,
                last_error="No input images present in batch.",
                settings=settings,
            )
            return

        if not (batch_root / "template.json").exists():
            batches_service.update_status(
                batch_id,
                BatchStatus.failed,
                last_error=(
                    "template.json is required before running OMR. "
                    "Upload or PUT one for this batch."
                ),
                settings=settings,
            )
            return

        batches_service.update_status(batch_id, BatchStatus.running, settings=settings)
        run_started_at = time.monotonic()
        batches_service.update_batch_metadata(
            batch_id,
            {
                "processed_files": 0,
                "total_files": len(input_images),
                "latest_processed_file": None,
                "latest_dynamic_dimensions": None,
                "dynamic_dimensions_by_file": {},
                "preprocess_failures": [],
                "run_started_at": time.time(),
            },
            settings,
        )
        preprocess_failures: list[str] = []
        preprocess_worker_errors: list[str] = []

        base_config = batches_service.get_json_document(batch_id, "config", settings) or {}
        template_payload = _load_template_payload(batch_root)
        metadata = batches_service.get_batch_metadata(batch_id, settings)
        rotation_degrees = int(metadata.get("rotation_degrees", 0))
        max_workers = _coerce_max_workers(
            (base_config.get("outputs") or {}).get("max_workers")
        )

        base_root = _prepare_runtime_base(batch_root)
        logger.info("OMR batch started | batch_id=%s | images=%d | workers=%d", batch_id, len(input_images), max_workers)

        dynamic_dimensions_by_file: dict[str, dict[str, int]] = {}
        latest_persisted_config = copy.deepcopy(base_config)
        latest_index_completed = 0
        latest_file_name: str | None = None
        latest_dimensions: dict[str, int] | None = None
        latest_runtime_config: dict[str, Any] = copy.deepcopy(base_config)
        aggregated_results_header: list[str] | None = None
        aggregated_results_rows: list[tuple[int, list[str]]] = []
        aggregated_mm_header: list[str] | None = None
        aggregated_mm_rows: list[tuple[int, list[str]]] = []
        aggregated_err_header: list[str] | None = None
        aggregated_err_rows: list[tuple[int, list[str]]] = []

        def copy_if_present(path_str: str | None, dest_dir: Path) -> None:
            if not path_str:
                return
            src = Path(path_str)
            if not src.exists():
                return
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / src.name)

        def submit_payload(idx: int, image: Path) -> dict[str, Any]:
            return {
                "batch_root": str(batch_root),
                "outputs_dir": str(outputs_dir),
                "image_path": str(image),
                "base_config": base_config,
                "template_payload": template_payload,
                "base_root": str(base_root),
                "rotation_degrees": rotation_degrees,
                "index": idx,
            }

        futures = {}
        next_index = 1
        image_iter = iter(input_images)
        _executor_gone = False  # set when executor shuts down mid-run (e.g. server reload)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            while len(futures) < max_workers:
                try:
                    image = next(image_iter)
                except StopIteration:
                    break
                try:
                    futures[executor.submit(_process_one_image, submit_payload(next_index, image))] = image
                except RuntimeError:
                    _executor_gone = True
                    break
                next_index += 1

            last_milestone = 0
            completed = 0
            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    image = futures.pop(future)
                    result = future.result()
                    completed += 1
                    _total = len(input_images)
                    _pct = completed * 100 // _total
                    _milestone = (_pct // 10) * 10
                    if _milestone > last_milestone and _milestone > 0:
                        last_milestone = _milestone
                        _elapsed_s = time.monotonic() - run_started_at
                        _rate_min = (completed / _elapsed_s * 60) if _elapsed_s > 0 else 0
                        logger.info("OMR progress | %d/%d | elapsed=%.1fs | rate=%.0f/min | failures=%d", completed, _total, _elapsed_s, _rate_min, len(preprocess_failures))
                    file_name = result.get("file_name") or image.name
                    result_index = int(result.get("index") or 0)
                    dyn = result.get("dynamic_dimensions") or {}
                    ended_in_errors = bool(result.get("ended_in_errors", False))
                    worker_error = result.get("error")
                    results_header = result.get("results_header") or []
                    results_rows = result.get("results_rows") or []
                    mm_header = result.get("mm_header") or []
                    mm_rows = result.get("mm_rows") or []
                    err_header = result.get("err_header") or []
                    err_rows = result.get("err_rows") or []

                    if results_header and aggregated_results_header is None:
                        aggregated_results_header = list(results_header)
                    if mm_header and aggregated_mm_header is None:
                        aggregated_mm_header = list(mm_header)
                    if err_header and aggregated_err_header is None:
                        aggregated_err_header = list(err_header)
                    aggregated_results_rows.extend(
                        (result_index, row) for row in results_rows
                    )
                    aggregated_mm_rows.extend((result_index, row) for row in mm_rows)
                    aggregated_err_rows.extend((result_index, row) for row in err_rows)

                    copy_if_present(
                        result.get("checked_image"),
                        outputs_dir / "CheckedOMRs",
                    )
                    copy_if_present(
                        result.get("error_image"),
                        outputs_dir / "Manual" / "ErrorFiles",
                    )
                    copy_if_present(
                        result.get("mm_image"),
                        outputs_dir / "Manual" / "MultiMarkedFiles",
                    )

                    if isinstance(dyn, dict) and dyn:
                        dynamic_dimensions_by_file[file_name] = dyn
                        runtime_config = result.get("runtime_config")
                        if result_index >= latest_index_completed:
                            latest_index_completed = result_index
                            latest_file_name = file_name
                            latest_dimensions = dyn
                            if isinstance(runtime_config, dict):
                                latest_runtime_config = runtime_config
                    if ended_in_errors:
                        preprocess_failures.append(file_name)
                    if worker_error:
                        preprocess_worker_errors.append(f"{file_name}: {worker_error}")

                    batches_service.update_batch_metadata(
                        batch_id,
                        {
                            "processed_files": completed,
                            "total_files": len(input_images),
                            "latest_processed_file": latest_file_name,
                            "latest_dynamic_dimensions": latest_dimensions,
                            # dynamic_dimensions_by_file grows O(n) — only
                            # written in the final completion call, not here.
                            "preprocess_failures": list(preprocess_failures),
                            "run_elapsed_s": round(time.monotonic() - run_started_at, 1),
                        },
                        settings,
                    ) if (completed % 5 == 0 or completed == len(input_images)) else None

                if _is_cancel_requested(batch_id, settings):
                    # Cancel futures that haven't been picked up by a worker
                    # yet so we don't block waiting for a full backlog of
                    # unstarted work.  In-flight futures in worker processes
                    # cannot be interrupted and will run to completion.
                    for _pending in list(futures.keys()):
                        _pending.cancel()
                    futures.clear()
                    _mark_cancelled(
                        batch_id,
                        "Stop requested. In-flight images will finish; pending work cancelled.",
                        settings,
                    )
                    return

                if not _executor_gone:
                    while len(futures) < max_workers:
                        try:
                            image = next(image_iter)
                        except StopIteration:
                            break
                        try:
                            futures[executor.submit(_process_one_image, submit_payload(next_index, image))] = image
                        except RuntimeError:
                            _executor_gone = True
                            break
                        next_index += 1

        results_dir = outputs_dir / "Results"
        manual_dir = outputs_dir / "Manual"
        results_dir.mkdir(parents=True, exist_ok=True)
        manual_dir.mkdir(parents=True, exist_ok=True)
        if aggregated_results_header is not None:
            time_now_hrs = time.strftime("%I%p", time.localtime())
            path = results_dir / f"Results_{time_now_hrs}.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(aggregated_results_header)
                writer.writerows(
                    row for _, row in sorted(aggregated_results_rows, key=lambda item: item[0])
                )
        if aggregated_mm_header is not None:
            path = manual_dir / "MultiMarkedFiles.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(aggregated_mm_header)
                writer.writerows(
                    row for _, row in sorted(aggregated_mm_rows, key=lambda item: item[0])
                )
        if aggregated_err_header is not None:
            path = manual_dir / "ErrorFiles.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(aggregated_err_header)
                writer.writerows(
                    row for _, row in sorted(aggregated_err_rows, key=lambda item: item[0])
                )

        _elapsed_total = time.monotonic() - run_started_at
        _rate_total = (completed * 60 / _elapsed_total) if _elapsed_total > 0 else 0
        logger.info("OMR batch complete | batch_id=%s | total=%d | failures=%d | elapsed=%.1fs | rate=%.0f/min", batch_id, completed, len(preprocess_failures), _elapsed_total, _rate_total)

        batches_service.save_json_document(batch_id, "config", latest_runtime_config, settings)

        if preprocess_worker_errors:
            batches_service.update_batch_metadata(
                batch_id,
                {"worker_errors": list(preprocess_worker_errors)},
                settings,
            )

        batches_service.update_batch_metadata(
            batch_id,
            {"run_elapsed_s": round(time.monotonic() - run_started_at, 1)},
            settings,
        )

        if preprocess_failures and len(preprocess_failures) == len(input_images):
            batches_service.update_status(
                batch_id,
                BatchStatus.failed,
                last_error=(
                    "Marker / page preprocessing failed for every input image. "
                    "Check that template preprocessors can locate markers on "
                    "your sheets. Failed files: "
                    + ", ".join(preprocess_failures)
                ),
                settings=settings,
            )
        elif preprocess_failures:
            batches_service.update_status(
                batch_id,
                BatchStatus.done,
                last_error=(
                    f"{len(preprocess_failures)} of {len(input_images)} file(s) "
                    "failed marker / page preprocessing and were moved to "
                    "outputs/Manual/ErrorFiles: "
                    + ", ".join(preprocess_failures)
                ),
                settings=settings,
            )
        else:
            batches_service.update_status(
                batch_id, BatchStatus.done, settings=settings
            )
        # Delete internal worker scratch dirs AFTER status is visible to
        # clients.  Doing this before the status update would delay the
        # 'done' signal by 30+ seconds on large batches.
        shutil.rmtree(outputs_dir / "_workers", ignore_errors=True)
    except BaseException as exc:
        logger.exception("OMR run failed for batch %s", batch_id)
        batches_service.update_batch_metadata(
            batch_id,
            {"run_elapsed_s": round(time.monotonic() - run_started_at, 1)},
            settings,
        )
        batches_service.update_status(
            batch_id,
            BatchStatus.failed,
            last_error=f"{type(exc).__name__}: {exc}",
            settings=settings,
        )
    finally:
        _cleanup_runtime_dir(runtime_dir)
        lock.release()


def restart_and_run_batch_sync(batch_id: str, settings: Settings | None = None) -> None:
    """Background task for restart: remove prior run artifacts then run OMR.

    Doing the heavy rmtree here (background) instead of in the HTTP handler
    means the 202 response is returned immediately and the user sees log
    activity right away instead of a 30-second silent hang.
    """
    settings = settings or get_settings()
    batch_root = batches_service.get_batch_root(batch_id, settings)
    logger.info("OMR restart: removing prior run artifacts | batch_id=%s", batch_id)
    for name in ("outputs", "_runtime"):
        target = batch_root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
    (batch_root / "outputs").mkdir(parents=True, exist_ok=True)
    logger.info("OMR restart: artifact cleanup complete, starting run | batch_id=%s", batch_id)
    run_batch_sync(batch_id, settings)


def queue_run(batch_id: str, settings: Settings | None = None) -> None:
    """Mark a batch as queued so the caller can hand off to a background task."""
    settings = settings or get_settings()
    batches_service.update_batch_metadata(
        batch_id,
        {
            "cancel_requested": False,
            "processed_files": 0,
            "total_files": 0,
            "latest_processed_file": None,
            "latest_dynamic_dimensions": None,
            "dynamic_dimensions_by_file": {},
        },
        settings,
    )
    batches_service.update_status(batch_id, BatchStatus.queued, settings=settings)


def request_cancel(batch_id: str, settings: Settings | None = None) -> BatchStatus:
    """Request cooperative cancellation for a queued or running batch."""
    settings = settings or get_settings()
    batch = batches_service.get_batch(batch_id, settings)
    if batch.status not in {BatchStatus.queued, BatchStatus.running}:
        raise batches_service.InvalidBatchRequest(
            "Only queued or running batches can be stopped."
        )

    batches_service.update_batch_metadata(
        batch_id,
        {"cancel_requested": True},
        settings,
    )

    if batch.status == BatchStatus.queued:
        _mark_cancelled(batch_id, "Cancelled before processing started.", settings)
        return BatchStatus.cancelled

    batches_service.update_status(
        batch_id,
        BatchStatus.running,
        last_error="Stop requested. The current image will finish first.",
        settings=settings,
    )
    return BatchStatus.running


def _read_csv_records(
    path: Path,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[str], list[dict[str, str]], int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return [], [], 0

            records: list[dict[str, str]] = []
            total = 0
            for row in reader:
                if not row or not any(str(value).strip() for value in row):
                    continue
                if total >= offset and (limit is None or len(records) < limit):
                    records.append(dict(zip(header, row)))
                total += 1
    except (FileNotFoundError, OSError):
        return [], [], 0

    return header, records, total


def _find_error_files_csvs(batch_id: str, settings: Settings) -> list[Path]:
    batch_root = batches_service.get_batch_root(batch_id, settings)
    outputs = batch_root / "outputs"
    if not outputs.exists():
        return []
    candidates = [
        path
        for manual_dir in outputs.rglob("Manual")
        if manual_dir.is_dir()
        and "_workers" not in manual_dir.relative_to(outputs).parts
        for path in manual_dir.glob("ErrorFiles*.csv")
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return candidates


def _result_row_from_record(
    record: dict[str, str], status: Literal["ok", "failed"] = "ok"
) -> ResultsRow:
    responses = {
        key: value
        for key, value in record.items()
        if key not in {"file_id", "input_path", "output_path", "score"}
    }
    error_reason = None
    if status == "failed":
        error_reason = "Marker preprocessing failed (image moved to Manual/ErrorFiles)."
    return ResultsRow(
        file_id=record.get("file_id") or "",
        input_path=record.get("input_path"),
        output_path=record.get("output_path"),
        score=record.get("score"),
        status=status,
        error_reason=error_reason,
        qc_flags=[],
        nr_count=0,
        nr_percent=0.0,
        responses=responses,
    )


def _compute_qc(
    *,
    record: dict[str, str],
    columns: list[str],
    candidate_regex: str | None,
) -> tuple[list[str], int, float]:
    q_columns = [col for col in columns if _QC_Q_COL.match(col or "")]
    nr_count = 0
    for col in q_columns:
        if (record.get(col) or "").strip() == "NR":
            nr_count += 1
    q_total = max(1, len(q_columns))
    nr_percent = nr_count / q_total

    flags: list[str] = []
    if nr_percent >= 0.70:
        flags.append("HIGH_NR")

    candidate_column_present = False
    candidate_value = ""
    for key in ("CandidateNumber", "candidateNumber", "candidate_number"):
        if key in record:
            candidate_column_present = True
            candidate_value = (record.get(key) or "").strip()
            break

    should_validate_candidate = candidate_column_present or candidate_regex is not None
    if should_validate_candidate and candidate_value == "":
        flags.append("BAD_CANDIDATE")
    elif should_validate_candidate and candidate_regex:
        try:
            if re.fullmatch(candidate_regex, candidate_value) is None:
                flags.append("BAD_CANDIDATE")
        except re.error:
            flags.append("BAD_CANDIDATE")

    return flags, int(nr_count), float(nr_percent)


def read_results(
    batch_id: str,
    settings: Settings | None = None,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> ResultsPayload:
    """Parse the latest ``Results_*.csv`` for a batch, if any."""
    settings = settings or get_settings()
    offset = max(0, int(offset))
    if limit is not None:
        limit = max(0, int(limit))
    csv_path = batches_service.find_results_csv(batch_id, settings)
    error_csvs = _find_error_files_csvs(batch_id, settings)
    if csv_path is None and not error_csvs:
        return ResultsPayload(
            batch_id=batch_id,
            columns=[],
            rows=[],
            generated_csv=None,
            total_rows=0,
            offset=offset,
            limit=limit,
            truncated=False,
        )

    header: list[str] = []
    rows: list[ResultsRow] = []
    total_rows = 0
    seen_error_file_ids: set[str] = set()
    config_doc = batches_service.get_json_document(batch_id, "config", settings) or {}
    candidate_regex = None
    if isinstance(config_doc, dict):
        outputs = config_doc.get("outputs")
        if isinstance(outputs, dict):
            regex_value = outputs.get("candidate_regex")
            if isinstance(regex_value, str) and regex_value.strip():
                candidate_regex = regex_value.strip()

    regular_total = 0
    if csv_path is not None:
        header, records, total_rows = _read_csv_records(
            csv_path, offset=offset, limit=limit
        )
        regular_total = total_rows
        for record in records:
            row = _result_row_from_record(record)
            flags, nr_count, nr_percent = _compute_qc(
                record=record, columns=header, candidate_regex=candidate_regex
            )
            row.qc_flags = flags
            row.nr_count = nr_count
            row.nr_percent = nr_percent
            rows.append(row)

    error_offset = max(0, offset - regular_total)
    remaining_limit = None if limit is None else max(0, limit - len(rows))
    for error_csv in error_csvs:
        error_header, error_records, error_total = _read_csv_records(
            error_csv,
            offset=error_offset,
            limit=remaining_limit,
        )
        total_rows += error_total
        error_offset = max(0, error_offset - error_total)
        if remaining_limit is not None:
            remaining_limit = max(0, remaining_limit - len(error_records))
        if not header:
            header = error_header
        for record in error_records:
            file_id = record.get("file_id") or ""
            if file_id in seen_error_file_ids:
                continue
            seen_error_file_ids.add(file_id)
            row = _result_row_from_record(record, status="failed")
            flags, nr_count, nr_percent = _compute_qc(
                record=record, columns=header, candidate_regex=candidate_regex
            )
            row.qc_flags = flags
            row.nr_count = nr_count
            row.nr_percent = nr_percent
            rows.append(row)

    return ResultsPayload(
        batch_id=batch_id,
        columns=header,
        rows=rows,
        generated_csv=csv_path.as_posix() if csv_path is not None else None,
        total_rows=total_rows,
        offset=offset,
        limit=limit,
        truncated=limit is not None and offset + len(rows) < total_rows,
    )


def results_csv_path(batch_id: str, settings: Settings | None = None) -> Path | None:
    """Return the raw path to the latest Results CSV (if any)."""
    settings = settings or get_settings()
    return batches_service.find_results_csv(batch_id, settings)
