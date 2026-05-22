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
from typing import Any

import cv2
from PIL import Image as _PILImage

from src.defaults.config import CONFIG_DEFAULTS
from webui.schemas import (
    BatchStatus,
    ResultsPayload,
    ResultsRow,
)
from webui.services import batches as batches_service
from webui.services.capacity import reserve_worker_capacity, worker_capacity_limit
from webui.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_COPIED_JSON_FILES = ("template.json", "config.json", "evaluation.json")
_RUNTIME_DIR_NAME = "_runtime"
_RUNTIME_BASE_DIR_NAME = "_base"
_DISPLAY_KEYS = ("display_height", "display_width", "processing_height", "processing_width")


def _batch_scratch_root(batch_root: Path, settings: Settings | None = None) -> Path:
    """Return the AV-friendly scratch root for this batch.

    By default this is ``%LOCALAPPDATA%\\OMRChecker\\cache\\<batch_id>\\`` on
    Windows (a single administrator-excludable directory) and
    ``~/.cache/omrchecker/<batch_id>/`` on POSIX. Falls back to the legacy
    ``<batch_root>`` location if settings are unavailable so this function
    can be called from contexts (e.g. worker subprocesses) that may not
    have an initialised Settings instance.
    """
    if settings is None:
        try:
            settings = get_settings()
        except Exception:  # noqa: BLE001 - never let cache-dir logic break OMR
            return batch_root
    try:
        return settings.batch_cache_dir(batch_root.name)
    except Exception:  # noqa: BLE001
        return batch_root


def _batch_runtime_root(batch_root: Path, settings: Settings | None = None) -> Path:
    return _batch_scratch_root(batch_root, settings) / _RUNTIME_DIR_NAME


def _batch_rotated_root(batch_root: Path, settings: Settings | None = None) -> Path:
    return _batch_scratch_root(batch_root, settings) / "_rotated"


def _batch_workers_root(outputs_dir: Path, settings: Settings | None = None) -> Path:
    """Per-image worker output dirs go under the scratch root, not outputs/.

    Worker outputs are transient: the parent process aggregates the rows
    and copies the kept annotated images back into ``outputs/CheckedOMRs/``,
    then the worker dir is deleted. Putting them under the scratch root
    keeps the real ``outputs/`` directory clean of intermediate files
    that Defender would otherwise scan.
    """
    batch_root = outputs_dir.parent
    return _batch_scratch_root(batch_root, settings) / "_workers"

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


def _filter_stable_input_images(paths: list[Path]) -> list[Path]:
    """Return only newly discovered pages whose size is stable across a tick."""
    if not paths:
        return []
    first_sizes: dict[Path, int] = {}
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 0:
            first_sizes[path] = size
    if not first_sizes:
        return []
    time.sleep(0.05)
    ready: list[Path] = []
    for path, first_size in first_sizes.items():
        try:
            second_size = path.stat().st_size
        except OSError:
            continue
        if second_size == first_size and second_size > 0:
            ready.append(path)
    return ready


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


def _prepare_runtime_base(batch_root: Path, settings: Settings | None = None) -> Path:
    """Build the shared ``_base`` runtime dir containing template + assets.

    Lives under the scratch cache root (e.g. ``%LOCALAPPDATA%\\OMRChecker``)
    so a single Defender exclusion covers it.
    """
    runtime_root = _batch_runtime_root(batch_root, settings)
    base_root = runtime_root / _RUNTIME_BASE_DIR_NAME
    if base_root.exists():
        shutil.rmtree(base_root, ignore_errors=True)
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
    rotated_dir: Path | None = None,
) -> None:
    """Copy *src* into *dst*, applying rotation and optional downscale.

    ``pre_resize_to`` is ``(width, height)`` and is only applied when the
    operation would make the image *smaller* (never upscales).

    When neither rotation nor a meaningful resize is needed, no decoded
    intermediate is written: a single copy is enough. This avoids one
    Defender-scanned write per image.
    """
    # Decide up front whether we actually need to do pixel work. If the
    # source is already small enough and no rotation is requested, the
    # OMR engine can read the source bytes directly via a copy — no
    # OpenCV decode, no extra Defender scan, no _rotated cache entry.
    needs_rotate = rotation_degrees != 0
    needs_resize = False
    if pre_resize_to is not None:
        try:
            with _PILImage.open(src) as _pil:
                _src_w, _src_h = _pil.size
        except Exception:  # noqa: BLE001
            _src_w = _src_h = None
        if _src_w is not None and _src_h is not None:
            proc_w, proc_h = pre_resize_to
            if _src_w > proc_w or _src_h > proc_h:
                needs_resize = True
    if not needs_rotate and not needs_resize:
        shutil.copy2(src, dst)
        return

    # Build a cache key that encodes both rotation and resize so the cached
    # file stays valid across different processing-dimension configs.
    resize_tag = f"_s{pre_resize_to[0]}x{pre_resize_to[1]}" if pre_resize_to else ""
    rot_tag = f"_rot{rotation_degrees}" if rotation_degrees else ""
    cache_key = f"{src.stem}{rot_tag}{resize_tag}{src.suffix.lower()}"

    if rotated_dir is None:
        # Legacy callers (kept for backward compatibility in tests) place the
        # cache next to the runtime dir, which lives under the scratch root.
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

    if needs_rotate:
        rotate_codes = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        rotate_code = rotate_codes.get(rotation_degrees)
        if rotate_code is None:
            raise ValueError(f"Unsupported rotation: {rotation_degrees}")
        image = cv2.rotate(image, rotate_code)

    if needs_resize and pre_resize_to is not None:
        proc_w, proc_h = pre_resize_to
        h, w = image.shape[:2]
        if w > proc_w or h > proc_h:
            image = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    if not cv2.imwrite(str(dst), image):
        raise ValueError(f"Could not write runtime image: {dst.as_posix()}")
    try:
        cv2.imwrite(str(cached), image)
    except Exception:  # noqa: BLE001
        pass


def _worker_runtime_dir(
    batch_root: Path,
    settings: Settings | None = None,
) -> Path:
    """Return the per-worker reusable runtime directory.

    Long-lived ``ProcessPoolExecutor`` workers were creating a fresh
    ``_runtime/<index>_<stem>/`` directory per image and ``rmtree``-ing it
    afterwards. At 5000 pages that is ~10,000 NTFS metadata ops + Defender
    scans of the staged image, template, evaluation, and config files
    inside each dir. Reusing a single dir per worker process and just
    overwriting its image + config on each task drops that to N (number
    of workers) directory ops total.
    """
    pid = os.getpid()
    return _batch_runtime_root(batch_root, settings) / f"worker_{pid}"


def _prepare_runtime_dir(
    batch_root: Path,
    image_path: Path,
    runtime_config: dict,
    index: int,
    rotation_degrees: int = 0,
    base_root: Path | None = None,
    pre_resize_to: tuple[int, int] | None = None,
    settings: Settings | None = None,
) -> Path:
    """Stage a single-image runtime directory for engine execution.

    The directory is **per worker process** and is reused across every image
    that worker processes. We clear the previous image + per-image config
    each time but keep the (read-only) template.json / evaluation.json /
    assets in place.
    """
    runtime_root = _worker_runtime_dir(batch_root, settings)
    runtime_root.mkdir(parents=True, exist_ok=True)

    if base_root is None:
        base_root = _prepare_runtime_base(batch_root, settings)
    # Ensure template.json / evaluation.json / assets exist in the worker
    # dir. We hardlink-or-copy them once (the file already existing is the
    # common path on the worker's 2nd..Nth task).
    for name in ("template.json", "evaluation.json"):
        src = base_root / name
        dst = runtime_root / name
        if src.exists() and not dst.exists():
            _link_or_copy(src, dst)
    for path in base_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {"template.json", "evaluation.json"}:
            continue
        rel = path.relative_to(base_root)
        dst = runtime_root / rel
        if not dst.exists():
            _link_or_copy(path, dst)

    # Remove any image left by the previous task so the engine's directory
    # scan only finds the current target.
    for child in runtime_root.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
            # Don't delete template-referenced assets (those are tracked
            # under base_root and re-linked above on the next iteration).
            relative_to_base = base_root / child.name
            if not relative_to_base.exists():
                try:
                    child.unlink()
                except OSError:
                    pass

    rotated_dir = _batch_rotated_root(batch_root, settings)
    _rotate_image_for_runtime(
        image_path,
        runtime_root / image_path.name,
        rotation_degrees,
        pre_resize_to=pre_resize_to,
        rotated_dir=rotated_dir,
    )
    _write_runtime_config(runtime_config, runtime_root / "config.json")
    return runtime_root


def _cleanup_runtime_dir(runtime_dir: Path | None) -> None:
    """Best-effort cleanup of the **per-image** files inside a runtime dir.

    The per-worker runtime dir itself is intentionally *not* removed: it
    will be reused for the next task on the same worker. We only clear
    image + per-task config to drop file-system pressure between tasks.
    """
    if not runtime_dir or not runtime_dir.exists():
        return
    for child in runtime_dir.iterdir():
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
            try:
                child.unlink()
            except OSError:
                pass


def _default_max_workers() -> int:
    capacity = worker_capacity_limit()
    # Allow explicit override via environment variable.
    env_val = os.environ.get("OMR_WEBUI_MAX_WORKERS")
    if env_val:
        try:
            requested = int(env_val)
            if requested > capacity:
                logger.warning(
                    "Clamping OMR_WEBUI_MAX_WORKERS from %d to worker capacity %d",
                    requested, capacity,
                )
            return max(1, min(32, requested, capacity))
        except (TypeError, ValueError):
            pass
    cpu = os.cpu_count() or 2
    # Leave 1 core free for the main process; cap at 16 for modern many-core
    # machines. Override via max_workers in config.json (accepts 1-32).
    return max(1, min(cpu - 1, 16, capacity))


def _coerce_max_workers(value: Any) -> int:
    capacity = worker_capacity_limit()
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _default_max_workers()
    if parsed > capacity:
        logger.warning(
            "Clamping configured max_workers from %d to worker capacity %d",
            parsed, capacity,
        )
    return max(1, min(32, parsed, capacity))


def _process_one_image(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint to process a single image.

    Uses a *per-worker* reusable runtime directory under the configurable
    scratch cache root (default ``%LOCALAPPDATA%\\OMRChecker\\cache``).
    Source images are hardlinked (not copied) into the runtime dir when
    no rotation or resize is required, eliminating one Defender-scanned
    write per processed page.

    With ``OMR_WEBUI_INMEMORY_PIPELINE=true`` (default) the worker also
    invokes the in-memory engine path that points the OMR engine directly
    at the source image instead of round-tripping through a staged copy.
    """
    cv2.setNumThreads(1)

    batch_root = Path(payload["batch_root"])
    outputs_dir = Path(payload["outputs_dir"])
    image_path = Path(payload["image_path"])
    base_config = payload.get("base_config") or {}
    template_payload = payload.get("template_payload") or {}
    base_root = Path(payload["base_root"])
    rotation_degrees = int(payload.get("rotation_degrees", 0))
    index = int(payload.get("index", 1))
    use_inmemory = bool(payload.get("inmemory_pipeline", False))

    settings: Settings | None = None
    try:
        settings = get_settings()
    except Exception:  # noqa: BLE001
        settings = None

    runtime_dir: Path | None = None
    logger.debug(
        "Worker | index=%d | file=%s | inmemory=%s",
        index, image_path.name, use_inmemory,
    )
    t_start = time.perf_counter()
    try:
        if not base_root.exists():
            base_root = _prepare_runtime_base(batch_root, settings)
        dynamic_dimensions = _compute_dynamic_dimensions(
            image_path, template_payload, rotation_degrees
        )
        runtime_config = _merge_dimensions_into_config(base_config, dynamic_dimensions)

        # Pre-resize: if the source image is meaningfully larger than the
        # processing dimensions, shrink it on the way in so the engine's
        # own resize step becomes a near-no-op. In the in-memory path we
        # skip this (the engine resizes the array anyway, with no I/O cost).
        src_w = dynamic_dimensions["source_width"]
        src_h = dynamic_dimensions["source_height"]
        proc_w = dynamic_dimensions["processing_width"]
        proc_h = dynamic_dimensions["processing_height"]
        pre_resize_to: tuple[int, int] | None = None
        if not use_inmemory and (src_w > proc_w * 1.1 or src_h > proc_h * 1.1):
            pre_resize_to = (proc_w, proc_h)

        worker_outputs_root = _batch_workers_root(outputs_dir, settings)
        worker_outputs_dir = worker_outputs_root / f"{index:04d}_{image_path.stem}"
        worker_outputs_dir.mkdir(parents=True, exist_ok=True)

        if use_inmemory:
            # In-memory path: point the engine directly at the source image
            # — no per-image staging copy, no _rotated cache write, no
            # runtime config file edits. Rotation, if any, is applied
            # in-process via cv2.rotate on the numpy array.
            from src.entry import entry_point_for_image
            entry_point_for_image(
                image_path=image_path,
                output_dir=worker_outputs_dir,
                template_payload=template_payload,
                config_payload=runtime_config,
                evaluation_path=(base_root / "evaluation.json")
                    if (base_root / "evaluation.json").exists() else None,
                template_dir=base_root,
                rotation_degrees=rotation_degrees,
            )
        else:
            runtime_dir = _prepare_runtime_dir(
                batch_root,
                image_path,
                runtime_config,
                index,
                rotation_degrees,
                base_root=base_root,
                pre_resize_to=pre_resize_to,
                settings=settings,
            )
            from main import entry_point_for_args
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
    # Snapshot once so the flag stays consistent for the entire run even if
    # settings are somehow mutated mid-run (which they shouldn't be).
    _pipeline_enabled = bool(getattr(settings, "pipeline_omr_with_split", True))
    lock = _lock_for(batch_id)
    if not lock.acquire(blocking=False):
        logger.info("Batch %s is already processing; skipping duplicate run", batch_id)
        return

    runtime_dir: Path | None = None
    run_started_at = time.monotonic()  # initialised early so except block can always reference it
    try:
        batch_root = batches_service.get_batch_root(batch_id, settings)
        outputs_dir = batch_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
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
        initial_split_total = int(metadata.get("pdf_split_total", 0) or 0)
        capacity_limit = worker_capacity_limit()
        # Record whether this run actually engages pipelining so the UI/log can
        # confirm it.  Without this flag operators have no way to tell whether
        # the pipeline_omr_with_split feature did anything for a given batch.
        pipeline_active = False
        if _pipeline_enabled and initial_split_total > 0:
            if capacity_limit <= 1:
                logger.info(
                    "Disabling pipelined OMR because worker capacity is one | "
                    "batch_id=%s",
                    batch_id,
                )
                _pipeline_enabled = False
            else:
                pipeline_active = True
                if max_workers >= capacity_limit:
                    max_workers = max(1, capacity_limit - 1)
                    logger.info(
                        "OMR workers clamped to leave split capacity | batch_id=%s | "
                        "workers=%d | capacity=%d",
                        batch_id, max_workers, capacity_limit,
                    )

        base_root = _prepare_runtime_base(batch_root)
        if pipeline_active:
            logger.info(
                "OMR PIPELINED MODE engaged | batch_id=%s | initial_pages=%d | "
                "split_total=%d | workers=%d (capacity=%d). Newly-arrived pages "
                "will be picked up as the splitter publishes them.",
                batch_id, len(input_images), initial_split_total, max_workers, capacity_limit,
            )
        else:
            logger.info(
                "OMR SEQUENTIAL MODE | batch_id=%s | images=%d | workers=%d | "
                "pipeline_enabled=%s | initial_split_total=%d (no in-flight split to overlap with)",
                batch_id, len(input_images), max_workers, _pipeline_enabled, initial_split_total,
            )
        # Persist the flag so the /status endpoint can surface it to the UI.
        batches_service.update_batch_metadata(
            batch_id,
            {"pipelined_run": pipeline_active},
            settings,
        )

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

        # Snapshot once outside the closure so workers all see the same value
        # even if settings are mutated mid-run (which they shouldn't be).
        _inmemory = bool(getattr(settings, "inmemory_pipeline", True))

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
                "inmemory_pipeline": _inmemory,
            }

        futures = {}
        next_index = 1
        image_iter = iter(input_images)
        _executor_gone = False  # set when executor shuts down mid-run (e.g. server reload)
        # Track every path that has been submitted to the executor so that
        # the pipeline re-discovery block never submits the same image twice.
        submitted_paths: set[Path] = set()
        # True once the current image_iter has been fully consumed.
        _iter_exhausted = False
        # True once we are certain no more split pages are incoming.
        # When pipelining is disabled we treat the split as already done so
        # the loop degenerates to the legacy ``while futures:`` behaviour.
        _split_done = not _pipeline_enabled
        split_error_seen: str | None = None

        with reserve_worker_capacity(
            max_workers,
            label=f"omr:{batch_id}",
        ) as reserved_workers:
            if reserved_workers != max_workers:
                logger.info(
                    "OMR worker count reduced by shared capacity | batch_id=%s | "
                    "requested=%d | granted=%d",
                    batch_id, max_workers, reserved_workers,
                )
            max_workers = reserved_workers
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                while len(futures) < max_workers:
                    try:
                        image = next(image_iter)
                    except StopIteration:
                        _iter_exhausted = True
                        break
                    try:
                        futures[executor.submit(_process_one_image, submit_payload(next_index, image))] = image
                    except RuntimeError:
                        _executor_gone = True
                        break
                    submitted_paths.add(image)
                    next_index += 1

                last_milestone = 0
                completed = 0
                # Exit when (a) all submitted futures have completed AND
                # (b) there are no more split pages incoming.  When
                # pipelining is disabled _split_done starts True so the
                # condition reduces to the legacy ``while futures:``.
                while futures or (_pipeline_enabled and not _split_done):
                    if not futures:
                        # All known images processed but the PDF split is
                        # still running.  Pause briefly before re-polling
                        # to avoid a tight busy-loop; the next refill will
                        # submit newly-discovered pages if any have arrived.
                        time.sleep(0.5)

                    if futures:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    else:
                        done = set()

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
                        _mark_cancelled(
                            batch_id,
                            "Stop requested. Running images will finish first.",
                            settings,
                        )
                        return

                    if not _executor_gone:
                        while len(futures) < max_workers:
                            try:
                                image = next(image_iter)
                            except StopIteration:
                                _iter_exhausted = True
                                break
                            # Guard against double-submission; should not
                            # happen with a linear iterator, but provides
                            # safety during pipeline re-discovery.
                            if image in submitted_paths:
                                continue
                            try:
                                futures[executor.submit(_process_one_image, submit_payload(next_index, image))] = image
                            except RuntimeError:
                                _executor_gone = True
                                break
                            submitted_paths.add(image)
                            next_index += 1

                        # ── Pipeline re-discovery ──────────────────────────
                        # When OMR is pipelined with an in-flight PDF split,
                        # re-check for newly-arrived pages each time the
                        # current iterator runs dry.  Re-discovery is gated
                        # on a completed future (natural pacing; no extra
                        # timer), so it never becomes a busy-loop.
                        #
                        # Re-discovery always performs one final directory scan
                        # even after pdf_split_total has cleared. This catches
                        # pages published just before the splitter marked
                        # itself complete.
                        if _pipeline_enabled and _iter_exhausted and not _split_done:
                            _split_meta = batches_service.get_batch_metadata(batch_id, settings)
                            _split_error = _split_meta.get("pdf_split_error")
                            if _split_error:
                                split_error_seen = str(_split_error)
                                _split_done = True
                            else:
                                _all_discovered = _discover_input_images(batch_id, settings)
                                _candidate_new_paths = [
                                    img for img in _all_discovered
                                    if img not in submitted_paths
                                ]
                                _new_paths = _filter_stable_input_images(_candidate_new_paths)
                                if _new_paths:
                                    # Extend tracking list so progress
                                    # calculations reflect the growing total
                                    _prev_total = len(input_images)
                                    input_images.extend(_new_paths)
                                    _new_total = len(input_images)
                                    # Re-anchor milestone tracking to the new
                                    # denominator.  Without this, items
                                    # completed under the old (smaller)
                                    # total push ``last_milestone`` ahead of
                                    # the current pct against the new
                                    # (larger) total — suppressing all
                                    # subsequent OMR progress logs until
                                    # ``completed`` catches up to the old
                                    # absolute milestone count.  Reset to
                                    # the current pct floored to the
                                    # nearest 10% so the very next 10%
                                    # boundary fires correctly.
                                    _current_pct = (
                                        completed * 100 // _new_total
                                        if _new_total else 0
                                    )
                                    last_milestone = (_current_pct // 10) * 10
                                    logger.info(
                                        "OMR input pool grew | batch_id=%s | "
                                        "previous_total=%d | new_total=%d | "
                                        "added=%d | completed=%d (%d%% of new total)",
                                        batch_id, _prev_total, _new_total,
                                        len(_new_paths), completed, _current_pct,
                                    )
                                    image_iter = iter(_new_paths)
                                    _iter_exhausted = False
                                elif int(_split_meta.get("pdf_split_total", 0) or 0) == 0:
                                    _split_done = True

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

        shutil.rmtree(outputs_dir / "_workers", ignore_errors=True)

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

        if split_error_seen:
            batches_service.update_status(
                batch_id,
                BatchStatus.failed,
                last_error=(
                    "PDF splitting failed while OMR was running. "
                    f"Processed {completed} file(s) before stopping. "
                    f"Split error: {split_error_seen}"
                ),
                settings=settings,
            )
        elif preprocess_failures and len(preprocess_failures) == len(input_images):
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


def recover_stale_batches(settings: Settings | None = None) -> list[str]:
    """Handle batches orphaned by a previous server crash.

    - ``queued`` batches are re-enqueued and returned so the caller can
      schedule them as background tasks.
    - ``running`` batches are marked ``failed`` with a restart message so
      the user knows the run was interrupted.

    Returns the list of batch IDs that need to be run as background tasks.
    """
    settings = settings or get_settings()
    to_requeue: list[str] = []
    for batch in batches_service.list_batches(settings):
        if batch.status == BatchStatus.queued:
            to_requeue.append(batch.id)
        elif batch.status == BatchStatus.running:
            batches_service.update_status(
                batch.id,
                BatchStatus.failed,
                last_error="Interrupted by server restart — please restart the batch.",
                settings=settings,
            )
    return to_requeue


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


def _read_csv_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            raw_rows = list(reader)
    except (FileNotFoundError, OSError):
        return [], []

    if not raw_rows:
        return [], []

    header, *data_rows = raw_rows
    records = [
        dict(zip(header, row))
        for row in data_rows
        if row and any(str(value).strip() for value in row)
    ]
    return header, records


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


def _result_row_from_record(record: dict[str, str], status: str = "ok") -> ResultsRow:
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


def read_results(batch_id: str, settings: Settings | None = None) -> ResultsPayload:
    """Parse the latest ``Results_*.csv`` for a batch, if any."""
    settings = settings or get_settings()
    csv_path = batches_service.find_results_csv(batch_id, settings)
    error_csvs = _find_error_files_csvs(batch_id, settings)
    if csv_path is None and not error_csvs:
        return ResultsPayload(
            batch_id=batch_id, columns=[], rows=[], generated_csv=None
        )

    header: list[str] = []
    rows: list[ResultsRow] = []
    seen_error_file_ids: set[str] = set()
    config_doc = batches_service.get_json_document(batch_id, "config", settings) or {}
    candidate_regex = None
    if isinstance(config_doc, dict):
        outputs = config_doc.get("outputs")
        if isinstance(outputs, dict):
            regex_value = outputs.get("candidate_regex")
            if isinstance(regex_value, str) and regex_value.strip():
                candidate_regex = regex_value.strip()

    if csv_path is not None:
        header, records = _read_csv_records(csv_path)
        for record in records:
            row = _result_row_from_record(record)
            flags, nr_count, nr_percent = _compute_qc(
                record=record, columns=header, candidate_regex=candidate_regex
            )
            row.qc_flags = flags
            row.nr_count = nr_count
            row.nr_percent = nr_percent
            rows.append(row)

    for error_csv in error_csvs:
        error_header, error_records = _read_csv_records(error_csv)
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
    )


def results_csv_path(batch_id: str, settings: Settings | None = None) -> Path | None:
    """Return the raw path to the latest Results CSV (if any)."""
    settings = settings or get_settings()
    return batches_service.find_results_csv(batch_id, settings)
