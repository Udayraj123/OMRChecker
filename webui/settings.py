"""Runtime configuration for the OMRChecker Web UI.

Values can be overridden via three layered sources (later wins):

1. Built-in field defaults (this file).
2. Environment variables (prefix ``OMR_WEBUI_``) or ``.env`` at repo root.
3. A JSON overrides file at ``cache_root / settings_overrides.json`` that
   the ``/settings`` page writes to. This survives restarts without
   requiring users to touch environment variables.

The runtime overrides file is loaded eagerly in :func:`get_settings`. The
``/api/v1/settings`` endpoints read/write it and call
:func:`reload_settings` so the next ``get_settings()`` call picks up the
new values without a process restart.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

# Settings safe to mutate at runtime via the /settings page. Anything not
# in this set requires a process restart (storage_root, cache_root, etc.)
# because services snapshot it at startup or in long-running tasks.
RUNTIME_MUTABLE_SETTINGS: frozenset[str] = frozenset({
    "pipeline_omr_with_split",
    "auto_start_omr_with_split",
    "auto_start_omr_min_pages",
    "auto_start_omr_require_config",
    "pdf_render_dpi",
    "pdf_render_grayscale",
    "pdf_page_format",
    "pdf_jpeg_quality",
    "inmemory_pipeline",
    "pdf_split_workers",
    "pdf_split_min_pages_for_parallel",
    "default_preset",
    "allow_directory_import",
})

OVERRIDES_FILENAME = "settings_overrides.json"


def _default_cache_root() -> Path:
    """Return an OS-appropriate, AV-friendly default scratch directory.

    On Windows we prefer ``%LOCALAPPDATA%\\OMRChecker\\cache`` so that an
    administrator can add a *single* Defender exclusion that covers every
    transient runtime/rotated/worker artifact the pipeline creates. We avoid
    ``%TEMP%`` (Defender scans it aggressively) and the repo directory
    (developer Documents folders are typically scanned too).
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "OMRChecker" / "cache"
    return Path.home() / ".cache" / "omrchecker"


class Settings(BaseSettings):
    """Application settings for the web UI."""

    model_config = SettingsConfigDict(
        env_prefix="OMR_WEBUI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    storage_root: Path = Field(
        default=REPO_ROOT / "webui" / "storage" / "batches",
        description="Directory where each batch's files and outputs live.",
    )

    allow_directory_import: bool = Field(
        default=False,
        description=(
            "When true, /api/v1/batches/{id}/files/import can read from an "
            "arbitrary server-side directory. Turn off in hosted deployments."
        ),
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins. Use an explicit list in production.",
    )

    max_upload_bytes: int = Field(
        default=1024 * 1024 * 1024,
        description="Per-file upload limit in bytes (default 1 GiB).",
    )

    pdf_render_dpi: int = Field(
        default=150,
        description=(
            "DPI used when rasterising PDF pages to PNG for OMR processing. "
            "150 DPI gives 44 %% less RAM/disk per page than 200 DPI while "
            "keeping ArUco markers well above the detection threshold. "
            "Override with OMR_WEBUI_PDF_RENDER_DPI."
        ),
    )

    pdf_render_grayscale: bool = Field(
        default=True,
        description=(
            "When true, PDF pages are rasterised as single-channel grayscale "
            "(1 byte/px) instead of RGB (3 bytes/px). The OMR engine reads "
            "all inputs as IMREAD_GRAYSCALE so colour information is discarded "
            "on read anyway. Disabling gives no quality benefit for OMR sheets "
            "but triples peak RAM. Override with OMR_WEBUI_PDF_RENDER_GRAYSCALE."
        ),
    )

    pdf_page_format: str = Field(
        default="jpeg",
        description=(
            "Image format for split PDF pages: 'jpeg' (default) or 'png'. "
            "Grayscale JPEG at quality 92 is ~5x smaller than PNG and visually "
            "lossless for OMR bubble detection, which dramatically reduces "
            "Windows Defender scan time on 1000+ page batches. "
            "Override with OMR_WEBUI_PDF_PAGE_FORMAT."
        ),
    )

    pdf_jpeg_quality: int = Field(
        default=92,
        ge=60,
        le=100,
        description=(
            "JPEG quality (60-100) used when pdf_page_format='jpeg'. 92 is "
            "visually lossless for scanned OMR sheets. Lower values shrink "
            "files further but risk JPEG artifacts on bubble edges. "
            "Override with OMR_WEBUI_PDF_JPEG_QUALITY."
        ),
    )

    cache_root: Path = Field(
        default_factory=_default_cache_root,
        description=(
            "Root directory for transient OMR scratch space: per-worker "
            "runtime dirs, rotated/resized image cache, and per-image worker "
            "outputs. Defaults to %LOCALAPPDATA%\\OMRChecker\\cache on "
            "Windows (single-path Defender exclusion friendly) and "
            "~/.cache/omrchecker on POSIX. Override with OMR_WEBUI_CACHE_ROOT."
        ),
    )

    inmemory_pipeline: bool = Field(
        default=True,
        description=(
            "When true (default), the OMR worker decodes images directly into "
            "memory and bypasses the per-image runtime directory entirely. "
            "This is the single biggest Defender mitigation: zero "
            "intermediate file writes per processed page. Set to false to "
            "fall back to the legacy directory-staged engine flow (useful "
            "for debugging or if a template references custom asset files). "
            "Override with OMR_WEBUI_INMEMORY_PIPELINE."
        ),
    )

    pipeline_omr_with_split: bool = Field(
        default=True,
        description=(
            "When true, OMR processing overlaps with in-flight PDF splits: "
            "pages produced by background split tasks are enqueued for OMR "
            "as they arrive instead of waiting for the entire split to "
            "finish. Set to false to restore the legacy single-snapshot "
            "discovery behaviour. Override with OMR_WEBUI_PIPELINE_OMR_WITH_SPLIT."
        ),
    )

    auto_start_omr_with_split: bool = Field(
        default=True,
        description=(
            "When true, the frontend automatically POSTs /process as soon as "
            "the configured minimum number of split pages exist AND the batch "
            "has a template + (optionally) config uploaded. This is what "
            "actually makes 'Pipelined OMR' visible end-to-end: pages start "
            "being scored while later pages are still being rendered. Set "
            "false to require the operator to click Run OMR manually. "
            "Override with OMR_WEBUI_AUTO_START_OMR_WITH_SPLIT."
        ),
    )

    auto_start_omr_min_pages: int = Field(
        default=10,
        ge=1,
        le=10_000,
        description=(
            "Minimum number of pages that must exist before auto-start "
            "fires. Set higher (e.g. 50) to amortise process-pool spawn "
            "cost over more pages; set to 1 for the earliest possible "
            "overlap. Ignored unless auto_start_omr_with_split is true. "
            "Override with OMR_WEBUI_AUTO_START_OMR_MIN_PAGES."
        ),
    )

    auto_start_omr_require_config: bool = Field(
        default=False,
        description=(
            "When true, auto-start also requires a config.json to be present "
            "(not just template.json). Most templates work without an "
            "explicit config, so the default is false. Override with "
            "OMR_WEBUI_AUTO_START_OMR_REQUIRE_CONFIG."
        ),
    )

    pdf_split_workers: int = Field(
        default=0,
        ge=0,
        le=32,
        description=(
            "Number of worker processes used to render PDF pages in parallel. "
            "0 (default) means auto: min(8, max(1, cpu_count // 2)). "
            "Render + JPEG encode is CPU-bound, so this typically gives a "
            "near-linear speedup up to the auto-cap. Set to 1 to force the "
            "legacy single-threaded loop (useful for debugging). "
            "Override with OMR_WEBUI_PDF_SPLIT_WORKERS."
        ),
    )

    pdf_split_min_pages_for_parallel: int = Field(
        default=16,
        ge=1,
        description=(
            "PDFs with fewer than this many pages always use the serial "
            "render path; process-pool spawn overhead dominates below this "
            "threshold. Override with OMR_WEBUI_PDF_SPLIT_MIN_PAGES_FOR_PARALLEL."
        ),
    )

    presets_dir: Path = Field(
        default=REPO_ROOT,
        description=(
            "Directory scanned for preset subdirectories. Each direct subdir "
            "that contains a template.json is treated as a named preset."
        ),
    )

    default_preset: str | None = Field(
        default="custom_25_definitive_final",
        description=(
            "Preset applied automatically when a new batch is created. "
            "Set to null or empty string to disable auto-apply."
        ),
    )

    def ensure_storage(self) -> Path:
        """Create and return the batches root directory."""
        self.storage_root.mkdir(parents=True, exist_ok=True)
        return self.storage_root

    def ensure_cache_root(self) -> Path:
        """Create and return the runtime/scratch cache root directory."""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self.cache_root

    def batch_cache_dir(self, batch_id: str) -> Path:
        """Return the per-batch scratch directory under the cache root."""
        path = self.ensure_cache_root() / batch_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def overrides_path(self) -> Path:
        """Return the path to the runtime overrides JSON file.

        The file lives next to ``storage_root`` (one level up, alongside
        the per-batch directories) rather than under ``cache_root``.
        Two reasons:

        1. Some Windows AV products (e.g. Symantec Endpoint Protection)
           aggressively quarantine newly-written small JSON files under
           ``%LOCALAPPDATA%`` via the ``.tmp`` -> atomic-rename pattern,
           which is what ``cache_root`` defaults to. ``storage_root``
           is the user's project directory which is typically already
           whitelisted by IT or simpler to add an exclusion for.
        2. Overrides aren't transient — they encode user-visible
           configuration that should outlive a cache wipe.
        """
        return self.storage_root.parent / OVERRIDES_FILENAME


def _resolve_overrides_path(root: Path) -> Path:
    """Map a passed root (cache_root or storage_root) to its overrides file.

    For backwards compatibility callers may pass either ``cache_root``
    (legacy: ``<cache_root>/settings_overrides.json``) or
    ``storage_root`` (current: ``<storage_root>.parent/settings_overrides.json``).
    We pick the location based on which one matches the live ``Settings``
    instance, falling back to ``<root>/OVERRIDES_FILENAME`` for unknown
    roots so direct unit tests keep working.
    """
    # The current convention is storage_root.parent. Tests that previously
    # passed cache_root land in the fallback branch below.
    try:
        live = get_settings()
        if root == live.storage_root or root == live.storage_root.parent:
            return live.overrides_path()
        if root == live.cache_root:
            # Legacy caller asked for cache_root location; respect it
            # but also check the new location so reads find a file
            # written by a newer caller.
            new_loc = live.overrides_path()
            if new_loc.exists():
                return new_loc
            return root / OVERRIDES_FILENAME
    except Exception:  # noqa: BLE001 — be defensive during init/teardown
        pass
    return root / OVERRIDES_FILENAME


def _load_overrides(root: Path) -> dict[str, Any]:
    """Read the runtime overrides JSON file if present.

    *root* is either ``storage_root`` (preferred) or ``cache_root`` (legacy);
    the function resolves the actual file path via
    :func:`_resolve_overrides_path` so callers can stay agnostic.

    Returns an empty dict if the file is missing, unreadable, or contains
    invalid JSON. Never raises — a corrupt overrides file must not
    prevent the server from starting.
    """
    path = _resolve_overrides_path(root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Ignoring corrupt settings overrides at %s: %s", path, exc
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Ignoring settings overrides at %s: top-level is %s, expected object",
            path, type(data).__name__,
        )
        return {}
    # Drop any keys that are not in the runtime-mutable allowlist to keep
    # the file forward-compatible (older overrides for removed settings
    # are silently ignored).
    return {k: v for k, v in data.items() if k in RUNTIME_MUTABLE_SETTINGS}


def write_overrides(overrides: dict[str, Any], root: Path) -> Path:
    """Persist *overrides* to the settings overrides JSON file.

    Only keys in :data:`RUNTIME_MUTABLE_SETTINGS` are written; everything
    else is dropped silently. Writes atomically via a ``.tmp`` swap so a
    crash mid-write can't leave a half-finished JSON file behind.

    *root* may be either ``cache_root`` (legacy) or ``storage_root``
    (current). The resolved file location is returned so the caller
    can log it or hand it to the operator on failure.
    """
    target = _resolve_overrides_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    filtered = {k: v for k, v in overrides.items() if k in RUNTIME_MUTABLE_SETTINGS}
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(filtered, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def reload_settings() -> "Settings":
    """Clear the memoised settings so the next ``get_settings()`` rebuilds.

    Call this after writing to the overrides file so subsequent requests
    pick up the new values without a process restart.
    """
    get_settings.cache_clear()
    return get_settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a memoised ``Settings`` instance.

    Construction order:

    1. Pydantic loads env vars + ``.env`` defaults.
    2. We then layer the JSON overrides file on top via a fresh
       ``model_copy(update=...)`` so the resulting instance reflects every
       layered source.

    The overrides file location is resolved via :meth:`Settings.overrides_path`
    (currently ``storage_root.parent/settings_overrides.json``). A legacy
    location at ``cache_root/settings_overrides.json`` is also checked
    so older deployments keep working through one restart.
    """
    base = Settings()
    overrides_path = base.overrides_path()
    if overrides_path.exists():
        try:
            data = json.loads(overrides_path.read_text(encoding="utf-8"))
            overrides = {k: v for k, v in data.items() if k in RUNTIME_MUTABLE_SETTINGS}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Ignoring corrupt settings overrides at %s: %s",
                overrides_path, exc,
            )
            overrides = {}
    else:
        # Legacy fallback: check the old cache_root location.
        legacy_path = base.cache_root / OVERRIDES_FILENAME
        if legacy_path.exists():
            try:
                data = json.loads(legacy_path.read_text(encoding="utf-8"))
                overrides = {
                    k: v for k, v in data.items() if k in RUNTIME_MUTABLE_SETTINGS
                }
                logger.info(
                    "Found legacy settings overrides at %s; the next save "
                    "will migrate them to %s",
                    legacy_path, overrides_path,
                )
            except (json.JSONDecodeError, OSError):
                overrides = {}
        else:
            overrides = {}

    if not overrides:
        return base
    try:
        return base.model_copy(update=overrides)
    except Exception as exc:  # noqa: BLE001 — defensive: never let overrides break startup
        logger.warning(
            "Ignoring %d settings overrides because model_copy failed: %s",
            len(overrides), exc,
        )
        return base
