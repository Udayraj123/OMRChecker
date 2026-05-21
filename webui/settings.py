"""Runtime configuration for the OMRChecker Web UI.

Values can be overridden via environment variables (prefix ``OMR_WEBUI_``)
or a ``.env`` file at the repo root. The defaults are chosen to match the
plan's local-first posture.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a memoised ``Settings`` instance."""
    return Settings()
