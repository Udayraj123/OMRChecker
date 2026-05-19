"""Runtime configuration for the OMRChecker Web UI.

Values can be overridden via environment variables (prefix ``OMR_WEBUI_``)
or a ``.env`` file at the repo root. The defaults are chosen to match the
plan's local-first posture.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a memoised ``Settings`` instance."""
    return Settings()
