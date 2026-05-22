"""Pydantic schemas for the runtime ``/api/v1/settings`` endpoints.

Kept deliberately small and focused so the response/request models stay
in lockstep with :data:`webui.settings.RUNTIME_MUTABLE_SETTINGS`. Adding
or removing a runtime-mutable setting requires editing both this file
and the allowlist in ``webui/settings.py``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import PydanticUndefined

from webui.settings import RUNTIME_MUTABLE_SETTINGS, Settings

# Allowed values for ``pdf_page_format``. Kept as a module-level alias so
# both the response and update models share the exact same enum.
PdfPageFormat = Literal["jpeg", "png"]


class RuntimeSettingsResponse(BaseModel):
    """Snapshot of every runtime-mutable setting.

    Returned by ``GET /api/v1/settings`` and by ``PUT /api/v1/settings``
    after the override file has been rewritten and the settings reloaded.
    Types mirror :class:`webui.settings.Settings`.
    """

    model_config = ConfigDict(extra="forbid")

    pipeline_omr_with_split: bool
    auto_start_omr_with_split: bool
    auto_start_omr_min_pages: int = Field(ge=1, le=10_000)
    auto_start_omr_require_config: bool
    pdf_render_dpi: int
    pdf_render_grayscale: bool
    pdf_page_format: PdfPageFormat
    pdf_jpeg_quality: int = Field(ge=60, le=100)
    inmemory_pipeline: bool
    pdf_split_workers: int = Field(ge=0, le=32)
    pdf_split_min_pages_for_parallel: int = Field(ge=1)
    default_preset: Optional[str] = None
    allow_directory_import: bool


class RuntimeSettingsUpdate(BaseModel):
    """Partial update body for ``PUT /api/v1/settings``.

    Every field is ``Optional`` so callers may PATCH a single value
    without re-sending the whole object. ``extra="forbid"`` ensures that
    unknown keys (e.g. typos, or attempts to mutate non-allowlisted
    settings such as ``storage_root``) return ``422 Unprocessable
    Entity`` rather than being silently dropped.

    Field-level validators mirror the ``Settings`` constraints so a bad
    value is rejected at the API boundary, not deep inside ``Settings``.
    """

    model_config = ConfigDict(extra="forbid")

    pipeline_omr_with_split: Optional[bool] = None
    auto_start_omr_with_split: Optional[bool] = None
    auto_start_omr_min_pages: Optional[int] = Field(default=None, ge=1, le=10_000)
    auto_start_omr_require_config: Optional[bool] = None
    pdf_render_dpi: Optional[int] = Field(default=None, ge=50, le=600)
    pdf_render_grayscale: Optional[bool] = None
    pdf_page_format: Optional[PdfPageFormat] = None
    pdf_jpeg_quality: Optional[int] = Field(default=None, ge=60, le=100)
    inmemory_pipeline: Optional[bool] = None
    pdf_split_workers: Optional[int] = Field(default=None, ge=0, le=32)
    pdf_split_min_pages_for_parallel: Optional[int] = Field(default=None, ge=1)
    default_preset: Optional[str] = None
    allow_directory_import: Optional[bool] = None


class SettingsMetaResponse(BaseModel):
    """Metadata used by the ``/settings`` UI to render labels + reset buttons.

    ``descriptions`` and ``defaults`` are populated from the
    :class:`Settings` model field info so they stay in sync with the
    source of truth (no copy/paste). ``mutable_keys`` is the
    alphabetically-sorted allowlist; the UI renders one form row per key.
    """

    model_config = ConfigDict(extra="forbid")

    descriptions: dict[str, str]
    defaults: dict[str, Any]
    mutable_keys: list[str]


def _field_default(name: str) -> Any:
    """Return the built-in default for ``name`` from the :class:`Settings` model.

    Falls back to invoking ``default_factory`` when the field uses one,
    so callers always get a concrete JSON-serialisable value rather than
    a Pydantic sentinel.
    """
    field_info = Settings.model_fields[name]
    if field_info.default is not PydanticUndefined:
        return field_info.default
    if field_info.default_factory is not None:
        return field_info.default_factory()  # type: ignore[misc]
    return None


def build_meta_response() -> SettingsMetaResponse:
    """Build a :class:`SettingsMetaResponse` from the live ``Settings`` schema."""
    descriptions: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    for key in RUNTIME_MUTABLE_SETTINGS:
        field_info = Settings.model_fields[key]
        descriptions[key] = field_info.description or ""
        defaults[key] = _field_default(key)
    return SettingsMetaResponse(
        descriptions=descriptions,
        defaults=defaults,
        mutable_keys=sorted(RUNTIME_MUTABLE_SETTINGS),
    )
