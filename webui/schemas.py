"""Pydantic schemas for the OMRChecker Web UI."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class BatchStatus(str, Enum):
    """Lifecycle states for a batch."""

    created = "created"
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class SourceMode(str, Enum):
    """How input files got into the batch."""

    upload = "upload"
    directory = "directory"
    mixed = "mixed"


class BatchCreate(BaseModel):
    """Payload for creating a new batch."""

    name: str = Field(min_length=1, max_length=120)


class Batch(BaseModel):
    """Persisted metadata about a batch."""

    id: str
    name: str
    status: BatchStatus = BatchStatus.created
    created_at: datetime
    updated_at: datetime
    source_mode: SourceMode | None = None
    source_dir: str | None = None
    last_error: str | None = None
    file_count: int = 0
    has_template: bool = False
    has_config: bool = False
    has_evaluation: bool = False
    rotation_degrees: Literal[0, 90, 180, 270] = 0


class BatchRotationUpdate(BaseModel):
    """Payload for rotating all batch inputs before processing."""

    rotation_degrees: Literal[0, 90, 180, 270] = 0


class FileRef(BaseModel):
    """A single image file inside a batch."""

    name: str
    size_bytes: int


class TemplateAssetRef(BaseModel):
    """A file referenced by ``template.json`` (e.g. marker image)."""

    name: str
    required: bool = True
    present: bool = False
    size_bytes: int | None = None


class DirectoryImportRequest(BaseModel):
    """Payload for importing scanned images from a server-side directory."""

    model_config = {"protected_namespaces": ()}

    source_dir: str = Field(min_length=1)
    copy_files: bool = Field(
        default=True,
        alias="copy",
        description="Copy files into the batch (true) or symlink where supported (false).",
    )


class ImportResult(BaseModel):
    """Result of a directory import."""

    imported: list[FileRef]
    skipped: list[str] = Field(default_factory=list)


class JsonDocument(BaseModel):
    """Wrapper for optional template/config/evaluation JSON blobs."""

    name: str
    content: dict[str, Any] | None = None


class ProcessAccepted(BaseModel):
    """Response after queueing a processing run."""

    batch_id: str
    status: BatchStatus


class DynamicDimensions(BaseModel):
    """Derived dimensions used for a single processed image."""

    source_height: int
    source_width: int
    display_height: int
    display_width: int
    processing_height: int
    processing_width: int


class BatchStatusResponse(BaseModel):
    """Polled status payload."""

    id: str
    status: BatchStatus
    last_error: str | None = None
    file_count: int
    updated_at: datetime
    processed_files: int = 0
    total_files: int = 0
    latest_processed_file: str | None = None
    latest_dynamic_dimensions: DynamicDimensions | None = None
    cancel_requested: bool = False
    preprocess_failures: list[str] = Field(default_factory=list)
    # Live timing fields (populated while status == "running")
    elapsed_s: float | None = None
    rate_per_min: float | None = None
    eta_s: float | None = None
    # PDF split progress (populated while a PDF is being split during upload)
    pdf_split_pages: int = 0
    pdf_split_total: int = 0
    # Set when a background PDF split fails; cleared on the next upload attempt
    pdf_split_error: str | None = None
    # True for the current run when OMR engaged pipelined mode (started while a
    # PDF split was still in flight). Set at run start and remains True for the
    # duration of the run so the UI can surface a "Pipelined" badge.
    pipelined_run: bool = False
    # Auto-start OMR (frontend reads these to decide whether to fire the
    # /process POST automatically once enough split pages exist).
    auto_start_omr_with_split: bool = True
    auto_start_omr_min_pages: int = 10
    auto_start_omr_require_config: bool = False
    # Mirror the Batch model so the auto-start gating check is a single
    # status fetch instead of an extra GET /batches/{id}.
    has_template: bool = False
    has_config: bool = False


class ResultsRow(BaseModel):
    """A single parsed row from the generated Results CSV.

    The OMR engine writes arbitrary per-template columns; those are kept
    in ``responses`` so this model stays stable across templates.
    """

    file_id: str
    input_path: str | None = None
    output_path: str | None = None
    score: str | None = None
    status: Literal["ok", "failed"] = "ok"
    error_reason: str | None = None
    qc_flags: list[str] = Field(default_factory=list)
    nr_count: int = 0
    nr_percent: float = 0.0
    responses: dict[str, str] = Field(default_factory=dict)


class ResultsPayload(BaseModel):
    """Results payload exposed by the API."""

    batch_id: str
    columns: list[str]
    rows: list[ResultsRow]
    generated_csv: str | None = None
    total_rows: int = 0
    offset: int = 0
    limit: int | None = None
    truncated: bool = False
