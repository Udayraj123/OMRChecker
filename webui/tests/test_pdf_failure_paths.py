"""Tests for PDF-upload failure paths (Fix #2) and long-filename truncation (Fix #4).

Covers:
- Corrupt PDF → 202 → status poll shows pdf_split_error containing the filename stem.
- Truncated / minimal-garbage PDF → same outcome.
- Valid 2-page PDF → pdf_split_error stays None after split.
- 250-char-stem PDF → file_count > 0 and all resulting filenames are < 200 chars.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_batch(client: TestClient, name: str = "PDF failure test") -> str:
    resp = client.post("/api/v1/batches", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _poll_status(client: TestClient, batch_id: str, timeout: float = 6.0) -> dict:
    """Poll until pdf_split_total reaches 0 (or pdf_split_error is set) or timeout."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/api/v1/batches/{batch_id}/status")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        # In TestClient, background tasks run synchronously before the first
        # status poll, so the very first response should already be settled.
        if last.get("pdf_split_error") or last.get("pdf_split_total", 0) == 0:
            return last
        time.sleep(0.1)
    return last


def _upload_pdf(
    client: TestClient,
    batch_id: str,
    filename: str,
    content: bytes,
    expected_status: int = 202,
) -> dict:
    resp = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", (filename, content, "application/pdf"))],
    )
    assert resp.status_code == expected_status, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Fix #2 – corrupt / truncated PDFs surface an error
# ---------------------------------------------------------------------------

def test_corrupt_pdf_sets_split_error(client: TestClient) -> None:
    """A corrupt PDF byte-stream must set pdf_split_error on the status endpoint."""
    fitz = pytest.importorskip("fitz")
    batch_id = _create_batch(client, "corrupt pdf test")

    corrupt = b"%PDF-1.4 garbage data that is not a valid pdf stream\x00\x01\x02"
    _upload_pdf(client, batch_id, "corrupt_scan.pdf", corrupt)

    status = _poll_status(client, batch_id)
    error = status.get("pdf_split_error")
    assert error, f"Expected pdf_split_error to be set; got status={status!r}"
    # The stem of the uploaded filename must appear in the error message so the
    # user knows which file failed.
    assert "corrupt_scan" in error, (
        f"Expected 'corrupt_scan' in error message, got: {error!r}"
    )
    # Progress fields must be cleared (unblocks the frontend poller).
    assert status["pdf_split_pages"] == 0
    assert status["pdf_split_total"] == 0


def test_truncated_pdf_sets_split_error(client: TestClient) -> None:
    """A valid PDF header followed by truncated content must set pdf_split_error."""
    fitz = pytest.importorskip("fitz")
    batch_id = _create_batch(client, "truncated pdf test")

    # Build a real PDF then keep only the first 64 bytes (truncated mid-stream).
    doc = fitz.open()
    doc.new_page(width=200, height=280)
    real_bytes = doc.tobytes()
    doc.close()
    truncated = real_bytes[:64]

    _upload_pdf(client, batch_id, "truncated_scan.pdf", truncated)

    status = _poll_status(client, batch_id)
    error = status.get("pdf_split_error")
    assert error, f"Expected pdf_split_error; got {status!r}"
    assert "truncated_scan" in error, f"Stem missing from error: {error!r}"


def test_valid_pdf_no_split_error(client: TestClient) -> None:
    """A valid 2-page PDF must leave pdf_split_error as None after a successful split."""
    fitz = pytest.importorskip("fitz")
    batch_id = _create_batch(client, "valid pdf test")

    doc = fitz.open()
    for label in ("Page 1", "Page 2"):
        page = doc.new_page(width=240, height=320)
        page.insert_text((72, 120), label)
    pdf_bytes = doc.tobytes()
    doc.close()

    _upload_pdf(client, batch_id, "good_scan.pdf", pdf_bytes)

    status = _poll_status(client, batch_id)
    assert status.get("pdf_split_error") is None, (
        f"Unexpected split error for valid PDF: {status['pdf_split_error']!r}"
    )
    # Two pages must have been produced.
    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    assert len(files_resp.json()) == 2


def test_corrupt_pdf_error_cleared_on_retry(client: TestClient) -> None:
    """Uploading a valid PDF after a failed corrupt upload clears pdf_split_error."""
    fitz = pytest.importorskip("fitz")
    batch_id = _create_batch(client, "retry test")

    # First upload a corrupt PDF to set the error flag.
    _upload_pdf(client, batch_id, "scan.pdf", b"%PDF garbage")
    bad_status = _poll_status(client, batch_id)
    assert bad_status.get("pdf_split_error"), "Expected error after corrupt upload"

    # Now upload a valid PDF with the same name — error must clear.
    doc = fitz.open()
    doc.new_page(width=200, height=280)
    good_bytes = doc.tobytes()
    doc.close()
    _upload_pdf(client, batch_id, "scan.pdf", good_bytes)

    good_status = _poll_status(client, batch_id)
    assert good_status.get("pdf_split_error") is None, (
        f"Error not cleared after successful re-upload: {good_status['pdf_split_error']!r}"
    )


# ---------------------------------------------------------------------------
# Fix #4 – long filenames are truncated and still produce pages
# ---------------------------------------------------------------------------

def test_long_stem_pdf_produces_files_with_short_names(
    client: TestClient, storage_root: Path
) -> None:
    """A 250-char-stem PDF must produce page images with filenames < 200 chars."""
    fitz = pytest.importorskip("fitz")
    batch_id = _create_batch(client, "long filename test")

    long_stem = "A" * 250
    long_filename = long_stem + ".pdf"

    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=200, height=280)
        page.insert_text((72, 140), f"Page {i + 1}")
    pdf_bytes = doc.tobytes()
    doc.close()

    _upload_pdf(client, batch_id, long_filename, pdf_bytes)

    # Background tasks run synchronously in TestClient, so files are ready now.
    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()
    assert len(files) > 0, "Expected at least one page image after long-stem PDF split"

    for f in files:
        assert len(f["name"]) < 200, (
            f"Filename too long ({len(f['name'])} chars): {f['name']!r}"
        )
