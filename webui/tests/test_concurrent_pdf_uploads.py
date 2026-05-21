"""Tests for concurrent same-stem PDF uploads (Fix #3).

Verifies that two PDFs uploaded simultaneously with the same filename to the
same batch do not clobber each other's rendered page images.

Strategy: Option A (per-(batch_id, stem) threading.Lock) is in place inside
_save_pdf_pages_as_images.  These tests exercise both the concurrent path and
a simple serial sanity-check.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_batch(client: TestClient, name: str = "Concurrent PDF test") -> str:
    resp = client.post("/api/v1/batches", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _make_pdf_bytes(page_count: int, width: int = 210, height: int = 297) -> bytes:
    """Build a minimal fitz PDF with *page_count* blank pages."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 140), f"Page {i + 1} ({width}x{height})")
    data = doc.tobytes()
    doc.close()
    return data


def _upload_pdf_sync(
    client: TestClient,
    batch_id: str,
    filename: str,
    content: bytes,
    results: list,
    index: int,
) -> None:
    """Thread target: upload *content* as *filename* and store the response."""
    resp = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", (filename, content, "application/pdf"))],
    )
    results[index] = resp


# ---------------------------------------------------------------------------
# Serial sanity-check: two sequential same-stem uploads yield pages from the
# second upload only (the first is replaced) — existing behaviour is preserved.
# ---------------------------------------------------------------------------

def test_serial_same_stem_second_upload_replaces_first(client: TestClient) -> None:
    """Serial re-upload of the same PDF stem replaces the first set of pages."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "Serial same-stem test")

    pdf_a = _make_pdf_bytes(3, width=210, height=297)
    pdf_b = _make_pdf_bytes(5, width=210, height=297)

    for pdf_bytes in (pdf_a, pdf_b):
        resp = client.post(
            f"/api/v1/batches/{batch_id}/files",
            files=[("files", ("scans.pdf", pdf_bytes, "application/pdf"))],
        )
        assert resp.status_code == 202, resp.text

    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()
    # The second upload (5 pages) replaced the first (3 pages).
    assert len(files) == 5, (
        f"Expected 5 pages after second upload replaced first; got {len(files)}"
    )
    names = [f["name"] for f in files]
    assert "scans_page_0005.jpg" in names
    # No duplicate names.
    assert len(names) == len(set(names)), f"Duplicate filenames detected: {names}"


# ---------------------------------------------------------------------------
# Concurrent upload: two *distinct* PDFs with the same filename to the same
# batch.  The lock serialises them so the output is one complete set of pages
# (whichever ran second wins), not an interleaved corrupt mix.
# ---------------------------------------------------------------------------

def test_concurrent_same_stem_no_collision(client: TestClient) -> None:
    """Concurrent same-stem PDFs must not produce duplicate or missing pages."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "Concurrent same-stem test")

    # Use different page counts so we can tell which upload won.
    pdf_5 = _make_pdf_bytes(5, width=200, height=280)
    pdf_7 = _make_pdf_bytes(7, width=200, height=280)

    results: list = [None, None]

    t1 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "scans.pdf", pdf_5, results, 0),
    )
    t2 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "scans.pdf", pdf_7, results, 1),
    )

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results[0] is not None and results[1] is not None, (
        "One or both upload threads did not complete in time"
    )
    assert results[0].status_code == 202, results[0].text
    assert results[1].status_code == 202, results[1].text

    # Allow background tasks a moment to settle (TestClient runs them
    # synchronously within each request, but both requests have already
    # returned their 202 responses by the time we reach here).
    time.sleep(0.5)

    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()
    names = [f["name"] for f in files]

    # No duplicate filenames (page clobbering would produce identical names).
    assert len(names) == len(set(names)), (
        f"Duplicate page filenames detected — concurrent clobbering occurred: {names}"
    )

    # The lock ensures exactly one complete set of pages landed (either 5 or 7).
    assert len(files) in {5, 7}, (
        f"Expected 5 or 7 pages (one complete set); got {len(files)}: {names}"
    )


# ---------------------------------------------------------------------------
# Concurrent upload of *different* stems must run fully in parallel (each
# should produce its own page set without interfering with the other).
# ---------------------------------------------------------------------------

def test_concurrent_different_stems_both_complete(client: TestClient) -> None:
    """Two PDFs with different stems uploaded concurrently must both produce pages."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "Concurrent different-stem test")

    pdf_a = _make_pdf_bytes(3, width=210, height=297)
    pdf_b = _make_pdf_bytes(4, width=148, height=210)

    results: list = [None, None]

    t1 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "alpha.pdf", pdf_a, results, 0),
    )
    t2 = threading.Thread(
        target=_upload_pdf_sync,
        args=(client, batch_id, "beta.pdf", pdf_b, results, 1),
    )

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results[0].status_code == 202
    assert results[1].status_code == 202

    files_resp = client.get(f"/api/v1/batches/{batch_id}/files")
    assert files_resp.status_code == 200
    files = files_resp.json()
    names = {f["name"] for f in files}

    alpha_pages = {f"alpha_page_{i:04d}.jpg" for i in range(1, 4)}
    beta_pages = {f"beta_page_{i:04d}.jpg" for i in range(1, 5)}

    assert alpha_pages <= names, (
        f"Missing alpha pages; got: {sorted(names)}"
    )
    assert beta_pages <= names, (
        f"Missing beta pages; got: {sorted(names)}"
    )
    assert len(names) == len(set(names)), "Duplicate filenames detected"
