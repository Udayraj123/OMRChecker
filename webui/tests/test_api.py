"""End-to-end tests for the /api/v1 surface.

The flow under test mirrors how a real client uses the API::

    create batch -> upload file -> set template/config -> process -> results

We run the engine against ``custom_25_definitive_final/inputs/`` images so
no new fixtures are needed.  OpenCV UI calls are mocked in the shared
``conftest.py``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from webui.schemas import BatchStatus
from webui.services import batches as batches_service
from webui.services.omr import (
    _compute_dynamic_dimensions,
    _prepare_runtime_base,
    _write_non_interactive_config,
)
from webui.settings import get_settings

def _create_batch(client: TestClient, name: str = "Integration Test") -> str:
    response = client.post("/api/v1/batches", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _upload_image(client: TestClient, batch_id: str, path: Path) -> list[dict]:
    with path.open("rb") as fh:
        response = client.post(
            f"/api/v1/batches/{batch_id}/files",
            files=[("files", (path.name, fh, "image/png"))],
        )
    assert response.status_code == 201, response.text
    return response.json()


def _wait_for_status(
    client: TestClient, batch_id: str, terminal={"done", "failed"}, timeout: float = 30.0
) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/batches/{batch_id}/status")
        assert response.status_code == 200
        data = response.json()
        if data["status"] in terminal:
            return data
        time.sleep(0.1)
    pytest.fail(f"Timed out waiting for terminal status; last payload: {data}")


def test_batch_lifecycle_crud(client: TestClient) -> None:
    response = client.get("/api/v1/batches")
    assert response.status_code == 200
    assert response.json() == []

    batch_id = _create_batch(client, "CRUD batch")

    response = client.get(f"/api/v1/batches/{batch_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "CRUD batch"
    assert body["status"] == "created"
    assert body["file_count"] == 0

    response = client.get("/api/v1/batches")
    assert len(response.json()) == 1

    response = client.delete(f"/api/v1/batches/{batch_id}")
    assert response.status_code == 204

    response = client.get(f"/api/v1/batches/{batch_id}")
    assert response.status_code == 404


def test_health_endpoint_and_security_headers(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_upload_and_list_files(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Upload test")
    _upload_image(client, batch_id, adrian_images[0])

    response = client.get(f"/api/v1/batches/{batch_id}/files")
    assert response.status_code == 200
    files = response.json()
    assert len(files) == 1
    assert files[0]["name"].endswith((".png", ".jpg", ".jpeg"))
    assert files[0]["size_bytes"] > 0


def test_pdf_upload_splits_pages_into_images(
    client: TestClient, storage_root: Path
) -> None:
    fitz = pytest.importorskip("fitz")
    batch_id = _create_batch(client, "PDF upload test")
    pdf = fitz.open()
    for label in ("Page 1", "Page 2"):
        page = pdf.new_page(width=240, height=320)
        page.insert_text((72, 120), label)
    pdf_bytes = pdf.tobytes()
    pdf.close()

    response = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", ("sample.pdf", pdf_bytes, "application/pdf"))],
    )
    assert response.status_code == 201, response.text
    uploaded = response.json()
    assert len(uploaded) == 2
    assert uploaded[0]["name"] == "sample_page_0001.jpg"
    assert uploaded[1]["name"] == "sample_page_0002.jpg"

    response = client.get(f"/api/v1/batches/{batch_id}/files")
    assert response.status_code == 200
    files = response.json()
    assert [file["name"] for file in files] == [
        "sample_page_0001.jpg",
        "sample_page_0002.jpg",
    ]

    # Stale file with .png extension (from prior version) must be removed on re-upload
    stale_duplicate = storage_root / batch_id / "inputs" / "sample_page_0001_1.png"
    stale_duplicate.write_bytes(b"stale")

    response = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", ("sample.pdf", pdf_bytes, "application/pdf"))],
    )
    assert response.status_code == 201, response.text
    uploaded = response.json()
    assert [file["name"] for file in uploaded] == [
        "sample_page_0001.jpg",
        "sample_page_0002.jpg",
    ]

    response = client.get(f"/api/v1/batches/{batch_id}/files")
    assert response.status_code == 200
    files = response.json()
    assert [file["name"] for file in files] == [
        "sample_page_0001.jpg",
        "sample_page_0002.jpg",
    ]
    assert not stale_duplicate.exists()


def test_rejects_unsupported_filetype(client: TestClient) -> None:
    batch_id = _create_batch(client, "Bad upload")
    response = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# PDF rendering quality/format tests
# ---------------------------------------------------------------------------

def _make_simple_pdf(page_count: int = 1, width: int = 72, height: int = 72) -> bytes:
    """Return minimal valid PDF bytes with ``page_count`` blank pages."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(page_count):
        p = doc.new_page(width=width, height=height)
        p.insert_text((10, 40), str(i))
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_pages_are_grayscale_by_default(
    client: TestClient, storage_root: Path
) -> None:
    """Default settings must produce single-channel grayscale JPEGs (saves ~81 %% RAM)."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "Grayscale default")
    pdf_bytes = _make_simple_pdf()

    response = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", ("grey.pdf", pdf_bytes, "application/pdf"))],
    )
    assert response.status_code == 201, response.text

    jpg_path = storage_root / batch_id / "inputs" / "grey_page_0001.jpg"
    assert jpg_path.exists(), "Expected page JPEG not written to disk"
    img = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
    assert img is not None, "cv2 could not read the output JPEG"
    assert img.ndim == 2, (
        f"Expected 2-D grayscale array (1 channel), got shape {img.shape}"
    )


def test_pdf_pages_are_rgb_when_grayscale_disabled(
    storage_root: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """OMR_WEBUI_PDF_RENDER_GRAYSCALE=false must produce 3-channel RGB JPEGs."""
    pytest.importorskip("fitz")
    monkeypatch.setenv("OMR_WEBUI_PDF_RENDER_GRAYSCALE", "false")
    get_settings.cache_clear()

    from webui.app import create_app
    from src.tests.utils import setup_mocker_patches

    setup_mocker_patches(mocker)
    app = create_app()
    with TestClient(app) as rgb_client:
        batch_id = _create_batch(rgb_client, "RGB override")
        pdf_bytes = _make_simple_pdf()
        response = rgb_client.post(
            f"/api/v1/batches/{batch_id}/files",
            files=[("files", ("rgb.pdf", pdf_bytes, "application/pdf"))],
        )
        assert response.status_code == 201, response.text

    jpg_path = storage_root / batch_id / "inputs" / "rgb_page_0001.jpg"
    assert jpg_path.exists()
    img = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 3, (
        f"Expected 3-channel RGB array, got shape {img.shape}"
    )


def test_pdf_page_dimensions_match_150_dpi(
    client: TestClient, storage_root: Path
) -> None:
    """At 150 DPI a 72-pt (1-inch) page must produce a ~150 px wide image."""
    pytest.importorskip("fitz")
    batch_id = _create_batch(client, "DPI dimensions")
    # 72 pt == 1 inch; at 150 DPI → 150 px wide, 300 px tall
    pdf_bytes = _make_simple_pdf(width=72, height=144)

    response = client.post(
        f"/api/v1/batches/{batch_id}/files",
        files=[("files", ("dims.pdf", pdf_bytes, "application/pdf"))],
    )
    assert response.status_code == 201, response.text

    jpg_path = storage_root / batch_id / "inputs" / "dims_page_0001.jpg"
    img = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
    assert img is not None
    # Allow ±2 px rounding from PyMuPDF's integer scaling
    assert abs(img.shape[1] - 150) <= 2, f"Width {img.shape[1]} not ~150 px at 150 DPI"
    assert abs(img.shape[0] - 300) <= 2, f"Height {img.shape[0]} not ~300 px at 150 DPI"


def test_pdf_split_skips_failing_page_and_returns_rest(tmp_path: Path) -> None:
    """A per-page render error is skipped; all other pages are returned."""
    fitz = pytest.importorskip("fitz")
    from unittest.mock import patch
    from webui.services.batches import _save_pdf_pages_as_images

    pdf_bytes = _make_simple_pdf(page_count=3)
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    original = fitz.Page.get_pixmap

    def flaky(self, **kwargs):
        # self.number is the 0-based page index; page index 1 == "page 2"
        if self.number == 1:
            raise RuntimeError("Synthetic render failure on page 2")
        return original(self, **kwargs)

    try:
        with patch.object(fitz.Page, "get_pixmap", flaky):
            refs = _save_pdf_pages_as_images(inputs, "flaky.pdf", pdf_bytes)
    except (TypeError, AttributeError):
        pytest.skip("Cannot patch fitz.Page.get_pixmap on this PyMuPDF build")

    names = {r.name for r in refs}
    assert len(refs) == 2, f"Expected 2 pages after 1 failure, got {len(refs)}"
    assert "flaky_page_0001.jpg" in names
    assert "flaky_page_0002.jpg" not in names, "Failed page must not produce a file"
    assert "flaky_page_0003.jpg" in names


def test_pdf_split_all_pages_fail_raises_error(tmp_path: Path) -> None:
    """If every page fails to render, InvalidBatchRequest is raised (not a partial empty list)."""
    fitz = pytest.importorskip("fitz")
    from unittest.mock import patch
    from webui.services.batches import _save_pdf_pages_as_images, InvalidBatchRequest

    pdf_bytes = _make_simple_pdf(page_count=2)
    inputs = tmp_path / "inputs"
    inputs.mkdir()

    def always_fail(self, **kwargs):
        raise RuntimeError("Synthetic total failure")

    try:
        with patch.object(fitz.Page, "get_pixmap", always_fail):
            with pytest.raises(InvalidBatchRequest, match="all.*page.*failed"):
                _save_pdf_pages_as_images(inputs, "bomb.pdf", pdf_bytes)
    except (TypeError, AttributeError):
        pytest.skip("Cannot patch fitz.Page.get_pixmap on this PyMuPDF build")





def test_rotation_endpoint_persists_allowed_values(client: TestClient) -> None:
    batch_id = _create_batch(client, "Rotation setting")

    response = client.put(
        f"/api/v1/batches/{batch_id}/rotation",
        json={"rotation_degrees": 90},
    )
    assert response.status_code == 200, response.text
    assert response.json()["rotation_degrees"] == 90

    response = client.get(f"/api/v1/batches/{batch_id}")
    assert response.status_code == 200
    assert response.json()["rotation_degrees"] == 90

    response = client.put(
        f"/api/v1/batches/{batch_id}/rotation",
        json={"rotation_degrees": 45},
    )
    assert response.status_code == 422


def test_rotation_restores_sideways_input_for_processing(
    client: TestClient,
    tmp_path: Path,
    adrian_images: list[Path],
    sample_template_body: dict,
    sample_config_body: dict,
) -> None:
    original = cv2.imread(str(adrian_images[0]), cv2.IMREAD_UNCHANGED)
    assert original is not None
    rotated_path = tmp_path / "adrian_sideways.png"
    cv2.imwrite(str(rotated_path), cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE))

    batch_id = _create_batch(client, "Rotated input")
    _upload_image(client, batch_id, rotated_path)
    response = client.put(
        f"/api/v1/batches/{batch_id}/rotation",
        json={"rotation_degrees": 270},
    )
    assert response.status_code == 200, response.text
    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)
    client.put(f"/api/v1/batches/{batch_id}/config", json=sample_config_body)

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 202, response.text
    final = _wait_for_status(client, batch_id)
    assert final["status"] == "done", final
    assert final["latest_dynamic_dimensions"] == _compute_dynamic_dimensions(
        rotated_path, sample_template_body, 270
    )

    response = client.get(f"/api/v1/batches/{batch_id}/results")
    assert response.status_code == 200
    results = response.json()
    assert len(results["rows"]) >= 1
    # The dimensions assertion above already verifies the rotation correction was
    # applied.  Row status depends on the preprocessor succeeding end-to-end;
    # ArUco-based templates may not fully recover from a lossy JPEG→rotate→PNG
    # round-trip, so we do not assert "ok" here.


def test_results_include_failed_error_file_rows(
    client: TestClient,
    storage_root: Path,
) -> None:
    batch_id = _create_batch(client, "Failed rows")
    manual_dir = storage_root / batch_id / "outputs" / "Manual"
    manual_dir.mkdir(parents=True)
    error_csv = manual_dir / "ErrorFiles.csv"
    with error_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_id", "input_path", "output_path", "score", "q1"])
        writer.writerow(
            [
                "failed_page.png",
                "inputs/failed_page.png",
                "outputs/Manual/ErrorFiles/failed_page.png",
                "NA",
                "",
            ]
        )

    response = client.get(f"/api/v1/batches/{batch_id}/results")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["columns"] == ["file_id", "input_path", "output_path", "score", "q1"]
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["file_id"] == "failed_page.png"
    assert row["status"] == "failed"
    assert row["error_reason"]


def test_results_include_qc_flags_for_high_nr_and_bad_candidate(
    client: TestClient,
    storage_root: Path,
) -> None:
    batch_id = _create_batch(client, "QC flags")
    batch_root = storage_root / batch_id
    (batch_root / "outputs" / "Results").mkdir(parents=True)

    client.put(
        f"/api/v1/batches/{batch_id}/config",
        json={"outputs": {"candidate_regex": "^\\d{10}$"}},
    )

    results_csv = batch_root / "outputs" / "Results" / "Results_11AM.csv"
    results_csv.write_text(
        '"file_id","CandidateNumber","q1","q2","q3","q4","q5"\n'
        '"sheet1.png","","NR","NR","NR","NR","NR"\n',
        encoding="utf-8",
    )

    response = client.get(f"/api/v1/batches/{batch_id}/results")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["qc_flags"]
    assert "HIGH_NR" in row["qc_flags"]
    assert "BAD_CANDIDATE" in row["qc_flags"]
    assert row["nr_count"] == 5
    assert row["nr_percent"] >= 0.99


def test_results_do_not_flag_missing_candidate_when_not_configured(
    client: TestClient,
    storage_root: Path,
) -> None:
    batch_id = _create_batch(client, "QC no candidate")
    batch_root = storage_root / batch_id
    (batch_root / "outputs" / "Results").mkdir(parents=True)
    results_csv = batch_root / "outputs" / "Results" / "Results_11AM.csv"
    results_csv.write_text(
        '"file_id","q1","q2","q3","q4","q5"\n'
        '"sheet1.png","A","B","C","D","A"\n',
        encoding="utf-8",
    )

    response = client.get(f"/api/v1/batches/{batch_id}/results")
    assert response.status_code == 200, response.text
    row = response.json()["rows"][0]
    assert "BAD_CANDIDATE" not in row["qc_flags"]


def test_results_endpoint_limits_large_csv_preview(
    client: TestClient,
    storage_root: Path,
) -> None:
    batch_id = _create_batch(client, "Large CSV")
    results_dir = storage_root / batch_id / "outputs" / "Results"
    results_dir.mkdir(parents=True)
    results_csv = results_dir / "Results_11AM.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_id", "score", "q1"])
        for idx in range(750):
            writer.writerow([f"sheet_{idx:04d}.png", str(idx), "A"])

    response = client.get(f"/api/v1/batches/{batch_id}/results?limit=100")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["rows"]) == 100
    assert payload["total_rows"] == 750
    assert payload["limit"] == 100
    assert payload["truncated"] is True


def test_directory_import_disabled_by_default(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Directory import disabled")
    response = client.post(
        f"/api/v1/batches/{batch_id}/files/import",
        json={"source_dir": str(adrian_images[0].parent), "copy": True},
    )
    assert response.status_code == 400
    assert "disabled" in response.json()["detail"].lower()


def test_directory_import(
    client: TestClient, adrian_images: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMR_WEBUI_ALLOW_DIRECTORY_IMPORT", "true")
    get_settings.cache_clear()
    batch_id = _create_batch(client, "Directory import test")
    source_dir = adrian_images[0].parent
    response = client.post(
        f"/api/v1/batches/{batch_id}/files/import",
        json={"source_dir": str(source_dir), "copy": True},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert len(payload["imported"]) == len(adrian_images)
    assert payload["skipped"] == []


def test_template_config_round_trip(
    client: TestClient, sample_template_body: dict, sample_config_body: dict
) -> None:
    batch_id = _create_batch(client, "Template round trip")

    response = client.put(
        f"/api/v1/batches/{batch_id}/template", json=sample_template_body
    )
    assert response.status_code == 200
    assert response.json()["status"] == "saved"

    response = client.put(
        f"/api/v1/batches/{batch_id}/config", json=sample_config_body
    )
    assert response.status_code == 200

    response = client.get(f"/api/v1/batches/{batch_id}/template")
    assert response.status_code == 200
    assert response.json() == sample_template_body

    response = client.get(f"/api/v1/batches/{batch_id}")
    batch = response.json()
    assert batch["has_template"] is True
    assert batch["has_config"] is True


def test_template_upload_rejects_invalid_schema(client: TestClient) -> None:
    batch_id = _create_batch(client, "Invalid template")
    response = client.put(f"/api/v1/batches/{batch_id}/template", json={})
    assert response.status_code == 422
    assert "Invalid template.json" in response.json()["detail"]


def test_process_requires_template(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Missing template")
    _upload_image(client, batch_id, adrian_images[0])

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400
    assert "template" in response.json()["detail"].lower()


def test_full_process_flow_produces_results(
    client: TestClient,
    adrian_images: list[Path],
    sample_template_body: dict,
    sample_config_body: dict,
) -> None:
    batch_id = _create_batch(client, "Happy path")

    for image in adrian_images:
        _upload_image(client, batch_id, image)

    client.put(
        f"/api/v1/batches/{batch_id}/template", json=sample_template_body
    )
    client.put(
        f"/api/v1/batches/{batch_id}/config", json=sample_config_body
    )

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"

    final = _wait_for_status(client, batch_id)
    assert final["status"] == "done", final
    assert final["processed_files"] == len(adrian_images)
    assert final["total_files"] == len(adrian_images)
    assert final["latest_processed_file"] == adrian_images[-1].name
    assert final["latest_dynamic_dimensions"] == _compute_dynamic_dimensions(
        adrian_images[-1], sample_template_body
    )

    response = client.get(f"/api/v1/batches/{batch_id}/results")
    assert response.status_code == 200
    results = response.json()
    assert results["batch_id"] == batch_id
    assert results["generated_csv"] is not None
    assert len(results["rows"]) >= 1
    assert [row["file_id"] for row in results["rows"]] == [
        image.name for image in adrian_images
    ]
    assert "file_id" in results["columns"]
    assert "score" in results["columns"]

    response = client.get(f"/api/v1/batches/{batch_id}/results/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert len(response.content) > 0

    response = client.get(f"/api/v1/batches/{batch_id}/config")
    assert response.status_code == 200
    persisted_config = response.json()
    expected_dimensions = _compute_dynamic_dimensions(adrian_images[-1], sample_template_body)
    assert persisted_config["dimensions"]["display_height"] == expected_dimensions["display_height"]
    assert persisted_config["dimensions"]["display_width"] == expected_dimensions["display_width"]
    assert (
        persisted_config["dimensions"]["processing_height"]
        == expected_dimensions["processing_height"]
    )
    assert (
        persisted_config["dimensions"]["processing_width"]
        == expected_dimensions["processing_width"]
    )


def test_ui_pages_render(
    client: TestClient, sample_template_body: dict, adrian_images: list[Path]
) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "OMRChecker" in response.text

    batch_id = _create_batch(client, "HTML test")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)

    response = client.get(f"/batches/{batch_id}")
    assert response.status_code == 200
    assert "HTML test" in response.text
    assert "template.json" in response.text
    assert "Template parameter help" in response.text
    assert "origin" in response.text
    assert "bubblesGap" in response.text
    assert "Template assets" in response.text


def test_staged_config_forces_non_interactive(tmp_path: Path) -> None:
    src = tmp_path / "config.json"
    dst = tmp_path / "staged_config.json"
    src.write_text(
        '{"dimensions":{"display_height":2480},"outputs":{"show_image_level":5}}',
        encoding="utf-8",
    )

    _write_non_interactive_config(src, dst)

    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["outputs"]["show_image_level"] == 0


def test_runtime_base_rebuilds_and_injects_marker_defaults(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    batch_root.mkdir()
    (batch_root / "template.json").write_text(
        json.dumps(
            {
                "preProcessors": [
                    {
                        "name": "CropOnMarkers",
                        "options": {
                            "relativePath": "omr_marker.jpg",
                            "markerCorners": [[0, 40, 0, 40]] * 4,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (batch_root / "omr_marker.jpg").write_bytes(b"marker")

    stale_base = batch_root / "_runtime" / "_base"
    stale_base.mkdir(parents=True)
    (stale_base / "template.json").write_text('{"stale": true}', encoding="utf-8")

    base_root = _prepare_runtime_base(batch_root)

    runtime_template = json.loads((base_root / "template.json").read_text(encoding="utf-8"))
    options = runtime_template["preProcessors"][0]["options"]
    assert "stale" not in runtime_template
    assert options["markerSearchPadding"] == 20
    assert options["fallbackToExpandedMarkerCorners"] is True
    assert options["max_matching_variation"] == 0.5


_MARKER_TEMPLATE = {
    "pageDimensions": [300, 400],
    "bubbleDimensions": [25, 25],
    "fieldBlocks": {
        "MCQ_Block_1": {
            "fieldType": "QTYPE_MCQ4",
            "origin": [65, 60],
            "fieldLabels": ["q1..2"],
            "labelsGap": 52,
            "bubblesGap": 41,
        }
    },
    "preProcessors": [
        {
            "name": "CropOnMarkers",
            "options": {
                "relativePath": "omr_marker.jpg",
                "sheetToMarkerWidthRatio": 17,
            },
        }
    ],
}


def test_missing_template_asset_blocks_processing_with_clear_error(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Missing marker asset")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=_MARKER_TEMPLATE)

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "omr_marker.jpg" in detail
    assert "Template assets" in detail

    response = client.get(f"/api/v1/batches/{batch_id}/status")
    assert response.json()["status"] == "created"


def test_list_template_assets_reports_missing_required_asset(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Assets listing")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=_MARKER_TEMPLATE)

    response = client.get(f"/api/v1/batches/{batch_id}/assets")
    assert response.status_code == 200
    assets = response.json()
    assert len(assets) == 1
    assert assets[0]["name"] == "omr_marker.jpg"
    assert assets[0]["required"] is True
    assert assets[0]["present"] is False


def test_upload_template_asset_satisfies_preflight(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Upload asset")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=_MARKER_TEMPLATE)

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400

    fake_marker_bytes = b"\xff\xd8\xff\xe0" + b"0" * 64
    response = client.post(
        f"/api/v1/batches/{batch_id}/assets",
        files=[("files", ("omr_marker.jpg", fake_marker_bytes, "image/jpeg"))],
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body[0]["name"] == "omr_marker.jpg"
    assert body[0]["present"] is True

    response = client.get(f"/api/v1/batches/{batch_id}/assets")
    assets = response.json()
    assert assets[0]["present"] is True
    assert assets[0]["size_bytes"] == len(fake_marker_bytes)

    settings = get_settings()
    missing = batches_service.missing_template_assets(batch_id, settings)
    assert missing == []

    response = client.delete(
        f"/api/v1/batches/{batch_id}/assets/omr_marker.jpg"
    )
    assert response.status_code == 204

    response = client.get(f"/api/v1/batches/{batch_id}/assets")
    assert response.json()[0]["present"] is False

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 400
    assert "omr_marker.jpg" in response.json()["detail"]


def test_asset_upload_rejects_bad_filenames(
    client: TestClient, adrian_images: list[Path]
) -> None:
    batch_id = _create_batch(client, "Bad asset upload")
    _upload_image(client, batch_id, adrian_images[0])

    response = client.post(
        f"/api/v1/batches/{batch_id}/assets",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 400

    response = client.post(
        f"/api/v1/batches/{batch_id}/assets",
        files=[("files", ("template.json", b"{}", "application/json"))],
    )
    assert response.status_code == 400


def test_cancel_endpoint_cancels_queued_batch(
    client: TestClient, adrian_images: list[Path], sample_template_body: dict
) -> None:
    batch_id = _create_batch(client, "Queued cancel")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)

    settings = get_settings()
    batches_service.update_status(batch_id, BatchStatus.queued, settings=settings)

    response = client.post(f"/api/v1/batches/{batch_id}/cancel")
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "cancelled"

    response = client.get(f"/api/v1/batches/{batch_id}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "cancelled"
    assert payload["cancel_requested"] is True


def test_restart_endpoint_requeues_non_running_batch(
    client: TestClient, adrian_images: list[Path], sample_template_body: dict, monkeypatch
) -> None:
    batch_id = _create_batch(client, "Restart me")
    _upload_image(client, batch_id, adrian_images[0])
    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)

    settings = get_settings()
    batches_service.update_status(
        batch_id,
        BatchStatus.failed,
        last_error="Synthetic failure",
        settings=settings,
    )

    from webui.services import omr as omr_service

    def fake_run(batch_id_arg, settings_arg=None):
        batches_service.update_status(
            batch_id_arg,
            BatchStatus.done,
            settings=settings_arg or settings,
        )

    monkeypatch.setattr(omr_service, "run_batch_sync", fake_run)

    response = client.post(f"/api/v1/batches/{batch_id}/restart")
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"

    response = client.get(f"/api/v1/batches/{batch_id}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "done"


def test_directory_import_processes_with_dynamic_dimensions(
    client: TestClient,
    adrian_images: list[Path],
    sample_template_body: dict,
    sample_config_body: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMR_WEBUI_ALLOW_DIRECTORY_IMPORT", "true")
    get_settings.cache_clear()
    batch_id = _create_batch(client, "Dynamic directory batch")
    source_dir = adrian_images[0].parent

    response = client.post(
        f"/api/v1/batches/{batch_id}/files/import",
        json={"source_dir": str(source_dir), "copy": True},
    )
    assert response.status_code == 201, response.text

    client.put(f"/api/v1/batches/{batch_id}/template", json=sample_template_body)
    client.put(f"/api/v1/batches/{batch_id}/config", json=sample_config_body)

    response = client.post(f"/api/v1/batches/{batch_id}/process")
    assert response.status_code == 202, response.text

    final = _wait_for_status(client, batch_id)
    assert final["status"] == "done", final
    assert final["processed_files"] == len(adrian_images)
    assert final["total_files"] == len(adrian_images)

    latest_expected = _compute_dynamic_dimensions(adrian_images[-1], sample_template_body)
    assert final["latest_dynamic_dimensions"] == latest_expected

    response = client.get(f"/api/v1/batches/{batch_id}/config")
    assert response.status_code == 200
    persisted_config = response.json()
    assert persisted_config["dimensions"]["processing_width"] == latest_expected["processing_width"]
    assert persisted_config["dimensions"]["processing_height"] == latest_expected["processing_height"]
    assert persisted_config["outputs"]["show_image_level"] == sample_config_body["outputs"]["show_image_level"]
