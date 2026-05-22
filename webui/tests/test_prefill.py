"""Tests for the prefill answer-sheet API paths."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from webui.services import prefill as prefill_service


def test_prefill_batch_accepts_large_csv_without_buffered_generation(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_generate(rows, dst_path, realism_preset="none"):
        calls["count"] = len(rows)
        calls["dst_path"] = dst_path
        calls["realism_preset"] = realism_preset
        dst_path.write_bytes(b"fake pdf")
        return {
            "count": len(rows),
            "successes": len(rows),
            "errors": [],
            "elapsed_s": 0.01,
            "size_bytes": dst_path.stat().st_size,
        }

    monkeypatch.setattr(prefill_service, "generate_batch_pdf_to_file", fake_generate)
    csv_lines = ["student_name,school_name,exam_name,candidate_number"]
    csv_lines.extend(
        f"Student {idx},School,Exam,{idx:010d}" for idx in range(4000)
    )

    response = client.post(
        "/api/v1/prefill/batch",
        data={"output_mode": "pdf", "csv_text": "\n".join(csv_lines)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["count"] == 4000
    assert payload["successes"] == 4000
    assert payload["download_url"].startswith("/api/v1/prefill/batch/download/")
    assert calls["count"] == 4000
    assert calls["realism_preset"] == "none"
