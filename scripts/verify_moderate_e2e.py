"""End-to-end check: prefill a 20-page moderate PDF, push it through OMR.

Used after the moderate-preset regression fix to confirm the engine's
preprocess-failure rate dropped from ~68% to near 0%.

Run with the live server already on http://127.0.0.1:5050.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:5050"
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
MANUAL_TEMPLATE_UPLOAD = "--manual-template" in sys.argv

TEMPLATE_DIR = ROOT / "custom_25_definitive_final"
TEMPLATE_JSON = TEMPLATE_DIR / "template.json"
CONFIG_JSON = TEMPLATE_DIR / "config.json"


def main() -> int:
    csv_lines = ["student_name,school_name,exam_name,candidate_number"]
    for i in range(N_ROWS):
        csv_lines.append(f"Student {i+1},School,Exam,{(9_000_000_000 + i):010d}")
    csv_text = "\n".join(csv_lines)

    with httpx.Client(timeout=120.0) as client:
        # ----- 1. Generate moderate PDF ---------------------------------------
        print(f"Generating {N_ROWS}-page moderate PDF...")
        r = client.post(
            f"{BASE}/api/v1/prefill/batch",
            data={
                "output_mode": "pdf",
                "realism_preset": "moderate",
                "csv_text": csv_text,
            },
        )
        r.raise_for_status()
        info = r.json()
        download_url = info["download_url"]
        generated_filename = info.get("filename") or "prefilled_sheets_moderate.pdf"
        print(f"  successes={info['successes']} elapsed={info['elapsed_s']:.2f}s")

        # ----- 2. Download the PDF --------------------------------------------
        r = client.get(f"{BASE}{download_url}")
        r.raise_for_status()
        pdf_bytes = r.content
        print(f"  PDF size={len(pdf_bytes)/1024:.1f} kB")

        # ----- 3. Create a batch and upload the PDF ---------------------------
        r = client.post(f"{BASE}/api/v1/batches", json={"name": "moderate-e2e-check"})
        r.raise_for_status()
        batch_id = r.json()["id"]
        print(f"  batch_id={batch_id}")

        files = {"files": (generated_filename, io.BytesIO(pdf_bytes), "application/pdf")}
        r = client.post(f"{BASE}/api/v1/batches/{batch_id}/files", files=files)
        r.raise_for_status()
        imp = r.json()
        print(f"  upload: status={imp.get('status')} pages={imp.get('pages')}")

        # ----- 4. Wait for files to settle ------------------------------------
        # Upload may have already extracted pages (file_count > 0). If a PDF
        # split is in-flight, pdf_split_total > 0 and we wait for it.
        for _ in range(180):
            time.sleep(1)
            r = client.get(f"{BASE}/api/v1/batches/{batch_id}/status")
            data = r.json()
            split_total = data.get("pdf_split_total") or 0
            split_done = data.get("pdf_split_pages") or 0
            file_count = data.get("file_count") or 0
            if split_total > 0 and split_done < split_total:
                continue
            if file_count >= N_ROWS:
                print(f"  files ready: file_count={file_count} (split {split_done}/{split_total})")
                break
            if data.get("pdf_split_error"):
                print(f"  split error: {data['pdf_split_error']}")
                return 1
        else:
            print(f"  files never settled: {data}")
            return 1

        # ----- 4b. Optional manual template upload ----------------------------
        # The normal generated-sheet workflow should NOT need this anymore:
        # uploads named prefilled_sheet(s)* auto-attach the built-in 25Q
        # ArUco template/config. Keep this flag for comparing old behaviour.
        if MANUAL_TEMPLATE_UPLOAD and TEMPLATE_JSON.exists():
            template_payload = TEMPLATE_JSON.read_bytes()
            r = client.put(
                f"{BASE}/api/v1/batches/{batch_id}/template",
                json=__import__("json").loads(template_payload),
            )
            if r.status_code != 200:
                print(f"  template upload failed: {r.status_code} {r.text[:200]}")
                return 1
            print(f"  template uploaded ({len(template_payload)} bytes)")
        if MANUAL_TEMPLATE_UPLOAD and CONFIG_JSON.exists():
            config_payload = CONFIG_JSON.read_bytes()
            r = client.put(
                f"{BASE}/api/v1/batches/{batch_id}/config",
                json=__import__("json").loads(config_payload),
            )
            if r.status_code != 200:
                print(f"  config upload failed: {r.status_code} {r.text[:200]}")
                return 1
            print(f"  config uploaded ({len(config_payload)} bytes)")

        # ----- 5. Kick off OMR ------------------------------------------------
        r = client.post(f"{BASE}/api/v1/batches/{batch_id}/process")
        if r.status_code not in {200, 202}:
            print(f"  process kickoff failed: {r.status_code} {r.text[:200]}")
            return 1
        print(f"  OMR processing started")

        # ----- 6. Wait for OMR completion -------------------------------------
        for _ in range(300):
            time.sleep(2)
            r = client.get(f"{BASE}/api/v1/batches/{batch_id}/status")
            data = r.json()
            status = data.get("status")
            if status == "done":
                total = data.get("total_files") or 0
                failures = data.get("preprocess_failures") or []
                fail_count = len(failures) if isinstance(failures, list) else int(failures)
                pct = 100.0 * fail_count / total if total else 0.0
                print(f"\n=== RESULT ===")
                print(f"  total={total} failures={fail_count} ({pct:.1f}%)")
                if fail_count == 0:
                    print("  PASS: zero preprocess failures on moderate")
                    return 0
                elif pct < 5.0:
                    print("  PASS: < 5% failure rate (acceptable)")
                    return 0
                else:
                    print("  FAIL: failure rate above 5%")
                    print(f"  failed: {failures[:10]}")
                    return 2
            if status in {"failed", "cancelled"}:
                print(f"\n  ERROR: status={status} last_error={data.get('last_error')}")
                return 1
        print("  OMR never finished")
        return 1


if __name__ == "__main__":
    sys.exit(main())
