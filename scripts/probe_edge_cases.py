"""Probe live server for edge-case failure modes.

Targets the running OMRChecker web UI to surface real-world bugs that
the unit tests don't cover:

  1. Malformed PDF bytes -> should return 202 then mark batch failed
     (or empty) without crashing the server.
  2. Two PDFs uploaded concurrently to the same batch.
  3. Pathologically long filename (sanitisation check).
  4. Empty PDF (zero pages) -> must produce a clean 4xx, not a 5xx.
  5. /api/v1/system/info sanity.

All probes clean up after themselves.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:5050")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    pdf_path = Path("omr_stress_artifacts/browser_test_5pages.pdf")
    if not pdf_path.exists():
        raise SystemExit(f"missing test PDF: {pdf_path}")
    pdf_bytes = pdf_path.read_bytes()

    def _create_batch(name: str) -> str:
        r = requests.post(base + "/api/v1/batches", json={"name": name})
        r.raise_for_status()
        return r.json()["id"]

    print("\n== 1) Invalid PDF bytes ==")
    bid = _create_batch("probe-bad-pdf")
    r = requests.post(
        f"{base}/api/v1/batches/{bid}/files",
        files=[("files", ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf"))],
    )
    print(f"  upload: status={r.status_code} body={r.text[:200]}")
    time.sleep(2)
    s = requests.get(f"{base}/api/v1/batches/{bid}/status").json()
    print(f"  status after 2s: status={s.get('status')} "
          f"file_count={s.get('file_count')} last_error={s.get('last_error')}")
    requests.delete(f"{base}/api/v1/batches/{bid}")

    print("\n== 2) Concurrent uploads to same batch ==")
    bid = _create_batch("probe-concurrent")
    errors = []
    def upload(name: str) -> None:
        try:
            r = requests.post(
                f"{base}/api/v1/batches/{bid}/files",
                files=[("files", (name, pdf_bytes, "application/pdf"))],
                timeout=30,
            )
            if r.status_code != 202:
                errors.append(f"{name}: {r.status_code} {r.text[:120]}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    t1 = threading.Thread(target=upload, args=("concA.pdf",))
    t2 = threading.Thread(target=upload, args=("concB.pdf",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    time.sleep(6)
    files = requests.get(f"{base}/api/v1/batches/{bid}/files").json()
    print(f"  upload thread errors: {errors}")
    print(f"  total files after both uploads: {len(files)} (expected 10)")
    distinct_stems = {f["name"].split("_page_")[0] for f in files}
    print(f"  distinct stems: {sorted(distinct_stems)}")
    requests.delete(f"{base}/api/v1/batches/{bid}")

    print("\n== 3) Long filename ==")
    bid = _create_batch("probe-longname")
    long_name = ("x" * 200) + ".pdf"
    r = requests.post(
        f"{base}/api/v1/batches/{bid}/files",
        files=[("files", (long_name, pdf_bytes, "application/pdf"))],
    )
    print(f"  upload: status={r.status_code}")
    time.sleep(2)
    files = requests.get(f"{base}/api/v1/batches/{bid}/files").json()
    first_name = files[0]["name"] if files else None
    print(f"  files: {len(files)} | first len={len(first_name) if first_name else 0}")
    requests.delete(f"{base}/api/v1/batches/{bid}")

    print("\n== 4) Truncated PDF (header only) ==")
    bid = _create_batch("probe-empty")
    truncated = b"%PDF-1.4\n%%EOF\n"
    r = requests.post(
        f"{base}/api/v1/batches/{bid}/files",
        files=[("files", ("trunc.pdf", truncated, "application/pdf"))],
    )
    print(f"  upload: status={r.status_code} body={r.text[:200]}")
    time.sleep(2)
    s = requests.get(f"{base}/api/v1/batches/{bid}/status").json()
    print(f"  status: status={s.get('status')} "
          f"file_count={s.get('file_count')} last_error={s.get('last_error')}")
    requests.delete(f"{base}/api/v1/batches/{bid}")

    print("\n== 5) System info ==")
    r = requests.get(f"{base}/api/v1/system/info")
    print(f"  status={r.status_code} body={r.text}")

    print("\nProbes complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
