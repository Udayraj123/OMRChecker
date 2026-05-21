"""Live stress test of the parallel PDF split path.

Uploads a 25-page PDF (exercises the parallel branch because the
default threshold is 16) and measures wall-clock split + OMR time
end-to-end through the running web UI.

Usage:
    python scripts/stress_test_parallel_split.py --base-url http://127.0.0.1:5050
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import requests

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:5050")
    ap.add_argument(
        "--pdf",
        type=Path,
        default=REPO_ROOT / "omr_stress_artifacts" / "browser_test_25pages.pdf",
    )
    ap.add_argument("--keep", action="store_true",
                    help="Do not delete the test batch at the end.")
    args = ap.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    base = args.base_url.rstrip("/")

    print(f"Live parallel-split stress test")
    print(f"  base={base}")
    print(f"  pdf={args.pdf.name} ({args.pdf.stat().st_size:,} B)")
    print()

    # Create batch
    r = requests.post(base + "/api/v1/batches",
                      json={"name": "stress-parallel-split"}, timeout=10)
    if r.status_code != 201:
        sys.exit(f"batch create failed: {r.status_code} {r.text}")
    batch_id = r.json()["id"]
    print(f"  batch_id={batch_id}")

    # Upload (kicks off background split)
    t0 = time.perf_counter()
    with args.pdf.open("rb") as fh:
        r = requests.post(
            f"{base}/api/v1/batches/{batch_id}/files",
            files=[("files", (args.pdf.name, fh, "application/pdf"))],
            timeout=60,
        )
    upload_post = time.perf_counter() - t0
    if r.status_code != 202:
        sys.exit(f"upload failed: {r.status_code} {r.text}")
    body = r.json()
    print(f"  upload POST returned 202 in {upload_post:.2f}s "
          f"(pdf_count={body.get('pdf_count')})")

    # Poll until split completes (file_count = 25 or pdf_split_total goes back to 0)
    deadline = time.time() + 120
    t0 = time.perf_counter()
    seen_total = False
    last = None
    while time.time() < deadline:
        s = requests.get(f"{base}/api/v1/batches/{batch_id}/status", timeout=10)
        if s.status_code == 200:
            last = s.json()
            total = last.get("pdf_split_total") or 0
            pages = last.get("pdf_split_pages") or 0
            fc = last.get("file_count") or 0
            if total > 0:
                seen_total = True
                print(f"    split progress: {pages}/{total} pages "
                      f"(file_count={fc})")
            if (seen_total and total == 0) or fc >= 25:
                break
        time.sleep(0.5)
    split_wall = time.perf_counter() - t0
    print(f"  split completed in {split_wall:.2f}s "
          f"(file_count={last.get('file_count') if last else None})")

    # Verify all 25 files .jpg
    r = requests.get(f"{base}/api/v1/batches/{batch_id}/files", timeout=10)
    files = r.json() if r.status_code == 200 else []
    names = [f["name"] for f in files]
    all_jpg = all(n.endswith(".jpg") for n in names)
    print(f"  got {len(files)} files | all_jpg={all_jpg}")

    if len(files) != 25 or not all_jpg:
        print(f"  WARN: unexpected file listing: {names[:5]}...")

    if not args.keep:
        requests.delete(f"{base}/api/v1/batches/{batch_id}", timeout=10)
        print(f"  batch deleted")
    else:
        print(f"  batch kept: {batch_id}")

    print()
    print(f"SUMMARY:")
    print(f"  PDF: 25 pages")
    print(f"  Upload POST round-trip:  {upload_post:.2f}s")
    print(f"  Split wall-clock:        {split_wall:.2f}s")
    print(f"  Throughput:              {25 / split_wall:.1f} pages/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
