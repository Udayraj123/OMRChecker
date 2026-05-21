"""Live HTTP smoke test against the running web UI.

Drives the same flow a real browser would: create a batch, POST a PDF,
verify 202 + processing flag, poll for split completion, GET the file
list, verify all `.jpg` extensions, then delete the batch.

This is the closest reliable equivalent to a true browser-level E2E
test now that browser automation cannot drive Windows native file
pickers. The HTTP contract verified here is exactly the contract the
frontend JavaScript depends on.

Run with::

    python scripts/smoke_test_pdf_upload.py --base-url http://127.0.0.1:5051
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:5051")
    ap.add_argument(
        "--pdf",
        type=Path,
        default=REPO_ROOT / "omr_stress_artifacts" / "browser_test_5pages.pdf",
    )
    ap.add_argument("--poll-timeout", type=float, default=60.0)
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("requests is required; install with `pip install requests`")

    if not args.pdf.exists():
        sys.exit(f"test PDF not found: {args.pdf}")

    base = args.base_url.rstrip("/")
    all_ok = True

    print(f"OMRChecker live HTTP smoke test | base={base}")
    print(f"  using {args.pdf.name} ({args.pdf.stat().st_size:,} B)")

    # --- T1: home page reachable ----------------------------------------
    print("\nT1 — home page reachable")
    r = requests.get(base + "/", timeout=10)
    all_ok &= _check("GET / returns 200", r.status_code == 200, f"got {r.status_code}")

    # --- T2: create batch ----------------------------------------------
    print("\nT2 — create batch")
    r = requests.post(base + "/api/v1/batches",
                      json={"name": "live-smoke-pdf-upload"}, timeout=10)
    all_ok &= _check("POST /api/v1/batches returns 201",
                     r.status_code == 201, f"got {r.status_code}")
    if r.status_code != 201:
        print(f"   body: {r.text[:200]}")
        return 1
    batch = r.json()
    batch_id = batch["id"]
    print(f"  batch_id={batch_id}")

    # --- T3: upload PDF, verify 202 + processing flag --------------------
    print("\nT3 — upload PDF, verify 202 + processing flag")
    with args.pdf.open("rb") as fh:
        r = requests.post(
            f"{base}/api/v1/batches/{batch_id}/files",
            files=[("files", (args.pdf.name, fh, "application/pdf"))],
            timeout=30,
        )
    all_ok &= _check("upload returns 202", r.status_code == 202,
                     f"got {r.status_code}")
    if r.status_code == 202:
        body = r.json()
        all_ok &= _check("response body has processing=true",
                         body.get("processing") is True,
                         f"got {body}")
        all_ok &= _check("response body has pdf_count=1",
                         body.get("pdf_count") == 1,
                         f"got pdf_count={body.get('pdf_count')}")

    # --- T4: poll for split completion ----------------------------------
    print("\nT4 — poll for split completion")
    deadline = time.time() + args.poll_timeout
    seen_total = False
    last_status = None
    while time.time() < deadline:
        s = requests.get(f"{base}/api/v1/batches/{batch_id}/status", timeout=10)
        if s.status_code != 200:
            time.sleep(0.3)
            continue
        last_status = s.json()
        total = last_status.get("pdf_split_total") or 0
        pages = last_status.get("pdf_split_pages") or 0
        if total > 0:
            seen_total = True
        if seen_total and total == 0:
            # split finished — total resets to 0
            break
        # also break when batch has files even if status quirks
        time.sleep(0.3)
    print(f"  last status: {last_status}")

    # --- T5: GET file listing, verify .jpg + count -----------------------
    print("\nT5 — file listing all .jpg")
    r = requests.get(f"{base}/api/v1/batches/{batch_id}/files", timeout=10)
    all_ok &= _check("files endpoint returns 200",
                     r.status_code == 200, f"got {r.status_code}")
    files = r.json() if r.status_code == 200 else []
    names = [f["name"] for f in files]
    all_ok &= _check("got 5 pages from 5-page PDF",
                     len(files) == 5,
                     f"got {len(files)} files")
    all_ok &= _check("all extensions are .jpg",
                     all(n.endswith(".jpg") for n in names),
                     f"names={names}")
    all_ok &= _check("no stale .png files in listing",
                     not any(n.endswith(".png") for n in names),
                     f"names={names}")

    # --- T6: verify scratch lives in cache_root, not batch dir ----------
    print("\nT6 — runtime scratch is outside batch dir")
    from webui.settings import get_settings
    settings = get_settings()
    cache_root = settings.cache_root
    print(f"  cache_root={cache_root}")
    all_ok &= _check("cache_root contains 'OMRChecker'",
                     "OMRChecker" in str(cache_root),
                     f"path={cache_root}")
    # The batch's own dir under storage_root should not contain _runtime/.
    batch_dir = settings.storage_root / batch_id
    runtime_in_batch = batch_dir / "_runtime"
    rotated_in_batch = batch_dir / "_rotated"
    all_ok &= _check("batch storage dir does NOT contain _runtime/",
                     not runtime_in_batch.exists(),
                     f"unexpected: {runtime_in_batch}")
    all_ok &= _check("batch storage dir does NOT contain _rotated/",
                     not rotated_in_batch.exists(),
                     f"unexpected: {rotated_in_batch}")

    # --- T7: apply default preset (custom_25_definitive_final) ----------
    print("\nT7 — apply default template via preset")
    preset_template = REPO_ROOT / "custom_25_definitive_final" / "template.json"
    if preset_template.exists():
        with preset_template.open("r", encoding="utf-8") as fh:
            template_payload = fh.read()
        import json as _json
        r = requests.put(
            f"{base}/api/v1/batches/{batch_id}/template",
            json=_json.loads(template_payload),
            timeout=10,
        )
        all_ok &= _check("template PUT returns 200",
                         r.status_code == 200, f"got {r.status_code}")
    else:
        print(f"  (skipped: no preset at {preset_template})")

    # --- T8: kick off OMR processing and wait for done ------------------
    print("\nT8 — process batch end-to-end (in-memory pipeline)")
    r = requests.post(f"{base}/api/v1/batches/{batch_id}/process", timeout=10)
    all_ok &= _check("process POST returns 202",
                     r.status_code == 202, f"got {r.status_code}")
    deadline = time.time() + 120
    final = None
    while time.time() < deadline:
        s = requests.get(f"{base}/api/v1/batches/{batch_id}/status", timeout=10)
        if s.status_code == 200:
            final = s.json()
            if final.get("status") in {"done", "failed", "cancelled"}:
                break
        time.sleep(0.3)
    print(f"  final status: {final.get('status') if final else None} "
          f"processed={final.get('processed_files') if final else None}/"
          f"{final.get('total_files') if final else None}")
    all_ok &= _check("batch finished with terminal status",
                     final is not None and final.get("status") in {"done", "failed"},
                     f"got {final}")
    all_ok &= _check("all 5 pages processed",
                     final is not None and final.get("processed_files") == 5,
                     f"got {final.get('processed_files') if final else None}")

    # --- T9: results endpoint returns rows ------------------------------
    print("\nT9 — results endpoint returns rows")
    r = requests.get(f"{base}/api/v1/batches/{batch_id}/results", timeout=10)
    all_ok &= _check("results endpoint returns 200",
                     r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        results = r.json()
        rows = results.get("rows") or []
        all_ok &= _check("results contain 5 rows",
                         len(rows) == 5, f"got {len(rows)}")
        if rows:
            sample = rows[0]
            all_ok &= _check("first row has file_id + responses payload",
                             "file_id" in sample and "responses" in sample,
                             f"keys={list(sample.keys())[:10]}")
            responses = sample.get("responses") or {}
            all_ok &= _check("responses dict includes question keys (q1, ...)",
                             any(k.lower().startswith("q") for k in responses),
                             f"responses keys={list(responses.keys())[:10]}")

    # --- T10: delete batch ----------------------------------------------
    print("\nT10 — delete batch")
    r = requests.delete(f"{base}/api/v1/batches/{batch_id}", timeout=10)
    all_ok &= _check("delete returns 204",
                     r.status_code == 204,
                     f"got {r.status_code}")

    print()
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
