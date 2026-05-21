"""Local benchmark for the PDF split + OMR pipeline.

Measures the wall-clock cost of:
  1. PDF -> per-page image rendering (the "split" step).
  2. OMR processing of those pages by the in-memory and legacy paths.

Usage::

    # Default: 100-page synthetic PDF
    python scripts/bench_pdf_split.py

    # Bigger run, with comparison between in-memory and legacy pipelines.
    python scripts/bench_pdf_split.py --pages 1000 --compare

The benchmark uses an existing preset (``custom_25_definitive_final``) so
it exercises the real engine, including ArUco detection. Pages that fail
preprocessing are still counted as "processed" for the throughput metric.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_pdf(page_count: int, source_image: Path) -> bytes:
    """Build an N-page PDF where every page is the same source image."""
    import fitz
    img_doc = fitz.open(str(source_image))
    if img_doc.is_pdf:
        raise SystemExit("source_image must be an image, not a PDF")
    pix = img_doc.convert_to_pdf()
    img_doc.close()
    one = fitz.open("pdf", pix)
    out = fitz.open()
    for _ in range(page_count):
        out.insert_pdf(one)
    one.close()
    data = out.tobytes()
    out.close()
    return data


def _benchmark(name: str, *, pages: int, source_image: Path, inmemory: bool) -> dict:
    import os
    os.environ["OMR_WEBUI_INMEMORY_PIPELINE"] = "true" if inmemory else "false"
    from webui.settings import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    print(f"\n=== {name}: {pages} pages | inmemory_pipeline={settings.inmemory_pipeline}")

    from webui.services import batches as batches_service
    from webui.services import omr as omr_service
    from webui.schemas import BatchStatus

    batch_id = batches_service.create_batch(f"bench-{name}-{pages}", settings).id

    # 1) PDF -> image split
    pdf_bytes = _make_pdf(pages, source_image)
    t0 = time.perf_counter()
    refs = batches_service.save_uploaded_file(
        batch_id, f"bench_{name}.pdf", pdf_bytes, settings
    )
    t_split = time.perf_counter() - t0
    print(f"  split: {len(refs)} pages in {t_split:.2f}s "
          f"({len(refs)/t_split:.0f} pages/s)")

    # 2) Apply the existing preset's template + config to the batch.
    preset_dir = REPO_ROOT / "custom_25_definitive_final"
    if (preset_dir / "template.json").exists():
        shutil.copy2(
            preset_dir / "template.json",
            batches_service.get_batch_root(batch_id, settings) / "template.json",
        )

    # 3) Run OMR.
    t0 = time.perf_counter()
    omr_service.run_batch_sync(batch_id, settings)
    t_omr = time.perf_counter() - t0
    meta = batches_service.get_batch_metadata(batch_id, settings)
    processed = meta.get("processed_files", 0)
    failed = len(meta.get("preprocess_failures") or [])
    print(f"  omr:   {processed} processed in {t_omr:.2f}s "
          f"({processed/t_omr:.0f} omr/s, {processed*60/t_omr:.0f} omr/min) "
          f"| failed={failed}")

    # Cleanup so repeated runs do not accumulate disk.
    try:
        batches_service.delete_batch(batch_id, settings)
    except Exception:  # noqa: BLE001
        pass

    return {
        "name": name,
        "pages": pages,
        "split_seconds": t_split,
        "omr_seconds": t_omr,
        "processed": processed,
        "failed": failed,
        "inmemory": inmemory,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=100)
    ap.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "custom_25_definitive_final" / "inputs" /
                "MCFormat25questions_page-0001_filled_sample.jpg",
        help="Source image to repeat across PDF pages (must exist).",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Run BOTH in-memory and legacy paths and compare.",
    )
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"source image not found: {args.source}")

    print(f"OMRChecker PDF + OMR benchmark | source={args.source.name}")

    inmem = _benchmark("inmemory", pages=args.pages, source_image=args.source, inmemory=True)
    if args.compare:
        legacy = _benchmark("legacy", pages=args.pages, source_image=args.source, inmemory=False)
        speedup_split = legacy["split_seconds"] / max(inmem["split_seconds"], 1e-9)
        speedup_omr = legacy["omr_seconds"] / max(inmem["omr_seconds"], 1e-9)
        print()
        print(f"=== Comparison ({args.pages} pages) ===")
        print(f"  split speedup (inmem vs legacy): {speedup_split:.2f}x")
        print(f"  omr   speedup (inmem vs legacy): {speedup_omr:.2f}x")
        print(f"  end-to-end (inmem): {inmem['split_seconds']+inmem['omr_seconds']:.2f}s")
        print(f"  end-to-end (legacy): {legacy['split_seconds']+legacy['omr_seconds']:.2f}s")


if __name__ == "__main__":
    main()
