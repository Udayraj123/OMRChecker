"""Count the number of files created by each OMR pipeline path.

The in-memory pipeline is expected to write meaningfully fewer files per
processed page than the legacy directory-staged pipeline. We measure that
directly by watching the on-disk file count in the scratch cache root.

Usage::

    python scripts/verify_inmemory_fewer_writes.py --pages 20

Output is a side-by-side table showing total files written under the
scratch cache root for each path.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*") if _.is_file())


def _run(pages: int, *, inmemory: bool) -> dict:
    os.environ["OMR_WEBUI_INMEMORY_PIPELINE"] = "true" if inmemory else "false"
    from webui.settings import get_settings
    get_settings.cache_clear()
    settings = get_settings()

    # Start from an empty scratch cache so the count reflects only this run.
    if settings.cache_root.exists():
        shutil.rmtree(settings.cache_root)
    settings.ensure_cache_root()

    from webui.services import batches as batches_service
    from webui.services import omr as omr_service

    source = REPO_ROOT / "custom_25_definitive_final" / "inputs" / \
             "MCFormat25questions_page-0001_filled_sample.jpg"
    if not source.exists():
        sys.exit(f"source image not found: {source}")

    import fitz
    img_doc = fitz.open(str(source))
    pdf_bytes_single = img_doc.convert_to_pdf()
    img_doc.close()
    one = fitz.open("pdf", pdf_bytes_single)
    out = fitz.open()
    for _ in range(pages):
        out.insert_pdf(one)
    pdf_bytes = out.tobytes()
    one.close()
    out.close()

    batch_id = batches_service.create_batch(
        f"inmem-verify-{'in' if inmemory else 'legacy'}", settings
    ).id
    batches_service.save_uploaded_file(
        batch_id, "verify.pdf", pdf_bytes, settings
    )

    shutil.copy2(
        REPO_ROOT / "custom_25_definitive_final" / "template.json",
        batches_service.get_batch_root(batch_id, settings) / "template.json",
    )

    omr_service.run_batch_sync(batch_id, settings)
    cache_files = _count_files(settings.cache_root)

    batches_service.delete_batch(batch_id, settings)
    return {
        "inmemory": inmemory,
        "pages": pages,
        "cache_files": cache_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=20)
    args = ap.parse_args()

    print(f"Verifying scratch-file counts | pages={args.pages}")
    legacy = _run(args.pages, inmemory=False)
    inmem = _run(args.pages, inmemory=True)

    print()
    print(f"  legacy   pipeline: {legacy['cache_files']:>5} scratch files "
          f"(~{legacy['cache_files'] / args.pages:.1f} per page)")
    print(f"  inmemory pipeline: {inmem['cache_files']:>5} scratch files "
          f"(~{inmem['cache_files'] / args.pages:.1f} per page)")
    delta = legacy["cache_files"] - inmem["cache_files"]
    if legacy["cache_files"]:
        pct = delta / legacy["cache_files"] * 100
        print(f"  reduction:        {delta:>5} fewer scratch files "
              f"({pct:.0f}% fewer Defender scan targets)")


if __name__ == "__main__":
    main()
