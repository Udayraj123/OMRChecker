"""Build a small multi-page PDF from the existing sample OMR sheet.

The resulting PDF is used by browser-based E2E tests as a real upload
payload — the same image repeated across N pages so we can exercise PDF
splitting without needing a unique large PDF asset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "omr_stress_artifacts" / "browser_test_5pages.pdf",
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "custom_25_definitive_final" / "inputs" /
                "MCFormat25questions_page-0001_filled_sample.jpg",
    )
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"source image not found: {args.source}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import fitz
    img_doc = fitz.open(str(args.source))
    pdf_bytes_single = img_doc.convert_to_pdf()
    img_doc.close()
    one = fitz.open("pdf", pdf_bytes_single)
    out = fitz.open()
    for _ in range(args.pages):
        out.insert_pdf(one)
    out.save(str(args.out))
    one.close()
    out.close()

    print(f"Wrote {args.pages}-page test PDF -> {args.out}")


if __name__ == "__main__":
    main()
