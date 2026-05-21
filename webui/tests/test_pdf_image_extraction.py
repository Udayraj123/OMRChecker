"""Tests for the embedded-image fast path in batches._try_extract_embedded_page_image.

Fix #1 — Direct embedded-image fast path
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gray_jpeg(width: int = 100, height: int = 100, quality: int = 90) -> bytes:
    """Return grayscale JPEG bytes for use as an embedded page image."""
    fitz = pytest.importorskip("fitz")
    import fitz as _fitz
    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, width, height), False)
    # set_rect requires a sequence for color; pass a 1-tuple for gray.
    pix.set_rect(pix.irect, (128,))
    try:
        return pix.tobytes("jpeg", jpg_quality=quality)
    except TypeError:
        return pix.tobytes("jpeg")


def _make_rgb_jpeg(width: int = 100, height: int = 100, quality: int = 90) -> bytes:
    """Return RGB JPEG bytes (red fill) for use as an embedded page image."""
    import fitz as _fitz
    pix = _fitz.Pixmap(_fitz.csRGB, (0, 0, width, height), False)
    pix.set_rect(pix.irect, (220, 50, 50))
    try:
        return pix.tobytes("jpeg", jpg_quality=quality)
    except TypeError:
        return pix.tobytes("jpeg")


def _make_gray_png(width: int = 100, height: int = 100) -> bytes:
    """Return grayscale PNG bytes for use as an embedded page image."""
    import fitz as _fitz
    pix = _fitz.Pixmap(_fitz.csGRAY, (0, 0, width, height), False)
    pix.set_rect(pix.irect, (200,))
    return pix.tobytes("png")


def _build_pdf_with_full_page_image(image_bytes: bytes, width: int = 100, height: int = 100) -> "tuple[Any, Any]":
    """Return (doc, page) for a PDF whose first page is entirely *image_bytes*."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=image_bytes)
    return doc, page


def _build_pdf_with_text_only(width: int = 100, height: int = 100) -> "tuple[Any, Any]":
    """Return (doc, page) for a PDF with text only — no embedded images."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_text((10, 50), "No images here — vector only")
    return doc, page


def _build_pdf_with_small_corner_image(
    page_width: int = 200,
    page_height: int = 300,
    image_size: int = 20,
) -> "tuple[Any, Any]":
    """Return (doc, page) where a small image occupies only the top-left corner.

    Coverage = 20×20 / (200×300) ≈ 0.67 % — well below the 90 % threshold.
    """
    import fitz
    gray_jpeg = _make_gray_jpeg(image_size, image_size)
    doc = fitz.open()
    page = doc.new_page(width=page_width, height=page_height)
    corner_rect = fitz.Rect(0, 0, image_size, image_size)
    page.insert_image(corner_rect, stream=gray_jpeg)
    return doc, page


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fast_path_extracts_embedded_jpeg_directly(tmp_path: Path) -> None:
    """A full-page embedded JPEG is returned without any re-encoding.

    The saved bytes must equal what PyMuPDF stores as the embedded stream
    (i.e. no second JPEG quantisation round-trip at our layer).
    """
    pytest.importorskip("fitz")
    src_jpeg = _make_gray_jpeg()
    doc, page = _build_pdf_with_full_page_image(src_jpeg)

    # What PyMuPDF actually stored (our reference — may differ from src_jpeg
    # if PyMuPDF normalised the header, but won't differ from what we extract).
    images = page.get_images(full=True)
    xref = images[0][0]
    ref_bytes = doc.extract_image(xref)["image"]

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is not None, "Fast path should return bytes for a full-page gray JPEG"
    assert result == ref_bytes, (
        "Fast path must return the raw embedded stream — no additional re-encoding"
    )
    doc.close()


def test_fast_path_falls_back_for_vector_pages(tmp_path: Path) -> None:
    """A page with text only (no embedded images) must trigger the fallback path."""
    pytest.importorskip("fitz")
    doc, page = _build_pdf_with_text_only()

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is None, (
        "Vector-only page must return None so the caller falls back to rasterisation"
    )
    doc.close()


def test_fast_path_falls_back_for_partial_image_pages(tmp_path: Path) -> None:
    """A small corner image (< 90 % coverage) must not trigger the fast path."""
    pytest.importorskip("fitz")
    doc, page = _build_pdf_with_small_corner_image(
        page_width=200, page_height=300, image_size=20
    )

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is None, (
        "Partial-coverage image must return None (coverage ≈ 0.67 %% < 90 %%)"
    )
    doc.close()


def test_fast_path_handles_format_mismatch(tmp_path: Path) -> None:
    """Embedded PNG with JPEG output requested → re-encoded to JPEG, not raw PNG."""
    pytest.importorskip("fitz")
    png_bytes = _make_gray_png()
    doc, page = _build_pdf_with_full_page_image(png_bytes)

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is not None, "PNG→JPEG conversion must succeed"
    # JPEG magic bytes: FF D8
    assert result[:2] == b"\xff\xd8", (
        f"Output must be a JPEG (expected FF D8 header), got {result[:4].hex()}"
    )
    # Must NOT be raw PNG bytes (PNG magic: 89 50 4E 47)
    assert result[:4] != b"\x89PNG", "Output must not be raw PNG"
    doc.close()


def test_fast_path_handles_color_to_grayscale_conversion(tmp_path: Path) -> None:
    """Embedded RGB JPEG with grayscale=True → output must be a grayscale JPEG."""
    pytest.importorskip("fitz")
    rgb_jpeg = _make_rgb_jpeg()
    doc, page = _build_pdf_with_full_page_image(rgb_jpeg)

    from webui.services.batches import _try_extract_embedded_page_image
    result = _try_extract_embedded_page_image(
        doc, page, 1,
        dpi=150, grayscale=True, ext=".jpg", jpeg_quality=90,
    )

    assert result is not None, "RGB→grayscale JPEG conversion must succeed"
    # Decode the output with PyMuPDF and check colorspace components.
    import fitz as _fitz
    pix = _fitz.Pixmap(result)
    n_components = pix.n - (1 if pix.alpha else 0)
    assert n_components == 1, (
        f"Expected 1 grayscale component in output, got {n_components} "
        f"(colorspace={pix.colorspace})"
    )
    doc.close()


# ---------------------------------------------------------------------------
# Integration: _save_pdf_pages_serial correctly writes fast-path pages
# ---------------------------------------------------------------------------

def test_serial_fast_path_produces_valid_image_files(tmp_path: Path) -> None:
    """_save_pdf_pages_serial must write a valid image when fast path triggers."""
    pytest.importorskip("fitz")
    import fitz as _fitz

    # Build a 2-page PDF where each page is a full-page grayscale JPEG.
    src_jpeg = _make_gray_jpeg(width=200, height=280)
    doc = _fitz.open()
    for _ in range(2):
        pg = doc.new_page(width=200, height=280)
        pg.insert_image(pg.rect, stream=src_jpeg)
    pdf_bytes = doc.tobytes()
    doc.close()

    inputs = tmp_path / "inputs"
    inputs.mkdir()

    from webui.services.batches import _save_pdf_pages_serial
    stored, failed = _save_pdf_pages_serial(
        inputs=inputs,
        safe_filename="scan.pdf",
        data=pdf_bytes,
        stem="scan",
        page_count=2,
        dpi=150,
        grayscale=True,
        ext=".jpg",
        jpeg_quality=90,
        batch_id=None,
        settings=None,
    )

    assert failed == [], f"No pages should fail; failed={failed}"
    assert len(stored) == 2
    for ref in stored:
        img_path = inputs / ref.name
        assert img_path.exists(), f"Expected {ref.name} on disk"
        # Verify it's a valid JPEG.
        assert img_path.read_bytes()[:2] == b"\xff\xd8", f"{ref.name} is not a JPEG"
