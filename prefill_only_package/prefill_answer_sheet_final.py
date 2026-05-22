#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import logging
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_TEMPLATE = Path('blank_template_reference.png')

# Relative coordinates for the current landscape blank template.
CFG = {
    'student_name_box': (0.040, 0.178, 0.245, 0.288),
    'school_name_box': (0.262, 0.178, 0.430, 0.288),
    'exam_name_box': (0.438, 0.178, 0.600, 0.330),

    'candidate_grid': (0.636, 0.121, 0.962, 0.400),  # x1, y1, x2, y2
    'candidate_header_band': (0.121, 0.200),         # y1, y2 relative to image
    'candidate_bubble_top': 0.2094,
    'candidate_bubble_step': 0.01985,

    # Text sizes (approximate; auto-reduced when needed)
    'student_start_size': 28,
    'school_start_size': 26,
    'exam_start_size': 25,
    'header_digit_size': 18,

    'header_digit_y_offset': 22,
    'bubble_row_y_offsets': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


_font_cache: dict = {}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-worker-process stamped-template cache (set by _init_worker_stamped).
# Using a process-level global avoids shipping the 1.8 MB template PNG through
# the IPC pipe on every task.  The initializer loads it once per worker.
# ---------------------------------------------------------------------------
_WORKER_STAMPED_ARR: 'np.ndarray | None' = None
_WORKER_LAYOUT: 'dict | None' = None  # pre-computed pixel geometry for fixed template size


def load_font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, size)
                _font_cache[key] = font
                return font
            except Exception:
                pass
    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def wrap_and_draw(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], start_size: int) -> None:
    x1, y1, x2, y2 = box
    words = text.split()
    max_width = x2 - x1
    max_height = y2 - y1
    for size in range(start_size, 11, -1):
        font = load_font(size)
        line_h = draw.textbbox((0, 0), 'Ag', font=font)[3]
        lines: list[str] = []
        current = ''
        for word in words:
            trial = word if not current else current + ' ' + word
            width = draw.textbbox((0, 0), trial, font=font)[2]
            if width <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * 4
        if total_h <= max_height:
            y = y1
            for line in lines:
                draw.text((x1, y), line, fill='black', font=font)
                y += line_h + 4
            return
    draw.text((x1, y1), text, fill='black', font=load_font(12))


def relative_box(rel_box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rel_box
    return round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)


# ArUco dictionary and ID convention — must match template.json.
_ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
# Corner order: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
_ARUCO_CORNER_IDS = [0, 1, 2, 3]
# Reference marker centres as fractions of the processing canvas (666 x 515).
# These mirror referenceMarkerCenters in template.json.
_REF_CENTERS_RELATIVE = [
    (13.5 / 666, 13.2 / 515),   # TL
    (651.5 / 666, 13.2 / 515),  # TR
    (13.5 / 666, 499.0 / 515),  # BL
    (651.5 / 666, 499.0 / 515), # BR
]
# Marker size relative to the longer image dimension (landscape).
# On the 1426×1103 reference template, this gives ~43 px; on the 666×515
# processing canvas it gives ~24 px — both large enough for reliable ArUco
# detection with the 60 px white-border padding used by the scanner.
# Minimum is clamped to 24 px.
_ARUCO_MARKER_SIZE_RATIO = 0.03


def draw_aruco_corners(img: Image.Image) -> Image.Image:
    """Stamp 4 ArUco fiducial markers at the corners of *img* and return the
    modified image.  The markers use ``DICT_4X4_50`` with IDs 0-3 corresponding
    to top-left, top-right, bottom-left, bottom-right.

    If the underlying template already has printed square markers at the
    corners, a white rectangle is drawn first to erase them before the ArUco
    pattern is stamped, so old ink does not interfere with detection.
    """
    img_cv = np.array(img.convert("RGB"))[:, :, ::-1].copy()  # PIL→BGR
    h, w = img_cv.shape[:2]

    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_ID)
    marker_px = max(24, int(max(w, h) * _ARUCO_MARKER_SIZE_RATIO))
    # ArUco requires a white quiet zone on all sides.
    quiet_zone = max(6, marker_px // 8)

    for corner_idx, marker_id in enumerate(_ARUCO_CORNER_IDS):
        rel_cx, rel_cy = _REF_CENTERS_RELATIVE[corner_idx]
        cx, cy = round(rel_cx * w), round(rel_cy * h)
        half = marker_px // 2
        # Clamp so the full marker fits and has quiet zone on all sides.
        x0 = max(quiet_zone, min(cx - half, w - marker_px - quiet_zone))
        y0 = max(quiet_zone, min(cy - half, h - marker_px - quiet_zone))
        x1 = x0 + marker_px
        y1 = y0 + marker_px
        if x1 > w or y1 > h or marker_px < 8:
            continue
        # Erase any existing printed marker in this region with white.
        img_cv[y0:y1, x0:x1] = 255
        # Generate ArUco marker at the required size.
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        # Convert to BGR for placement in the colour image.
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        img_cv[y0:y1, x0:x1] = marker_bgr

    # BGR→RGB, then back to PIL
    return Image.fromarray(img_cv[:, :, ::-1])


def aruco_marker_boxes(w: int, h: int) -> list[dict]:
    """Return pixel-space ArUco marker boxes for a rendered sheet.

    Kept in sync with :func:`draw_aruco_corners` so downstream scan
    simulation can occlude / blur markers intentionally for robustness
    testing without duplicating the placement math.
    """
    marker_px = max(24, int(max(w, h) * _ARUCO_MARKER_SIZE_RATIO))
    quiet_zone = max(6, marker_px // 8)
    boxes = []
    for corner_idx, marker_id in enumerate(_ARUCO_CORNER_IDS):
        rel_cx, rel_cy = _REF_CENTERS_RELATIVE[corner_idx]
        cx, cy = round(rel_cx * w), round(rel_cy * h)
        half = marker_px // 2
        x0 = max(quiet_zone, min(cx - half, w - marker_px - quiet_zone))
        y0 = max(quiet_zone, min(cy - half, h - marker_px - quiet_zone))
        boxes.append({
            'corner': corner_idx,
            'marker_id': marker_id,
            'x0': x0,
            'y0': y0,
            'x1': x0 + marker_px,
            'y1': y0 + marker_px,
        })
    return boxes


def load_stamped_template(template_path: Path) -> Image.Image:
    """Open the template and stamp ArUco corners once.

    Call this once before a batch loop and pass copies of the result to
    individual sheet workers instead of re-stamping on every sheet.
    """
    img = Image.open(template_path).convert('RGB')
    return draw_aruco_corners(img)


def _build_layout(w: int, h: int) -> dict:
    """Pre-compute all pixel-space layout values for a template of size (w, h).
    Cached once per process so repeated calls for the same template size are free.
    """
    gx1, gy1, gx2, gy2 = relative_box(CFG['candidate_grid'], w, h)
    grid_w = gx2 - gx1
    header_font = load_font(CFG['header_digit_size'], bold=True)
    # Pre-compute digit text extents for 0-9 using a throw-away ImageDraw.
    _tmp = Image.new('RGB', (1, 1))
    _drw = ImageDraw.Draw(_tmp)
    digit_sizes = {
        str(d): _drw.textbbox((0, 0), str(d), font=header_font)[2:4]
        for d in range(10)
    }
    return {
        'student_box': relative_box(CFG['student_name_box'], w, h),
        'school_box': relative_box(CFG['school_name_box'], w, h),
        'exam_box': relative_box(CFG['exam_name_box'], w, h),
        'col_centers': [round(gx1 + (i + 0.5) * grid_w / 10) for i in range(10)],
        'header_y': round(((CFG['candidate_header_band'][0] + CFG['candidate_header_band'][1]) / 2) * h),
        'row_centers': [round((CFG['candidate_bubble_top'] + i * CFG['candidate_bubble_step']) * h) for i in range(10)],
        'header_font': header_font,
        'digit_sizes': digit_sizes,
        'radius': max(8, round(h * 0.009)),
        'y_offset': CFG['header_digit_y_offset'],
    }


_LAYOUT_CACHE: dict[tuple[int, int], dict] = {}


def _get_layout(w: int, h: int) -> dict:
    key = (w, h)
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = _build_layout(w, h)
    return _LAYOUT_CACHE[key]


def candidate_bubble_geometry(
    w: int,
    h: int,
    candidate_number: str | None = None,
) -> list[dict]:
    """Return pixel-space geometry for all candidate-number bubbles.

    Ordering is column-major: candidate digit column 0..9, then digit row
    0..9. If ``candidate_number`` is provided, the selected row for each
    column is marked with ``filled=True``.
    """
    L = _get_layout(w, h)
    filled_by_column: dict[int, int] = {}
    if candidate_number:
        for idx, digit in enumerate(candidate_number[:10]):
            if digit.isdigit():
                filled_by_column[idx] = int(digit)

    bubbles: list[dict] = []
    for column, cx in enumerate(L['col_centers']):
        for digit, cy in enumerate(L['row_centers']):
            bubbles.append({
                'column': column,
                'digit': digit,
                'cx': cx,
                'cy': cy,
                'radius': L['radius'],
                'filled': filled_by_column.get(column) == digit,
            })
    return bubbles


def _draw_sheet_content(
    img: Image.Image,
    student_name: str,
    school_name: str,
    exam_name: str,
    candidate_number: str,
) -> Image.Image:
    """Draw text fields and candidate bubbles onto *img* (already ArUco-stamped)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    L = _get_layout(w, h)

    wrap_and_draw(draw, student_name, L['student_box'], CFG['student_start_size'])
    wrap_and_draw(draw, school_name, L['school_box'], CFG['school_start_size'])
    wrap_and_draw(draw, exam_name, L['exam_box'], CFG['exam_start_size'])

    col_centers = L['col_centers']
    header_y = L['header_y']
    row_centers = L['row_centers']
    header_font = L['header_font']
    digit_sizes = L['digit_sizes']
    radius = L['radius']
    y_offset = L['y_offset']

    for idx, digit in enumerate(candidate_number):
        cx = col_centers[idx]
        tw, th = digit_sizes[digit]
        draw.text(
            (cx - tw / 2, header_y - th / 2 - 1 + y_offset),
            digit,
            fill='black',
            font=header_font,
        )
        cy = row_centers[int(digit)]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill='black')

    return img


def _prefill_worker(payload: dict) -> bytes:
    """ProcessPoolExecutor worker: render one sheet from pre-stamped template bytes.

    Expected payload keys: stamped_bytes, student_name, school_name, exam_name,
    candidate_number.  Returns PNG bytes with fast (level-1) compression.
    """
    img = Image.open(io.BytesIO(payload['stamped_bytes'])).convert('RGB')
    img = _draw_sheet_content(
        img,
        payload['student_name'],
        payload['school_name'],
        payload['exam_name'],
        payload['candidate_number'],
    )
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fast worker path — zero IPC overhead for the template.
# The ProcessPoolExecutor is created with initializer=_init_worker_stamped so
# each worker process decodes the stamped PIL image once at startup.  Task
# payloads then only carry the small per-student text fields.
# ---------------------------------------------------------------------------

def _init_worker_stamped(arr_bytes: bytes, shape: tuple) -> None:
    """Initializer for fast worker pool: decode stamped template once per process."""
    global _WORKER_STAMPED_ARR, _WORKER_LAYOUT
    arr = np.frombuffer(arr_bytes, dtype=np.uint8).reshape(shape).copy()
    _WORKER_STAMPED_ARR = arr
    # Pre-warm the layout cache for this template size.
    h, w = shape[:2]
    _WORKER_LAYOUT = _get_layout(w, h)


def _prefill_worker_fast(payload: dict) -> bytes:
    """Like _prefill_worker but uses the worker-global pre-decoded template.

    Payload keys: student_name, school_name, exam_name, candidate_number.
    Optional keys:
      output_format: 'png' (default) | 'jpeg'
      jpeg_quality:  int 1-95 (default 85), only used when output_format='jpeg'
    No stamped_bytes key needed — template lives in _WORKER_STAMPED_ARR.
    """
    img = Image.fromarray(_WORKER_STAMPED_ARR.copy())
    img = _draw_sheet_content(
        img,
        payload['student_name'],
        payload['school_name'],
        payload['exam_name'],
        payload['candidate_number'],
    )
    buf = io.BytesIO()
    fmt = payload.get('output_format', 'png').lower()
    if fmt == 'jpeg':
        img.save(buf, format='JPEG', quality=payload.get('jpeg_quality', 75))
    else:
        img.save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def prefill_sheet(template_path: Path, student_name: str, school_name: str, exam_name: str, candidate_number: str) -> Image.Image:
    logger.debug("prefill_sheet | candidate=%s", candidate_number)
    if len(candidate_number) != 10 or not candidate_number.isdigit():
        raise ValueError('Candidate number must be exactly 10 digits.')
    img = Image.open(template_path).convert('RGB')
    img = draw_aruco_corners(img)
    return _draw_sheet_content(img, student_name, school_name, exam_name, candidate_number)


def save_pdf(images: Iterable[Image.Image], output_path: Path) -> None:
    imgs = [im.convert('RGB') for im in images]
    if not imgs:
        raise ValueError('No images to save.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(output_path, format='PDF', save_all=True, append_images=imgs[1:], resolution=300.0)


def save_pdf_fast(png_bytes_list: list, output_path: Path, chunk_size: int = 500) -> None:
    """Assemble a list of PNG byte strings into a PDF using PyMuPDF.

    Uses insert_image() to embed PNGs directly (no re-encoding) and saves
    without deflate re-compression (PNG data is already zlib-compressed).
    Processes pages in chunks to keep peak RAM bounded.
    """
    import fitz  # PyMuPDF
    import struct
    import tempfile

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _png_dims(data: bytes) -> tuple[int, int]:
        """Read width/height from PNG IHDR chunk (bytes 16-23) without decoding."""
        return struct.unpack('>II', data[16:24])

    def _build_chunk_doc(chunk: list) -> 'fitz.Document':
        doc = fitz.open()
        for data in chunk:
            w, h = _png_dims(data)
            page = doc.new_page(width=w, height=h)
            page.insert_image(page.rect, stream=data)
        return doc

    if len(png_bytes_list) <= chunk_size:
        doc = _build_chunk_doc(png_bytes_list)
        doc.save(str(output_path), garbage=0)
        doc.close()
        return

    # Large page count — write chunks to temp files then merge.
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_paths: list[str] = []
        for chunk_start in range(0, len(png_bytes_list), chunk_size):
            chunk = png_bytes_list[chunk_start:chunk_start + chunk_size]
            chunk_path = str(Path(tmpdir) / f'chunk_{chunk_start:06d}.pdf')
            chunk_doc = _build_chunk_doc(chunk)
            chunk_doc.save(chunk_path, garbage=0)
            chunk_doc.close()
            chunk_paths.append(chunk_path)
            print(f'  PDF chunk {chunk_start + len(chunk)}/{len(png_bytes_list)} assembled')

        # Merge all chunk PDFs into the final file.
        final_doc = fitz.open()
        for chunk_path in chunk_paths:
            src = fitz.open(chunk_path)
            final_doc.insert_pdf(src)
            src.close()
        final_doc.save(str(output_path), garbage=0)
        final_doc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prefill the current landscape multiple choice answer sheet template.')
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--student-name', type=str)
    parser.add_argument('--school-name', type=str)
    parser.add_argument('--exam-name', type=str)
    parser.add_argument('--candidate-number', type=str)
    parser.add_argument('--output', type=Path, help='Single-sheet output PNG or PDF')
    parser.add_argument('--csv', type=Path, help='Batch CSV with columns: student_name,school_name,exam_name,candidate_number[,output_file]')
    parser.add_argument('--output-dir', type=Path, help='Directory for individual PNGs in batch mode')
    parser.add_argument('--combined-pdf', type=Path, help='Combined PDF in batch mode')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        import time
        from concurrent.futures import ProcessPoolExecutor

        rows = list(csv.DictReader(args.csv.open('r', newline='', encoding='utf-8-sig')))
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Prefill batch starting | count=%d template=%s", len(rows), args.template)
        # Pre-stamp template ONCE — ArUco stamping is the biggest per-sheet cost.
        t_stamp = time.perf_counter()
        stamped_img = load_stamped_template(args.template)
        stamped_buf = io.BytesIO()
        stamped_img.save(stamped_buf, format='PNG', compress_level=1)
        stamped_bytes = stamped_buf.getvalue()
        logger.info("Template ArUco-stamped | elapsed=%.1fs", time.perf_counter() - t_stamp)

        payloads = [
            {
                'stamped_bytes': stamped_bytes,
                'student_name': row['student_name'].strip(),
                'school_name': row['school_name'].strip(),
                'exam_name': row['exam_name'].strip(),
                'candidate_number': str(row['candidate_number']).strip(),
            }
            for row in rows
        ]

        max_workers = max(1, min((os.cpu_count() or 2) - 1, 8))
        print(f'Generating {len(rows)} sheets with {max_workers} workers...')
        t0 = time.perf_counter()

        png_bytes_list: list[bytes] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, png_bytes in enumerate(
                executor.map(_prefill_worker, payloads, chunksize=20), start=1
            ):
                png_bytes_list.append(png_bytes)
                if i % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (i / elapsed) * 60 if elapsed > 0 else 0
                    print(f'  {i}/{len(rows)} done ({elapsed:.1f}s, {i / elapsed:.1f} sheets/s)')
                    logger.info("Prefill progress | %d/%d sheets done | elapsed=%.1fs | rate=%.0f sheets/min", i, len(rows), elapsed, rate)

        elapsed = time.perf_counter() - t0
        rate = (len(rows) / elapsed) * 60 if elapsed > 0 else 0
        print(f'Generated {len(rows)} sheets in {elapsed:.1f}s ({len(rows) / elapsed:.1f} sheets/s)')
        logger.info("Prefill batch complete | count=%d | elapsed=%.1fs | rate=%.0f sheets/min | output=%s", len(rows), elapsed, rate, args.combined_pdf or args.output or args.output_dir)

        if args.output_dir:
            for i, (row, png_bytes) in enumerate(zip(rows, png_bytes_list), start=0):
                filename = (row.get('output_file') or f'sheet_{i:04d}.png').strip()
                (args.output_dir / filename).write_bytes(png_bytes)

        if args.combined_pdf:
            save_pdf_fast(png_bytes_list, args.combined_pdf)
            print(f'Created {args.combined_pdf}')
        elif args.output:
            save_pdf_fast(png_bytes_list, args.output)
            print(f'Created {args.output}')
        elif not args.output_dir:
            raise SystemExit('For batch mode, provide --combined-pdf, --output, or --output-dir.')
        return

    required = [args.student_name, args.school_name, args.exam_name, args.candidate_number, args.output]
    if any(v is None for v in required):
        raise SystemExit('For a single sheet, provide --student-name, --school-name, --exam-name, --candidate-number, and --output.')

    image = prefill_sheet(args.template, args.student_name, args.school_name, args.exam_name, args.candidate_number)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.pdf':
        save_pdf([image], args.output)
    else:
        image.save(args.output)
    print(f'Created {args.output}')


if __name__ == '__main__':
    main()
