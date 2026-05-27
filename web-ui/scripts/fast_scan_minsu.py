import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:
    import fitz
except Exception:
    fitz = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None


FIELD_TYPE_VALUES = {
    "QTYPE_MCQ5": ["A", "B", "C", "D", "E"],
    "QTYPE_MCQ4": ["A", "B", "C", "D"],
}

PAGE_WIDTH = 547
PAGE_HEIGHT = 834
PDF_RENDER_SCALE = float(os.environ.get("OMR_PDF_RENDER_SCALE", "1.5"))
ENABLE_LRN_CELL_FALLBACK = os.environ.get("OMR_LRN_CELL_FALLBACK") == "1"
MAX_WORKERS = max(1, int(os.environ.get("OMR_SCAN_WORKERS", str(min(os.cpu_count() or 1, 4)))))
WORKER_TEMPLATE = None
WORKER_REFERENCE = None
WORKER_OCR = None


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def marker_centers(image):
    _, threshold = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, 8)
    height, width = image.shape[:2]
    candidates = []

    for index in range(1, count):
        _x, _y, marker_width, marker_height, area = stats[index]
        if not 20 <= area <= 3000:
            continue
        if marker_width < 4 or marker_height < 4:
            continue
        ratio = marker_width / max(marker_height, 1)
        if 0.5 <= ratio <= 1.8:
            candidates.append(centroids[index])

    if len(candidates) < 4:
        return None

    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    return np.array(
        [
            min(candidates, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2)
            for x, y in corners
        ],
        dtype=np.float32,
    )


def reference_homography(image, reference, page_width, page_height):
    source_points = marker_centers(image)
    reference_image = cv2.resize(reference, (page_width, page_height))
    reference_points = marker_centers(reference_image)

    if source_points is None or reference_points is None:
        return None

    homography, _mask = cv2.findHomography(source_points, reference_points)
    return homography


def align_to_reference(image, reference, page_width, page_height):
    homography = reference_homography(image, reference, page_width, page_height)
    if homography is None:
        return cv2.resize(image, (page_width, page_height))
    return cv2.warpPerspective(image, homography, (page_width, page_height), borderValue=255)


def align_gray_and_color(gray, color, reference, page_width, page_height):
    homography = reference_homography(gray, reference, page_width, page_height)
    if homography is None:
        aligned_gray = cv2.resize(gray, (page_width, page_height))
        aligned_color = cv2.resize(color, (page_width, page_height)) if color is not None else None
        return aligned_gray, aligned_color

    aligned_gray = cv2.warpPerspective(gray, homography, (page_width, page_height), borderValue=255)
    aligned_color = None
    if color is not None:
        aligned_color = cv2.warpPerspective(color, homography, (page_width, page_height), borderValue=(255, 255, 255))
    return aligned_gray, aligned_color


def parse_field_labels(field_labels):
    labels = []
    for field_label in field_labels:
        match = re.fullmatch(r"([A-Za-z_]+)(\d+)\.\.(\d+)", field_label)
        if match:
            prefix, start, end = match.groups()
            labels.extend(f"{prefix}{index}" for index in range(int(start), int(end) + 1))
        else:
            labels.append(field_label)
    return labels


def sample_bubble_darkness(image, center_x, center_y):
    radius = 4
    x1 = max(0, int(round(center_x - radius)))
    x2 = min(image.shape[1], int(round(center_x + radius + 1)))
    y1 = max(0, int(round(center_y - radius)))
    y2 = min(image.shape[0], int(round(center_y + radius + 1)))
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float((255 - patch).mean())


def read_block(image, block, default_bubble_dimensions):
    bubble_values = FIELD_TYPE_VALUES.get(block.get("fieldType"), block.get("bubbleValues", []))
    if not bubble_values:
        return {}

    origin_x, origin_y = block["origin"]
    bubble_width, bubble_height = block.get("bubbleDimensions", default_bubble_dimensions)
    bubbles_gap = float(block["bubblesGap"])
    labels_gap = float(block["labelsGap"])
    labels = parse_field_labels(block["fieldLabels"])
    responses = {}

    for row_index, label in enumerate(labels):
        center_y = origin_y + row_index * labels_gap + bubble_height / 2
        scores = []
        for column_index, value in enumerate(bubble_values):
            center_x = origin_x + column_index * bubbles_gap + bubble_width / 2
            scores.append((value, sample_bubble_darkness(image, center_x, center_y)))

        ranked = sorted(scores, key=lambda item: item[1], reverse=True)
        best_value, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        responses[label] = best_value if best_score >= 45 and best_score - second_score >= 8 else ""

    return responses


def scaled_box(box, image_width, image_height):
    x, y, w, h = box
    sx = image_width / PAGE_WIDTH
    sy = image_height / PAGE_HEIGHT
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    )


def ocr_cell(ocr, image, box):
    if ocr is None:
        return ""

    x, y, w, h = box
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return ""

    crop = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    result, _ = ocr(crop)
    if not result:
        return ""

    best = max(result, key=lambda item: item[2])
    text = re.sub(r"[^A-Za-z0-9]", "", best[1]).upper()
    return text[:1]


def read_cells(ocr, image, normalized_boxes, allowed):
    height, width = image.shape[:2]
    chars = []
    for box in normalized_boxes:
        text = ocr_cell(ocr, image, scaled_box(box, width, height))
        chars.append(text if text and text in allowed else "")
    return chars


def join_application(chars):
    prefix = "".join(chars[:3])
    year = "".join(chars[4:8])
    serial = "".join(chars[9:14])
    parts = [part for part in [prefix, year, serial] if part]
    return "-".join(parts)


def join_date(chars):
    digits = "".join(chars)
    if len(digits) >= 8:
        return f"{digits[:2]}-{digits[2:4]}-{digits[4:8]}"
    return digits


LRN_BOXES = [
    (274.0 + index * 22.85, 86.0, 22.0, 22.0)
    for index in range(12)
]

LRN_STRIP_BOX = (286.0, 88.0, 205.0, 23.0)
DIGIT_TRANSLATION = str.maketrans({ord(chr(0xFF10 + index)): str(index) for index in range(10)})


def normalize_digits(text):
    return re.sub(r"[^0-9]", "", text.translate(DIGIT_TRANSLATION))


def compose_lrn_digit_line(color_image):
    height, width = color_image.shape[:2]
    parts = []

    for box in LRN_BOXES:
        x, y, w, h = scaled_box(box, width, height)
        crop = color_image[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        inner = crop[5:18, 5:17]
        if inner.size == 0:
            return None

        inner = cv2.resize(inner, (42, 48), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        parts.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    spacer = np.full((48, 10, 3), 255, np.uint8)
    line = parts[0]
    for part in parts[1:]:
        line = np.hstack([line, spacer, part])
    return line


def read_lrn(ocr, color_image):
    if color_image is None or ocr is None:
        return ""

    digit_line = compose_lrn_digit_line(color_image)
    if digit_line is not None:
        result, _ = ocr(digit_line, use_det=False, use_cls=False, use_rec=True)
        if result:
            text = "".join(item[0] if isinstance(item[0], str) else item[1] for item in result)
            digits = normalize_digits(text)
            if len(digits) >= 12:
                return digits[:12]

    height, width = color_image.shape[:2]
    x, y, w, h = scaled_box(LRN_STRIP_BOX, width, height)
    crop = color_image[y : y + h, x : x + w]
    if crop.size:
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result, _ = ocr(crop)
        if result:
            text = "".join(item[1] for item in result)
            digits = normalize_digits(text)
            if len(digits) >= 12:
                return digits[:12]
            if digits and not ENABLE_LRN_CELL_FALLBACK:
                return digits

    if not ENABLE_LRN_CELL_FALLBACK:
        return ""
    return "".join(read_cells(ocr, color_image, LRN_BOXES, "0123456789"))


def build_subject(values, prefix, count):
    return "".join(values.get(f"{prefix}{index + 1}", "") for index in range(count))


def count_responses(values, prefix, count):
    return sum(1 for index in range(count) if values.get(f"{prefix}{index + 1}", ""))


def scan_image_arrays(gray, color, file_name, source_file_name, template, reference, ocr):
    page_width, page_height = template["pageDimensions"]
    aligned, aligned_color = align_gray_and_color(gray, color, reference, page_width, page_height)
    normalized = cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX)
    default_bubble_dimensions = template.get("bubbleDimensions", [5, 5])

    responses = {}
    for block in template["fieldBlocks"].values():
        responses.update(read_block(normalized, block, default_bubble_dimensions))

    lrn = read_lrn(ocr, aligned_color)

    return {
        "applicationNumber": "",
        "lrn": lrn,
        "surname": "",
        "name": "",
        "middleName": "",
        "examDate": "",
        "sourceFileName": source_file_name,
        "languageProficiency": build_subject(responses, "L", 35),
        "mathematics": build_subject(responses, "M", 35),
        "science": build_subject(responses, "S", 35),
        "logicAndAbstractReasoning": build_subject(responses, "LA", 20),
        "generalKnowledge": build_subject(responses, "G", 35),
        "mechanicalReasoning": build_subject(responses, "ME", 20),
        "languageProficiencyDetected": count_responses(responses, "L", 35),
        "mathematicsDetected": count_responses(responses, "M", 35),
        "scienceDetected": count_responses(responses, "S", 35),
        "logicAndAbstractReasoningDetected": count_responses(responses, "LA", 20),
        "generalKnowledgeDetected": count_responses(responses, "G", 35),
        "mechanicalReasoningDetected": count_responses(responses, "ME", 20),
        "fileName": file_name,
        "checkedImagePath": "",
    }


def pixmap_to_images(pixmap):
    channels = pixmap.n
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, channels)
    if channels == 4:
        rgb = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    elif channels == 3:
        rgb = array
    else:
        gray = array[:, :, 0] if channels > 1 else array
        return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def scan_entry(entry, template, reference, ocr):
    if "pdfPath" in entry:
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for fast PDF scanning.")

        document = fitz.open(entry["pdfPath"])
        results = []
        base_name = entry["fileName"]
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE), alpha=False)
            gray, color = pixmap_to_images(pixmap)
            page_name = f"{base_name}-page-{index + 1}.png"
            results.append(scan_image_arrays(gray, color, page_name, entry["sourceFileName"], template, reference, ocr))
        return results

    image_path = Path(entry["imagePath"])
    return [
        scan_image_arrays(
            read_gray(image_path),
            cv2.imread(str(image_path)),
            entry["fileName"],
            entry.get("sourceFileName", entry["fileName"]),
            template,
            reference,
            ocr,
        )
    ]


def init_worker(template_path, reference_path):
    global WORKER_TEMPLATE, WORKER_REFERENCE, WORKER_OCR
    WORKER_TEMPLATE = json.loads(Path(template_path).read_text(encoding="utf-8"))
    WORKER_REFERENCE = read_gray(Path(reference_path))
    WORKER_OCR = RapidOCR() if RapidOCR is not None else None


def scan_task(task):
    kind = task["kind"]
    if kind == "pdf-page":
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for fast PDF scanning.")

        document = fitz.open(task["pdfPath"])
        page = document[task["pageIndex"]]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE), alpha=False)
        gray, color = pixmap_to_images(pixmap)
        return scan_image_arrays(
            gray,
            color,
            task["fileName"],
            task["sourceFileName"],
            WORKER_TEMPLATE,
            WORKER_REFERENCE,
            WORKER_OCR,
        )

    image_path = Path(task["imagePath"])
    return scan_image_arrays(
        read_gray(image_path),
        cv2.imread(str(image_path)),
        task["fileName"],
        task["sourceFileName"],
        WORKER_TEMPLATE,
        WORKER_REFERENCE,
        WORKER_OCR,
    )


def build_tasks(entries):
    tasks = []
    for entry in entries:
        if "pdfPath" in entry:
            if fitz is None:
                raise RuntimeError("PyMuPDF is required for fast PDF scanning.")

            document = fitz.open(entry["pdfPath"])
            base_name = entry["fileName"]
            for index in range(document.page_count):
                tasks.append(
                    {
                        "kind": "pdf-page",
                        "pdfPath": entry["pdfPath"],
                        "fileName": f"{base_name}-page-{index + 1}.png",
                        "sourceFileName": entry["sourceFileName"],
                        "pageIndex": index,
                    }
                )
            continue

        tasks.append(
            {
                "kind": "image",
                "imagePath": entry["imagePath"],
                "fileName": entry["fileName"],
                "sourceFileName": entry.get("sourceFileName", entry["fileName"]),
            }
        )
    return tasks


def main():
    manifest_path = Path(sys.argv[1])
    template_path = Path(sys.argv[2])
    reference_path = Path(sys.argv[3])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    tasks = build_tasks(manifest["images"])

    if MAX_WORKERS == 1 or len(tasks) <= 1:
        init_worker(template_path, reference_path)
        results = [scan_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=min(MAX_WORKERS, len(tasks)),
            initializer=init_worker,
            initargs=(template_path, reference_path),
        ) as executor:
            results = list(executor.map(scan_task, tasks))
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
