import json
import re
import sys
from pathlib import Path

import cv2

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None


PAGE_WIDTH = 547
PAGE_HEIGHT = 834


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
        if text and text in allowed:
            chars.append(text)
        else:
            chars.append("")
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


def extract_metadata(image_path):
    image = cv2.imread(str(image_path))
    if image is None or RapidOCR is None:
        return {
            "applicationNumber": "",
            "lrn": "",
            "examDate": "",
        }

    ocr = RapidOCR()

    app_boxes = [
        (290.0, 64.2, 11.6, 13.5),
        (302.6, 64.2, 11.6, 13.5),
        (315.2, 64.7, 11.6, 13.0),
        (327.3, 64.7, 11.6, 13.0),
        (339.4, 64.7, 11.6, 13.5),
        (351.9, 64.7, 12.1, 14.0),
        (364.5, 65.2, 11.6, 13.5),
        (376.6, 65.7, 11.6, 13.5),
        (388.7, 65.7, 11.6, 13.5),
        (400.8, 65.7, 12.6, 13.5),
        (413.9, 65.2, 12.1, 14.0),
        (426.4, 64.7, 11.6, 14.0),
        (439.0, 64.7, 11.6, 14.0),
        (451.1, 64.2, 12.6, 14.0),
        (464.6, 63.7, 11.6, 14.0),
        (476.7, 63.7, 11.6, 13.5),
    ]
    lrn_boxes = [
        (290.0, 91.8, 15.8, 14.0),
        (306.8, 91.8, 15.8, 14.0),
        (323.6, 91.8, 15.8, 14.0),
        (339.8, 92.3, 15.8, 14.0),
        (356.6, 92.8, 15.8, 13.5),
        (372.9, 93.3, 15.8, 13.5),
        (389.2, 93.3, 15.8, 13.5),
        (405.5, 93.3, 16.3, 13.5),
        (422.2, 93.3, 15.8, 14.0),
        (438.5, 92.8, 16.3, 14.0),
        (455.8, 92.3, 15.8, 14.0),
        (472.5, 92.3, 15.8, 14.0),
    ]
    date_boxes = [
        (290.0, 119.4, 23.3, 14.0),
        (313.8, 119.4, 22.8, 14.0),
        (342.8, 119.9, 22.8, 14.0),
        (366.0, 119.9, 22.8, 13.5),
        (394.8, 120.4, 22.8, 13.5),
        (418.5, 120.4, 22.8, 13.5),
        (441.8, 120.4, 22.8, 13.5),
        (465.1, 120.4, 23.3, 13.0),
    ]

    app_chars = read_cells(ocr, image, app_boxes, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    lrn_chars = read_cells(ocr, image, lrn_boxes, "0123456789")
    date_chars = read_cells(ocr, image, date_boxes, "0123456789")

    return {
        "applicationNumber": join_application(app_chars),
        "lrn": "".join(lrn_chars),
        "examDate": join_date(date_chars),
    }


if __name__ == "__main__":
    metadata = extract_metadata(Path(sys.argv[1]))
    print(json.dumps(metadata))
