import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


FIELD_TYPE_VALUES = {
    "QTYPE_MCQ5": ["A", "B", "C", "D", "E"],
    "QTYPE_MCQ4": ["A", "B", "C", "D"],
}


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
        x, y, marker_width, marker_height, area = stats[index]
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


def align_to_reference(image, reference, page_width, page_height):
    source_points = marker_centers(image)
    reference_image = cv2.resize(reference, (page_width, page_height))
    reference_points = marker_centers(reference_image)

    if source_points is None or reference_points is None:
        return cv2.resize(image, (page_width, page_height))

    homography, _mask = cv2.findHomography(source_points, reference_points)
    if homography is None:
        return cv2.resize(image, (page_width, page_height))

    return cv2.warpPerspective(image, homography, (page_width, page_height), borderValue=255)


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


def main():
    image_path = Path(sys.argv[1])
    template_path = Path(sys.argv[2])
    reference_path = Path(sys.argv[3])

    template = json.loads(template_path.read_text(encoding="utf-8"))
    page_width, page_height = template["pageDimensions"]
    image = read_gray(image_path)
    reference = read_gray(reference_path)
    aligned = align_to_reference(image, reference, page_width, page_height)
    normalized = cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX)
    default_bubble_dimensions = template.get("bubbleDimensions", [5, 5])

    responses = {}
    for block in template["fieldBlocks"].values():
        responses.update(read_block(normalized, block, default_bubble_dimensions))

    print(json.dumps(responses))


if __name__ == "__main__":
    main()
