"""
Detects L-shaped corner (anchor) markers and warps the sheet to a top-down
view (issue #270).

Real photographs make faint L-markers hard to find directly (background
texture swamps the thin strokes), so a hybrid strategy is used:

1. Detect the page quadrilateral as a robust anchor (always available).
2. Within an anchor region near each page corner, try to refine the corner
   onto the actual L-marker bracket.
3. Fall back to the page corner when no marker is confidently found.
4. Reuse ImageUtils.four_point_transform for the warp.

This implements the issue requirements (anchor regions, morphology, Canny
+ contours, outer bounding geometry, shared warp) while degrading gracefully
on noisy real-world scans.
"""
import cv2
import numpy as np

from src.constants.image_processing import (
    DEFAULT_CONTOUR_COLOR,
    DEFAULT_CONTOUR_LINE_WIDTH,
)
from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


def normalize(image):
    return cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX)


class CropOnLMarkers(ImagePreprocessor):
    """Crops/warps a sheet using its page corners, refined by L-markers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        marker_ops = self.options
        self.morph_kernel = tuple(int(x) for x in marker_ops.get("morphKernel", [7, 7]))
        self.anchor_ratio = float(marker_ops.get("anchorRatio", 0.18))
        self.min_page_ratio = float(marker_ops.get("minPageRatio", 0.2))

    def __str__(self):
        return "CropOnLMarkers"

    def apply_filter(self, image, file_path):
        config = self.tuning_config

        quad = self.find_page_quad(image)
        if quad is None:
            logger.error(
                f"\tError: could not find page quadrilateral for "
                f"'{file_path}'. Check that the full sheet is visible."
            )
            return None

        ordered = ImageUtils.order_points(quad)
        refined = self.refine_corners(image, ordered)

        if config.outputs.show_image_level >= 4:
            debug = (
                image.copy()
                if len(image.shape) == 3
                else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            )
            cv2.drawContours(
                debug,
                [refined.astype(int)],
                -1,
                DEFAULT_CONTOUR_COLOR,
                DEFAULT_CONTOUR_LINE_WIDTH,
            )
            for x, y in refined:
                cv2.circle(debug, (int(x), int(y)), 12, (0, 0, 255), -1)
            InteractionUtils.show("L-markers", debug, config=config)

        return ImageUtils.four_point_transform(image, refined)

    def find_page_quad(self, image):
        """Return the page four corners as a float32 array, or None."""
        height, width = image.shape[:2]
        blurred = normalize(cv2.GaussianBlur(image, (5, 5), 0))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(closed, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours = ImageUtils.grab_contours(
            cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        min_area = self.min_page_ratio * height * width
        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(contour) > min_area:
                return approx.reshape(4, 2).astype("float32")
        return None

    def refine_corners(self, image, ordered):
        """Nudge each ordered page corner onto its L-marker when found."""
        diag = np.linalg.norm(ordered[2] - ordered[0])
        win = max(120, int(diag * self.anchor_ratio))
        center = ordered.mean(axis=0)
        height, width = image.shape[:2]

        refined = []
        for corner in ordered:
            marker = self.find_marker_in_window(
                image, corner, center, win, width, height
            )
            refined.append(marker if marker is not None else corner)
        return np.array(refined, dtype="float32")

    def find_marker_in_window(self, image, corner, center, win, width, height):
        """Find an L-bracket near a corner; return its outer vertex or None."""
        cx, cy = int(corner[0]), int(corner[1])
        toward = np.sign(center - corner).astype(int)
        x0 = max(0, cx - (win if toward[0] >= 0 else win // 3))
        y0 = max(0, cy - (win if toward[1] >= 0 else win // 3))
        x1 = min(width, cx + (win if toward[0] < 0 else win // 3))
        y1 = min(height, cy + (win if toward[1] < 0 else win // 3))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return None

        patch = image[y0:y1, x0:x1]
        thresh = cv2.adaptiveThreshold(
            cv2.GaussianBlur(patch, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8,
        )
        contours = ImageUtils.grab_contours(
            cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )

        best = None
        best_score = -1.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 0.5 * patch.shape[0] * patch.shape[1]:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 40 or bh < 40:
                continue
            aspect = bw / float(bh)
            fill = area / float(bw * bh)
            if 0.4 < aspect < 2.5 and fill < 0.55:
                score = bw * bh * (1.0 - fill)
                if score > best_score:
                    best_score = score
                    best = (x0 + bx, y0 + by, bw, bh)

        if best is None:
            return None
        return self.bracket_vertex(best, center)

    @staticmethod
    def bracket_vertex(bbox, center):
        """Return the bbox corner furthest from the page centre (the L elbow)."""
        bx, by, bw, bh = bbox
        candidates = [
            (bx, by),
            (bx + bw, by),
            (bx + bw, by + bh),
            (bx, by + bh),
        ]
        return max(
            candidates,
            key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2,
        )
