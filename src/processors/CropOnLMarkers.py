import os

import cv2
import numpy as np

from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


class CropOnLMarkers(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options

        # Size of each arm of the L-shape (in pixels on the processing image)
        self.marker_size = int(options.get("marker_size", 30))

        # How thick each arm of the L is
        self.thickness = int(options.get("thickness", 10))

        # Minimum area of a contour to be considered a valid L marker
        self.min_contour_area = int(options.get("min_contour_area", 200))

        # Max area - filters out huge blobs that are not markers
        self.max_contour_area = int(options.get("max_contour_area", 6000))

        # Optional: path to exclude (e.g. a reference image)
        self.exclude_image_path = options.get("excludeImage", None)
        if self.exclude_image_path:
            self.exclude_image_path = os.path.join(
                self.relative_dir, self.exclude_image_path
            )

    @staticmethod
    def exclude_files():
        # No extra reference image needed for L-marker detection
        return []

    def apply_filter(self, image, file_path):
        config = self.tuning_config

        # Step 1: Blur + threshold to get clean black/white image
        gray = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 2: Morphological close - connects broken edges of L-shapes
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        if config.outputs.show_image_level >= 2:
            InteractionUtils.show("CropOnLMarkers: binary", binary, config=config)

        # Step 3: Find all contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 4: Filter contours by area to get only L-marker candidates
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_contour_area < area < self.max_contour_area:
                candidates.append(cnt)

        if len(candidates) < 4:
            logger.warning(
                f"CropOnLMarkers: Found only {len(candidates)} candidates,"  # noqa: E231
                " need 4. Check min/max contour area settings."
            )
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show(
                    "CropOnLMarkers: not enough markers",
                    image,
                    config=config,
                )
            return None

        # Step 5: Find the inner corner (turning point) of each L candidate
        # We get the bounding box and pick the corner closest to image center
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2  # image center

        inner_corners = []
        for cnt in candidates:
            x, y, bw, bh = cv2.boundingRect(cnt)

            # The 4 possible corners of the bounding box
            box_corners = [
                [x, y],  # top-left
                [x + bw, y],  # top-right
                [x, y + bh],  # bottom-left
                [x + bw, y + bh],  # bottom-right
            ]

            # Inner corner = corner of bounding box closest to image center
            inner = min(
                box_corners,
                key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
            )
            inner_corners.append(inner)

        # Step 6: Pick one marker per quadrant (TL, TR, BL, BR)
        quadrant_corners = self._pick_one_per_quadrant(inner_corners, cx, cy)

        if quadrant_corners is None or len(quadrant_corners) != 4:
            logger.warning("CropOnLMarkers: Could not find one marker per quadrant.")
            return None

        logger.info(f"CropOnLMarkers: Detected inner corners: {quadrant_corners}")

        # Step 7: Crop and warp using the 4 inner corners
        centres = np.array(quadrant_corners, dtype="float32")
        image = ImageUtils.four_point_transform(image, centres)

        return image

    def _pick_one_per_quadrant(self, corners, cx, cy):
        """
        From a list of (x, y) points, pick exactly one per quadrant.
        Quadrants: top-left, top-right, bottom-left, bottom-right.
        If multiple points fall in same quadrant, pick the one farthest
        from image center (closest to the actual sheet corner).
        """
        quadrants = {
            "tl": [],  # x < cx, y < cy
            "tr": [],  # x >= cx, y < cy
            "bl": [],  # x < cx, y >= cy
            "br": [],  # x >= cx, y >= cy
        }

        for pt in corners:
            x, y = pt[0], pt[1]
            if x < cx and y < cy:
                quadrants["tl"].append(pt)
            elif x >= cx and y < cy:
                quadrants["tr"].append(pt)
            elif x < cx and y >= cy:
                quadrants["bl"].append(pt)
            else:
                quadrants["br"].append(pt)

        result = []
        for key, pts in quadrants.items():
            if not pts:
                return None
            # Pick the one farthest from center
            best = max(pts, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            result.append(best)

        return result
