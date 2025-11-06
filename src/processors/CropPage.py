import cv2
import numpy as np

from config.config_loader import load_config

# Load config
config = load_config()

from src.constants.image_processing import (
    MIN_PAGE_AREA_THRESHOLD,
    MAX_COSINE_THRESHOLD,
    PAGE_THRESHOLD_PARAMS,
    CANNY_PARAMS,
    DEFAULT_GAUSSIAN_BLUR_KERNEL,
)

class CropPage:
    def __init__(self):
        # Load values from config with fallback to constants
        self.min_page_area_threshold = config["preprocessing"].get("min_page_area_threshold", MIN_PAGE_AREA_THRESHOLD)
        self.max_cosine_threshold = config["preprocessing"].get("max_cosine_threshold", MAX_COSINE_THRESHOLD)

        self.page_threshold_params = {
            "thresh": config["preprocessing"].get("threshold", PAGE_THRESHOLD_PARAMS["threshold_value"]),
            "maxval": PAGE_THRESHOLD_PARAMS["max_pixel_value"],
        }

        self.canny_params = {
            "threshold1": config["preprocessing"].get("canny_min", CANNY_PARAMS["lower_threshold"]),
            "threshold2": config["preprocessing"].get("canny_max", CANNY_PARAMS["upper_threshold"]),
        }

        self.gaussian_blur_kernel = config["preprocessing"].get("blur_kernel", DEFAULT_GAUSSIAN_BLUR_KERNEL)

    def apply_threshold(self, image):
        """ Apply trunc threshold on the page """
        _ret, image = cv2.threshold(
            image,
            self.page_threshold_params["thresh"],
            self.page_threshold_params["maxval"],
            cv2.THRESH_TRUNC,
        )
        return image

    def detect_edges(self, image):
        """ Apply Canny edge detection using config values """
        blurred = cv2.GaussianBlur(image, self.gaussian_blur_kernel, 0)
        edges = cv2.Canny(
            blurred,
            self.canny_params["threshold1"],
            self.canny_params["threshold2"]
        )
        return edges

    def is_valid_contour(self, contour):
        """ Validate contour based on config thresholds """
        if cv2.contourArea(contour) < self.min_page_area_threshold:
            return False

        # Calculate max cosine for contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)

        max_cosine = 0
        if len(approx) == 4:
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                p3 = approx[(i + 2) % 4][0]

                v1 = p1 - p2
                v2 = p3 - p2
                cosine = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                max_cosine = max(max_cosine, cosine)

        return max_cosine < self.max_cosine_threshold
