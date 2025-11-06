import cv2
import numpy as np

from config.config_loader import load_config
config = load_config()

from src.constants.image_processing import (
    MIN_PAGE_AREA_THRESHOLD,
    MAX_COSINE_THRESHOLD,
    CANNY_PARAMS,
    DEFAULT_GAUSSIAN_BLUR_KERNEL,
)

class CropOnMarkers:
    def __init__(self, marker_ops=None):
        # Load config or fallback to defaults
        self.min_matching_threshold = marker_ops.get(
            "min_matching_threshold",
            config["omr"].get("bubble_threshold", 0.3)
        ) if marker_ops else config["omr"].get("bubble_threshold", 0.3)

        self.gaussian_blur_kernel = config["preprocessing"].get(
            "blur_kernel",
            DEFAULT_GAUSSIAN_BLUR_KERNEL
        )

        self.canny_params = {
            "threshold1": config["preprocessing"].get("canny_min", CANNY_PARAMS["lower_threshold"]),
            "threshold2": config["preprocessing"].get("canny_max", CANNY_PARAMS["upper_threshold"]),
        }

        self.threshold_circles = []
    def preprocess_image(self, image):
        """ Apply Gaussian Blur and Canny edge detection with config values """
        blurred = cv2.GaussianBlur(image, self.gaussian_blur_kernel, 0)
        edges = cv2.Canny(
            blurred,
            self.canny_params["threshold1"],
            self.canny_params["threshold2"]
        )
        return edges

    def find_marker_matches(self, reference_markers, detected_markers):
        """ Match detected markers with reference markers based on config threshold """
        matched_markers = []

        for ref_marker in reference_markers:
            best_match = None
            best_score = -1

            for det_marker in detected_markers:
                score = self.calculate_marker_similarity(ref_marker, det_marker)

                if score > best_score and score >= self.min_matching_threshold:
                    best_match = det_marker
                    best_score = score

            if best_match is not None:
                matched_markers.append((ref_marker, best_match, best_score))

        return matched_markers

    def calculate_marker_similarity(self, marker1, marker2):
        """ Calculate similarity between markers (dummy implementation to be improved) """
        # Convert to grayscale for comparison
        m1 = cv2.cvtColor(marker1, cv2.COLOR_BGR2GRAY) if len(marker1.shape) == 3 else marker1
        m2 = cv2.cvtColor(marker2, cv2.COLOR_BGR2GRAY) if len(marker2.shape) == 3 else marker2

        if m1.shape != m2.shape:
            return 0

        diff = cv2.absdiff(m1, m2)
        score = 1 - (np.sum(diff) / (m1.size * 255))
        return max(0, min(1, score))  # Clamp between 0 and 1

    def compute_threshold_from_circles(self):
        """ Compute adaptive threshold based on detected circle intensities """
        if not self.threshold_circles:
            return None

        avg_threshold = sum(self.threshold_circles) / len(self.threshold_circles)

        # Store for analysis
        self.threshold_circles.append(avg_threshold)

        # Apply config-based adjustment if enabled
        adjusted_threshold = avg_threshold * config["omr"].get("adaptive_threshold_factor", 1.0)

        return adjusted_threshold
    def process(self, image, reference_markers):
        """
        Main method to crop image based on detected markers.
        1. Preprocess image
        2. Detect markers
        3. Match with reference
        4. Compute threshold
        """

        # Step 1: Preprocess image with config-based values
        edges = self.preprocess_image(image)

        # Step 2: Detect contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_markers = []
        for contour in contours:
            # Threshold based on area
            if cv2.contourArea(contour) < MIN_PAGE_AREA_THRESHOLD:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

            if len(approx) == 4:
                detected_markers.append(approx)

        # Step 3: Match markers
        matches = self.find_marker_matches(reference_markers, detected_markers)

        if not matches:
            print("No valid markers detected based on config thresholds.")
            return None

        # Step 4: Compute threshold if enabled
        computed_threshold = self.compute_threshold_from_circles()

        return {
            "matched_markers": matches,
            "computed_threshold": computed_threshold,
        }
    
