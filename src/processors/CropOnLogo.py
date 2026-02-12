import os

import cv2
import numpy as np

from src.constants.image_processing import (
    DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER,
    DEFAULT_NORMALIZE_PARAMS,
    DEFAULT_WHITE_COLOR,
)
from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils


class CropOnLogo(ImagePreprocessor):
    """
    Aligns the image so that a reference logo appears at a fixed position.
    Uses template matching to find the logo, then applies a translation so
    the logo's top-left corner moves to the expected origin. This gives a
    stable reference for the grid defined in the template.

    Logo image requirements:
      - Content: Exact crop of the logo as it appears on the scanned sheets
        (same design, no extra borders). The template is matched as-is.
      - Scale: Same apparent size as on the resized sheet. Sheets are resized
        to processing_width (default 666 px) before matching. If the logo
        occupies e.g. 1/10 of the sheet width, the logo file should be ~66 px
        wide, or set sheetToLogoWidthRatio (sheet_width / logo_width) in options.
      - Format: Any format OpenCV reads (e.g. PNG, JPG). Loaded in grayscale.
      - Orientation: Same as in the scans (no rotation).

    Template usage (in preProcessors):
      {"name": "CropOnLogo", "options": {"relativePath": "logo.png"}}
      {"name": "CropOnLogo", "options": {"relativePath": "logo.png", "expected_origin": [50, 30], "min_matching_threshold": 0.5}}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logo_ops = self.options
        self.logo_path = os.path.join(
            self.relative_dir, logo_ops.get("relativePath", "logo.png")
        )
        self.expected_origin = tuple(
            int(x) for x in logo_ops.get("expected_origin", [0, 0])
        )
        self.min_matching_threshold = float(
            logo_ops.get("min_matching_threshold", 0.4)
        )
        self.logo = self._load_logo(logo_ops)

    def __str__(self):
        return self.logo_path

    def exclude_files(self):
        return [self.logo_path]

    def _load_logo(self, logo_ops):
        """Loads and preprocesses the logo image for template matching."""
        if not os.path.exists(self.logo_path):
            logger.error(
                "Logo not found at path provided in template:",
                self.logo_path,
            )
            exit(31)
        logo = cv2.imread(self.logo_path, cv2.IMREAD_GRAYSCALE)
        if logo is None:
            logger.error("Failed to read logo image:", self.logo_path)
            exit(32)
        config = self.tuning_config
        if "sheetToLogoWidthRatio" in logo_ops:
            logo = ImageUtils.resize_util(
                logo,
                config.dimensions.processing_width
                / int(logo_ops["sheetToLogoWidthRatio"]),
            )
        logo = cv2.GaussianBlur(
            logo,
            DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER["kernel_size"],
            DEFAULT_GAUSSIAN_BLUR_PARAMS_MARKER["sigma_x"],
        )
        logo = cv2.normalize(
            logo,
            None,
            alpha=DEFAULT_NORMALIZE_PARAMS["alpha"],
            beta=DEFAULT_NORMALIZE_PARAMS["beta"],
            norm_type=cv2.NORM_MINMAX,
        )
        return logo

    def _find_logo(self, image):
        """
        Finds the logo in the image via template matching.
        Returns (x, y) of the top-left corner of the best match, or None.
        """
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(image, self.logo, cv2.TM_CCOEFF_NORMED)
        max_val = res.max()
        if max_val < self.min_matching_threshold:
            return None, max_val
        pt = np.argwhere(res == max_val)[0]
        return (int(pt[1]), int(pt[0])), max_val

    def apply_filter(self, image, file_path):
        """
        Detects the logo, then translates the image so the logo's top-left
        is at expected_origin. Keeps image dimensions unchanged; new border
        is filled with white.
        """
        config = self.tuning_config
        img_gray = (
            image
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
        img_norm = ImageUtils.normalize_util(img_gray)
        found_pt, match_val = self._find_logo(img_norm)
        if found_pt is None:
            logger.error(
                "%s Logo not found (match %.3f < %.3f). Check relativePath and expected_origin.",
                file_path,
                match_val,
                self.min_matching_threshold,
            )
            return None
        logger.info(
            "CropOnLogo: found at %s (match %.3f), aligning to %s",
            found_pt,
            match_val,
            self.expected_origin,
        )
        dx = self.expected_origin[0] - found_pt[0]
        dy = self.expected_origin[1] - found_pt[1]
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        h, w = image.shape[:2]
        aligned = cv2.warpAffine(
            image,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(DEFAULT_WHITE_COLOR,) * (3 if image.ndim == 3 else 1),
        )
        return aligned
