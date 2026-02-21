"""
Image Warping Strategies

Extracted from WarpOnPointsCommon to provide focused, testable
implementations of different warping/transformation methods.

Each strategy encapsulates a specific approach to transforming images
based on control and destination points.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import cv2
import numpy as np

from src.utils.logger import logger


class WarpStrategy(ABC):
    """
    Abstract base class for image warping strategies.

    Each strategy implements a specific method for transforming an image
    from control points to destination points.
    """

    @abstractmethod
    def warp_image(
        self,
        image: np.ndarray,
        colored_image: Optional[np.ndarray],
        control_points: np.ndarray,
        destination_points: np.ndarray,
        warped_dimensions: Tuple[int, int],
        debug_image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply warping transformation to images.

        Args:
            image: Grayscale input image
            colored_image: Optional colored version
            control_points: Source points in the original image
            destination_points: Target points in the warped image
            warped_dimensions: (width, height) of output image
            debug_image: Optional debug/overlay image to warp with same transform
            **kwargs: Strategy-specific parameters

        Returns:
            Tuple of (warped_gray_image, warped_colored_image, warped_debug_image)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this warping strategy"""
        pass


class PerspectiveTransformStrategy(WarpStrategy):
    """
    Perspective transformation using 4-point homography.

    This is the most common method for document rectification.
    Requires exactly 4 control points forming a quadrilateral.
    """

    def __init__(self, interpolation_flag: int = cv2.INTER_LINEAR):
        """
        Initialize perspective transform strategy.

        Args:
            interpolation_flag: OpenCV interpolation method
                - cv2.INTER_LINEAR: Bilinear (default, good balance)
                - cv2.INTER_CUBIC: Bicubic (slower, higher quality)
                - cv2.INTER_NEAREST: Nearest neighbor (fastest, lower quality)
        """
        self.interpolation_flag = interpolation_flag

    def get_name(self) -> str:
        return "PerspectiveTransform"

    def warp_image(
        self,
        image: np.ndarray,
        colored_image: Optional[np.ndarray],
        control_points: np.ndarray,
        destination_points: np.ndarray,
        warped_dimensions: Tuple[int, int],
        debug_image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply perspective transformation.

        Uses cv2.getPerspectiveTransform and cv2.warpPerspective.
        """
        if len(control_points) != 4:
            raise ValueError(
                f"PerspectiveTransform requires exactly 4 control points, "
                f"got {len(control_points)}"
            )

        # Ensure float32 for OpenCV
        control_pts = np.float32(control_points)
        dest_pts = np.float32(destination_points)

        # Compute perspective transformation matrix (once)
        transform_matrix = cv2.getPerspectiveTransform(control_pts, dest_pts)

        # Apply to grayscale image
        warped_image = cv2.warpPerspective(
            image, transform_matrix, warped_dimensions, flags=self.interpolation_flag
        )

        # Apply to colored image if provided
        warped_colored_image = None
        if colored_image is not None:
            warped_colored_image = cv2.warpPerspective(
                colored_image,
                transform_matrix,
                warped_dimensions,
                flags=self.interpolation_flag,
            )

        # Apply same transform to debug image if provided
        warped_debug_image = None
        if debug_image is not None:
            warped_debug_image = cv2.warpPerspective(
                debug_image,
                transform_matrix,
                warped_dimensions,
                flags=self.interpolation_flag,
            )

        logger.debug(
            f"Applied perspective transform: {image.shape[:2]} -> {warped_image.shape[:2]}"
        )

        return warped_image, warped_colored_image, warped_debug_image


class HomographyStrategy(WarpStrategy):
    """
    Homography-based transformation using N points.

    More flexible than perspective transform, can use more than 4 points.
    Uses cv2.findHomography with least-squares fitting.
    """

    def __init__(
        self,
        interpolation_flag: int = cv2.INTER_LINEAR,
        use_ransac: bool = False,
        ransac_threshold: float = 3.0,
    ):
        """
        Initialize homography strategy.

        Args:
            interpolation_flag: OpenCV interpolation method
            use_ransac: Use RANSAC for robust estimation
            ransac_threshold: RANSAC reprojection threshold (pixels)
        """
        self.interpolation_flag = interpolation_flag
        self.use_ransac = use_ransac
        self.ransac_threshold = ransac_threshold

    def get_name(self) -> str:
        return "Homography"

    def warp_image(
        self,
        image: np.ndarray,
        colored_image: Optional[np.ndarray],
        control_points: np.ndarray,
        destination_points: np.ndarray,
        warped_dimensions: Tuple[int, int],
        debug_image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply homography transformation.

        Computes homography matrix using cv2.findHomography.
        """
        if len(control_points) < 4:
            raise ValueError(
                f"Homography requires at least 4 control points, "
                f"got {len(control_points)}"
            )

        # Ensure float32
        control_pts = np.float32(control_points)
        dest_pts = np.float32(destination_points)

        # Compute homography (once)
        method = cv2.RANSAC if self.use_ransac else 0
        homography, mask = cv2.findHomography(
            control_pts,
            dest_pts,
            method=method,
            ransacReprojThreshold=self.ransac_threshold if self.use_ransac else None,
        )

        if homography is None:
            raise ValueError("Failed to compute homography matrix")

        transform_matrix = np.float32(homography)

        # Apply warping
        warped_image = cv2.warpPerspective(
            image, transform_matrix, warped_dimensions, flags=self.interpolation_flag
        )

        warped_colored_image = None
        if colored_image is not None:
            warped_colored_image = cv2.warpPerspective(
                colored_image,
                transform_matrix,
                warped_dimensions,
                flags=self.interpolation_flag,
            )

        warped_debug_image = None
        if debug_image is not None:
            warped_debug_image = cv2.warpPerspective(
                debug_image,
                transform_matrix,
                warped_dimensions,
                flags=self.interpolation_flag,
            )

        inliers = np.sum(mask) if mask is not None else len(control_points)
        logger.debug(f"Applied homography with {inliers}/{len(control_points)} inliers")

        return warped_image, warped_colored_image, warped_debug_image


class GridDataRemapStrategy(WarpStrategy):
    """
    Grid-based interpolation using scipy.interpolate.griddata.

    Creates a dense warp field by interpolating between sparse control points.
    Useful for non-linear transformations and when you have many control points.
    """

    def __init__(self, interpolation_method: str = "cubic"):
        """
        Initialize griddata remap strategy.

        Args:
            interpolation_method: 'linear', 'nearest', or 'cubic'
        """
        self.interpolation_method = interpolation_method

    def get_name(self) -> str:
        return "GridDataRemap"

    def warp_image(
        self,
        image: np.ndarray,
        colored_image: Optional[np.ndarray],
        control_points: np.ndarray,
        destination_points: np.ndarray,
        warped_dimensions: Tuple[int, int],
        debug_image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply grid-based interpolation.

        Uses scipy.interpolate.griddata to create a dense mapping.
        """
        from scipy.interpolate import griddata

        w, h = warped_dimensions

        # Create meshgrid for destination coordinates
        grid_y, grid_x = np.mgrid[0 : h - 1 : complex(h), 0 : w - 1 : complex(w)]

        # Interpolate source coordinates for each destination pixel (once)
        grid_z = griddata(
            points=destination_points,  # Where we want to be
            values=control_points,  # Where pixels come from
            xi=(grid_x, grid_y),  # Interpolate for all dest pixels
            method=self.interpolation_method,
        )

        grid_z = grid_z.astype("float32")

        # Apply remap
        warped_image = cv2.remap(
            image, map1=grid_z, map2=None, interpolation=cv2.INTER_CUBIC
        )

        warped_colored_image = None
        if colored_image is not None:
            warped_colored_image = cv2.remap(
                colored_image, map1=grid_z, map2=None, interpolation=cv2.INTER_CUBIC
            )

        warped_debug_image = None
        if debug_image is not None:
            warped_debug_image = cv2.remap(
                debug_image, map1=grid_z, map2=None, interpolation=cv2.INTER_CUBIC
            )

        logger.debug(
            f"Applied griddata remap with {len(control_points)} control points"
        )

        return warped_image, warped_colored_image, warped_debug_image


class DocRefineRectifyStrategy(WarpStrategy):
    """
    Document rectification using custom scanline-based approach.

    Uses edge contours to create a detailed warp field that preserves
    document structure better than simple perspective transform.
    """

    def get_name(self) -> str:
        return "DocRefineRectify"

    def warp_image(
        self,
        image: np.ndarray,
        colored_image: Optional[np.ndarray],
        control_points: np.ndarray,
        destination_points: np.ndarray,
        warped_dimensions: Tuple[int, int],
        debug_image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply document rectification.

        Requires 'edge_contours_map' in kwargs containing edge contours.
        """
        from src.processors.helpers.rectify import rectify

        edge_contours_map = kwargs.get("edge_contours_map")
        if edge_contours_map is None:
            raise ValueError("DocRefineRectify requires 'edge_contours_map' in kwargs")

        # Create dense warp map using rectify algorithm (once)
        scaled_map = rectify(edge_contours_map=edge_contours_map)

        # Apply remap
        warped_image = cv2.remap(
            image, map1=scaled_map, map2=None, interpolation=cv2.INTER_NEAREST
        )

        warped_colored_image = None
        if colored_image is not None:
            warped_colored_image = cv2.remap(
                colored_image, map1=scaled_map, map2=None, interpolation=cv2.INTER_CUBIC
            )

        warped_debug_image = None
        if debug_image is not None:
            warped_debug_image = cv2.remap(
                debug_image, map1=scaled_map, map2=None, interpolation=cv2.INTER_NEAREST
            )

        logger.debug("Applied doc-refine rectification")

        return warped_image, warped_colored_image, warped_debug_image


# Factory for creating strategies
class WarpStrategyFactory:
    """
    Factory for creating warp strategy instances.

    Centralizes strategy creation and configuration.
    """

    _strategies = {
        "PERSPECTIVE_TRANSFORM": PerspectiveTransformStrategy,
        "HOMOGRAPHY": HomographyStrategy,
        "REMAP_GRIDDATA": GridDataRemapStrategy,
        "DOC_REFINE": DocRefineRectifyStrategy,
    }

    @classmethod
    def create(cls, method_name: str, **config) -> WarpStrategy:
        """
        Create a warp strategy by name.

        Args:
            method_name: Strategy name (e.g., 'PERSPECTIVE_TRANSFORM')
            **config: Strategy-specific configuration

        Returns:
            Configured WarpStrategy instance

        Raises:
            ValueError: If method_name is unknown
        """
        strategy_class = cls._strategies.get(method_name)

        if strategy_class is None:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown warp method '{method_name}'. Available: {available}"
            )

        return strategy_class(**config)

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """Return list of available warp method names"""
        return list(cls._strategies.keys())
