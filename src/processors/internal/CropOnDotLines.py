import cv2
import numpy as np

from src.processors.constants import (
    DOT_AREA_TYPES_IN_ORDER,
    EDGE_TYPES_IN_ORDER,
    LINE_AREA_TYPES_IN_ORDER,
    TARGET_EDGE_FOR_LINE,
    AreaTemplate,
    EdgeType,
    ScannerType,
    WarpMethod,
)
from src.processors.internal.CropOnPatchesCommon import CropOnPatchesCommon
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.math import MathUtils

# from rich.table import Table
# from src.utils.logger import console


class CropOnDotLines(CropOnPatchesCommon):
    __is_internal_preprocessor__ = True

    scan_area_templates_for_layout = {
        "ONE_LINE_TWO_DOTS": [
            AreaTemplate.topRightDot,
            AreaTemplate.bottomRightDot,
            AreaTemplate.leftLine,
        ],
        "TWO_DOTS_ONE_LINE": [
            AreaTemplate.rightLine,
            AreaTemplate.topLeftDot,
            AreaTemplate.bottomLeftDot,
        ],
        "TWO_LINES": [
            AreaTemplate.leftLine,
            AreaTemplate.rightLine,
        ],
        "TWO_LINES_HORIZONTAL": [
            AreaTemplate.topLine,
            AreaTemplate.bottomLine,
        ],
        "FOUR_DOTS": DOT_AREA_TYPES_IN_ORDER,
    }

    default_scan_area_descriptions = {
        **{
            area_template: {
                "scannerType": ScannerType.PATCH_DOT,
                "selector": "SELECT_CENTER",
                "maxPoints": 2,  # for cropping
            }
            for area_template in DOT_AREA_TYPES_IN_ORDER
        },
        **{
            area_template: {
                "scannerType": ScannerType.PATCH_LINE,
                "selector": "LINE_OUTER_EDGE",
            }
            for area_template in LINE_AREA_TYPES_IN_ORDER
        },
        "CUSTOM": {},
    }

    default_points_selector_map = {
        "CENTERS": {
            AreaTemplate.topLeftDot: "SELECT_CENTER",
            AreaTemplate.topRightDot: "SELECT_CENTER",
            AreaTemplate.bottomRightDot: "SELECT_CENTER",
            AreaTemplate.bottomLeftDot: "SELECT_CENTER",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_WIDTHS": {
            AreaTemplate.topLeftDot: "SELECT_TOP_RIGHT",
            AreaTemplate.topRightDot: "SELECT_TOP_LEFT",
            AreaTemplate.bottomRightDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.bottomLeftDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.leftLine: "LINE_INNER_EDGE",
            AreaTemplate.rightLine: "LINE_INNER_EDGE",
        },
        "INNER_HEIGHTS": {
            AreaTemplate.topLeftDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.topRightDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.bottomRightDot: "SELECT_TOP_RIGHT",
            AreaTemplate.bottomLeftDot: "SELECT_TOP_LEFT",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_CORNERS": {
            AreaTemplate.topLeftDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.topRightDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.bottomRightDot: "SELECT_TOP_LEFT",
            AreaTemplate.bottomLeftDot: "SELECT_TOP_RIGHT",
            AreaTemplate.leftLine: "LINE_INNER_EDGE",
            AreaTemplate.rightLine: "LINE_INNER_EDGE",
        },
        "OUTER_CORNERS": {
            AreaTemplate.topLeftDot: "SELECT_TOP_LEFT",
            AreaTemplate.topRightDot: "SELECT_TOP_RIGHT",
            AreaTemplate.bottomRightDot: "SELECT_BOTTOM_RIGHT",
            AreaTemplate.bottomLeftDot: "SELECT_BOTTOM_LEFT",
            AreaTemplate.leftLine: "LINE_OUTER_EDGE",
            AreaTemplate.rightLine: "LINE_OUTER_EDGE",
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tuning_options = self.tuning_options
        self.line_kernel_morph = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(tuning_options.get("lineKernel", [2, 10]))
        )
        self.dot_kernel_morph = cv2.getStructuringElement(
            cv2.MORPH_RECT, tuple(tuning_options.get("dotKernel", [5, 5]))
        )

    def validate_scan_areas(self):
        # TODO: more validations here at child class level (customOptions etc)
        return super().validate_scan_areas()

    def validate_and_remap_options_schema(self, options):
        layout_type = options["type"]
        tuning_options = options.get("tuningOptions", {})
        parsed_options = {
            "pointsLayout": layout_type,
            "enableCropping": True,
            "tuningOptions": {
                "warpMethod": tuning_options.get(
                    "warpMethod", WarpMethod.PERSPECTIVE_TRANSFORM
                )
            },
        }

        # TODO: add default values for provided options["scanAreas"]? like get "maxPoints" from options["lineMaxPoints"]
        # inject scanAreas
        parsed_options["scanAreas"] = [
            # Note: currently at CropOnMarkers/CropOnDotLines level, 'scanAreas' is extended provided areas
            *options.get("scanAreas", []),
            *[
                {
                    "areaTemplate": area_template,
                    "areaDescription": options.get(area_template, {}),
                    "customOptions": {
                        # TODO: get customOptions here
                    },
                }
                for area_template in self.scan_area_templates_for_layout[layout_type]
            ],
        ]
        return parsed_options

    edge_selector_map = {
        AreaTemplate.topLine: {
            "LINE_INNER_EDGE": EdgeType.BOTTOM,
            "LINE_OUTER_EDGE": EdgeType.TOP,
        },
        AreaTemplate.leftLine: {
            "LINE_INNER_EDGE": EdgeType.RIGHT,
            "LINE_OUTER_EDGE": EdgeType.LEFT,
        },
        AreaTemplate.bottomLine: {
            "LINE_INNER_EDGE": EdgeType.TOP,
            "LINE_OUTER_EDGE": EdgeType.BOTTOM,
        },
        AreaTemplate.rightLine: {
            "LINE_INNER_EDGE": EdgeType.LEFT,
            "LINE_OUTER_EDGE": EdgeType.RIGHT,
        },
    }

    @staticmethod
    def select_edge_from_scan_area(area_description, edge_type):
        destination_rectangle = MathUtils.get_rectangle_points_from_box(
            area_description["origin"], area_description["dimensions"]
        )

        destination_line = MathUtils.select_edge_from_rectangle(
            destination_rectangle, edge_type
        )
        return destination_line

    def find_and_select_points_from_line(
        self, image, area_template, area_description, _file_path
    ):
        area_label = area_description["label"]
        points_selector = area_description.get(
            "selector", self.default_points_selector.get(area_label, None)
        )

        line_edge_contours_map = self.find_line_corners_and_contours(
            image, area_description
        )

        selected_edge_type = self.edge_selector_map[area_label][points_selector]
        target_edge_type = TARGET_EDGE_FOR_LINE[area_template]
        selected_contour = line_edge_contours_map[selected_edge_type]
        destination_line = self.select_edge_from_scan_area(
            area_description, selected_edge_type
        )
        # Ensure clockwise order after extraction
        if selected_edge_type != target_edge_type:
            selected_contour.reverse()
            destination_line.reverse()

        max_points = area_description.get("maxPoints", None)

        # Extrapolates the destination_line to get approximate destination points
        (
            control_points,
            destination_points,
        ) = ImageUtils.get_control_destination_points_from_contour(
            selected_contour, destination_line, max_points
        )
        # Temp
        # table = Table(
        #     title=f"{area_label}: {destination_line}",
        #     show_header=True,
        #     show_lines=False,
        # )
        # table.add_column("Control", style="cyan", no_wrap=True)
        # table.add_column("Destination", style="magenta")
        # for c, d in zip(control_points, destination_points):
        #     table.add_row(str(c), str(d))
        # console.print(table, justify="center")

        return control_points, destination_points, selected_contour

    def find_line_corners_and_contours(self, image, area_description):
        area_label = area_description["label"]
        config = self.tuning_config
        tuning_options = self.tuning_options
        area, area_start = self.compute_scan_area_util(image, area_description)

        # Make boxes darker (less gamma)
        darker_image = ImageUtils.adjust_gamma(area, config.thresholding.GAMMA_LOW)

        # Lines are expected to be fairly dark
        line_threshold = tuning_options.get("lineThreshold", 180)

        _, thresholded = cv2.threshold(
            darker_image, line_threshold, 255, cv2.THRESH_TRUNC
        )
        normalised = ImageUtils.normalize(thresholded)

        # add white padding
        kernel_height, kernel_width = self.line_kernel_morph.shape[:2]
        white, pad_range = ImageUtils.pad_image_from_center(
            normalised, kernel_width, kernel_height, 255
        )

        # Threshold-Normalize after morph + white padding
        _, white_thresholded = cv2.threshold(
            white, line_threshold, 255, cv2.THRESH_TRUNC
        )
        white_normalised = ImageUtils.normalize(white_thresholded)

        # Open : erode then dilate
        line_morphed = cv2.morphologyEx(
            white_normalised, cv2.MORPH_OPEN, self.line_kernel_morph, iterations=3
        )

        # remove white padding
        line_morphed = line_morphed[
            pad_range[0] : pad_range[1], pad_range[2] : pad_range[3]
        ]

        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [
                darker_image,
                normalised,
                white_thresholded,
                line_morphed,
            ]
        elif config.outputs.show_image_level == 4:
            InteractionUtils.show(
                f"morph_opened_{area_label}", line_morphed, pause=False
            )

        (
            _,
            edge_contours_map,
        ) = self.find_morph_corners_and_contours_map(
            area_start, line_morphed, area_description
        )

        if edge_contours_map is None:
            raise Exception(
                f"No line match found at origin: {area_description['origin']} with dimensions: {area_description['dimensions']}"
            )
        return edge_contours_map

    def find_dot_corners_from_options(self, image, area_description, _file_path):
        config = self.tuning_config
        tuning_options = self.tuning_options
        area_label = area_description["label"]

        area, area_start = self.compute_scan_area_util(image, area_description)

        # TODO: simple colored thresholding to clear out noise?

        dot_blur_kernel = tuning_options.get("dotBlurKernel", None)
        if dot_blur_kernel:
            area = cv2.GaussianBlur(area, dot_blur_kernel, 0)

        # Open : erode then dilate
        morph_c = cv2.morphologyEx(
            area, cv2.MORPH_OPEN, self.dot_kernel_morph, iterations=3
        )

        # TODO: try pyrDown to 64 values and find the outlier for black threshold?
        # Dots are expected to be fairly dark
        dot_threshold = tuning_options.get("dotThreshold", 150)
        _, thresholded = cv2.threshold(morph_c, dot_threshold, 255, cv2.THRESH_TRUNC)
        normalised = ImageUtils.normalize(thresholded)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack += [area, morph_c, thresholded, normalised]
        elif config.outputs.show_image_level == 4:
            InteractionUtils.show(
                f"threshold_normalised: {area_label}", normalised, pause=False
            )

        corners, _ = self.find_morph_corners_and_contours_map(
            area_start, normalised, area_description
        )
        if corners is None:
            if config.outputs.show_image_level >= 1:
                hstack = ImageUtils.get_padded_hstack(
                    [self.debug_image, area, thresholded]
                )
                InteractionUtils.show(
                    f"No patch/dot debug hstack",
                    ImageUtils.get_padded_hstack(self.debug_hstack),
                    pause=0,
                )
                InteractionUtils.show(f"No patch/dot found:", hstack, pause=1)

            raise Exception(
                f"No patch/dot found at origin: {area_description['origin']} with dimensions: {area_description['dimensions']}"
            )

        return corners

    # TODO: >> create a ScanArea class and move some methods there
    def find_morph_corners_and_contours_map(self, area_start, area, area_description):
        scanner_type, area_label = (
            area_description["scannerType"],
            area_description["label"],
        )
        config = self.tuning_config
        canny_edges = cv2.Canny(area, 185, 55)

        if config.outputs.show_image_level >= 5:
            self.debug_hstack.append(canny_edges)

        # Should mostly return a single contour in the area
        all_contours = ImageUtils.grab_contours(
            cv2.findContours(
                canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )  # cv2.CHAIN_APPROX_NONE)
        )
        # Note: skipping convexHull here because we want to preserve the curves

        if len(all_contours) == 0:
            return None, None

        ordered_patch_corners, edge_contours_map = None, None
        largest_contour = sorted(all_contours, key=cv2.contourArea, reverse=True)[0]
        if config.outputs.show_image_level >= 5:
            h, w = canny_edges.shape[:2]
            contour_overlay = 255 * np.ones((h, w), np.uint8)
            DrawingUtils.draw_contour(contour_overlay, largest_contour)
            self.debug_hstack.append(contour_overlay)

        # Convert to list of 2d points
        bounding_contour = np.vstack(largest_contour).squeeze()

        # TODO: see if bounding_hull is still needed
        bounding_hull = cv2.convexHull(bounding_contour)

        if scanner_type == ScannerType.PATCH_DOT:
            # Bounding rectangle will not be rotated
            x, y, w, h = cv2.boundingRect(bounding_hull)
            patch_corners = MathUtils.get_rectangle_points(x, y, w, h)
            (
                ordered_patch_corners,
                edge_contours_map,
            ) = ImageUtils.split_patch_contour_on_corners(
                patch_corners, bounding_contour
            )
        elif scanner_type == ScannerType.PATCH_LINE:
            # Rotated rectangle can correct slight rotations better
            rotated_rect = cv2.minAreaRect(bounding_hull)
            # TODO: less confidence if angle = rotated_rect[2] is too skew
            rotated_rect_points = cv2.boxPoints(rotated_rect)
            patch_corners = np.intp(rotated_rect_points)
            (
                ordered_patch_corners,
                edge_contours_map,
            ) = ImageUtils.split_patch_contour_on_corners(
                patch_corners, bounding_contour
            )
        else:
            raise Exception(f"Unsupported scanner type: {scanner_type}")

        # TODO: less confidence if given dimensions differ from matched block size (also give a warning)
        if config.outputs.show_image_level >= 5:
            if ordered_patch_corners is not None:
                corners_contour_overlay = canny_edges.copy()
                DrawingUtils.draw_contour(
                    corners_contour_overlay, ordered_patch_corners
                )
                self.debug_hstack.append(corners_contour_overlay)

            InteractionUtils.show(
                f"Debug Largest Patch: {area_label}",
                ImageUtils.get_padded_hstack(self.debug_hstack),
                0,
                resize_to_height=True,
                config=config,
            )
            self.debug_vstack.append(self.debug_hstack)
            self.debug_hstack = []

        absolute_corners = MathUtils.shift_points_from_origin(
            area_start, ordered_patch_corners
        )

        shifted_edge_contours_map = {
            edge_type: MathUtils.shift_points_from_origin(
                area_start, edge_contours_map[edge_type]
            )
            for edge_type in EDGE_TYPES_IN_ORDER
        }

        return absolute_corners, shifted_edge_contours_map
