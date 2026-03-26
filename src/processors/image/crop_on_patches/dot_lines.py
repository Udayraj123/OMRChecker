from typing import ClassVar

from src.utils.exceptions import ImageProcessingError, TemplateValidationError
from src.processors.constants import (
    DOT_ZONE_TYPES_IN_ORDER,
    LINE_ZONE_TYPES_IN_ORDER,
    TARGET_EDGE_FOR_LINE,
    EdgeType,
    ScannerType,
    SelectorType,
    WarpMethod,
    ZonePreset,
)
from src.processors.image.crop_on_patches.common import CropOnPatchesCommon
from src.processors.image.crop_on_patches.dot_line_detection import (
    detect_dot_corners,
    detect_line_corners_and_edges,
    create_structuring_element,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.json_conversion import camel_to_snake
from src.utils.math import MathUtils
from src.processors.image.crop_on_patches.patch_utils import compute_scan_zone


class CropOnDotLines(CropOnPatchesCommon):
    __is_internal_preprocessor__: ClassVar = True

    scan_zone_presets_for_layout: ClassVar = {
        "ONE_LINE_TWO_DOTS": [
            ZonePreset.topRightDot,
            ZonePreset.bottomRightDot,
            ZonePreset.leftLine,
        ],
        "TWO_DOTS_ONE_LINE": [
            ZonePreset.rightLine,
            ZonePreset.topLeftDot,
            ZonePreset.bottomLeftDot,
        ],
        "TWO_LINES": [
            ZonePreset.leftLine,
            ZonePreset.rightLine,
        ],
        "TWO_LINES_HORIZONTAL": [
            ZonePreset.topLine,
            ZonePreset.bottomLine,
        ],
        "FOUR_DOTS": DOT_ZONE_TYPES_IN_ORDER,
        # "BOUNDING_RECTANGLE" : DOT_ZONE_TYPES_IN_ORDER,
    }

    default_scan_zone_descriptions: ClassVar = {
        **{
            zone_preset: {
                "scanner_type": ScannerType.PATCH_DOT,
                "selector": "SELECT_CENTER",
                "max_points": 2,  # for cropping
            }
            for zone_preset in DOT_ZONE_TYPES_IN_ORDER
        },
        **{
            zone_preset: {
                "scanner_type": ScannerType.PATCH_LINE,
                "selector": "LINE_OUTER_EDGE",
            }
            for zone_preset in LINE_ZONE_TYPES_IN_ORDER
        },
        "CUSTOM": {},
    }

    default_points_selector_map: ClassVar = {
        "CENTERS": {
            ZonePreset.topLeftDot: "SELECT_CENTER",
            ZonePreset.topRightDot: "SELECT_CENTER",
            ZonePreset.bottomRightDot: "SELECT_CENTER",
            ZonePreset.bottomLeftDot: "SELECT_CENTER",
            ZonePreset.leftLine: "LINE_OUTER_EDGE",
            ZonePreset.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_WIDTHS": {
            ZonePreset.topLeftDot: "SELECT_TOP_RIGHT",
            ZonePreset.topRightDot: "SELECT_TOP_LEFT",
            ZonePreset.bottomRightDot: "SELECT_BOTTOM_LEFT",
            ZonePreset.bottomLeftDot: "SELECT_BOTTOM_RIGHT",
            ZonePreset.leftLine: "LINE_INNER_EDGE",
            ZonePreset.rightLine: "LINE_INNER_EDGE",
        },
        "INNER_HEIGHTS": {
            ZonePreset.topLeftDot: "SELECT_BOTTOM_LEFT",
            ZonePreset.topRightDot: "SELECT_BOTTOM_RIGHT",
            ZonePreset.bottomRightDot: "SELECT_TOP_RIGHT",
            ZonePreset.bottomLeftDot: "SELECT_TOP_LEFT",
            ZonePreset.leftLine: "LINE_OUTER_EDGE",
            ZonePreset.rightLine: "LINE_OUTER_EDGE",
        },
        "INNER_CORNERS": {
            ZonePreset.topLeftDot: "SELECT_BOTTOM_RIGHT",
            ZonePreset.topRightDot: "SELECT_BOTTOM_LEFT",
            ZonePreset.bottomRightDot: "SELECT_TOP_LEFT",
            ZonePreset.bottomLeftDot: "SELECT_TOP_RIGHT",
            ZonePreset.leftLine: "LINE_INNER_EDGE",
            ZonePreset.rightLine: "LINE_INNER_EDGE",
        },
        "OUTER_CORNERS": {
            ZonePreset.topLeftDot: "SELECT_TOP_LEFT",
            ZonePreset.topRightDot: "SELECT_TOP_RIGHT",
            ZonePreset.bottomRightDot: "SELECT_BOTTOM_RIGHT",
            ZonePreset.bottomLeftDot: "SELECT_BOTTOM_LEFT",
            ZonePreset.leftLine: "LINE_OUTER_EDGE",
            ZonePreset.rightLine: "LINE_OUTER_EDGE",
        },
    }

    def __init__(self, options, *args, **kwargs) -> None:
        # Parent's __init__ will call validate_and_remap_options_schema via polymorphism
        super().__init__(options, *args, **kwargs)
        tuning_options = self.tuning_options
        self.line_kernel_morph = create_structuring_element(
            "rect", tuple(tuning_options.get("line_kernel", [2, 10]))
        )
        self.dot_kernel_morph = create_structuring_element(
            "rect", tuple(tuning_options.get("dot_kernel", [5, 5]))
        )

    def validate_scan_zones(self):
        # TODO: more validations here at child class level (customOptions etc)
        return super().validate_scan_zones()

    def validate_and_remap_options_schema(self, options):
        layout_type = options["type"]
        parsed_options = self._build_base_parsed_options(
            options,
            layout_type,
            cropping_enabled=options.get("cropping_enabled", True),
            default_warp_method=WarpMethod.PERSPECTIVE_TRANSFORM,
        )

        # TODO: add default values for provided options["scan_zones"]? like get "maxPoints" from options["line_max_points"]
        # inject scanZones
        parsed_options["scan_zones"] = [
            # Note: currently at CropOnMarkers/CropOnDotLines level, 'scanZones' is extended provided zones
            *options.get("scan_zones", []),
            *[
                {
                    "zone_preset": zone_preset,
                    "zone_description": options[camel_to_snake(zone_preset)],
                    "custom_options": {
                        # TODO: get customOptions here
                    },
                }
                for zone_preset in CropOnDotLines.scan_zone_presets_for_layout[
                    layout_type
                ]
                if camel_to_snake(zone_preset) in options
            ],
        ]
        return parsed_options

    edge_selector_map: ClassVar = {
        ZonePreset.topLine: {
            "LINE_INNER_EDGE": EdgeType.BOTTOM,
            "LINE_OUTER_EDGE": EdgeType.TOP,
        },
        ZonePreset.leftLine: {
            "LINE_INNER_EDGE": EdgeType.RIGHT,
            "LINE_OUTER_EDGE": EdgeType.LEFT,
        },
        ZonePreset.bottomLine: {
            "LINE_INNER_EDGE": EdgeType.TOP,
            "LINE_OUTER_EDGE": EdgeType.BOTTOM,
        },
        ZonePreset.rightLine: {
            "LINE_INNER_EDGE": EdgeType.LEFT,
            "LINE_OUTER_EDGE": EdgeType.RIGHT,
        },
    }

    @staticmethod
    def select_edge_from_scan_zone(zone_description, edge_type):
        destination_rectangle = MathUtils.get_rectangle_points_from_box(
            zone_description["origin"], zone_description["dimensions"]
        )

        return MathUtils.select_edge_from_rectangle(destination_rectangle, edge_type)

    def find_and_select_points_from_line(
        self, image, zone_preset, zone_description, _file_path
    ):
        zone_label = zone_description["label"]
        points_selector = zone_description.get(
            "selector",
            self.default_points_selector.get(zone_label, SelectorType.LINE_OUTER_EDGE),
        )

        line_edge_contours_map = self.find_line_corners_and_contours(
            image, zone_description
        )

        selected_edge_type = self.edge_selector_map[zone_label][points_selector]
        target_edge_type = TARGET_EDGE_FOR_LINE[zone_preset]
        selected_contour = line_edge_contours_map[selected_edge_type]
        destination_line = self.select_edge_from_scan_zone(
            zone_description, selected_edge_type
        )
        # Ensure clockwise order after extraction
        if selected_edge_type != target_edge_type:
            selected_contour.reverse()
            destination_line.reverse()

        max_points = zone_description.get("max_points", None)

        # Extrapolates the destination_line to get approximate destination points
        (
            control_points,
            destination_points,
        ) = ImageUtils.get_control_destination_points_from_contour(
            selected_contour, destination_line, max_points
        )
        # Temp
        # table = Table(
        #     title=f"{zone_label}: {destination_line}",
        #     show_header=True,
        #     show_lines=False,
        # )
        # table.add_column("Control", style="cyan", no_wrap=True)
        # table.add_column("Destination", style="magenta")
        # for c, d in zip(control_points, destination_points):
        #     table.add_row(str(c), str(d))
        # console.print(table, justify="center")

        return control_points, destination_points, selected_contour

    def find_line_corners_and_contours(self, image, zone_description):
        """
        Detect line corners and edge contours using extracted detection module.

        This is now a thin wrapper around detect_line_corners_and_edges that:
        1. Validates blur kernel
        2. Applies optional Gaussian blur
        3. Calls the extracted detection function
        4. Handles debug visualization
        """
        zone_label = zone_description["label"]
        config = self.tuning_config
        tuning_options = self.tuning_options

        zone, zone_start, _ = compute_scan_zone(image, zone_description)

        # Validate and apply blur if configured
        line_blur_kernel = tuning_options.get("line_blur_kernel", None)
        if line_blur_kernel:
            zone_h, zone_w = zone.shape
            blur_h, blur_w = line_blur_kernel

            if not (zone_h > blur_h and zone_w > blur_w):
                msg = f"The zone '{zone_label}' is smaller than provided lineBlurKernel: {zone.shape} < {line_blur_kernel}"
                raise TemplateValidationError(
                    msg,
                    context={
                        "zone_label": zone_label,
                        "zone_shape": zone.shape,
                        "line_blur_kernel": line_blur_kernel,
                    },
                )

        # Use extracted detection module
        line_threshold = tuning_options.get("line_threshold", 180)
        _, edge_contours_map = detect_line_corners_and_edges(
            zone,
            zone_start,
            self.line_kernel_morph,
            config.thresholding.gamma_low,
            line_threshold=line_threshold,
            blur_kernel=line_blur_kernel,
        )

        if edge_contours_map is None:
            msg = f"No line match found at origin: {zone_description['origin']} with dimensions: {zone_description['dimensions']}"
            raise ImageProcessingError(
                msg,
                context={
                    "origin": zone_description["origin"],
                    "dimensions": zone_description["dimensions"],
                },
            )

        return edge_contours_map

    def find_dot_corners_from_options(self, image, zone_description, _file_path):
        """
        Detect dot corners using extracted detection module.

        This is now a thin wrapper around detect_dot_corners that:
        1. Validates blur kernel
        2. Applies optional Gaussian blur
        3. Calls the extracted detection function
        4. Handles debug visualization and error messages
        """
        config = self.tuning_config
        tuning_options = self.tuning_options
        zone_label = zone_description["label"]

        zone, zone_start, _ = compute_scan_zone(image, zone_description)

        # Validate and apply blur if configured
        dot_blur_kernel = tuning_options.get("dot_blur_kernel", None)
        if dot_blur_kernel:
            zone_h, zone_w = zone.shape
            blur_h, blur_w = dot_blur_kernel

            if not (zone_h > blur_h and zone_w > blur_w):
                msg = f"The zone '{zone_label}' is smaller than provided dotBlurKernel: {zone.shape} < {dot_blur_kernel}"
                raise TemplateValidationError(
                    msg,
                    context={
                        "zone_label": zone_label,
                        "zone_shape": zone.shape,
                        "dot_blur_kernel": dot_blur_kernel,
                    },
                )

        # Use extracted detection module
        dot_threshold = tuning_options.get("dot_threshold", 150)
        corners = detect_dot_corners(
            zone,
            zone_start,
            self.dot_kernel_morph,
            dot_threshold=dot_threshold,
            blur_kernel=dot_blur_kernel,
        )

        if corners is None:
            if config.outputs.show_image_level >= 1:
                if len(self.debug_hstack) > 0:
                    InteractionUtils.show(
                        "No patch/dot debug hstack",
                        ImageUtils.get_padded_hstack(self.debug_hstack),
                        pause=0,
                    )
                # Show debug information for troubleshooting
                hstack = ImageUtils.get_padded_hstack([self.debug_image, zone])
                InteractionUtils.show(
                    f"No patch/dot found for {zone_label}", hstack, pause=1
                )

            msg = f"No patch/dot found at origin: {zone_description['origin']} with dimensions: {zone_description['dimensions']}"
            raise ImageProcessingError(
                msg,
                context={
                    "origin": zone_description["origin"],
                    "dimensions": zone_description["dimensions"],
                },
            )

        return corners
