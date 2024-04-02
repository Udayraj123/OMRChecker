import cv2

from src.processors.constants import (
    DOT_AREA_TYPES_IN_ORDER,
    LINE_AREA_TYPES_IN_ORDER,
    MARKER_AREA_TYPES_IN_ORDER,
    ScannerType,
)
from src.processors.interfaces.ImageTemplatePreprocessor import (
    ImageTemplatePreprocessor,
)
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.math import MathUtils
from src.utils.parsing import OVERRIDE_MERGER


# Internal Processor for separation of code
class WarpOnPointsCommon(ImageTemplatePreprocessor):
    __is_internal_preprocessor__ = True

    # TODO: these should be divided into child class accessors!
    default_scan_area_descriptions = {
        **{
            marker_type: {
                "scannerType": ScannerType.TEMPLATE_MATCH,
                "selector": "TEMPLATE_CENTER",
            }
            for marker_type in MARKER_AREA_TYPES_IN_ORDER
        },
        **{
            marker_type: {
                "scannerType": ScannerType.PATCH_DOT,
                "selector": "DOT_CENTER",
            }
            for marker_type in DOT_AREA_TYPES_IN_ORDER
        },
        **{
            marker_type: {
                "scannerType": "PATCH_LINE",
                "selector": "LINE_OUTER_EDGE",
            }
            for marker_type in LINE_AREA_TYPES_IN_ORDER
        },
        "CUSTOM": {},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options
        self.scan_areas = self.parse_scan_areas_with_defaults(options["scanAreas"])
        self.validate_scan_areas()
        self.validate_points_layouts()

    def exclude_files(self):
        return []

    def prepare_image(self, image):
        return image

    def parse_scan_areas_with_defaults(self, scan_areas):
        scan_areas_with_defaults = []
        for scan_area in scan_areas:
            area_template, area_description = scan_area["areaTemplate"], scan_area.get(
                "areaDescription", {}
            )
            area_description["label"] = area_description.get("label", area_template)
            custom_options = area_description.get("customOptions", {})
            scan_areas_with_defaults.append(
                {
                    "areaTemplate": area_template,
                    "areaDescription": OVERRIDE_MERGER.merge(
                        self.default_scan_area_descriptions[area_template],
                        area_description,
                    ),
                    "customOptions": custom_options,
                }
            )

        self.scan_areas = scan_areas_with_defaults

    def validate_scan_areas(self):
        seen_labels = set()
        repeat_labels = set()
        for scan_area in self.scan_areas:
            area_label = scan_area["areaDescription"]["label"]
            if area_label in seen_labels:
                repeat_labels.add(area_label)
            seen_labels.add(area_label)
        if len(repeat_labels) > 0:
            raise Exception(f"Found repeated labels in scanAreas: {repeat_labels}")

        # TODO: more validations in child classes

    # TODO: check if this needs to move into child for working properly (accessing self attributes declared in child in parent's constructor)
    def validate_points_layouts(self):
        options = self.options
        points_layout = options["pointsLayout"]
        if (
            points_layout not in self.scan_area_templates_for_layout
            and points_layout != "CUSTOM"
        ):
            raise Exception(
                f"Invalid pointsLayout provided: {points_layout} for {self}"
            )

        expected_templates = set(self.scan_area_templates_for_layout[points_layout])
        provided_templates = set(
            [scan_area["areaTemplate"] for scan_area in self.scan_areas]
        )
        not_provided_area_templates = expected_templates.difference(provided_templates)

        if len(not_provided_area_templates) > 0:
            logger.error(f"not_provided_area_templates={not_provided_area_templates}")
            raise Exception(
                f"Missing a few scanAreaTemplates for the pointsLayout {points_layout}"
            )

    def apply_filter(self, image, colored_image, _template, file_path):
        config = self.tuning_config

        self.debug_image = image.copy()
        self.debug_hstack = []
        self.debug_vstack = []

        image = self.prepare_image(image)

        # TODO: Save intuitive meta data
        # self.append_save_image(3,warped_image)

        (
            control_points,
            destination_points,
        ) = self.extract_control_destination_points(image, file_path)

        (
            parsed_destination_points,
            warped_dimensions,
        ) = self.parse_destination_points_for_image(
            image, control_points, destination_points
        )

        logger.info(
            f"control_points={control_points}, destination_points={destination_points}, warped_dimensions={warped_dimensions}"
        )

        # Find and pass control points in a defined order
        transform_matrix = cv2.getPerspectiveTransform(
            control_points, parsed_destination_points
        )

        # Crop the image
        warped_image = cv2.warpPerspective(image, transform_matrix, warped_dimensions)

        if config.outputs.show_colored_outputs:
            colored_image = cv2.warpPerspective(
                colored_image, transform_matrix, warped_dimensions
            )

        # self.append_save_image(1,warped_image)

        if config.outputs.show_image_level >= 4:
            hstack = ImageUtils.get_padded_hstack([self.debug_image, warped_image])
            InteractionUtils.show(
                f"warped_image: {file_path}", hstack, 1, 1, config=config
            )

        return warped_image, colored_image, _template

    def parse_destination_points_for_image(self, image, destination_points):
        config = self.tuning_config
        h, w = image.shape[:2]
        parsed_destination_points, warped_dimensions = destination_points, (w, h)
        enable_cropping = config.get("enableCropping", False)
        if enable_cropping:
            # TODO: exclude the destination points marked with excludeFromCropping (using a small class for each point?)
            # Also exclude corresponding points from control points (Note: may need to be done in a second pass after alignment warping)
            # But if warping supports alignment of negative points, this will work as-is (try it!)

            # TODO: Give a warning if the destination_points do not form a convex polygon!

            #   get bounding box on the destination points (with validation?)
            (
                destination_box,
                rectangle_dimensions,
            ) = MathUtils.get_bounding_box_of_points(destination_points)
            warped_dimensions = rectangle_dimensions
            # Shift the destination points to enable the cropping
            parsed_destination_points = MathUtils.shift_origin_for_points(
                destination_box[0], destination_points
            )

            # Note: control points remain the same (wrt image shape!)

        return (
            parsed_destination_points,
            warped_dimensions,
        )
