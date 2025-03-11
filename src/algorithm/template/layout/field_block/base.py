from copy import copy as shallowcopy
from typing import List

from src.algorithm.template.layout.field.base import Field, ScanBox
from src.algorithm.template.layout.field.bubble_field import BubbleField
from src.algorithm.template.layout.field.ocr_field import OCRField
from src.algorithm.template.layout.field_block.field_block_drawing import (
    FieldBlockDrawing,
)
from src.processors.constants import FieldDetectionType
from src.utils.parsing import default_dump, parse_fields


class FieldBlock:
    field_detection_type_to_field_class = {
        FieldDetectionType.BUBBLES_THRESHOLD: BubbleField,
        FieldDetectionType.OCR: OCRField,
    }

    def __init__(self, block_name, field_block_object, field_blocks_offset):
        self.name = block_name
        # TODO: Move plot_bin_name into child class
        self.plot_bin_name = block_name
        self.shifts = [0, 0]
        self.setup_field_block(field_block_object, field_blocks_offset)
        self.generate_fields()
        self.drawing = FieldBlockDrawing(self)

    # Make deepcopy for only parts that are mutated by Processor
    def get_copy_for_shifting(self):
        copied_field_block = shallowcopy(self)
        # No need to deepcopy self.fields since they are not using shifts yet,
        # also we are resetting them anyway before runs.
        return copied_field_block

    def reset_all_shifts(self):
        self.shifts = [0, 0]
        for field in self.fields:
            field.reset_all_shifts()

    # Need this at runtime as we have allowed mutation of template via pre-processors
    def get_shifted_origin(self):
        origin, shifts = self.origin, self.shifts
        return [origin[0] + shifts[0], origin[1] + shifts[1]]

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "bubble_dimensions",
                "dimensions",
                "empty_value",
                "fields",
                "name",
                "origin",
                # "shifted_origin",
                # "plot_bin_name",
            ]
        }

    def setup_field_block(self, field_block_object, field_blocks_offset):
        # case mapping
        (
            direction,
            empty_value,
            field_detection_type,
            field_labels,
            labels_gap,
            origin,
        ) = map(
            field_block_object.get,
            [
                "direction",
                "emptyValue",
                "fieldDetectionType",
                "fieldLabels",
                "labelsGap",
                "origin",
            ],
        )

        self.direction = direction
        self.empty_value = empty_value
        self.field_detection_type = field_detection_type
        self.labels_gap = labels_gap
        offset_x, offset_y = field_blocks_offset
        self.origin = [origin[0] + offset_x, origin[1] + offset_y]

        self.parsed_field_labels = parse_fields(
            f"Field Block Labels: {self.name}", field_labels
        )
        # TODO: conditionally set below items based on field
        if field_detection_type == FieldDetectionType.BUBBLES_THRESHOLD:
            self.setup_bubbles_field_block(field_block_object)
        elif field_detection_type == FieldDetectionType.OCR:
            self.setup_ocr_field_block(field_block_object)
        # TODO: support barcode, photo blob, etc custom field types
        # logger.info(
        #     "field_detection_type", field_detection_type, "labels_gap", labels_gap
        # )

    # TODO: move into a BubblesFieldBlock class? But provision to allow multiple field detection types in future.
    def setup_bubbles_field_block(self, field_block_object):
        (
            alignment_object,
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            bubble_field_type,
        ) = map(
            field_block_object.get,
            [
                "alignment",
                "bubbleDimensions",
                "bubbleValues",
                "bubblesGap",
                "bubbleFieldType",
            ],
        )
        # Setup custom props
        self.bubble_dimensions = bubble_dimensions
        self.bubble_values = bubble_values
        self.bubbles_gap = bubbles_gap
        self.bubble_field_type = bubble_field_type

        # Setup alignment
        DEFAULT_ALIGNMENT = {
            # TODO: copy defaults from template's maxDisplacement value
        }
        self.alignment = (
            alignment_object if alignment_object is not None else DEFAULT_ALIGNMENT
        )

    def setup_ocr_field_block(self, field_block_object):
        (scan_zone,) = map(
            field_block_object.get,
            [
                "scanZone",
            ],
        )
        # Setup custom props
        self.scan_zone = scan_zone
        # TODO: compute scan zone?

    def generate_fields(
        self,
    ):
        # TODO: refactor this dependency
        field_block = self
        direction = self.direction
        empty_value = self.empty_value
        field_detection_type = self.field_detection_type
        labels_gap = self.labels_gap

        _v = 0 if (direction == "vertical") else 1
        self.fields: List[Field] = []
        # Generate the bubble grid
        lead_point = [float(self.origin[0]), float(self.origin[1])]
        for field_label in self.parsed_field_labels:
            origin = lead_point.copy()
            FieldClass = self.field_detection_type_to_field_class[field_detection_type]
            self.fields.append(
                FieldClass(
                    direction,
                    empty_value,
                    field_block,
                    field_detection_type,
                    field_label,
                    origin,
                )
            )
            lead_point[_v] += labels_gap

        # TODO: validate for field block overflow outside template dimensions
        self.update_bounding_box()

    def update_bounding_box(self):
        all_scan_boxes: List[ScanBox] = []
        for field in self.fields:
            all_scan_boxes += field.scan_boxes

        # TODO: see if shapely should be used for these
        self.bounding_box_origin = [
            min(scan_box.origin[0] for scan_box in all_scan_boxes),
            min(scan_box.origin[1] for scan_box in all_scan_boxes),
        ]

        # Note: we ignore the margins of the scan boxes when visualizing
        self.bounding_box_dimensions = [
            max(
                scan_box.origin[0] + scan_box.dimensions[0]
                for scan_box in all_scan_boxes
            )
            - self.bounding_box_origin[0],
            max(
                scan_box.origin[1] + scan_box.dimensions[1]
                for scan_box in all_scan_boxes
            )
            - self.bounding_box_origin[1],
        ]
