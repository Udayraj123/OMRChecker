from abc import abstractmethod
from typing import List

from src.utils.parsing import default_dump


class Field:
    """
    Container for a Field on the OMR i.e. a group of ScanBoxes with a collective field_label

    """

    def __init__(
        self,
        direction,
        empty_value,
        field_block,
        field_detection_type,
        field_label,
        origin,
    ):
        self.direction = direction
        self.empty_value = empty_value

        # reference to get shifts at runtime
        self.field_block = field_block

        self.field_detection_type = field_detection_type
        self.field_label = field_label
        self.id = f"{field_block.name}::{field_label}"
        self.name = field_label
        self.plot_bin_name = field_label

        self.origin = origin

        self.scan_boxes: List[ScanBox] = []
        # Child class would populate scan_boxes
        self.setup_scan_boxes(field_block)
        self.drawing = self.get_drawing_instance()

    @abstractmethod
    def setup_scan_boxes(self, field_block):
        raise Exception("Not implemented")

    @abstractmethod
    def get_drawing_instance(self):
        raise Exception("Not implemented")

    def reset_all_shifts(self):
        # Note: no shifts needed at bubble level
        for bubble in self.scan_boxes:
            bubble.reset_shifts()

    def __str__(self):
        return self.id

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "id",
                "field_label",
                "field_detection_type",
                "direction",
                "scan_boxes",
            ]
        }


class ScanBox:
    """
    The smallest unit in the template layout.
    TODO: update docs
    """

    def __init__(self, field_index, field: Field, origin, dimensions, margins):
        self.field_index = field_index
        self.dimensions = dimensions
        self.margins = margins
        self.origin = origin

        # self.bounding_box_dimensions = [
        #     origin[0] + margins["left"] + margins["right"] + dimensions[0],
        #     origin[1] + margins["top"] + margins["bottom"] + dimensions[1],
        # ]

        self.x = round(origin[0])
        self.y = round(origin[1])
        self.field = field
        self.field_label = field.field_label
        self.field_detection_type = field.field_detection_type
        self.name = f"{self.field_label}_{self.field_index}"

    def __str__(self):
        return self.name

    def reset_shifts(self):
        self.shifts = [0, 0]

    def get_shifted_position(self, shifts=None):
        if shifts is None:
            shifts = self.field.field_block.shifts
        return [
            self.x + self.shifts[0] + shifts[0],
            self.y + self.shifts[1] + shifts[1],
        ]

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "field_label",
                "field_detection_type",
                # for item_reference_name
                "name",
                "x",
                "y",
                "origin",
            ]
        }
