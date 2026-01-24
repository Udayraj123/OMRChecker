from src.processors.layout.field.base import Field, ScanBox
from src.processors.layout.field.field_drawing import BubbleFieldDrawing
from src.utils.constants import ZERO_MARGINS
from src.utils.parsing import default_dump


class BubbleField(Field):
    def __init__(
        # ruff: noqa: PLR0913
        self,
        direction,
        empty_value,
        field_block,
        field_detection_type,
        field_label,
        origin,
    ) -> None:
        # Call super().__init__() first (proper initialization order)
        super().__init__(
            direction,
            empty_value,
            field_block,
            field_detection_type,
            field_label,
            origin,
        )

        # Set subclass-specific properties after super().__init__()
        # This ensures proper initialization order and avoids accessing self before parent init
        self.bubble_dimensions = field_block.bubble_dimensions
        self.bubble_values = field_block.bubble_values
        self.bubbles_gap = field_block.bubbles_gap
        self.bubble_field_type = field_block.bubble_field_type
        # self.plot_bin_name = field_label

    def get_drawing_instance(self):
        return BubbleFieldDrawing(self)

    def setup_scan_boxes(self, field_block) -> None:
        # Use field_block parameters instead of self properties
        # This allows setup_scan_boxes to work even if called before subclass properties are set
        bubble_values = field_block.bubble_values
        bubble_dimensions = field_block.bubble_dimensions
        bubbles_gap = field_block.bubbles_gap
        bubble_field_type = field_block.bubble_field_type

        if not bubble_values:
            raise ValueError("bubble_values is required and must not be empty")

        # populate the field bubbles
        h = 1 if (self.direction == "vertical") else 0

        bubble_point = self.origin.copy()
        self.scan_boxes: list[BubblesScanBox] = []

        # Temporarily set properties so BubblesScanBox constructor can access them.
        # setup_scan_boxes() is called from base Field constructor (before BubbleField sets these).
        # BubblesScanBox constructor accesses field.bubble_dimensions and field.bubble_field_type.
        # These will be set again (to the same values) after super() returns in BubbleField constructor.
        self.bubble_dimensions = bubble_dimensions
        self.bubble_field_type = bubble_field_type

        for field_index, bubble_value in enumerate(bubble_values):
            bubble_origin = bubble_point.copy()
            scan_box = BubblesScanBox(field_index, self, bubble_origin, bubble_value)
            self.scan_boxes.append(scan_box)
            bubble_point[h] += bubbles_gap

        # Note: We don't restore original values because they'll be overwritten in the constructor anyway.


class BubblesScanBox(ScanBox):
    """(TODO: update docs).

    Container for a Point Box on the OMR
    field_label is the point's property- field to which this point belongs to
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1).
    """

    def __init__(self, field_index, field: BubbleField, origin, bubble_value) -> None:
        dimensions = field.bubble_dimensions
        margins = ZERO_MARGINS
        super().__init__(field_index, field, origin, dimensions, margins)
        self.bubble_field_type = field.bubble_field_type
        self.bubble_dimensions = field.bubble_dimensions

        self.shifts = [0, 0]
        self.bubble_value = bubble_value
        self.name = f"{self.field_label}_{self.bubble_value}"
        self.plot_bin_name = self.field_label

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "field_label",
                "bubble_value",
                # for item_reference_name
                "name",
                "x",
                "y",
                "origin",
                # "plot_bin_name",
                # "bubble_field_type",
                # "field_index",
                # "bubble_dimensions",
            ]
        }
