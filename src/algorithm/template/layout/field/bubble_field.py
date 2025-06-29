from src.algorithm.template.layout.field.base import Field, ScanBox
from src.algorithm.template.layout.field.field_drawing import BubbleFieldDrawing
from src.utils.constants import ZERO_MARGINS
from src.utils.parsing import default_dump


class BubbleField(Field):
    def __init__(
        self,
        direction,
        empty_value,
        field_block,
        field_detection_type,
        field_label,
        origin,
    ):
        self.bubble_dimensions = field_block.bubble_dimensions
        self.bubble_values = field_block.bubble_values
        self.bubbles_gap = field_block.bubbles_gap
        self.bubble_field_type = field_block.bubble_field_type
        # self.plot_bin_name = field_label

        super().__init__(
            direction,
            empty_value,
            field_block,
            field_detection_type,
            field_label,
            origin,
        )

    def get_drawing_instance(self):
        return BubbleFieldDrawing(self)

    def setup_scan_boxes(self, field_block):
        # populate the field bubbles
        _h = 1 if (self.direction == "vertical") else 0

        field = self
        bubble_point = self.origin.copy()
        scan_boxes: list[BubblesScanBox] = []
        for field_index, bubble_value in enumerate(self.bubble_values):
            bubble_origin = bubble_point.copy()
            scan_box = BubblesScanBox(field_index, field, bubble_origin, bubble_value)
            scan_boxes.append(scan_box)
            bubble_point[_h] += self.bubbles_gap

        self.scan_boxes = scan_boxes


class BubblesScanBox(ScanBox):
    """(TODO: update docs)
    Container for a Point Box on the OMR
    field_label is the point's property- field to which this point belongs to
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """

    def __init__(self, field_index, field: BubbleField, origin, bubble_value):
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
