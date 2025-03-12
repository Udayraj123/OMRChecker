from src.utils.drawing import DrawingUtils


class FieldDrawing:
    def __init__(self, field):
        self.field = field

    def draw_scan_boxes(self, marked_image, shifts, thickness_factor, border):
        field = self.field
        FieldDrawing.draw_scan_boxes_util(
            field, marked_image, shifts, thickness_factor, border
        )

    @staticmethod
    def draw_scan_boxes_util(field, marked_image, shifts, thickness_factor, border):
        scan_boxes = field.scan_boxes
        for unit_bubble in scan_boxes:
            shifted_position = unit_bubble.get_shifted_position(shifts)
            dimensions = unit_bubble.dimensions
            DrawingUtils.draw_box(
                marked_image,
                shifted_position,
                dimensions,
                thickness_factor,
                border=border,
            )


class BubbleFieldDrawing(FieldDrawing):
    pass


class OCRFieldDrawing(FieldDrawing):
    # TODO: Implement custom drawing of the layout for OCR fields
    pass
