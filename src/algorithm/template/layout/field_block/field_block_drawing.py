from src.utils.constants import CLR_BLACK
from src.utils.drawing import DrawingUtils


class FieldBlockDrawing:
    def __init__(self, field_block):
        self.field_block = field_block

    def draw_field_block(self, marked_image, shifted=True, thickness=3, border=3):
        field_block = self.field_block
        # TODO: get this field block using a bounding box of all bubbles instead. (remove shift at field block level)
        FieldBlockDrawing.draw_bounding_rectangle(
            field_block, marked_image, shifted, border
        )

        thickness_factor = 1 / 10
        for field in field_block.fields:
            field.drawing.draw_scan_boxes(
                marked_image, field_block.shifts, thickness_factor, border
            )

        FieldBlockDrawing.draw_field_block_label(
            field_block, marked_image, shifted, thickness
        )

    @staticmethod
    def draw_bounding_rectangle(field_block, marked_image, shifted, border):
        (
            bounding_box_origin,
            bounding_box_dimensions,
        ) = map(
            lambda attr: getattr(field_block, attr),
            [
                "bounding_box_origin",
                "bounding_box_dimensions",
            ],
        )
        block_position = (
            field_block.get_shifted_origin() if shifted else bounding_box_origin
        )
        if not shifted:
            DrawingUtils.draw_box(
                marked_image,
                block_position,
                bounding_box_dimensions,
                color=CLR_BLACK,
                style="BOX_HOLLOW",
                thickness_factor=0,
                border=border,
            )

    @staticmethod
    def draw_field_block_label(field_block, marked_image, shifted, thickness):
        (
            field_block_name,
            bounding_box_origin,
            bounding_box_dimensions,
        ) = map(
            lambda attr: getattr(field_block, attr),
            [
                "name",
                "bounding_box_origin",
                "bounding_box_dimensions",
            ],
        )

        block_position = (
            field_block.get_shifted_origin() if shifted else bounding_box_origin
        )
        text_position = lambda size_x, size_y: (
            int(block_position[0] + bounding_box_dimensions[0] - size_x),
            int(block_position[1] - size_y),
        )
        text = field_block_name
        if shifted:
            text = f"({field_block.shifts}){field_block_name}"

        DrawingUtils.draw_text(marked_image, text, text_position, thickness=thickness)
