class TemplateDrawing:
    def __init__(self, template):
        self.template = template

    def draw_field_blocks_layout(
        self, image, shifted=True, shouldCopy=True, thickness=3, border=3
    ):
        template = self.template
        return TemplateDrawing.draw_field_blocks_layout_util(
            template, image, shifted, shouldCopy, thickness, border
        )

    @staticmethod
    def draw_field_blocks_layout_util(
        template, image, shifted=True, shouldCopy=True, thickness=3, border=3
    ):
        marked_image = image.copy() if shouldCopy else image
        for field_block in template.field_blocks:
            field_block.drawing.draw_field_block(
                marked_image, shifted, thickness, border
            )

        return marked_image
