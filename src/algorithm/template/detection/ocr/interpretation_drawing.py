from src.algorithm.template.detection.base.interpretation_drawing import (
    FieldInterpretationDrawing,
)


class OCRFieldInterpretationDrawing(FieldInterpretationDrawing):
    def __init__(self, field_interpretation):
        super().__init__(field_interpretation)

    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config_for_response
    ):
        # field_label = self.field.field_label
        return

    @staticmethod
    def draw_field_interpretation_util():
        pass
