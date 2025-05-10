from abc import abstractmethod

from src.algorithm.template.detection.base.interpretation import FieldInterpretation


class FieldInterpretationDrawing:
    def __init__(
        self,
        field_interpretation: FieldInterpretation,
    ):
        self.field_interpretation = field_interpretation
        self.tuning_config = field_interpretation.tuning_config
        self.field = field_interpretation.field

    @abstractmethod
    def draw_field_interpretation(
        self, marked_image, image_type, evaluation_meta, evaluation_config_for_response
    ):
        raise Exception("Not implemented")
