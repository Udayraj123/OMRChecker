class FieldInterpretationDrawing:
    def __init__(
        self,
        field_interpretation,
    ):
        self.field_interpretation = field_interpretation
        self.tuning_config = field_interpretation.tuning_config
        self.field = field_interpretation.field
