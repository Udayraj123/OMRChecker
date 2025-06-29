from src.algorithm.template.layout.field.base import Field, ScanBox
from src.algorithm.template.layout.field.field_drawing import BarcodeFieldDrawing
from src.utils.parsing import default_dump
from src.utils.shapes import ShapeUtils


class BarcodeField(Field):
    def __init__(
        self,
        direction,
        empty_value,
        field_block,
        field_detection_type,
        field_label,
        origin,
    ):
        super().__init__(
            direction,
            empty_value,
            field_block,
            field_detection_type,
            field_label,
            origin,
        )

    def get_drawing_instance(self):
        return BarcodeFieldDrawing(self)

    def setup_scan_boxes(self, field_block):
        scan_zone = field_block.scan_zone
        origin = field_block.origin
        field = self
        # TODO: support for multiple scan zones per field (grid structure)
        field_index = 0
        scan_box = BarcodeScanBox(field_index, field, origin, scan_zone)
        self.scan_boxes: list[BarcodeScanBox] = [scan_box]

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "field_label",
                "direction",
                "scan_boxes",
            ]
        }


class BarcodeScanBox(ScanBox):
    def __init__(self, field_index, field: BarcodeField, origin, scan_zone):
        dimensions = scan_zone["dimensions"]
        margins = scan_zone["margins"]
        super().__init__(field_index, field, origin, dimensions, margins)
        self.zone_description = {"origin": origin, "label": self.name, **scan_zone}
        # Compute once for reuse
        self.scan_zone_rectangle = ShapeUtils.compute_scan_zone_rectangle(
            self.zone_description, include_margins=True
        )
