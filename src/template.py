"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

from src.constants import FIELD_TYPES
from src.core import ImageInstanceOps
from src.logger import logger
from src.processors.manager import PROCESSOR_MANAGER
from src.utils.parsing import (
    custom_sort_output_columns,
    open_template_with_defaults,
    parse_fields,
)


class Bubble:
    """
    Container for a Point Box on the OMR

    field_label is the point's property- question to which this point belongs to
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """

    def __init__(self, pt, field_label, field_type, field_value):
        self.x = round(pt[0])
        self.y = round(pt[1])
        self.field_label = field_label
        self.field_type = field_type
        self.field_value = field_value

    def __str__(self):
        return str([self.x, self.y])


class FieldBlock:
    def __init__(
        self,
        # TODO: reduce these params?
        block_name,
        bubble_dimensions,
        bubble_values,
        bubbles_gap,
        direction,
        empty_val,
        field_labels,
        field_type,
        labels_gap,
        origin,
    ):
        self.name = block_name
        self.empty_val = empty_val
        self.origin = origin
        self.shift = 0

        _h, _v = (1, 0) if (direction == "vertical") else (0, 1)

        num_values = len(bubble_values)
        num_fields = len(field_labels)
        values_dimension = bubbles_gap * (num_values - 1) + bubble_dimensions[_h]
        fields_dimension = labels_gap * (num_fields - 1) + bubble_dimensions[_v]
        q_block_dims = (
            [fields_dimension, values_dimension]
            if (direction == "vertical")
            else [values_dimension, fields_dimension]
        )

        traverse_bubbles = []
        o = [float(origin[0]), float(origin[1])]

        for field_label in field_labels:
            pt = o.copy()
            field_bubbles = []
            for bubble_value in bubble_values:
                field_bubbles.append(
                    Bubble(pt.copy(), field_label, field_type, bubble_value)
                )
                pt[_h] += bubbles_gap
            # For diagonal endpoint of the Field Block
            pt[_h] += bubble_dimensions[_h] - bubbles_gap
            pt[_v] += bubble_dimensions[_v]

            block_bounding_box = [o.copy(), pt.copy()]
            traverse_bubbles.append((block_bounding_box, field_bubbles))

            o[_v] += labels_gap

        self.traverse_bubbles = traverse_bubbles
        self.dimensions = tuple(round(x) for x in q_block_dims)


class Template:
    def __init__(self, template_path, tuning_config):
        self.image_instance_ops = ImageInstanceOps(tuning_config)
        json_obj = open_template_with_defaults(template_path)
        self.path = template_path
        (
            self.bubble_dimensions,
            self.global_empty_val,
            self.options,
            self.output_columns,
            self.page_dimensions,
        ) = map(
            json_obj.get,
            [
                "bubbleDimensions",
                "emptyVal",
                "options",
                "outputColumns",
                "pageDimensions",
            ],
        )
        # TODO: refactor into smaller functions

        # load image pre_processors
        self.pre_processors = []
        for pre_processor in json_obj["preProcessors"]:
            ProcessorClass = PROCESSOR_MANAGER.processors[pre_processor["name"]]
            pre_processor_instance = ProcessorClass(
                options=pre_processor["options"],
                relative_dir=template_path.parent,
                image_instance_ops=self.image_instance_ops,
            )
            self.pre_processors.append(pre_processor_instance)

        # Add field_blocks
        self.field_blocks = []
        self.all_parsed_labels = set()
        for block_name, field_block in json_obj["fieldBlocks"].items():
            self.parse_and_add_field_block(block_name, field_block)

        all_parsed_custom_labels = set()
        self.custom_labels = {}
        for custom_label, label_strings in json_obj["customLabels"].items():
            parsed_labels = parse_fields(f"Custom Label: {custom_label}", label_strings)
            parsed_labels_set = set(parsed_labels)
            self.custom_labels[custom_label] = parsed_labels

            missing_custom_labels = parsed_labels_set.difference(self.all_parsed_labels)
            if len(missing_custom_labels) > 0:
                logger.critical(
                    f"For '{custom_label}', Missing labels - {missing_custom_labels}"
                )
                raise Exception(
                    f"Missing field block label(s) in the given template for {missing_custom_labels} from '{custom_label}'"
                )

            if not all_parsed_custom_labels.isdisjoint(parsed_labels_set):
                # Note: this can be made a warning, but it's a choice
                logger.critical(
                    f"field strings overlap for labels: {label_strings} and existing custom labels: {all_parsed_custom_labels}"
                )
                raise Exception(
                    f"The field strings for custom label {custom_label} overlap with other existing custom labels"
                )

            all_parsed_custom_labels.update(parsed_labels)

        self.non_custom_labels = self.all_parsed_labels.difference(
            all_parsed_custom_labels
        )

        non_custom_columns, all_custom_columns = (
            list(self.non_custom_labels),
            list(json_obj["customLabels"].keys()),
        )

        if len(self.output_columns) == 0:
            all_template_columns = non_custom_columns + all_custom_columns
            # Typical case: sort alpha-numerical (natural sort)
            self.output_columns = sorted(
                all_template_columns, key=custom_sort_output_columns
            )
        else:
            output_columns_set = set(self.output_columns)
            all_custom_columns_set = set(all_custom_columns)
            missing_output_columns = sorted(
                output_columns_set.difference(all_custom_columns_set).difference(
                    self.all_parsed_labels
                )
            )
            if len(missing_output_columns) > 0:
                logger.critical(f"Missing output columns: {missing_output_columns}")
                raise Exception(
                    f"Some columns are missing in the field blocks for the given output columns"
                )

            all_template_columns_set = set(non_custom_columns + all_custom_columns)
            missing_label_columns = sorted(
                all_template_columns_set.difference(output_columns_set)
            )
            if len(missing_label_columns) > 0:
                logger.warning(
                    f"Some label columns are not covered in the given output columns: {missing_label_columns}"
                )

    # Expects bubble_dimensions to be set already
    def parse_and_add_field_block(self, block_name, field_block):
        if "fieldType" in field_block:
            field_block = {**field_block, **FIELD_TYPES[field_block["fieldType"]]}
        else:
            field_block = {**field_block, "fieldType": "__CUSTOM__"}

        field_block = {
            **{
                "direction": "vertical",
                "emptyVal": self.global_empty_val,
                "bubbleDimensions": self.bubble_dimensions,
            },
            **field_block,
        }

        # case mapping
        (
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            direction,
            empty_val,
            field_labels,
            field_type,
            labels_gap,
            origin,
        ) = map(
            field_block.get,
            [
                "bubbleDimensions",
                "bubbleValues",
                "bubblesGap",
                "direction",
                "emptyVal",
                "fieldLabels",
                "fieldType",
                "labelsGap",
                "origin",
            ],
        )

        parsed_field_labels = parse_fields(
            f"Field Block Labels: {block_name}", field_labels
        )
        field_labels_set = set(parsed_field_labels)
        if not self.all_parsed_labels.isdisjoint(field_labels_set):
            logger.critical(
                f"An overlap found between field string: {field_labels} in block '{block_name}' and existing labels: {self.all_parsed_labels}"
            )
            raise Exception(
                f"The field strings for field block {block_name} overlap with other existing fields"
            )
        self.all_parsed_labels.update(field_labels_set)

        block_instance = FieldBlock(
            block_name,
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            direction,
            empty_val,
            parsed_field_labels,
            field_type,
            labels_gap,
            origin,
        )

        page_width, page_height = self.page_dimensions
        block_width, block_height = block_instance.dimensions
        [block_start_x, block_start_y] = block_instance.origin

        block_end_x, block_end_y = (
            block_start_x + block_width,
            block_start_y + block_height,
        )

        if (
            block_end_x >= page_width
            or block_end_y >= page_height
            or block_start_x <= 0
            or block_start_y <= 0
        ):
            raise Exception(
                f"Field block '{block_name}' with origin {block_instance.origin} and dimensions {block_instance.dimensions} overflows the template dimensions {self.page_dimensions}"
            )

        self.field_blocks.append(block_instance)

    def __str__(self):
        return str(self.path)
