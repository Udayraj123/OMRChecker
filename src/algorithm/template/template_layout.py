import os

from src.processors.manager import PROCESSOR_MANAGER
from src.utils.constants import BUILTIN_FIELD_TYPES, CUSTOM_FIELD_TYPE
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.parsing import (
    custom_sort_output_columns,
    default_dump,
    open_template_with_defaults,
    parse_fields,
)


# TODO: make a child class TemplateLayout to keep template only about the layouts & json data?
class TemplateLayout:
    def __init__(self, template_path, tuning_config):
        self.path = template_path
        # TODO: fill these for external use
        # self.fields = []
        # self.all_field_detection_types = []

        json_object = open_template_with_defaults(template_path)
        (
            custom_labels_object,
            field_blocks_object,
            alignment_object,
            output_columns_array,
            pre_processors_object,
            self.bubble_dimensions,
            self.global_empty_val,
            self.template_dimensions,
            self.options,
            self.output_image_shape,
            self.field_blocks_offset,
            custom_field_types,
        ) = map(
            json_object.get,
            [
                "customLabels",
                "fieldBlocks",
                "alignment",
                "outputColumns",
                "preProcessors",
                "bubbleDimensions",
                "emptyValue",
                "templateDimensions",
                "options",
                "outputImageShape",
                "fieldBlocksOffset",
                "customFieldTypes",
                # TODO: support for "sortFiles" key
            ],
        )

        page_width, page_height = self.template_dimensions

        self.processing_image_shape = json_object.get(
            "processingImageShape", [page_height, page_width]
        )

        self.parse_output_columns(output_columns_array)

        # TODO: move outside
        self.setup_pre_processors(pre_processors_object, template_path.parent)

        self.parse_custom_field_types(custom_field_types)
        self.validate_field_blocks(field_blocks_object)
        self.setup_field_blocks(field_blocks_object)

        self.parse_custom_labels(custom_labels_object)

        non_custom_columns, all_custom_columns = (
            list(self.non_custom_labels),
            list(custom_labels_object.keys()),
        )

        if len(self.output_columns) == 0:
            self.fill_output_columns(non_custom_columns, all_custom_columns)

        self.validate_template_columns(non_custom_columns, all_custom_columns)

        # TODO: this is dependent on other calls to finish
        self.setup_alignment(alignment_object, template_path.parent, tuning_config)

    def get_copy_for_shifting(self):
        # Copy template for this instance op
        template_layout = shallowcopy(self)
        # Make deepcopy for only parts that are mutated by Processor
        template_layout.field_blocks = [field_block.get_copy_for_shifting() for field_block in self.field_blocks]
        
        return template_layout

    # TODO: separate out preprocessing?
    def apply_preprocessors(
        self, file_path, gray_image, colored_image, original_template_layout
    ):
        config = self.tuning_config

        template_layout = original_template_layout.get_copy_for_shifting()

        # Reset the shifts in the copied template_layout
        template_layout.reset_all_shifts()

        # resize to conform to common preprocessor input requirements
        gray_image = ImageUtils.resize_to_shape(
            gray_image, template_layout.processing_image_shape
        )
        if config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.resize_to_shape(
                colored_image, template_layout.processing_image_shape
            )

        show_preprocessors_diff = config.outputs.show_preprocessors_diff
        # run pre_processors in sequence
        for pre_processor in template_layout.pre_processors:
            pre_processor_name = pre_processor.get_class_name()

            # Show Before Preview
            if show_preprocessors_diff[pre_processor_name]:
                InteractionUtils.show(
                    f"Before {pre_processor_name}: {file_path}",
                    (
                        colored_image
                        if config.outputs.colored_outputs_enabled
                        else gray_image
                    ),
                )

            # Apply filter
            (
                out_omr,
                colored_image,
                next_template_layout,
            ) = pre_processor.resize_and_apply_filter(
                gray_image, colored_image, template_layout, file_path
            )
            gray_image = out_omr
            template_layout = next_template_layout

            # Show After Preview
            if show_preprocessors_diff[pre_processor_name]:
                InteractionUtils.show(
                    f"After {pre_processor_name}: {file_path}",
                    (
                        colored_image
                        if config.outputs.colored_outputs_enabled
                        else gray_image
                    ),
                )

        if template_layout.output_image_shape:
            # resize to output requirements
            gray_image = ImageUtils.resize_to_shape(
                gray_image, template_layout.output_image_shape
            )
            if config.outputs.colored_outputs_enabled:
                colored_image = ImageUtils.resize_to_shape(
                    colored_image, template_layout.output_image_shape
                )

        return gray_image, colored_image, template_layout

    def parse_output_columns(self, output_columns_array):
        self.output_columns = parse_fields(f"Output Columns", output_columns_array)

    def setup_pre_processors(self, pre_processors_object, relative_dir):
        # load image pre_processors
        self.pre_processors = []
        for pre_processor in pre_processors_object:
            ImageTemplateProcessorClass = PROCESSOR_MANAGER.processors[
                pre_processor["name"]
            ]
            pre_processor_instance = ImageTemplateProcessorClass(
                options=pre_processor["options"],
                relative_dir=relative_dir,
                save_image_ops=self.template.save_image_ops,
                default_processing_image_shape=self.processing_image_shape,
            )
            self.pre_processors.append(pre_processor_instance)

    def parse_custom_field_types(self, custom_field_types):
        if custom_field_types is None:
            self.field_types_data = BUILTIN_FIELD_TYPES
        else:
            self.field_types_data = {
                **BUILTIN_FIELD_TYPES,
                **custom_field_types,
            }

    def validate_field_blocks(self, field_blocks_object):
        for block_name, field_block_object in field_blocks_object.items():
            field_type = field_block_object["fieldType"]
            if (
                field_type not in self.field_types_data
                and field_type != CUSTOM_FIELD_TYPE
            ):
                logger.critical(
                    f"Cannot find definition for {field_type} in customFieldTypes"
                )
                raise Exception(
                    f"Invalid field type: {field_type} in block {block_name}"
                )

    # TODO: move out to template_alignment.py
    def setup_alignment(self, alignment_object, relative_dir, tuning_config):
        self.alignment = alignment_object
        self.alignment["margins"] = alignment_object["margins"]
        self.alignment["reference_image_path"] = None
        relative_path = self.alignment.get("referenceImage", None)

        # TODO: add more setup steps here

        if relative_path is not None:
            self.alignment["reference_image_path"] = os.path.join(
                relative_dir, relative_path
            )
            # logger.debug(self.alignment)
            gray_alignment_image, colored_alignment_image = ImageUtils.read_image_util(
                self.alignment["reference_image_path"], tuning_config
            )
            # InteractionUtils.show("gray_alignment_image", gray_alignment_image)

            # TODO: shouldn't pass self
            (
                processed_gray_alignment_image,
                processed_colored_alignment_image,
                _,
            ) = self.template.apply_preprocessors(
                self.alignment["reference_image_path"],
                gray_alignment_image,
                colored_alignment_image,
                self,
            )
            # Pre-processed alignment image
            self.alignment["gray_alignment_image"] = processed_gray_alignment_image
            self.alignment["colored_alignment_image"] = (
                processed_colored_alignment_image
            )

    def setup_field_blocks(self, field_blocks_object):
        # Add field_blocks
        self.field_blocks = []
        self.all_parsed_labels = set()
        # TODO: add support for parsing "conditionalSets" with their matcher
        for block_name, field_block_object in field_blocks_object.items():
            self.parse_and_add_field_block(block_name, field_block_object)

    def parse_custom_labels(self, custom_labels_object):
        all_parsed_custom_labels = set()
        self.custom_labels = {}
        for custom_label, label_strings in custom_labels_object.items():
            parsed_labels = parse_fields(f"Custom Label: {custom_label}", label_strings)
            parsed_labels_set = set(parsed_labels)
            self.custom_labels[custom_label] = parsed_labels

            missing_custom_labels = sorted(
                parsed_labels_set.difference(self.all_parsed_labels)
            )
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
                    f"The field strings for custom label '{custom_label}' overlap with other existing custom labels"
                )

            all_parsed_custom_labels.update(parsed_labels)

        self.non_custom_labels = self.all_parsed_labels.difference(
            all_parsed_custom_labels
        )

    def fill_output_columns(self, non_custom_columns, all_custom_columns):
        all_template_columns = non_custom_columns + all_custom_columns
        # Typical case: sort alpha-numerical (natural sort)
        self.output_columns = sorted(
            all_template_columns, key=custom_sort_output_columns
        )

    def validate_template_columns(self, non_custom_columns, all_custom_columns):
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

    def parse_and_add_field_block(self, block_name, field_block_object):
        field_block_object = self.pre_fill_field_block(field_block_object)
        block_instance = FieldBlock(
            block_name, field_block_object, self.field_blocks_offset
        )
        # TODO: support custom field types like Barcode and OCR
        self.field_blocks.append(block_instance)
        self.validate_parsed_labels(field_block_object["fieldLabels"], block_instance)

    def pre_fill_field_block(self, field_block_object):
        field_type = field_block_object.get("fieldType", CUSTOM_FIELD_TYPE)
        #  TODO: support for if field_type == "BARCODE":

        if field_type in self.field_types_data:
            field_type_data = self.field_types_data[field_type]
            field_block_object = {
                **field_block_object,
                **field_type_data,
            }

        return {
            "fieldType": field_type,
            # "direction": "vertical",
            "emptyValue": self.global_empty_val,
            "bubbleDimensions": self.bubble_dimensions,
            **field_block_object,
        }

    def validate_parsed_labels(self, field_labels, block_instance):
        parsed_field_labels, block_name = (
            block_instance.parsed_field_labels,
            block_instance.name,
        )
        field_labels_set = set(parsed_field_labels)
        if not self.all_parsed_labels.isdisjoint(field_labels_set):
            overlap = field_labels_set.intersection(self.all_parsed_labels)
            # Note: in case of two fields pointing to same column, use a custom column instead of same field labels.
            logger.critical(
                f"An overlap found between field string: {field_labels} in block '{block_name}' and existing labels: {self.all_parsed_labels}"
            )
            raise Exception(
                f"The field strings for field block {block_name} overlap with other existing fields: {overlap}"
            )
        self.all_parsed_labels.update(field_labels_set)

        page_width, page_height = self.template_dimensions
        block_width, block_height = block_instance.dimensions
        [block_start_x, block_start_y] = block_instance.origin

        block_end_x, block_end_y = (
            block_start_x + block_width,
            block_start_y + block_height,
        )

        if (
            block_end_x >= page_width
            or block_end_y >= page_height
            or block_start_x < 0
            or block_start_y < 0
        ):
            raise Exception(
                f"Overflowing field block '{block_name}' with origin {block_instance.origin} and dimensions {block_instance.dimensions} in template with dimensions {self.template_dimensions}"
            )

    def reset_and_setup_for_directory(self, output_dir, output_mode):
        """Reset all mutations to the template, and setup output directories"""
        self.reset_all_shifts()
        self.reset_and_setup_outputs(output_dir, output_mode)

    def reset_all_shifts(self):
        # Note: field blocks offset is static and independent of "shifts"
        for field_block in self.field_blocks:
            field_block.reset_all_shifts()

    def reset_and_setup_outputs(self, output_dir, output_mode):
        self.directory_handler.reset_path_utils(output_dir, output_mode)

    def __str__(self):
        return str(self.path)

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "template_dimensions",
                "field_blocks",
                # Not needed as local props are overridden -
                # "bubble_dimensions",
                # 'options',
                # "global_empty_val",
            ]
        }


class FieldBlock:
    def __init__(self, block_name, field_block_object, field_blocks_offset):
        self.name = block_name
        # TODO: Move plot_bin_name into child class
        self.plot_bin_name = block_name
        self.shifts = [0, 0]
        self.setup_field_block(field_block_object, field_blocks_offset)
    
    def get_copy_for_shifting(self):
        copied_field_block = shallowcopy(self)
        # No need to deepcopy self.fields since they are not using shifts yet,
        # also we are resetting them anyway before runs.
        return copied_field_block

    def reset_all_shifts(self):
        self.shifts = [0, 0]
        for field in self.fields:
            field.reset_all_shifts()

    # Need this at runtime as we have allowed mutation of template via pre-processors
    def get_shifted_origin(self):
        origin, shifts = self.origin, self.shifts
        return [origin[0] + shifts[0], origin[1] + shifts[1]]

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "bubble_dimensions",
                "dimensions",
                "empty_value",
                "fields",
                "name",
                "origin",
                # "shifted_origin",
                # "plot_bin_name",
            ]
        }

    def setup_field_block(self, field_block_object, field_blocks_offset):
        # case mapping
        (
            alignment_object,
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            direction,
            field_labels,
            field_type,
            labels_gap,
            origin,
            empty_value,
        ) = map(
            field_block_object.get,
            [
                "alignment",
                "bubbleDimensions",
                "bubbleValues",
                "bubblesGap",
                "direction",
                "fieldLabels",
                "fieldType",
                "labelsGap",
                "origin",
                "emptyValue",
            ],
        )
        self.parsed_field_labels = parse_fields(
            f"Field Block Labels: {self.name}", field_labels
        )
        offset_x, offset_y = field_blocks_offset
        self.origin = [origin[0] + offset_x, origin[1] + offset_y]
        self.bubble_dimensions = bubble_dimensions
        self.empty_value = empty_value
        self.direction = direction
        self.bubbles_gap = bubbles_gap
        self.labels_gap = labels_gap
        # TODO: support barcode, ocr, etc custom field types
        self.field_type = field_type
        self.setup_alignment(alignment_object)

        self.calculate_block_dimensions(
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            direction,
            labels_gap,
        )
        field_block = self
        self.generate_bubble_grid(
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            direction,
            empty_value,
            field_block,
            field_type,
            labels_gap,
        )

    def setup_alignment(self, alignment_object):
        DEFAULT_ALIGNMENT = {
            # TODO: copy defaults from template's maxDisplacement value
        }
        self.alignment = (
            alignment_object if alignment_object is not None else DEFAULT_ALIGNMENT
        )

    def calculate_block_dimensions(
        self,
        bubble_dimensions,
        bubble_values,
        bubbles_gap,
        direction,
        labels_gap,
    ):
        _h, _v = (1, 0) if (direction == "vertical") else (0, 1)

        values_dimension = int(
            bubbles_gap * (len(bubble_values) - 1) + bubble_dimensions[_h]
        )
        fields_dimension = int(
            labels_gap * (len(self.parsed_field_labels) - 1) + bubble_dimensions[_v]
        )
        self.dimensions = (
            [fields_dimension, values_dimension]
            if (direction == "vertical")
            else [values_dimension, fields_dimension]
        )
        # TODO: validate for field block overflow outside template dimensions

    def generate_bubble_grid(
        self,
        bubble_dimensions,
        bubble_values,
        bubbles_gap,
        direction,
        empty_value,
        field_block,
        field_type,
        labels_gap,
    ):
        field_block = self
        _v = 0 if (direction == "vertical") else 1
        self.fields = []
        # Generate the bubble grid
        lead_point = [float(self.origin[0]), float(self.origin[1])]
        for field_label in self.parsed_field_labels:
            origin = lead_point.copy()
            self.fields.append(
                Field(
                    bubble_dimensions,
                    bubble_values,
                    bubbles_gap,
                    direction,
                    empty_value,
                    field_block,
                    field_label,
                    field_type,
                    origin,
                )
            )
            # TODO: fill this? -
            # self.field_detectioself.append(FieldInterpreter(field_label, field_type, field_bubbles, direction))
            # self.field_detector.append(FieldTypeDetector(field_label, field_type, field_bubbles, direction))
            lead_point[_v] += labels_gap


class Field:
    """
    Container for a Field on the OMR i.e. a group of FieldBubbles with a collective field_label

    """

    def __init__(
        self,
        bubble_dimensions,
        bubble_values,
        bubbles_gap,
        direction,
        empty_value,
        field_block,
        field_label,
        field_type,
        origin,
    ):
        self.bubble_dimensions = bubble_dimensions
        self.bubble_values = bubble_values
        self.bubbles_gap = bubbles_gap
        self.direction = direction
        self.empty_value = empty_value
        self.origin = origin
        # reference to get shifts at runtime
        self.field_block = field_block
        self.field_label = field_label
        self.field_type = field_type
        self.populate_bubbles()

    def populate_bubbles(self):
        _h = 1 if (self.direction == "vertical") else 0
        field = self
        bubble_point = self.origin.copy()
        field_bubbles = []
        for bubble_index, bubble_value in enumerate(self.bubble_values):
            bubble_origin = bubble_point.copy()
            field_bubbles.append(
                FieldBubble(bubble_index, bubble_origin, bubble_value, field)
            )
            bubble_point[_h] += self.bubbles_gap
        self.field_bubbles = field_bubbles

    def reset_all_shifts(self):
        # Note: no shifts needed at bubble level
        for bubble in self.field_bubbles:
            bubble.reset_shifts()

    def __str__(self):
        return self.field_label

    # Make the class serializable
    def to_json(self):
        return {
            key: default_dump(getattr(self, key))
            for key in [
                "field_label",
                "field_type",
                "direction",
                "field_bubbles",
            ]
        }


class FieldBubble:
    """
    Container for a Point Box on the OMR

    field_label is the point's property- field to which this point belongs to
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """

    def __init__(self, bubble_origin, bubble_index, bubble_value, field):

        self.field = field
        self.field_label = field.field_label
        self.field_type = field.field_type
        self.bubble_dimensions = field.bubble_dimensions

        self.name = f"{self.field_label}_{bubble_value}"
        self.plot_bin_name = self.field_label
        self.x = round(bubble_origin[0])
        self.y = round(bubble_origin[1])
        self.shifts = [0, 0]
        self.bubble_value = bubble_value
        self.bubble_index = bubble_index

    def __str__(self):
        return self.name  # f"{self.field_label}: [{self.x}, {self.y}]"

    def reset_shifts(self):
        self.shifts = [0, 0]

    def get_shifted_position(self, shifts = None):
        # field_shifts = self.field.shifts
        if shifts is None:
            shifts = self.field.field_block.shifts
        return [
            self.x + self.shifts[0] + shifts[0],
            self.y + self.shifts[1] + shifts[1],
        ]

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
                # "plot_bin_name",
                # "field_type",
                # "bubble_index",
                # "bubble_dimensions",
            ]
        }
