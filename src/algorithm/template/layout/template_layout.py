from copy import copy as shallowcopy
from pathlib import Path
from typing import TYPE_CHECKING

from src.algorithm.template.layout.field_block.base import FieldBlock
from src.processors.constants import FieldDetectionType
from src.processors.manager import PROCESSOR_MANAGER
from src.utils.constants import BUILTIN_BUBBLE_FIELD_TYPES
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger
from src.utils.parsing import (
    custom_sort_output_columns,
    default_dump,
    open_template_with_defaults,
    parse_fields,
)

if TYPE_CHECKING:
    from src.algorithm.template.layout.field.base import Field


class TemplateLayout:
    # TODO: pass 'set_layout' arg as part of 'output_mode' value 'set_layout' and consume it in template + pre_processors
    def __init__(self, template, template_path, tuning_config) -> None:
        self.path = template_path
        self.template = template
        self.tuning_config = tuning_config

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
            custom_bubble_field_types,
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
                "customBubbleFieldTypes",
                # TODO: support for "sortFiles" key
            ],
        )

        page_width, page_height = self.template_dimensions

        # Default processingImageShape will be the page dimensions
        self.processing_image_shape = json_object.get(
            "processingImageShape", [page_height, page_width]
        )

        self.parse_output_columns(output_columns_array)

        # TODO: move outside
        self.setup_pre_processors(pre_processors_object, template_path.parent)

        self.parse_custom_bubble_field_types(custom_bubble_field_types)
        self.validate_field_blocks(field_blocks_object)
        self.setup_layout(field_blocks_object)

        self.parse_custom_labels(custom_labels_object)

        non_custom_columns, all_custom_columns = (
            list(self.non_custom_labels),
            list(custom_labels_object.keys()),
        )

        if len(self.output_columns) == 0:
            self.fill_output_columns(non_custom_columns, all_custom_columns)

        self.validate_template_columns(non_custom_columns, all_custom_columns)

        # TODO: this is dependent on other calls to finish
        self.setup_alignment(alignment_object, template_path.parent)

    def get_copy_for_shifting(self):
        # Copy template for this instance op
        template_layout = shallowcopy(self)
        # Make deepcopy for only parts that are mutated by Processor
        template_layout.field_blocks = [
            field_block.get_copy_for_shifting() for field_block in self.field_blocks
        ]

        return template_layout

    # TODO: separate out preprocessing into a class?
    def apply_preprocessors(self, file_path, gray_image, colored_image):
        config = self.tuning_config

        next_template_layout = self.get_copy_for_shifting()

        # Reset the shifts in the copied next_template_layout
        next_template_layout.reset_all_shifts()

        # resize to conform to common preprocessor input requirements
        gray_image = ImageUtils.resize_to_shape(
            next_template_layout.processing_image_shape, gray_image
        )
        if config.outputs.colored_outputs_enabled:
            colored_image = ImageUtils.resize_to_shape(
                next_template_layout.processing_image_shape, colored_image
            )

        show_preprocessors_diff = config.outputs.show_preprocessors_diff
        # run pre_processors in sequence
        for pre_processor in next_template_layout.pre_processors:
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
                gray_image,
                colored_image,
                next_template_layout,
            ) = pre_processor.resize_and_apply_filter(
                gray_image, colored_image, next_template_layout, file_path
            )

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

        template_layout = next_template_layout

        if template_layout.output_image_shape:
            # resize to output requirements
            gray_image = ImageUtils.resize_to_shape(
                template_layout.output_image_shape, gray_image
            )
            if config.outputs.colored_outputs_enabled:
                colored_image = ImageUtils.resize_to_shape(
                    template_layout.output_image_shape, colored_image
                )

        return gray_image, colored_image, template_layout

    def parse_output_columns(self, output_columns_array) -> None:
        self.output_columns = parse_fields("Output Columns", output_columns_array)

    def setup_pre_processors(self, pre_processors_object, relative_dir) -> None:
        # load image pre_processors
        self.pre_processors = []
        for pre_processor in pre_processors_object:
            # ruff: noqa: N806
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

    def parse_custom_bubble_field_types(self, custom_bubble_field_types) -> None:
        if custom_bubble_field_types is None:
            self.bubble_field_types_data = BUILTIN_BUBBLE_FIELD_TYPES
        else:
            self.bubble_field_types_data = {
                **BUILTIN_BUBBLE_FIELD_TYPES,
                **custom_bubble_field_types,
            }

    def validate_field_blocks(self, field_blocks_object) -> None:
        for block_name, field_block_object in field_blocks_object.items():
            # TODO: Check for validations if any for OCR
            if (
                field_block_object["fieldDetectionType"]
                == FieldDetectionType.BUBBLES_THRESHOLD
            ):
                bubble_field_type = field_block_object["bubbleFieldType"]
                if bubble_field_type not in self.bubble_field_types_data:
                    logger.critical(
                        f"Cannot find definition for {bubble_field_type} in customBubbleFieldTypes"
                    )
                    msg = f"Invalid bubble field type: {bubble_field_type} in block {block_name}"
                    raise Exception(msg)
            field_labels = field_block_object["fieldLabels"]
            if len(field_labels) > 1 and "labelsGap" not in field_block_object:
                logger.critical(
                    f"More than one fieldLabels({field_labels}) provided, but labelsGap not present for block {block_name}"
                )
                msg = f"More than one fieldLabels provided, but labelsGap not present for block {block_name}"
                raise Exception(msg)

    # TODO: move out to template_alignment.py
    def setup_alignment(self, alignment_object, relative_dir) -> None:
        tuning_config = self.tuning_config
        self.alignment = alignment_object
        self.alignment["margins"] = alignment_object["margins"]
        self.alignment["reference_image_path"] = None
        relative_path = self.alignment.get("referenceImage", None)

        # TODO: add more setup steps here

        if relative_path is not None:
            self.alignment["reference_image_path"] = Path(relative_dir, relative_path)
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
            ) = self.apply_preprocessors(
                self.alignment["reference_image_path"],
                gray_alignment_image,
                colored_alignment_image,
            )
            # Pre-processed alignment image
            self.alignment["gray_alignment_image"] = processed_gray_alignment_image
            self.alignment["colored_alignment_image"] = (
                processed_colored_alignment_image
            )

    def setup_layout(self, field_blocks_object) -> None:
        # TODO: try for better readability here
        self.all_fields: list[Field] = []
        all_field_detection_types = set()
        # TODO: see if labels part can be moved out of template layout?
        self.all_parsed_labels = set()
        # Add field_blocks
        self.field_blocks: list[FieldBlock] = []
        # TODO: add support for parsing "conditionalSets" with their matcher
        for block_name, field_block_object in field_blocks_object.items():
            block_instance = self.parse_and_add_field_block(
                block_name, field_block_object
            )
            # TODO: validation for duplicate field labels?
            self.all_fields.extend(block_instance.fields)
            all_field_detection_types.add(block_instance.field_detection_type)

        self.all_field_detection_types = list(all_field_detection_types)

    # TODO: see if labels part can be moved out of template layout?
    def parse_custom_labels(self, custom_labels_object) -> None:
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
                msg = f"Missing field block label(s) in the given template for {missing_custom_labels} from '{custom_label}'"
                raise Exception(msg)

            if not all_parsed_custom_labels.isdisjoint(parsed_labels_set):
                # Note: this can be made a warning, but it's a choice
                logger.critical(
                    f"field strings overlap for labels: {label_strings} and existing custom labels: {all_parsed_custom_labels}"
                )
                msg = f"The field strings for custom label '{custom_label}' overlap with other existing custom labels"
                raise Exception(msg)

            all_parsed_custom_labels.update(parsed_labels)

        self.non_custom_labels = self.all_parsed_labels.difference(
            all_parsed_custom_labels
        )

    def get_concatenated_omr_response(self, raw_omr_response):
        # Multi-column/multi-row questions which need to be concatenated
        concatenated_omr_response = {}
        for field_label, concatenate_keys in self.custom_labels.items():
            custom_label = "".join([raw_omr_response[k] for k in concatenate_keys])
            concatenated_omr_response[field_label] = custom_label

        for field_label in self.non_custom_labels:
            concatenated_omr_response[field_label] = raw_omr_response[field_label]

        return concatenated_omr_response

    def fill_output_columns(self, non_custom_columns, all_custom_columns) -> None:
        all_template_columns = non_custom_columns + all_custom_columns
        # Typical case: sort alpha-numerical (natural sort)
        self.output_columns = sorted(
            all_template_columns, key=custom_sort_output_columns
        )

    def validate_template_columns(self, non_custom_columns, all_custom_columns) -> None:
        output_columns_set = set(self.output_columns)
        all_custom_columns_set = set(all_custom_columns)

        missing_output_columns = sorted(
            output_columns_set.difference(all_custom_columns_set).difference(
                self.all_parsed_labels
            )
        )
        if len(missing_output_columns) > 0:
            logger.critical(f"Missing output columns: {missing_output_columns}")
            msg = "Some columns are missing in the field blocks for the given output columns"
            raise Exception(msg)

        all_template_columns_set = set(non_custom_columns + all_custom_columns)
        missing_label_columns = sorted(
            all_template_columns_set.difference(output_columns_set)
        )
        if len(missing_label_columns) > 0:
            logger.warning(
                f"Some label columns are not covered in the given output columns: {missing_label_columns}"
            )

    def parse_and_add_field_block(self, block_name, field_block_object):
        field_block_object = self.prefill_field_block(field_block_object)
        block_instance = FieldBlock(
            block_name, field_block_object, self.field_blocks_offset
        )
        # TODO: support custom field types like Barcode and OCR
        self.field_blocks.append(block_instance)
        self.validate_parsed_field_block(
            field_block_object["fieldLabels"], block_instance
        )
        return block_instance

    def prefill_field_block(self, field_block_object):
        filled_field_block_object = {
            **field_block_object,
        }

        if (
            field_block_object["fieldDetectionType"]
            == FieldDetectionType.BUBBLES_THRESHOLD
        ):
            bubble_field_type = field_block_object["bubbleFieldType"]
            field_type_data = self.bubble_field_types_data[bubble_field_type]
            filled_field_block_object = {
                "bubbleFieldType": bubble_field_type,
                # "direction": "vertical",
                "emptyValue": self.global_empty_val,
                "bubbleDimensions": self.bubble_dimensions,
                **filled_field_block_object,
                **field_type_data,
            }
        elif (
            field_block_object["fieldDetectionType"] == FieldDetectionType.OCR
            or field_block_object["fieldDetectionType"] == FieldDetectionType.BARCODE_QR
        ):
            filled_field_block_object = {
                "emptyValue": self.global_empty_val,
                "labelsGap": 0,
                **filled_field_block_object,
            }

        return filled_field_block_object

    def validate_parsed_field_block(self, field_labels, block_instance) -> None:
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
            msg = f"The field strings for field block {block_name} overlap with other existing fields: {overlap}"
            raise Exception(msg)
        self.all_parsed_labels.update(field_labels_set)

        page_width, page_height = self.template_dimensions
        block_width, block_height = block_instance.bounding_box_dimensions
        [block_start_x, block_start_y] = block_instance.bounding_box_origin

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
            msg = f"Overflowing field block '{block_name}' with origin {block_instance.bounding_box_origin} and dimensions {block_instance.bounding_box_dimensions} in template with dimensions {self.template_dimensions}"
            raise Exception(msg)

    def reset_all_shifts(self) -> None:
        # Note: field blocks offset is static and independent of "shifts"
        for field_block in self.field_blocks:
            field_block.reset_all_shifts()

    def __str__(self) -> str:
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
