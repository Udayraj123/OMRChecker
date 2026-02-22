from copy import copy as shallowcopy
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.exceptions import OMRCheckerError
from src.processors.constants import FieldDetectionType
from src.processors.layout.field_block.base import FieldBlock
from src.processors.manager import PROCESSOR_MANAGER
from src.schemas.models.template import OutputColumnsConfig
from src.utils.constants import BUILTIN_BUBBLE_FIELD_TYPES
from src.utils.image import ImageUtils
from src.utils.logger import logger
from src.utils.parsing import (
    alphanumerical_sort_key,
    default_dump,
    open_template_with_defaults,
    parse_fields,
)

if TYPE_CHECKING:
    from src.processors.layout.field.base import Field


class TemplateLayout:
    # TODO: pass 'set_layout' arg as part of 'output_mode' value 'set_layout' and consume it in template + pre_processors
    def __init__(self, template, template_path, tuning_config) -> None:
        self.path = template_path
        self.template = template
        self.tuning_config = tuning_config
        self.template_dimensions: list[int] = [0, 0]

        # Set circular reference immediately for preprocessors that may need it
        # during initialization (e.g., alignment preprocessing)
        self.template.template_layout = self

        template_config = open_template_with_defaults(template_path)
        # Required properties
        self.bubble_dimensions = template_config.bubble_dimensions
        self.template_dimensions = template_config.template_dimensions
        # Properties with defaults
        field_blocks_object = template_config.field_blocks
        pre_processors_object = template_config.pre_processors
        alignment_object = template_config.alignment
        custom_bubble_field_types = template_config.custom_bubble_field_types
        custom_labels_object = template_config.custom_labels
        output_columns_config = template_config.output_columns
        self.field_blocks_offset = template_config.field_blocks_offset
        self.global_empty_val = template_config.empty_value

        # Properties without defaults
        page_width, page_height = self.template_dimensions
        # Use processing_image_shape from config, or default to page dimensions
        self.processing_image_shape = template_config.processing_image_shape or [
            page_height,
            page_width,
        ]
        self.output_image_shape = template_config.output_image_shape
        # TODO: support for "sortFiles" key

        self.parse_output_columns(output_columns_config)

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
            self.fill_output_columns(
                non_custom_columns, all_custom_columns, output_columns_config
            )

        self.validate_template_columns(non_custom_columns, all_custom_columns)

        # TODO: this is dependent on other calls to finish
        self.setup_alignment(alignment_object, template_path.parent)

    def get_exclude_files(self):
        excluded_files = []
        if self.alignment["reference_image_path"] is not None:
            # Note: reference_image_path is already Path()
            excluded_files.append(self.alignment["reference_image_path"])
        return excluded_files

    def get_copy_for_shifting(self):
        # Copy template for this instance op
        template_layout = shallowcopy(self)
        # Make deepcopy for only parts that are mutated by Processor
        template_layout.field_blocks = [
            field_block.get_copy_for_shifting() for field_block in self.field_blocks
        ]

        return template_layout

    def parse_output_columns(self, output_columns_config: "OutputColumnsConfig"):
        custom_order = output_columns_config.custom_order
        sort_type = output_columns_config.sort_type

        # Make sure sort_type is set to CUSTOM if output columns are custom
        if len(custom_order) > 0 and sort_type != "CUSTOM":
            logger.critical(
                "Custom output columns are passed but sort_type is not"
                "CUSTOM: {sort_type}. Please set sortType to CUSTOM in outputColumns."
            )
            msg = f"Invalid sort type: {sort_type} for custom columns"
            raise OMRCheckerError(
                msg,
                context={"sort_type": sort_type},
            )

        self.output_columns = parse_fields("Output Columns", custom_order)

    def setup_pre_processors(self, pre_processors_object, relative_dir) -> None:
        # load image pre_processors
        self.pre_processors = []
        for pre_processor in pre_processors_object:
            # ruff: noqa: N806
            ImageTemplateProcessorClass = PROCESSOR_MANAGER[pre_processor["name"]]
            pre_processor_instance = ImageTemplateProcessorClass(
                options=pre_processor["options"],
                relative_dir=relative_dir,
                save_image_ops=self.template.save_image_ops,
                default_processing_image_shape=self.processing_image_shape,
            )
            self.pre_processors.append(pre_processor_instance)

    def parse_custom_bubble_field_types(self, custom_bubble_field_types) -> None:
        if not custom_bubble_field_types:
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
                field_block_object["field_detection_type"]
                == FieldDetectionType.BUBBLES_THRESHOLD
            ):
                bubble_field_type = field_block_object["bubble_field_type"]
                if bubble_field_type not in self.bubble_field_types_data:
                    logger.critical(
                        f"Cannot find definition for {bubble_field_type} in customBubbleFieldTypes"
                    )
                    msg = f"Invalid bubble field type: {bubble_field_type} in block {block_name}. Have you defined customBubbleFieldTypes?"
                    raise OMRCheckerError(
                        msg,
                        context={
                            "bubble_field_type": bubble_field_type,
                            "block_name": block_name,
                        },
                    )
            field_labels = field_block_object["field_labels"]
            if len(field_labels) > 1 and "labels_gap" not in field_block_object:
                logger.critical(
                    f"More than one fieldLabels({field_labels}) provided, but labels_gap not present for block {block_name}"
                )
                msg = f"More than one fieldLabels provided, but labels_gap not present for block {block_name}"
                raise OMRCheckerError(
                    msg,
                    context={
                        "field_labels": field_labels,
                        "block_name": block_name,
                    },
                )

    # TODO: move out to template_alignment.py
    def setup_alignment(self, alignment_object, relative_dir) -> None:
        from dataclasses import asdict
        from src.schemas.models.template import AlignmentConfig

        tuning_config = self.tuning_config
        # Convert AlignmentConfig dataclass to dict for mutability
        if isinstance(alignment_object, AlignmentConfig):
            self.alignment = asdict(alignment_object)
        else:
            self.alignment = alignment_object

        self.alignment["margins"] = self.alignment["margins"]
        self.alignment["reference_image_path"] = None
        relative_path = self.alignment.get("reference_image", None)

        # TODO: add more setup steps here

        if relative_path is not None:
            self.alignment["reference_image_path"] = Path(relative_dir, relative_path)
            # logger.debug(self.alignment)
            gray_alignment_image, colored_alignment_image = ImageUtils.read_image_util(
                self.alignment["reference_image_path"], tuning_config
            )
            # InteractionUtils.show("gray_alignment_image", gray_alignment_image)

            # Use PreprocessingCoordinator to preprocess the reference image
            from src.processors.base import ProcessingContext
            from src.processors.image.coordinator import PreprocessingCoordinator

            coordinator = PreprocessingCoordinator(self.template)

            context = ProcessingContext(
                file_path=self.alignment["reference_image_path"],
                gray_image=gray_alignment_image,
                colored_image=colored_alignment_image,
                template=self.template,
            )

            context = coordinator.process(context)

            processed_gray_alignment_image = context.gray_image
            processed_colored_alignment_image = context.colored_image

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
                raise OMRCheckerError(
                    msg,
                    context={
                        "custom_label": custom_label,
                        "missing_labels": list(missing_custom_labels),
                    },
                )

            if not all_parsed_custom_labels.isdisjoint(parsed_labels_set):
                # Note: this can be made a warning, but it's a choice
                logger.critical(
                    f"field strings overlap for labels: {label_strings} and existing custom labels: {all_parsed_custom_labels}"
                )
                msg = f"The field strings for custom label '{custom_label}' overlap with other existing custom labels"
                raise OMRCheckerError(
                    msg,
                    context={
                        "custom_label": custom_label,
                        "label_strings": label_strings,
                    },
                )

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

    def fill_output_columns(
        self, non_custom_columns, all_custom_columns, output_columns
    ):
        all_template_columns = non_custom_columns + all_custom_columns
        sort_type = output_columns.sort_type

        sort_order = output_columns.sort_order
        reverse = sort_order == "DESC"

        if sort_type == "ALPHANUMERIC":
            self.output_columns = sorted(
                all_template_columns, key=alphanumerical_sort_key, reverse=reverse
            )
        else:
            self.output_columns = sorted(all_template_columns, reverse=reverse)

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
            raise OMRCheckerError(
                msg,
                context={"missing_output_columns": list(missing_output_columns)},
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
        field_block_object = self.prefill_field_block(field_block_object)
        block_instance = FieldBlock(
            block_name, field_block_object, self.field_blocks_offset
        )
        # TODO: support custom field types like Barcode and OCR
        self.field_blocks.append(block_instance)
        self.validate_parsed_field_block(
            field_block_object["field_labels"], block_instance
        )
        return block_instance

    def prefill_field_block(self, field_block_object):
        filled_field_block_object = {
            **field_block_object,
        }

        if (
            field_block_object["field_detection_type"]
            == FieldDetectionType.BUBBLES_THRESHOLD
        ):
            bubble_field_type = field_block_object["bubble_field_type"]
            field_type_data = self.bubble_field_types_data[bubble_field_type]
            filled_field_block_object = {
                "bubble_field_type": bubble_field_type,
                # "direction": "vertical",
                "empty_value": self.global_empty_val,
                "bubble_dimensions": self.bubble_dimensions,
                **filled_field_block_object,
                **field_type_data,
            }
        elif (
            field_block_object["field_detection_type"] == FieldDetectionType.OCR
            or field_block_object["field_detection_type"]
            == FieldDetectionType.BARCODE_QR
        ):
            filled_field_block_object = {
                "empty_value": self.global_empty_val,
                "labels_gap": 0,
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
            raise OMRCheckerError(
                msg,
                context={
                    "block_name": block_name,
                    "field_labels": field_labels,
                    "overlap": list(overlap),
                },
            )
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
            raise OMRCheckerError(
                msg,
                context={
                    "block_name": block_name,
                    "bounding_box_origin": block_instance.bounding_box_origin,
                    "bounding_box_dimensions": block_instance.bounding_box_dimensions,
                    "template_dimensions": self.template_dimensions,
                },
            )

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
