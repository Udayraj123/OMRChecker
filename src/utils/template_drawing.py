import cv2

from src.schemas.constants import AnswerType
from src.utils.constants import (
    CLR_BLACK,
    CLR_GRAY,
    CLR_NEAR_BLACK,
    CLR_WHITE,
    MARKED_TEMPLATE_TRANSPARENCY,
    TEXT_SIZE,
)
from src.utils.drawing import DrawingUtils
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


class TemplateDrawing:
    @staticmethod
    def draw_template_layout(
        gray_image, colored_image, template, config, *args, **kwargs
    ):
        final_marked = TemplateDrawing.draw_template_layout_util(
            gray_image, "GRAYSCALE", template, config, *args, **kwargs
        )

        colored_final_marked = colored_image
        if config.outputs.colored_outputs_enabled:
            save_marked_dir = kwargs.get("save_marked_dir", None)
            # TODO: get dedicated path from top args
            kwargs["save_marked_dir"] = (
                save_marked_dir.joinpath("colored")
                if save_marked_dir is not None
                else None
            )

            colored_final_marked = TemplateDrawing.draw_template_layout_util(
                colored_final_marked,
                "COLORED",
                template,
                config,
                *args,
                **kwargs,
            )
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show(
                    "Final Marked Bubbles",
                    final_marked,
                    0,
                    resize_to_height=True,
                    config=config,
                )
                InteractionUtils.show(
                    "Final Marked Bubbles (Colored)",
                    colored_final_marked,
                    1,
                    resize_to_height=True,
                    config=config,
                )
        elif config.outputs.show_image_level >= 1:
            InteractionUtils.show(
                "Final Marked Bubbles",
                final_marked,
                1,
                resize_to_height=True,
                config=config,
            )

        template.save_image_ops.append_save_image(
            "Marked Template", range(1, 7), final_marked, colored_final_marked
        )

        return final_marked, colored_final_marked

    @staticmethod
    def draw_template_layout_util(
        image,
        image_type,
        template,
        config,
        file_id=None,
        field_number_to_field_bubble_means=None,
        save_marked_dir=None,
        evaluation_meta=None,
        evaluation_config=None,
        shifted=False,
        border=-1,
    ):
        marked_image = ImageUtils.resize_to_dimensions(
            image, template.template_dimensions
        )

        transparent_layer = marked_image.copy()
        should_draw_field_block_rectangles = field_number_to_field_bubble_means is None
        should_draw_marked_bubbles = field_number_to_field_bubble_means is not None
        should_draw_question_verdicts = (
            should_draw_marked_bubbles and evaluation_meta is not None
        )

        should_save_detections = (
            config.outputs.save_detections and save_marked_dir is not None
        )

        if should_draw_field_block_rectangles:
            marked_image = TemplateDrawing.draw_field_blocks_layout(
                marked_image, template, shifted, shouldCopy=False, border=border
            )
            return marked_image

        if should_draw_marked_bubbles:
            if config.outputs.save_image_level >= 1:
                marked_image_copy = marked_image.copy()
                marked_image_copy = (
                    TemplateDrawing.draw_marked_bubbles_with_evaluation_meta(
                        marked_image_copy,
                        image_type,
                        template,
                        field_number_to_field_bubble_means,
                        evaluation_meta=None,
                        evaluation_config=None,
                    )
                )
                if image_type == "GRAYSCALE":
                    template.save_image_ops.append_save_image(
                        f"Marked Image", range(2, 7), marked_image_copy
                    )
                else:
                    template.save_image_ops.append_save_image(
                        f"Marked Image", range(2, 7), colored_image=marked_image_copy
                    )

            marked_image = TemplateDrawing.draw_marked_bubbles_with_evaluation_meta(
                marked_image,
                image_type,
                template,
                field_number_to_field_bubble_means,
                evaluation_meta,
                evaluation_config,
            )

        if should_save_detections:
            # TODO: migrate after support for multi_marked bucket based on identifier config
            # if multi_roll:
            #     save_marked_dir = save_marked_dir.joinpath("_MULTI_")
            image_path = str(save_marked_dir.joinpath(file_id))
            logger.info(f"Saving Image to '{image_path}'")
            ImageUtils.save_img(image_path, marked_image)

        if should_draw_question_verdicts:
            marked_image = TemplateDrawing.draw_evaluation_summary(
                marked_image, evaluation_meta, evaluation_config
            )

        # Translucent
        cv2.addWeighted(
            marked_image,
            MARKED_TEMPLATE_TRANSPARENCY,
            transparent_layer,
            1 - MARKED_TEMPLATE_TRANSPARENCY,
            0,
            marked_image,
        )

        return marked_image

    @staticmethod
    def draw_field_blocks_layout(
        image, template, shifted=True, shouldCopy=True, thickness=3, border=3
    ):
        marked_image = image.copy() if shouldCopy else image
        for field_block in template.field_blocks:
            field_block_name, origin, dimensions, bubble_dimensions = map(
                lambda attr: getattr(field_block, attr),
                [
                    "name",
                    "origin",
                    "dimensions",
                    "bubble_dimensions",
                ],
            )
            block_position = field_block.get_shifted_origin() if shifted else origin

            # Field block bounding rectangle
            DrawingUtils.draw_box(
                marked_image,
                block_position,
                dimensions,
                color=CLR_BLACK,
                style="BOX_HOLLOW",
                thickness_factor=0,
                border=border,
            )

            for field in field_block.fields:
                field_bubbles = field.field_bubbles
                for unit_bubble in field_bubbles:
                    shifted_position = unit_bubble.get_shifted_position(
                        field_block.shifts
                    )
                    DrawingUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        thickness_factor=1 / 10,
                        border=border,
                    )

            if shifted:
                text_position = lambda size_x, size_y: (
                    int(block_position[0] + dimensions[0] - size_x),
                    int(block_position[1] - size_y),
                )
                text = f"({field_block.shifts}){field_block_name}"
                DrawingUtils.draw_text(marked_image, text, text_position, thickness)

        return marked_image

    @staticmethod
    def draw_marked_bubbles_with_evaluation_meta(
        marked_image,
        image_type,
        template,
        field_number_to_field_bubble_means,
        evaluation_meta,
        evaluation_config,
    ):
        should_draw_question_verdicts = (
            evaluation_meta is not None and evaluation_config is not None
        )
        absolute_field_number = 0
        for field_block in template.field_blocks:
            for field in field_block.fields:
                field_label = field.field_label
                field_bubble_means = field_number_to_field_bubble_means[
                    absolute_field_number
                ]
                absolute_field_number += 1

                question_has_verdict = (
                    evaluation_meta is not None
                    and field_label in evaluation_meta["questions_meta"]
                )

                # linked_custom_labels = [custom_label if (field_label in field_labels) else None for (custom_label, field_labels) in template.custom_labels.items()]
                # is_part_of_custom_label = len(linked_custom_labels) > 0
                # TODO: replicate verdict: question_has_verdict = len([if field_label in questions_meta else None for field_label in linked_custom_labels])

                if (
                    should_draw_question_verdicts
                    and question_has_verdict
                    and evaluation_config.draw_question_verdicts["enabled"]
                ):
                    question_meta = evaluation_meta["questions_meta"][field_label]
                    # Draw answer key items
                    TemplateDrawing.draw_field_with_question_meta(
                        marked_image,
                        image_type,
                        field_bubble_means,
                        field_block,
                        question_meta,
                        evaluation_config,
                    )
                else:
                    TemplateDrawing.draw_field_bubbles_and_detections(
                        marked_image, field_bubble_means, field_block, evaluation_config
                    )

        return marked_image

    @staticmethod
    def draw_field_with_question_meta(
        marked_image,
        image_type,
        field_bubble_means,
        field_block,
        question_meta,
        evaluation_config,
    ):
        bubble_dimensions = tuple(field_block.bubble_dimensions)
        bonus_type = question_meta["bonus_type"]
        for bubble_detection in field_bubble_means:
            bubble = bubble_detection.item_reference
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            field_value = str(bubble.field_value)

            # Enhanced bounding box for expected answer:
            if TemplateDrawing.is_part_of_some_answer(question_meta, field_value):
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    CLR_BLACK,
                    style="BOX_HOLLOW",
                    thickness_factor=0,
                )

            # Filled box in case of marked bubble or bonus case
            if bubble_detection.is_marked or bonus_type is not None:
                (
                    verdict_symbol,
                    verdict_color,
                    verdict_symbol_color,
                    thickness_factor,
                ) = evaluation_config.get_evaluation_meta_for_question(
                    question_meta, bubble_detection, image_type
                )

                # Bounding box for marked bubble or bonus bubble
                if verdict_color != "":
                    position, position_diagonal = DrawingUtils.draw_box(
                        marked_image,
                        shifted_position,
                        bubble_dimensions,
                        color=verdict_color,
                        style="BOX_FILLED",
                        thickness_factor=thickness_factor,
                    )

                # Symbol for the marked bubble or bonus bubble
                if verdict_symbol != "":
                    DrawingUtils.draw_symbol(
                        marked_image,
                        verdict_symbol,
                        position,
                        position_diagonal,
                        color=verdict_symbol_color,
                    )

                # Symbol of the field value for marked bubble
                if (
                    bubble_detection.is_marked
                    and evaluation_config.draw_detected_bubble_texts["enabled"]
                ):
                    DrawingUtils.draw_text(
                        marked_image,
                        field_value,
                        shifted_position,
                        text_size=TEXT_SIZE,
                        color=CLR_NEAR_BLACK,
                        thickness=int(1 + 3.5 * TEXT_SIZE),
                    )
            else:
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    style="BOX_HOLLOW",
                    thickness_factor=1 / 10,
                )

        if evaluation_config.draw_answer_groups["enabled"]:
            TemplateDrawing.draw_answer_groups(
                marked_image,
                image_type,
                question_meta,
                field_bubble_means,
                field_block,
                evaluation_config,
            )

    @staticmethod
    def draw_field_bubbles_and_detections(
        marked_image, field_bubble_means, field_block, evaluation_config
    ):
        bubble_dimensions = tuple(field_block.bubble_dimensions)
        for bubble_detection in field_bubble_means:
            bubble = bubble_detection.item_reference
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            field_value = str(bubble.field_value)

            if bubble_detection.is_marked:
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    color=CLR_GRAY,
                    style="BOX_FILLED",
                    thickness_factor=1 / 12,
                )
                if (
                    # Note: this mimics the default true behavior for draw_detected_bubble_texts
                    evaluation_config is None
                    or evaluation_config.draw_detected_bubble_texts["enabled"]
                ):
                    DrawingUtils.draw_text(
                        marked_image,
                        field_value,
                        shifted_position,
                        text_size=TEXT_SIZE,
                        color=CLR_NEAR_BLACK,
                        thickness=int(1 + 3.5 * TEXT_SIZE),
                    )
            else:
                DrawingUtils.draw_box(
                    marked_image,
                    shifted_position,
                    bubble_dimensions,
                    style="BOX_HOLLOW",
                    thickness_factor=1 / 10,
                )

    @staticmethod
    def draw_evaluation_summary(marked_image, evaluation_meta, evaluation_config):
        if evaluation_config.draw_answers_summary["enabled"]:
            (
                formatted_answers_summary,
                position,
                size,
                thickness,
            ) = evaluation_config.get_formatted_answers_summary()
            DrawingUtils.draw_text(
                marked_image,
                formatted_answers_summary,
                position,
                text_size=size,
                thickness=thickness,
            )

        if evaluation_config.draw_score["enabled"]:
            (
                formatted_score,
                position,
                size,
                thickness,
            ) = evaluation_config.get_formatted_score(evaluation_meta["score"])
            DrawingUtils.draw_text(
                marked_image,
                formatted_score,
                position,
                text_size=size,
                thickness=thickness,
            )

        return marked_image

    @staticmethod
    def is_part_of_some_answer(question_meta, field_value):
        if question_meta["bonus_type"] is not None:
            return True
        matched_groups = TemplateDrawing.get_matched_answer_groups(
            question_meta, field_value
        )
        return len(matched_groups) > 0

    @staticmethod
    def get_matched_answer_groups(question_meta, field_value):
        matched_groups = []
        answer_type, answer_item = map(
            question_meta.get, ["answer_type", "answer_item"]
        )

        if answer_type == AnswerType.STANDARD:
            # Note: implicit check on concatenated answer
            if field_value in str(answer_item):
                matched_groups.append(0)
        if answer_type == AnswerType.MULTIPLE_CORRECT:
            for answer_index, allowed_answer in enumerate(answer_item):
                if field_value in allowed_answer:
                    matched_groups.append(answer_index)
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            for answer_index, (allowed_answer, score) in enumerate(answer_item):
                if field_value in allowed_answer and score > 0:
                    matched_groups.append(answer_index)
        return matched_groups

    @staticmethod
    def draw_answer_groups(
        marked_image,
        image_type,
        question_meta,
        field_bubble_means,
        field_block,
        evaluation_config,
    ):
        # Note: currently draw_answer_groups is limited for questions with upto 4 bubbles
        answer_type = question_meta["answer_type"]
        if answer_type == AnswerType.STANDARD:
            return
        box_edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
        color_sequence = evaluation_config.draw_answer_groups["color_sequence"]
        bubble_dimensions = field_block.bubble_dimensions
        if image_type == "GRAYSCALE":
            color_sequence = [CLR_WHITE] * len(color_sequence)
        for bubble_detection in field_bubble_means:
            bubble = bubble_detection.item_reference
            shifted_position = tuple(bubble.get_shifted_position(field_block.shifts))
            field_value = str(bubble.field_value)
            matched_groups = TemplateDrawing.get_matched_answer_groups(
                question_meta, field_value
            )
            for answer_index in matched_groups:
                box_edge = box_edges[answer_index % 4]
                color = color_sequence[answer_index % 4]
                DrawingUtils.draw_group(
                    marked_image, shifted_position, bubble_dimensions, box_edge, color
                )
