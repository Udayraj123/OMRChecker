import ast
import os
import re

import pandas as pd
from rich.table import Table

from src.algorithm.evaluation.answer_matcher import AnswerMatcher
from src.algorithm.evaluation.section_marking_scheme import SectionMarkingScheme
from src.schemas.constants import (
    DEFAULT_SECTION_KEY,
    SCHEMA_VERDICTS_IN_ORDER,
    AnswerType,
    MarkingSchemeType,
    Verdict,
)
from src.utils.constants import CLR_BLACK, CLR_WHITE
from src.utils.image import ImageUtils
from src.utils.logger import console, logger
from src.utils.math import MathUtils
from src.utils.parsing import parse_fields


class EvaluationConfigForSet:
    """Note: this instance will be reused for multiple omr sheets"""

    def __init__(
        self,
        set_name,
        curr_dir,
        merged_evaluation_json,
        template,
        tuning_config,
        parent_evaluation_config=None,
    ):
        self.set_name = set_name
        (
            options,
            outputs_configuration,
            marking_schemes,
            source_type,
        ) = map(
            merged_evaluation_json.get,
            [
                "options",
                "outputs_configuration",
                "marking_schemes",
                "source_type",
            ],
        )

        (
            self.draw_answers_summary,
            self.draw_detected_bubble_texts,
            self.draw_question_verdicts,
            self.draw_score,
            self.should_explain_scoring,
            self.should_export_explanation_csv,
        ) = map(
            outputs_configuration.get,
            [
                "draw_answers_summary",
                "draw_detected_bubble_texts",
                "draw_question_verdicts",
                "draw_score",
                "should_explain_scoring",
                "should_export_explanation_csv",
            ],
        )
        if self.draw_question_verdicts["enabled"]:
            self.parse_draw_question_verdicts()

        self.exclude_files = []
        self.has_custom_marking = False
        self.has_streak_marking = False
        self.allow_streak = False
        self.has_conditional_sets = parent_evaluation_config is not None

        # TODO: separate handlers for these two type
        if source_type == "local":
            # Assign self answers_in_order so that child configs can refer for merging
            (
                local_questions_in_order,
                local_answers_in_order,
            ) = self.parse_local_question_answers(options)
        elif source_type == "image_and_csv" or source_type == "csv":
            (
                local_questions_in_order,
                local_answers_in_order,
            ) = self.parse_csv_question_answers(
                curr_dir, options, tuning_config, template
            )

        # Merge set's questions with parent questions(if any)
        (
            self.questions_in_order,
            self.answers_in_order,
        ) = self.merge_parsed_questions_and_schemes_with_parent(
            parent_evaluation_config, local_questions_in_order, local_answers_in_order
        )

        self.validate_questions()

        self.set_parsed_marking_schemes(
            marking_schemes, parent_evaluation_config, template
        )

        self.validate_marking_schemes()

        self.question_to_answer_matcher = self.parse_answers_and_map_questions()
        self.validate_answers(tuning_config)
        self.reset_evaluation()
        self.validate_format_strings()

    def get_exclude_files(self):
        return self.exclude_files

    @staticmethod
    def parse_local_question_answers(options):
        questions_in_order = EvaluationConfigForSet.parse_questions_in_order(
            options["questions_in_order"]
        )
        answers_in_order = options["answers_in_order"]
        return questions_in_order, answers_in_order

    def parse_csv_question_answers(self, curr_dir, options, tuning_config, template):
        questions_in_order = None
        csv_path = curr_dir.joinpath(options["answer_key_csv_path"])
        if not os.path.exists(csv_path):
            logger.warning(f"Answer key csv does not exist at: '{csv_path}'.")

        answer_key_image_path = options.get("answer_key_image_path", None)
        if os.path.exists(csv_path):
            # TODO: CSV parsing/validation for each row with a (qNo, <ans string/>) pair
            answer_key = pd.read_csv(
                csv_path,
                header=None,
                names=["question", "answer"],
                converters={
                    "question": lambda question: question.strip(),
                    "answer": self.parse_answer_column,
                },
            )

            questions_in_order = answer_key["question"].to_list()
            answers_in_order = answer_key["answer"].to_list()
        elif not answer_key_image_path:
            raise Exception(
                f"Answer key csv not found at '{csv_path}' and answer key image not provided to generate the csv"
            )
        else:
            # Attempt answer key image to generate the csv
            image_path = str(curr_dir.joinpath(answer_key_image_path))
            if not os.path.exists(image_path):
                raise Exception(f"Answer key image not found at '{image_path}'")

            self.exclude_files.append(image_path)

            logger.info(f"Attempting to generate answer key from image: '{image_path}'")
            # TODO: use a common function for below changes?
            gray_image, colored_image = ImageUtils.read_image_util(
                image_path, tuning_config
            )
            (
                gray_image,
                colored_image,
                template,
            ) = template.apply_preprocessors(image_path, gray_image, colored_image)
            if gray_image is None:
                raise Exception(f"Could not read answer key from image {image_path}")

            _, concatenated_omr_response = template.read_omr_response(
                gray_image, colored_image, image_path
            )

            empty_value = template.global_empty_val
            empty_answer_regex = (
                rf"{re.escape(empty_value)}+" if empty_value != "" else r"^$"
            )

            if "questions_in_order" in options:
                questions_in_order = self.parse_questions_in_order(
                    options["questions_in_order"]
                )
                empty_answered_questions = [
                    question
                    for question in questions_in_order
                    if re.search(
                        empty_answer_regex, concatenated_omr_response[question]
                    )
                ]
                if len(empty_answered_questions) > 0:
                    logger.error(
                        f"Found empty answers for the questions: {empty_answered_questions}, empty value used: '{empty_value}'"
                    )
                    raise Exception(
                        f"Found empty answers in file '{image_path}'. Please check your template again in the --setLayout mode."
                    )
            else:
                logger.warning(
                    f"questions_in_order not provided, proceeding to use non-empty values as answer key"
                )
                questions_in_order = sorted(
                    question
                    for (question, answer) in concatenated_omr_response.items()
                    if not re.search(empty_answer_regex, answer)
                )
            answers_in_order = [
                concatenated_omr_response[question] for question in questions_in_order
            ]
            # TODO: save the CSV
        return questions_in_order, answers_in_order

    @staticmethod
    def parse_answer_column(answer_column):
        # Remove all whitespaces
        answer_column = answer_column.replace(" ", "")
        if answer_column[0] == "[":
            # multiple-correct-weighted or multiple-correct
            parsed_answer = ast.literal_eval(answer_column)
        elif "," in answer_column:
            # multiple-correct
            parsed_answer = answer_column.split(",")
        else:
            # single-correct
            parsed_answer = answer_column
        return parsed_answer

    def parse_draw_question_verdicts(self):
        (
            verdict_colors,
            verdict_symbol_colors,
            draw_answer_groups,
        ) = map(
            self.draw_question_verdicts.get,
            [
                "verdict_colors",
                "verdict_symbol_colors",
                "draw_answer_groups",
            ],
        )

        self.verdict_colors = {
            "correct": MathUtils.to_bgr(verdict_colors["correct"]),
            # Note: neutral value will default to incorrect value
            "neutral": MathUtils.to_bgr(
                verdict_colors["neutral"]
                if verdict_colors["neutral"] is not None
                else verdict_colors["incorrect"]
            ),
            "incorrect": MathUtils.to_bgr(verdict_colors["incorrect"]),
            "bonus": MathUtils.to_bgr(verdict_colors["bonus"]),
        }
        self.verdict_symbol_colors = {
            "positive": MathUtils.to_bgr(verdict_symbol_colors["positive"]),
            "neutral": MathUtils.to_bgr(verdict_symbol_colors["neutral"]),
            "negative": MathUtils.to_bgr(verdict_symbol_colors["negative"]),
            "bonus": MathUtils.to_bgr(verdict_symbol_colors["bonus"]),
        }

        # Shallow copy
        self.draw_answer_groups = {**draw_answer_groups}
        self.draw_answer_groups["color_sequence"] = list(
            map(MathUtils.to_bgr, draw_answer_groups["color_sequence"])
        )

    @staticmethod
    def parse_questions_in_order(questions_in_order):
        return parse_fields("questions_in_order", questions_in_order)

    @staticmethod
    def merge_parsed_questions_and_schemes_with_parent(
        parent_evaluation_config, local_questions_in_order, local_answers_in_order
    ):
        if parent_evaluation_config is None:
            return local_questions_in_order, local_answers_in_order
        parent_questions_in_order, parent_answers_in_order = (
            parent_evaluation_config.questions_in_order,
            parent_evaluation_config.answers_in_order,
        )
        local_question_to_answer_item = {
            question: answer_item
            for question, answer_item in zip(
                local_questions_in_order, local_answers_in_order
            )
        }

        merged_questions_in_order, merged_answers_in_order = [], []
        # Append existing questions from parent
        for parent_question, parent_answer_item in zip(
            parent_questions_in_order, parent_answers_in_order
        ):
            merged_questions_in_order.append(parent_question)
            # override from child set if present
            if parent_question in local_question_to_answer_item:
                merged_answers_in_order.append(
                    local_question_to_answer_item[parent_question]
                )
            else:
                merged_answers_in_order.append(parent_answer_item)

        parent_questions_set = set(parent_questions_in_order)

        # Append new questions from child set at the end
        for question, answer_item in zip(
            local_questions_in_order, local_answers_in_order
        ):
            if question not in parent_questions_set:
                merged_questions_in_order.append(question)
                merged_answers_in_order.append(answer_item)

        return merged_questions_in_order, merged_answers_in_order

    def validate_questions(self):
        questions_in_order, answers_in_order = (
            self.questions_in_order,
            self.answers_in_order,
        )

        # for question, answer in zip(questions_in_order, answers_in_order):
        #     if question in template.custom_labels:
        #         # TODO: get all bubble values for the custom label
        #         if len(answer) != len(firstBubbleValues):
        #           logger.warning(f"The question {question} is a custom label and its answer does not have same length as the custom label")

        len_questions_in_order, len_answers_in_order = len(questions_in_order), len(
            answers_in_order
        )
        if len_questions_in_order != len_answers_in_order:
            logger.critical(
                f"questions_in_order({len_questions_in_order}): {questions_in_order}\nanswers_in_order({len_answers_in_order}): {answers_in_order}"
            )
            raise Exception(
                f"Unequal lengths for questions_in_order and answers_in_order ({len_questions_in_order} != {len_answers_in_order})"
            )

    def set_parsed_marking_schemes(
        self, marking_schemes, parent_evaluation_config, template
    ):
        self.section_marking_schemes, self.question_to_scheme = {}, {}

        # Precedence to local marking schemes (Note: default scheme is compulsory)
        for section_key, section_scheme in marking_schemes.items():
            section_marking_scheme = SectionMarkingScheme(
                section_key, section_scheme, self.set_name, template.global_empty_val
            )
            if section_key == DEFAULT_SECTION_KEY:
                self.default_marking_scheme = section_marking_scheme
            else:
                self.section_marking_schemes[section_key] = section_marking_scheme
                for q in section_marking_scheme.questions:
                    self.question_to_scheme[q] = section_marking_scheme
                self.has_custom_marking = True

                if (
                    section_marking_scheme.marking_type
                    == MarkingSchemeType.VERDICT_LEVEL_STREAK
                ):
                    self.has_streak_marking = True

        if parent_evaluation_config is not None:
            self.update_marking_schemes_from_parent(parent_evaluation_config)

    def update_marking_schemes_from_parent(self, parent_evaluation_config):
        # Parse parents schemes to inject the question_to_scheme map
        parent_marking_schemes = parent_evaluation_config.section_marking_schemes
        # Loop over parent's custom marking schemes to map remaining questions if any
        for (
            parent_section_key,
            parent_section_marking_scheme,
        ) in parent_marking_schemes.items():
            # Skip the default marking scheme (as it won't have any questions)
            if parent_section_key == DEFAULT_SECTION_KEY:
                continue

            section_key = f"parent-{parent_section_key}"
            questions_subset = [
                question
                for question in parent_section_marking_scheme.questions
                if question not in self.question_to_scheme
            ]

            if len(questions_subset) == 0:
                continue

            # Override the custom marking flag
            self.has_custom_marking = True

            # Make a deepclone with updated questions array
            subset_marking_scheme = (
                # TODO: does deepcopy affect streak of SectionScheme
                parent_section_marking_scheme.deepcopy_with_questions(questions_subset)
            )
            self.section_marking_schemes[section_key] = subset_marking_scheme

            if (
                subset_marking_scheme.marking_type
                == MarkingSchemeType.VERDICT_LEVEL_STREAK
            ):
                self.has_streak_marking = True

            for q in parent_section_marking_scheme.questions:
                self.question_to_scheme[q] = subset_marking_scheme

    def validate_marking_schemes(self):
        section_marking_schemes = self.section_marking_schemes
        section_questions = set()
        for section_key, section_scheme in section_marking_schemes.items():
            if section_key == DEFAULT_SECTION_KEY:
                continue
            current_set = set(section_scheme.questions)
            if not section_questions.isdisjoint(current_set):
                raise Exception(
                    f"Section '{section_key}' has overlapping question(s) with other sections locally"
                )
            section_questions = section_questions.union(current_set)

        all_questions = set(self.questions_in_order)
        missing_questions = sorted(section_questions.difference(all_questions))
        if len(missing_questions) > 0:
            logger.critical(f"Missing answer key for: {missing_questions}")
            raise Exception(
                f"Some questions are missing in the answer key for the given marking scheme(s)"
            )

    def parse_answers_and_map_questions(self):
        answers_in_order = self.answers_in_order
        question_to_answer_matcher = {}
        for question, answer_item in zip(self.questions_in_order, answers_in_order):
            section_marking_scheme = self.get_marking_scheme_for_question(question)
            answer_matcher = AnswerMatcher(answer_item, section_marking_scheme)
            question_to_answer_matcher[question] = answer_matcher
            if answer_matcher.answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
                self.has_custom_marking = True
                if section_marking_scheme.section_key != DEFAULT_SECTION_KEY:
                    logger.warning(
                        f"The custom scheme '{section_marking_scheme}' contains the question '{question}' but the given answer weights will be used instead: f{answer_item}"
                    )
        return question_to_answer_matcher

    def get_marking_scheme_for_question(self, question):
        return self.question_to_scheme.get(question, self.default_marking_scheme)

    def validate_answers(self, tuning_config):
        answer_matcher_map, answers_in_order = (
            self.question_to_answer_matcher,
            self.answers_in_order,
        )
        if tuning_config.outputs.filter_out_multimarked_files:
            contains_multi_marked_answer = False
            for question, answer_item in zip(self.questions_in_order, answers_in_order):
                answer_type = answer_matcher_map[question].answer_type
                if answer_type == AnswerType.STANDARD:
                    if len(answer_item) > 1:
                        contains_multi_marked_answer = True
                if answer_type == AnswerType.MULTIPLE_CORRECT:
                    for single_answer in answer_item:
                        if len(single_answer) > 1:
                            contains_multi_marked_answer = True
                            break
                if answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
                    for single_answer, _answer_score in answer_item:
                        if len(single_answer) > 1:
                            contains_multi_marked_answer = True

                if contains_multi_marked_answer:
                    raise Exception(
                        f"Provided answer key contains multiple correct answer(s), but config.filter_out_multimarked_files is True. Scoring will get skipped."
                    )

    def validate_format_strings(self):
        answers_summary_format_string = self.draw_answers_summary[
            "answers_summary_format_string"
        ]
        try:
            # TODO: Support for total_positive, total_negative,
            # TODO: Same aggregates section-wise: correct/incorrect verdict counts in formatted_answers_summary
            answers_summary_format_string.format(**self.schema_verdict_counts)
        except:  # NOQA
            raise Exception(
                f"The format string should contain only allowed variables {SCHEMA_VERDICTS_IN_ORDER}. answers_summary_format_string={answers_summary_format_string}"
            )

        score_format_string = self.draw_score["score_format_string"]
        try:
            score_format_string.format(score=0)
        except:  # NOQA
            raise Exception(
                f"The format string should contain only allowed variables ['score']. score_format_string={score_format_string}"
            )

    # Public function: Externally called methods with higher abstraction level.
    def prepare_and_validate_omr_response(
        self, concatenated_omr_response, allow_streak=False
    ):
        self.allow_streak = allow_streak
        self.reset_evaluation()

        omr_response_keys = set(concatenated_omr_response.keys())
        all_questions = set(self.questions_in_order)
        missing_questions = sorted(all_questions.difference(omr_response_keys))
        if len(missing_questions) > 0:
            logger.critical(
                f"Missing OMR response for: {missing_questions} in omr response keys: {omr_response_keys}"
            )
            raise Exception(
                f"Some question keys are missing in the OMR response for the given answer key"
            )

        prefixed_omr_response_questions = set(
            [k for k in concatenated_omr_response.keys() if k.startswith("q")]
        )
        missing_prefixed_questions = sorted(
            prefixed_omr_response_questions.difference(all_questions)
        )
        if len(missing_prefixed_questions) > 0:
            logger.warning(
                f"No answer given for potential questions in OMR response: {missing_prefixed_questions} ('q' prefix). You may rename the field in evaluation.json to avoid this warning."
            )

    # Public function
    def match_answer_for_question(
        self,
        current_score,
        question,
        marked_answer,
    ):
        answer_matcher = self.question_to_answer_matcher[question]
        (
            question_verdict,
            delta,
            current_streak,
            updated_streak,
        ) = answer_matcher.get_verdict_marking(marked_answer, self.allow_streak)
        question_schema_verdict = AnswerMatcher.get_schema_verdict(
            answer_matcher.answer_type, question_verdict, delta
        )
        self.schema_verdict_counts[question_schema_verdict] += 1

        self.conditionally_add_explanation(
            answer_matcher,
            delta,
            marked_answer,
            question_schema_verdict,
            question_verdict,
            question,
            current_score,
            current_streak,
            updated_streak,
        )
        return delta, question_verdict, answer_matcher, question_schema_verdict

    def conditionally_add_explanation(
        self,
        answer_matcher,
        delta,
        marked_answer,
        question_schema_verdict,
        question_verdict,
        question,
        current_score,
        current_streak,
        updated_streak,
    ):
        if self.should_explain_scoring:
            next_score = current_score + delta
            # Conditionally add cells
            row = [
                item
                for item in [
                    (
                        answer_matcher.get_section_explanation()
                        if self.has_custom_marking
                        else None
                    ),
                    question,
                    marked_answer,
                    str(answer_matcher),
                    f"{question_schema_verdict.title()} ({question_verdict})",
                    str(round(delta, 2)),
                    str(round(next_score, 2)),
                    (
                        answer_matcher.get_matched_set_name()
                        if self.has_conditional_sets
                        else None
                    ),
                    (
                        f"{current_streak} -> {updated_streak}"
                        if self.has_streak_marking and self.allow_streak
                        else None
                    ),
                ]
                if item is not None
            ]
            self.explanation_table.add_row(*row)

    def conditionally_print_explanation(self):
        if self.should_explain_scoring:
            console.print(self.explanation_table, justify="center")

    def get_should_explain_scoring(self):
        return self.should_explain_scoring

    def get_should_export_explanation_csv(self):
        return self.should_export_explanation_csv

    def get_explanation_table(self):
        return self.explanation_table

    def get_formatted_answers_summary(self, answers_summary_format_string=None):
        if answers_summary_format_string is None:
            answers_summary_format_string = self.draw_answers_summary[
                "answers_summary_format_string"
            ]
        answers_format = answers_summary_format_string.format(
            **self.schema_verdict_counts
        )
        position = self.draw_answers_summary["position"]
        size = self.draw_answers_summary["size"]
        thickness = int(self.draw_answers_summary["size"] * 2)
        return answers_format, position, size, thickness

    def get_formatted_score(self, score):
        score_format = self.draw_score["score_format_string"].format(
            score=round(score, 2)
        )
        position = self.draw_score["position"]
        size = self.draw_score["size"]
        thickness = int(self.draw_score["size"] * 2)
        return score_format, position, size, thickness

    def reset_evaluation(self):
        self.explanation_table = None

        for section_scheme in self.section_marking_schemes.values():
            section_scheme.reset_all_streaks()

        self.schema_verdict_counts = {
            schema_verdict: 0 for schema_verdict in SCHEMA_VERDICTS_IN_ORDER
        }
        self.prepare_explanation_table()

    def prepare_explanation_table(self):
        # TODO: provide a way to export this as csv/pdf
        if not self.should_explain_scoring:
            return
        table = Table(title="Evaluation Explanation Table", show_lines=True)
        if self.has_custom_marking:
            table.add_column("Marking Scheme")
        table.add_column("Question")
        table.add_column("Marked")
        table.add_column("Answer(s)")
        table.add_column("Verdict")
        table.add_column("Delta")
        table.add_column("Score")
        # TODO: Add max and min score in explanation (row-wise and total)
        if self.has_conditional_sets:
            table.add_column("Set Mapping")
        if self.has_streak_marking and self.allow_streak:
            table.add_column("Streak")
        self.explanation_table = table

    def get_evaluation_meta_for_question(
        self, question_meta, is_field_marked, image_type
    ):
        # TODO: take config for CROSS_TICKS vs BUBBLE_BOUNDARY and call appropriate util
        (
            symbol_positive,
            symbol_negative,
            symbol_neutral,
            symbol_bonus,
            symbol_unmarked,
        ) = ("+", "-", "o", "*", "")
        thickness_factor = 1 / 12

        bonus_type, question_verdict, question_schema_verdict = map(
            question_meta.get,
            ["bonus_type", "question_verdict", "question_schema_verdict"],
        )

        color_correct, color_incorrect, color_neutral, color_bonus = map(
            self.verdict_colors.get, ["correct", "incorrect", "neutral", "bonus"]
        )
        (
            symbol_color_positive,
            symbol_color_negative,
            symbol_color_neutral,
            symbol_color_bonus,
        ) = map(
            self.verdict_symbol_colors.get,
            ["positive", "negative", "neutral", "bonus"],
        )
        if is_field_marked:
            # Always render symbol as per delta (regardless of bonus) for marked bubbles
            if question_meta["delta"] > 0:
                symbol = symbol_positive
            elif question_meta["delta"] < 0:
                symbol = symbol_negative
            else:
                symbol = symbol_neutral

            # Update colors for marked bubbles
            if image_type == "GRAYSCALE":
                color = CLR_WHITE
                symbol_color = CLR_BLACK
            else:
                # Apply colors as per delta
                if question_meta["delta"] > 0:
                    color, symbol_color = (
                        color_correct,
                        symbol_color_positive,
                    )
                elif question_meta["delta"] < 0:
                    color, symbol_color = (
                        color_incorrect,
                        symbol_color_negative,
                    )
                else:
                    color, symbol_color = (
                        color_neutral,
                        symbol_color_neutral,
                    )

                # Override bonus colors if bubble was marked but verdict was not correct
                if bonus_type is not None and question_verdict in [
                    Verdict.UNMARKED,
                    Verdict.NO_ANSWER_MATCH,
                ]:
                    color, symbol_color = (
                        color_bonus,
                        symbol_color_bonus,
                    )
        else:
            symbol = symbol_unmarked
            # In case of unmarked bubbles, we draw the symbol only if bonus type is BONUS_FOR_ALL
            if bonus_type == "BONUS_FOR_ALL":
                symbol = symbol_positive
            elif bonus_type == "BONUS_ON_ATTEMPT":
                if question_schema_verdict == Verdict.UNMARKED:
                    # Case of bonus on attempt with blank question - show neutral symbol on all bubbles(on attempt)
                    symbol = symbol_neutral
                else:
                    # Case of bonus on attempt with one or more marked bubbles - show bonus symbol on remaining bubbles
                    symbol = symbol_bonus

            # Apply bonus colors for all bubbles
            if bonus_type is not None:
                color, symbol_color = (
                    color_bonus,
                    symbol_color_bonus,
                )

        return symbol, color, symbol_color, thickness_factor
