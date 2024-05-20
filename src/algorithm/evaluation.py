"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import ast
import os
import re
from copy import deepcopy

import pandas as pd
from rich.table import Table

from src.schemas.constants import (
    BONUS_SECTION_PREFIX,
    DEFAULT_SECTION_KEY,
    SCHEMA_VERDICTS_IN_ORDER,
    VERDICT_TO_SCHEMA_VERDICT,
    VERDICTS_IN_ORDER,
    AnswerType,
    SchemaVerdict,
    Verdict,
)
from src.utils.constants import CLR_BLACK, CLR_WHITE
from src.utils.image import ImageUtils
from src.utils.logger import console, logger
from src.utils.math import MathUtils
from src.utils.parsing import (
    get_concatenated_response,
    open_evaluation_with_defaults,
    parse_fields,
    parse_float_or_fraction,
)


class AnswerMatcher:
    def __init__(self, answer_item, section_marking_scheme):
        self.section_marking_scheme = section_marking_scheme
        self.answer_type = self.get_answer_type(answer_item)
        self.parse_and_set_answer_item(answer_item)
        self.set_local_marking_defaults(section_marking_scheme)

    @staticmethod
    def is_a_marking_score(answer_element):
        # Note: strict type checking is already done at schema validation level,
        # Here we focus on overall struct type
        return type(answer_element) == str or type(answer_element) == int

    @staticmethod
    def is_standard_answer(answer_element):
        return type(answer_element) == str and len(answer_element) >= 1

    def get_answer_type(self, answer_item):
        if self.is_standard_answer(answer_item):
            return AnswerType.STANDARD
        elif type(answer_item) == list:
            if (
                # Array of answer elements: ['A', 'B', 'AB']
                len(answer_item) >= 2
                and all(
                    self.is_standard_answer(answers_or_score)
                    for answers_or_score in answer_item
                )
            ):
                return AnswerType.MULTIPLE_CORRECT
            elif (
                # Array of two-tuples: [['A', 1], ['B', 1], ['C', 3], ['AB', 2]]
                len(answer_item) >= 1
                and all(
                    type(answer_and_score) == list and len(answer_and_score) == 2
                    for answer_and_score in answer_item
                )
                and all(
                    self.is_standard_answer(allowed_answer)
                    and self.is_a_marking_score(answer_score)
                    for allowed_answer, answer_score in answer_item
                )
            ):
                return AnswerType.MULTIPLE_CORRECT_WEIGHTED

        logger.critical(
            f"Unable to determine answer type for answer item: {answer_item}"
        )
        raise Exception("Unable to determine answer type")

    def parse_and_set_answer_item(self, answer_item):
        answer_type = self.answer_type
        if answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            # parse answer scores for weighted answers
            self.answer_item = [
                [allowed_answer, parse_float_or_fraction(answer_score)]
                for allowed_answer, answer_score in answer_item
            ]
        else:
            self.answer_item = answer_item

    def set_local_marking_defaults(self, section_marking_scheme):
        answer_type = self.answer_type
        self.empty_val = section_marking_scheme.empty_val
        # Make a copy of section marking locally
        self.marking = deepcopy(section_marking_scheme.marking)
        if answer_type == AnswerType.STANDARD:
            # no local overrides needed
            pass
        elif answer_type == AnswerType.MULTIPLE_CORRECT:
            allowed_answers = self.answer_item
            # override marking scheme scores for each allowed answer
            for allowed_answer in allowed_answers:
                self.marking[f"{Verdict.ANSWER_MATCH}-{allowed_answer}"] = self.marking[
                    Verdict.ANSWER_MATCH
                ]
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            for allowed_answer, parsed_answer_score in self.answer_item:
                self.marking[
                    f"{Verdict.ANSWER_MATCH}-{allowed_answer}"
                ] = parsed_answer_score

    def get_marking_scheme(self):
        return self.section_marking_scheme

    def get_section_explanation(self):
        answer_type = self.answer_type
        if answer_type in [AnswerType.STANDARD, AnswerType.MULTIPLE_CORRECT]:
            return self.section_marking_scheme.section_key
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            return f"Custom Weights: {self.marking}"

    def get_verdict_marking(self, marked_answer):
        answer_type = self.answer_type
        question_verdict = Verdict.NO_ANSWER_MATCH
        if answer_type == AnswerType.STANDARD:
            question_verdict = self.get_standard_verdict(marked_answer)
        elif answer_type == AnswerType.MULTIPLE_CORRECT:
            question_verdict = self.get_multiple_correct_verdict(marked_answer)
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            question_verdict = self.get_multiple_correct_weighted_verdict(marked_answer)
        return question_verdict, self.marking[question_verdict]

    def get_standard_verdict(self, marked_answer):
        allowed_answer = self.answer_item
        if marked_answer == self.empty_val:
            return Verdict.UNMARKED
        elif marked_answer == allowed_answer:
            return Verdict.ANSWER_MATCH
        else:
            return Verdict.NO_ANSWER_MATCH

    def get_multiple_correct_verdict(self, marked_answer):
        allowed_answers = self.answer_item
        if marked_answer == self.empty_val:
            return Verdict.UNMARKED
        elif marked_answer in allowed_answers:
            # e.g. ANSWER-MATCH-BC
            return f"{Verdict.ANSWER_MATCH}-{marked_answer}"
        else:
            return Verdict.NO_ANSWER_MATCH

    def get_multiple_correct_weighted_verdict(self, marked_answer):
        allowed_answers = [
            allowed_answer for allowed_answer, _answer_score in self.answer_item
        ]
        if marked_answer == self.empty_val:
            return Verdict.UNMARKED
        elif marked_answer in allowed_answers:
            return f"{Verdict.ANSWER_MATCH}-{marked_answer}"
        else:
            return Verdict.NO_ANSWER_MATCH

    def __str__(self):
        return f"{self.answer_item}"


class SectionMarkingScheme:
    def __init__(self, section_key, section_scheme, empty_val):
        # TODO: get local empty_val from qblock
        self.empty_val = empty_val
        self.section_key = section_key
        # DEFAULT marking scheme follows a shorthand
        if section_key == DEFAULT_SECTION_KEY:
            self.questions = None
            self.marking = self.parse_verdict_marking_from_scheme(section_scheme)
        else:
            self.questions = parse_fields(section_key, section_scheme["questions"])
            self.marking = self.parse_verdict_marking_from_scheme(
                section_scheme["marking"]
            )

    def __str__(self):
        return self.section_key

    def parse_verdict_marking_from_scheme(self, section_scheme):
        parsed_marking = {}
        for verdict in VERDICTS_IN_ORDER:
            schema_verdict = VERDICT_TO_SCHEMA_VERDICT[verdict]
            schema_verdict_marking = parse_float_or_fraction(
                section_scheme[schema_verdict]
            )
            if (
                schema_verdict_marking > 0
                and schema_verdict == SchemaVerdict.INCORRECT
                and not self.section_key.startswith(BONUS_SECTION_PREFIX)
            ):
                logger.warning(
                    f"Found positive marks({round(schema_verdict_marking, 2)}) for incorrect answer in the schema '{self.section_key}'. For Bonus sections, prefer adding a prefix 'BONUS_' to the scheme name."
                )

            # translated marking e.g. parsed[MATCHED] -> section_scheme['correct']
            parsed_marking[verdict] = schema_verdict_marking

        return parsed_marking

    def get_bonus_type(self):
        if self.marking[Verdict.NO_ANSWER_MATCH] <= 0:
            return None
        elif self.marking[Verdict.UNMARKED] > 0:
            return "BONUS_FOR_ALL"
        elif self.marking[Verdict.UNMARKED] == 0:
            return "BONUS_ON_ATTEMPT"
        else:
            return None


class EvaluationConfig:
    """Note: this instance will be reused for multiple omr sheets"""

    def __init__(self, curr_dir, evaluation_path, template, tuning_config):
        self.path = evaluation_path
        evaluation_json = open_evaluation_with_defaults(evaluation_path)

        (
            options,
            outputs_configuration,
            marking_schemes,
            source_type,
        ) = map(
            evaluation_json.get,
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
        ) = map(
            outputs_configuration.get,
            [
                "draw_answers_summary",
                "draw_detected_bubble_texts",
                "draw_question_verdicts",
                "draw_score",
                "should_explain_scoring",
            ],
        )
        if self.draw_question_verdicts["enabled"]:
            self.parse_draw_question_verdicts()

        self.has_custom_marking = False
        self.exclude_files = []

        # TODO: separate handlers for these two types
        if source_type == "image_and_csv" or source_type == "csv":
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

                self.questions_in_order = answer_key["question"].to_list()
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

                logger.info(
                    f"Attempting to generate answer key from image: '{image_path}'"
                )
                # TODO: use a common function for below changes?
                gray_image, colored_image = ImageUtils.read_image_util(
                    image_path, tuning_config
                )
                (
                    gray_image,
                    colored_image,
                    template,
                ) = template.image_instance_ops.apply_preprocessors(
                    image_path, gray_image, colored_image, template
                )
                if gray_image is None:
                    raise Exception(
                        f"Could not read answer key from image {image_path}"
                    )

                (response_dict, *_) = template.image_instance_ops.read_omr_response(
                    gray_image, colored_image, template, image_path
                )
                omr_response = get_concatenated_response(response_dict, template)

                empty_val = template.global_empty_val
                empty_answer_regex = (
                    rf"{re.escape(empty_val)}+" if empty_val != "" else r"^$"
                )

                if "questions_in_order" in options:
                    self.questions_in_order = self.parse_questions_in_order(
                        options["questions_in_order"]
                    )
                    empty_answered_questions = [
                        question
                        for question in self.questions_in_order
                        if re.search(empty_answer_regex, omr_response[question])
                    ]
                    if len(empty_answered_questions) > 0:
                        logger.error(
                            f"Found empty answers for the questions: {empty_answered_questions}, empty value used: '{empty_val}'"
                        )
                        raise Exception(
                            f"Found empty answers in file '{image_path}'. Please check your template again in the --setLayout mode."
                        )
                else:
                    logger.warning(
                        f"questions_in_order not provided, proceeding to use non-empty values as answer key"
                    )
                    self.questions_in_order = sorted(
                        question
                        for (question, answer) in omr_response.items()
                        if not re.search(empty_answer_regex, answer)
                    )
                answers_in_order = [
                    omr_response[question] for question in self.questions_in_order
                ]
                # TODO: save the CSV
        else:
            self.questions_in_order = self.parse_questions_in_order(
                options["questions_in_order"]
            )
            answers_in_order = options["answers_in_order"]

        self.validate_questions(answers_in_order)

        self.set_marking_schemes(marking_schemes, template)

        self.validate_marking_schemes()

        self.question_to_answer_matcher = self.parse_answers_and_map_questions(
            answers_in_order
        )
        self.validate_answers(answers_in_order, tuning_config)
        self.reset_evaluation()
        self.validate_format_strings()

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

        self.draw_answer_groups = draw_answer_groups
        self.draw_answer_groups["color_sequence"] = list(
            map(MathUtils.to_bgr, draw_answer_groups["color_sequence"])
        )

    def parse_questions_in_order(self, questions_in_order):
        return parse_fields("questions_in_order", questions_in_order)

    def validate_questions(self, answers_in_order):
        questions_in_order = self.questions_in_order

        # for question, answer in zip(questions_in_order, answers_in_order):
        #     if question in template.custom_label:
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

    def set_marking_schemes(self, marking_schemes, template):
        self.section_marking_schemes, self.question_to_scheme = {}, {}
        for section_key, section_scheme in marking_schemes.items():
            section_marking_scheme = SectionMarkingScheme(
                section_key, section_scheme, template.global_empty_val
            )
            if section_key != DEFAULT_SECTION_KEY:
                self.section_marking_schemes[section_key] = section_marking_scheme
                for q in section_marking_scheme.questions:
                    self.question_to_scheme[q] = section_marking_scheme
                self.has_custom_marking = True
            else:
                self.default_marking_scheme = section_marking_scheme

    def validate_marking_schemes(self):
        section_marking_schemes = self.section_marking_schemes
        section_questions = set()
        for section_key, section_scheme in section_marking_schemes.items():
            if section_key == DEFAULT_SECTION_KEY:
                continue
            current_set = set(section_scheme.questions)
            if not section_questions.isdisjoint(current_set):
                raise Exception(
                    f"Section '{section_key}' has overlapping question(s) with other sections"
                )
            section_questions = section_questions.union(current_set)

        all_questions = set(self.questions_in_order)
        missing_questions = sorted(section_questions.difference(all_questions))
        if len(missing_questions) > 0:
            logger.critical(f"Missing answer key for: {missing_questions}")
            raise Exception(
                f"Some questions are missing in the answer key for the given marking scheme"
            )

    def parse_answers_and_map_questions(self, answers_in_order):
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

    def validate_answers(self, answers_in_order, tuning_config):
        answer_matcher_map = self.question_to_answer_matcher
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

    def __str__(self):
        return str(self.path)

    def get_exclude_files(self):
        return self.exclude_files

    # Externally called methods have higher abstraction level.
    def prepare_and_validate_omr_response(self, omr_response):
        self.reset_evaluation()

        omr_response_questions = set(omr_response.keys())
        all_questions = set(self.questions_in_order)
        missing_questions = sorted(all_questions.difference(omr_response_questions))
        if len(missing_questions) > 0:
            logger.critical(f"Missing OMR response for: {missing_questions}")
            raise Exception(
                f"Some questions are missing in the OMR response for the given answer key"
            )

        prefixed_omr_response_questions = set(
            [k for k in omr_response.keys() if k.startswith("q")]
        )
        missing_prefixed_questions = sorted(
            prefixed_omr_response_questions.difference(all_questions)
        )
        if len(missing_prefixed_questions) > 0:
            logger.warning(
                f"No answer given for potential questions in OMR response: {missing_prefixed_questions}"
            )

    def match_answer_for_question(self, current_score, question, marked_answer):
        answer_matcher = self.question_to_answer_matcher[question]
        question_verdict, delta = answer_matcher.get_verdict_marking(marked_answer)
        question_schema_verdict = self.get_schema_verdict(
            answer_matcher, question_verdict, delta
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
    ):
        if self.should_explain_scoring:
            next_score = current_score + delta
            # Conditionally add cells
            row = [
                item
                for item in [
                    question,
                    marked_answer,
                    str(answer_matcher),
                    f"{question_schema_verdict.title()} ({question_verdict})",
                    str(round(delta, 2)),
                    str(round(next_score, 2)),
                    (
                        answer_matcher.get_section_explanation()
                        if self.has_custom_marking
                        else None
                    ),
                ]
                if item is not None
            ]
            self.explanation_table.add_row(*row)

    def get_schema_verdict(self, answer_matcher, question_verdict, delta):
        # Note: Negative custom weights should be considered as incorrect schema verdict(special case)
        if (
            delta < 0
            and answer_matcher.answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED
        ):
            return SchemaVerdict.INCORRECT
        else:
            for verdict in VERDICTS_IN_ORDER:
                # using startswith to handle cases like matched-A
                if question_verdict.startswith(verdict):
                    schema_verdict = VERDICT_TO_SCHEMA_VERDICT[verdict]
                    return schema_verdict

    def conditionally_print_explanation(self):
        if self.should_explain_scoring:
            console.print(self.explanation_table, justify="center")

    def get_should_explain_scoring(self):
        return self.should_explain_scoring

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
        self.schema_verdict_counts = {
            schema_verdict: 0 for schema_verdict in SCHEMA_VERDICTS_IN_ORDER
        }
        self.prepare_explanation_table()

    def prepare_explanation_table(self):
        # TODO: provide a way to export this as csv/pdf
        if not self.should_explain_scoring:
            return
        table = Table(title="Evaluation Explanation Table", show_lines=True)
        table.add_column("Question")
        table.add_column("Marked")
        table.add_column("Answer(s)")
        table.add_column("Verdict")
        table.add_column("Delta")
        table.add_column("Score")
        # TODO: Add max and min score in explanation (row-wise and total)
        if self.has_custom_marking:
            table.add_column("Marking Scheme")
        self.explanation_table = table

    def get_evaluation_meta_for_question(
        self, question_meta, bubble_detection, image_type
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
            ["correct", "incorrect", "neutral", "bonus"],
        )
        if bubble_detection.is_marked:
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


def evaluate_concatenated_response(concatenated_response, evaluation_config):
    evaluation_config.prepare_and_validate_omr_response(concatenated_response)
    current_score = 0.0
    questions_meta = {}
    for question in evaluation_config.questions_in_order:
        marked_answer = concatenated_response[question]
        (
            delta,
            question_verdict,
            answer_matcher,
            question_schema_verdict,
        ) = evaluation_config.match_answer_for_question(
            current_score, question, marked_answer
        )
        marking_scheme = evaluation_config.get_marking_scheme_for_question(question)
        bonus_type = marking_scheme.get_bonus_type()
        current_score += delta
        questions_meta[question] = {
            "question_verdict": question_verdict,
            "marked_answer": marked_answer,
            "delta": delta,
            "current_score": current_score,
            "answer_item": answer_matcher.answer_item,
            "answer_type": answer_matcher.answer_type,
            "bonus_type": bonus_type,
            "question_schema_verdict": question_schema_verdict,
        }

    evaluation_config.conditionally_print_explanation()
    formatted_answers_summary, *_ = evaluation_config.get_formatted_answers_summary()
    evaluation_meta = {
        "score": current_score,
        "questions_meta": questions_meta,
        # "schema_verdict_counts": evaluation_config.schema_verdict_counts,
        "formatted_answers_summary": formatted_answers_summary,
    }
    return current_score, evaluation_meta
