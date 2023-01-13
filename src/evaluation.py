import os
import re
from copy import deepcopy
from fractions import Fraction

import pandas as pd
from rich.table import Table

from src.logger import console, logger
from src.schemas.evaluation_schema import (
    BONUS_SECTION_PREFIX,
    DEFAULT_SECTION_KEY,
    MARKING_VERDICT_TYPES,
)
from src.utils.parsing import open_evaluation_with_validation


def parse_float_or_fraction(result):
    if type(result) == str and "/" in result:
        result = float(Fraction(result))
    else:
        result = float(result)
    return result


class AnswerMatcher:
    def __init__(self, answer_item, marking_scheme):
        self.answer_type = self.get_answer_type(answer_item)
        self.parsed_answer = self.parse_answer_item(answer_item)
        self.set_defaults_from_scheme(marking_scheme)
        self.marking_scheme = marking_scheme

    def get_marking_scheme(self):
        return self.marking_scheme

    def get_section_explanation(self):
        answer_type = self.answer_type
        if answer_type in ["standard", "multiple-correct"]:
            return self.marking_scheme.section_key
        elif answer_type == "multi-weighted":
            return f"Custom\n{self.marking}"

    def get_answer_type(self, answer_item):
        item_type = type(answer_item)
        if item_type == str:
            return "standard"
        elif item_type == list:
            if (
                len(answer_item) >= 2
                and type(answer_item[0]) == str
                and type(answer_item[1]) == str
            ):
                return "multiple-correct"
            elif (
                len(answer_item) == 2
                and type(answer_item[0]) in [str, list]
                and type(answer_item[1]) == list
            ):
                return "multi-weighted"
            # elif (
            #     len(answer_item) == 3
            #     and type(answer_item[0]) == list
            #     and type(answer_item[1]) == list
            # ):
            #     return "multi-weighted"
            else:
                logger.critical(
                    f"Unable to determine answer type for answer item: {answer_item}"
                )
                raise Exception("Unable to determine answer type")

    def parse_answer_item(self, answer_item):
        return answer_item

    def set_defaults_from_scheme(self, marking_scheme):
        answer_type = self.answer_type
        self.empty_val = marking_scheme.empty_val
        parsed_answer = self.parsed_answer
        self.marking = deepcopy(marking_scheme.marking)
        # TODO: reuse part of parse_scheme_marking here -
        if answer_type == "standard":
            # no local overrides
            pass
        elif answer_type == "multiple-correct":
            # no local overrides
            for allowed_answer in parsed_answer:
                self.marking[f"correct-{allowed_answer}"] = self.marking["correct"]
        elif answer_type == "multi-weighted":
            # TODO: think about streaks in multi-weighted scenario (or invalidate such cases)
            custom_marking = list(map(parse_float_or_fraction, parsed_answer[1]))
            verdict_types_length = min(len(MARKING_VERDICT_TYPES), len(custom_marking))
            # override the given marking
            for i in range(verdict_types_length):
                verdict_type = MARKING_VERDICT_TYPES[i]
                # Note: copies over the marking regardless of streak or not -
                self.marking[verdict_type] = custom_marking[i]

            if type(parsed_answer[0] == str):
                allowed_answer = parsed_answer[0]
                self.marking[f"correct-{allowed_answer}"] = self.marking["correct"]
            else:
                for allowed_answer in parsed_answer[0]:
                    self.marking[f"correct-{allowed_answer}"] = self.marking["correct"]

    def get_verdict_marking(self, marked_answer):
        answer_type = self.answer_type
        if answer_type == "standard":
            question_verdict = self.get_standard_verdict(marked_answer)
            return question_verdict, self.marking[question_verdict]
        elif answer_type == "multiple-correct":
            question_verdict = self.get_multiple_correct_verdict(marked_answer)
            return question_verdict, self.marking[question_verdict]
        elif answer_type == "multi-weighted":
            question_verdict = self.get_multi_weighted_verdict(marked_answer)
            return question_verdict, self.marking[question_verdict]

    def get_standard_verdict(self, marked_answer):
        parsed_answer = self.parsed_answer
        if marked_answer == self.empty_val:
            return "unmarked"
        elif marked_answer == parsed_answer:
            return "correct"
        else:
            return "incorrect"

    def get_multiple_correct_verdict(self, marked_answer):
        parsed_answer = self.parsed_answer
        if marked_answer == self.empty_val:
            return "unmarked"
        elif marked_answer in parsed_answer:
            return f"correct-{marked_answer}"
        else:
            return "incorrect"

    def get_multi_weighted_verdict(self, marked_answer):
        return self.get_multiple_correct_verdict(marked_answer)

    def __str__(self):
        answer_type, parsed_answer = self.answer_type, self.parsed_answer
        if answer_type == "multiple-correct":
            return str(parsed_answer)
        elif answer_type == "multi-weighted":
            return f"{parsed_answer[0]}"
            # TODO: multi-lines in multi-weights
        return parsed_answer


class SectionMarkingScheme:
    def __init__(self, section_key, section_scheme, empty_val):
        # TODO: get local empty_val from qblock
        self.empty_val = empty_val
        self.section_key = section_key
        self.is_streak_scheme_present = False
        self.reset_side_effects()
        # DEFAULT marking scheme follows a shorthand
        if section_key == DEFAULT_SECTION_KEY:
            self.questions = None
            self.marking = self.parse_scheme_marking(section_scheme)
            if self.is_streak_scheme_present:
                raise Exception(
                    f"Default schema '{DEFAULT_SECTION_KEY}' cannot have streak marking. Create a new section and specify questions range in it."
                )

        else:
            self.questions = self.parse_questions(
                section_key, section_scheme["questions"]
            )
            self.marking = self.parse_scheme_marking(section_scheme["marking"])

    def reset_side_effects(self):
        self.streaks = {
            "correct": 0,
            "incorrect": 0,
            "unmarked": 0,
        }

    def parse_scheme_marking(self, marking):
        parsed_marking = {}
        for verdict_type in MARKING_VERDICT_TYPES:
            verdict_marking = marking[verdict_type]
            if type(verdict_marking) == str:
                verdict_marking = parse_float_or_fraction(verdict_marking)
                section_key = self.section_key
                if (
                    verdict_marking > 0
                    and verdict_type == "incorrect"
                    and not section_key.startswith(BONUS_SECTION_PREFIX)
                ):
                    logger.warning(
                        f"Found positive marks({round(verdict_marking, 2)}) for incorrect answer in the schema '{section_key}'. For Bonus sections, add a prefix 'BONUS_' to them."
                    )
            elif type(verdict_marking) == list:
                self.is_streak_scheme_present = True
                verdict_marking = list(map(parse_float_or_fraction, verdict_marking))

            parsed_marking[verdict_type] = verdict_marking

        return parsed_marking

    def has_streak_scheme(self):
        return self.is_streak_scheme_present

    @staticmethod
    def parse_questions(key, questions):
        parsed_questions = []
        questions_set = set()
        for question_string in questions:
            questions_array = SectionMarkingScheme.parse_question_string(
                question_string
            )
            current_set = set(questions_array)
            if not questions_set.isdisjoint(current_set):
                raise Exception(
                    f"Given question string '{question_string}' has overlapping question(s) with other questions in '{key}': {questions}"
                )
            parsed_questions.extend(questions_array)
        return parsed_questions

    @staticmethod
    def parse_question_string(question_string):
        if "." in question_string:
            question_prefix, start, end = re.findall(
                r"([^\.\d]+)(\d+)\.{2,3}(\d+)", question_string
            )[0]
            start, end = int(start), int(end)
            if start >= end:
                raise Exception(
                    f"Invalid range in question string: '{question_string}', start: {start} is not less than end: {end}"
                )
            return [
                f"{question_prefix}{question_number}"
                for question_number in range(start, end + 1)
            ]
        else:
            return [question_string]

    def match_answer(self, marked_answer, answer_matcher):
        question_verdict, verdict_marking = answer_matcher.get_verdict_marking(
            marked_answer
        )

        if type(verdict_marking) == list:
            current_streak = self.streaks[question_verdict]
            delta = verdict_marking[min(current_streak, len(verdict_marking) - 1)]
        else:
            delta = verdict_marking

        if question_verdict in MARKING_VERDICT_TYPES:  # TODO: refine this check
            self.update_streaks_for_verdict(question_verdict)

        return delta, question_verdict

    def update_streaks_for_verdict(self, question_verdict):
        current_streak = self.streaks[question_verdict]
        for verdict_type in MARKING_VERDICT_TYPES:
            if question_verdict == verdict_type:
                # increase current streak
                self.streaks[verdict_type] = current_streak + 1
            else:
                # reset other streaks
                self.streaks[verdict_type] = 0


class EvaluationConfig:
    """Note: this instance will be reused for multiple omr sheets"""

    def __init__(self, local_evaluation_path, template, curr_dir):
        evaluation_json = open_evaluation_with_validation(local_evaluation_path)
        options, marking_scheme, source_type = map(
            evaluation_json.get, ["options", "marking_scheme", "source_type"]
        )
        self.should_explain_scoring = options.get("should_explain_scoring", False)
        self.is_streak_scheme_present = False
        self.exclude_files = []

        marking_scheme = marking_scheme
        if source_type == "csv":
            csv_path = curr_dir.joinpath(options["answer_key_csv_path"])

            if not os.path.exists(csv_path):
                logger.warning(f"Note: Answer key csv does not exist at: '{csv_path}'.")

                answer_key_image_path = options.get("answer_key_image_path", None)
                if not answer_key_image_path:
                    raise Exception(f"Answer key csv not found at '{csv_path}'")

                image_path = curr_dir.joinpath(answer_key_image_path)

                if os.path.exists(image_path):
                    raise Exception(f"Answer key image not found at '{image_path}'")

                # TODO: trigger parent's omr reading for 'image_path' with evaluation_columns (only for regenerate)
                # TODO: think about upcoming plugins as we'd be going out of the execution flow
                logger.debug(f"Attempting to generate csv from image: '{image_path}'")

                self.exclude_files.extend(image_path)

            # TODO: CSV parsing/validation for each row with a (qNo, <ans string/>) pair
            answer_key = pd.read_csv(csv_path, dtype=str, header=None)

            self.questions_in_order = answer_key[0].to_list()
            # TODO: parse array syntax here -
            answers_in_order = answer_key[1].to_list()
            self.validate_questions(answers_in_order)
        else:
            self.questions_in_order = self.parse_questions_in_order(
                options["questions_in_order"]
            )
            answers_in_order = options["answers_in_order"]
        self.validate_questions(answers_in_order)

        self.marking_scheme, self.question_to_scheme = {}, {}
        for (section_key, section_scheme) in marking_scheme.items():
            section_marking_scheme = SectionMarkingScheme(
                section_key, section_scheme, template.global_empty_val
            )
            if section_key != DEFAULT_SECTION_KEY:
                self.marking_scheme[section_key] = section_marking_scheme
                for q in section_marking_scheme.questions:
                    # check the answer key for custom scheme here?
                    self.question_to_scheme[q] = section_marking_scheme
            else:
                self.default_marking_scheme = section_marking_scheme

            if section_marking_scheme.has_streak_scheme():
                self.is_streak_scheme_present = True

        self.validate_marking_scheme()

        self.question_to_answer_matcher = self.parse_answers_and_map_questions(
            answers_in_order
        )

    def get_marking_scheme_for_question(self, question):
        return self.question_to_scheme.get(question, self.default_marking_scheme)

    def match_answer_for_question(self, current_score, question, marked_answer):
        answer_matcher = self.question_to_answer_matcher[question]
        question_marking_scheme = answer_matcher.get_marking_scheme()
        delta, question_verdict = question_marking_scheme.match_answer(
            marked_answer, answer_matcher
        )
        self.conditionally_add_explanation(
            answer_matcher,
            delta,
            marked_answer,
            question_marking_scheme,
            question_verdict,
            question,
            current_score,
        )
        return delta

    def get_should_explain_scoring(self):
        return self.should_explain_scoring

    def get_exclude_files(self):
        return self.exclude_files

    def validate_questions(self, answers_in_order):
        questions_in_order = self.questions_in_order
        if len(questions_in_order) != len(answers_in_order):
            logger.critical(
                f"questions_in_order: {questions_in_order}\nanswers_in_order: {answers_in_order}"
            )
            raise Exception(
                f"Unequal lengths for questions_in_order and answers_in_order"
            )

    def validate_marking_scheme(self):
        marking_scheme = self.marking_scheme
        section_questions = set()
        for (section_key, section_scheme) in marking_scheme.items():
            if section_key == DEFAULT_SECTION_KEY:
                continue
            current_set = set(section_scheme.questions)
            if not section_questions.isdisjoint(current_set):
                raise Exception(
                    f"Section '{section_key}' has overlapping question(s) with other sections"
                )
            section_questions = section_questions.union(current_set)
            questions_count = len(section_scheme.questions)
            for verdict_type in MARKING_VERDICT_TYPES:
                marking = section_scheme.marking[verdict_type]
                if type(marking) == list and questions_count > len(marking):
                    logger.critical(
                        f"Section '{section_key}' with {questions_count} questions is more than the capacity for '{verdict_type}' marking with {len(marking)} scores."
                    )
                    raise Exception(
                        f"Section '{section_key}' has more questions than streak for '{verdict_type}' type"
                    )

        all_questions = set(self.questions_in_order)
        missing_questions = sorted(section_questions.difference(all_questions))
        if len(missing_questions) > 0:
            logger.critical(f"Missing answer key for: {missing_questions}")
            raise Exception(
                f"Some questions are missing in the answer key for the given marking scheme"
            )

    def prepare_and_validate_omr_response(self, omr_response):
        self.reset_explanation_table()
        self.reset_sections()
        omr_response_questions = set(omr_response.keys())
        all_questions = set(self.questions_in_order)
        missing_questions = sorted(all_questions.difference(omr_response_questions))
        if len(missing_questions) > 0:
            logger.critical(f"Missing OMR response for: {missing_questions}")
            # TODO: support for else case after skipping non-scored columns when evaluation_columns is provided
            raise Exception(
                f"Some questions are missing in the OMR response for the given answer key"
            )

    def parse_answers_and_map_questions(self, answers_in_order):
        question_to_answer_matcher = {}
        for (question, answer_item) in zip(self.questions_in_order, answers_in_order):
            section_marking_scheme = self.get_marking_scheme_for_question(question)
            question_to_answer_matcher[question] = AnswerMatcher(
                answer_item, section_marking_scheme
            )
        return question_to_answer_matcher

    def parse_questions_in_order(self, questions_in_order):
        return SectionMarkingScheme.parse_questions(
            "questions_in_order", questions_in_order
        )

    def prepare_explanation_table(self):
        # TODO: provide a way to export this as csv

        if not self.should_explain_scoring:
            return
        table = Table(show_lines=True)
        table.add_column("Question")
        table.add_column("Marked")
        table.add_column("Answer(s)")
        table.add_column("Verdict")
        table.add_column("Delta")
        table.add_column("Score")
        table.add_column("Section")
        if self.is_streak_scheme_present:
            table.add_column("Section Streak", justify="right")
        self.explanation_table = table

    def conditionally_add_explanation(
        self,
        answer_matcher,
        delta,
        marked_answer,
        question_marking_scheme,
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
                    str.title(question_verdict),
                    str(round(delta, 2)),
                    str(round(next_score, 2)),
                    answer_matcher.get_section_explanation(),
                    str(question_marking_scheme.streaks[question_verdict])
                    if self.is_streak_scheme_present
                    else None,
                ]
                if item is not None
            ]
            self.explanation_table.add_row(*row)

    def conditionally_print_explanation(self):
        if self.should_explain_scoring:
            console.print(self.explanation_table, justify="center")

    def reset_explanation_table(self):
        self.explanation_table = None
        self.prepare_explanation_table()

    def reset_sections(self):
        for section_scheme in self.marking_scheme.values():
            section_scheme.reset_side_effects()


def evaluate_concatenated_response(concatenated_response, evaluation_config):
    evaluation_config.prepare_and_validate_omr_response(concatenated_response)
    current_score = 0.0
    for question in evaluation_config.questions_in_order:
        marked_answer = concatenated_response[question]
        delta = evaluation_config.match_answer_for_question(
            current_score, question, marked_answer
        )
        current_score += delta

    evaluation_config.conditionally_print_explanation()

    return current_score
