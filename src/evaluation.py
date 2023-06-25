import ast
import os
import re
from copy import deepcopy

import cv2
import pandas as pd
from rich.table import Table

from src.logger import console, logger
from src.schemas.constants import (
    BONUS_SECTION_PREFIX,
    DEFAULT_SECTION_KEY,
    MARKING_VERDICT_TYPES,
)
from src.utils.parsing import (
    get_concatenated_response,
    open_evaluation_with_validation,
    parse_fields,
    parse_float_or_fraction,
)


class AnswerMatcher:
    def __init__(self, answer_item, marking_scheme):
        self.answer_type = self.get_answer_type(answer_item)
        self.parsed_answer = self.parse_answer_item(answer_item)
        self.set_defaults_from_scheme(marking_scheme)
        self.marking_scheme = marking_scheme

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
                and type(answer_item[0]) == str
                and type(answer_item[1]) == list
            ):
                return "multiple-correct-weighted"
            elif (
                len(answer_item) == 2
                and type(answer_item[0]) in list
                and type(answer_item[1]) == list
            ):
                return "answer-weights"
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
        elif answer_type == "multiple-correct-weighted":
            custom_marking = list(map(parse_float_or_fraction, parsed_answer[1]))
            verdict_types_length = min(len(MARKING_VERDICT_TYPES), len(custom_marking))
            # override the given marking
            for i in range(verdict_types_length):
                verdict_type = MARKING_VERDICT_TYPES[i]
                self.marking[verdict_type] = custom_marking[i]

            if type(parsed_answer[0] == str):
                allowed_answer = parsed_answer[0]
                self.marking[f"correct-{allowed_answer}"] = self.marking["correct"]
            else:
                for allowed_answer in parsed_answer[0]:
                    self.marking[f"correct-{allowed_answer}"] = self.marking["correct"]

    def get_marking_scheme(self):
        return self.marking_scheme

    def get_section_explanation(self):
        answer_type = self.answer_type
        if answer_type in ["standard", "multiple-correct"]:
            return self.marking_scheme.section_key
        elif answer_type == "multiple-correct-weighted":
            return f"Custom: {self.marking}"

    def get_verdict_marking(self, marked_answer):
        answer_type = self.answer_type
        if answer_type == "standard":
            question_verdict = self.get_standard_verdict(marked_answer)
            return question_verdict, self.marking[question_verdict]
        elif answer_type == "multiple-correct":
            question_verdict = self.get_multiple_correct_verdict(marked_answer)
            return question_verdict, self.marking[question_verdict]
        elif answer_type == "multiple-correct-weighted":
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

    def get_multi_weighted_verdict(self, marked_answer):
        return self.get_multiple_correct_verdict(marked_answer)

    def get_multiple_correct_verdict(self, marked_answer):
        parsed_answer = self.parsed_answer
        if marked_answer == self.empty_val:
            return "unmarked"
        elif marked_answer in parsed_answer:
            return f"correct-{marked_answer}"
        else:
            return "incorrect"

    def __str__(self):
        answer_type, parsed_answer = self.answer_type, self.parsed_answer
        if answer_type == "multiple-correct":
            return str(parsed_answer)
        elif answer_type == "multiple-correct-weighted":
            return f"{parsed_answer[0]}"
            # TODO: multi-lines in multi-weights
        return parsed_answer


class SectionMarkingScheme:
    def __init__(self, section_key, section_scheme, empty_val):
        # TODO: get local empty_val from qblock
        self.empty_val = empty_val
        self.section_key = section_key
        # DEFAULT marking scheme follows a shorthand
        if section_key == DEFAULT_SECTION_KEY:
            self.questions = None
            self.marking = self.parse_scheme_marking(section_scheme)
        else:
            self.questions = parse_fields(section_key, section_scheme["questions"])
            self.marking = self.parse_scheme_marking(section_scheme["marking"])

    def parse_scheme_marking(self, marking):
        parsed_marking = {}
        for verdict_type in MARKING_VERDICT_TYPES:
            verdict_marking = parse_float_or_fraction(marking[verdict_type])
            if (
                verdict_marking > 0
                and verdict_type == "incorrect"
                and not self.section_key.startswith(BONUS_SECTION_PREFIX)
            ):
                logger.warning(
                    f"Found positive marks({round(verdict_marking, 2)}) for incorrect answer in the schema '{self.section_key}'. For Bonus sections, add a prefix 'BONUS_' to them."
                )
            parsed_marking[verdict_type] = verdict_marking

        return parsed_marking

    def match_answer(self, marked_answer, answer_matcher):
        question_verdict, verdict_marking = answer_matcher.get_verdict_marking(
            marked_answer
        )

        return verdict_marking, question_verdict


class EvaluationConfig:
    """Note: this instance will be reused for multiple omr sheets"""

    def __init__(self, curr_dir, evaluation_path, template, tuning_config):
        self.path = evaluation_path
        evaluation_json = open_evaluation_with_validation(evaluation_path)
        options, marking_scheme, source_type = map(
            evaluation_json.get, ["options", "marking_scheme", "source_type"]
        )
        self.should_explain_scoring = options.get("should_explain_scoring", False)
        self.has_non_default_section = False
        self.exclude_files = []

        marking_scheme = marking_scheme

        if source_type == "csv":
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
                    converters={"question": str, "answer": self.parse_answer_column},
                )

                self.questions_in_order = answer_key["question"].to_list()
                answers_in_order = answer_key["answer"].to_list()
            elif not answer_key_image_path:
                raise Exception(f"Answer key csv not found at '{csv_path}'")
            else:
                image_path = str(curr_dir.joinpath(answer_key_image_path))
                if not os.path.exists(image_path):
                    raise Exception(f"Answer key image not found at '{image_path}'")

                # self.exclude_files.append(image_path)

                logger.debug(
                    f"Attempting to generate answer key from image: '{image_path}'"
                )
                # TODO: use a common function for below changes?
                in_omr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                in_omr = template.image_instance_ops.apply_preprocessors(
                    image_path, in_omr, template
                )
                if in_omr is None:
                    raise Exception(
                        f"Could not read answer key from image {image_path}"
                    )
                (
                    response_dict,
                    _final_marked,
                    _multi_marked,
                    _multi_roll,
                ) = template.image_instance_ops.read_omr_response(
                    template,
                    image=in_omr,
                    name=image_path,
                    save_dir=None,
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
                            f"Found empty answers for questions: {empty_answered_questions}, empty value used: '{empty_val}'"
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

        self.marking_scheme, self.question_to_scheme = {}, {}
        for section_key, section_scheme in marking_scheme.items():
            section_marking_scheme = SectionMarkingScheme(
                section_key, section_scheme, template.global_empty_val
            )
            if section_key != DEFAULT_SECTION_KEY:
                self.marking_scheme[section_key] = section_marking_scheme
                for q in section_marking_scheme.questions:
                    # TODO: check the answer key for custom scheme here?
                    self.question_to_scheme[q] = section_marking_scheme
                self.has_non_default_section = True
            else:
                self.default_marking_scheme = section_marking_scheme

        self.validate_marking_scheme()

        self.question_to_answer_matcher = self.parse_answers_and_map_questions(
            answers_in_order
        )
        self.validate_answers(answers_in_order, tuning_config)

    def __str__(self):
        return str(self.path)

    # Externally called methods have higher abstraction level.
    def prepare_and_validate_omr_response(self, omr_response):
        self.reset_explanation_table()

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
        self.conditionally_add_explanation(
            answer_matcher,
            delta,
            marked_answer,
            question_verdict,
            question,
            current_score,
        )
        return delta

    def conditionally_print_explanation(self):
        if self.should_explain_scoring:
            console.print(self.explanation_table, justify="center")

    def get_should_explain_scoring(self):
        return self.should_explain_scoring

    def get_exclude_files(self):
        return self.exclude_files

    @staticmethod
    def parse_answer_column(answer_column):
        if answer_column[0] == "[":
            parsed_answer = ast.literal_eval(answer_column)
        elif "," in answer_column:
            parsed_answer = answer_column.split(",")
        else:
            parsed_answer = answer_column
        return parsed_answer

    def parse_questions_in_order(self, questions_in_order):
        return parse_fields("questions_in_order", questions_in_order)

    def validate_answers(self, answers_in_order, tuning_config):
        answer_matcher_map = self.question_to_answer_matcher
        if tuning_config.outputs.filter_out_multimarked_files:
            multi_marked_answer = False
            for question, answer_item in zip(self.questions_in_order, answers_in_order):
                answer_type = answer_matcher_map[question].answer_type
                if answer_type == "standard":
                    if len(answer_item) > 1:
                        multi_marked_answer = True
                if answer_type == "multiple-correct":
                    for single_answer in answer_item:
                        if len(single_answer) > 1:
                            multi_marked_answer = True
                            break
                if answer_type == "multiple-correct-weighted":
                    if len(answer_item[0]) > 1:
                        multi_marked_answer = True

                if multi_marked_answer:
                    raise Exception(
                        f"Answer key contains multiple correct answer(s), but filter_out_multimarked_files is True. Scoring will get skipped."
                    )

    def validate_questions(self, answers_in_order):
        questions_in_order = self.questions_in_order
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

    def validate_marking_scheme(self):
        marking_scheme = self.marking_scheme
        section_questions = set()
        for section_key, section_scheme in marking_scheme.items():
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
            question_to_answer_matcher[question] = AnswerMatcher(
                answer_item, section_marking_scheme
            )
        return question_to_answer_matcher

    # Then unfolding lower abstraction levels
    def reset_explanation_table(self):
        self.explanation_table = None
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
        if self.has_non_default_section:
            table.add_column("Section")
        self.explanation_table = table

    def get_marking_scheme_for_question(self, question):
        return self.question_to_scheme.get(question, self.default_marking_scheme)

    def conditionally_add_explanation(
        self,
        answer_matcher,
        delta,
        marked_answer,
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
                    answer_matcher.get_section_explanation()
                    if self.has_non_default_section
                    else None,
                ]
                if item is not None
            ]
            self.explanation_table.add_row(*row)


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
