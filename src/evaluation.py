import os
from fractions import Fraction

import numpy as np
import pandas as pd
from rich.table import Table

from src.logger import console, logger
from src.utils.parsing import open_evaluation_with_validation


def parse_float_or_fraction(result):
    if "/" in result:
        result = float(Fraction(result))
    else:
        result = float(result)
    return result


class SectionMarkingScheme:
    DEFAULT_MARKING_SCHEME = "DEFAULT"
    MARKING_VERDICT_TYPES = ["correct", "incorrect", "unmarked"]

    def __init__(self, section_key, section_scheme, empty_val):
        # todo: get local empty_val from qblock
        self.empty_val = empty_val
        self.section_key = section_key
        self.streaks = {
            "correct": 0,
            "incorrect": 0,
            "unmarked": 0,
        }

        # DEFAULT marking scheme follows a shorthand
        if section_key == self.DEFAULT_MARKING_SCHEME:
            self.questions = None
            self.marking = self.parse_scheme_marking(section_scheme)
        else:
            self.questions = self.parse_scheme_questions(section_scheme["questions"])
            self.marking = self.parse_scheme_marking(section_scheme["marking"])

    def parse_scheme_marking(self, marking):
        parsed_marking = {}
        for verdict_type in self.MARKING_VERDICT_TYPES:
            result = marking[verdict_type]
            if type(result) == str:
                result = parse_float_or_fraction(result)

                if result > 0 and verdict_type == "incorrect":
                    logger.warning(
                        f"Found positive marks({round(result, 2)}) for incorrect answer in the schema '{self.section_key}'. Is this a Bonus Question?"
                    )
            elif type(result) == list:
                result = map(parse_float_or_fraction, result)

            parsed_marking[verdict_type] = result

        return parsed_marking

    def parse_scheme_questions(self, questions):
        # TODO: simplify this in schema itself
        # if(questions === "all"): <- handle this case in top parsing + validation

        # TODO: parse the range operator regex here
        parsed_questions = questions
        return parsed_questions

    def get_question_verdict(self, marked_answer, correct_answer):
        if marked_answer == self.empty_val:
            return "unmarked"
        elif marked_answer == correct_answer:
            return "correct"
        else:
            # TODO: support for multi-weighted
            return "incorrect"

    def update_streaks_for_verdict(self, question_verdict):
        current_streak = self.streaks[question_verdict]
        for verdict_type in self.MARKING_VERDICT_TYPES:
            if question_verdict == verdict_type:
                # increase current streak
                self.streaks[verdict_type] = current_streak + 1
            else:
                # reset other streaks
                self.streaks[verdict_type] = 0

    def match_answer(self, marked_answer, correct_answer):
        question_verdict = self.get_question_verdict(marked_answer, correct_answer)
        current_streak = self.streaks[question_verdict]
        marking = self.marking[question_verdict]
        if type(marking) == list:
            # TODO: add validation for max questions in streak scheme (len(questions) < len(marking))
            delta = marking[min(current_streak, len(marking) - 1)]
        else:
            delta = marking
        self.update_streaks_for_verdict(question_verdict)
        return delta, question_verdict


class EvaluationConfig:
    def __init__(self, local_evaluation_path, template, curr_dir):
        evaluation_json = open_evaluation_with_validation(
            local_evaluation_path, template, curr_dir
        )
        options = evaluation_json["options"]
        self.should_explain_scoring = options.get("should_explain_scoring", False)
        if self.should_explain_scoring:
            self.prepare_explanation_table()
        marking_scheme = evaluation_json["marking_scheme"]
        if evaluation_json["source_type"] == "csv":
            csv_path = curr_dir.joinpath(options["answer_key_path"])
            if not os.path.exists(csv_path):
                raise Exception(f"Answer key not found at {csv_path}")

            # TODO: CSV parsing
            answer_key = pd.read_csv(csv_path, dtype=str)
            self.questions_in_order = answer_key[0].to_list()
            self.answers_in_order = answer_key[1].to_list()
            # TODO: later parse complex answer schemes from csv itself (ans strings)
            # TODO: validate each row to contain a (qNo, <ans string/>) pair
        else:
            # - if source_type = custom, template should have all valid qNos (in instance?)
            self.questions_in_order = self.parse_questions_in_order(
                options["questions_in_order"]
            )
            self.answers_in_order = options["answers_in_order"]
        self.marking_scheme = {}
        for (section_key, section_scheme) in marking_scheme.items():
            # instance will allow easy readability, extensibility as well as streak sample
            self.marking_scheme[section_key] = SectionMarkingScheme(
                section_key, section_scheme, template.global_empty_val
            )

    def parse_questions_in_order(self, questions_in_order):
        # TODO: parse range operators here as well
        return questions_in_order

    def prepare_explanation_table(self):
        table = Table(show_lines=True)
        table.add_column("Question")
        table.add_column("Marked")
        table.add_column("Correct")
        table.add_column("Verdict")
        table.add_column("Delta")
        table.add_column("Score")
        table.add_column("Section")
        table.add_column("Section Streak", justify="right")
        self.explanation_table = table

    def conditionally_add_explanation(
        self,
        correct_answer,
        delta,
        marked_answer,
        question_marking_scheme,
        question_verdict,
        question,
        score,
    ):
        if self.should_explain_scoring:
            next_score = score + delta
            self.explanation_table.add_row(
                question,
                marked_answer,
                correct_answer,
                str.title(question_verdict),
                str(round(delta, 2)),
                str(round(next_score, 2)),
                question_marking_scheme.section_key,
                str(question_marking_scheme.streaks[question_verdict]),
            )

    def conditionally_print_explanation(self):
        if self.should_explain_scoring:
            console.print(self.explanation_table, justify="center")


def evaluate_concatenated_response(concatenated_response, evaluation_config):
    # first go with answers_in_order object, later just export it as csv to get csv format for docs.
    questions_in_order, answers_in_order, marking_scheme = map(
        evaluation_config.__dict__.get,
        ["questions_in_order", "answers_in_order", "marking_scheme"],
    )
    QUESTION_WISE_SCHEMES = {}
    for (section_key, section_marking_scheme) in marking_scheme.items():
        if section_key == SectionMarkingScheme.DEFAULT_MARKING_SCHEME:
            default_marking_scheme = marking_scheme[section_key]
        else:
            for q in section_marking_scheme.questions:
                QUESTION_WISE_SCHEMES[q] = section_marking_scheme

    score = 0.0
    for q_index, question in enumerate(questions_in_order):
        correct_answer = answers_in_order[q_index]
        # TODO: add validation for existence of each question in response keys
        marked_answer = concatenated_response[question]
        question_marking_scheme = QUESTION_WISE_SCHEMES.get(
            question, default_marking_scheme
        )
        delta, question_verdict = question_marking_scheme.match_answer(
            marked_answer, correct_answer
        )
        evaluation_config.conditionally_add_explanation(
            correct_answer,
            delta,
            marked_answer,
            question_marking_scheme,
            question_verdict,
            question,
            score,
        )
        score += delta

    evaluation_config.conditionally_print_explanation()

    return score


# Evaluate accuracy based on OMRDataset file generated through review
# portal on the same set of images
def evaluate_correctness(outputs_namespace):
    # TODO: test_file WOULD BE RELATIVE TO INPUT SUBDIRECTORY NOW-
    test_file = "inputs/OMRDataset.csv"
    if os.path.exists(test_file):
        logger.info(f"Starting evaluation for: '{test_file}'")

        test_cols = ["file_id"] + outputs_namespace.resp_cols
        y_df = (
            pd.read_csv(test_file, dtype=str)[test_cols]
            .replace(np.nan, "", regex=True)
            .set_index("file_id")
        )

        if np.any(y_df.index.duplicated):
            y_df_filtered = y_df.loc[~y_df.index.duplicated(keep="first")]
            logger.warning(
                f"WARNING: Found duplicate File-ids in file '{test_file}'. Removed {y_df.shape[0] - y_df_filtered.shape[0]} rows from testing data. Rows remaining: {y_df_filtered.shape[0]}"
            )
            y_df = y_df_filtered

        x_df = pd.DataFrame(
            outputs_namespace.OUTPUT_SET, dtype=str, columns=test_cols
        ).set_index("file_id")
        # print("x_df",x_df.head())
        # print("\ny_df",y_df.head())
        intersection = y_df.index.intersection(x_df.index)

        # Checking if the merge is okay
        if intersection.size == x_df.index.size:
            y_df = y_df.loc[intersection]
            x_df["TestResult"] = (x_df == y_df).all(axis=1).astype(int)
            logger.info(x_df.head())
            logger.info(
                f"\n\t Accuracy on the {test_file} Dataset: {round((x_df['TestResult'].sum() / x_df.shape[0]),6)}"
            )
        else:
            logger.error(
                "\nERROR: Insufficient Testing Data: Have you appended MultiMarked data yet?"
            )
            logger.error(
                f"Missing File-ids: {list(x_df.index.difference(intersection))}"
            )
