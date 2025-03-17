from copy import deepcopy

from src.schemas.constants import (
    VERDICT_TO_SCHEMA_VERDICT,
    VERDICTS_IN_ORDER,
    AnswerType,
    SchemaVerdict,
    Verdict,
)
from src.utils.logger import logger
from src.utils.parsing import parse_float_or_fraction


class AnswerMatcher:
    def __init__(self, answer_item, section_marking_scheme):
        self.section_marking_scheme = section_marking_scheme
        self.answer_type = self.get_answer_type(answer_item)
        self.parse_and_set_answer_item(answer_item)
        self.set_local_marking_defaults(section_marking_scheme)

    @staticmethod
    def is_part_of_some_answer(question_meta, answer_string):
        if question_meta["bonus_type"] is not None:
            return True
        matched_groups = AnswerMatcher.get_matched_answer_groups(
            question_meta, answer_string
        )
        return len(matched_groups) > 0

    @staticmethod
    def get_matched_answer_groups(question_meta, answer_string):
        matched_groups = []
        answer_type, answer_item = map(
            question_meta.get, ["answer_type", "answer_item"]
        )

        if answer_type == AnswerType.STANDARD:
            # Note: implicit check on concatenated answer
            if answer_string in str(answer_item):
                matched_groups.append(0)
        if answer_type == AnswerType.MULTIPLE_CORRECT:
            for answer_index, allowed_answer in enumerate(answer_item):
                if answer_string in allowed_answer:
                    matched_groups.append(answer_index)
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            for answer_index, (allowed_answer, score) in enumerate(answer_item):
                if answer_string in allowed_answer and score > 0:
                    matched_groups.append(answer_index)
        return matched_groups

    @staticmethod
    def is_a_marking_score(answer_element):
        # Note: strict type checking is already done at schema validation level,
        # Here we focus on overall struct type
        return type(answer_element) == str or type(answer_element) == int

    @staticmethod
    def is_standard_answer(answer_element):
        return type(answer_element) == str and len(answer_element) >= 1

    @staticmethod
    def get_schema_verdict(answer_type, question_verdict, delta=None):
        # Note: Negative custom weights should be considered as incorrect schema verdict(special case)
        if delta < 0 and answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            return SchemaVerdict.INCORRECT
        else:
            for verdict in VERDICTS_IN_ORDER:
                # using startswith to handle cases like matched-A
                if question_verdict.startswith(verdict):
                    schema_verdict = VERDICT_TO_SCHEMA_VERDICT[verdict]
                    return schema_verdict

    @staticmethod
    def parse_verdict_marking(marking):
        if type(marking) == list:
            return list(map(parse_float_or_fraction, marking))

        return parse_float_or_fraction(marking)

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
                [
                    allowed_answer,
                    AnswerMatcher.parse_verdict_marking(answer_score),
                ]
                for allowed_answer, answer_score in answer_item
            ]
        else:
            self.answer_item = answer_item

    def set_local_marking_defaults(self, section_marking_scheme):
        answer_type = self.answer_type
        self.empty_value = section_marking_scheme.empty_value
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

    # def get_marking_scheme(self):
    #     return self.section_marking_scheme

    def get_section_explanation(self):
        answer_type = self.answer_type
        if answer_type in [AnswerType.STANDARD, AnswerType.MULTIPLE_CORRECT]:
            return self.section_marking_scheme.section_key
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            return f"Custom Weights: {self.marking}"

    def get_matched_set_name(self):
        return self.section_marking_scheme.set_name

    def get_verdict_marking(self, marked_answer, allow_streak=False):
        answer_type = self.answer_type
        question_verdict = Verdict.NO_ANSWER_MATCH
        if answer_type == AnswerType.STANDARD:
            question_verdict = self.get_standard_verdict(marked_answer)
        elif answer_type == AnswerType.MULTIPLE_CORRECT:
            question_verdict = self.get_multiple_correct_verdict(marked_answer)
        elif answer_type == AnswerType.MULTIPLE_CORRECT_WEIGHTED:
            question_verdict = self.get_multiple_correct_weighted_verdict(marked_answer)

        (
            delta,
            current_streak,
            updated_streak,
        ) = self.section_marking_scheme.get_delta_and_update_streak(
            self.marking, answer_type, question_verdict, allow_streak
        )

        return question_verdict, delta, current_streak, updated_streak

    def get_standard_verdict(self, marked_answer):
        allowed_answer = self.answer_item
        if marked_answer == self.empty_value:
            return Verdict.UNMARKED
        elif marked_answer == allowed_answer:
            return Verdict.ANSWER_MATCH
        else:
            return Verdict.NO_ANSWER_MATCH

    def get_multiple_correct_verdict(self, marked_answer):
        allowed_answers = self.answer_item
        if marked_answer == self.empty_value:
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
        if marked_answer == self.empty_value:
            return Verdict.UNMARKED
        elif marked_answer in allowed_answers:
            return f"{Verdict.ANSWER_MATCH}-{marked_answer}"
        else:
            return Verdict.NO_ANSWER_MATCH

    def __str__(self):
        return f"{self.answer_item}"
