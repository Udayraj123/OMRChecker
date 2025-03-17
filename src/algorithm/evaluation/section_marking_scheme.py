from copy import deepcopy

from src.algorithm.evaluation.answer_matcher import AnswerMatcher
from src.schemas.constants import (
    BONUS_SECTION_PREFIX,
    DEFAULT_SECTION_KEY,
    SCHEMA_VERDICTS_IN_ORDER,
    VERDICT_TO_SCHEMA_VERDICT,
    VERDICTS_IN_ORDER,
    MarkingSchemeType,
    SchemaVerdict,
    Verdict,
)
from src.utils.logger import logger
from src.utils.parsing import parse_fields


class SectionMarkingScheme:
    def __init__(self, section_key, section_scheme, set_name, empty_value):
        # TODO: get local empty_value from qblock
        self.empty_value = empty_value
        self.section_key = section_key
        self.set_name = set_name
        self.marking_type = section_scheme.get("marking_type", "default")
        self.reset_all_streaks()

        # DEFAULT marking scheme follows a shorthand
        if section_key == DEFAULT_SECTION_KEY:
            self.questions = None
            self.marking = self.parse_verdict_marking_from_scheme(section_scheme)
        else:
            self.questions = parse_fields(section_key, section_scheme["questions"])
            self.marking = self.parse_verdict_marking_from_scheme(
                section_scheme["marking"]
            )
        self.validate_marking_scheme()

    def reset_all_streaks(self):
        # TODO: support for MarkingSchemeType.SECTION_LEVEL_STREAK
        if self.marking_type == MarkingSchemeType.VERDICT_LEVEL_STREAK:
            self.streaks = {
                schema_verdict: 0 for schema_verdict in SCHEMA_VERDICTS_IN_ORDER
            }
        else:
            self.section_level_streak = 0
            self.previous_streak_verdict = None

    def get_delta_for_verdict(
        self, answer_matcher_marking, question_verdict, current_streak
    ):
        if type(answer_matcher_marking[question_verdict]) == list:
            return answer_matcher_marking[question_verdict][current_streak]
        else:
            if current_streak > 0:
                logger.warning(
                    f"Non zero streak({current_streak}) for verdict {question_verdict} in scheme {self}. Using non-streak score for this verdict."
                )
            return answer_matcher_marking[question_verdict]

    def get_delta_and_update_streak(
        self, answer_matcher_marking, answer_type, question_verdict, allow_streak
    ):
        schema_verdict = AnswerMatcher.get_schema_verdict(
            answer_type, question_verdict, 0
        )

        # TODO: support for MarkingSchemeType.SECTION_LEVEL_STREAK

        if self.marking_type == MarkingSchemeType.VERDICT_LEVEL_STREAK:
            current_streak = self.streaks[schema_verdict]
            # reset all
            self.reset_all_streaks()

            if allow_streak:
                # increase only current verdict streak
                if schema_verdict != SchemaVerdict.UNMARKED:
                    self.streaks[schema_verdict] = current_streak + 1

            delta = self.get_delta_for_verdict(
                answer_matcher_marking, question_verdict, current_streak
            )
            updated_streak = self.streaks[schema_verdict]

        elif self.marking_type == MarkingSchemeType.SECTION_LEVEL_STREAK:
            # TODO: add tests for this type

            # Note: this is assuming that the parent is calling in order of section question.
            current_streak = self.section_level_streak
            previous_verdict = self.previous_streak_verdict
            # reset all
            self.reset_all_streaks()

            if allow_streak:
                # increase only current verdict streak
                if previous_verdict is None or schema_verdict == previous_verdict:
                    self.section_level_streak = current_streak + 1

            delta = self.get_delta_for_verdict(
                answer_matcher_marking, question_verdict, current_streak
            )
            updated_streak = self.section_level_streak
        else:
            current_streak, updated_streak = 0, 0
            delta = self.get_delta_for_verdict(
                answer_matcher_marking, question_verdict, current_streak
            )

        return delta, current_streak, updated_streak

    def deepcopy_with_questions(self, questions):
        clone = deepcopy(self)
        clone.update_questions(questions)
        return clone

    def update_questions(self, questions):
        self.questions = questions
        self.validate_marking_scheme()

    def validate_marking_scheme(self):
        # TODO: add validation on maximum streak possible vs length of provided section verdict marking
        pass

    def __str__(self):
        return self.section_key

    def parse_verdict_marking_from_scheme(self, section_scheme):
        parsed_marking = {}
        for verdict in VERDICTS_IN_ORDER:
            schema_verdict = VERDICT_TO_SCHEMA_VERDICT[verdict]
            schema_verdict_marking = AnswerMatcher.parse_verdict_marking(
                section_scheme[schema_verdict]
            )
            if (
                self.marking_type == MarkingSchemeType.DEFAULT
                and schema_verdict_marking > 0
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
        if self.marking_type == MarkingSchemeType.VERDICT_LEVEL_STREAK:
            return None
        elif self.marking[Verdict.NO_ANSWER_MATCH] <= 0:
            return None
        elif self.marking[Verdict.UNMARKED] > 0:
            return "BONUS_FOR_ALL"
        elif self.marking[Verdict.UNMARKED] == 0:
            return "BONUS_ON_ATTEMPT"
        else:
            return None
