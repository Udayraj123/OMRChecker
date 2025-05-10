# TODO: create class for this to manage evaluation_meta for the directory? (Similar to TemplateFileRunner?)


class QuestionMeta:
    def __init__(
        self,
        question,
        question_verdict,
        marked_answer,
        delta,
        current_score,
        answer_matcher,
        bonus_type,
        question_schema_verdict,
    ):
        self.question = question
        self.question_verdict = question_verdict
        self.marked_answer = marked_answer
        self.delta = delta
        self.current_score = current_score
        self.answer_item = answer_matcher.answer_item
        self.answer_type = answer_matcher.answer_type
        self.bonus_type = bonus_type
        self.question_schema_verdict = question_schema_verdict

    def to_dict(self):
        return {
            "question_verdict": self.question_verdict,
            "marked_answer": self.marked_answer,
            "delta": self.delta,
            "current_score": self.current_score,
            "answer_item": self.answer_item,
            "answer_type": self.answer_type,
            "bonus_type": self.bonus_type,
            "question_schema_verdict": self.question_schema_verdict,
        }


class EvaluationMeta:
    def __init__(self):
        self.score = 0.0
        self.questions_meta = {}

    def add_question_meta(self, question, question_meta):
        self.questions_meta[question] = question_meta.to_dict()

    def to_dict(self, formatted_answers_summary):
        return {
            "score": self.score,
            "questions_meta": self.questions_meta,
            "formatted_answers_summary": formatted_answers_summary,
        }


# A utility to calculate score and metadata
def evaluate_concatenated_response(
    concatenated_response, evaluation_config_for_response
):
    evaluation_config_for_response.prepare_and_validate_omr_response(
        concatenated_response, allow_streak=True
    )
    evaluation_meta = EvaluationMeta()
    for question in evaluation_config_for_response.questions_in_order:
        marked_answer = concatenated_response[question]
        (
            delta,
            question_verdict,
            answer_matcher,
            question_schema_verdict,
        ) = evaluation_config_for_response.match_answer_for_question(
            evaluation_meta.score, question, marked_answer
        )
        marking_scheme = evaluation_config_for_response.get_marking_scheme_for_question(
            question
        )
        bonus_type = marking_scheme.get_bonus_type()
        evaluation_meta.score += delta
        question_meta = QuestionMeta(
            question,
            question_verdict,
            marked_answer,
            delta,
            evaluation_meta.score,
            answer_matcher,
            bonus_type,
            question_schema_verdict,
        )
        evaluation_meta.add_question_meta(question, question_meta)

    evaluation_config_for_response.conditionally_print_explanation()
    (
        formatted_answers_summary,
        *_,
    ) = evaluation_config_for_response.get_formatted_answers_summary()
    return evaluation_meta.score, evaluation_meta.to_dict(formatted_answers_summary)
