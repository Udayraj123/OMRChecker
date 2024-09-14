# TODO: create class for this to manage evaluation_meta for the directory? (Similar to TemplateFileRunner?)


# A utility to calculate score and metadata
def evaluate_concatenated_response(
    concatenated_response, evaluation_config_for_response
):
    evaluation_config_for_response.prepare_and_validate_omr_response(
        concatenated_response, allow_streak=True
    )
    current_score = 0.0
    questions_meta = {}
    for question in evaluation_config_for_response.questions_in_order:
        marked_answer = concatenated_response[question]
        (
            delta,
            question_verdict,
            answer_matcher,
            question_schema_verdict,
        ) = evaluation_config_for_response.match_answer_for_question(
            current_score, question, marked_answer
        )
        marking_scheme = evaluation_config_for_response.get_marking_scheme_for_question(
            question
        )
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

    evaluation_config_for_response.conditionally_print_explanation()
    (
        formatted_answers_summary,
        *_,
    ) = evaluation_config_for_response.get_formatted_answers_summary()
    evaluation_meta = {
        "score": current_score,
        "questions_meta": questions_meta,
        # "schema_verdict_counts": evaluation_config_for_response.schema_verdict_counts,
        "formatted_answers_summary": formatted_answers_summary,
    }
    return current_score, evaluation_meta
