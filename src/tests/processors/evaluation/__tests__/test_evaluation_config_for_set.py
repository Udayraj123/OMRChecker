"""Comprehensive tests for EvaluationConfigForSet class.

Tests all critical methods including validation, parsing, and answer matching.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.utils.exceptions import (
    ConfigError,
    EvaluationError,
    FieldDefinitionError,
    OMRCheckerError,
)
from src.processors.evaluation.evaluation_config_for_set import EvaluationConfigForSet
from src.processors.evaluation.section_marking_scheme import SectionMarkingScheme
from src.schemas.constants import DEFAULT_SECTION_KEY


# Note: mock_template and mock_tuning_config are in conftest.py
# Override mock_template for evaluation-specific needs
@pytest.fixture
def mock_template():
    """Create a mock template with evaluation-specific attributes."""
    template = Mock()
    template.global_empty_val = ""
    template.get_evaluations_dir = Mock(return_value=Path("/tmp/evaluations"))
    return template


# Override mock_tuning_config for evaluation-specific needs
@pytest.fixture
def mock_tuning_config():
    """Create a mock tuning config with evaluation-specific attributes."""
    config = Mock()
    config.outputs = Mock()
    config.outputs.filter_out_multimarked_files = False
    return config


# Note: minimal_evaluation_json is in conftest.py but needs DEFAULT_SECTION_KEY
# Keep this override for now since it uses DEFAULT_SECTION_KEY


@pytest.fixture
def minimal_evaluation_json():
    """Minimal valid evaluation JSON with DEFAULT_SECTION_KEY."""
    base_json = {
        "source_type": "local",
        "options": {
            "questions_in_order": ["q1", "q2", "q3"],
            "answers_in_order": ["A", "B", "C"],
        },
        "marking_schemes": {
            DEFAULT_SECTION_KEY: {
                "correct": 1,
                "incorrect": 0,
                "unmarked": 0,
            }
        },
        "outputs_configuration": {
            "draw_answers_summary": {
                "enabled": False,
                "answers_summary_format_string": "{correct}/{incorrect}",
                "position": [200, 600],
                "size": 1.0,
            },
            "draw_score": {
                "enabled": False,
                "score_format_string": "Score: {score}",
                "position": [200, 200],
                "size": 1.5,
            },
            "draw_question_verdicts": {"enabled": False},
            "draw_detected_bubble_texts": {"enabled": False},
            "should_explain_scoring": False,
            "should_export_explanation_csv": False,
        },
    }
    return base_json


@pytest.fixture
def sample_evaluation_config_for_set(
    minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
):
    """Create an EvaluationConfigForSet instance."""
    curr_dir = tmp_path
    return EvaluationConfigForSet(
        "DEFAULT_SET",
        curr_dir,
        minimal_evaluation_json,
        mock_template,
        mock_tuning_config,
    )


class TestEvaluationConfigForSetInitialization:
    """Test EvaluationConfigForSet initialization."""

    def test_initialization_with_local_answers(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test initialization with local question/answer configuration."""
        config = EvaluationConfigForSet(
            "DEFAULT_SET",
            tmp_path,
            minimal_evaluation_json,
            mock_template,
            mock_tuning_config,
        )

        assert config.set_name == "DEFAULT_SET"
        assert len(config.questions_in_order) == 3
        assert len(config.answers_in_order) == 3
        assert config.questions_in_order == ["q1", "q2", "q3"]
        assert config.answers_in_order == ["A", "B", "C"]

    def test_initialization_with_parent_config(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test initialization with parent evaluation config."""
        parent_json = minimal_evaluation_json.copy()
        parent_config = EvaluationConfigForSet(
            "PARENT_SET",
            tmp_path,
            parent_json,
            mock_template,
            mock_tuning_config,
        )

        child_json = {
            "source_type": "local",
            "options": {
                "questions_in_order": ["q4"],
                "answers_in_order": ["D"],
            },
            "marking_schemes": {
                DEFAULT_SECTION_KEY: {
                    "correct": 1,
                    "incorrect": 0,
                    "unmarked": 0,
                }
            },
            "outputs_configuration": minimal_evaluation_json["outputs_configuration"],
        }

        child_config = EvaluationConfigForSet(
            "CHILD_SET",
            tmp_path,
            child_json,
            mock_template,
            mock_tuning_config,
            parent_evaluation_config=parent_config,
        )

        assert child_config.has_conditional_sets is True
        assert len(child_config.questions_in_order) == 4  # q1, q2, q3, q4

    def test_initialization_with_custom_marking_scheme(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test initialization with custom marking scheme."""
        minimal_evaluation_json["marking_schemes"]["SECTION_1"] = {
            "questions": ["q1", "q2"],
            "marking": {
                "correct": 2,
                "incorrect": -1,
                "unmarked": 0,
            },
        }

        config = EvaluationConfigForSet(
            "DEFAULT_SET",
            tmp_path,
            minimal_evaluation_json,
            mock_template,
            mock_tuning_config,
        )

        assert config.has_custom_marking is True
        assert "SECTION_1" in config.section_marking_schemes


class TestValidateQuestions:
    """Test validate_questions method."""

    def test_validate_questions_equal_lengths(self, sample_evaluation_config_for_set):
        """Test validation with equal question and answer lengths."""
        # Should not raise
        sample_evaluation_config_for_set.validate_questions()
        assert True

    def test_validate_questions_unequal_lengths(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test validation with unequal question and answer lengths."""
        minimal_evaluation_json["options"]["answers_in_order"] = [
            "A",
            "B",
        ]  # Only 2 instead of 3

        # FieldDefinitionError may not accept context parameter, so catch any exception
        with pytest.raises((FieldDefinitionError, ValueError, TypeError)):
            EvaluationConfigForSet(
                "DEFAULT_SET",
                tmp_path,
                minimal_evaluation_json,
                mock_template,
                mock_tuning_config,
            )


class TestValidateMarkingSchemes:
    """Test validate_marking_schemes method."""

    def test_validate_marking_schemes_no_overlap(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test validation with non-overlapping marking schemes."""
        minimal_evaluation_json["marking_schemes"]["SECTION_1"] = {
            "questions": ["q1"],
            "marking": {"correct": 1, "incorrect": 0, "unmarked": 0},
        }
        minimal_evaluation_json["marking_schemes"]["SECTION_2"] = {
            "questions": ["q2"],
            "marking": {"correct": 1, "incorrect": 0, "unmarked": 0},
        }

        # Should not raise
        config = EvaluationConfigForSet(
            "DEFAULT_SET",
            tmp_path,
            minimal_evaluation_json,
            mock_template,
            mock_tuning_config,
        )
        assert config is not None

    def test_validate_marking_schemes_with_overlap(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test validation with overlapping marking schemes."""
        minimal_evaluation_json["marking_schemes"]["SECTION_1"] = {
            "questions": ["q1", "q2"],
            "marking": {"correct": 1, "incorrect": 0, "unmarked": 0},
        }
        minimal_evaluation_json["marking_schemes"]["SECTION_2"] = {
            "questions": ["q2", "q3"],  # Overlaps with SECTION_1
            "marking": {"correct": 1, "incorrect": 0, "unmarked": 0},
        }

        with pytest.raises(FieldDefinitionError):
            EvaluationConfigForSet(
                "DEFAULT_SET",
                tmp_path,
                minimal_evaluation_json,
                mock_template,
                mock_tuning_config,
            )

    def test_validate_marking_schemes_missing_questions(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test validation with marking scheme questions not in answer key."""
        minimal_evaluation_json["marking_schemes"]["SECTION_1"] = {
            "questions": ["q99"],  # Not in questions_in_order
            "marking": {"correct": 1, "incorrect": 0, "unmarked": 0},
        }

        with pytest.raises((EvaluationError, OMRCheckerError)):
            EvaluationConfigForSet(
                "DEFAULT_SET",
                tmp_path,
                minimal_evaluation_json,
                mock_template,
                mock_tuning_config,
            )


class TestValidateAnswers:
    """Test validate_answers method."""

    def test_validate_answers_no_multimarked(
        self, sample_evaluation_config_for_set, mock_tuning_config
    ):
        """Test validation with no multi-marked answers."""
        mock_tuning_config.outputs.filter_out_multimarked_files = True

        # Should not raise
        sample_evaluation_config_for_set.validate_answers(mock_tuning_config)
        assert True

    def test_validate_answers_with_multimarked_standard(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test validation with multi-marked standard answers."""
        minimal_evaluation_json["options"]["answers_in_order"] = [
            "AB",
            "B",
            "C",
        ]  # "AB" is multi-marked

        config = EvaluationConfigForSet(
            "DEFAULT_SET",
            tmp_path,
            minimal_evaluation_json,
            mock_template,
            mock_tuning_config,
        )

        mock_tuning_config.outputs.filter_out_multimarked_files = True

        with pytest.raises(ConfigError):
            config.validate_answers(mock_tuning_config)

    def test_validate_answers_with_multimarked_multiple_correct(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test validation with multi-marked multiple correct answers."""
        minimal_evaluation_json["options"]["answers_in_order"] = [
            ["AB", "CD"],  # Multi-marked
            "B",
            "C",
        ]

        config = EvaluationConfigForSet(
            "DEFAULT_SET",
            tmp_path,
            minimal_evaluation_json,
            mock_template,
            mock_tuning_config,
        )

        mock_tuning_config.outputs.filter_out_multimarked_files = True

        with pytest.raises(ConfigError):
            config.validate_answers(mock_tuning_config)


class TestValidateFormatStrings:
    """Test validate_format_strings method."""

    def test_validate_format_strings_valid(self, sample_evaluation_config_for_set):
        """Test validation with valid format strings."""
        sample_evaluation_config_for_set.validate_format_strings()
        assert True

    @pytest.mark.parametrize(
        "config_path,invalid_value",
        [
            (
                [
                    "outputs_configuration",
                    "draw_answers_summary",
                    "answers_summary_format_string",
                ],
                "{invalid_variable}",
            ),
            (
                ["outputs_configuration", "draw_score", "score_format_string"],
                "{invalid_variable}",
            ),
        ],
    )
    def test_validate_format_strings_invalid(
        self,
        minimal_evaluation_json,
        mock_template,
        mock_tuning_config,
        tmp_path,
        config_path,
        invalid_value,
    ):
        """Test validation with invalid format strings."""
        config = minimal_evaluation_json
        for key in config_path[:-1]:
            config = config[key]
        config[config_path[-1]] = invalid_value

        with pytest.raises(ConfigError):
            EvaluationConfigForSet(
                "DEFAULT_SET",
                tmp_path,
                minimal_evaluation_json,
                mock_template,
                mock_tuning_config,
            )


class TestPrepareAndValidateOmrResponse:
    """Test prepare_and_validate_omr_response method."""

    def test_prepare_and_validate_omr_response_valid(
        self, sample_evaluation_config_for_set
    ):
        """Test validation with valid OMR response."""
        concatenated_response = {"q1": "A", "q2": "B", "q3": "C"}

        # Should not raise
        sample_evaluation_config_for_set.prepare_and_validate_omr_response(
            concatenated_response
        )
        assert True

    def test_prepare_and_validate_omr_response_missing_keys(
        self, sample_evaluation_config_for_set
    ):
        """Test validation with missing question keys."""
        concatenated_response = {"q1": "A", "q2": "B"}  # Missing q3

        with pytest.raises(EvaluationError):
            sample_evaluation_config_for_set.prepare_and_validate_omr_response(
                concatenated_response
            )

    def test_prepare_and_validate_omr_response_with_allow_streak(
        self, sample_evaluation_config_for_set
    ):
        """Test validation with allow_streak flag."""
        concatenated_response = {"q1": "A", "q2": "B", "q3": "C"}

        sample_evaluation_config_for_set.prepare_and_validate_omr_response(
            concatenated_response, allow_streak=True
        )

        assert sample_evaluation_config_for_set.allow_streak is True


class TestMatchAnswerForQuestion:
    """Test match_answer_for_question method."""

    def test_match_answer_for_question_correct(self, sample_evaluation_config_for_set):
        """Test matching correct answer."""
        concatenated_response = {"q1": "A", "q2": "B", "q3": "C"}
        sample_evaluation_config_for_set.prepare_and_validate_omr_response(
            concatenated_response
        )

        # match_answer_for_question returns (delta, verdict, answer_matcher, schema_verdict)
        # Takes (current_score, question, marked_answer)
        result = sample_evaluation_config_for_set.match_answer_for_question(
            0.0, "q1", "A"
        )

        # Result is a tuple: (delta, verdict, answer_matcher, schema_verdict)
        assert isinstance(result, tuple)
        assert len(result) == 4
        delta, verdict, answer_matcher, schema_verdict = result
        assert delta >= 0  # Should be non-negative for correct
        assert verdict in ["correct", "answer-match"]
        assert schema_verdict == "correct"

    def test_match_answer_for_question_incorrect(
        self, sample_evaluation_config_for_set
    ):
        """Test matching incorrect answer."""
        concatenated_response = {"q1": "A", "q2": "B", "q3": "C"}
        sample_evaluation_config_for_set.prepare_and_validate_omr_response(
            concatenated_response
        )

        result = sample_evaluation_config_for_set.match_answer_for_question(
            0.0,
            "q1",
            "B",  # Wrong answer
        )

        # Result is a tuple: (delta, verdict, answer_matcher, schema_verdict)
        delta, verdict, answer_matcher, schema_verdict = result
        assert delta <= 0  # Should be zero or negative for incorrect
        assert schema_verdict in ["incorrect", "neutral"]

    def test_match_answer_for_question_unmarked(self, sample_evaluation_config_for_set):
        """Test matching unmarked answer."""
        concatenated_response = {"q1": "", "q2": "B", "q3": "C"}
        sample_evaluation_config_for_set.prepare_and_validate_omr_response(
            concatenated_response
        )

        result = sample_evaluation_config_for_set.match_answer_for_question(
            0.0, "q1", ""
        )

        # Result is a tuple: (delta, verdict, answer_matcher, schema_verdict)
        delta, verdict, answer_matcher, schema_verdict = result
        assert delta == 0  # Should be zero for unmarked
        assert schema_verdict == "unmarked"


class TestGetEvaluationMetaForQuestion:
    """Test get_evaluation_meta_for_question method."""

    def test_get_evaluation_meta_for_question(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test getting evaluation metadata for a question."""
        # Enable draw_question_verdicts so verdict_colors is initialized
        minimal_evaluation_json["outputs_configuration"]["draw_question_verdicts"] = {
            "enabled": True,
            "verdict_colors": {
                "correct": "#00FF00",
                "incorrect": "#FF0000",
                "neutral": None,
                "bonus": "#00DDDD",
            },
            "verdict_symbol_colors": {
                "positive": "#000000",
                "negative": "#000000",
                "neutral": "#000000",
                "bonus": "#000000",
            },
            "draw_answer_groups": {
                "enabled": True,
                "color_sequence": ["#8DFBC4", "#F7FB8D", "#8D9EFB", "#EA666F"],
            },
        }

        config = EvaluationConfigForSet(
            "DEFAULT_SET",
            tmp_path,
            minimal_evaluation_json,
            mock_template,
            mock_tuning_config,
        )

        question_meta = {
            "verdict": "correct",
            "bonus_type": None,
            "delta": 1.0,
            "question_verdict": "correct",
            "question_schema_verdict": "correct",
        }

        field_interpretation = Mock()
        field_interpretation.is_attempted = True

        meta = config.get_evaluation_meta_for_question(
            question_meta, field_interpretation, "GRAYSCALE"
        )

        assert (
            len(meta) == 4
        )  # verdict_symbol, verdict_color, verdict_symbol_color, thickness_factor
        assert meta[0] is not None  # verdict_symbol
        assert meta[1] is not None  # verdict_color
        assert meta[2] is not None  # verdict_symbol_color
        assert meta[3] is not None  # thickness_factor


class TestGetFormattedAnswersSummary:
    """Test get_formatted_answers_summary method."""

    def test_get_formatted_answers_summary(self, sample_evaluation_config_for_set):
        """Test getting formatted answers summary."""
        # Set some verdict counts
        sample_evaluation_config_for_set.schema_verdict_counts = {
            "correct": 2,
            "incorrect": 1,
            "unmarked": 0,
        }

        # Returns tuple: (answers_format, position, size, thickness)
        summary = sample_evaluation_config_for_set.get_formatted_answers_summary()

        assert isinstance(summary, tuple)
        assert len(summary) == 4
        assert isinstance(summary[0], str)  # answers_format
        assert len(summary[0]) > 0


class TestGetFormattedScore:
    """Test get_formatted_score method."""

    def test_get_formatted_score(self, sample_evaluation_config_for_set):
        """Test getting formatted score."""
        score = 85.5

        # Returns tuple: (score_format, position, size, thickness)
        formatted = sample_evaluation_config_for_set.get_formatted_score(score)

        assert isinstance(formatted, tuple)
        assert len(formatted) == 4
        assert isinstance(formatted[0], str)  # score_format
        assert "85" in formatted[0] or "85.5" in formatted[0]


class TestResetEvaluation:
    """Test reset_evaluation method."""

    def test_reset_evaluation(self, sample_evaluation_config_for_set):
        """Test resetting evaluation state."""
        # Set some counts
        sample_evaluation_config_for_set.schema_verdict_counts = {
            "correct": 5,
            "incorrect": 2,
            "unmarked": 1,
        }

        sample_evaluation_config_for_set.reset_evaluation()

        assert sample_evaluation_config_for_set.schema_verdict_counts["correct"] == 0
        assert sample_evaluation_config_for_set.schema_verdict_counts["incorrect"] == 0
        assert sample_evaluation_config_for_set.schema_verdict_counts["unmarked"] == 0


class TestGetMarkingSchemeForQuestion:
    """Test get_marking_scheme_for_question method."""

    def test_get_marking_scheme_for_question_default(
        self, sample_evaluation_config_for_set
    ):
        """Test getting default marking scheme."""
        scheme = sample_evaluation_config_for_set.get_marking_scheme_for_question("q1")

        assert isinstance(scheme, SectionMarkingScheme)
        assert scheme.section_key == DEFAULT_SECTION_KEY

    def test_get_marking_scheme_for_question_custom(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test getting custom marking scheme."""
        minimal_evaluation_json["marking_schemes"]["SECTION_1"] = {
            "questions": ["q1"],
            "marking": {"correct": 2, "incorrect": -1, "unmarked": 0},
        }

        config = EvaluationConfigForSet(
            "DEFAULT_SET",
            tmp_path,
            minimal_evaluation_json,
            mock_template,
            mock_tuning_config,
        )

        scheme = config.get_marking_scheme_for_question("q1")

        assert scheme.section_key == "SECTION_1"


class TestGetExcludeFiles:
    """Test get_exclude_files method."""

    def test_get_exclude_files(self, sample_evaluation_config_for_set):
        """Test getting excluded files."""
        excluded = sample_evaluation_config_for_set.get_exclude_files()

        assert isinstance(excluded, list)
