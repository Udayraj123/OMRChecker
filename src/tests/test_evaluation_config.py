"""Tests for EvaluationConfig class."""

import pytest

from src.exceptions import ConfigError
from src.processors.evaluation.evaluation_config import EvaluationConfig
from src.schemas.constants import DEFAULT_SECTION_KEY


@pytest.fixture
def mock_template():
    """Create a mock template."""
    template = type("Template", (), {})()
    template.global_empty_val = ""
    return template


@pytest.fixture
def mock_tuning_config():
    """Create a mock tuning config."""
    config = type("Config", (), {})()
    config.outputs = type("Outputs", (), {})()
    config.outputs.filter_out_multimarked_files = False
    return config


@pytest.fixture
def minimal_evaluation_json():
    """Minimal valid evaluation JSON."""
    return {
        "source_type": "local",
        "options": {
            "questions_in_order": ["q1", "q2"],
            "answers_in_order": ["A", "B"],
        },
        "marking_schemes": {
            DEFAULT_SECTION_KEY: {
                "correct": 1,
                "incorrect": 0,
                "unmarked": 0,
            }
        },
        "outputs_configuration": {},
    }


class TestEvaluationConfigValidation:
    """Test EvaluationConfig validation."""

    def test_reject_conditional_set_with_answers_but_no_questions(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test that conditional set with answers_in_order but no questions_in_order is rejected."""
        evaluation_json = minimal_evaluation_json.copy()
        evaluation_json["conditional_sets"] = [
            {
                "name": "Set A",
                "matcher": {
                    "formatString": "{set_type}",
                    "matchRegex": "^A$",
                },
                "evaluation": {
                    "options": {
                        "answers_in_order": ["B", "C"],  # Missing questions_in_order
                    },
                },
            },
        ]

        with pytest.raises(ConfigError) as exc_info:
            EvaluationConfig(
                tmp_path,
                tmp_path / "evaluation.json",
                evaluation_json,
                mock_template,
                mock_tuning_config,
            )

        assert "answers_in_order" in str(exc_info.value)
        assert "questions_in_order" in str(exc_info.value)

    def test_reject_conditional_set_with_questions_but_no_answers(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test that conditional set with questions_in_order but no answers_in_order is rejected."""
        evaluation_json = minimal_evaluation_json.copy()
        evaluation_json["conditional_sets"] = [
            {
                "name": "Set A",
                "matcher": {
                    "formatString": "{set_type}",
                    "matchRegex": "^A$",
                },
                "evaluation": {
                    "options": {
                        "questions_in_order": ["q1", "q2"],  # Missing answers_in_order
                    },
                },
            },
        ]

        with pytest.raises(ConfigError) as exc_info:
            EvaluationConfig(
                tmp_path,
                tmp_path / "evaluation.json",
                evaluation_json,
                mock_template,
                mock_tuning_config,
            )

        assert "questions_in_order" in str(exc_info.value)
        assert "answers_in_order" in str(exc_info.value)

    def test_accept_conditional_set_with_both_questions_and_answers(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test that conditional set with both questions_in_order and answers_in_order is accepted."""
        evaluation_json = minimal_evaluation_json.copy()
        evaluation_json["conditional_sets"] = [
            {
                "name": "Set A",
                "matcher": {
                    "formatString": "{set_type}",
                    "matchRegex": "^A$",
                },
                "evaluation": {
                    "options": {
                        "questions_in_order": ["q1", "q2"],
                        "answers_in_order": ["B", "C"],
                    },
                },
            },
        ]

        # Should not raise
        config = EvaluationConfig(
            tmp_path,
            tmp_path / "evaluation.json",
            evaluation_json,
            mock_template,
            mock_tuning_config,
        )

        assert config is not None
        assert "Set A" in config.set_mapping
