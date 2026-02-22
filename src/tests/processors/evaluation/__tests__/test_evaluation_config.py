"""Tests for EvaluationConfig class."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.utils.exceptions import EvaluationValidationError
from src.processors.evaluation.evaluation_config import EvaluationConfig
from src.schemas.constants import DEFAULT_SECTION_KEY


@pytest.fixture
def mock_template():
    """Create a mock template."""
    template = Mock()
    template.global_empty_val = ""
    template.get_evaluations_dir = Mock(return_value=Path("/tmp/evaluations"))
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
        "sourceType": "local",
        "options": {
            "questionsInOrder": ["q1", "q2"],
            "answersInOrder": ["A", "B"],
        },
        "markingSchemes": {
            DEFAULT_SECTION_KEY: {
                "correct": 1,
                "incorrect": 0,
                "unmarked": 0,
            }
        },
        "outputsConfiguration": {},
    }


class TestEvaluationConfigValidation:
    """Test EvaluationConfig validation."""

    def test_reject_conditional_set_with_answers_but_no_questions(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test that conditional set with answersInOrder but no questionsInOrder is rejected."""
        evaluation_json = minimal_evaluation_json.copy()
        evaluation_json["conditionalSets"] = [
            {
                "name": "Set A",
                "matcher": {
                    "formatString": "{set_type}",
                    "matchRegex": "^A$",
                },
                "evaluation": {
                    "sourceType": "local",
                    "options": {
                        "answersInOrder": ["B", "C"],  # Missing questionsInOrder
                    },
                    "markingSchemes": {
                        DEFAULT_SECTION_KEY: {
                            "correct": 1,
                            "incorrect": 0,
                            "unmarked": 0,
                        }
                    },
                    "outputsConfiguration": {},
                },
            },
        ]

        # Write evaluation JSON to file
        eval_file = tmp_path / "evaluation.json"
        with open(eval_file, "w") as f:
            json.dump(evaluation_json, f)

        # Schema validation happens first and will catch missing questionsInOrder
        with pytest.raises(EvaluationValidationError) as exc_info:
            EvaluationConfig(
                tmp_path,
                eval_file,
                mock_template,
                mock_tuning_config,
            )

        assert "questionsInOrder" in str(exc_info.value)

    def test_reject_conditional_set_with_questions_but_no_answers(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test that conditional set with questionsInOrder but no answersInOrder is rejected."""
        evaluation_json = minimal_evaluation_json.copy()
        evaluation_json["conditionalSets"] = [
            {
                "name": "Set A",
                "matcher": {
                    "formatString": "{set_type}",
                    "matchRegex": "^A$",
                },
                "evaluation": {
                    "sourceType": "local",
                    "options": {
                        "questionsInOrder": ["q1", "q2"],  # Missing answersInOrder
                    },
                    "markingSchemes": {
                        DEFAULT_SECTION_KEY: {
                            "correct": 1,
                            "incorrect": 0,
                            "unmarked": 0,
                        }
                    },
                    "outputsConfiguration": {},
                },
            },
        ]

        # Write evaluation JSON to file
        eval_file = tmp_path / "evaluation.json"
        with open(eval_file, "w") as f:
            json.dump(evaluation_json, f)

        # Schema validation happens first and will catch missing answersInOrder
        with pytest.raises(EvaluationValidationError) as exc_info:
            EvaluationConfig(
                tmp_path,
                eval_file,
                mock_template,
                mock_tuning_config,
            )

        assert "answersInOrder" in str(exc_info.value)

    def test_accept_conditional_set_with_both_questions_and_answers(
        self, minimal_evaluation_json, mock_template, mock_tuning_config, tmp_path
    ):
        """Test that conditional set with both questionsInOrder and answersInOrder is accepted."""
        evaluation_json = minimal_evaluation_json.copy()
        evaluation_json["conditionalSets"] = [
            {
                "name": "Set A",
                "matcher": {
                    "formatString": "{set_type}",
                    "matchRegex": "^A$",
                },
                "evaluation": {
                    "sourceType": "local",
                    "options": {
                        "questionsInOrder": ["q1", "q2"],
                        "answersInOrder": ["B", "C"],
                    },
                    "markingSchemes": {
                        DEFAULT_SECTION_KEY: {
                            "correct": 1,
                            "incorrect": 0,
                            "unmarked": 0,
                        }
                    },
                    "outputsConfiguration": {},
                },
            },
        ]

        # Write evaluation JSON to file
        eval_file = tmp_path / "evaluation.json"
        with open(eval_file, "w") as f:
            json.dump(evaluation_json, f)

        # Should not raise
        config = EvaluationConfig(
            tmp_path,
            eval_file,
            mock_template,
            mock_tuning_config,
        )

        assert config is not None
        assert "Set A" in config.set_mapping
