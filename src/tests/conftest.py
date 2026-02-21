"""Shared pytest fixtures for all tests.

This module provides common fixtures used across multiple test files to reduce
code duplication and improve maintainability.
"""

from unittest.mock import Mock

import pytest

from src.schemas.models.config import Config


@pytest.fixture
def mock_template():
    """Create a mock template object."""
    template = Mock()
    template.tuning_config = Config()
    template.all_fields = []
    template.all_field_detection_types = []
    return template


@pytest.fixture
def mock_tuning_config():
    """Create a mock tuning config."""
    return Config()


@pytest.fixture
def minimal_template_json():
    """Minimal valid template JSON."""
    return {
        "templateDimensions": [1000, 800],
        "bubbleDimensions": [20, 20],
        "emptyValue": "",
        "fieldBlocksOffset": [0, 0],
        "fieldBlocks": {
            "block1": {
                "fieldDetectionType": "BUBBLES_THRESHOLD",
                "origin": [100, 100],
                "fieldLabels": ["q1", "q2"],
                "bubbleFieldType": "QTYPE_MCQ4",
                "bubblesGap": 30,
                "labelsGap": 50,
            }
        },
        "preProcessors": [],
        "alignment": {"margins": {"top": 0, "right": 0, "bottom": 0, "left": 0}},
        "customBubbleFieldTypes": {},
        "customLabels": {},
        "outputColumns": {"sortType": "ALPHANUMERIC", "customOrder": []},
    }


@pytest.fixture
def minimal_evaluation_json():
    """Minimal valid evaluation JSON."""
    return {
        "source_type": "local",
        "options": {
            "questions_in_order": ["q1", "q2", "q3"],
            "answers_in_order": ["A", "B", "C"],
        },
        "marking_schemes": {
            "DEFAULT": {
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


@pytest.fixture
def minimal_args():
    """Minimal args dictionary for Template initialization."""
    return {
        "debug": False,
        "outputMode": "default",
        "setLayout": False,
        "mode": "process",
        "input_paths": [],
        "output_dir": "./outputs",
        "collect_training_data": False,
        "confidence_threshold": 0.8,
    }


@pytest.fixture
def temp_template_path(tmp_path):
    """Create a temporary template JSON file path."""
    template_path = tmp_path / "template.json"
    return template_path
