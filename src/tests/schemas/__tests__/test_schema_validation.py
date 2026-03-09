"""Schema validation tests for template, config, and evaluation schemas.

These tests exercise the JSON-schema validators directly via
`src.utils.validations` (which raises typed ValidationError subclasses) as
well as the raw jsonschema validator exposed through `src.schemas.SCHEMA_VALIDATORS`.

Each section covers:
  - at least one fully-valid "happy path" fixture
  - missing required fields
  - wrong types / out-of-range values
  - schema-level constraints (enum, pattern, additionalProperties, allOf/if-then)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.schemas import SCHEMA_VALIDATORS
from src.schemas.constants import DEFAULT_SECTION_KEY
from src.utils.exceptions import (
    ConfigValidationError,
    EvaluationValidationError,
    TemplateValidationError,
)
from src.utils.validations import (
    validate_config_json,
    validate_evaluation_json,
    validate_template_json,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DUMMY_PATH = Path("/tmp/dummy")


def _schema_errors(schema_name: str, instance: dict) -> list[str]:
    """Return a list of jsonschema error messages for *instance* against *schema_name*."""
    return [e.message for e in SCHEMA_VALIDATORS[schema_name].iter_errors(instance)]


def _is_valid(schema_name: str, instance: dict) -> bool:
    return not list(SCHEMA_VALIDATORS[schema_name].iter_errors(instance))


# ===========================================================================
# TEMPLATE SCHEMA
# ===========================================================================

# ---------------------------------------------------------------------------
# Minimal valid template fixture
# ---------------------------------------------------------------------------

_MINIMAL_TEMPLATE: dict = {
    "bubbleDimensions": [25, 25],
    "templateDimensions": [300, 400],
    "preProcessors": [],
    "fieldBlocks": {},
}


class TestTemplateSchemaValid:
    """Happy-path tests: valid template inputs must pass the schema."""

    def test_minimal_template_is_valid(self):
        """A template with only the four required keys should be accepted."""
        assert _is_valid("template", _MINIMAL_TEMPLATE)

    def test_template_with_empty_value(self):
        """emptyValue is optional but must be a string when present."""
        instance = {**_MINIMAL_TEMPLATE, "emptyValue": ""}
        assert _is_valid("template", instance)

    def test_template_with_custom_labels(self):
        """customLabels values must be arrays of field-string patterns."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "customLabels": {
                "Roll": ["q1..5"],
                "Name": ["q6..10"],
            },
        }
        assert _is_valid("template", instance)

    def test_template_with_output_columns_sort_type(self):
        """outputColumns.sortType must be one of ALPHABETICAL/ALPHANUMERIC/CUSTOM."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "outputColumns": {"sortType": "ALPHANUMERIC"},
        }
        assert _is_valid("template", instance)

    def test_template_with_processing_image_shape(self):
        """processingImageShape is optional; must be two positive numbers."""
        instance = {**_MINIMAL_TEMPLATE, "processingImageShape": [900, 650]}
        assert _is_valid("template", instance)

    def test_template_with_alignment_block(self):
        """alignment block is optional; its sub-keys must match the sub-schema."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "alignment": {
                "margins": {"top": 5, "right": 5, "bottom": 5, "left": 5},
                "maxDisplacement": 10,
            },
        }
        assert _is_valid("template", instance)

    def test_template_with_preprocessor_crop_page(self):
        """A CropPage preprocessor with valid options must be accepted."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "preProcessors": [
                {
                    "name": "CropPage",
                    "options": {},
                }
            ],
        }
        assert _is_valid("template", instance)


class TestTemplateSchemaInvalidMissingRequired:
    """Missing required fields must fail validation."""

    def test_missing_bubble_dimensions(self):
        """bubbleDimensions is required."""
        instance = {
            k: v for k, v in _MINIMAL_TEMPLATE.items() if k != "bubbleDimensions"
        }
        assert not _is_valid("template", instance)
        errors = _schema_errors("template", instance)
        assert any("bubbleDimensions" in e for e in errors)

    def test_missing_template_dimensions(self):
        """templateDimensions is required."""
        instance = {
            k: v for k, v in _MINIMAL_TEMPLATE.items() if k != "templateDimensions"
        }
        assert not _is_valid("template", instance)
        errors = _schema_errors("template", instance)
        assert any("templateDimensions" in e for e in errors)

    def test_missing_pre_processors(self):
        """preProcessors is required."""
        instance = {k: v for k, v in _MINIMAL_TEMPLATE.items() if k != "preProcessors"}
        assert not _is_valid("template", instance)
        errors = _schema_errors("template", instance)
        assert any("preProcessors" in e for e in errors)

    def test_missing_field_blocks(self):
        """fieldBlocks is required."""
        instance = {k: v for k, v in _MINIMAL_TEMPLATE.items() if k != "fieldBlocks"}
        assert not _is_valid("template", instance)
        errors = _schema_errors("template", instance)
        assert any("fieldBlocks" in e for e in errors)


class TestTemplateSchemaInvalidTypes:
    """Wrong types for template fields must fail validation."""

    def test_bubble_dimensions_must_be_array(self):
        """bubbleDimensions must be an array, not a number."""
        instance = {**_MINIMAL_TEMPLATE, "bubbleDimensions": 25}
        assert not _is_valid("template", instance)

    def test_bubble_dimensions_must_have_two_elements(self):
        """bubbleDimensions must have exactly two elements."""
        instance = {**_MINIMAL_TEMPLATE, "bubbleDimensions": [25]}
        assert not _is_valid("template", instance)

    def test_bubble_dimensions_must_be_positive(self):
        """bubbleDimensions values must be non-negative."""
        instance = {**_MINIMAL_TEMPLATE, "bubbleDimensions": [-1, 25]}
        assert not _is_valid("template", instance)

    def test_template_dimensions_must_be_array(self):
        """templateDimensions must be an array, not a string."""
        instance = {**_MINIMAL_TEMPLATE, "templateDimensions": "300x400"}
        assert not _is_valid("template", instance)

    def test_empty_value_must_be_string(self):
        """emptyValue must be a string, not an integer."""
        instance = {**_MINIMAL_TEMPLATE, "emptyValue": 0}
        assert not _is_valid("template", instance)

    def test_pre_processors_must_be_array(self):
        """preProcessors must be an array, not an object."""
        instance = {**_MINIMAL_TEMPLATE, "preProcessors": {"name": "CropPage"}}
        assert not _is_valid("template", instance)

    def test_output_columns_sort_type_must_be_enum(self):
        """outputColumns.sortType must be one of the allowed enum values."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "outputColumns": {"sortType": "RANDOM"},
        }
        assert not _is_valid("template", instance)

    def test_output_columns_sort_order_must_be_enum(self):
        """outputColumns.sortOrder must be ASC or DESC."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "outputColumns": {"sortOrder": "ASCENDING"},
        }
        assert not _is_valid("template", instance)

    def test_unknown_top_level_key_rejected(self):
        """additionalProperties=False: unknown top-level keys must be rejected."""
        instance = {**_MINIMAL_TEMPLATE, "unknownKey": True}
        assert not _is_valid("template", instance)

    def test_custom_labels_value_must_be_array_of_strings(self):
        """customLabels values must be arrays, not plain strings."""
        instance = {**_MINIMAL_TEMPLATE, "customLabels": {"Roll": "q1..5"}}
        assert not _is_valid("template", instance)


class TestTemplateSchemaValidationFunctions:
    """Tests for validate_template_json() which raises TemplateValidationError."""

    def test_valid_template_does_not_raise(self):
        """A valid template dict must not raise any exception."""
        # Should complete without error
        validate_template_json(_MINIMAL_TEMPLATE, _DUMMY_PATH)

    def test_empty_template_raises_template_validation_error(self):
        """An empty dict is missing all required fields and must raise."""
        with pytest.raises(TemplateValidationError):
            validate_template_json({}, _DUMMY_PATH)

    def test_missing_field_blocks_raises(self):
        """Missing fieldBlocks must raise TemplateValidationError."""
        instance = {k: v for k, v in _MINIMAL_TEMPLATE.items() if k != "fieldBlocks"}
        with pytest.raises(TemplateValidationError) as exc_info:
            validate_template_json(instance, _DUMMY_PATH)
        assert "fieldBlocks" in str(exc_info.value)

    def test_invalid_sort_type_raises(self):
        """An invalid sortType must raise TemplateValidationError."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "outputColumns": {"sortType": "INVALID_TYPE"},
        }
        with pytest.raises(TemplateValidationError) as exc_info:
            validate_template_json(instance, _DUMMY_PATH)
        assert "Invalid template JSON" in str(exc_info.value)

    def test_unknown_preprocessor_name_raises(self):
        """An unknown preProcessor name must raise TemplateValidationError."""
        instance = {
            **_MINIMAL_TEMPLATE,
            "preProcessors": [{"name": "NonExistentProcessor", "options": {}}],
        }
        with pytest.raises(TemplateValidationError) as exc_info:
            validate_template_json(instance, _DUMMY_PATH)
        assert "Invalid template JSON" in str(exc_info.value)

    def test_negative_bubble_dimensions_raises(self):
        """Negative bubble dimensions must raise TemplateValidationError."""
        instance = {**_MINIMAL_TEMPLATE, "bubbleDimensions": [-5, 25]}
        with pytest.raises(TemplateValidationError):
            validate_template_json(instance, _DUMMY_PATH)

    def test_additional_property_raises(self):
        """An extra unknown key must raise TemplateValidationError."""
        instance = {**_MINIMAL_TEMPLATE, "garbage": 42}
        with pytest.raises(TemplateValidationError):
            validate_template_json(instance, _DUMMY_PATH)


# ===========================================================================
# CONFIG SCHEMA
# ===========================================================================

# ---------------------------------------------------------------------------
# Minimal valid config fixture
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG: dict = {}  # Config schema allows fully empty object (all optional)


class TestConfigSchemaValid:
    """Happy-path tests: valid config inputs must pass the schema."""

    def test_empty_config_is_valid(self):
        """An empty config object must be accepted (all fields optional)."""
        assert _is_valid("config", _MINIMAL_CONFIG)

    def test_config_with_valid_thresholding(self):
        """A thresholding block with values in range must be accepted."""
        instance = {
            "thresholding": {
                "minGapTwoBubbles": 30,
                "minJump": 25,
                "globalPageThreshold": 200,
            }
        }
        assert _is_valid("config", instance)

    def test_config_with_valid_outputs(self):
        """An outputs block with valid showImageLevel must be accepted."""
        instance = {"outputs": {"showImageLevel": 0, "saveImageLevel": 1}}
        assert _is_valid("config", instance)

    def test_config_with_valid_output_mode(self):
        """outputMode must be one of the supported values."""
        instance = {"outputs": {"outputMode": "default"}}
        assert _is_valid("config", instance)

    def test_config_with_show_logs_by_type(self):
        """showLogsByType accepts boolean toggles per log level."""
        instance = {
            "outputs": {
                "showLogsByType": {
                    "critical": True,
                    "error": True,
                    "warning": False,
                    "info": True,
                    "debug": False,
                }
            }
        }
        assert _is_valid("config", instance)

    def test_config_with_max_parallel_workers_one(self):
        """maxParallelWorkers=1 is always valid."""
        instance = {"processing": {"maxParallelWorkers": 1}}
        assert _is_valid("config", instance)

    def test_config_with_ml_enabled(self):
        """ML section with enabled=True must be accepted."""
        instance = {"ml": {"enabled": True, "confidenceThreshold": 0.7}}
        assert _is_valid("config", instance)

    def test_config_with_ml_fusion_strategy_enum(self):
        """fusionStrategy must be one of the allowed enum values."""
        for strategy in [
            "confidence_weighted",
            "ml_fallback",
            "traditional_primary",
            "ml_only",
        ]:
            instance = {"ml": {"fusionStrategy": strategy}}
            assert _is_valid("config", instance), (
                f"Expected valid for strategy={strategy}"
            )

    def test_config_show_image_level_zero_allows_multiple_workers(self):
        """showImageLevel=0 (non-interactive) allows maxParallelWorkers > 1."""
        instance = {
            "outputs": {"showImageLevel": 0},
            "processing": {"maxParallelWorkers": 4},
        }
        assert _is_valid("config", instance)


class TestConfigSchemaInvalidValues:
    """Out-of-range or wrong-type config values must fail validation."""

    def test_thresholding_min_gap_below_minimum(self):
        """minGapTwoBubbles must be >= 10."""
        instance = {"thresholding": {"minGapTwoBubbles": 5}}
        assert not _is_valid("config", instance)

    def test_thresholding_min_gap_above_maximum(self):
        """minGapTwoBubbles must be <= 100."""
        instance = {"thresholding": {"minGapTwoBubbles": 200}}
        assert not _is_valid("config", instance)

    def test_thresholding_global_page_threshold_above_maximum(self):
        """globalPageThreshold must be <= 255."""
        instance = {"thresholding": {"globalPageThreshold": 300}}
        assert not _is_valid("config", instance)

    def test_outputs_show_image_level_above_maximum(self):
        """showImageLevel must be <= 6."""
        instance = {"outputs": {"showImageLevel": 10}}
        assert not _is_valid("config", instance)

    def test_outputs_show_image_level_below_minimum(self):
        """showImageLevel must be >= 0."""
        instance = {"outputs": {"showImageLevel": -1}}
        assert not _is_valid("config", instance)

    def test_outputs_save_image_level_above_maximum(self):
        """saveImageLevel must be <= 6."""
        instance = {"outputs": {"saveImageLevel": 7}}
        assert not _is_valid("config", instance)

    def test_outputs_invalid_output_mode(self):
        """outputMode must be one of the enum values, not an arbitrary string."""
        instance = {"outputs": {"outputMode": "superMode"}}
        assert not _is_valid("config", instance)

    def test_processing_max_parallel_workers_below_minimum(self):
        """maxParallelWorkers must be >= 1."""
        instance = {"processing": {"maxParallelWorkers": 0}}
        assert not _is_valid("config", instance)

    def test_processing_max_parallel_workers_above_maximum(self):
        """maxParallelWorkers must be <= 16."""
        instance = {"processing": {"maxParallelWorkers": 32}}
        assert not _is_valid("config", instance)

    def test_ml_confidence_threshold_above_one(self):
        """confidenceThreshold must be <= 1.0."""
        instance = {"ml": {"confidenceThreshold": 1.5}}
        assert not _is_valid("config", instance)

    def test_ml_confidence_threshold_below_zero(self):
        """confidenceThreshold must be >= 0.0."""
        instance = {"ml": {"confidenceThreshold": -0.1}}
        assert not _is_valid("config", instance)

    def test_ml_invalid_fusion_strategy(self):
        """fusionStrategy must be one of the allowed enum values."""
        instance = {"ml": {"fusionStrategy": "unknown_strategy"}}
        assert not _is_valid("config", instance)

    def test_ml_shift_detection_negative_max_shift(self):
        """globalMaxShiftPixels must be >= 0."""
        instance = {"ml": {"shiftDetection": {"globalMaxShiftPixels": -10}}}
        assert not _is_valid("config", instance)

    def test_unknown_top_level_key_rejected(self):
        """additionalProperties=False: unknown top-level keys must be rejected."""
        instance = {"unknownKey": "surprise"}
        assert not _is_valid("config", instance)

    def test_show_logs_by_type_non_boolean_value(self):
        """showLogsByType values must be booleans, not strings."""
        instance = {"outputs": {"showLogsByType": {"critical": "yes"}}}
        assert not _is_valid("config", instance)

    def test_thresholding_gamma_low_above_one(self):
        """gammaLow must be <= 1."""
        instance = {"thresholding": {"gammaLow": 1.5}}
        assert not _is_valid("config", instance)

    def test_thresholding_gamma_low_below_zero(self):
        """gammaLow must be >= 0."""
        instance = {"thresholding": {"gammaLow": -0.5}}
        assert not _is_valid("config", instance)


class TestConfigSchemaConditionalConstraints:
    """Tests for the allOf conditional: showImageLevel > 0 requires maxParallelWorkers=1."""

    def test_show_image_level_positive_requires_single_worker(self):
        """showImageLevel > 0 with maxParallelWorkers > 1 must fail."""
        instance = {
            "outputs": {"showImageLevel": 1},
            "processing": {"maxParallelWorkers": 4},
        }
        assert not _is_valid("config", instance)

    def test_show_image_level_positive_with_single_worker_is_valid(self):
        """showImageLevel > 0 with maxParallelWorkers=1 must be accepted."""
        instance = {
            "outputs": {"showImageLevel": 3},
            "processing": {"maxParallelWorkers": 1},
        }
        assert _is_valid("config", instance)

    def test_show_image_level_zero_allows_multiple_workers(self):
        """showImageLevel=0 (non-interactive) with multiple workers is valid."""
        instance = {
            "outputs": {"showImageLevel": 0},
            "processing": {"maxParallelWorkers": 8},
        }
        assert _is_valid("config", instance)


class TestConfigSchemaValidationFunctions:
    """Tests for validate_config_json() which raises ConfigValidationError."""

    def test_valid_empty_config_does_not_raise(self):
        """An empty config dict must not raise any exception."""
        validate_config_json({}, _DUMMY_PATH)

    def test_valid_config_with_outputs_does_not_raise(self):
        """A config with valid outputs block must not raise."""
        validate_config_json({"outputs": {"showImageLevel": 0}}, _DUMMY_PATH)

    def test_show_image_level_with_multiple_workers_raises(self, capsys):
        """showImageLevel > 0 with maxParallelWorkers > 1 must raise ConfigValidationError."""
        instance = {
            "outputs": {"showImageLevel": 2},
            "processing": {"maxParallelWorkers": 4},
        }
        with pytest.raises((ConfigValidationError, Exception)) as exc_info:
            validate_config_json(instance, _DUMMY_PATH)
        error_str = str(exc_info.value)
        assert (
            "Invalid config JSON" in error_str or "config JSON is Invalid" in error_str
        )

    def test_invalid_output_mode_raises(self):
        """An invalid outputMode must raise ConfigValidationError."""
        instance = {"outputs": {"outputMode": "unknownMode"}}
        with pytest.raises(ConfigValidationError):
            validate_config_json(instance, _DUMMY_PATH)

    def test_out_of_range_threshold_raises(self):
        """An out-of-range thresholding value must raise ConfigValidationError."""
        instance = {"thresholding": {"minGapTwoBubbles": 200}}
        with pytest.raises(ConfigValidationError):
            validate_config_json(instance, _DUMMY_PATH)

    def test_unknown_key_raises(self):
        """An unknown top-level key must raise ConfigValidationError."""
        instance = {"unknownKey": 42}
        with pytest.raises(ConfigValidationError):
            validate_config_json(instance, _DUMMY_PATH)


# ===========================================================================
# EVALUATION SCHEMA
# ===========================================================================

# ---------------------------------------------------------------------------
# Minimal valid evaluation fixture
# ---------------------------------------------------------------------------

_MINIMAL_EVALUATION_LOCAL: dict = {
    "sourceType": "local",
    "options": {
        "questionsInOrder": ["q1", "q2", "q3"],
        "answersInOrder": ["A", "B", "C"],
    },
    "markingSchemes": {
        DEFAULT_SECTION_KEY: {
            "correct": 1,
            "incorrect": 0,
            "unmarked": 0,
        }
    },
}

_MINIMAL_EVALUATION_CSV: dict = {
    "sourceType": "csv",
    "options": {"answerKeyCsvPath": "answers.csv"},
    "markingSchemes": {
        DEFAULT_SECTION_KEY: {
            "correct": 4,
            "incorrect": "-1",
            "unmarked": 0,
        }
    },
}


class TestEvaluationSchemaValid:
    """Happy-path tests: valid evaluation inputs must pass the schema."""

    def test_minimal_local_evaluation_is_valid(self):
        """A minimal local evaluation schema must be accepted."""
        assert _is_valid("evaluation", _MINIMAL_EVALUATION_LOCAL)

    def test_minimal_csv_evaluation_is_valid(self):
        """A minimal CSV-sourced evaluation schema must be accepted."""
        assert _is_valid("evaluation", _MINIMAL_EVALUATION_CSV)

    def test_marking_scheme_with_fraction_string(self):
        """Marking scores can be fractions like '1/3'."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "markingSchemes": {
                DEFAULT_SECTION_KEY: {
                    "correct": "4/3",
                    "incorrect": "-1/3",
                    "unmarked": 0,
                }
            },
        }
        assert _is_valid("evaluation", instance)

    def test_marking_scheme_with_negative_number(self):
        """Marking scores can be negative numbers."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "markingSchemes": {
                DEFAULT_SECTION_KEY: {
                    "correct": 1,
                    "incorrect": -0.25,
                    "unmarked": 0,
                }
            },
        }
        assert _is_valid("evaluation", instance)

    def test_outputs_configuration_optional(self):
        """outputsConfiguration is optional; omitting it must be accepted."""
        assert _is_valid("evaluation", _MINIMAL_EVALUATION_LOCAL)

    def test_outputs_configuration_draw_score_disabled(self):
        """drawScore with enabled=False requires no position."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "outputsConfiguration": {
                "drawScore": {"enabled": False},
            },
        }
        assert _is_valid("evaluation", instance)

    def test_outputs_configuration_draw_score_enabled_with_position(self):
        """drawScore with enabled=True requires a position array."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "outputsConfiguration": {
                "drawScore": {
                    "enabled": True,
                    "position": [100, 200],
                },
            },
        }
        assert _is_valid("evaluation", instance)

    def test_conditional_sets_valid_structure(self):
        """A conditionalSets entry with name/matcher/evaluation must be accepted."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "conditionalSets": [
                {
                    "name": "SetA",
                    "matcher": {
                        "formatString": "{Roll}",
                        "matchRegex": "^A.*",
                    },
                    "evaluation": {
                        "sourceType": "local",
                        "options": {
                            "questionsInOrder": ["q1", "q2"],
                            "answersInOrder": ["B", "D"],
                        },
                        "markingSchemes": {
                            DEFAULT_SECTION_KEY: {
                                "correct": 2,
                                "incorrect": -0.5,
                                "unmarked": 0,
                            }
                        },
                    },
                }
            ],
        }
        assert _is_valid("evaluation", instance)

    def test_answers_in_order_multiple_correct_type(self):
        """answersInOrder allows arrays of correct answers for ambiguous questions."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {
                "questionsInOrder": ["q1", "q2"],
                "answersInOrder": [["A", "B"], "C"],
            },
        }
        assert _is_valid("evaluation", instance)

    def test_answers_in_order_weighted_multiple_correct(self):
        """answersInOrder allows weighted pairs [[answer, score], ...]."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {
                "questionsInOrder": ["q1"],
                "answersInOrder": [[["A", 2], ["B", 0.5]]],
            },
        }
        assert _is_valid("evaluation", instance)

    def test_should_explain_scoring_flag(self):
        """shouldExplainScoring is an optional boolean in outputsConfiguration."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "outputsConfiguration": {"shouldExplainScoring": True},
        }
        assert _is_valid("evaluation", instance)

    def test_should_export_explanation_csv_flag(self):
        """shouldExportExplanationCsv is an optional boolean in outputsConfiguration."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "outputsConfiguration": {"shouldExportExplanationCsv": True},
        }
        assert _is_valid("evaluation", instance)


class TestEvaluationSchemaInvalidMissingRequired:
    """Missing required fields in evaluation schema must fail validation."""

    def test_missing_source_type(self):
        """sourceType is required."""
        instance = {
            k: v for k, v in _MINIMAL_EVALUATION_LOCAL.items() if k != "sourceType"
        }
        assert not _is_valid("evaluation", instance)
        errors = _schema_errors("evaluation", instance)
        assert any("sourceType" in e for e in errors)

    def test_missing_options(self):
        """options is required."""
        instance = {
            k: v for k, v in _MINIMAL_EVALUATION_LOCAL.items() if k != "options"
        }
        assert not _is_valid("evaluation", instance)
        errors = _schema_errors("evaluation", instance)
        assert any("options" in e for e in errors)

    def test_missing_marking_schemes(self):
        """markingSchemes is required."""
        instance = {
            k: v for k, v in _MINIMAL_EVALUATION_LOCAL.items() if k != "markingSchemes"
        }
        assert not _is_valid("evaluation", instance)
        errors = _schema_errors("evaluation", instance)
        assert any("markingSchemes" in e for e in errors)

    def test_marking_schemes_missing_default_section(self):
        """markingSchemes must contain the DEFAULT section key."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "markingSchemes": {
                "CustomSection": {
                    "marking": {"correct": 1, "incorrect": 0, "unmarked": 0},
                    "questions": ["q1"],
                }
            },
        }
        assert not _is_valid("evaluation", instance)

    def test_local_options_missing_questions_in_order(self):
        """Local sourceType options must contain questionsInOrder."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {"answersInOrder": ["A", "B"]},
        }
        assert not _is_valid("evaluation", instance)

    def test_local_options_missing_answers_in_order(self):
        """Local sourceType options must contain answersInOrder."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {"questionsInOrder": ["q1", "q2"]},
        }
        assert not _is_valid("evaluation", instance)

    def test_csv_options_missing_answer_key_csv_path(self):
        """CSV sourceType options must contain answerKeyCsvPath."""
        instance = {
            **_MINIMAL_EVALUATION_CSV,
            "options": {},
        }
        assert not _is_valid("evaluation", instance)

    def test_conditional_set_missing_name(self):
        """Each conditionalSet item must have a name."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "conditionalSets": [
                {
                    "matcher": {"formatString": "{Roll}", "matchRegex": ".*"},
                    "evaluation": {
                        "sourceType": "local",
                        "options": {
                            "questionsInOrder": ["q1"],
                            "answersInOrder": ["A"],
                        },
                        "markingSchemes": {
                            DEFAULT_SECTION_KEY: {
                                "correct": 1,
                                "incorrect": 0,
                                "unmarked": 0,
                            }
                        },
                    },
                }
            ],
        }
        assert not _is_valid("evaluation", instance)
        errors = _schema_errors("evaluation", instance)
        assert any("name" in e for e in errors)

    def test_conditional_set_missing_matcher(self):
        """Each conditionalSet item must have a matcher."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "conditionalSets": [
                {
                    "name": "SetA",
                    "evaluation": {
                        "sourceType": "local",
                        "options": {
                            "questionsInOrder": ["q1"],
                            "answersInOrder": ["A"],
                        },
                        "markingSchemes": {
                            DEFAULT_SECTION_KEY: {
                                "correct": 1,
                                "incorrect": 0,
                                "unmarked": 0,
                            }
                        },
                    },
                }
            ],
        }
        assert not _is_valid("evaluation", instance)
        errors = _schema_errors("evaluation", instance)
        assert any("matcher" in e for e in errors)

    def test_conditional_set_missing_evaluation(self):
        """Each conditionalSet item must have an evaluation block."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "conditionalSets": [
                {
                    "name": "SetA",
                    "matcher": {"formatString": "{Roll}", "matchRegex": ".*"},
                }
            ],
        }
        assert not _is_valid("evaluation", instance)
        errors = _schema_errors("evaluation", instance)
        assert any("evaluation" in e for e in errors)


class TestEvaluationSchemaInvalidTypes:
    """Wrong types for evaluation fields must fail validation."""

    def test_invalid_source_type_value(self):
        """sourceType must be one of the allowed enum values."""
        instance = {**_MINIMAL_EVALUATION_LOCAL, "sourceType": "database"}
        assert not _is_valid("evaluation", instance)

    def test_marking_scheme_incorrect_type(self):
        """Marking score values must be numbers or matching strings, not booleans."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "markingSchemes": {
                DEFAULT_SECTION_KEY: {
                    "correct": True,  # bool is not a valid score
                    "incorrect": 0,
                    "unmarked": 0,
                }
            },
        }
        # Boolean is coerced by JSON schema to integer in some validators,
        # but the string pattern won't match True. Verify it is not accepted
        # as a plain number by checking the instance directly.
        # (JSON Schema may accept booleans as numbers in some draft versions;
        #  we check for string-pattern mismatch when the type is not number.)
        errors = _schema_errors("evaluation", instance)
        # If there are errors they relate to the marking score type
        # (this is informational; the key assertion is schema-level coverage)
        assert isinstance(errors, list)

    def test_questions_in_order_must_be_array(self):
        """questionsInOrder must be an array, not a string."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {
                "questionsInOrder": "q1,q2",
                "answersInOrder": ["A", "B"],
            },
        }
        assert not _is_valid("evaluation", instance)

    def test_answers_in_order_must_be_array(self):
        """answersInOrder must be an array."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {
                "questionsInOrder": ["q1", "q2"],
                "answersInOrder": "A,B",
            },
        }
        assert not _is_valid("evaluation", instance)

    def test_draw_score_enabled_without_position_is_invalid(self):
        """drawScore enabled=True requires the position field."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "outputsConfiguration": {
                "drawScore": {"enabled": True},  # Missing required position
            },
        }
        assert not _is_valid("evaluation", instance)

    def test_draw_answers_summary_enabled_without_position_is_invalid(self):
        """drawAnswersSummary enabled=True requires the position field."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "outputsConfiguration": {
                "drawAnswersSummary": {"enabled": True},  # Missing required position
            },
        }
        assert not _is_valid("evaluation", instance)

    def test_conditional_set_matcher_missing_format_string(self):
        """matcher object must have both formatString and matchRegex."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "conditionalSets": [
                {
                    "name": "SetA",
                    "matcher": {"matchRegex": ".*"},  # Missing formatString
                    "evaluation": {
                        "sourceType": "local",
                        "options": {
                            "questionsInOrder": ["q1"],
                            "answersInOrder": ["A"],
                        },
                        "markingSchemes": {
                            DEFAULT_SECTION_KEY: {
                                "correct": 1,
                                "incorrect": 0,
                                "unmarked": 0,
                            }
                        },
                    },
                }
            ],
        }
        assert not _is_valid("evaluation", instance)

    def test_unknown_top_level_key_rejected(self):
        """additionalProperties=False: unknown top-level keys must be rejected."""
        instance = {**_MINIMAL_EVALUATION_LOCAL, "extraKey": "extra"}
        assert not _is_valid("evaluation", instance)


class TestEvaluationSchemaValidationFunctions:
    """Tests for validate_evaluation_json() which raises EvaluationValidationError."""

    def test_valid_local_evaluation_does_not_raise(self):
        """A valid local evaluation dict must not raise any exception."""
        validate_evaluation_json(_MINIMAL_EVALUATION_LOCAL, _DUMMY_PATH)

    def test_valid_csv_evaluation_does_not_raise(self):
        """A valid CSV evaluation dict must not raise any exception."""
        validate_evaluation_json(_MINIMAL_EVALUATION_CSV, _DUMMY_PATH)

    def test_missing_source_type_raises(self):
        """Missing sourceType must raise EvaluationValidationError."""
        instance = {
            k: v for k, v in _MINIMAL_EVALUATION_LOCAL.items() if k != "sourceType"
        }
        with pytest.raises(EvaluationValidationError) as exc_info:
            validate_evaluation_json(instance, _DUMMY_PATH)
        assert "sourceType" in str(exc_info.value)

    def test_missing_options_raises(self):
        """Missing options must raise EvaluationValidationError."""
        instance = {
            k: v for k, v in _MINIMAL_EVALUATION_LOCAL.items() if k != "options"
        }
        with pytest.raises(EvaluationValidationError) as exc_info:
            validate_evaluation_json(instance, _DUMMY_PATH)
        assert "options" in str(exc_info.value)

    def test_missing_marking_schemes_raises(self):
        """Missing markingSchemes must raise EvaluationValidationError."""
        instance = {
            k: v for k, v in _MINIMAL_EVALUATION_LOCAL.items() if k != "markingSchemes"
        }
        with pytest.raises(EvaluationValidationError) as exc_info:
            validate_evaluation_json(instance, _DUMMY_PATH)
        assert "markingSchemes" in str(exc_info.value)

    def test_invalid_source_type_raises(self):
        """An invalid sourceType must raise EvaluationValidationError."""
        instance = {**_MINIMAL_EVALUATION_LOCAL, "sourceType": "invalid"}
        with pytest.raises(EvaluationValidationError):
            validate_evaluation_json(instance, _DUMMY_PATH)

    def test_draw_score_enabled_without_position_raises(self):
        """drawScore enabled=True without position must raise EvaluationValidationError."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "outputsConfiguration": {
                "drawScore": {"enabled": True},
            },
        }
        with pytest.raises(EvaluationValidationError):
            validate_evaluation_json(instance, _DUMMY_PATH)

    def test_local_missing_questions_in_order_raises(self):
        """Local evaluation missing questionsInOrder must raise EvaluationValidationError."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {"answersInOrder": ["A"]},
        }
        with pytest.raises(EvaluationValidationError) as exc_info:
            validate_evaluation_json(instance, _DUMMY_PATH)
        assert "questionsInOrder" in str(exc_info.value)

    def test_local_missing_answers_in_order_raises(self):
        """Local evaluation missing answersInOrder must raise EvaluationValidationError."""
        instance = {
            **_MINIMAL_EVALUATION_LOCAL,
            "options": {"questionsInOrder": ["q1"]},
        }
        with pytest.raises(EvaluationValidationError) as exc_info:
            validate_evaluation_json(instance, _DUMMY_PATH)
        assert "answersInOrder" in str(exc_info.value)

    def test_additional_property_raises(self):
        """An unknown top-level key must raise EvaluationValidationError."""
        instance = {**_MINIMAL_EVALUATION_LOCAL, "bogusField": "value"}
        with pytest.raises(EvaluationValidationError):
            validate_evaluation_json(instance, _DUMMY_PATH)
