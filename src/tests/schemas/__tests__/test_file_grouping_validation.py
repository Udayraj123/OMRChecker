"""Tests for FileGroupingConfig validation."""

from unittest.mock import MagicMock

import pytest

from src.schemas.models.config import FileGroupingConfig, GroupingRule


class TestFileGroupingConfigValidation:
    """Test suite for FileGroupingConfig validation."""

    def test_disabled_config_skips_validation(self):
        """Test that disabled config doesn't perform validation."""
        config = FileGroupingConfig(enabled=False)
        errors = config.validate()
        assert len(errors) == 0

    def test_valid_config_with_builtin_fields(self):
        """Test that config with only built-in fields passes validation."""
        rules = [
            GroupingRule(
                name="Test Rule",
                priority=1,
                destination_pattern="output/{file_name}",
                matcher={"formatString": "{is_multi_marked}", "matchRegex": ".*"},
            )
        ]
        config = FileGroupingConfig(
            enabled=True,
            rules=rules,
            default_pattern="ungrouped/{original_name}",
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_score_field_without_evaluation_fails(self):
        """Test that using {score} without evaluation produces error."""
        rules = [
            GroupingRule(
                name="High Scorers",
                priority=1,
                destination_pattern="scores/{score}",
                matcher={"formatString": "{score}", "matchRegex": "^[8-9]"},
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate(has_evaluation=False)
        assert len(errors) > 0
        assert any("score" in error and "evaluation" in error for error in errors)

    def test_score_field_with_evaluation_passes(self):
        """Test that using {score} with evaluation passes."""
        rules = [
            GroupingRule(
                name="High Scorers",
                priority=1,
                destination_pattern="scores/{score}",
                matcher={"formatString": "{score}", "matchRegex": "^[8-9]"},
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate(has_evaluation=True)
        # Should only fail if template fields are checked
        score_errors = [e for e in errors if "score" in e and "evaluation" in e]
        assert len(score_errors) == 0

    def test_invalid_template_field_fails(self):
        """Test that using non-existent template field produces error."""
        mock_template = MagicMock()
        mock_template.all_fields = ["roll_number", "name", "booklet_code"]

        rules = [
            GroupingRule(
                name="Test Rule",
                priority=1,
                destination_pattern="output/{invalid_field}",
                matcher={"formatString": "{roll_number}", "matchRegex": ".*"},
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate(template=mock_template)
        assert len(errors) > 0
        assert any("invalid_field" in error for error in errors)

    def test_valid_template_field_passes(self):
        """Test that using valid template field passes."""
        mock_template = MagicMock()
        mock_template.all_fields = ["roll_number", "name", "booklet_code"]

        rules = [
            GroupingRule(
                name="Test Rule",
                priority=1,
                destination_pattern="booklet_{booklet_code}/roll_{roll_number}",
                matcher={"formatString": "{roll_number}", "matchRegex": ".*"},
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate(template=mock_template)
        assert len(errors) == 0

    def test_invalid_regex_fails(self):
        """Test that invalid regex pattern produces error."""
        rules = [
            GroupingRule(
                name="Bad Regex",
                priority=1,
                destination_pattern="output/{file_name}",
                matcher={
                    "formatString": "{file_name}",
                    "matchRegex": "[invalid(regex",  # Invalid regex
                },
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate()
        assert len(errors) > 0
        assert any("regex" in error.lower() for error in errors)

    def test_invalid_action_fails(self):
        """Test that invalid action produces error."""
        rules = [
            GroupingRule(
                name="Bad Action",
                priority=1,
                destination_pattern="output/{file_name}",
                matcher={"formatString": "{file_name}", "matchRegex": ".*"},
                action="move",  # Invalid - should be symlink or copy
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate()
        assert len(errors) > 0
        assert any("action" in error.lower() for error in errors)

    def test_invalid_collision_strategy_fails(self):
        """Test that invalid collision strategy produces error."""
        rules = [
            GroupingRule(
                name="Bad Strategy",
                priority=1,
                destination_pattern="output/{file_name}",
                matcher={"formatString": "{file_name}", "matchRegex": ".*"},
                collision_strategy="delete",  # Invalid
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate()
        assert len(errors) > 0
        assert any("collision" in error.lower() for error in errors)

    def test_duplicate_priorities_fails(self):
        """Test that duplicate rule priorities produce error."""
        rules = [
            GroupingRule(
                name="Rule 1",
                priority=1,
                destination_pattern="output1/{file_name}",
                matcher={"formatString": "{file_name}", "matchRegex": ".*"},
            ),
            GroupingRule(
                name="Rule 2",
                priority=1,  # Duplicate!
                destination_pattern="output2/{file_name}",
                matcher={"formatString": "{file_name}", "matchRegex": ".*"},
            ),
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate()
        assert len(errors) > 0
        assert any(
            "duplicate" in error.lower() and "priority" in error.lower()
            for error in errors
        )

    def test_invalid_pattern_syntax_fails(self):
        """Test that invalid pattern syntax produces error."""
        rules = [
            GroupingRule(
                name="Bad Pattern",
                priority=1,
                destination_pattern="output/{unclosed_brace",  # Invalid syntax
                matcher={"formatString": "{file_name}", "matchRegex": ".*"},
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate()
        assert len(errors) > 0
        assert any("syntax" in error.lower() for error in errors)

    def test_empty_matcher_format_string_fails(self):
        """Test that empty matcher format string produces error."""
        rules = [
            GroupingRule(
                name="Empty Matcher",
                priority=1,
                destination_pattern="output/{file_name}",
                matcher={"formatString": "", "matchRegex": ".*"},
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate()
        assert len(errors) > 0
        assert any("empty" in error.lower() for error in errors)

    def test_helpful_error_message_suggests_available_fields(self):
        """Test that error message suggests available fields."""
        mock_template = MagicMock()
        mock_template.all_fields = ["roll_number", "name"]

        rules = [
            GroupingRule(
                name="Test Rule",
                priority=1,
                destination_pattern="output/{wrong_field}",
                matcher={"formatString": "{roll_number}", "matchRegex": ".*"},
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate(template=mock_template)
        assert len(errors) > 0
        # Check that error suggests available fields
        error_text = " ".join(errors)
        assert "roll_number" in error_text or "Available" in error_text

    def test_multiple_errors_all_reported(self):
        """Test that all errors are reported, not just the first one."""
        mock_template = MagicMock()
        mock_template.all_fields = ["valid_field"]

        rules = [
            GroupingRule(
                name="Multiple Errors",
                priority=1,
                destination_pattern="output/{invalid_field}",
                matcher={
                    "formatString": "{another_invalid}",
                    "matchRegex": "[bad(regex",
                },
                action="invalid_action",
            )
        ]
        config = FileGroupingConfig(enabled=True, rules=rules)

        errors = config.validate(template=mock_template)
        # Should report multiple errors
        assert len(errors) >= 4  # At least: 2 bad fields, bad regex, bad action

    def test_default_pattern_validation(self):
        """Test that default pattern is also validated."""
        config = FileGroupingConfig(
            enabled=True,
            default_pattern="output/{nonexistent_field}",
        )

        mock_template = MagicMock()
        mock_template.all_fields = ["roll_number"]

        errors = config.validate(template=mock_template)
        assert len(errors) > 0
        assert any("default_pattern" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
