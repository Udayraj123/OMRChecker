"""Tests for validation utility functions."""

import pytest

from src.utils.validations import (
    get_camel_case_hint,
    suggest_camel_case,
    to_camel_case,
)


class TestToCamelCase:
    """Tests for to_camel_case function."""

    def test_snake_case_conversion(self) -> None:
        assert to_camel_case("field_labels") == "fieldLabels"
        assert to_camel_case("bubble_dimensions") == "bubbleDimensions"
        assert to_camel_case("pre_processors") == "preProcessors"

    def test_kebab_case_conversion(self) -> None:
        assert to_camel_case("field-labels") == "fieldLabels"
        assert to_camel_case("bubble-dimensions") == "bubbleDimensions"

    def test_space_separated_conversion(self) -> None:
        assert to_camel_case("field labels") == "fieldLabels"
        assert to_camel_case("bubble dimensions") == "bubbleDimensions"

    def test_pascal_case_to_camel_case(self) -> None:
        assert to_camel_case("FieldLabels") == "fieldLabels"
        assert to_camel_case("BubbleDimensions") == "bubbleDimensions"

    def test_already_camel_case(self) -> None:
        assert to_camel_case("fieldLabels") == "fieldLabels"
        assert to_camel_case("bubbleDimensions") == "bubbleDimensions"

    def test_single_word(self) -> None:
        assert to_camel_case("field") == "field"
        assert to_camel_case("Field") == "field"

    def test_empty_string(self) -> None:
        assert to_camel_case("") == ""


class TestSuggestCamelCase:
    """Tests for suggest_camel_case function."""

    def test_returns_suggestion_for_snake_case(self) -> None:
        assert suggest_camel_case("field_labels") == "fieldLabels"
        assert suggest_camel_case("bubble_dimensions") == "bubbleDimensions"

    def test_returns_suggestion_for_pascal_case(self) -> None:
        assert suggest_camel_case("FieldLabels") == "fieldLabels"

    def test_returns_none_for_camel_case(self) -> None:
        assert suggest_camel_case("fieldLabels") is None
        assert suggest_camel_case("bubbleDimensions") is None

    def test_returns_none_for_lowercase(self) -> None:
        assert suggest_camel_case("field") is None
        assert suggest_camel_case("dimensions") is None


class TestGetCamelCaseHint:
    """Tests for get_camel_case_hint function."""

    def test_returns_hint_for_non_camel_case(self) -> None:
        hint = get_camel_case_hint("field_labels")
        assert "fieldLabels" in hint
        assert "Did you mean" in hint

    def test_returns_empty_string_for_camel_case(self) -> None:
        assert get_camel_case_hint("fieldLabels") == ""
        assert get_camel_case_hint("bubbleDimensions") == ""

    def test_returns_empty_string_for_lowercase(self) -> None:
        assert get_camel_case_hint("field") == ""
