"""Tests for custom exception hierarchy.

This module tests all custom exceptions to ensure they:
1. Store the correct context information
2. Generate appropriate error messages
3. Maintain the proper exception hierarchy
4. Can be caught at different levels of specificity
"""

from pathlib import Path

import pytest

from src.utils.exceptions import (
    AlignmentError,
    AnswerKeyError,
    BarcodeDetectionError,
    BubbleDetectionError,
    ConfigError,
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigValidationError,
    EvaluationConfigLoadError,
    EvaluationConfigNotFoundError,
    EvaluationError,
    EvaluationValidationError,
    FieldDefinitionError,
    FileSizeLimitError,
    FileWriteError,
    ImageProcessingError,
    ImageReadError,
    InputDirectoryNotFoundError,
    InputError,
    InputFileNotFoundError,
    InvalidConfigValueError,
    MarkerDetectionError,
    OCRError,
    OMRCheckerError,
    OutputDirectoryError,
    OutputError,
    PathTraversalError,
    PreprocessorError,
    ProcessingError,
    SchemaValidationError,
    ScoringError,
    SecurityError,
    TemplateError,
    TemplateLoadError,
    TemplateNotFoundError,
    TemplateValidationError,
    ValidationError,
)


class TestBaseException:
    """Tests for the base OMRCheckerError exception."""

    def test_base_exception_with_message_only(self):
        """Test base exception with just a message."""
        exc = OMRCheckerError("Test error message")
        assert str(exc) == "Test error message"
        assert exc.message == "Test error message"
        assert exc.context == {}

    def test_base_exception_with_context(self):
        """Test base exception with context information."""
        context = {"file": "test.jpg", "line": 42}
        exc = OMRCheckerError("Test error", context=context)
        assert exc.message == "Test error"
        assert exc.context == context
        assert "file=test.jpg" in str(exc)
        assert "line=42" in str(exc)

    @pytest.mark.parametrize(
        "exception_class,args",
        [
            (InputError, ("test",)),
            (ProcessingError, ("test",)),
            (ValidationError, ("test",)),
            (TemplateError, ("test",)),
            (EvaluationError, ("test",)),
            (ConfigError, ("test",)),
            (SecurityError, ("test",)),
        ],
    )
    def test_base_exception_is_catchable(self, exception_class, args):
        """Test that all custom exceptions can be caught as OMRCheckerError."""
        with pytest.raises(OMRCheckerError):
            raise exception_class(*args)


class TestInputOutputExceptions:
    """Tests for input/output related exceptions."""

    def test_input_directory_not_found_error(self):
        """Test InputDirectoryNotFoundError with path information."""
        path = Path("/nonexistent/directory")
        exc = InputDirectoryNotFoundError(path)

        assert isinstance(exc, InputError)
        assert isinstance(exc, OMRCheckerError)
        assert exc.path == path
        assert str(path) in str(exc)
        assert exc.context["path"] == str(path)

    @pytest.mark.parametrize(
        "file_type,expected_in_str",
        [
            (None, False),
            ("template", True),
            ("config", True),
            ("evaluation", True),
        ],
    )
    def test_input_file_not_found_error(self, file_type, expected_in_str):
        """Test InputFileNotFoundError with and without file type."""
        path = Path("/path/to/missing.json")
        if file_type:
            exc = InputFileNotFoundError(path, file_type=file_type)
            assert exc.file_type == file_type
            assert file_type in str(exc)
            assert exc.context["file_type"] == file_type
        else:
            exc = InputFileNotFoundError(path)
            assert not hasattr(exc, "file_type") or exc.file_type is None

        assert isinstance(exc, InputError)
        assert exc.path == path
        assert "missing.json" in str(exc)

    def test_image_read_error_with_reason(self):
        """Test ImageReadError with specific reason."""
        path = Path("/path/to/corrupt.jpg")
        reason = "File is corrupted"
        exc = ImageReadError(path, reason)

        assert isinstance(exc, InputError)
        assert exc.path == path
        assert exc.reason == reason
        assert reason in str(exc)
        assert exc.context["reason"] == reason

    def test_output_directory_error(self):
        """Test OutputDirectoryError with path and reason."""
        path = Path("/output/dir")
        reason = "Permission denied"
        exc = OutputDirectoryError(path, reason)

        assert isinstance(exc, OutputError)
        assert exc.path == path
        assert exc.reason == reason
        assert str(path) in str(exc)
        assert reason in str(exc)

    def test_file_write_error(self):
        """Test FileWriteError with optional reason."""
        path = Path("/output/file.csv")
        exc = FileWriteError(path, reason="Disk full")

        assert isinstance(exc, OutputError)
        assert exc.path == path
        assert "Disk full" in str(exc)


class TestValidationExceptions:
    """Tests for validation related exceptions."""

    def test_template_validation_error_with_errors_list(self):
        """Test TemplateValidationError with list of validation errors."""
        path = Path("/path/to/template.json")
        errors = ["Missing field 'dimensions'", "Invalid preprocessor name"]
        exc = TemplateValidationError(path, errors=errors)

        assert isinstance(exc, ValidationError)
        assert exc.path == path
        assert exc.errors == errors
        assert len(exc.context["errors"]) == 2

    def test_config_validation_error_with_reason(self):
        """Test ConfigValidationError with general reason."""
        path = Path("/path/to/config.json")
        reason = "Invalid JSON structure"
        exc = ConfigValidationError(path, reason=reason)

        assert isinstance(exc, ValidationError)
        assert exc.reason == reason
        assert reason in str(exc)

    def test_evaluation_validation_error(self):
        """Test EvaluationValidationError."""
        path = Path("/path/to/evaluation.json")
        errors = ["Missing answer key"]
        exc = EvaluationValidationError(path, errors=errors)

        assert isinstance(exc, ValidationError)
        assert exc.errors == errors

    def test_schema_validation_error(self):
        """Test SchemaValidationError with schema name."""
        schema_name = "template_v2"
        errors = ["Field 'bubbles' is required"]
        data_path = Path("/data/template.json")
        exc = SchemaValidationError(schema_name, errors, data_path)

        assert isinstance(exc, ValidationError)
        assert exc.schema_name == schema_name
        assert exc.errors == errors
        assert exc.data_path == data_path
        assert schema_name in str(exc)


class TestProcessingExceptions:
    """Tests for processing related exceptions."""

    def test_marker_detection_error(self):
        """Test MarkerDetectionError with file path."""
        file_path = Path("/images/sample.jpg")
        reason = "Insufficient contrast"
        exc = MarkerDetectionError(file_path, reason)

        assert isinstance(exc, ProcessingError)
        assert exc.file_path == file_path
        assert exc.reason == reason
        assert "sample.jpg" in str(exc)
        assert reason in str(exc)

    @pytest.mark.parametrize(
        "operation,file_path,reason",
        [
            ("rotation", None, None),
            ("cropping", Path("/images/test.jpg"), "Invalid coordinates"),
            ("resize", Path("/images/photo.jpg"), None),
            ("normalize", None, "Out of range values"),
        ],
    )
    def test_image_processing_error(self, operation, file_path, reason):
        """Test ImageProcessingError with various parameter combinations."""
        if file_path and reason:
            exc = ImageProcessingError(operation, file_path, reason)
            assert exc.file_path == file_path
            assert exc.reason == reason
            assert all(x in str(exc) for x in [operation, file_path.name, reason])
        elif file_path:
            exc = ImageProcessingError(operation, file_path)
            assert exc.file_path == file_path
            assert exc.reason is None
            assert operation in str(exc)
            assert file_path.name in str(exc)
        elif reason:
            exc = ImageProcessingError(operation, None, reason)
            assert exc.file_path is None
            assert exc.reason == reason
            assert operation in str(exc)
            assert reason in str(exc)
        else:
            exc = ImageProcessingError(operation)
            assert exc.file_path is None
            assert exc.reason is None
            assert operation in str(exc)

        assert exc.operation == operation

    def test_alignment_error(self):
        """Test AlignmentError."""
        file_path = Path("/images/skewed.jpg")
        exc = AlignmentError(file_path, "Could not find alignment points")

        assert isinstance(exc, ProcessingError)
        assert exc.file_path == file_path

    def test_bubble_detection_error_with_field(self):
        """Test BubbleDetectionError with field ID."""
        file_path = Path("/images/omr.jpg")
        field_id = "q1"
        reason = "No bubbles detected in field"
        exc = BubbleDetectionError(file_path, field_id, reason)

        assert isinstance(exc, ProcessingError)
        assert exc.field_id == field_id
        assert field_id in str(exc)

    def test_ocr_error(self):
        """Test OCRError."""
        file_path = Path("/images/text.jpg")
        exc = OCRError(file_path, field_id="name", reason="Low confidence")

        assert isinstance(exc, ProcessingError)
        assert exc.field_id == "name"

    def test_barcode_detection_error(self):
        """Test BarcodeDetectionError."""
        file_path = Path("/images/barcode.jpg")
        exc = BarcodeDetectionError(file_path, reason="Unreadable barcode")

        assert isinstance(exc, ProcessingError)
        assert "barcode.jpg" in str(exc)


class TestTemplateExceptions:
    """Tests for template related exceptions."""

    def test_template_not_found_error(self):
        """Test TemplateNotFoundError."""
        search_path = Path("/project/inputs")
        exc = TemplateNotFoundError(search_path)

        assert isinstance(exc, TemplateError)
        assert exc.search_path == search_path
        assert "template.json" in str(exc).lower()
        assert str(search_path) in str(exc)

    def test_template_load_error(self):
        """Test TemplateLoadError with reason."""
        path = Path("/templates/invalid.json")
        reason = "Malformed JSON"
        exc = TemplateLoadError(path, reason)

        assert isinstance(exc, TemplateError)
        assert exc.path == path
        assert exc.reason == reason
        assert reason in str(exc)

    def test_preprocessor_error(self):
        """Test PreprocessorError with all details."""
        preprocessor_name = "CropOnMarkers"
        file_path = Path("/images/test.jpg")
        reason = "Markers not found"
        exc = PreprocessorError(preprocessor_name, file_path, reason)

        assert isinstance(exc, TemplateError)
        assert exc.preprocessor_name == preprocessor_name
        assert exc.file_path == file_path
        assert all(x in str(exc) for x in [preprocessor_name, "test.jpg", reason])

    def test_field_definition_error(self):
        """Test FieldDefinitionError."""
        field_id = "q1_bubbles"
        reason = "Invalid bubble coordinates"
        template_path = Path("/templates/template.json")
        exc = FieldDefinitionError(field_id, reason, template_path)

        assert isinstance(exc, TemplateError)
        assert exc.field_id == field_id
        assert exc.reason == reason
        assert exc.template_path == template_path
        assert field_id in str(exc)


class TestEvaluationExceptions:
    """Tests for evaluation related exceptions."""

    def test_evaluation_config_not_found_error(self):
        """Test EvaluationConfigNotFoundError."""
        search_path = Path("/project/inputs/sample")
        exc = EvaluationConfigNotFoundError(search_path)

        assert isinstance(exc, EvaluationError)
        assert exc.search_path == search_path
        assert "evaluation.json" in str(exc)

    def test_evaluation_config_load_error(self):
        """Test EvaluationConfigLoadError."""
        path = Path("/configs/evaluation.json")
        reason = "Invalid answer format"
        exc = EvaluationConfigLoadError(path, reason)

        assert isinstance(exc, EvaluationError)
        assert exc.path == path
        assert reason in str(exc)

    @pytest.mark.parametrize(
        "reason,question_id",
        [
            ("Answer key file not found", None),
            ("Multiple correct answers not allowed", "q5"),
            ("Invalid answer format", "q10"),
            ("Missing answer for question", None),
        ],
    )
    def test_answer_key_error(self, reason, question_id):
        """Test AnswerKeyError with and without question identifier."""
        if question_id:
            exc = AnswerKeyError(reason, question_id)
            assert exc.question_id == question_id
            assert question_id in str(exc)
        else:
            exc = AnswerKeyError(reason)
            assert exc.question_id is None

        assert isinstance(exc, EvaluationError)
        assert exc.reason == reason
        assert reason in str(exc)

    def test_scoring_error_complete(self):
        """Test ScoringError with all details."""
        reason = "Invalid marking scheme"
        file_path = Path("/responses/student1.jpg")
        question_id = "q10"
        exc = ScoringError(reason, file_path, question_id)

        assert isinstance(exc, EvaluationError)
        assert exc.reason == reason
        assert exc.file_path == file_path
        assert exc.question_id == question_id
        assert all(x in str(exc) for x in [reason, "student1.jpg", question_id])


class TestSecurityExceptions:
    """Tests for security related exceptions."""

    @pytest.mark.parametrize(
        "path_str,base_path_str",
        [
            ("../../etc/passwd", None),
            ("/allowed/dir/../../../sensitive/data", "/allowed/dir"),
            ("../config.json", "/project"),
            ("../../../../root", None),
        ],
    )
    def test_path_traversal_error(self, path_str, base_path_str):
        """Test PathTraversalError with and without base path."""
        path = Path(path_str)
        if base_path_str:
            base_path = Path(base_path_str)
            exc = PathTraversalError(path, base_path)
            assert exc.base_path == base_path
            assert str(base_path) in str(exc)
        else:
            exc = PathTraversalError(path)
            assert exc.base_path is None

        assert isinstance(exc, SecurityError)
        assert exc.path == path
        assert "traversal" in str(exc).lower()

    def test_file_size_limit_error(self):
        """Test FileSizeLimitError with size information."""
        path = Path("/uploads/huge.jpg")
        size = 100_000_000  # 100 MB
        limit = 10_000_000  # 10 MB
        exc = FileSizeLimitError(path, size, limit)

        assert isinstance(exc, SecurityError)
        assert exc.path == path
        assert exc.size == size
        assert exc.limit == limit
        assert str(size) in str(exc)
        assert str(limit) in str(exc)


class TestConfigExceptions:
    """Tests for configuration related exceptions."""

    def test_config_not_found_error(self):
        """Test ConfigNotFoundError."""
        path = Path("/config/app.json")
        exc = ConfigNotFoundError(path)

        assert isinstance(exc, ConfigError)
        assert exc.path == path
        assert str(path) in str(exc)

    def test_config_load_error(self):
        """Test ConfigLoadError."""
        path = Path("/config/settings.json")
        reason = "Syntax error on line 5"
        exc = ConfigLoadError(path, reason)

        assert isinstance(exc, ConfigError)
        assert exc.path == path
        assert exc.reason == reason
        assert reason in str(exc)

    def test_invalid_config_value_error(self):
        """Test InvalidConfigValueError."""
        key = "max_workers"
        value = -1
        reason = "Must be a positive integer"
        exc = InvalidConfigValueError(key, value, reason)

        assert isinstance(exc, ConfigError)
        assert exc.key == key
        assert exc.value == value
        assert exc.reason == reason
        assert key in str(exc)
        assert reason in str(exc)


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catching behavior."""

    @pytest.mark.parametrize(
        "exception_class,category_class,args,kwargs",
        [
            (InputDirectoryNotFoundError, InputError, (Path("/missing"),), {}),
            (ImageReadError, InputError, (Path("/corrupt.jpg"), "Corrupted"), {}),
            (
                MarkerDetectionError,
                ProcessingError,
                (Path("/test.jpg"), "No markers"),
                {},
            ),
            (AlignmentError, ProcessingError, (Path("/test.jpg"), "Failed"), {}),
            (TemplateValidationError, ValidationError, (Path("/template.json"),), {}),
            (
                ConfigValidationError,
                ValidationError,
                (Path("/config.json"),),
                {"reason": "Invalid"},
            ),
        ],
    )
    def test_catch_by_category(self, exception_class, category_class, args, kwargs):
        """Test that exceptions can be caught by their category."""
        with pytest.raises(category_class):
            raise exception_class(*args, **kwargs)

    def test_catch_all_omrchecker_errors(self):
        """Test that all custom exceptions inherit from OMRCheckerError."""
        exceptions_to_test = [
            InputDirectoryNotFoundError(Path("/test")),
            ImageReadError(Path("/test.jpg")),
            TemplateNotFoundError(Path("/dir")),
            MarkerDetectionError(Path("/omr.jpg")),
            ConfigValidationError(Path("/config.json")),
            PathTraversalError(Path("../../etc")),
            ScoringError("test error"),
        ]

        for exc in exceptions_to_test:
            with pytest.raises(OMRCheckerError):
                raise exc

    def test_exception_context_preservation(self):
        """Test that exception context is preserved through the hierarchy."""
        path = Path("/test/file.jpg")
        reason = "Test reason"
        exc = ImageProcessingError("rotation", path, reason)

        # Context should be accessible
        assert exc.context["operation"] == "rotation"
        assert exc.context["file_path"] == str(path)
        assert exc.context["reason"] == reason

        # Should still be catchable as base exception
        with pytest.raises(OMRCheckerError) as exc_info:
            raise exc

        caught_exc = exc_info.value
        assert caught_exc.context["operation"] == "rotation"


class TestExceptionMessages:
    """Tests for exception message formatting."""

    def test_messages_contain_relevant_information(self):
        """Test that exception messages contain all relevant information."""
        # Test with path
        path = Path("/path/to/file.jpg")
        exc = InputFileNotFoundError(path, "template")
        msg = str(exc)
        assert "file.jpg" in msg
        assert "template" in msg

        # Test with multiple pieces of information
        exc = BubbleDetectionError(
            Path("/omr.jpg"), field_id="q1", reason="No bubbles found"
        )
        msg = str(exc)
        assert "omr.jpg" in msg
        assert "q1" in msg
        assert "No bubbles found" in msg

    def test_message_without_optional_fields(self):
        """Test that messages work correctly without optional fields."""
        exc = ImageProcessingError("cropping")
        msg = str(exc)
        assert "cropping" in msg
        # Should not crash even without file_path or reason

    def test_context_in_string_representation(self):
        """Test that context is included in string representation."""
        context = {"operation": "resize", "dimensions": "800x600"}
        exc = OMRCheckerError("Processing failed", context=context)
        msg = str(exc)
        assert "operation=resize" in msg
        assert "dimensions=800x600" in msg
