"""Custom exception hierarchy for OMRChecker.

This module defines all custom exceptions used throughout the application,
providing better error handling, debugging, and user feedback.
"""

from pathlib import Path
from typing import Any


class OMRCheckerError(Exception):
    """Base exception class for all OMRChecker errors.

    All custom exceptions in OMRChecker should inherit from this class.
    This allows catching all application-specific errors with a single except clause.

    Attributes:
        message: Human-readable error message
        context: Additional context information about the error
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            context: Optional dictionary with additional error context
        """
        self.message = message
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


# ============================================================================
# Input/Output Exceptions
# ============================================================================


class InputError(OMRCheckerError):
    """Base class for input-related errors."""


class InputDirectoryNotFoundError(InputError):
    """Raised when the specified input directory does not exist."""

    def __init__(self, path: Path) -> None:
        """Initialize the exception.

        Args:
            path: The path that was not found
        """
        self.path = path
        super().__init__(
            f"Input directory does not exist: '{path}'", context={"path": str(path)}
        )


class InputFileNotFoundError(InputError):
    """Raised when an expected input file is not found."""

    def __init__(self, path: Path, file_type: str | None = None) -> None:
        """Initialize the exception.

        Args:
            path: The file path that was not found
            file_type: Optional description of the file type
        """
        self.path = path
        self.file_type = file_type
        file_desc = f"{file_type} " if file_type else ""
        super().__init__(
            f"Input {file_desc}file not found: '{path}'",
            context={"path": str(path), "file_type": file_type},
        )


class ImageReadError(InputError):
    """Raised when an image file cannot be read or decoded."""

    def __init__(self, path: Path, reason: str | None = None) -> None:
        """Initialize the exception.

        Args:
            path: Path to the image file
            reason: Optional reason why the image could not be read
        """
        self.path = path
        self.reason = reason
        msg = f"Unable to read image: '{path}'"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg, context={"path": str(path), "reason": reason})


class OutputError(OMRCheckerError):
    """Base class for output-related errors."""


class OutputDirectoryError(OutputError):
    """Raised when there are issues with output directory operations."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the exception.

        Args:
            path: Path to the output directory
            reason: Reason for the error
        """
        self.path = path
        self.reason = reason
        super().__init__(
            f"Output directory error at '{path}': {reason}",
            context={"path": str(path), "reason": reason},
        )


class FileWriteError(OutputError):
    """Raised when a file cannot be written."""

    def __init__(self, path: Path, reason: str | None = None) -> None:
        """Initialize the exception.

        Args:
            path: Path to the file that couldn't be written
            reason: Optional reason for the failure
        """
        self.path = path
        self.reason = reason
        msg = f"Failed to write file: '{path}'"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg, context={"path": str(path), "reason": reason})


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationError(OMRCheckerError):
    """Base class for validation-related errors."""


class TemplateValidationError(ValidationError):
    """Raised when a template JSON file fails validation."""

    def __init__(
        self, path: Path, errors: list[str] | None = None, reason: str | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            path: Path to the invalid template file
            errors: List of validation error messages
            reason: Optional general reason for validation failure
        """
        self.path = path
        self.errors = errors or []
        self.reason = reason
        msg = f"Invalid template JSON: '{path}'"
        if reason:
            msg += f" - {reason}"
        super().__init__(
            msg,
            context={"path": str(path), "errors": self.errors, "reason": reason},
        )


class ConfigValidationError(ValidationError):
    """Raised when a config JSON file fails validation."""

    def __init__(
        self, path: Path, errors: list[str] | None = None, reason: str | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            path: Path to the invalid config file
            errors: List of validation error messages
            reason: Optional general reason for validation failure
        """
        self.path = path
        self.errors = errors or []
        self.reason = reason
        msg = f"Invalid config JSON: '{path}'"
        if reason:
            msg += f" - {reason}"
        super().__init__(
            msg,
            context={"path": str(path), "errors": self.errors, "reason": reason},
        )


class EvaluationValidationError(ValidationError):
    """Raised when an evaluation JSON file fails validation."""

    def __init__(
        self, path: Path, errors: list[str] | None = None, reason: str | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            path: Path to the invalid evaluation file
            errors: List of validation error messages
            reason: Optional general reason for validation failure
        """
        self.path = path
        self.errors = errors or []
        self.reason = reason
        msg = f"Invalid evaluation JSON: '{path}'"
        if reason:
            msg += f" - {reason}"
        super().__init__(
            msg,
            context={"path": str(path), "errors": self.errors, "reason": reason},
        )


class SchemaValidationError(ValidationError):
    """Raised when data fails schema validation."""

    def __init__(
        self, schema_name: str, errors: list[str], data_path: Path | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            schema_name: Name of the schema that failed
            errors: List of validation error messages
            data_path: Optional path to the data file
        """
        self.schema_name = schema_name
        self.errors = errors
        self.data_path = data_path
        msg = f"Schema validation failed for '{schema_name}'"
        if data_path:
            msg += f" at '{data_path}'"
        super().__init__(
            msg,
            context={
                "schema": schema_name,
                "errors": errors,
                "data_path": str(data_path) if data_path else None,
            },
        )


# ============================================================================
# Processing Exceptions
# ============================================================================


class ProcessingError(OMRCheckerError):
    """Base class for processing-related errors."""


class MarkerDetectionError(ProcessingError):
    """Raised when markers cannot be detected on an OMR sheet."""

    def __init__(self, file_path: Path, reason: str | None = None) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the OMR image file
            reason: Optional reason for detection failure
        """
        self.file_path = file_path
        self.reason = reason
        msg = f"Marker detection failed for '{file_path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, context={"file_path": str(file_path), "reason": reason})


class ImageProcessingError(ProcessingError):
    """Raised when image processing operations fail."""

    def __init__(
        self, operation: str, file_path: Path | None = None, reason: str | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            operation: Name of the processing operation that failed
            file_path: Optional path to the image being processed
            reason: Optional reason for the failure
        """
        self.operation = operation
        self.file_path = file_path
        self.reason = reason
        msg = f"Image processing failed during '{operation}'"
        if file_path:
            msg += f" for '{file_path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            context={
                "operation": operation,
                "file_path": str(file_path) if file_path else None,
                "reason": reason,
            },
        )


class AlignmentError(ProcessingError):
    """Raised when image alignment fails."""

    def __init__(self, file_path: Path, reason: str | None = None) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the image file
            reason: Optional reason for alignment failure
        """
        self.file_path = file_path
        self.reason = reason
        msg = f"Image alignment failed for '{file_path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, context={"file_path": str(file_path), "reason": reason})


class BubbleDetectionError(ProcessingError):
    """Raised when bubble detection fails."""

    def __init__(
        self, file_path: Path, field_id: str | None = None, reason: str | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the OMR image
            field_id: Optional field identifier where detection failed
            reason: Optional reason for failure
        """
        self.file_path = file_path
        self.field_id = field_id
        self.reason = reason
        msg = f"Bubble detection failed for '{file_path}'"
        if field_id:
            msg += f" at field '{field_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            context={
                "file_path": str(file_path),
                "field_id": field_id,
                "reason": reason,
            },
        )


class OCRError(ProcessingError):
    """Raised when OCR processing fails."""

    def __init__(
        self, file_path: Path, field_id: str | None = None, reason: str | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the image
            field_id: Optional field identifier where OCR failed
            reason: Optional reason for failure
        """
        self.file_path = file_path
        self.field_id = field_id
        self.reason = reason
        msg = f"OCR processing failed for '{file_path}'"
        if field_id:
            msg += f" at field '{field_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            context={
                "file_path": str(file_path),
                "field_id": field_id,
                "reason": reason,
            },
        )


class BarcodeDetectionError(ProcessingError):
    """Raised when barcode detection fails."""

    def __init__(
        self, file_path: Path, field_id: str | None = None, reason: str | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the image
            field_id: Optional field identifier where barcode detection failed
            reason: Optional reason for failure
        """
        self.file_path = file_path
        self.field_id = field_id
        self.reason = reason
        msg = f"Barcode detection failed for '{file_path}'"
        if field_id:
            msg += f" at field '{field_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            context={
                "file_path": str(file_path),
                "field_id": field_id,
                "reason": reason,
            },
        )


# ============================================================================
# Template Exceptions
# ============================================================================


class TemplateError(OMRCheckerError):
    """Base class for template-related errors."""


class TemplateNotFoundError(TemplateError):
    """Raised when a template file is not found in the directory tree."""

    def __init__(self, search_path: Path) -> None:
        """Initialize the exception.

        Args:
            search_path: Directory path where template was searched
        """
        self.search_path = search_path
        super().__init__(
            f"No template.json found in directory tree of '{search_path}'",
            context={"search_path": str(search_path)},
        )


class TemplateLoadError(TemplateError):
    """Raised when a template file cannot be loaded or parsed."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the exception.

        Args:
            path: Path to the template file
            reason: Reason for load failure
        """
        self.path = path
        self.reason = reason
        super().__init__(
            f"Failed to load template '{path}': {reason}",
            context={"path": str(path), "reason": reason},
        )


class PreprocessorError(TemplateError):
    """Raised when a preprocessor operation fails."""

    def __init__(
        self,
        preprocessor_name: str,
        file_path: Path | None = None,
        reason: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            preprocessor_name: Name of the preprocessor that failed
            file_path: Optional path to the file being processed
            reason: Optional reason for failure
        """
        self.preprocessor_name = preprocessor_name
        self.file_path = file_path
        self.reason = reason
        msg = f"Preprocessor '{preprocessor_name}' failed"
        if file_path:
            msg += f" for '{file_path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            msg,
            context={
                "preprocessor": preprocessor_name,
                "file_path": str(file_path) if file_path else None,
                "reason": reason,
            },
        )


class FieldDefinitionError(TemplateError):
    """Raised when there's an issue with field definitions in the template."""

    def __init__(
        self, field_id: str, reason: str, template_path: Path | None = None
    ) -> None:
        """Initialize the exception.

        Args:
            field_id: Identifier of the problematic field
            reason: Description of the issue
            template_path: Optional path to the template file
        """
        self.field_id = field_id
        self.reason = reason
        self.template_path = template_path
        msg = f"Invalid field definition '{field_id}': {reason}"
        if template_path:
            msg += f" in '{template_path}'"
        super().__init__(
            msg,
            context={
                "field_id": field_id,
                "reason": reason,
                "template_path": str(template_path) if template_path else None,
            },
        )


class TemplateConfigurationError(TemplateError):
    """Raised when there's an issue with template configuration (preprocessors, scan zones, etc.)."""

    def __init__(self, message: str, **context) -> None:
        """Initialize the exception.

        Args:
            message: Description of the configuration issue
            **context: Additional context information
        """
        self.context_data = context
        super().__init__(message, context=context)


# ============================================================================
# Evaluation Exceptions
# ============================================================================


class EvaluationError(OMRCheckerError):
    """Base class for evaluation-related errors."""


class EvaluationConfigNotFoundError(EvaluationError):
    """Raised when an evaluation config is expected but not found."""

    def __init__(self, search_path: Path) -> None:
        """Initialize the exception.

        Args:
            search_path: Directory path where evaluation config was searched
        """
        self.search_path = search_path
        super().__init__(
            f"No evaluation.json found at '{search_path}'",
            context={"search_path": str(search_path)},
        )


class EvaluationConfigLoadError(EvaluationError):
    """Raised when an evaluation config cannot be loaded."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the exception.

        Args:
            path: Path to the evaluation config file
            reason: Reason for load failure
        """
        self.path = path
        self.reason = reason
        super().__init__(
            f"Failed to load evaluation config '{path}': {reason}",
            context={"path": str(path), "reason": reason},
        )


class AnswerKeyError(EvaluationError):
    """Raised when there's an issue with the answer key."""

    def __init__(self, reason: str, question_id: str | None = None) -> None:
        """Initialize the exception.

        Args:
            reason: Description of the issue
            question_id: Optional identifier of the problematic question
        """
        self.reason = reason
        self.question_id = question_id
        msg = f"Answer key error: {reason}"
        if question_id:
            msg += f" (question: {question_id})"
        super().__init__(msg, context={"reason": reason, "question_id": question_id})


class ScoringError(EvaluationError):
    """Raised when score calculation fails."""

    def __init__(
        self,
        reason: str,
        file_path: Path | None = None,
        question_id: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            reason: Description of the scoring issue
            file_path: Optional path to the OMR file being scored
            question_id: Optional identifier of the problematic question
        """
        self.reason = reason
        self.file_path = file_path
        self.question_id = question_id
        msg = f"Scoring failed: {reason}"
        if file_path:
            msg += f" for '{file_path}'"
        if question_id:
            msg += f" at question '{question_id}'"
        super().__init__(
            msg,
            context={
                "reason": reason,
                "file_path": str(file_path) if file_path else None,
                "question_id": question_id,
            },
        )


# ============================================================================
# Security Exceptions
# ============================================================================


class SecurityError(OMRCheckerError):
    """Base class for security-related errors."""


class PathTraversalError(SecurityError):
    """Raised when a path traversal attempt is detected."""

    def __init__(self, path: Path, base_path: Path | None = None) -> None:
        """Initialize the exception.

        Args:
            path: The suspicious path
            base_path: Optional base path that was being protected
        """
        self.path = path
        self.base_path = base_path
        msg = f"Path traversal detected: '{path}'"
        if base_path:
            msg += f" (base: '{base_path}')"
        super().__init__(
            msg,
            context={
                "path": str(path),
                "base_path": str(base_path) if base_path else None,
            },
        )


class FileSizeLimitError(SecurityError):
    """Raised when a file exceeds size limits."""

    def __init__(self, path: Path, size: int, limit: int) -> None:
        """Initialize the exception.

        Args:
            path: Path to the oversized file
            size: Actual file size in bytes
            limit: Maximum allowed size in bytes
        """
        self.path = path
        self.size = size
        self.limit = limit
        super().__init__(
            f"File '{path}' exceeds size limit: {size} bytes > {limit} bytes",
            context={"path": str(path), "size": size, "limit": limit},
        )


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigError(OMRCheckerError):
    """Base class for configuration-related errors."""


class ConfigNotFoundError(ConfigError):
    """Raised when a required configuration file is not found."""

    def __init__(self, path: Path) -> None:
        """Initialize the exception.

        Args:
            path: Path where config file was expected
        """
        self.path = path
        super().__init__(
            f"Configuration file not found: '{path}'", context={"path": str(path)}
        )


class ConfigLoadError(ConfigError):
    """Raised when a configuration file cannot be loaded."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the exception.

        Args:
            path: Path to the config file
            reason: Reason for load failure
        """
        self.path = path
        self.reason = reason
        super().__init__(
            f"Failed to load configuration '{path}': {reason}",
            context={"path": str(path), "reason": reason},
        )


class InvalidConfigValueError(ConfigError):
    """Raised when a configuration value is invalid."""

    def __init__(self, key: str, value: object, reason: str) -> None:
        """Initialize the exception.

        Args:
            key: Configuration key
            value: Invalid value
            reason: Reason why the value is invalid
        """
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid configuration value for '{key}': {value} - {reason}",
            context={"key": key, "value": str(value), "reason": reason},
        )
