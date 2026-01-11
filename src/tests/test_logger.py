"""Tests for logger utility functions."""

from src.utils.logger import Logger


class TestLogger:
    """Test suite for logger utility functions."""

    def test_logger_creation(self) -> None:
        """Test logger creation with default settings."""
        log = Logger("test_logger")
        assert log is not None
        assert log.show_logs_by_type is not None

    def test_logger_debug_message(self) -> None:
        """Test debug level logging."""
        log = Logger("test_debug")
        # Should not raise exception
        log.debug("Test debug message")

    def test_logger_info_message(self) -> None:
        """Test info level logging."""
        log = Logger("test_info")
        log.info("Test info message")

    def test_logger_warning_message(self) -> None:
        """Test warning level logging."""
        log = Logger("test_warning")
        log.warning("Test warning message")

    def test_logger_error_message(self) -> None:
        """Test error level logging."""
        log = Logger("test_error")
        log.error("Test error message")

    def test_logger_critical_message(self) -> None:
        """Test critical level logging."""
        log = Logger("test_critical")
        log.critical("Test critical message")

    def test_logger_set_levels(self) -> None:
        """Test setting log levels."""
        log = Logger("test_levels")
        log.set_log_levels({"debug": False, "info": True})
        assert log.show_logs_by_type["debug"] is False
        assert log.show_logs_by_type["info"] is True

    def test_logger_reset_levels(self) -> None:
        """Test resetting log levels."""
        log = Logger("test_reset")
        log.set_log_levels({"debug": False})
        log.reset_log_levels()
        assert log.show_logs_by_type["debug"] is True

    def test_logger_multiple_messages(self) -> None:
        """Test logging multiple messages at once."""
        log = Logger("test_multi")
        log.info("Message 1", "Message 2", "Message 3")

    def test_logger_custom_separator(self) -> None:
        """Test logging with custom separator."""
        log = Logger("test_sep")
        log.info("Part1", "Part2", sep="-")
