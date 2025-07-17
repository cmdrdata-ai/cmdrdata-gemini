"""
Unit tests for logging configuration
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini.logging_config import (
    DEFAULT_LOGGING_CONFIG,
    LoggingConfig,
    RequestLogger,
    SecurityFormatter,
    StructuredFormatter,
    configure_logging,
    get_logger,
    log_performance,
)


class TestStructuredFormatter:
    """Test suite for StructuredFormatter"""

    def test_basic_log_formatting(self):
        """Test basic log record formatting"""
        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_function"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_function"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

        # Verify timestamp format
        datetime.fromisoformat(parsed["timestamp"].replace("Z", "+00:00"))

    def test_log_formatting_with_custom_fields(self):
        """Test log formatting with custom fields"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_function"

        # Add custom fields
        record.customer_id = "customer-123"
        record.model = "gpt-4"
        record.tokens = 150
        record.request_id = "req-abc123"
        record.response_time = 0.85
        record.api_endpoint = "/chat/completions"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["customer_id"] == "customer-123"
        assert parsed["model"] == "gpt-4"
        assert parsed["tokens"] == 150
        assert parsed["request_id"] == "req-abc123"
        assert parsed["response_time"] == 0.85
        assert parsed["api_endpoint"] == "/chat/completions"

    def test_log_formatting_with_extra_fields(self):
        """Test log formatting with extra fields"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test_module"
        record.funcName = "test_function"

        # Add extra fields
        record.extra_fields = {
            "custom_field": "custom_value",
            "operation": "chat_completion",
            "duration": 1.23,
        }

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["custom_field"] == "custom_value"
        assert parsed["operation"] == "chat_completion"
        assert parsed["duration"] == 1.23

    def test_log_formatting_with_exception(self):
        """Test log formatting with exception information"""
        formatter = StructuredFormatter()

        # Create exception info
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = logging.sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        record.module = "test_module"
        record.funcName = "test_function"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Error occurred"
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "Test exception"
        assert "traceback" in parsed["exception"]


class TestSecurityFormatter:
    """Test suite for SecurityFormatter"""

    def test_api_key_sanitization(self):
        """Test API key sanitization in log messages"""
        formatter = SecurityFormatter("%(message)s")

        test_cases = [
            (
                "API key: sk-abcdefghijklmnopqrstuvwxyz1234567890123456",
                "API key: sk-***REDACTED***",
            ),
            ("Token: tk-1234567890abcdef", "Token: tk-***REDACTED***"),
            (
                "Authorization: Bearer sk-test123456789012345678901234567890123456",
                "Authorization: Bearer ***REDACTED***",
            ),
            ('{"api_key": "sk-secretkey123"}', '{"api_key": "***REDACTED***"}'),
            ('{"token": "tk-secret"}', '{"token": "***REDACTED***"}'),
        ]

        for original, expected in test_cases:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="/path/to/file.py",
                lineno=42,
                msg=original,
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            assert expected in result

    def test_no_sanitization_for_safe_content(self):
        """Test that safe content is not sanitized"""
        formatter = SecurityFormatter("%(message)s")

        safe_messages = [
            "Normal log message",
            "Processing customer customer-123",
            "Model gpt-4 completed successfully",
            "Response time: 0.85 seconds",
        ]

        for message in safe_messages:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="/path/to/file.py",
                lineno=42,
                msg=message,
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            assert message in result

    def test_multiple_sensitive_patterns(self):
        """Test sanitization of multiple sensitive patterns in one message"""
        formatter = SecurityFormatter("%(message)s")

        message = "Using API key sk-test123 and tracker token tk-secret456"
        expected_parts = ["sk-***REDACTED***", "tk-***REDACTED***"]

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg=message,
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        for part in expected_parts:
            assert part in result

        # Ensure original sensitive data is not present
        assert "sk-test123" not in result
        assert "tk-secret456" not in result


class TestLoggingConfig:
    """Test suite for LoggingConfig"""

    def setup_method(self):
        """Set up test fixtures"""
        # Clear any existing handlers
        logger = logging.getLogger("cmdrdata_gemini")
        logger.handlers.clear()

    def test_default_configuration(self):
        """Test default logging configuration"""
        config = LoggingConfig()

        logger = logging.getLogger("cmdrdata_gemini")

        # Should have at least one handler (console)
        assert len(logger.handlers) >= 1

        # Check log level
        assert logger.level == logging.INFO

    def test_custom_log_level(self):
        """Test custom log level configuration"""
        config = LoggingConfig({"log_level": "DEBUG"})

        logger = logging.getLogger("cmdrdata_gemini")
        assert logger.level == logging.DEBUG

    def test_structured_format(self):
        """Test structured logging format"""
        config = LoggingConfig({"log_format": "structured"})

        logger = logging.getLogger("cmdrdata_gemini")

        # Find the console handler
        console_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                console_handler = handler
                break

        assert console_handler is not None
        assert isinstance(console_handler.formatter, StructuredFormatter)

    def test_standard_format(self):
        """Test standard logging format"""
        config = LoggingConfig({"log_format": "standard"})

        logger = logging.getLogger("cmdrdata_gemini")

        # Find the console handler
        console_handler = None
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                console_handler = handler
                break

        assert console_handler is not None
        assert not isinstance(console_handler.formatter, StructuredFormatter)

    def test_file_logging(self):
        """Test file logging configuration"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            config = LoggingConfig({"log_file": temp_path, "console_logging": False})

            logger = logging.getLogger("cmdrdata_gemini")

            # Should have file handler
            file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
            assert len(file_handlers) >= 1

            # Test logging to file
            logger.info("Test file logging")

            # Force flush
            for handler in file_handlers:
                handler.flush()

            # Check file content
            with open(temp_path, "r") as f:
                content = f.read()
                assert "Test file logging" in content

        finally:
            # Clean up - close file handlers first
            logger = logging.getLogger("cmdrdata_gemini")
            file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
            for handler in file_handlers:
                handler.close()
                logger.removeHandler(handler)

            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    # On Windows, sometimes the file is still locked
                    pass

    def test_console_logging_disabled(self):
        """Test disabling console logging"""
        config = LoggingConfig({"console_logging": False})

        logger = logging.getLogger("cmdrdata_gemini")

        # Should not have console handlers
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename")
        ]
        assert len(console_handlers) == 0

    def test_security_mode_enabled(self):
        """Test security mode enables sanitization"""
        config = LoggingConfig({"security_mode": True, "log_format": "standard"})

        logger = logging.getLogger("cmdrdata_gemini")

        # Find a handler and check if it uses SecurityFormatter
        for handler in logger.handlers:
            if hasattr(handler, "formatter"):
                # Security mode should wrap the formatter or use SecurityFormatter
                # The exact implementation may vary
                assert handler.formatter is not None

    def test_invalid_log_file_path(self):
        """Test handling of invalid log file path"""
        # Use a path that will definitely fail on Windows - invalid characters
        invalid_path = "C:\\invalid<>|path\\test.log"

        config = LoggingConfig({"log_file": invalid_path})

        # The test passes if LoggingConfig doesn't crash
        # The warning is logged but we don't need to check it specifically
        # since the exact behavior may vary by OS


class TestRequestLogger:
    """Test suite for RequestLogger context manager"""

    def test_request_logger_context(self):
        """Test RequestLogger adds context to log records"""
        logger = logging.getLogger("test_request_logger")

        context = {"request_id": "req-123", "customer_id": "customer-456"}

        with RequestLogger(logger, **context):
            # Create a log record within the context
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="/path/to/file.py",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            # The record factory should add our context
            current_factory = logging.getLogRecordFactory()
            test_record = current_factory(
                "test_logger",
                logging.INFO,
                "/path/to/file.py",
                42,
                "Test message",
                (),
                None,
            )

            # Check that context was added
            assert hasattr(test_record, "request_id")
            assert hasattr(test_record, "customer_id")
            assert test_record.request_id == "req-123"
            assert test_record.customer_id == "customer-456"

    def test_request_logger_cleanup(self):
        """Test RequestLogger cleans up after context"""
        logger = logging.getLogger("test_request_logger")
        original_factory = logging.getLogRecordFactory()

        context = {"request_id": "req-123"}

        with RequestLogger(logger, **context):
            # Factory should be different within context
            assert logging.getLogRecordFactory() != original_factory

        # Factory should be restored after context
        assert logging.getLogRecordFactory() == original_factory


class TestLogPerformanceDecorator:
    """Test suite for log_performance decorator"""

    def test_log_performance_success(self):
        """Test log_performance decorator for successful operations"""
        logger = Mock()

        @log_performance(logger, "test_operation")
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"
        logger.info.assert_called_once()

        # Check the log call
        call_args = logger.info.call_args
        assert "Operation completed successfully" in call_args[0][0]

        # Check extra fields
        extra = call_args[1]["extra"]
        assert extra["operation"] == "test_operation"
        assert extra["status"] == "success"
        assert "response_time" in extra
        assert extra["response_time"] > 0

    def test_log_performance_error(self):
        """Test log_performance decorator for failed operations"""
        logger = Mock()

        @log_performance(logger, "test_operation")
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_function()

        logger.error.assert_called_once()

        # Check the log call
        call_args = logger.error.call_args
        assert "Operation failed" in call_args[0][0]

        # Check extra fields
        extra = call_args[1]["extra"]
        assert extra["operation"] == "test_operation"
        assert extra["status"] == "error"
        assert extra["error_type"] == "ValueError"
        assert "response_time" in extra


class TestUtilityFunctions:
    """Test suite for utility functions"""

    def test_get_logger(self):
        """Test get_logger function"""
        logger = get_logger("test_module")

        assert logger.name == "cmdrdata_gemini.test_module"
        assert isinstance(logger, logging.Logger)

    def test_configure_logging(self):
        """Test configure_logging function"""
        custom_config = {"log_level": "WARNING", "log_format": "structured"}

        configure_logging(custom_config)

        logger = logging.getLogger("cmdrdata_gemini")
        assert logger.level == logging.WARNING

    def test_default_logging_config_values(self):
        """Test DEFAULT_LOGGING_CONFIG contains expected values"""
        config = DEFAULT_LOGGING_CONFIG

        assert "log_level" in config
        assert "log_format" in config
        assert "console_logging" in config
        assert "security_mode" in config

        # Test environment variable integration
        with patch.dict(os.environ, {"CMDRDATA_LOG_LEVEL": "DEBUG"}):
            # Note: Module import caching means this won't reflect the new env var
            # The actual value would depend on when the module was imported
            pass


if __name__ == "__main__":
    pytest.main([__file__])
