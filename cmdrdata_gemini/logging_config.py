"""
Structured logging configuration for cmdrdata-openai
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import ConfigurationError


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter"""

    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add custom fields if present
        if hasattr(record, "customer_id"):
            log_entry["customer_id"] = record.customer_id
        if hasattr(record, "model"):
            log_entry["model"] = record.model
        if hasattr(record, "tokens"):
            log_entry["tokens"] = record.tokens
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "response_time"):
            log_entry["response_time"] = record.response_time
        if hasattr(record, "api_endpoint"):
            log_entry["api_endpoint"] = record.api_endpoint

        # Add any additional context
        extra_fields = getattr(record, "extra_fields", {})
        if extra_fields:
            log_entry.update(extra_fields)

        return json.dumps(log_entry)


class SecurityFormatter(logging.Formatter):
    """Security-focused formatter that sanitizes sensitive data"""

    SENSITIVE_FIELDS = {
        "api_key",
        "token",
        "password",
        "secret",
        "key",
        "auth",
        "authorization",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize the message
        message = record.getMessage()

        # Basic sanitization - replace likely API keys
        import re

        patterns = [
            (r"sk-[a-zA-Z0-9]{6,}", "sk-***REDACTED***"),
            (r"tk-[a-zA-Z0-9]{6,}", "tk-***REDACTED***"),
            (r"Bearer\s+[a-zA-Z0-9_-]+", "Bearer ***REDACTED***"),
            (r'"api_key":\s*"[^"]*"', '"api_key": "***REDACTED***"'),
            (r'"token":\s*"[^"]*"', '"token": "***REDACTED***"'),
        ]

        for pattern, replacement in patterns:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)

        # Create sanitized record
        sanitized_record = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=message,
            args=(),
            exc_info=record.exc_info,
        )

        # Copy safe attributes
        for attr in ["module", "funcName", "created", "msecs", "thread", "threadName"]:
            if hasattr(record, attr):
                setattr(sanitized_record, attr, getattr(record, attr))

        return super().format(sanitized_record)


class LoggingConfig:
    """Centralized logging configuration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.setup_logging()

    def setup_logging(self):
        """Configure logging based on configuration"""
        # Get configuration values
        log_level = self.config.get("log_level", "INFO").upper()
        log_format = self.config.get(
            "log_format", "structured"
        )  # structured or standard
        log_file = self.config.get("log_file")
        console_logging = self.config.get("console_logging", True)
        security_mode = self.config.get("security_mode", True)

        # Create root logger
        logger = logging.getLogger("cmdrdata_openai")
        logger.setLevel(getattr(logging, log_level))

        # Clear existing handlers
        logger.handlers.clear()

        # Choose formatter
        if log_format == "structured":
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Apply security formatter if needed
        if security_mode:
            if log_format == "structured":
                # Apply security sanitization to structured formatter
                original_format = formatter.format

                def secure_format(record):
                    # Sanitize before formatting
                    self._sanitize_record(record)
                    return original_format(record)

                formatter.format = secure_format
            else:
                formatter = SecurityFormatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )

        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_file:
            try:
                # Create log directory if it doesn't exist
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                # Rotating file handler
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")

    def _sanitize_record(self, record: logging.LogRecord):
        """Sanitize log record for security"""
        # This is a placeholder for more sophisticated sanitization
        # In a real implementation, you'd want to recursively sanitize
        # all arguments and extra fields
        pass

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with the configured settings"""
        return logging.getLogger(f"cmdrdata_openai.{name}")


# Context manager for request-scoped logging
class RequestLogger:
    """Context manager for adding request context to logs"""

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Performance monitoring decorator
def log_performance(logger: logging.Logger, operation: str):
    """Decorator to log performance metrics"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                end_time = time.time()

                logger.info(
                    f"Operation completed successfully",
                    extra={
                        "operation": operation,
                        "response_time": end_time - start_time,
                        "status": "success",
                    },
                )

                return result
            except Exception as e:
                end_time = time.time()

                logger.error(
                    f"Operation failed: {str(e)}",
                    extra={
                        "operation": operation,
                        "response_time": end_time - start_time,
                        "status": "error",
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return wrapper

    return decorator


# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    "log_level": os.getenv("CMDRDATA_LOG_LEVEL", "INFO"),
    "log_format": os.getenv("CMDRDATA_LOG_FORMAT", "structured"),
    "log_file": os.getenv("CMDRDATA_LOG_FILE"),
    "console_logging": os.getenv("CMDRDATA_CONSOLE_LOGGING", "true").lower() == "true",
    "security_mode": os.getenv("CMDRDATA_SECURITY_MODE", "true").lower() == "true",
}

# Initialize default logging
try:
    _default_config = LoggingConfig(DEFAULT_LOGGING_CONFIG)
except Exception as e:
    # Fallback to basic logging if configuration fails
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("cmdrdata_openai").warning(
        f"Could not configure structured logging: {e}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    return logging.getLogger(f"cmdrdata_openai.{name}")


# Module logger for internal logging
logger = get_logger(__name__)


def configure_logging(config: Dict[str, Any]):
    """Reconfigure logging with new settings"""
    global _default_config
    _default_config = LoggingConfig(config)
