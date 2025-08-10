"""
Input validation and sanitization for cmdrdata-openai
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

from .exceptions import SecurityError, ValidationError

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation and sanitization"""

    # Security patterns
    SUSPICIOUS_PATTERNS = [
        r"<script[^>]*>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"data:",  # Data URLs
        r"vbscript:",  # VBScript URLs
        r"on\w+\s*=",  # Event handlers
        r"expression\s*\(",  # CSS expressions
        r'[<>"\']',  # HTML/XML characters in IDs
    ]

    # API key patterns
    API_KEY_PATTERNS = {
        "openai": r"^sk-[a-zA-Z0-9]{48}$",
        "cmdrdata": r"^(tk-[a-zA-Z0-9]{32,64}|cmd-live-v\d+-[a-zA-Z0-9_-]+)$",  # Support both formats
        "generic": r"^[a-zA-Z0-9_-]{20,}$",
    }

    @staticmethod
    def validate_api_key(api_key: str, key_type: str = "generic") -> bool:
        """
        Validate API key format

        Args:
            api_key: The API key to validate
            key_type: Type of API key (openai, cmdrdata, generic)

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If validation fails
        """
        if not api_key or not isinstance(api_key, str):
            raise ValidationError("API key must be a non-empty string")

        # Check for suspicious patterns
        for pattern in InputValidator.SUSPICIOUS_PATTERNS:
            if re.search(pattern, api_key, re.IGNORECASE):
                raise SecurityError(f"API key contains suspicious pattern: {pattern}")

        # Check format
        pattern = InputValidator.API_KEY_PATTERNS.get(
            key_type, InputValidator.API_KEY_PATTERNS["generic"]
        )
        if not re.match(pattern, api_key):
            raise ValidationError(f"Invalid {key_type} API key format")

        return True

    @staticmethod
    def validate_customer_id(customer_id: str) -> bool:
        """
        Validate customer ID format

        Args:
            customer_id: The customer ID to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not customer_id or not isinstance(customer_id, str):
            raise ValidationError("Customer ID must be a non-empty string")

        # Check length
        if len(customer_id) > 255:
            raise ValidationError("Customer ID must be 255 characters or less")

        # Check for suspicious patterns
        for pattern in InputValidator.SUSPICIOUS_PATTERNS:
            if re.search(pattern, customer_id, re.IGNORECASE):
                raise SecurityError(
                    f"Customer ID contains suspicious pattern: {pattern}"
                )

        # Basic format check - alphanumeric, hyphens, underscores, dots
        if not re.match(r"^[a-zA-Z0-9._-]+$", customer_id):
            raise ValidationError(
                "Customer ID can only contain alphanumeric characters, hyphens, underscores, and dots"
            )

        return True

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format and security

        Args:
            url: The URL to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")

        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            raise ValidationError("URL must use HTTP or HTTPS protocol")

        # Check for suspicious patterns
        for pattern in InputValidator.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                raise SecurityError(f"URL contains suspicious pattern: {pattern}")

        return True

    @staticmethod
    def validate_timeout(timeout: Union[int, float]) -> bool:
        """
        Validate timeout value

        Args:
            timeout: The timeout value to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(timeout, (int, float)):
            raise ValidationError("Timeout must be a number")

        if timeout <= 0:
            raise ValidationError("Timeout must be positive")

        if timeout > 300:  # 5 minutes
            raise ValidationError("Timeout cannot exceed 300 seconds")

        return True

    @staticmethod
    def validate_model_name(model: str) -> bool:
        """
        Validate OpenAI model name

        Args:
            model: The model name to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not model or not isinstance(model, str):
            raise ValidationError("Model name must be a non-empty string")

        # Check for suspicious patterns
        for pattern in InputValidator.SUSPICIOUS_PATTERNS:
            if re.search(pattern, model, re.IGNORECASE):
                raise SecurityError(
                    f"Model name contains suspicious pattern: {pattern}"
                )

        # Basic format check - alphanumeric, hyphens, underscores, dots
        if not re.match(r"^[a-zA-Z0-9._-]+$", model):
            raise ValidationError(
                "Model name can only contain alphanumeric characters, hyphens, underscores, and dots"
            )

        return True

    @staticmethod
    def validate_token_count(tokens: int) -> bool:
        """
        Validate token count

        Args:
            tokens: The token count to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(tokens, int):
            raise ValidationError("Token count must be an integer")

        if tokens < 0:
            raise ValidationError("Token count cannot be negative")

        if tokens > 1000000:  # 1M tokens - reasonable upper limit
            raise ValidationError("Token count exceeds maximum limit")

        return True

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata dictionary

        Args:
            metadata: The metadata to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")

        # Check size
        if len(str(metadata)) > 10000:  # 10KB limit
            raise ValidationError("Metadata size exceeds limit")

        # Check keys and values
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValidationError("Metadata keys must be strings")

            # Check for suspicious patterns in keys
            for pattern in InputValidator.SUSPICIOUS_PATTERNS:
                if re.search(pattern, key, re.IGNORECASE):
                    raise SecurityError(
                        f"Metadata key contains suspicious pattern: {pattern}"
                    )

            # Check values
            if isinstance(value, str):
                for pattern in InputValidator.SUSPICIOUS_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise SecurityError(
                            f"Metadata value contains suspicious pattern: {pattern}"
                        )

        return True

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitize string value

        Args:
            value: The string to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)

        # Remove null bytes
        value = value.replace("\x00", "")

        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
            logger.warning(f"String truncated to {max_length} characters")

        return value

    @staticmethod
    def validate_chat_messages(messages: List[Dict[str, Any]]) -> bool:
        """
        Validate chat messages format

        Args:
            messages: List of chat messages

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(messages, list):
            raise ValidationError("Messages must be a list")

        if not messages:
            raise ValidationError("Messages list cannot be empty")

        valid_roles = {"system", "user", "assistant", "function"}

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(f"Message {i} must be a dictionary")

            if "role" not in message:
                raise ValidationError(f"Message {i} missing required 'role' field")

            if message["role"] not in valid_roles:
                raise ValidationError(
                    f"Message {i} has invalid role: {message['role']}"
                )

            if "content" not in message:
                raise ValidationError(f"Message {i} missing required 'content' field")

            # Validate content
            content = message["content"]
            if not isinstance(content, str):
                raise ValidationError(f"Message {i} content must be a string")

            # Check for suspicious patterns
            for pattern in InputValidator.SUSPICIOUS_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    raise SecurityError(
                        f"Message {i} content contains suspicious pattern: {pattern}"
                    )

        return True


def validate_input(validation_func: Callable) -> Callable:
    """
    Decorator for input validation

    Args:
        validation_func: Function to validate inputs

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                validation_func(*args, **kwargs)
            except (ValidationError, SecurityError) as e:
                logger.error(f"Input validation failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected validation error: {e}")
                raise ValidationError(f"Validation failed: {e}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
