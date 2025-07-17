"""
Security utilities and best practices for cmdrdata-openai
"""

import hashlib
import hmac
import ipaddress
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from .exceptions import AuthenticationError, SecurityError, ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


class SecurityConfig:
    """Security configuration settings"""

    def __init__(self):
        self.max_request_size = int(
            os.getenv("CMDRDATA_MAX_REQUEST_SIZE", "1048576")
        )  # 1MB
        self.rate_limit_window = int(
            os.getenv("CMDRDATA_RATE_LIMIT_WINDOW", "60")
        )  # 1 minute
        self.max_requests_per_window = int(
            os.getenv("CMDRDATA_MAX_REQUESTS_PER_WINDOW", "100")
        )
        self.allowed_origins = os.getenv("CMDRDATA_ALLOWED_ORIGINS", "*").split(",")
        self.blocked_ips = os.getenv("CMDRDATA_BLOCKED_IPS", "").split(",")
        self.require_https = (
            os.getenv("CMDRDATA_REQUIRE_HTTPS", "true").lower() == "true"
        )
        self.api_key_rotation_days = int(
            os.getenv("CMDRDATA_API_KEY_ROTATION_DAYS", "90")
        )


class APIKeyManager:
    """Secure API key management and validation"""

    # Known API key patterns for different providers
    API_KEY_PATTERNS = {
        "openai": {
            "pattern": r"^sk-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}$",
            "legacy_pattern": r"^sk-[a-zA-Z0-9]{48}$",
            "description": "OpenAI API key",
        },
        "anthropic": {
            "pattern": r"^sk-ant-api03-[a-zA-Z0-9_-]{95}$",
            "description": "Anthropic API key",
        },
        "google": {
            "pattern": r"^AIza[a-zA-Z0-9_-]{35}$",
            "description": "Google Gen AI API key",
        },
        "cmdrdata": {
            "pattern": r"^cmd-live-v1-[a-zA-Z0-9_-]+-[a-zA-Z0-9_-]+$",
            "legacy_pattern": r"^tk-[a-zA-Z0-9]{32,64}$",
            "description": "cmdrdata tracking key",
        },
    }

    # Suspicious patterns that might indicate injection attempts
    SUSPICIOUS_PATTERNS = [
        r"<script[^>]*>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"data:",  # Data URLs
        r"vbscript:",  # VBScript URLs
        r"on\w+\s*=",  # Event handlers
        r"expression\s*\(",  # CSS expressions
        r"(\r\n|\n|\r)",  # Line breaks
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",  # Control characters
        r"(union|select|insert|update|delete|drop|create|alter)\s+",  # SQL keywords
        r"(\||&|;|`|\$\(|\${)",  # Command injection
    ]

    @classmethod
    def validate_api_key(cls, api_key: str, provider: str) -> Dict[str, Any]:
        """
        Validate API key format and security

        Args:
            api_key: The API key to validate
            provider: The provider type (openai, anthropic, etc.)

        Returns:
            Dict with validation results

        Raises:
            SecurityError: If key appears malicious
            ValidationError: If key format is invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValidationError("API key must be a non-empty string")

        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, api_key, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in API key: {pattern}")
                raise SecurityError(f"API key contains suspicious pattern: {pattern}")

        # Check length (reasonable bounds)
        if len(api_key) < 20 or len(api_key) > 500:
            raise ValidationError("API key length is outside acceptable range")

        # Provider-specific validation
        if provider in cls.API_KEY_PATTERNS:
            pattern_info = cls.API_KEY_PATTERNS[provider]
            pattern = pattern_info["pattern"]

            # Check legacy patterns for providers that support them
            if not re.match(pattern, api_key):
                legacy_pattern = pattern_info.get("legacy_pattern")
                if legacy_pattern and re.match(legacy_pattern, api_key):
                    if provider == "openai":
                        logger.warning("Using legacy OpenAI API key format")
                    elif provider == "cmdrdata":
                        logger.warning("Using legacy cmdrdata API key format")
                    return {
                        "valid": True,
                        "provider": provider,
                        "format": "legacy",
                        "description": pattern_info["description"],
                    }

            if not re.match(pattern, api_key):
                raise ValidationError(f"Invalid {provider} API key format")

        return {
            "valid": True,
            "provider": provider,
            "format": "standard",
            "description": cls.API_KEY_PATTERNS.get(provider, {}).get(
                "description", "Unknown provider"
            ),
        }

    @classmethod
    def sanitize_api_key_for_logging(cls, api_key: str) -> str:
        """
        Sanitize API key for safe logging

        Args:
            api_key: The API key to sanitize

        Returns:
            Sanitized key for logging
        """
        if not api_key:
            return "[EMPTY]"

        if len(api_key) < 8:
            return "[REDACTED]"

        # Show first 3 and last 3 characters
        return f"{api_key[:3]}...{api_key[-3:]}"

    @classmethod
    def generate_tracking_key(cls) -> str:
        """
        Generate a secure tracking key

        Returns:
            Secure tracking key
        """
        # Generate 32 bytes of random data
        key_bytes = secrets.token_bytes(32)
        # Encode as hex and add prefix
        return f"tk-{key_bytes.hex()}"

    @classmethod
    def hash_api_key(cls, api_key: str, salt: Optional[str] = None) -> str:
        """
        Hash API key for secure storage

        Args:
            api_key: The API key to hash
            salt: Optional salt (generates if not provided)

        Returns:
            Hashed API key
        """
        if not salt:
            salt = secrets.token_hex(16)

        # Use PBKDF2 with SHA-256
        key_hash = hashlib.pbkdf2_hmac(
            "sha256", api_key.encode(), salt.encode(), 100000
        )
        return f"{salt}:{key_hash.hex()}"

    @classmethod
    def verify_api_key_hash(cls, api_key: str, stored_hash: str) -> bool:
        """
        Verify API key against stored hash

        Args:
            api_key: The API key to verify
            stored_hash: The stored hash to verify against

        Returns:
            True if key matches hash
        """
        try:
            salt, key_hash = stored_hash.split(":", 1)
            test_hash = hashlib.pbkdf2_hmac(
                "sha256", api_key.encode(), salt.encode(), 100000
            )
            return hmac.compare_digest(key_hash, test_hash.hex())
        except (ValueError, TypeError):
            return False


class InputSanitizer:
    """Sanitize and validate user inputs"""

    # Maximum lengths for different input types
    MAX_LENGTHS = {
        "customer_id": 255,
        "model_name": 100,
        "message_content": 100000,  # 100KB
        "metadata_key": 100,
        "metadata_value": 1000,
        "url": 2048,
        "general_string": 1000,
    }

    # Allowed characters for different input types
    ALLOWED_PATTERNS = {
        "customer_id": r"^[a-zA-Z0-9._-]+$",
        "model_name": r"^[a-zA-Z0-9._-]+$",
        "alphanumeric": r"^[a-zA-Z0-9]+$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    }

    @classmethod
    def sanitize_string(
        cls, value: str, input_type: str = "general_string", strict: bool = False
    ) -> str:
        """
        Sanitize string input

        Args:
            value: The string to sanitize
            input_type: Type of input for validation rules
            strict: Whether to apply strict validation

        Returns:
            Sanitized string

        Raises:
            ValidationError: If input is invalid
            SecurityError: If input contains suspicious patterns
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")

        original_value = value

        # Check for suspicious patterns
        for pattern in APIKeyManager.SUSPICIOUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in input: {pattern}")
                if strict:
                    raise SecurityError(f"Input contains suspicious pattern: {pattern}")
                else:
                    # Remove suspicious content
                    value = re.sub(pattern, "", value, flags=re.IGNORECASE)

        # Check length
        max_length = cls.MAX_LENGTHS.get(input_type, cls.MAX_LENGTHS["general_string"])
        if len(value) > max_length:
            if strict:
                raise ValidationError(f"Input too long. Maximum length: {max_length}")
            else:
                value = value[:max_length]
                logger.warning(f"Input truncated to {max_length} characters")

        # Apply pattern validation
        if input_type in cls.ALLOWED_PATTERNS:
            pattern = cls.ALLOWED_PATTERNS[input_type]
            if not re.match(pattern, value):
                if strict:
                    raise ValidationError(
                        f"Input does not match required pattern for {input_type}"
                    )
                else:
                    # Extract valid characters based on pattern
                    if input_type == "customer_id":
                        value = re.sub(r"[^a-zA-Z0-9._-]", "", value)
                    elif input_type == "model_name":
                        value = re.sub(r"[^a-zA-Z0-9._-]", "", value)
                    elif input_type == "alphanumeric":
                        value = re.sub(r"[^a-zA-Z0-9]", "", value)

        # Remove null bytes and other control characters
        value = value.replace("\x00", "")
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", value)

        # Log if value was modified
        if value != original_value:
            logger.info(
                f"Input sanitized: {len(original_value)} -> {len(value)} characters"
            )

        return value

    @classmethod
    def validate_url(cls, url: str, allowed_schemes: Optional[Set[str]] = None) -> str:
        """
        Validate and sanitize URL

        Args:
            url: The URL to validate
            allowed_schemes: Set of allowed URL schemes

        Returns:
            Validated URL

        Raises:
            ValidationError: If URL is invalid
            SecurityError: If URL is suspicious
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")

        # Check length
        if len(url) > cls.MAX_LENGTHS["url"]:
            raise ValidationError(
                f"URL too long. Maximum length: {cls.MAX_LENGTHS['url']}"
            )

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")

        # Check for basic URL structure - must have scheme and netloc
        if not parsed.scheme and not parsed.netloc:
            raise ValidationError("Invalid URL format")

        # Check scheme
        if allowed_schemes is None:
            allowed_schemes = {"http", "https"}

        if parsed.scheme not in allowed_schemes:
            raise ValidationError("URL must use HTTP or HTTPS protocol")

        # Check for suspicious patterns
        for pattern in APIKeyManager.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                raise SecurityError(f"URL contains suspicious pattern: {pattern}")

        # Check for private IP addresses in production
        if parsed.hostname:
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private and os.getenv("CMDRDATA_ENVIRONMENT") == "production":
                    raise SecurityError(
                        "Private IP addresses not allowed in production"
                    )
            except ValueError:
                # Not an IP address, hostname is fine
                pass

        return url

    @classmethod
    def sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata dictionary

        Args:
            metadata: The metadata to sanitize

        Returns:
            Sanitized metadata

        Raises:
            ValidationError: If metadata is invalid
        """
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")

        # Check size
        serialized_size = len(str(metadata))
        if serialized_size > 10000:  # 10KB limit
            raise ValidationError("Metadata size exceeds limit")

        sanitized = {}

        for key, value in metadata.items():
            # Sanitize key
            if not isinstance(key, str):
                raise ValidationError("Metadata keys must be strings")

            sanitized_key = cls.sanitize_string(key, "metadata_key", strict=True)

            # Sanitize value
            if isinstance(value, str):
                sanitized_value = cls.sanitize_string(value, "metadata_value")
            elif isinstance(value, (int, float, bool)):
                sanitized_value = value
            elif value is None:
                sanitized_value = None
            else:
                # Convert to string and sanitize
                sanitized_value = cls.sanitize_string(str(value), "metadata_value")

            sanitized[sanitized_key] = sanitized_value

        return sanitized

    @classmethod
    def validate_customer_id(cls, customer_id: str) -> bool:
        """Validate customer ID format"""
        if not customer_id or not isinstance(customer_id, str):
            raise ValidationError("Customer ID must be a non-empty string")

        if len(customer_id) > cls.MAX_LENGTHS["customer_id"]:
            raise ValidationError(
                f"Customer ID must be {cls.MAX_LENGTHS['customer_id']} characters or less"
            )

        # Check for suspicious patterns
        for pattern in APIKeyManager.SUSPICIOUS_PATTERNS:
            if re.search(pattern, customer_id, re.IGNORECASE):
                raise SecurityError(
                    f"Customer ID contains suspicious pattern: {pattern}"
                )

        # Check format
        if not re.match(cls.ALLOWED_PATTERNS["customer_id"], customer_id):
            raise ValidationError(
                "Customer ID can only contain letters, numbers, dots, underscores, and hyphens"
            )

        return True

    @classmethod
    def validate_timeout(cls, timeout: float) -> bool:
        """Validate timeout value"""
        if not isinstance(timeout, (int, float)):
            raise ValidationError("Timeout must be a number")

        if timeout <= 0:
            raise ValidationError("Timeout must be positive")

        if timeout > 300:
            raise ValidationError("Timeout cannot exceed 300 seconds")

        return True

    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """Validate model name format"""
        if not model_name or not isinstance(model_name, str):
            raise ValidationError("Model name must be a non-empty string")

        # Check for suspicious patterns
        for pattern in APIKeyManager.SUSPICIOUS_PATTERNS:
            if re.search(pattern, model_name, re.IGNORECASE):
                raise SecurityError(
                    f"Model name contains suspicious pattern: {pattern}"
                )

        # Check format
        if not re.match(cls.ALLOWED_PATTERNS["model_name"], model_name):
            raise ValidationError(
                "Model name can only contain letters, numbers, dots, underscores, and hyphens"
            )

        return True

    @classmethod
    def validate_token_count(cls, token_count: int) -> bool:
        """Validate token count"""
        if not isinstance(token_count, int):
            raise ValidationError("Token count must be an integer")

        if token_count < 0:
            raise ValidationError("Token count cannot be negative")

        if token_count > 1000000:
            raise ValidationError("Token count exceeds maximum limit")

        return True

    @classmethod
    def validate_metadata(cls, metadata: dict) -> bool:
        """Validate metadata dictionary"""
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")

        # Check size
        serialized_size = len(str(metadata))
        if serialized_size > 10000:
            raise ValidationError("Metadata size exceeds limit")

        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValidationError("Metadata keys must be strings")

            # Check for suspicious patterns in keys
            for pattern in APIKeyManager.SUSPICIOUS_PATTERNS:
                if re.search(pattern, key, re.IGNORECASE):
                    raise SecurityError(
                        f"Metadata key contains suspicious pattern: {pattern}"
                    )

            # Check for suspicious patterns in values
            if isinstance(value, str):
                for pattern in APIKeyManager.SUSPICIOUS_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise SecurityError(
                            f"Metadata value contains suspicious pattern: {pattern}"
                        )

        return True

    @classmethod
    def validate_chat_messages(cls, messages: list) -> bool:
        """Validate chat messages format"""
        if not isinstance(messages, list):
            raise ValidationError("Messages must be a list")

        if not messages:
            raise ValidationError("Messages list cannot be empty")

        valid_roles = {"system", "user", "assistant"}

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

            if not isinstance(message["content"], str):
                raise ValidationError(f"Message {i} content must be a string")

            # Check for suspicious patterns in content
            for pattern in APIKeyManager.SUSPICIOUS_PATTERNS:
                if re.search(pattern, message["content"], re.IGNORECASE):
                    raise SecurityError(
                        f"Message {i} content contains suspicious pattern: {pattern}"
                    )

        return True


def validate_input(validation_func):
    """Decorator for input validation"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                validation_func(*args, **kwargs)
            except (ValidationError, SecurityError):
                raise
            except Exception:
                raise ValidationError("Validation failed")
            return func(*args, **kwargs)

        return wrapper

    return decorator


class RateLimiter:
    """Rate limiting for API requests"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for given identifier

        Args:
            identifier: Unique identifier (IP, user ID, etc.)

        Returns:
            True if request is allowed
        """
        now = time.time()

        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time
                for req_time in self.requests[identifier]
                if now - req_time < self.window_seconds
            ]
        else:
            self.requests[identifier] = []

        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True

        return False

    def get_reset_time(self, identifier: str) -> Optional[float]:
        """
        Get when rate limit resets for identifier

        Args:
            identifier: Unique identifier

        Returns:
            Unix timestamp when limit resets
        """
        if identifier not in self.requests or not self.requests[identifier]:
            return None

        oldest_request = min(self.requests[identifier])
        return oldest_request + self.window_seconds


def require_valid_api_key(provider: str):
    """
    Decorator to validate API key

    Args:
        provider: API provider name
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract API key from kwargs or instance
            api_key = kwargs.get("api_key")
            if not api_key and args:
                # Try to get from instance
                instance = args[0]
                api_key = getattr(instance, "api_key", None)

            if not api_key:
                raise AuthenticationError("API key is required")

            # Validate API key
            try:
                APIKeyManager.validate_api_key(api_key, provider)
            except (ValidationError, SecurityError) as e:
                raise AuthenticationError(f"Invalid API key: {e}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limited(max_requests: int = 100, window_seconds: int = 60):
    """
    Decorator for rate limiting

    Args:
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds
    """
    limiter = RateLimiter(max_requests, window_seconds)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier (could be IP, user ID, etc.)
            identifier = kwargs.get("identifier", "default")

            if not limiter.is_allowed(identifier):
                reset_time = limiter.get_reset_time(identifier)
                raise SecurityError(
                    f"Rate limit exceeded. Try again after {reset_time}",
                    error_code="RATE_LIMIT_EXCEEDED",
                    details={"reset_time": reset_time},
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def secure_compare(a: str, b: str) -> bool:
    """
    Secure string comparison that prevents timing attacks

    Args:
        a: First string
        b: Second string

    Returns:
        True if strings are equal
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_secure_token(length: int = 32) -> str:
    """
    Generate cryptographically secure token

    Args:
        length: Token length in bytes

    Returns:
        Secure token as hex string
    """
    return secrets.token_hex(length)


def validate_request_signature(
    request_body: bytes, signature: str, secret: str
) -> bool:
    """
    Validate request signature for webhook security

    Args:
        request_body: Raw request body
        signature: Provided signature
        secret: Webhook secret

    Returns:
        True if signature is valid
    """
    expected_signature = hmac.new(
        secret.encode("utf-8"), request_body, hashlib.sha256
    ).hexdigest()

    return secure_compare(signature, expected_signature)


# Global security configuration
security_config = SecurityConfig()

# Global rate limiter
global_rate_limiter = RateLimiter(
    max_requests=security_config.max_requests_per_window,
    window_seconds=security_config.rate_limit_window,
)
