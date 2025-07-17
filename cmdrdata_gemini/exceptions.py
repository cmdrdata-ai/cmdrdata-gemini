"""
Custom exceptions for cmdrdata-openai package
"""

from typing import Any, Dict, Optional


class CMDRDataError(Exception):
    """Base exception for all cmdrdata-gemini errors"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        base_message = self.message
        if self.error_code:
            base_message = f"[{self.error_code}] {base_message}"
        return base_message


class ConfigurationError(CMDRDataError):
    """Raised when there's a configuration issue"""

    pass


class AuthenticationError(CMDRDataError):
    """Raised when authentication fails"""

    pass


class ValidationError(CMDRDataError):
    """Raised when input validation fails"""

    pass


class RateLimitError(CMDRDataError):
    """Raised when rate limits are exceeded"""

    pass


class TrackingError(CMDRDataError):
    """Raised when usage tracking fails"""

    pass


class NetworkError(CMDRDataError):
    """Raised when network operations fail"""

    pass


class TimeoutError(CMDRDataError):
    """Raised when operations time out"""

    pass


class RetryExhaustedError(CMDRDataError):
    """Raised when retry attempts are exhausted"""

    pass


class CircuitBreakerError(CMDRDataError):
    """Raised when circuit breaker is open"""

    pass


class SecurityError(CMDRDataError):
    """Raised when security validation fails"""

    pass


class CompatibilityError(CMDRDataError):
    """Raised when version compatibility check fails"""

    pass
