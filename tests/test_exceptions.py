"""
Unit tests for custom exceptions
"""

import pytest

from cmdrdata_gemini.exceptions import (
    CMDRDataError, ConfigurationError, AuthenticationError, ValidationError,
    RateLimitError, TrackingError, NetworkError, TimeoutError,
    RetryExhaustedError, CircuitBreakerError, SecurityError, CompatibilityError
)


class TestCMDRDataError:
    """Test suite for base CMDRDataError"""
    
    def test_basic_initialization(self):
        """Test basic error initialization"""
        error = CMDRDataError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}
    
    def test_initialization_with_error_code(self):
        """Test error initialization with error code"""
        error = CMDRDataError("Test error", error_code="ERR001")
        
        assert str(error) == "[ERR001] Test error"
        assert error.message == "Test error"
        assert error.error_code == "ERR001"
        assert error.details == {}
    
    def test_initialization_with_details(self):
        """Test error initialization with details"""
        details = {"field": "value", "count": 42}
        error = CMDRDataError("Test error", details=details)
        
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.details == details
    
    def test_initialization_with_all_parameters(self):
        """Test error initialization with all parameters"""
        details = {"reason": "validation_failed", "field": "api_key"}
        error = CMDRDataError(
            "Invalid API key format",
            error_code="VALIDATION_001",
            details=details
        )
        
        assert str(error) == "[VALIDATION_001] Invalid API key format"
        assert error.message == "Invalid API key format"
        assert error.error_code == "VALIDATION_001"
        assert error.details == details
    
    def test_details_default_value(self):
        """Test that details defaults to empty dict"""
        error = CMDRDataError("Test error", details=None)
        
        assert error.details == {}
        assert isinstance(error.details, dict)


class TestSpecificExceptions:
    """Test suite for specific exception types"""
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from CMDRDataError"""
        error = ConfigurationError("Config error")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, ConfigurationError)
        assert str(error) == "Config error"
    
    def test_authentication_error_inheritance(self):
        """Test AuthenticationError inherits from CMDRDataError"""
        error = AuthenticationError("Auth failed", error_code="AUTH001")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, AuthenticationError)
        assert str(error) == "[AUTH001] Auth failed"
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from CMDRDataError"""
        error = ValidationError("Validation failed")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, ValidationError)
        assert str(error) == "Validation failed"
    
    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inherits from CMDRDataError"""
        error = RateLimitError("Rate limit exceeded")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, RateLimitError)
        assert str(error) == "Rate limit exceeded"
    
    def test_tracking_error_inheritance(self):
        """Test TrackingError inherits from CMDRDataError"""
        error = TrackingError("Tracking failed")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, TrackingError)
        assert str(error) == "Tracking failed"
    
    def test_network_error_inheritance(self):
        """Test NetworkError inherits from CMDRDataError"""
        error = NetworkError("Network connection failed")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, NetworkError)
        assert str(error) == "Network connection failed"
    
    def test_timeout_error_inheritance(self):
        """Test TimeoutError inherits from CMDRDataError"""
        error = TimeoutError("Request timed out")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, TimeoutError)
        assert str(error) == "Request timed out"
    
    def test_retry_exhausted_error_inheritance(self):
        """Test RetryExhaustedError inherits from CMDRDataError"""
        error = RetryExhaustedError("All retry attempts failed")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, RetryExhaustedError)
        assert str(error) == "All retry attempts failed"
    
    def test_circuit_breaker_error_inheritance(self):
        """Test CircuitBreakerError inherits from CMDRDataError"""
        error = CircuitBreakerError("Circuit breaker is open")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, CircuitBreakerError)
        assert str(error) == "Circuit breaker is open"
    
    def test_security_error_inheritance(self):
        """Test SecurityError inherits from CMDRDataError"""
        error = SecurityError("Security validation failed")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, SecurityError)
        assert str(error) == "Security validation failed"
    
    def test_compatibility_error_inheritance(self):
        """Test CompatibilityError inherits from CMDRDataError"""
        error = CompatibilityError("Version incompatible")
        
        assert isinstance(error, CMDRDataError)
        assert isinstance(error, CompatibilityError)
        assert str(error) == "Version incompatible"


class TestExceptionUsage:
    """Test realistic exception usage scenarios"""
    
    def test_validation_error_with_field_details(self):
        """Test ValidationError with field-specific details"""
        error = ValidationError(
            "Invalid input format",
            error_code="VALIDATION_001",
            details={
                "field": "customer_id",
                "value": "invalid@id",
                "expected_format": "alphanumeric with hyphens and underscores"
            }
        )
        
        assert str(error) == "[VALIDATION_001] Invalid input format"
        assert error.details["field"] == "customer_id"
        assert error.details["value"] == "invalid@id"
        assert "expected_format" in error.details
    
    def test_network_error_with_retry_details(self):
        """Test NetworkError with retry information"""
        error = NetworkError(
            "Connection failed",
            error_code="NETWORK_001",
            details={
                "host": "api.example.com",
                "port": 443,
                "retry_count": 3,
                "last_error": "Connection timeout"
            }
        )
        
        assert str(error) == "[NETWORK_001] Connection failed"
        assert error.details["host"] == "api.example.com"
        assert error.details["retry_count"] == 3
    
    def test_rate_limit_error_with_timing_details(self):
        """Test RateLimitError with timing information"""
        error = RateLimitError(
            "Rate limit exceeded",
            error_code="RATE_LIMIT_001",
            details={
                "limit": 100,
                "window": 60,
                "current_count": 105,
                "reset_time": 1234567890
            }
        )
        
        assert str(error) == "[RATE_LIMIT_001] Rate limit exceeded"
        assert error.details["limit"] == 100
        assert error.details["current_count"] == 105
    
    def test_authentication_error_with_key_details(self):
        """Test AuthenticationError with key information"""
        error = AuthenticationError(
            "Invalid API key",
            error_code="AUTH_001",
            details={
                "key_prefix": "sk-abc",
                "key_type": "openai",
                "validation_error": "incorrect_format"
            }
        )
        
        assert str(error) == "[AUTH_001] Invalid API key"
        assert error.details["key_prefix"] == "sk-abc"
        assert error.details["key_type"] == "openai"
    
    def test_circuit_breaker_error_with_state_details(self):
        """Test CircuitBreakerError with state information"""
        error = CircuitBreakerError(
            "Circuit breaker is open",
            error_code="CIRCUIT_001",
            details={
                "failure_count": 5,
                "failure_threshold": 5,
                "state": "OPEN",
                "last_failure_time": 1234567890,
                "recovery_timeout": 60
            }
        )
        
        assert str(error) == "[CIRCUIT_001] Circuit breaker is open"
        assert error.details["failure_count"] == 5
        assert error.details["state"] == "OPEN"
    
    def test_retry_exhausted_error_with_attempt_details(self):
        """Test RetryExhaustedError with attempt information"""
        error = RetryExhaustedError(
            "All retry attempts failed",
            error_code="RETRY_001",
            details={
                "max_attempts": 3,
                "total_time": 15.5,
                "last_exception": "ConnectionError",
                "attempts": [
                    {"attempt": 1, "delay": 1.0, "error": "timeout"},
                    {"attempt": 2, "delay": 2.0, "error": "connection_reset"},
                    {"attempt": 3, "delay": 4.0, "error": "timeout"}
                ]
            }
        )
        
        assert str(error) == "[RETRY_001] All retry attempts failed"
        assert error.details["max_attempts"] == 3
        assert len(error.details["attempts"]) == 3


class TestExceptionChaining:
    """Test exception chaining and raising scenarios"""
    
    def test_exception_can_be_raised_and_caught(self):
        """Test that custom exceptions can be raised and caught"""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test validation error")
        
        assert str(exc_info.value) == "Test validation error"
        assert isinstance(exc_info.value, ValidationError)
        assert isinstance(exc_info.value, CMDRDataError)
    
    def test_exception_chaining_with_cause(self):
        """Test exception chaining with __cause__"""
        original_error = ValueError("Original error")
        
        try:
            raise original_error
        except ValueError as e:
            chained_error = TrackingError("Tracking failed due to validation error")
            chained_error.__cause__ = e
            
            with pytest.raises(TrackingError) as exc_info:
                raise chained_error
            
            assert str(exc_info.value) == "Tracking failed due to validation error"
            assert exc_info.value.__cause__ is original_error
    
    def test_multiple_exception_types_in_try_except(self):
        """Test catching multiple exception types"""
        def raise_different_errors(error_type):
            if error_type == "validation":
                raise ValidationError("Validation failed")
            elif error_type == "network":
                raise NetworkError("Network failed")
            elif error_type == "auth":
                raise AuthenticationError("Auth failed")
        
        # Test catching specific exception types
        with pytest.raises(ValidationError):
            raise_different_errors("validation")
        
        with pytest.raises(NetworkError):
            raise_different_errors("network")
        
        # Test catching base exception type
        with pytest.raises(CMDRDataError):
            raise_different_errors("auth")
    
    def test_exception_hierarchy_catching(self):
        """Test that exceptions can be caught by their base class"""
        specific_errors = [
            ValidationError("Validation error"),
            NetworkError("Network error"),
            AuthenticationError("Auth error"),
            TrackingError("Tracking error")
        ]
        
        for error in specific_errors:
            with pytest.raises(CMDRDataError):
                raise error


if __name__ == "__main__":
    pytest.main([__file__])