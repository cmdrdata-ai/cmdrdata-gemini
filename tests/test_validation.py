"""
Unit tests for input validation
"""

from unittest.mock import patch

import pytest

from cmdrdata_gemini.exceptions import SecurityError, ValidationError
from cmdrdata_gemini.security import APIKeyManager, InputSanitizer, validate_input


class TestAPIKeyManager:
    """Test suite for APIKeyManager"""

    def test_validate_api_key_openai_success(self):
        """Test successful OpenAI API key validation"""
        valid_key = "sk-" + "a" * 48  # Legacy format
        result = APIKeyManager.validate_api_key(valid_key, "openai")
        assert result["valid"] is True
        assert result["provider"] == "openai"

    def test_validate_api_key_cmdrdata_success(self):
        """Test successful cmdrdata API key validation"""
        valid_key = "tk-" + "a" * 32
        result = APIKeyManager.validate_api_key(valid_key, "cmdrdata")
        assert result["valid"] is True

    def test_validate_api_key_generic_success(self):
        """Test successful generic API key validation"""
        valid_key = "a" * 32
        result = APIKeyManager.validate_api_key(valid_key, "generic")
        assert result["valid"] is True

    def test_validate_api_key_empty_string(self):
        """Test API key validation with empty string"""
        with pytest.raises(ValidationError, match="API key must be a non-empty string"):
            APIKeyManager.validate_api_key("", "openai")

    def test_validate_api_key_none(self):
        """Test API key validation with None"""
        with pytest.raises(ValidationError, match="API key must be a non-empty string"):
            APIKeyManager.validate_api_key(None, "openai")

    def test_validate_api_key_suspicious_patterns(self):
        """Test API key validation with suspicious patterns"""
        malicious_keys = [
            "sk-<script>alert('xss')</script>" + "a" * 20,
            "sk-javascript:alert(1)" + "a" * 25,
            "sk-" + "a" * 40 + "\r\ninjection",
            "sk-" + "a" * 40 + "\x00null",
        ]

        for key in malicious_keys:
            with pytest.raises(
                SecurityError, match="API key contains suspicious pattern"
            ):
                APIKeyManager.validate_api_key(key, "openai")

    def test_validate_api_key_invalid_format(self):
        """Test API key validation with invalid format"""
        # Test length validation
        with pytest.raises(
            ValidationError, match="API key length is outside acceptable range"
        ):
            APIKeyManager.validate_api_key("short", "openai")

        # Test format validation with proper length
        with pytest.raises(ValidationError, match="Invalid openai API key format"):
            APIKeyManager.validate_api_key("wrong-prefix-" + "a" * 40, "openai")

    def test_validate_customer_id_success(self):
        """Test successful customer ID validation"""
        valid_ids = [
            "customer-123",
            "user_456",
            "tenant.789",
            "a1b2c3",
            "org-uuid-1234-5678",
        ]

        for customer_id in valid_ids:
            assert InputSanitizer.validate_customer_id(customer_id) is True

    def test_validate_customer_id_empty_string(self):
        """Test customer ID validation with empty string"""
        with pytest.raises(
            ValidationError, match="Customer ID must be a non-empty string"
        ):
            InputSanitizer.validate_customer_id("")

    def test_validate_customer_id_too_long(self):
        """Test customer ID validation with too long string"""
        long_id = "a" * 256
        with pytest.raises(
            ValidationError, match="Customer ID must be 255 characters or less"
        ):
            InputSanitizer.validate_customer_id(long_id)

    def test_validate_customer_id_suspicious_patterns(self):
        """Test customer ID validation with suspicious patterns"""
        malicious_ids = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "customer\r\ninjection",
            "customer\x00null",
        ]

        for customer_id in malicious_ids:
            with pytest.raises(
                SecurityError, match="Customer ID contains suspicious pattern"
            ):
                InputSanitizer.validate_customer_id(customer_id)

    def test_validate_customer_id_invalid_characters(self):
        """Test customer ID validation with invalid characters"""
        invalid_ids = [
            "customer@domain.com",
            "customer id with spaces",
            "customer/with/slashes",
            "customer#hash",
        ]

        for customer_id in invalid_ids:
            with pytest.raises(ValidationError, match="Customer ID can only contain"):
                InputSanitizer.validate_customer_id(customer_id)

    def test_validate_url_success(self):
        """Test successful URL validation"""
        valid_urls = [
            "https://api.example.com/endpoint",
            "http://localhost:8080/api",
            "https://subdomain.example.com:443/path",
        ]

        for url in valid_urls:
            assert InputSanitizer.validate_url(url) == url

    def test_validate_url_empty_string(self):
        """Test URL validation with empty string"""
        with pytest.raises(ValidationError, match="URL must be a non-empty string"):
            InputSanitizer.validate_url("")

    def test_validate_url_invalid_scheme(self):
        """Test URL validation with invalid scheme"""
        with pytest.raises(
            ValidationError, match="URL must use HTTP or HTTPS protocol"
        ):
            InputSanitizer.validate_url("ftp://example.com")

    def test_validate_url_suspicious_patterns(self):
        """Test URL validation with suspicious patterns"""
        malicious_urls = [
            "https://example.com/<script>alert('xss')</script>",
            "https://example.com/javascript:alert(1)",
            "https://example.com/path\r\ninjection",
        ]

        for url in malicious_urls:
            with pytest.raises(SecurityError, match="URL contains suspicious pattern"):
                InputSanitizer.validate_url(url)

    def test_validate_url_malformed(self):
        """Test URL validation with malformed URL"""
        with pytest.raises(ValidationError, match="Invalid URL format"):
            InputSanitizer.validate_url("not a valid url")

    def test_validate_timeout_success(self):
        """Test successful timeout validation"""
        valid_timeouts = [0.1, 1, 5.5, 30, 300]

        for timeout in valid_timeouts:
            assert InputSanitizer.validate_timeout(timeout) is True

    def test_validate_timeout_invalid_type(self):
        """Test timeout validation with invalid type"""
        with pytest.raises(ValidationError, match="Timeout must be a number"):
            InputSanitizer.validate_timeout("5")

    def test_validate_timeout_negative(self):
        """Test timeout validation with negative value"""
        with pytest.raises(ValidationError, match="Timeout must be positive"):
            InputSanitizer.validate_timeout(-1)

    def test_validate_timeout_too_large(self):
        """Test timeout validation with too large value"""
        with pytest.raises(ValidationError, match="Timeout cannot exceed 300 seconds"):
            InputSanitizer.validate_timeout(301)

    def test_validate_model_name_success(self):
        """Test successful model name validation"""
        valid_models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "custom_model_v1"]

        for model in valid_models:
            assert InputSanitizer.validate_model_name(model) is True

    def test_validate_model_name_empty_string(self):
        """Test model name validation with empty string"""
        with pytest.raises(
            ValidationError, match="Model name must be a non-empty string"
        ):
            InputSanitizer.validate_model_name("")

    def test_validate_model_name_suspicious_patterns(self):
        """Test model name validation with suspicious patterns"""
        malicious_models = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "gpt-4\r\ninjection",
        ]

        for model in malicious_models:
            with pytest.raises(
                SecurityError, match="Model name contains suspicious pattern"
            ):
                InputSanitizer.validate_model_name(model)

    def test_validate_model_name_invalid_characters(self):
        """Test model name validation with invalid characters"""
        invalid_models = ["model with spaces", "model@version", "model/version"]

        for model in invalid_models:
            with pytest.raises(ValidationError, match="Model name can only contain"):
                InputSanitizer.validate_model_name(model)

    def test_validate_token_count_success(self):
        """Test successful token count validation"""
        valid_counts = [0, 1, 100, 1000, 100000]

        for count in valid_counts:
            assert InputSanitizer.validate_token_count(count) is True

    def test_validate_token_count_invalid_type(self):
        """Test token count validation with invalid type"""
        with pytest.raises(ValidationError, match="Token count must be an integer"):
            InputSanitizer.validate_token_count(5.5)

    def test_validate_token_count_negative(self):
        """Test token count validation with negative value"""
        with pytest.raises(ValidationError, match="Token count cannot be negative"):
            InputSanitizer.validate_token_count(-1)

    def test_validate_token_count_too_large(self):
        """Test token count validation with too large value"""
        with pytest.raises(ValidationError, match="Token count exceeds maximum limit"):
            InputSanitizer.validate_token_count(1000001)

    def test_validate_metadata_success(self):
        """Test successful metadata validation"""
        valid_metadata = {
            "string_key": "string_value",
            "int_key": 123,
            "float_key": 45.6,
            "bool_key": True,
            "null_key": None,
        }

        assert InputSanitizer.validate_metadata(valid_metadata) is True

    def test_validate_metadata_invalid_type(self):
        """Test metadata validation with invalid type"""
        with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
            InputSanitizer.validate_metadata("not a dict")

    def test_validate_metadata_too_large(self):
        """Test metadata validation with too large data"""
        large_metadata = {"key": "a" * 10000}
        with pytest.raises(ValidationError, match="Metadata size exceeds limit"):
            InputSanitizer.validate_metadata(large_metadata)

    def test_validate_metadata_non_string_keys(self):
        """Test metadata validation with non-string keys"""
        with pytest.raises(ValidationError, match="Metadata keys must be strings"):
            InputSanitizer.validate_metadata({123: "value"})

    def test_validate_metadata_suspicious_keys(self):
        """Test metadata validation with suspicious keys"""
        malicious_metadata = {"<script>": "value"}
        with pytest.raises(
            SecurityError, match="Metadata key contains suspicious pattern"
        ):
            InputSanitizer.validate_metadata(malicious_metadata)

    def test_validate_metadata_suspicious_values(self):
        """Test metadata validation with suspicious values"""
        malicious_metadata = {"key": "<script>alert('xss')</script>"}
        with pytest.raises(
            SecurityError, match="Metadata value contains suspicious pattern"
        ):
            InputSanitizer.validate_metadata(malicious_metadata)

    def test_sanitize_string_basic(self):
        """Test basic string sanitization"""
        result = InputSanitizer.sanitize_string("hello world", "general_string")
        assert result == "hello world"

    def test_sanitize_string_with_null_bytes(self):
        """Test string sanitization with null bytes"""
        result = InputSanitizer.sanitize_string("hello\x00world", "general_string")
        assert result == "helloworld"

    def test_sanitize_string_truncation(self):
        """Test string sanitization with truncation"""
        long_string = "a" * 2000
        result = InputSanitizer.sanitize_string(long_string[:100], "general_string")
        assert len(result) == 100

    def test_sanitize_string_non_string_input(self):
        """Test string sanitization with non-string input"""
        with pytest.raises(ValidationError, match="Expected string, got"):
            InputSanitizer.sanitize_string(123, "general_string")

    def test_validate_chat_messages_success(self):
        """Test successful chat messages validation"""
        valid_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        assert InputSanitizer.validate_chat_messages(valid_messages) is True

    def test_validate_chat_messages_empty_list(self):
        """Test chat messages validation with empty list"""
        with pytest.raises(ValidationError, match="Messages list cannot be empty"):
            InputSanitizer.validate_chat_messages([])

    def test_validate_chat_messages_invalid_type(self):
        """Test chat messages validation with invalid type"""
        with pytest.raises(ValidationError, match="Messages must be a list"):
            InputSanitizer.validate_chat_messages("not a list")

    def test_validate_chat_messages_invalid_message_type(self):
        """Test chat messages validation with invalid message type"""
        with pytest.raises(ValidationError, match="Message 0 must be a dictionary"):
            InputSanitizer.validate_chat_messages(["not a dict"])

    def test_validate_chat_messages_missing_role(self):
        """Test chat messages validation with missing role"""
        messages = [{"content": "Hello!"}]
        with pytest.raises(
            ValidationError, match="Message 0 missing required 'role' field"
        ):
            InputSanitizer.validate_chat_messages(messages)

    def test_validate_chat_messages_invalid_role(self):
        """Test chat messages validation with invalid role"""
        messages = [{"role": "invalid", "content": "Hello!"}]
        with pytest.raises(ValidationError, match="Message 0 has invalid role"):
            InputSanitizer.validate_chat_messages(messages)

    def test_validate_chat_messages_missing_content(self):
        """Test chat messages validation with missing content"""
        messages = [{"role": "user"}]
        with pytest.raises(
            ValidationError, match="Message 0 missing required 'content' field"
        ):
            InputSanitizer.validate_chat_messages(messages)

    def test_validate_chat_messages_non_string_content(self):
        """Test chat messages validation with non-string content"""
        messages = [{"role": "user", "content": 123}]
        with pytest.raises(ValidationError, match="Message 0 content must be a string"):
            InputSanitizer.validate_chat_messages(messages)

    def test_validate_chat_messages_suspicious_content(self):
        """Test chat messages validation with suspicious content"""
        messages = [{"role": "user", "content": "<script>alert('xss')</script>"}]
        with pytest.raises(
            SecurityError, match="Message 0 content contains suspicious pattern"
        ):
            InputSanitizer.validate_chat_messages(messages)


class TestValidateInputDecorator:
    """Test suite for validate_input decorator"""

    def test_validate_input_decorator_success(self):
        """Test successful validation with decorator"""

        def validation_func(*args, **kwargs):
            # Mock validation that passes
            pass

        @validate_input(validation_func)
        def test_function(arg1, arg2):
            return "success"

        result = test_function("test1", "test2")
        assert result == "success"

    def test_validate_input_decorator_validation_error(self):
        """Test validation error with decorator"""

        def validation_func(*args, **kwargs):
            raise ValidationError("Validation failed")

        @validate_input(validation_func)
        def test_function(arg1, arg2):
            return "success"

        with pytest.raises(ValidationError, match="Validation failed"):
            test_function("test1", "test2")

    def test_validate_input_decorator_security_error(self):
        """Test security error with decorator"""

        def validation_func(*args, **kwargs):
            raise SecurityError("Security check failed")

        @validate_input(validation_func)
        def test_function(arg1, arg2):
            return "success"

        with pytest.raises(SecurityError, match="Security check failed"):
            test_function("test1", "test2")

    def test_validate_input_decorator_unexpected_error(self):
        """Test unexpected error with decorator"""

        def validation_func(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        @validate_input(validation_func)
        def test_function(arg1, arg2):
            return "success"

        with pytest.raises(ValidationError, match="Validation failed"):
            test_function("test1", "test2")


if __name__ == "__main__":
    pytest.main([__file__])
