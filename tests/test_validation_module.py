"""
Unit tests for validation module
"""

import pytest

from cmdrdata_gemini.exceptions import SecurityError, ValidationError
from cmdrdata_gemini.validation import InputValidator, validate_input


class TestInputValidator:
    """Test suite for InputValidator"""

    def test_validate_api_key_openai_success(self):
        """Test successful OpenAI API key validation"""
        valid_key = "sk-" + "a" * 48
        assert InputValidator.validate_api_key(valid_key, "openai") is True

    def test_validate_api_key_cmdrdata_success(self):
        """Test successful cmdrdata API key validation"""
        valid_key = "tk-" + "a" * 32
        assert InputValidator.validate_api_key(valid_key, "cmdrdata") is True

    def test_validate_api_key_generic_success(self):
        """Test successful generic API key validation"""
        valid_key = "a" * 32
        assert InputValidator.validate_api_key(valid_key, "generic") is True

    def test_validate_api_key_empty_string(self):
        """Test API key validation with empty string"""
        with pytest.raises(ValidationError, match="API key must be a non-empty string"):
            InputValidator.validate_api_key("", "openai")

    def test_validate_api_key_none(self):
        """Test API key validation with None"""
        with pytest.raises(ValidationError, match="API key must be a non-empty string"):
            InputValidator.validate_api_key(None, "openai")

    def test_validate_api_key_suspicious_patterns(self):
        """Test API key validation with suspicious patterns"""
        malicious_keys = [
            "sk-<script>alert('xss')</script>" + "a" * 20,
            "sk-javascript:alert(1)" + "a" * 25,
            "sk-data:text/html" + "a" * 30,
            "sk-vbscript:msgbox" + "a" * 28,
            'sk-onclick="alert(1)"' + "a" * 25,
            "sk-expression(alert)" + "a" * 27,
        ]
        for key in malicious_keys:
            with pytest.raises(
                SecurityError, match="API key contains suspicious pattern"
            ):
                InputValidator.validate_api_key(key, "openai")

    def test_validate_api_key_invalid_format(self):
        """Test API key validation with invalid format"""
        with pytest.raises(ValidationError, match="Invalid openai API key format"):
            InputValidator.validate_api_key("invalid-key", "openai")

    def test_validate_customer_id_success(self):
        """Test successful customer ID validation"""
        valid_ids = ["customer123", "user_456", "org.789", "test-customer"]
        for customer_id in valid_ids:
            assert InputValidator.validate_customer_id(customer_id) is True

    def test_validate_customer_id_empty(self):
        """Test customer ID validation with empty string"""
        with pytest.raises(
            ValidationError, match="Customer ID must be a non-empty string"
        ):
            InputValidator.validate_customer_id("")

    def test_validate_customer_id_none(self):
        """Test customer ID validation with None"""
        with pytest.raises(
            ValidationError, match="Customer ID must be a non-empty string"
        ):
            InputValidator.validate_customer_id(None)

    def test_validate_customer_id_too_long(self):
        """Test customer ID validation with too long string"""
        long_id = "a" * 256
        with pytest.raises(
            ValidationError, match="Customer ID must be 255 characters or less"
        ):
            InputValidator.validate_customer_id(long_id)

    def test_validate_customer_id_suspicious_patterns(self):
        """Test customer ID validation with suspicious patterns"""
        malicious_ids = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            'onclick="alert(1)"',
        ]
        for customer_id in malicious_ids:
            with pytest.raises(
                SecurityError, match="Customer ID contains suspicious pattern"
            ):
                InputValidator.validate_customer_id(customer_id)

    def test_validate_customer_id_invalid_characters(self):
        """Test customer ID validation with invalid characters"""
        invalid_ids = ["customer@123", "user#456", "org/789", "test customer"]
        for customer_id in invalid_ids:
            with pytest.raises(ValidationError, match="Customer ID can only contain"):
                InputValidator.validate_customer_id(customer_id)

    def test_validate_url_success(self):
        """Test successful URL validation"""
        valid_urls = [
            "https://api.openai.com",
            "http://localhost:8080",
            "https://example.com/api/v1/endpoint",
        ]
        for url in valid_urls:
            assert InputValidator.validate_url(url) is True

    def test_validate_url_empty(self):
        """Test URL validation with empty string"""
        with pytest.raises(ValidationError, match="URL must be a non-empty string"):
            InputValidator.validate_url("")

    def test_validate_url_none(self):
        """Test URL validation with None"""
        with pytest.raises(ValidationError, match="URL must be a non-empty string"):
            InputValidator.validate_url(None)

    def test_validate_url_invalid_scheme(self):
        """Test URL validation with invalid scheme"""
        invalid_urls = [
            "ftp://example.com",
            "file:///etc/passwd",
            "javascript:alert(1)",
        ]
        for url in invalid_urls:
            with pytest.raises(
                ValidationError, match="URL must use HTTP or HTTPS protocol"
            ):
                InputValidator.validate_url(url)

    def test_validate_url_malformed(self):
        """Test URL validation with malformed URL"""
        with pytest.raises(
            ValidationError, match="URL must use HTTP or HTTPS protocol"
        ):
            InputValidator.validate_url("not a valid url")

    def test_validate_url_suspicious_patterns(self):
        """Test URL validation with suspicious patterns"""
        malicious_urls = [
            "https://example.com/<script>alert('xss')</script>",
            "https://example.com/javascript:alert(1)",
        ]
        for url in malicious_urls:
            with pytest.raises(SecurityError, match="URL contains suspicious pattern"):
                InputValidator.validate_url(url)

    def test_validate_timeout_success(self):
        """Test successful timeout validation"""
        valid_timeouts = [1, 10.5, 60, 299.9]
        for timeout in valid_timeouts:
            assert InputValidator.validate_timeout(timeout) is True

    def test_validate_timeout_invalid_type(self):
        """Test timeout validation with invalid type"""
        with pytest.raises(ValidationError, match="Timeout must be a number"):
            InputValidator.validate_timeout("30")

    def test_validate_timeout_negative(self):
        """Test timeout validation with negative value"""
        with pytest.raises(ValidationError, match="Timeout must be positive"):
            InputValidator.validate_timeout(-1)

    def test_validate_timeout_zero(self):
        """Test timeout validation with zero value"""
        with pytest.raises(ValidationError, match="Timeout must be positive"):
            InputValidator.validate_timeout(0)

    def test_validate_timeout_too_large(self):
        """Test timeout validation with too large value"""
        with pytest.raises(ValidationError, match="Timeout cannot exceed 300 seconds"):
            InputValidator.validate_timeout(301)

    def test_validate_model_name_success(self):
        """Test successful model name validation"""
        valid_models = [
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "text-davinci-003",
            "claude-3-opus",
        ]
        for model in valid_models:
            assert InputValidator.validate_model_name(model) is True

    def test_validate_model_name_empty(self):
        """Test model name validation with empty string"""
        with pytest.raises(
            ValidationError, match="Model name must be a non-empty string"
        ):
            InputValidator.validate_model_name("")

    def test_validate_model_name_none(self):
        """Test model name validation with None"""
        with pytest.raises(
            ValidationError, match="Model name must be a non-empty string"
        ):
            InputValidator.validate_model_name(None)

    def test_validate_model_name_suspicious_patterns(self):
        """Test model name validation with suspicious patterns"""
        malicious_models = [
            "gemini-2.5<script>alert('xss')</script>",
            "javascript:alert(1)",
        ]
        for model in malicious_models:
            with pytest.raises(
                SecurityError, match="Model name contains suspicious pattern"
            ):
                InputValidator.validate_model_name(model)

    def test_validate_model_name_invalid_characters(self):
        """Test model name validation with invalid characters"""
        invalid_models = ["gemini@2.5", "gemini#1.5", "text/davinci", "gemini 2.5"]
        for model in invalid_models:
            with pytest.raises(ValidationError, match="Model name can only contain"):
                InputValidator.validate_model_name(model)

    def test_validate_token_count_success(self):
        """Test successful token count validation"""
        valid_counts = [0, 1, 100, 1000, 999999]
        for count in valid_counts:
            assert InputValidator.validate_token_count(count) is True

    def test_validate_token_count_invalid_type(self):
        """Test token count validation with invalid type"""
        with pytest.raises(ValidationError, match="Token count must be an integer"):
            InputValidator.validate_token_count(10.5)

    def test_validate_token_count_negative(self):
        """Test token count validation with negative value"""
        with pytest.raises(ValidationError, match="Token count cannot be negative"):
            InputValidator.validate_token_count(-1)

    def test_validate_token_count_too_large(self):
        """Test token count validation with too large value"""
        with pytest.raises(ValidationError, match="Token count exceeds maximum limit"):
            InputValidator.validate_token_count(1000001)

    def test_validate_metadata_success(self):
        """Test successful metadata validation"""
        valid_metadata = {
            "user_id": "123",
            "session": "abc-def",
            "version": "1.0",
        }
        assert InputValidator.validate_metadata(valid_metadata) is True

    def test_validate_metadata_not_dict(self):
        """Test metadata validation with non-dict"""
        with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
            InputValidator.validate_metadata("not a dict")

    def test_validate_metadata_too_large(self):
        """Test metadata validation with too large data"""
        large_metadata = {"key": "x" * 20000}
        with pytest.raises(ValidationError, match="Metadata size exceeds limit"):
            InputValidator.validate_metadata(large_metadata)

    def test_validate_metadata_non_string_keys(self):
        """Test metadata validation with non-string keys"""
        invalid_metadata = {123: "value"}
        with pytest.raises(ValidationError, match="Metadata keys must be strings"):
            InputValidator.validate_metadata(invalid_metadata)

    def test_validate_metadata_suspicious_keys(self):
        """Test metadata validation with suspicious keys"""
        malicious_metadata = {"<script>alert('xss')</script>": "value"}
        with pytest.raises(
            SecurityError, match="Metadata key contains suspicious pattern"
        ):
            InputValidator.validate_metadata(malicious_metadata)

    def test_validate_metadata_suspicious_values(self):
        """Test metadata validation with suspicious values"""
        malicious_metadata = {"key": "<script>alert('xss')</script>"}
        with pytest.raises(
            SecurityError, match="Metadata value contains suspicious pattern"
        ):
            InputValidator.validate_metadata(malicious_metadata)

    def test_sanitize_string_success(self):
        """Test successful string sanitization"""
        assert InputValidator.sanitize_string("hello world") == "hello world"
        assert InputValidator.sanitize_string(123) == "123"
        assert InputValidator.sanitize_string("hello\x00world") == "helloworld"

    def test_sanitize_string_truncation(self):
        """Test string sanitization with truncation"""
        long_string = "a" * 2000
        result = InputValidator.sanitize_string(long_string, max_length=1000)
        assert len(result) == 1000
        assert result == "a" * 1000

    def test_validate_chat_messages_success(self):
        """Test successful chat messages validation"""
        valid_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert InputValidator.validate_chat_messages(valid_messages) is True

    def test_validate_chat_messages_not_list(self):
        """Test chat messages validation with non-list"""
        with pytest.raises(ValidationError, match="Messages must be a list"):
            InputValidator.validate_chat_messages("not a list")

    def test_validate_chat_messages_empty(self):
        """Test chat messages validation with empty list"""
        with pytest.raises(ValidationError, match="Messages list cannot be empty"):
            InputValidator.validate_chat_messages([])

    def test_validate_chat_messages_invalid_format(self):
        """Test chat messages validation with invalid format"""
        invalid_messages = ["not a dict"]
        with pytest.raises(ValidationError, match="Message 0 must be a dictionary"):
            InputValidator.validate_chat_messages(invalid_messages)

    def test_validate_chat_messages_missing_role(self):
        """Test chat messages validation with missing role"""
        messages = [{"content": "Hello"}]
        with pytest.raises(
            ValidationError, match="Message 0 missing required 'role' field"
        ):
            InputValidator.validate_chat_messages(messages)

    def test_validate_chat_messages_invalid_role(self):
        """Test chat messages validation with invalid role"""
        messages = [{"role": "invalid", "content": "Hello"}]
        with pytest.raises(ValidationError, match="Message 0 has invalid role"):
            InputValidator.validate_chat_messages(messages)

    def test_validate_chat_messages_missing_content(self):
        """Test chat messages validation with missing content"""
        messages = [{"role": "user"}]
        with pytest.raises(
            ValidationError, match="Message 0 missing required 'content' field"
        ):
            InputValidator.validate_chat_messages(messages)

    def test_validate_chat_messages_invalid_content_type(self):
        """Test chat messages validation with invalid content type"""
        messages = [{"role": "user", "content": 123}]
        with pytest.raises(ValidationError, match="Message 0 content must be a string"):
            InputValidator.validate_chat_messages(messages)

    def test_validate_chat_messages_suspicious_content(self):
        """Test chat messages validation with suspicious content"""
        messages = [{"role": "user", "content": "<script>alert('xss')</script>"}]
        with pytest.raises(
            SecurityError, match="Message 0 content contains suspicious pattern"
        ):
            InputValidator.validate_chat_messages(messages)


class TestValidateInputDecorator:
    """Test suite for validate_input decorator"""

    def test_decorator_success(self):
        """Test decorator with successful validation"""

        def validation_func(value):
            if value < 0:
                raise ValidationError("Value must be positive")

        @validate_input(validation_func)
        def process_value(value):
            return value * 2

        assert process_value(5) == 10

    def test_decorator_validation_error(self):
        """Test decorator with validation error"""

        def validation_func(value):
            if value < 0:
                raise ValidationError("Value must be positive")

        @validate_input(validation_func)
        def process_value(value):
            return value * 2

        with pytest.raises(ValidationError, match="Value must be positive"):
            process_value(-5)

    def test_decorator_security_error(self):
        """Test decorator with security error"""

        def validation_func(value):
            if "script" in value:
                raise SecurityError("Suspicious content detected")

        @validate_input(validation_func)
        def process_value(value):
            return value.upper()

        with pytest.raises(SecurityError, match="Suspicious content detected"):
            process_value("<script>")

    def test_decorator_unexpected_error(self):
        """Test decorator with unexpected error"""

        def validation_func(value):
            raise RuntimeError("Unexpected error")

        @validate_input(validation_func)
        def process_value(value):
            return value

        with pytest.raises(
            ValidationError, match="Validation failed: Unexpected error"
        ):
            process_value("test")
