"""
Security tests for cmdrdata-openai
"""

import pytest
import time
from unittest.mock import Mock, patch

from cmdrdata_gemini.security import (
    APIKeyManager, InputSanitizer, RateLimiter, SecurityConfig,
    require_valid_api_key, rate_limited, secure_compare,
    generate_secure_token, validate_request_signature
)
from cmdrdata_gemini.exceptions import SecurityError, ValidationError, AuthenticationError


class TestAPIKeyManager:
    """Test API key management and validation"""
    
    def test_validate_openai_api_key_success(self):
        """Test valid OpenAI API key validation"""
        # Test with legacy format
        legacy_key = "sk-" + "a" * 48
        result = APIKeyManager.validate_api_key(legacy_key, 'openai')
        assert result['valid'] is True
        assert result['provider'] == 'openai'
        assert result['format'] == 'legacy'
    
    def test_validate_cmdrdata_api_key_success(self):
        """Test valid cmdrdata API key validation"""
        valid_key = "tk-" + "a" * 32
        result = APIKeyManager.validate_api_key(valid_key, 'cmdrdata')
        assert result['valid'] is True
        assert result['provider'] == 'cmdrdata'
    
    def test_validate_api_key_empty_string(self):
        """Test validation with empty string"""
        with pytest.raises(ValidationError, match="API key must be a non-empty string"):
            APIKeyManager.validate_api_key("", 'openai')
    
    def test_validate_api_key_none(self):
        """Test validation with None"""
        with pytest.raises(ValidationError, match="API key must be a non-empty string"):
            APIKeyManager.validate_api_key(None, 'openai')
    
    def test_validate_api_key_too_short(self):
        """Test validation with too short key"""
        with pytest.raises(ValidationError, match="API key length is outside acceptable range"):
            APIKeyManager.validate_api_key("short", 'openai')
    
    def test_validate_api_key_too_long(self):
        """Test validation with too long key"""
        long_key = "a" * 501
        with pytest.raises(ValidationError, match="API key length is outside acceptable range"):
            APIKeyManager.validate_api_key(long_key, 'openai')
    
    def test_validate_api_key_suspicious_patterns(self):
        """Test detection of suspicious patterns"""
        malicious_keys = [
            "sk-<script>alert('xss')</script>",
            "sk-javascript:alert(1)",
            "sk-data:text/html,<script>alert(1)</script>",
            "sk-test\r\ninjection",
            "sk-test\x00null"
        ]
        
        for key in malicious_keys:
            with pytest.raises(SecurityError, match="API key contains suspicious pattern"):
                APIKeyManager.validate_api_key(key, 'openai')
    
    def test_validate_api_key_invalid_format(self):
        """Test validation with invalid format"""
        invalid_key = "invalid-key-format-123456789012345678901234567890"
        with pytest.raises(ValidationError, match="Invalid openai API key format"):
            APIKeyManager.validate_api_key(invalid_key, 'openai')
    
    def test_sanitize_api_key_for_logging(self):
        """Test API key sanitization for logging"""
        key = "sk-abcdefghijklmnopqrstuvwxyz1234567890123456"
        sanitized = APIKeyManager.sanitize_api_key_for_logging(key)
        assert sanitized == "sk-...456"
        
        # Test with empty key
        assert APIKeyManager.sanitize_api_key_for_logging("") == "[EMPTY]"
        
        # Test with short key
        assert APIKeyManager.sanitize_api_key_for_logging("short") == "[REDACTED]"
    
    def test_generate_tracking_key(self):
        """Test tracking key generation"""
        key = APIKeyManager.generate_tracking_key()
        assert key.startswith("tk-")
        assert len(key) == 67  # tk- + 64 hex chars
        
        # Test uniqueness
        key2 = APIKeyManager.generate_tracking_key()
        assert key != key2
    
    def test_hash_and_verify_api_key(self):
        """Test API key hashing and verification"""
        api_key = "sk-test-key-123456789012345678901234567890"
        
        # Hash the key
        hashed = APIKeyManager.hash_api_key(api_key)
        assert ":" in hashed
        
        # Verify correct key
        assert APIKeyManager.verify_api_key_hash(api_key, hashed) is True
        
        # Verify wrong key
        wrong_key = "sk-wrong-key-123456789012345678901234567890"
        assert APIKeyManager.verify_api_key_hash(wrong_key, hashed) is False
        
        # Test with invalid hash format
        assert APIKeyManager.verify_api_key_hash(api_key, "invalid-hash") is False


class TestInputSanitizer:
    """Test input sanitization and validation"""
    
    def test_sanitize_string_basic(self):
        """Test basic string sanitization"""
        result = InputSanitizer.sanitize_string("hello world", 'general_string')
        assert result == "hello world"
    
    def test_sanitize_string_remove_suspicious(self):
        """Test removal of suspicious patterns"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "test\r\ninjection",
            "test\x00null"
        ]
        
        for input_str in malicious_inputs:
            result = InputSanitizer.sanitize_string(input_str, 'general_string', strict=False)
            assert "<script>" not in result
            assert "javascript:" not in result
            assert "\r\n" not in result
            assert "\x00" not in result
    
    def test_sanitize_string_strict_mode(self):
        """Test strict mode raises exceptions"""
        with pytest.raises(SecurityError, match="Input contains suspicious pattern"):
            InputSanitizer.sanitize_string("<script>alert('xss')</script>", 'general_string', strict=True)
    
    def test_sanitize_string_length_limit(self):
        """Test string length limiting"""
        long_string = "a" * 2000
        result = InputSanitizer.sanitize_string(long_string, 'general_string')
        assert len(result) == InputSanitizer.MAX_LENGTHS['general_string']
    
    def test_sanitize_string_strict_length_limit(self):
        """Test strict length limit raises exception"""
        long_string = "a" * 2000
        with pytest.raises(ValidationError, match="Input too long"):
            InputSanitizer.sanitize_string(long_string, 'general_string', strict=True)
    
    def test_sanitize_string_pattern_validation(self):
        """Test pattern validation"""
        # Valid customer ID
        result = InputSanitizer.sanitize_string("customer-123", 'customer_id')
        assert result == "customer-123"
        
        # Invalid characters in customer ID (non-strict)
        result = InputSanitizer.sanitize_string("customer@123", 'customer_id', strict=False)
        assert "@" not in result
        
        # Invalid characters in customer ID (strict)
        with pytest.raises(ValidationError, match="Input does not match required pattern"):
            InputSanitizer.sanitize_string("customer@123", 'customer_id', strict=True)
    
    def test_sanitize_string_non_string_input(self):
        """Test handling of non-string input"""
        with pytest.raises(ValidationError, match="Expected string, got"):
            InputSanitizer.sanitize_string(123, 'general_string')
    
    def test_validate_url_valid(self):
        """Test valid URL validation"""
        valid_urls = [
            "https://api.example.com/endpoint",
            "http://localhost:8080/api",
            "https://subdomain.example.com:443/path"
        ]
        
        for url in valid_urls:
            result = InputSanitizer.validate_url(url)
            assert result == url
    
    def test_validate_url_invalid_scheme(self):
        """Test URL validation with invalid scheme"""
        with pytest.raises(ValidationError, match="URL must use HTTP or HTTPS protocol"):
            InputSanitizer.validate_url("ftp://example.com")
    
    def test_validate_url_custom_schemes(self):
        """Test URL validation with custom allowed schemes"""
        result = InputSanitizer.validate_url("ftp://example.com", allowed_schemes={'ftp', 'https'})
        assert result == "ftp://example.com"
    
    def test_validate_url_suspicious_patterns(self):
        """Test URL validation with suspicious patterns"""
        malicious_urls = [
            "https://example.com/<script>alert('xss')</script>",
            "https://example.com/javascript:alert(1)",
            "https://example.com/path\r\ninjection"
        ]
        
        for url in malicious_urls:
            with pytest.raises(SecurityError, match="URL contains suspicious pattern"):
                InputSanitizer.validate_url(url)
    
    def test_validate_url_too_long(self):
        """Test URL validation with too long URL"""
        long_url = "https://example.com/" + "a" * 2048
        with pytest.raises(ValidationError, match="URL too long"):
            InputSanitizer.validate_url(long_url)
    
    def test_validate_url_private_ip_production(self):
        """Test URL validation blocks private IPs in production"""
        with patch.dict('os.environ', {'CMDRDATA_ENVIRONMENT': 'production'}):
            with pytest.raises(SecurityError, match="Private IP addresses not allowed"):
                InputSanitizer.validate_url("https://192.168.1.1/api")
    
    def test_sanitize_metadata_valid(self):
        """Test valid metadata sanitization"""
        metadata = {
            "key1": "value1",
            "key2": 123,
            "key3": True,
            "key4": None
        }
        
        result = InputSanitizer.sanitize_metadata(metadata)
        assert result == metadata
    
    def test_sanitize_metadata_invalid_type(self):
        """Test metadata sanitization with invalid type"""
        with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
            InputSanitizer.sanitize_metadata("not a dict")
    
    def test_sanitize_metadata_non_string_keys(self):
        """Test metadata sanitization with non-string keys"""
        with pytest.raises(ValidationError, match="Metadata keys must be strings"):
            InputSanitizer.sanitize_metadata({123: "value"})
    
    def test_sanitize_metadata_too_large(self):
        """Test metadata sanitization with too large data"""
        large_metadata = {"key": "a" * 10000}
        with pytest.raises(ValidationError, match="Metadata size exceeds limit"):
            InputSanitizer.sanitize_metadata(large_metadata)
    
    def test_sanitize_metadata_suspicious_content(self):
        """Test metadata sanitization with suspicious content"""
        metadata = {
            "key": "<script>alert('xss')</script>",
            "another": "normal value"
        }
        
        result = InputSanitizer.sanitize_metadata(metadata)
        assert "<script>" not in result["key"]
        assert result["another"] == "normal value"


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # Should allow first 3 requests
        for i in range(3):
            assert limiter.is_allowed("test_user") is True
        
        # Should deny 4th request
        assert limiter.is_allowed("test_user") is False
    
    def test_rate_limiter_different_identifiers(self):
        """Test rate limiter with different identifiers"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Should allow requests for different users
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user2") is True
        
        # Should deny after limit for each user
        assert limiter.is_allowed("user1") is False
        assert limiter.is_allowed("user2") is False
    
    def test_rate_limiter_window_reset(self):
        """Test rate limiter window reset"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # Use up the limit
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is True
        assert limiter.is_allowed("test_user") is False
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should allow requests again
        assert limiter.is_allowed("test_user") is True
    
    def test_rate_limiter_get_reset_time(self):
        """Test getting rate limit reset time"""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        
        # No requests yet
        assert limiter.get_reset_time("test_user") is None
        
        # Make a request
        assert limiter.is_allowed("test_user") is True
        reset_time = limiter.get_reset_time("test_user")
        assert reset_time is not None
        assert reset_time > time.time()


class TestSecurityDecorators:
    """Test security decorators"""
    
    def test_require_valid_api_key_decorator(self):
        """Test API key validation decorator"""
        @require_valid_api_key('openai')
        def test_function(api_key=None):
            return "success"
        
        # Should work with valid key
        valid_key = "sk-" + "a" * 48
        result = test_function(api_key=valid_key)
        assert result == "success"
        
        # Should fail with invalid key
        with pytest.raises(AuthenticationError):
            test_function(api_key="invalid-key")
        
        # Should fail with no key
        with pytest.raises(AuthenticationError):
            test_function()
    
    def test_rate_limited_decorator(self):
        """Test rate limiting decorator"""
        @rate_limited(max_requests=2, window_seconds=60)
        def test_function(identifier="default"):
            return "success"
        
        # Should allow first 2 requests
        assert test_function() == "success"
        assert test_function() == "success"
        
        # Should deny 3rd request
        with pytest.raises(SecurityError, match="Rate limit exceeded"):
            test_function()


class TestSecurityUtilities:
    """Test security utility functions"""
    
    def test_secure_compare(self):
        """Test secure string comparison"""
        assert secure_compare("hello", "hello") is True
        assert secure_compare("hello", "world") is False
        assert secure_compare("", "") is True
    
    def test_generate_secure_token(self):
        """Test secure token generation"""
        token = generate_secure_token()
        assert len(token) == 64  # 32 bytes = 64 hex chars
        assert all(c in '0123456789abcdef' for c in token)
        
        # Test custom length
        token = generate_secure_token(16)
        assert len(token) == 32  # 16 bytes = 32 hex chars
        
        # Test uniqueness
        token1 = generate_secure_token()
        token2 = generate_secure_token()
        assert token1 != token2
    
    def test_validate_request_signature(self):
        """Test request signature validation"""
        secret = "test-secret-key"
        request_body = b"test request body"
        
        # Generate valid signature
        import hashlib
        import hmac
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            request_body,
            hashlib.sha256
        ).hexdigest()
        
        # Should validate correct signature
        assert validate_request_signature(request_body, expected_signature, secret) is True
        
        # Should reject wrong signature
        assert validate_request_signature(request_body, "wrong-signature", secret) is False
        
        # Should reject wrong secret
        assert validate_request_signature(request_body, expected_signature, "wrong-secret") is False


class TestSecurityConfig:
    """Test security configuration"""
    
    def test_security_config_defaults(self):
        """Test security configuration defaults"""
        config = SecurityConfig()
        
        assert config.max_request_size == 1048576  # 1MB
        assert config.rate_limit_window == 60
        assert config.max_requests_per_window == 100
        assert config.allowed_origins == ['*']
        assert config.require_https is True
        assert config.api_key_rotation_days == 90
    
    def test_security_config_environment_override(self):
        """Test security configuration from environment"""
        with patch.dict('os.environ', {
            'CMDRDATA_MAX_REQUEST_SIZE': '2097152',
            'CMDRDATA_RATE_LIMIT_WINDOW': '120',
            'CMDRDATA_MAX_REQUESTS_PER_WINDOW': '200',
            'CMDRDATA_REQUIRE_HTTPS': 'false'
        }):
            config = SecurityConfig()
            assert config.max_request_size == 2097152
            assert config.rate_limit_window == 120
            assert config.max_requests_per_window == 200
            assert config.require_https is False


if __name__ == "__main__":
    pytest.main([__file__])