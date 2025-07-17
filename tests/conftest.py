"""
Pytest configuration and shared fixtures
"""

import pytest
import os
import logging
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    response = Mock()
    response.model = "gpt-4"
    response.id = "chatcmpl-test123"
    response.created = 1234567890
    response.system_fingerprint = "fp_test"
    
    # Mock usage
    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 15
    response.usage.total_tokens = 25
    
    # Mock choices
    choice = Mock()
    choice.finish_reason = "stop"
    choice.message = Mock()
    choice.message.content = "Hello! How can I help you today?"
    choice.message.role = "assistant"
    response.choices = [choice]
    
    return response


@pytest.fixture
def mock_completion_response():
    """Mock OpenAI completion (legacy) response"""
    response = Mock()
    response.model = "text-davinci-003"
    response.id = "cmpl-test123"
    response.created = 1234567890
    
    # Mock usage
    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 15
    response.usage.total_tokens = 25
    
    # Mock choices
    choice = Mock()
    choice.finish_reason = "stop"
    choice.text = "This is a completion response."
    response.choices = [choice]
    
    return response


@pytest.fixture
def valid_api_keys():
    """Valid API keys for testing"""
    return {
        'google': "AIza" + "A" * 35,
        'cmdrdata': "tk-" + "C" * 32,
        'generic': "x" * 32
    }


@pytest.fixture
def invalid_api_keys():
    """Invalid API keys for testing"""
    return {
        'too_short': "AIza-short",
        'wrong_prefix': "wrong-" + "a" * 40,
        'malicious': "AIza<script>alert('xss')</script>" + "a" * 20,
        'with_injection': "AIza" + "a" * 30 + "\r\ninjected",
        'empty': "",
        'none': None
    }


@pytest.fixture
def sample_customer_ids():
    """Sample customer IDs for testing"""
    return {
        'valid': [
            "customer-123",
            "user_456",
            "tenant.789",
            "org-uuid-1234-5678",
            "a1b2c3"
        ],
        'invalid': [
            "customer@domain.com",
            "customer id with spaces",
            "customer/with/slashes",
            "customer#hash",
            "<script>alert('xss')</script>",
            "customer\r\ninjection"
        ]
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {
        'valid': {
            "request_id": "req-123",
            "user_agent": "test-client/1.0",
            "session_id": "session-456",
            "feature_flag": True,
            "retry_count": 2
        },
        'invalid': {
            "malicious_key": "<script>alert('xss')</script>",
            "injection_value": "normal_key"
        },
        'large': {
            "large_field": "x" * 5000  # Too large
        }
    }


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing"""
    return {
        'valid': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
            {"role": "user", "content": "What's the weather like?"}
        ],
        'invalid_role': [
            {"role": "invalid_role", "content": "Hello!"}
        ],
        'missing_content': [
            {"role": "user"}
        ],
        'malicious_content': [
            {"role": "user", "content": "<script>alert('xss')</script>"}
        ]
    }


@pytest.fixture
def mock_tracker():
    """Mock UsageTracker instance"""
    tracker = Mock()
    tracker.track_usage.return_value = True
    tracker.track_usage_async.return_value = True
    tracker.track_usage_background.return_value = None
    tracker.get_health_status.return_value = {
        "healthy": True,
        "endpoint": "https://api.example.com",
        "timeout": 5.0
    }
    return tracker


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = Mock()
    
    # Mock chat completions
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock()
    
    # Mock legacy completions
    client.completions = Mock()
    client.completions.create = Mock()
    
    # Mock other endpoints
    client.models = Mock()
    client.models.list = Mock()
    client.files = Mock()
    client.files.list = Mock()
    
    return client


@pytest.fixture
def mock_async_openai_client():
    """Mock AsyncOpenAI client"""
    from unittest.mock import AsyncMock
    
    client = Mock()
    
    # Mock chat completions
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = AsyncMock()
    
    # Mock legacy completions
    client.completions = Mock()
    client.completions.create = AsyncMock()
    
    # Mock other endpoints
    client.models = Mock()
    client.models.list = AsyncMock()
    client.files = Mock()
    client.files.list = AsyncMock()
    
    return client


@pytest.fixture
def temp_log_file(tmp_path):
    """Temporary log file for testing"""
    log_file = tmp_path / "test.log"
    return str(log_file)


@pytest.fixture
def env_vars():
    """Environment variables for testing"""
    return {
        'OPENAI_API_KEY': 'sk-test-openai-key-123456789012345678901234567890',
        'CMDRDATA_API_KEY': 'tk-test-cmdrdata-key-123456789012345678901234',
        'CMDRDATA_LOG_LEVEL': 'DEBUG',
        'CMDRDATA_LOG_FORMAT': 'structured',
        'CMDRDATA_SECURITY_MODE': 'true'
    }


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing"""
    fixed_time = datetime(2023, 1, 1, 12, 0, 0)
    
    with patch('cmdrdata_gemini.tracker.datetime') as mock_dt:
        mock_dt.utcnow.return_value = fixed_time
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield mock_dt


@pytest.fixture
def network_errors():
    """Common network errors for testing"""
    import requests
    try:
        import httpx
        return [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timed out"),
            requests.exceptions.RequestException("Request failed"),
            httpx.ConnectError("HTTPX connection failed"),
            httpx.TimeoutException("HTTPX timeout"),
            httpx.RequestError("HTTPX request failed")
        ]
    except ImportError:
        return [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timed out"),
            requests.exceptions.RequestException("Request failed")
        ]


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Cleanup logging configuration after each test"""
    yield
    
    # Clear handlers from our loggers
    for logger_name in ['cmdrdata_gemini', 'cmdrdata_gemini.test']:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test"""
    original_env = os.environ.copy()
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def performance_config():
    """Performance configuration for testing"""
    return {
        'cache_size': 100,
        'cache_ttl': timedelta(minutes=5),
        'max_connections': 10,
        'request_timeout': 30.0,
        'rate_limit': 1000,
        'rate_window': 60
    }


@pytest.fixture
def retry_config():
    """Retry configuration for testing"""
    from cmdrdata_gemini.retry import RetryConfig, RetryPolicy
    
    return RetryConfig(
        max_attempts=3,
        initial_delay=0.01,  # Fast for testing
        max_delay=1.0,
        policy=RetryPolicy.EXPONENTIAL_BACKOFF,
        jitter=False  # Disable for predictable testing
    )


@pytest.fixture
def circuit_breaker_config():
    """Circuit breaker configuration for testing"""
    from cmdrdata_gemini.retry import CircuitBreaker
    
    return CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=0.1,  # Fast recovery for testing
        expected_exception=Exception
    )


# Pytest markers for different test types
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Pytest collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test file names
    for item in items:
        # Mark unit tests
        if "test_unit" in item.nodeid or "/unit/" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark integration tests
        if "test_integration" in item.nodeid or "/integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "test_performance" in item.nodeid or "/performance/" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Mark security tests
        if "test_security" in item.nodeid or "/security/" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        # Mark slow tests
        if "test_slow" in item.nodeid or item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)