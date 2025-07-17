"""
Unit tests for retry logic and circuit breaker
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini.exceptions import (
    CircuitBreakerError,
    NetworkError,
    RetryExhaustedError,
    TimeoutError,
)
from cmdrdata_gemini.retry import (
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
    DEFAULT_CIRCUIT_BREAKER,
    DEFAULT_RETRY_CONFIG,
    CircuitBreaker,
    CircuitBreakerState,
    RetryConfig,
    RetryPolicy,
    retry_with_backoff,
)


class TestRetryConfig:
    """Test suite for RetryConfig"""

    def test_retry_config_initialization_defaults(self):
        """Test RetryConfig initialization with defaults"""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.policy == RetryPolicy.EXPONENTIAL_BACKOFF
        assert Exception in config.retryable_exceptions

    def test_retry_config_initialization_custom(self):
        """Test RetryConfig initialization with custom values"""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            policy=RetryPolicy.FIXED_INTERVAL,
            retryable_exceptions=[ConnectionError, TimeoutError],
        )

        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.policy == RetryPolicy.FIXED_INTERVAL
        assert config.retryable_exceptions == [ConnectionError, TimeoutError]

    def test_calculate_delay_fixed_interval(self):
        """Test delay calculation for fixed interval policy"""
        config = RetryConfig(
            initial_delay=2.0, policy=RetryPolicy.FIXED_INTERVAL, jitter=False
        )

        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(3) == 2.0
        assert config.calculate_delay(10) == 2.0

    def test_calculate_delay_linear_backoff(self):
        """Test delay calculation for linear backoff policy"""
        config = RetryConfig(
            initial_delay=1.0, policy=RetryPolicy.LINEAR_BACKOFF, jitter=False
        )

        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 3.0

    def test_calculate_delay_exponential_backoff(self):
        """Test delay calculation for exponential backoff policy"""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            jitter=False,
        )

        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
        assert config.calculate_delay(4) == 8.0

    def test_calculate_delay_max_delay_limit(self):
        """Test delay calculation respects max_delay limit"""
        config = RetryConfig(
            initial_delay=10.0,
            max_delay=15.0,
            exponential_base=2.0,
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            jitter=False,
        )

        assert config.calculate_delay(1) == 10.0
        assert config.calculate_delay(2) == 15.0  # Capped at max_delay
        assert config.calculate_delay(3) == 15.0  # Still capped

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter"""
        config = RetryConfig(
            initial_delay=10.0, policy=RetryPolicy.EXPONENTIAL_BACKOFF, jitter=True
        )

        delay1 = config.calculate_delay(1)
        delay2 = config.calculate_delay(1)

        # With jitter, delays should be different
        # (There's a small chance they could be equal, but very unlikely)
        assert delay1 >= 9.0  # Should be within jitter range
        assert delay1 <= 11.0
        assert delay2 >= 9.0
        assert delay2 <= 11.0

    def test_calculate_delay_jitter_policy(self):
        """Test delay calculation for jitter policy"""
        config = RetryConfig(
            initial_delay=1.0, exponential_base=2.0, policy=RetryPolicy.JITTER
        )

        delay = config.calculate_delay(2)
        # Should be exponential backoff with built-in jitter
        assert delay >= 1.0  # At least half of expected value
        assert delay <= 2.0  # At most the full expected value

    def test_should_retry_with_retryable_exceptions(self):
        """Test should_retry with retryable exceptions"""
        config = RetryConfig(retryable_exceptions=[ConnectionError, ValueError])

        assert config.should_retry(ConnectionError()) is True
        assert config.should_retry(ValueError()) is True
        assert config.should_retry(RuntimeError()) is False

    def test_should_retry_with_exception_hierarchy(self):
        """Test should_retry with exception hierarchy"""
        config = RetryConfig(retryable_exceptions=[Exception])

        assert config.should_retry(ConnectionError()) is True
        assert config.should_retry(ValueError()) is True
        assert config.should_retry(RuntimeError()) is True


class TestCircuitBreaker:
    """Test suite for CircuitBreaker"""

    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization"""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ConnectionError,
        )

        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0
        assert cb.expected_exception == ConnectionError
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_successful_execution(self):
        """Test circuit breaker with successful execution"""
        cb = CircuitBreaker(failure_threshold=2)

        async with cb:
            # Simulate successful operation
            pass

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker failure counting"""
        cb = CircuitBreaker(failure_threshold=2)

        # First failure
        try:
            async with cb:
                raise ConnectionError("Test failure")
        except ConnectionError:
            pass

        assert cb.failure_count == 1
        assert cb.state == CircuitBreakerState.CLOSED

        # Second failure - should open circuit
        try:
            async with cb:
                raise ConnectionError("Test failure")
        except ConnectionError:
            pass

        assert cb.failure_count == 2
        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state_rejection(self):
        """Test circuit breaker rejects requests when open"""
        cb = CircuitBreaker(failure_threshold=1)

        # Cause failure to open circuit
        try:
            async with cb:
                raise ConnectionError("Test failure")
        except ConnectionError:
            pass

        assert cb.state == CircuitBreakerState.OPEN

        # Next request should be rejected
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            async with cb:
                pass

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Cause failure to open circuit
        try:
            async with cb:
                raise ConnectionError("Test failure")
        except ConnectionError:
            pass

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next request should transition to half-open
        async with cb:
            # Successful operation should close the circuit
            pass

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Cause failure to open circuit
        try:
            async with cb:
                raise ConnectionError("Test failure")
        except ConnectionError:
            pass

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Fail in half-open state should reopen circuit
        try:
            async with cb:
                raise ConnectionError("Test failure")
        except ConnectionError:
            pass

        assert cb.state == CircuitBreakerState.OPEN


class TestRetryDecorator:
    """Test suite for retry_with_backoff decorator"""

    @pytest.mark.asyncio
    async def test_retry_decorator_async_success(self):
        """Test retry decorator with successful async function"""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        async def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_function()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator_async_eventual_success(self):
        """Test retry decorator with eventual success"""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await test_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_decorator_async_max_attempts_exceeded(self):
        """Test retry decorator when max attempts are exceeded"""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=2, initial_delay=0.01))
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(RetryExhaustedError, match="Failed after 2 attempts"):
            await test_function()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_async_non_retryable_exception(self):
        """Test retry decorator with non-retryable exception"""
        call_count = 0

        @retry_with_backoff(
            RetryConfig(
                max_attempts=3,
                initial_delay=0.01,
                retryable_exceptions=[ConnectionError],
            )
        )
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError, match="Non-retryable error"):
            await test_function()

        assert call_count == 1

    def test_retry_decorator_sync_success(self):
        """Test retry decorator with successful sync function"""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_function()

        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_sync_eventual_success(self):
        """Test retry decorator with eventual success in sync function"""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = test_function()

        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_sync_max_attempts_exceeded(self):
        """Test retry decorator when max attempts are exceeded in sync function"""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_attempts=2, initial_delay=0.01))
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(RetryExhaustedError, match="Failed after 2 attempts"):
            test_function()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_with_circuit_breaker(self):
        """Test retry decorator with circuit breaker"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        call_count = 0

        @retry_with_backoff(
            RetryConfig(max_attempts=5, initial_delay=0.01), circuit_breaker=cb
        )
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(RetryExhaustedError):
            await test_function()

        # Should have failed before max attempts due to circuit breaker
        assert call_count <= 5  # Should not exceed max attempts
        assert cb.state == CircuitBreakerState.OPEN


class TestDefaultConfigurations:
    """Test suite for default retry configurations"""

    def test_default_retry_config(self):
        """Test default retry configuration"""
        config = DEFAULT_RETRY_CONFIG

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 30.0
        assert config.policy == RetryPolicy.EXPONENTIAL_BACKOFF

    def test_aggressive_retry_config(self):
        """Test aggressive retry configuration"""
        config = AGGRESSIVE_RETRY_CONFIG

        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_conservative_retry_config(self):
        """Test conservative retry configuration"""
        config = CONSERVATIVE_RETRY_CONFIG

        assert config.max_attempts == 2
        assert config.initial_delay == 2.0
        assert config.max_delay == 10.0
        assert config.policy == RetryPolicy.FIXED_INTERVAL

    def test_default_circuit_breaker(self):
        """Test default circuit breaker configuration"""
        cb = DEFAULT_CIRCUIT_BREAKER

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 60.0
        assert cb.expected_exception == Exception


if __name__ == "__main__":
    pytest.main([__file__])
