"""
Retry logic and circuit breaker implementation for robust API calls
"""

import asyncio
import logging
import random
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Optional, Type, Union

from .exceptions import (
    CircuitBreakerError,
    NetworkError,
    RetryExhaustedError,
    TimeoutError,
)

logger = logging.getLogger(__name__)


class RetryPolicy(Enum):
    """Retry policy types"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    LINEAR_BACKOFF = "linear_backoff"
    JITTER = "jitter"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.policy = policy
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            NetworkError,
            Exception,  # Generic fallback
        ]

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt"""
        if self.policy == RetryPolicy.FIXED_INTERVAL:
            delay = self.initial_delay
        elif self.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.initial_delay * attempt
        elif self.policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
        elif self.policy == RetryPolicy.JITTER:
            delay = self.initial_delay * (self.exponential_base ** (attempt - 1))
            delay = delay * (0.5 + random.random() * 0.5)  # Add jitter
        else:
            delay = self.initial_delay

        # Apply jitter if enabled
        if self.jitter and self.policy != RetryPolicy.JITTER:
            jitter_factor = 0.1  # 10% jitter
            delay = delay * (1 + random.uniform(-jitter_factor, jitter_factor))

        return min(delay, self.max_delay)

    def should_retry(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry"""
        return any(
            isinstance(exception, exc_type) for exc_type in self.retryable_exceptions
        )


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker moved to HALF_OPEN state")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        async with self._lock:
            if exc_type is None:
                # Success
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker moved to CLOSED state")
            elif issubclass(exc_type, self.expected_exception):
                # Failure
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"Circuit breaker moved to OPEN state after {self.failure_count} failures"
                    )

        return False  # Don't suppress exceptions


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
):
    """
    Decorator for retry logic with exponential backoff and circuit breaker

    Args:
        config: Retry configuration
        circuit_breaker: Circuit breaker instance
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    if circuit_breaker:
                        async with circuit_breaker:
                            return await func(*args, **kwargs)
                    else:
                        return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e):
                        logger.debug(f"Exception {type(e).__name__} is not retryable")
                        raise

                    if attempt >= config.max_attempts:
                        logger.error(
                            f"Max retry attempts ({config.max_attempts}) exceeded"
                        )
                        break

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed with {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

            # All retries exhausted
            raise RetryExhaustedError(
                f"Failed after {config.max_attempts} attempts. Last error: {str(last_exception)}",
                details={
                    "last_exception": str(last_exception),
                    "attempts": config.max_attempts,
                },
            ) from last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e):
                        logger.debug(f"Exception {type(e).__name__} is not retryable")
                        raise

                    if attempt >= config.max_attempts:
                        logger.error(
                            f"Max retry attempts ({config.max_attempts}) exceeded"
                        )
                        break

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed with {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # All retries exhausted
            raise RetryExhaustedError(
                f"Failed after {config.max_attempts} attempts. Last error: {str(last_exception)}",
                details={
                    "last_exception": str(last_exception),
                    "attempts": config.max_attempts,
                },
            ) from last_exception

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Default retry configurations for different scenarios
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    policy=RetryPolicy.EXPONENTIAL_BACKOFF,
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=60.0,
    policy=RetryPolicy.EXPONENTIAL_BACKOFF,
    jitter=True,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=2, initial_delay=2.0, max_delay=10.0, policy=RetryPolicy.FIXED_INTERVAL
)

# Default circuit breaker for tracking endpoints
DEFAULT_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=5, recovery_timeout=60.0, expected_exception=Exception
)
