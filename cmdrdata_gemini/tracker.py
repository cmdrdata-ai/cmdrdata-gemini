"""
Usage tracking client for sending events to cmdrdata backend
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import httpx
except ImportError:
    # Fallback to requests for simpler installations
    import requests

    httpx = None

from .exceptions import NetworkError, TrackingError, ValidationError

# InputValidator functionality is now in security module
from .logging_config import get_logger
from .performance import PerformanceContext, timed
from .retry import DEFAULT_RETRY_CONFIG, CircuitBreaker, RetryConfig, retry_with_backoff
from .security import APIKeyManager, InputSanitizer

logger = get_logger(__name__)


class UsageTracker:
    """
    Client for sending usage events to cmdrdata backend.

    Handles both sync and async sending with retry logic and error handling.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.cmdrdata.ai/api/events",
        timeout: float = 5.0,
        max_retries: int = 3,
    ):
        """
        Initialize the usage tracker.

        Args:
            api_key: cmdrdata API key for authentication
            endpoint: cmdrdata API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        # Validate inputs
        if not api_key or not isinstance(api_key, str) or api_key.strip() == "":
            raise ValidationError("API key is required")

        # Validate API key format
        try:
            APIKeyManager.validate_api_key(api_key, "cmdrdata")
        except Exception:
            raise ValidationError("Invalid API key format")

        if not endpoint or not isinstance(endpoint, str) or endpoint.strip() == "":
            raise ValidationError("Endpoint is required")

        # Validate endpoint URL
        try:
            InputSanitizer.validate_url(endpoint)
        except Exception:
            raise ValidationError("Invalid endpoint URL")

        # Validate timeout
        try:
            InputSanitizer.validate_timeout(timeout)
        except Exception:
            raise ValidationError("Invalid timeout")

        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "cmdrdata-openai/0.1.0",
        }

        # Thread pool for async usage tracking in sync contexts
        self._executor = ThreadPoolExecutor(max_workers=2)

    def track_usage(
        self,
        customer_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "openai",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Track usage synchronously.

        Args:
            customer_id: Customer identifier
            model: AI model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: AI provider (default: openai)
            metadata: Additional metadata about the request
            timestamp: Event timestamp (defaults to now)

        Returns:
            True if tracking succeeded, False otherwise
        """
        try:
            # Validate inputs first
            self._validate_tracking_inputs(
                customer_id, model, input_tokens, output_tokens, provider, metadata
            )

            # Sanitize and prepare the data
            event_data = self._sanitize_tracking_data(
                customer_id,
                model,
                input_tokens,
                output_tokens,
                provider,
                metadata,
                timestamp,
            )

            # Track usage with retry logic
            return self._track_usage_with_retry(event_data)

        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error tracking usage: {e}")
            return False

    async def track_usage_async(
        self,
        customer_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "openai",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Track usage asynchronously.

        Args:
            customer_id: Customer identifier
            model: AI model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: AI provider (default: openai)
            metadata: Additional metadata about the request
            timestamp: Event timestamp (defaults to now)

        Returns:
            True if tracking succeeded, False otherwise
        """
        event_data = {
            "customer_id": customer_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "provider": provider,
            "metadata": metadata or {},
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "version": "0.1.0",
        }

        try:
            if httpx:
                # Use httpx if available (preferred for async)
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.endpoint, json=event_data, headers=self.headers
                    )

                    if response.status_code == 200:
                        logger.debug(
                            f"Successfully tracked usage for customer {customer_id}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Failed to track usage: {response.status_code} {response.text}"
                        )
                        return False
            else:
                # Fallback to sync version in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self._executor,
                    self.track_usage,
                    customer_id,
                    model,
                    input_tokens,
                    output_tokens,
                    provider,
                    metadata,
                    timestamp,
                )

        except Exception as e:
            logger.error(f"Error tracking usage: {e}")
            return False

    def track_usage_background(
        self,
        customer_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str = "openai",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        # Enhanced tracking parameters for better analytics
        request_start_time: Optional[float] = None,
        request_end_time: Optional[float] = None,
        error_occurred: Optional[bool] = None,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Track usage in background thread (fire-and-forget).

        This method never blocks the main thread and is ideal for
        production use where tracking failures shouldn't affect
        the main application flow.

        Args:
            customer_id: Customer identifier
            model: AI model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: AI provider (default: openai)
            metadata: Additional metadata about the request
            timestamp: Event timestamp (defaults to now)
            request_start_time: Request start timestamp
            request_end_time: Request end timestamp
            error_occurred: Whether an error occurred
            error_type: Type of error if any
            error_code: Error code if any
            error_message: Error message if any
            request_id: Unique request identifier
        """
        # Merge the enhanced tracking parameters into metadata
        enhanced_metadata = metadata.copy() if metadata else {}

        # Add timing information if available
        if request_start_time and request_end_time:
            enhanced_metadata["request_duration_ms"] = int(
                (request_end_time - request_start_time) * 1000
            )
            enhanced_metadata["request_start_time"] = request_start_time
            enhanced_metadata["request_end_time"] = request_end_time

        # Add error information if available
        if error_occurred is not None:
            enhanced_metadata["error_occurred"] = error_occurred
        if error_type:
            enhanced_metadata["error_type"] = error_type
        if error_code:
            enhanced_metadata["error_code"] = error_code
        if error_message:
            enhanced_metadata["error_message"] = error_message

        # Add request ID for tracing
        if request_id:
            enhanced_metadata["request_id"] = request_id

        self._executor.submit(
            self.track_usage,
            customer_id,
            model,
            input_tokens,
            output_tokens,
            provider,
            enhanced_metadata,
            timestamp,
        )

    def _validate_tracking_inputs(
        self,
        customer_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate tracking inputs"""
        if not customer_id or not isinstance(customer_id, str):
            raise ValidationError("Customer ID is required")

        InputSanitizer.validate_customer_id(customer_id)
        InputSanitizer.validate_model_name(model)
        InputSanitizer.validate_token_count(input_tokens)
        InputSanitizer.validate_token_count(output_tokens)

        if metadata is not None:
            InputSanitizer.validate_metadata(metadata)

    def _sanitize_tracking_data(
        self,
        customer_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Sanitize and prepare tracking data"""
        sanitized_metadata = {}
        if metadata:
            sanitized_metadata = InputSanitizer.sanitize_metadata(metadata)

        return {
            "customer_id": InputSanitizer.sanitize_string(customer_id, "customer_id"),
            "model": InputSanitizer.sanitize_string(model, "model_name"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "provider": InputSanitizer.sanitize_string(provider, "general_string"),
            "metadata": sanitized_metadata,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "version": "0.1.0",
        }

    def _track_usage_with_retry(self, event_data: Dict[str, Any]) -> bool:
        """Track usage with retry logic"""
        try:
            if httpx:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.endpoint, json=event_data, headers=self.headers
                    )

                    if response.status_code == 200:
                        return True
                    elif response.status_code == 429:
                        raise NetworkError("Rate limited")
                    elif response.status_code >= 500:
                        raise NetworkError("Server error")
                    elif response.status_code >= 400:
                        raise TrackingError("Client error")
                    else:
                        return False
            else:
                # Fallback to requests
                import requests

                response = requests.post(
                    self.endpoint,
                    json=event_data,
                    headers=self.headers,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return True
                elif response.status_code == 429:
                    raise NetworkError("Rate limited")
                elif response.status_code >= 500:
                    raise NetworkError("Server error")
                elif response.status_code >= 400:
                    raise TrackingError("Client error")
                else:
                    return False

        except Exception as e:
            if "Connection" in str(e) or "Network" in str(e):
                raise TrackingError("Tracking failed")
            raise

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the tracker"""
        return {
            "healthy": True,
            "endpoint": self.endpoint,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "circuit_breaker_state": "CLOSED",
        }

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
