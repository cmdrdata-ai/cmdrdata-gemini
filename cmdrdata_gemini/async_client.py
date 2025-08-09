"""
Async Tracked Google Gen AI client with automatic usage tracking
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from .context import get_effective_customer_id
from .exceptions import ConfigurationError, ValidationError
from .logging_config import get_logger
from .performance import PerformanceContext
from .security import APIKeyManager, InputSanitizer
from .tracker import UsageTracker
from .version_compat import check_compatibility

logger = get_logger(__name__)


class AsyncTrackedGemini:
    """
    Async drop-in replacement for Google Gen AI AsyncClient with automatic usage tracking.

    This client maintains 100% API compatibility with the original Google Gen AI AsyncClient
    while transparently tracking usage for customer billing and analytics.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cmdrdata_api_key: Optional[str] = None,
        cmdrdata_endpoint: str = "https://api.cmdrdata.ai/api/events",
        track_usage: bool = True,
        **kwargs,
    ):
        """
        Initialize the async tracked Google Gen AI client.

        Args:
            api_key: Google Gen AI API key (or set GEMINI_API_KEY env var)
            cmdrdata_api_key: cmdrdata API key for usage tracking
            cmdrdata_endpoint: cmdrdata API endpoint URL
            track_usage: Whether to enable usage tracking
            **kwargs: Additional arguments passed to Google Gen AI AsyncClient
        """
        # Check version compatibility
        if not check_compatibility():
            logger.warning(
                "Google Gen AI SDK version may not be fully supported. "
                "Please check compatibility warnings."
            )

        # Import Google Gen AI here to provide better error messages
        try:
            from google import genai
        except ImportError:
            raise ConfigurationError(
                "Google Gen AI SDK not found. Please install it: pip install google-genai>=0.1.0"
            )

        # Validate API keys if provided
        if api_key:
            try:
                APIKeyManager.validate_api_key(api_key, "google")
            except Exception as e:
                raise ValidationError(f"Invalid Google Gen AI API key: {e}")

        if cmdrdata_api_key:
            try:
                APIKeyManager.validate_api_key(cmdrdata_api_key, "cmdrdata")
            except Exception as e:
                raise ValidationError(f"Invalid cmdrdata API key: {e}")

        # Initialize the original Google Gen AI AsyncClient
        try:
            if api_key:
                kwargs["api_key"] = api_key
            # Note: Google Gen AI may use AsyncClient in the future
            # For now, we'll use the regular Client but wrap async operations
            self._original_client = genai.Client(**kwargs)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google Gen AI client: {e}")

        # Set up usage tracking
        self._tracker = None
        self._track_usage = track_usage and cmdrdata_api_key is not None

        if self._track_usage:
            try:
                self._tracker = UsageTracker(
                    api_key=cmdrdata_api_key,
                    endpoint=cmdrdata_endpoint,
                )
                logger.info("Usage tracking enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize usage tracking: {e}")
                self._track_usage = False

        # Performance monitoring
        self._performance = PerformanceContext("async_gemini_client")

    async def _track_generate_content(
        self,
        result: Any,
        customer_id: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Track generate_content usage asynchronously"""
        if not self._track_usage or not self._tracker:
            return

        try:
            effective_customer_id = get_effective_customer_id(customer_id)

            if hasattr(result, "usage_metadata") and result.usage_metadata:
                # Clean up model name
                clean_model = model or "unknown"
                if isinstance(clean_model, str) and clean_model.startswith("models/"):
                    clean_model = clean_model[7:]  # Remove "models/" prefix

                await self._tracker.track_usage_async(
                    customer_id=effective_customer_id,
                    model=clean_model,
                    input_tokens=getattr(
                        result.usage_metadata, "prompt_token_count", 0
                    ),
                    output_tokens=getattr(
                        result.usage_metadata, "candidates_token_count", 0
                    ),
                    provider="google",
                    metadata={
                        "response_id": getattr(result, "id", None),
                        "model_version": getattr(result, "model_version", None),
                        "safety_ratings": getattr(result, "safety_ratings", None),
                        "finish_reason": (
                            getattr(result.candidates[0], "finish_reason", None)
                            if hasattr(result, "candidates") and result.candidates
                            else None
                        ),
                        "total_token_count": getattr(
                            result.usage_metadata, "total_token_count", 0
                        ),
                    },
                    timestamp=datetime.utcnow(),
                )

        except Exception as e:
            logger.warning(f"Failed to track usage for generate_content: {e}")

    async def _track_count_tokens(
        self,
        result: Any,
        customer_id: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Track count_tokens usage asynchronously"""
        if not self._track_usage or not self._tracker:
            return

        try:
            effective_customer_id = get_effective_customer_id(customer_id)

            if hasattr(result, "total_tokens"):
                # Clean up model name
                clean_model = model or "unknown"
                if isinstance(clean_model, str) and clean_model.startswith("models/"):
                    clean_model = clean_model[7:]  # Remove "models/" prefix

                await self._tracker.track_usage_async(
                    customer_id=effective_customer_id,
                    model=clean_model,
                    input_tokens=result.total_tokens,
                    output_tokens=0,  # No generation, just counting
                    provider="google",
                    metadata={
                        "operation": "count_tokens",
                        "total_tokens": result.total_tokens,
                    },
                    timestamp=datetime.utcnow(),
                )

        except Exception as e:
            logger.warning(f"Failed to track usage for count_tokens: {e}")

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically forward attribute access to the underlying client.
        """
        # Handle models attribute specially to add tracking
        if name == "models":
            return AsyncTrackedModels(
                self._original_client.models,
                self._track_generate_content if self._track_usage else None,
                self._track_count_tokens if self._track_usage else None,
            )

        # For other attributes, just forward to the original client
        try:
            return getattr(self._original_client, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self._original_client).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to the underlying client"""
        if name.startswith("_") or name in [
            "api_key",
            "base_url",
            "timeout",
            "max_retries",
            "default_headers",
        ]:
            object.__setattr__(self, name, value)
        else:
            setattr(self._original_client, name, value)

    def __dir__(self):
        """Return attributes from both proxy and underlying client"""
        proxy_attrs = [
            attr for attr in object.__dir__(self) if not attr.startswith("_")
        ]
        client_attrs = dir(self._original_client)
        return sorted(set(proxy_attrs + client_attrs))

    def get_usage_tracker(self) -> Optional[UsageTracker]:
        """Get the usage tracker instance (for testing/debugging)"""
        return self._tracker

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics"""
        return self._performance.get_all_stats()

    async def __aenter__(self):
        """Async context manager entry"""
        if hasattr(self._original_client, "__aenter__"):
            await self._original_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if hasattr(self._original_client, "__aexit__"):
            return await self._original_client.__aexit__(exc_type, exc_val, exc_tb)

    def __repr__(self):
        """Return a helpful representation"""
        tracking_status = "enabled" if self._track_usage else "disabled"
        return f"AsyncTrackedGemini(tracking={tracking_status})"


class AsyncTrackedModels:
    """Wrapper for models API with usage tracking"""

    def __init__(self, original_models, track_generate_func, track_count_func):
        self._original_models = original_models
        self._track_generate_func = track_generate_func
        self._track_count_func = track_count_func

    async def generate_content(
        self, customer_id: Optional[str] = None, track_usage: bool = True, **kwargs
    ):
        """Generate content with optional usage tracking"""
        # Call the original generate_content method in a thread pool since it's sync
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._original_models.generate_content(**kwargs)
        )

        # Track usage if enabled
        if track_usage and self._track_generate_func:
            await self._track_generate_func(
                result=result,
                customer_id=customer_id,
                **kwargs,
            )

        return result

    async def count_tokens(
        self, customer_id: Optional[str] = None, track_usage: bool = True, **kwargs
    ):
        """Count tokens with optional usage tracking"""
        # Call the original count_tokens method in a thread pool since it's sync
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._original_models.count_tokens(**kwargs)
        )

        # Track usage if enabled
        if track_usage and self._track_count_func:
            await self._track_count_func(
                result=result,
                customer_id=customer_id,
                **kwargs,
            )

        return result

    def __getattr__(self, name: str) -> Any:
        """Forward other method calls to the original models object"""
        return getattr(self._original_models, name)
