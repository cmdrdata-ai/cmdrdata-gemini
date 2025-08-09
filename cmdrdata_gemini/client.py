"""
Tracked Google Gen AI client with automatic usage tracking
"""

import logging
from typing import Any, Dict, Optional

from .context import get_effective_customer_id
from .exceptions import ConfigurationError, ValidationError
from .logging_config import get_logger
from .performance import PerformanceContext
from .proxy import GEMINI_TRACK_METHODS, TrackedProxy
from .security import APIKeyManager, InputSanitizer
from .tracker import UsageTracker
from .version_compat import check_compatibility

logger = get_logger(__name__)


class TrackedGemini:
    """
    Drop-in replacement for Google Gen AI client with automatic usage tracking.

    This client maintains 100% API compatibility with the original Google Gen AI client
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
        Initialize the tracked Google Gen AI client.

        Args:
            api_key: Google Gen AI API key (or set GEMINI_API_KEY env var)
            cmdrdata_api_key: cmdrdata API key for usage tracking
            cmdrdata_endpoint: cmdrdata API endpoint URL
            track_usage: Whether to enable usage tracking
            **kwargs: Additional arguments passed to Google Gen AI client
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

        # Initialize the original Google Gen AI client
        try:
            if api_key:
                kwargs["api_key"] = api_key
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

        # Initialize tracked attributes cache
        self._tracked_attributes = {}

        # Performance monitoring
        self._performance = PerformanceContext("gemini_client")

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically forward attribute access to the underlying client.
        If the attribute should be tracked, wrap it with the proxy.
        """
        # Check if we've already wrapped this attribute
        if name in self._tracked_attributes:
            return self._tracked_attributes[name]

        # Get the attribute from the original client
        try:
            attr = getattr(self._original_client, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self._original_client).__name__}' object has no attribute '{name}'"
            )

        # If tracking is enabled and this attribute should be tracked, wrap it
        if self._track_usage and self._tracker:
            # Check if this attribute or any sub-attributes should be tracked
            relevant_track_methods = {}
            for k, v in GEMINI_TRACK_METHODS.items():
                if k == name:
                    relevant_track_methods[k] = v
                elif k.startswith(f"{name}."):
                    # Strip the prefix for nested methods
                    relevant_track_methods[k[len(name) + 1 :]] = v

            if relevant_track_methods:
                wrapped_attr = TrackedProxy(attr, self._tracker, relevant_track_methods)
                self._tracked_attributes[name] = wrapped_attr
                return wrapped_attr

        # For everything else, just return the original attribute
        self._tracked_attributes[name] = attr
        return attr

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

    def __repr__(self):
        """Return a helpful representation"""
        tracking_status = "enabled" if self._track_usage else "disabled"
        return f"TrackedGemini(tracking={tracking_status})"
