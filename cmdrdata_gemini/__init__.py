"""
cmdrdata-gemini: Transparent usage tracking for Google Gemini API

This package provides drop-in replacements for Google's Gen AI Python SDK clients
that automatically track usage for customer billing and analytics.

Key Features:
- 100% API compatibility with original Google Gen AI clients
- Transparent usage tracking with zero code changes required
- Non-blocking I/O - tracking never slows down your application
- Thread-safe customer context management
- Comprehensive error handling and retry logic
- Production-ready with extensive testing

Basic Usage:
    from cmdrdata_gemini import TrackedGemini
    
    client = TrackedGemini(
        api_key="your-gemini-key",
        cmdrdata_api_key="your-cmdrdata-key"
    )
    
    # Same API as regular Google Gen AI client, with automatic tracking
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain how AI works"
    )

Customer Context:
    from cmdrdata_gemini.context import customer_context
    
    with customer_context("customer-123"):
        response = client.models.generate_content(...)  # Automatically tracked
"""

__version__ = "0.1.0"

# Main client exports
from .client import TrackedGemini
from .async_client import AsyncTrackedGemini

# Context management
from .context import (
    customer_context,
    set_customer_context,
    get_customer_context,
    clear_customer_context,
)

# Exceptions
from .exceptions import (
    CMDRDataError,
    ValidationError,
    ConfigurationError,
    NetworkError,
    TrackingError,
)

# Version compatibility
from .version_compat import check_compatibility, get_compatibility_info

__all__ = [
    # Main clients
    "TrackedGemini",
    "AsyncTrackedGemini",
    # Context management
    "customer_context",
    "set_customer_context", 
    "get_customer_context",
    "clear_customer_context",
    # Exceptions
    "CMDRDataError",
    "ValidationError",
    "ConfigurationError", 
    "NetworkError",
    "TrackingError",
    # Compatibility
    "check_compatibility",
    "get_compatibility_info",
]


def get_version() -> str:
    """Get the current version of cmdrdata-gemini"""
    return __version__