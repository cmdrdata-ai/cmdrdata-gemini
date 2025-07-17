"""
Customer context management for TrackedOpenAI

Provides thread-safe context management for customer IDs using
Python's contextvars for proper async support.
"""

import contextvars
from contextlib import contextmanager
from typing import Optional

# Context variable for storing customer ID in current context
_customer_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "customer_id", default=None
)


def set_customer_context(customer_id: str) -> None:
    """
    Set the customer ID for the current context.

    This customer ID will be used for all TrackedOpenAI calls
    within the current thread/async context until cleared.

    Args:
        customer_id: The customer identifier to track usage for
    """
    _customer_context.set(customer_id)


def get_customer_context() -> Optional[str]:
    """
    Get the current customer ID from context.

    Returns:
        The customer ID if set, None otherwise
    """
    return _customer_context.get()


def clear_customer_context() -> None:
    """
    Clear the customer ID from the current context.

    After calling this, TrackedOpenAI calls will need an explicit
    customer_id parameter or a new context to be set.
    """
    _customer_context.set(None)


@contextmanager
def customer_context(customer_id: str):
    """
    Context manager for temporarily setting a customer ID.

    Usage:
        with customer_context("customer-123"):
            response = client.chat.completions.create(...)
            # customer-123 is automatically tracked
        # customer context is automatically cleared

    Args:
        customer_id: The customer identifier for this context
    """
    previous_customer = get_customer_context()
    set_customer_context(customer_id)
    try:
        yield
    finally:
        if previous_customer is not None:
            set_customer_context(previous_customer)
        else:
            clear_customer_context()


def get_effective_customer_id(
    explicit_customer_id: Optional[str] = ...,
) -> Optional[str]:
    """
    Get the effective customer ID to use for tracking.

    Priority:
    1. Explicit customer_id parameter (highest priority)
    2. Customer ID from context
    3. None (no tracking)

    Args:
        explicit_customer_id: Customer ID passed explicitly to the API call

    Returns:
        The customer ID to use for tracking, or None if not available
    """
    # Use ... as sentinel to distinguish between None explicitly passed vs not passed
    if explicit_customer_id is not ...:
        return explicit_customer_id

    return get_customer_context()
