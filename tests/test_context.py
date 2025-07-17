"""
Tests for customer context management
"""

import pytest
from cmdrdata_gemini.context import (
    set_customer_context,
    get_customer_context,
    clear_customer_context,
    customer_context,
    get_effective_customer_id
)


def test_basic_context_operations():
    """Test basic context set/get/clear operations"""
    # Initially no context
    assert get_customer_context() is None
    
    # Set context
    set_customer_context("test-customer")
    assert get_customer_context() == "test-customer"
    
    # Clear context
    clear_customer_context()
    assert get_customer_context() is None


def test_context_manager():
    """Test context manager functionality"""
    # Set initial context
    set_customer_context("initial-customer")
    
    # Use context manager
    with customer_context("temp-customer"):
        assert get_customer_context() == "temp-customer"
    
    # Should restore original context
    assert get_customer_context() == "initial-customer"
    
    # Test with no initial context
    clear_customer_context()
    with customer_context("temp-customer"):
        assert get_customer_context() == "temp-customer"
    
    # Should be None after exiting
    assert get_customer_context() is None


def test_effective_customer_id_priority():
    """Test customer ID resolution priority"""
    # No context, no explicit ID
    clear_customer_context()
    assert get_effective_customer_id() is None
    
    # Context only
    set_customer_context("context-customer")
    assert get_effective_customer_id() == "context-customer"
    
    # Explicit ID overrides context
    assert get_effective_customer_id("explicit-customer") == "explicit-customer"
    
    # Explicit None overrides context
    assert get_effective_customer_id(None) is None


def test_nested_context_managers():
    """Test nested context managers"""
    set_customer_context("initial")
    
    with customer_context("level1"):
        assert get_customer_context() == "level1"
        
        with customer_context("level2"):
            assert get_customer_context() == "level2"
        
        # Should restore level1
        assert get_customer_context() == "level1"
    
    # Should restore initial
    assert get_customer_context() == "initial"


def test_context_isolation():
    """Test that context is properly isolated"""
    import threading
    import time
    
    results = {}
    
    def worker(customer_id, results_dict, key):
        set_customer_context(customer_id)
        time.sleep(0.1)  # Simulate some work
        results_dict[key] = get_customer_context()
    
    # Start multiple threads with different contexts
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=worker,
            args=(f"customer-{i}", results, f"thread-{i}")
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Each thread should have its own context
    assert results["thread-0"] == "customer-0"
    assert results["thread-1"] == "customer-1"
    assert results["thread-2"] == "customer-2"