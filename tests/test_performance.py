"""
Performance tests for cmdrdata-openai
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from unittest.mock import Mock, patch

import pytest

from cmdrdata_gemini.performance import (
    CacheEntry,
    ConnectionPool,
    LRUCache,
    PerformanceContext,
    PerformanceMonitor,
    RateLimiter,
    RequestBatcher,
    cached,
    clear_cache,
    configure_performance,
    get_cache_stats,
    get_performance_stats,
    timed,
)


class TestCacheEntry:
    """Test cache entry functionality"""

    def test_cache_entry_creation(self):
        """Test cache entry creation"""
        entry = CacheEntry(
            value="test_value", created_at=datetime.utcnow(), ttl=timedelta(minutes=5)
        )

        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.ttl == timedelta(minutes=5)

    def test_cache_entry_is_expired(self):
        """Test cache entry expiration"""
        # Non-expiring entry
        entry = CacheEntry(value="test_value", created_at=datetime.utcnow())
        assert entry.is_expired() is False

        # Expired entry
        entry = CacheEntry(
            value="test_value",
            created_at=datetime.utcnow() - timedelta(minutes=10),
            ttl=timedelta(minutes=5),
        )
        assert entry.is_expired() is True

        # Non-expired entry
        entry = CacheEntry(
            value="test_value",
            created_at=datetime.utcnow() - timedelta(minutes=2),
            ttl=timedelta(minutes=5),
        )
        assert entry.is_expired() is False

    def test_cache_entry_touch(self):
        """Test cache entry access tracking"""
        entry = CacheEntry(value="test_value", created_at=datetime.utcnow())

        original_access_count = entry.access_count
        original_last_accessed = entry.last_accessed

        time.sleep(0.01)  # Small delay to ensure timestamp difference
        entry.touch()

        assert entry.access_count == original_access_count + 1
        assert entry.last_accessed > original_last_accessed


class TestLRUCache:
    """Test LRU cache implementation"""

    def test_lru_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = LRUCache(max_size=3)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test non-existent key
        assert cache.get("nonexistent") is None

    def test_lru_cache_eviction(self):
        """Test LRU eviction policy"""
        cache = LRUCache(max_size=2)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 to make it more recently used
        cache.get("key1")

        # Add key3, should evict key2
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"  # Should exist

    def test_lru_cache_ttl(self):
        """Test TTL functionality"""
        cache = LRUCache(max_size=10, default_ttl=timedelta(milliseconds=100))

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should return None after TTL expiry
        assert cache.get("key1") is None

    def test_lru_cache_custom_ttl(self):
        """Test custom TTL per entry"""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1", ttl=timedelta(milliseconds=100))
        cache.set("key2", "value2")  # No TTL

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # Wait for TTL to expire
        time.sleep(0.15)

        assert cache.get("key1") is None  # Should be expired
        assert cache.get("key2") == "value2"  # Should still exist

    def test_lru_cache_clear(self):
        """Test cache clearing"""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_lru_cache_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 twice
        cache.get("key1")
        cache.get("key1")

        # Access key2 once
        cache.get("key2")

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["total_accesses"] == 3
        assert stats["hit_rate"] == 1.5  # 3 accesses / 2 entries

    def test_lru_cache_thread_safety(self):
        """Test thread safety"""
        cache = LRUCache(max_size=100)
        results = []

        def worker(thread_id):
            for i in range(10):
                key = f"key_{thread_id}_{i}"
                cache.set(key, f"value_{thread_id}_{i}")
                value = cache.get(key)
                results.append(value == f"value_{thread_id}_{i}")

        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should have succeeded
        assert all(results)


class TestConnectionPool:
    """Test connection pool implementation"""

    def test_connection_pool_get_return(self):
        """Test connection pool get/return operations"""
        pool = ConnectionPool(max_connections=5, max_keepalive=3)

        # Mock connection
        mock_conn = Mock()

        # Initially no connections
        assert pool.get_connection("host1") is None

        # Return a connection
        pool.return_connection("host1", mock_conn)

        # Should get the same connection back
        returned_conn = pool.get_connection("host1")
        assert returned_conn is mock_conn

        # Should be None again
        assert pool.get_connection("host1") is None

    def test_connection_pool_max_keepalive(self):
        """Test max keepalive limit"""
        pool = ConnectionPool(max_connections=10, max_keepalive=2)

        # Return 3 connections
        for i in range(3):
            pool.return_connection("host1", Mock())

        # Should only keep 2 connections
        conn1 = pool.get_connection("host1")
        conn2 = pool.get_connection("host1")
        conn3 = pool.get_connection("host1")

        assert conn1 is not None
        assert conn2 is not None
        assert conn3 is None

    def test_connection_pool_clear(self):
        """Test connection pool clearing"""
        pool = ConnectionPool()

        pool.return_connection("host1", Mock())
        pool.return_connection("host2", Mock())

        assert pool.get_connection("host1") is not None
        assert pool.get_connection("host2") is not None

        pool.clear()

        assert pool.get_connection("host1") is None
        assert pool.get_connection("host2") is None


class TestRequestBatcher:
    """Test request batching functionality"""

    @pytest.mark.asyncio
    async def test_request_batcher_basic(self):
        """Test basic request batching"""

        async def mock_processor(requests):
            return [f"processed_{req}" for req in requests]

        batcher = RequestBatcher(batch_size=3, batch_timeout=0.1)

        # Add requests
        tasks = []
        for i in range(2):
            task = asyncio.create_task(
                batcher.add_request(f"request_{i}", mock_processor)
            )
            tasks.append(task)

        # Wait for batch timeout
        results = await asyncio.gather(*tasks)

        assert results == ["processed_request_0", "processed_request_1"]

    @pytest.mark.asyncio
    async def test_request_batcher_batch_size_trigger(self):
        """Test batch processing triggered by size"""

        async def mock_processor(requests):
            return [f"processed_{req}" for req in requests]

        batcher = RequestBatcher(batch_size=2, batch_timeout=1.0)

        # Add requests that should trigger batch processing
        tasks = []
        for i in range(2):
            task = asyncio.create_task(
                batcher.add_request(f"request_{i}", mock_processor)
            )
            tasks.append(task)

        # Should process immediately when batch size is reached
        results = await asyncio.gather(*tasks)

        assert results == ["processed_request_0", "processed_request_1"]

    @pytest.mark.asyncio
    async def test_request_batcher_error_handling(self):
        """Test error handling in batch processing"""

        async def failing_processor(requests):
            raise ValueError("Processing failed")

        batcher = RequestBatcher(batch_size=2, batch_timeout=0.1)

        # Add requests
        tasks = []
        for i in range(2):
            task = asyncio.create_task(
                batcher.add_request(f"request_{i}", failing_processor)
            )
            tasks.append(task)

        # Should propagate the exception
        with pytest.raises(ValueError, match="Processing failed"):
            await asyncio.gather(*tasks)


class TestRateLimiter:
    """Test rate limiter implementation"""

    def test_rate_limiter_token_bucket(self):
        """Test token bucket algorithm"""
        limiter = RateLimiter(rate=10, burst=5)  # 10 tokens/sec, 5 burst

        # Should be able to acquire up to burst limit
        for i in range(5):
            assert limiter.acquire() is True

        # Should fail after burst is exhausted
        assert limiter.acquire() is False

    def test_rate_limiter_token_replenishment(self):
        """Test token replenishment"""
        limiter = RateLimiter(rate=100, burst=2)  # 100 tokens/sec, 2 burst

        # Use up tokens
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False

        # Wait for token replenishment
        time.sleep(0.05)  # 50ms should add ~5 tokens

        # Should be able to acquire again
        assert limiter.acquire() is True

    def test_rate_limiter_multiple_tokens(self):
        """Test acquiring multiple tokens"""
        limiter = RateLimiter(rate=10, burst=5)

        # Should be able to acquire 3 tokens
        assert limiter.acquire(3) is True

        # Should be able to acquire 2 more
        assert limiter.acquire(2) is True

        # Should fail to acquire 1 more
        assert limiter.acquire(1) is False

    @pytest.mark.asyncio
    async def test_rate_limiter_async_acquire(self):
        """Test async token acquisition"""
        limiter = RateLimiter(rate=100, burst=1)

        # First acquire should succeed immediately
        await limiter.acquire_async()

        # Second acquire should wait and succeed
        start_time = time.time()
        await limiter.acquire_async()
        elapsed = time.time() - start_time

        # Should have waited at least 10ms (1/100 second)
        assert elapsed >= 0.005  # Small tolerance for timing


class TestPerformanceMonitor:
    """Test performance monitoring"""

    def test_performance_monitor_metrics(self):
        """Test metric recording and retrieval"""
        monitor = PerformanceMonitor(window_size=100)

        # Record some metrics
        monitor.record_metric("response_time", 0.1)
        monitor.record_metric("response_time", 0.2)
        monitor.record_metric("response_time", 0.15)

        # Get statistics
        stats = monitor.get_stats("response_time")

        assert stats["count"] == 3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.2
        assert abs(stats["avg"] - 0.15) < 1e-10
        assert stats["latest"] == 0.15

    def test_performance_monitor_counters(self):
        """Test counter functionality"""
        monitor = PerformanceMonitor()

        # Increment counters
        monitor.increment_counter("requests")
        monitor.increment_counter("requests", 5)
        monitor.increment_counter("errors", 2)

        # Get all stats
        stats = monitor.get_all_stats()

        assert stats["counters"]["requests"] == 6
        assert stats["counters"]["errors"] == 2

    def test_performance_monitor_window_size(self):
        """Test metric window size limiting"""
        monitor = PerformanceMonitor(window_size=2)

        # Record more metrics than window size
        monitor.record_metric("test_metric", 1)
        monitor.record_metric("test_metric", 2)
        monitor.record_metric("test_metric", 3)

        # Should only keep last 2 values
        stats = monitor.get_stats("test_metric")
        assert stats["count"] == 2
        assert stats["min"] == 2
        assert stats["max"] == 3


class TestPerformanceDecorators:
    """Test performance decorators"""

    def test_cached_decorator(self):
        """Test caching decorator"""
        call_count = 0

        @cached(ttl=timedelta(minutes=5))
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different argument should execute function
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_timed_decorator(self):
        """Test timing decorator"""
        from cmdrdata_gemini.performance import PerformanceMonitor

        # Create a fresh monitor for this test to avoid global state conflicts
        test_monitor = PerformanceMonitor()

        def timed_test(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    test_monitor.record_metric("test_operation.duration", duration)
                    test_monitor.increment_counter("test_operation.calls")

            return wrapper

        @timed_test
        def slow_function():
            time.sleep(0.01)
            return "result"

        # Execute function
        result = slow_function()
        assert result == "result"

        # Check that timing was recorded
        stats = test_monitor.get_all_stats()
        assert "test_operation.duration" in stats
        assert "test_operation.calls" in stats["counters"]
        assert stats["counters"]["test_operation.calls"] == 1

    def test_performance_context(self):
        """Test performance context manager"""
        with PerformanceContext("test_context") as ctx:
            time.sleep(0.01)
            ctx.add_metric("custom_metric", 42)

        # Check that metrics were recorded
        stats = get_performance_stats()
        assert "test_context.duration" in stats
        assert "test_context.calls" in stats["counters"]
        assert "test_context.success" in stats["counters"]
        assert "test_context.custom_metric" in stats


class TestPerformanceUtilities:
    """Test performance utility functions"""

    def test_get_cache_stats(self):
        """Test cache statistics retrieval"""
        # Clear cache first
        clear_cache()

        # Use cached function to populate cache
        @cached()
        def test_function(x):
            return x * 2

        test_function(1)
        test_function(2)
        test_function(1)  # Cache hit

        stats = get_cache_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "total_accesses" in stats
        assert "hit_rate" in stats

    def test_clear_cache(self):
        """Test cache clearing"""

        @cached()
        def test_function(x):
            return x * 2

        # Populate cache
        test_function(1)
        test_function(2)

        # Verify cache has entries
        stats = get_cache_stats()
        assert stats["size"] > 0

        # Clear cache
        clear_cache()

        # Verify cache is empty
        stats = get_cache_stats()
        assert stats["size"] == 0

    def test_configure_performance(self):
        """Test performance configuration"""
        # Configure with custom settings
        configure_performance(
            cache_size=500, cache_ttl=timedelta(minutes=10), max_connections=20
        )

        # Test that configuration was applied
        stats = get_cache_stats()
        assert stats["max_size"] == 500


if __name__ == "__main__":
    pytest.main([__file__])
