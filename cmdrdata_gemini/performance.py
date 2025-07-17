"""
Performance optimizations and caching for cmdrdata-openai
"""

import asyncio
import hashlib
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    value: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[timedelta] = None

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if not self.ttl:
            return False
        return datetime.utcnow() - self.created_at > self.ttl

    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class LRUCache:
    """Thread-safe LRU cache implementation with TTL support"""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[timedelta] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                self._remove_key(key)
                return None

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            entry.touch()
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in cache"""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self._remove_key(key)

            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                self._evict_lru()

            # Add new entry
            entry = CacheEntry(
                value=value, created_at=datetime.utcnow(), ttl=ttl or self.default_ttl
            )

            self.cache[key] = entry
            self.access_order.append(key)

    def _remove_key(self, key: str) -> None:
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)

    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]

    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "hit_rate": total_accesses / max(1, len(self.cache)),
            }


class ConnectionPool:
    """HTTP connection pool for better performance"""

    def __init__(self, max_connections: int = 10, max_keepalive: int = 5):
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self.connections: Dict[str, List[Any]] = defaultdict(list)
        self.lock = threading.RLock()

    def get_connection(self, host: str) -> Optional[Any]:
        """Get a connection from the pool"""
        with self.lock:
            if host in self.connections and self.connections[host]:
                return self.connections[host].pop()
            return None

    def return_connection(self, host: str, connection: Any) -> None:
        """Return a connection to the pool"""
        with self.lock:
            if len(self.connections[host]) < self.max_keepalive:
                self.connections[host].append(connection)

    def clear(self) -> None:
        """Clear all connections"""
        with self.lock:
            self.connections.clear()


class RequestBatcher:
    """Batch multiple requests together for better performance"""

    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[Tuple[Any, asyncio.Future]] = []
        self.lock = asyncio.Lock()
        self.batch_task: Optional[asyncio.Task] = None

    async def add_request(self, request: Any, processor: Callable) -> Any:
        """Add a request to the batch"""
        future = asyncio.Future()
        should_process_immediately = False

        async with self.lock:
            self.pending_requests.append((request, future))

            # Check if batch is full
            if len(self.pending_requests) >= self.batch_size:
                if self.batch_task and not self.batch_task.done():
                    self.batch_task.cancel()
                should_process_immediately = True
            elif not self.batch_task or self.batch_task.done():
                # Start batch task if not already running
                self.batch_task = asyncio.create_task(
                    self._process_batch(processor, immediate=False)
                )

        # Process immediately if batch is full (outside the lock)
        if should_process_immediately:
            await self._process_batch(processor, immediate=True)

        return await future

    async def _process_batch(
        self, processor: Callable, immediate: bool = False
    ) -> None:
        """Process the current batch"""
        if not immediate:
            await asyncio.sleep(self.batch_timeout)

        async with self.lock:
            if not self.pending_requests:
                return

            batch = self.pending_requests.copy()
            self.pending_requests.clear()

        try:
            # Process all requests in batch
            requests = [item[0] for item in batch]
            results = await processor(requests)

            # Return results to futures
            for (_, future), result in zip(batch, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # Set exception for all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: float, burst: int):
        self.rate = rate  # tokens per second
        self.burst = burst  # max tokens
        self.tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        with self.lock:
            now = time.time()

            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def acquire_async(self, tokens: int = 1) -> None:
        """Async version that waits for tokens"""
        while not self.acquire(tokens):
            await asyncio.sleep(0.01)


class PerformanceMonitor:
    """Monitor and collect performance metrics"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def record_metric(
        self, name: str, value: float, timestamp: Optional[float] = None
    ) -> None:
        """Record a metric value"""
        with self.lock:
            self.metrics[name].append(
                {"value": value, "timestamp": timestamp or time.time()}
            )

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter"""
        with self.lock:
            self.counters[name] += value

    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        with self.lock:
            if name not in self.metrics:
                return {}

            values = [entry["value"] for entry in self.metrics[name]]
            if not values:
                return {}

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else None,
            }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics"""
        with self.lock:
            stats = {}
            for name in self.metrics:
                # Calculate stats directly without calling get_stats to avoid deadlock
                values = [entry["value"] for entry in self.metrics[name]]
                if values:
                    stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "latest": values[-1] if values else None,
                    }
                else:
                    stats[name] = {}

            stats["counters"] = self.counters.copy()
            return stats


# Global instances
_cache = LRUCache(max_size=1000, default_ttl=timedelta(minutes=5))
_performance_monitor = PerformanceMonitor()
_connection_pool = ConnectionPool()


def cached(ttl: Optional[timedelta] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {"func": func.__name__, "args": args, "kwargs": kwargs}
                cache_key = hashlib.md5(
                    json.dumps(key_data, sort_keys=True, default=str).encode()
                ).hexdigest()

            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}")

            return result

        return wrapper

    return decorator


def timed(metric_name: Optional[str] = None):
    """Decorator for timing function execution"""

    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                _performance_monitor.record_metric(f"{name}.duration", duration)
                _performance_monitor.increment_counter(f"{name}.calls")

        return wrapper

    return decorator


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return _cache.stats()


def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics"""
    return _performance_monitor.get_all_stats()


def clear_cache() -> None:
    """Clear all caches"""
    _cache.clear()


def configure_performance(
    cache_size: int = 1000,
    cache_ttl: Optional[timedelta] = None,
    max_connections: int = 10,
) -> None:
    """Configure performance settings"""
    global _cache, _connection_pool

    _cache = LRUCache(max_size=cache_size, default_ttl=cache_ttl)
    _connection_pool = ConnectionPool(max_connections=max_connections)

    logger.info(
        f"Performance configured: cache_size={cache_size}, max_connections={max_connections}"
    )


# Context manager for performance tracking
class PerformanceContext:
    """Context manager for tracking performance of operations"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.metrics = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        _performance_monitor.record_metric(f"{self.operation_name}.duration", duration)
        _performance_monitor.increment_counter(f"{self.operation_name}.calls")

        if exc_type:
            _performance_monitor.increment_counter(f"{self.operation_name}.errors")
        else:
            _performance_monitor.increment_counter(f"{self.operation_name}.success")

    def add_metric(self, name: str, value: float):
        """Add a custom metric to this operation"""
        _performance_monitor.record_metric(f"{self.operation_name}.{name}", value)
