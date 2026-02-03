"""
Security Framework Phase 2 - Memory and Performance Fixes
Author: Erdinc Erdogan
Purpose: Implements BoundedCache, BoundedHistory, and WeakRef patterns for memory leak prevention, HFT latency optimization, and data integrity in high-frequency systems.
References:
- Memory Leak Prevention Patterns
- Thread-Safe Bounded Collections
- TTL-Based Cache Eviction
Usage:
    cache = BoundedCache(maxsize=1000, ttl_seconds=3600)
    cache.set('key', value)
    history = BoundedHistory(maxsize=10000)
"""

import time
import weakref
import hashlib
from collections import deque
from functools import lru_cache
from typing import Optional, Any, Dict, List, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import threading

# ============================================================================
# MEMORY LEAK PREVENTION FRAMEWORK
# ============================================================================

K = TypeVar('K')
V = TypeVar('V')

class BoundedCache(Generic[K, V]):
    """Thread-safe bounded cache with automatic eviction and TTL"""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        self._cache: Dict[Any, tuple] = {}
        self._access_order: deque = deque()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: K, default: V = None) -> V:
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return default
    
    def set(self, key: K, value: V) -> None:
        with self._lock:
            while len(self._cache) >= self._maxsize:
                if self._access_order:
                    oldest_key = self._access_order.popleft()
                    self._cache.pop(oldest_key, None)
                else:
                    break
            self._cache[key] = (value, time.time())
            self._access_order.append(key)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: K) -> bool:
        return key in self._cache


class BoundedHistory:
    """Bounded history list with automatic cleanup"""
    
    def __init__(self, maxsize: int = 10000):
        self._history: deque = deque(maxlen=maxsize)
        self._lock = threading.RLock()
    
    def append(self, item: Any) -> None:
        with self._lock:
            self._history.append(item)
    
    def extend(self, items: List[Any]) -> None:
        with self._lock:
            self._history.extend(items)
    
    def get_recent(self, n: int = 100) -> List[Any]:
        with self._lock:
            return list(self._history)[-n:]
    
    def get_all(self) -> List[Any]:
        with self._lock:
            return list(self._history)
    
    def clear(self) -> None:
        with self._lock:
            self._history.clear()
    
    def __len__(self) -> int:
        return len(self._history)
    
    def __iter__(self):
        return iter(self._history)


# ============================================================================
# HFT-COMPLIANT TIMESTAMP FRAMEWORK
# ============================================================================

class HFTTimestamp:
    """High-precision timestamp for HFT operations"""
    
    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()
    
    @staticmethod
    def now_us() -> int:
        return time.perf_counter_ns() // 1000
    
    @staticmethod
    def now_ms() -> int:
        return time.perf_counter_ns() // 1_000_000
    
    @staticmethod
    def now_seconds() -> float:
        return time.perf_counter()
    
    @staticmethod
    def utc_now() -> datetime:
        return datetime.utcnow()
    
    @staticmethod
    def elapsed_ns(start_ns: int) -> int:
        return time.perf_counter_ns() - start_ns
    
    @staticmethod
    def elapsed_us(start_ns: int) -> float:
        return (time.perf_counter_ns() - start_ns) / 1000
    
    @staticmethod
    def elapsed_ms(start_ns: int) -> float:
        return (time.perf_counter_ns() - start_ns) / 1_000_000


class DeterministicRandom:
    """Deterministic random number generator for HFT"""
    
    def __init__(self, seed: int = None):
        import random
        self._rng = random.Random(seed if seed else int(time.perf_counter_ns() % 2**32))
        self._seed = seed
        self._lock = threading.RLock()
    
    def random(self) -> float:
        with self._lock:
            return self._rng.random()
    
    def randint(self, a: int, b: int) -> int:
        with self._lock:
            return self._rng.randint(a, b)
    
    def choice(self, seq: List[Any]) -> Any:
        with self._lock:
            return self._rng.choice(seq)
    
    def shuffle(self, seq: List[Any]) -> None:
        with self._lock:
            self._rng.shuffle(seq)
    
    def reseed(self, seed: int) -> None:
        with self._lock:
            self._seed = seed
            self._rng.seed(seed)


# ============================================================================
# PERFORMANCE OPTIMIZATION FRAMEWORK
# ============================================================================

class AsyncSleepManager:
    """Non-blocking sleep manager for trading operations"""
    
    @staticmethod
    async def sleep_async(seconds: float) -> None:
        import asyncio
        await asyncio.sleep(seconds)
    
    @staticmethod
    def sleep_with_callback(seconds: float, callback: callable) -> threading.Timer:
        timer = threading.Timer(seconds, callback)
        timer.start()
        return timer
    
    @staticmethod
    def sleep_interruptible(seconds: float, stop_event: threading.Event) -> bool:
        return stop_event.wait(timeout=seconds)


class ConnectionPool:
    """Connection pool for HTTP/WebSocket connections"""
    
    def __init__(self, max_connections: int = 10):
        self._pool: deque = deque(maxlen=max_connections)
        self._max_connections = max_connections
        self._lock = threading.RLock()
    
    def get_connection(self) -> Optional[Any]:
        with self._lock:
            return self._pool.pop() if self._pool else None
    
    def return_connection(self, conn: Any) -> None:
        with self._lock:
            if len(self._pool) < self._max_connections:
                self._pool.append(conn)
    
    def close_all(self) -> None:
        with self._lock:
            while self._pool:
                conn = self._pool.pop()
                try:
                    conn.close()
                except:
                    pass


# ============================================================================
# DATA INTEGRITY FRAMEWORK
# ============================================================================

class DataValidator:
    """Data validation framework for trading data integrity"""
    
    @staticmethod
    def validate_price(price: float, min_price: float = 0, max_price: float = 1e12) -> bool:
        return isinstance(price, (int, float)) and min_price <= price <= max_price
    
    @staticmethod
    def validate_quantity(quantity: float, min_qty: float = 0, max_qty: float = 1e12) -> bool:
        return isinstance(quantity, (int, float)) and min_qty <= quantity <= max_qty
    
    @staticmethod
    def validate_timestamp(ts: int, max_age_seconds: int = 60) -> bool:
        current_ns = time.perf_counter_ns()
        age_ns = current_ns - ts
        return age_ns >= 0 and age_ns < max_age_seconds * 1e9
    
    @staticmethod
    def checksum(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


# Global deterministic random instance
_deterministic_rng = DeterministicRandom()

# ============================================================================
# END PHASE 2 SECURITY FRAMEWORK
# ============================================================================
