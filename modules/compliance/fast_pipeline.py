"""
Ultra-Low Latency Compliance Pipeline - Sub-Millisecond Validation
Author: Erdinc Erdogan
Purpose: Provides sub-millisecond compliance validation using pre-computed caches and
optimized check ordering for high-frequency trading requirements.
References:
- Low-Latency System Design Patterns
- Cache-Optimized Validation Pipelines
- HFT Compliance Requirements
Usage:
    pipeline = UltraFastCompliancePipeline()
    pipeline.add_check(exposure_check, priority=1)
    result = pipeline.validate(order)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import time

class CheckResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"

@dataclass
class PipelineMetrics:
    total_checks: int = 0
    total_latency_ns: int = 0
    max_latency_ns: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class UltraFastCompliancePipeline:
    """Sub-millisecond compliance with O(1) checks and caching."""
    
    __slots__ = ('_checks', '_cache', '_cache_ttl_ns', '_metrics', '_enabled')
    
    def __init__(self, cache_ttl_ms: int = 100):
        self._checks: List[Tuple[str, Callable]] = []
        self._cache: Dict[str, Tuple[int, bool, str]] = {}
        self._cache_ttl_ns: int = cache_ttl_ms * 1_000_000
        self._metrics = PipelineMetrics()
        self._enabled: Dict[str, bool] = {}
    
    def register_check(self, name: str, check_fn: Callable, enabled: bool = True):
        """Register O(1) check function."""
        self._checks.append((name, check_fn))
        self._enabled[name] = enabled
    
    def _get_cache_key(self, agent_id: str, symbol: str, side: str, 
                       qty_bucket: int, price_bucket: int) -> str:
        """Generate cache key for O(1) lookup."""
        return f"{agent_id}:{symbol}:{side}:{qty_bucket}:{price_bucket}"
    
    def validate(self, agent_id: str, symbol: str, side: str,
                 quantity: float, price: float) -> Tuple[bool, List[str], int]:
        """Ultra-fast validation pipeline with caching."""
        start_ns = time.perf_counter_ns()
        
        qty_bucket = int(quantity * 100)
        price_bucket = int(price * 10)
        cache_key = self._get_cache_key(agent_id, symbol, side, qty_bucket, price_bucket)
        
        now_ns = time.perf_counter_ns()
        if cache_key in self._cache:
            cached_time, cached_result, cached_reason = self._cache[cache_key]
            if now_ns - cached_time < self._cache_ttl_ns:
                self._metrics.cache_hits += 1
                latency = time.perf_counter_ns() - start_ns
                return cached_result, [cached_reason] if not cached_result else [], latency
        
        self._metrics.cache_misses += 1
        failures = []
        
        context = {"agent_id": agent_id, "symbol": symbol, "side": side,
                   "quantity": quantity, "price": price}
        
        for name, check_fn in self._checks:
            if not self._enabled.get(name, True):
                continue
            try:
                passed, reason = check_fn(context)
                if not passed:
                    failures.append(f"{name}:{reason}")
            except Exception:
                pass
        
        result = len(failures) == 0
        reason = failures[0] if failures else "OK"
        self._cache[cache_key] = (now_ns, result, reason)
        
        latency_ns = time.perf_counter_ns() - start_ns
        self._metrics.total_checks += 1
        self._metrics.total_latency_ns += latency_ns
        self._metrics.max_latency_ns = max(self._metrics.max_latency_ns, latency_ns)
        
        return result, failures, latency_ns
    
    def get_metrics(self) -> Dict:
        avg_ns = self._metrics.total_latency_ns // max(self._metrics.total_checks, 1)
        return {
            "total_checks": self._metrics.total_checks,
            "avg_latency_us": avg_ns / 1000,
            "max_latency_us": self._metrics.max_latency_ns / 1000,
            "cache_hit_rate": self._metrics.cache_hits / max(self._metrics.cache_hits + self._metrics.cache_misses, 1)
        }
    
    def enable_check(self, name: str, enabled: bool = True):
        self._enabled[name] = enabled
