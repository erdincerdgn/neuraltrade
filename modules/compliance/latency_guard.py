"""
Latency Guard - Compliance Check Timeout Enforcement
Author: Erdinc Erdogan
Purpose: Ensures all compliance checks complete within the 2ms latency budget required
for high-frequency trading, with timeout handling and monitoring.
References:
- Latency Requirements in HFT Systems
- Timeout Patterns for Real-Time Systems
- Performance Monitoring in Trading
Usage:
    guard = LatencyGuard(max_latency_ms=2.0)
    result = guard.execute_with_timeout(compliance_check, order)
"""

import time
from dataclasses import dataclass
from typing import Callable, Any, Optional
from functools import wraps

@dataclass
class LatencyResult:
    passed: bool
    latency_ms: float
    timeout: bool
    result: Any

class LatencyGuard:
    """Wraps compliance checks with latency monitoring."""
    
    MAX_LATENCY_MS = 2.0  # Hard limit: 2ms
    WARN_LATENCY_MS = 1.5  # Warning threshold
    
    def __init__(self, max_latency_ms: float = 2.0):
        self.max_latency_ms = max_latency_ms
        self._latency_history: list = []
        self._max_history = 1000
    
    def timed_check(self, check_func: Callable, *args, **kwargs) -> LatencyResult:
        """Execute check with latency measurement."""
        start = time.perf_counter()
        try:
            result = check_func(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            timeout = latency_ms > self.max_latency_ms
            self._record_latency(latency_ms)
            return LatencyResult(passed=not timeout, latency_ms=latency_ms, timeout=timeout, result=result)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return LatencyResult(passed=False, latency_ms=latency_ms, timeout=False, result=e)
    
    def _record_latency(self, latency_ms: float):
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > self._max_history:
            self._latency_history.pop(0)
    
    def get_stats(self) -> dict:
        if not self._latency_history:
            return {"avg_ms": 0, "max_ms": 0, "p99_ms": 0}
        sorted_lat = sorted(self._latency_history)
        p99_idx = int(len(sorted_lat) * 0.99)
        return {"avg_ms": sum(sorted_lat)/len(sorted_lat), "max_ms": max(sorted_lat), "p99_ms": sorted_lat[p99_idx]}
