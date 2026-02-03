"""
Atomic Compliance Gate - Thread-Safe Order Validation
Author: Erdinc Erdogan
Purpose: Provides thread-safe compliance checking with spinlock mechanisms to prevent
race conditions during concurrent order validation in multi-agent systems.
References:
- Spinlock and Lock-Free Programming Patterns
- Atomic Operations in Concurrent Systems
- Thread Safety in Trading Systems
Usage:
    gate = AtomicComplianceGate(max_concurrent=1000)
    gate.register_check(exposure_check)
    result = gate.validate_order(order)
"""
import threading
import time
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass

@dataclass
class GateMetrics:
    total_requests: int = 0
    blocked_by_lock: int = 0
    race_prevented: int = 0

class AtomicComplianceGate:
    """Thread-safe compliance gate with atomic operations."""
    
    __slots__ = ('_lock', '_in_flight', '_metrics', '_checks', '_max_concurrent')
    
    def __init__(self, max_concurrent: int = 1000):
        self._lock = threading.Lock()
        self._in_flight: Dict[str, float] = {}  # order_id -> start_time
        self._metrics = GateMetrics()
        self._checks: List[Callable] = []
        self._max_concurrent = max_concurrent
    
    def register_check(self, check_fn: Callable):
        self._checks.append(check_fn)
    
    def acquire(self, order_id: str, timeout_ms: float = 10) -> Tuple[bool, str]:
        """Atomic acquire with timeout - prevents race conditions."""
        deadline = time.perf_counter() + (timeout_ms / 1000)
        
        while time.perf_counter() < deadline:
            acquired = self._lock.acquire(blocking=False)
            if acquired:
                try:
                    self._metrics.total_requests += 1
                    if len(self._in_flight) >= self._max_concurrent:
                        self._metrics.blocked_by_lock += 1
                        return False, "GATE-001: Max concurrent exceeded"
                    if order_id in self._in_flight:
                        self._metrics.race_prevented += 1
                        return False, "GATE-002: Duplicate order in flight"
                    self._in_flight[order_id] = time.perf_counter()
                    return True, "OK"
                finally:
                    self._lock.release()
            time.sleep(0.0001)  # 100μs spinlock
        
        self._metrics.blocked_by_lock += 1
        return False, "GATE-003: Lock timeout"
    
    def release(self, order_id: str):
        with self._lock:
            self._in_flight.pop(order_id, None)
    
    def validate_atomic(self, order_id: str, ctx: Dict) -> Tuple[bool, List[str]]:
        """Atomic validation - order cannot bypass during context switch."""
        acquired, msg = self.acquire(order_id)
        if not acquired:
            return False, [msg]
        try:
            failures = []
            for check in self._checks:
                ok, reason = check(ctx)
                if not ok:
                    failures.append(reason)
            return len(failures) == 0, failures
        finally:
            self.release(order_id)
    
    def get_metrics(self) -> Dict:
        return {"total": self._metrics.total_requests, 
                "blocked": self._metrics.blocked_by_lock,
                "race_prevented": self._metrics.race_prevented,
                "in_flight": len(self._in_flight)}
