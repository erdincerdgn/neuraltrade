"""
Safe Chaos Gate - Deadlock Prevention with Graceful Degradation
Author: Erdinc Erdogan
Purpose: Implements timeout-based locking with graceful degradation to prevent deadlocks
in the chaos gating system while maintaining trading system availability.
References:
- Deadlock Prevention Algorithms
- Timeout-Based Locking Patterns
- Graceful Degradation in Trading Systems
Usage:
    gate = SafeChaosGate(timeout_seconds=0.1)
    result = gate.compute_safe_position(chaos_factor=0.8, risk_factor=0.5)
"""

import numpy as np
from threading import RLock
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class SafePositionResult:
    """Thread-safe position sizing result."""
    request_id: int
    chaos_factor: float
    risk_factor: float
    final_factor: float
    acquired_lock: bool
    used_fallback: bool

class SafeChaosGate:
    """Deadlock-free chaos gate with timeout and fallback."""
    
    DEFAULT_FACTOR = 1.0
    LOCK_TIMEOUT = 0.1  # 100ms timeout
    
    def __init__(self, lyapunov_threshold: float = 0.5, hurst_trend: float = 0.6, hurst_mr: float = 0.4):
        self.lyapunov_threshold = lyapunov_threshold
        self.hurst_trend = hurst_trend
        self.hurst_mr = hurst_mr
        
        self._lock = RLock()
        self._sequence_counter = 0
        self._pending_chaos = self.DEFAULT_FACTOR
        self._pending_risk = self.DEFAULT_FACTOR
        self._last_valid_factor = self.DEFAULT_FACTOR
    
    def _try_acquire(self) -> bool:
        """Try to acquire lock with timeout."""
        return self._lock.acquire(timeout=self.LOCK_TIMEOUT)
    
    def _release(self):
        """Release lock if held."""
        try:
            self._lock.release()
        except RuntimeError:
            pass  # Lock not held
    
    def compute_chaos_factor(self, hurst: float, lyapunov: float, complexity: float) -> float:
        """Compute chaos-based position factor."""
        factor = 1.0
        if lyapunov > self.lyapunov_threshold:
            factor *= 0.5
        if hurst > self.hurst_trend:
            factor *= 1.2
        elif hurst < self.hurst_mr:
            factor *= 0.8
        if complexity > 20:
            factor *= 0.7
        return np.clip(factor, 0.1, 1.5)
    
    def submit_chaos_factor(self, hurst: float, lyapunov: float, complexity: float) -> int:
        """Submit chaos factor with timeout protection."""
        if self._try_acquire():
            try:
                self._sequence_counter += 1
                self._pending_chaos = self.compute_chaos_factor(hurst, lyapunov, complexity)
                return self._sequence_counter
            finally:
                self._release()
        return -1  # Failed to acquire lock
    
    def submit_risk_factor(self, risk_factor: float) -> int:
        """Submit risk factor with timeout protection."""
        if self._try_acquire():
            try:
                self._sequence_counter += 1
                self._pending_risk = np.clip(risk_factor, 0.1, 2.0)
                return self._sequence_counter
            finally:
                self._release()
        return -1
    
    def resolve_position_factor(self) -> SafePositionResult:
        """Resolve final position factor with fallback."""
        acquired = self._try_acquire()
        
        if acquired:
            try:
                self._sequence_counter += 1
                final = min(self._pending_chaos, self._pending_risk)
                self._last_valid_factor = final
                
                return SafePositionResult(
                    request_id=self._sequence_counter,
                    chaos_factor=self._pending_chaos,
                    risk_factor=self._pending_risk,
                    final_factor=final,
                    acquired_lock=True,
                    used_fallback=False
                )
            finally:
                self._release()
        else:
            # Fallback: use last valid factor
            return SafePositionResult(
                request_id=-1,
                chaos_factor=self._pending_chaos,
                risk_factor=self._pending_risk,
                final_factor=self._last_valid_factor,
                acquired_lock=False,
                used_fallback=True
            )
    
    def get_status(self) -> dict:
        """Get gate status (non-blocking)."""
        return {
            "pending_chaos": self._pending_chaos,
            "pending_risk": self._pending_risk,
            "last_valid": self._last_valid_factor,
            "sequence": self._sequence_counter
        }
