"""
Atomic Chaos Gate - Thread-Safe Position Sizing
Author: Erdinc Erdogan
Purpose: Provides thread-safe position sizing using atomic operations and sequence counters
to prevent race conditions between the Chaos Gate and Risk Manager.
References:
- Atomic Operations and Lock-Free Programming
- Thread Safety in Financial Systems
- Sequence Counters for Concurrency Control
Usage:
    gate = AtomicChaosGate()
    result = gate.compute_position_size(chaos_factor=0.8, risk_factor=0.5)
"""

import numpy as np
from threading import Lock, RLock
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class PositionSizeRequest:
    request_id: int
    timestamp: float
    chaos_factor: float
    risk_factor: float
    final_factor: float
    source: str

class AtomicChaosGate:
    """Thread-safe chaos gating with atomic position sizing."""
    
    def __init__(self, lyapunov_threshold: float = 0.5, hurst_trend: float = 0.6, hurst_mr: float = 0.4):
        self.lyapunov_threshold = lyapunov_threshold
        self.hurst_trend = hurst_trend
        self.hurst_mr = hurst_mr
        
        # Thread safety
        self._lock = RLock()
        self._sequence_counter = 0
        self._last_request: Optional[PositionSizeRequest] = None
        
        # Conflict resolution
        self._pending_chaos_factor = 1.0
        self._pending_risk_factor = 1.0
        self._last_update_time = 0.0
    
    def _get_sequence_id(self) -> int:
        """Atomic sequence counter."""
        with self._lock:
            self._sequence_counter += 1
            return self._sequence_counter
    
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
        """Submit chaos factor atomically."""
        with self._lock:
            seq_id = self._get_sequence_id()
            self._pending_chaos_factor = self.compute_chaos_factor(hurst, lyapunov, complexity)
            self._last_update_time = time.time()
            return seq_id
    
    def submit_risk_factor(self, risk_factor: float) -> int:
        """Submit risk factor atomically."""
        with self._lock:
            seq_id = self._get_sequence_id()
            self._pending_risk_factor = np.clip(risk_factor, 0.1, 2.0)
            self._last_update_time = time.time()
            return seq_id
    
    def resolve_position_factor(self) -> PositionSizeRequest:
        """Resolve final position factor atomically."""
        with self._lock:
            seq_id = self._get_sequence_id()
            
            # Conflict resolution: Use minimum of chaos and risk factors
            # This ensures conservative sizing when either system signals caution
            final_factor = min(self._pending_chaos_factor, self._pending_risk_factor)
            
            request = PositionSizeRequest(
                request_id=seq_id,
                timestamp=time.time(),
                chaos_factor=self._pending_chaos_factor,
                risk_factor=self._pending_risk_factor,
                final_factor=final_factor,
                source="ATOMIC_RESOLUTION"
            )
            
            self._last_request = request
            return request
    
    def get_last_request(self) -> Optional[PositionSizeRequest]:
        """Get last resolved request (thread-safe read)."""
        with self._lock:
            return self._last_request
