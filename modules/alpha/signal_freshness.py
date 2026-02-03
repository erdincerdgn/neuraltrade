"""
Signal Freshness Validator - TTL-Based Signal Aging System
Author: Erdinc Erdogan
Purpose: Implements time-to-live (TTL) mechanism with exponential decay for trading signals,
ensuring only fresh and valid signals are used in decision-making processes.
References:
- Exponential Decay Models in Signal Processing
- Time-To-Live (TTL) Protocols in Distributed Systems
- Signal Aging and Staleness Detection
Usage:
    validator = SignalFreshnessValidator(ttl_seconds=300, decay_rate=0.1)
    result = validator.validate(signal)
    if result.status == SignalStatus.FRESH: process_signal(result.signal)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import IntEnum
import time


class SignalStatus(IntEnum):
    """Signal freshness status."""
    FRESH = 0       # Within optimal window
    AGING = 1       # Approaching expiry, apply decay
    STALE = 2       # Past TTL, should be discarded
    EXPIRED = 3     # Way past TTL, must be rejected


@dataclass
class Signal:
    """Trading signal with timestamp and metadata."""
    signal_id: str
    agent_type: str
    direction: float          # -1 to 1 (bearish to bullish)
    confidence: float         # 0 to 1
    timestamp: float          # Unix timestamp
    features_hash: str = ""   # Hash of input features for dedup
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidatedSignal:
    """Signal after freshness validation."""
    signal: Signal
    status: SignalStatus
    age_seconds: float
    decay_factor: float
    adjusted_confidence: float
    adjusted_direction: float
    is_valid: bool
    rejection_reason: Optional[str] = None


class SignalFreshnessValidator:
    """
    Signal Freshness Validator with TTL and Exponential Decay.
    
    Implements the decay function:
    S_adj = S_raw × e^(-λ × Δt)
    
    Where:
    - S_raw = original signal strength
    - λ = decay_rate (configurable per regime)
    - Δt = time since signal generation
    
    Key Features:
    1. Configurable TTL per signal type
    2. Exponential decay for aging signals
    3. Regime-adaptive decay rates
    4. Duplicate signal detection
    5. Signal queue management
    
    Usage:
        validator = SignalFreshnessValidator(ttl_seconds=30)
        validated = validator.validate(signal)
        if validated.is_valid:
            process_signal(validated)
    """
    
    # Default TTL by agent type (seconds)
    DEFAULT_TTL = {
        "BULL": 30.0,
        "BEAR": 30.0,
        "NEUTRAL": 45.0,
        "MOMENTUM": 20.0,
        "MEAN_REVERSION": 60.0,
    }
    
    # Decay rates by volatility regime
    REGIME_DECAY_RATES = {
        "LOW_VOL": 0.02,      # Slow decay in calm markets
        "NORMAL": 0.05,       # Standard decay
        "HIGH_VOL": 0.10,     # Fast decay in volatile markets
        "CRISIS": 0.20,       # Very fast decay in crisis
    }
    
    def __init__(
        self,
        default_ttl: float = 30.0,
        decay_rate: float = 0.05,
        fresh_threshold: float = 0.5,    # Fraction of TTL considered "fresh"
        aging_threshold: float = 0.8,    # Fraction of TTL before "stale"
        min_confidence_threshold: float = 0.1,
        enable_dedup: bool = True,
        dedup_window: int = 100,
        regime_adaptive: bool = True
    ):
        self.default_ttl = default_ttl
        self.decay_rate = decay_rate
        self.fresh_threshold = fresh_threshold
        self.aging_threshold = aging_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_dedup = enable_dedup
        self.regime_adaptive = regime_adaptive
        
        # Deduplication tracking
        self.recent_signals: deque = deque(maxlen=dedup_window)
        
        # Statistics
        self.total_validated: int = 0
        self.total_rejected: int = 0
        self.rejection_reasons: Dict[str, int] = {}
        
        # Current regime (updated externally)
        self.current_regime: str = "NORMAL"
        
    def validate(
        self,
        signal: Signal,
        current_time: Optional[float] = None
    ) -> ValidatedSignal:
        """
        Validate signal freshness and apply decay.
        
        Args:
            signal: Signal to validate
            current_time: Current timestamp (uses time.time() if None)
            
        Returns:
            ValidatedSignal with adjusted values and status
        """
        if current_time is None:
            current_time = time.time()
        
        self.total_validated += 1
        
        # Calculate age
        age_seconds = current_time - signal.timestamp
        
        # Get TTL for this signal type
        ttl = self.DEFAULT_TTL.get(signal.agent_type, self.default_ttl)
        
        # Check for duplicate
        if self.enable_dedup and self._is_duplicate(signal):
            return self._create_rejected_signal(
                signal, age_seconds, "DUPLICATE"
            )
        
        # Determine status based on age
        status = self._determine_status(age_seconds, ttl)
        
        # Reject expired signals
        if status == SignalStatus.EXPIRED:
            return self._create_rejected_signal(
                signal, age_seconds, "EXPIRED"
            )
        
        # Reject stale signals
        if status == SignalStatus.STALE:
            return self._create_rejected_signal(
                signal, age_seconds, "STALE"
            )
        
        # Calculate decay factor
        decay_rate = self._get_decay_rate()
        decay_factor = self._calculate_decay(age_seconds, decay_rate)
        
        # Apply decay to confidence and direction
        adjusted_confidence = signal.confidence * decay_factor
        adjusted_direction = signal.direction * decay_factor
        
        # Check minimum confidence threshold
        if adjusted_confidence < self.min_confidence_threshold:
            return self._create_rejected_signal(
                signal, age_seconds, "LOW_CONFIDENCE"
            )
        
        # Track for deduplication
        if self.enable_dedup:
            self.recent_signals.append(signal.features_hash)
        
        return ValidatedSignal(
            signal=signal,
            status=status,
            age_seconds=age_seconds,
            decay_factor=decay_factor,
            adjusted_confidence=adjusted_confidence,
            adjusted_direction=adjusted_direction,
            is_valid=True,
            rejection_reason=None
        )
    
    def validate_batch(
        self,
        signals: List[Signal],
        current_time: Optional[float] = None
    ) -> List[ValidatedSignal]:
        """Validate multiple signals."""
        if current_time is None:
            current_time = time.time()
        
        return [self.validate(s, current_time) for s in signals]
    
    def filter_valid(
        self,
        signals: List[Signal],
        current_time: Optional[float] = None
    ) -> List[ValidatedSignal]:
        """Validate and return only valid signals."""
        validated = self.validate_batch(signals, current_time)
        return [v for v in validated if v.is_valid]
    
    def set_regime(self, regime: str):
        """Update current volatility regime for adaptive decay."""
        if regime in self.REGIME_DECAY_RATES:
            self.current_regime = regime
    
    def _determine_status(self, age_seconds: float, ttl: float) -> SignalStatus:
        """Determine signal status based on age."""
        age_ratio = age_seconds / ttl
        
        if age_ratio < self.fresh_threshold:
            return SignalStatus.FRESH
        elif age_ratio < self.aging_threshold:
            return SignalStatus.AGING
        elif age_ratio < 1.0:
            return SignalStatus.STALE
        else:
            return SignalStatus.EXPIRED
    
    def _calculate_decay(self, age_seconds: float, decay_rate: float) -> float:
        """
        Calculate exponential decay factor.
        
        S_adj = S_raw × e^(-λ × Δt)
        """
        decay_factor = np.exp(-decay_rate * age_seconds)
        return np.clip(decay_factor, 0.0, 1.0)
    
    def _get_decay_rate(self) -> float:
        """Get decay rate based on current regime."""
        if self.regime_adaptive:
            return self.REGIME_DECAY_RATES.get(self.current_regime, self.decay_rate)
        return self.decay_rate
    
    def _is_duplicate(self, signal: Signal) -> bool:
        """Check if signal is a duplicate."""
        if not signal.features_hash:
            return False
        return signal.features_hash in self.recent_signals
    
    def _create_rejected_signal(
        self,
        signal: Signal,
        age_seconds: float,
        reason: str
    ) -> ValidatedSignal:
        """Create a rejected signal result."""
        self.total_rejected += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        
        return ValidatedSignal(
            signal=signal,
            status=SignalStatus.EXPIRED if reason in ["EXPIRED", "STALE"] else SignalStatus.STALE,
            age_seconds=age_seconds,
            decay_factor=0.0,
            adjusted_confidence=0.0,
            adjusted_direction=0.0,
            is_valid=False,
            rejection_reason=reason
        )
    
    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        rejection_rate = self.total_rejected / max(self.total_validated, 1)
        return {
            "total_validated": self.total_validated,
            "total_rejected": self.total_rejected,
            "rejection_rate": rejection_rate,
            "rejection_reasons": dict(self.rejection_reasons),
            "current_regime": self.current_regime,
            "current_decay_rate": self._get_decay_rate(),
        }
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.total_validated = 0
        self.total_rejected = 0
        self.rejection_reasons = {}


class SignalQueue:
    """
    Priority queue for validated signals.
    
    Maintains a queue of fresh signals sorted by adjusted confidence,
    automatically removing stale signals.
    """
    
    def __init__(
        self,
        validator: SignalFreshnessValidator,
        max_size: int = 100
    ):
        self.validator = validator
        self.max_size = max_size
        self.queue: List[ValidatedSignal] = []
        
    def add(self, signal: Signal) -> bool:
        """Add signal to queue if valid."""
        validated = self.validator.validate(signal)
        
        if not validated.is_valid:
            return False
        
        self.queue.append(validated)
        
        # Sort by adjusted confidence (descending)
        self.queue.sort(key=lambda x: x.adjusted_confidence, reverse=True)
        
        # Trim to max size
        if len(self.queue) > self.max_size:
            self.queue = self.queue[:self.max_size]
        
        return True
    
    def get_top(self, n: int = 1) -> List[ValidatedSignal]:
        """Get top N signals by confidence."""
        self._refresh()
        return self.queue[:n]
    
    def get_consensus(self) -> Tuple[float, float]:
        """
        Get consensus direction and confidence from queue.
        
        Returns:
            Tuple of (weighted_direction, average_confidence)
        """
        self._refresh()
        
        if not self.queue:
            return 0.0, 0.0
        
        total_weight = sum(s.adjusted_confidence for s in self.queue)
        if total_weight < 1e-10:
            return 0.0, 0.0
        
        weighted_direction = sum(
            s.adjusted_direction * s.adjusted_confidence 
            for s in self.queue
        ) / total_weight
        
        avg_confidence = total_weight / len(self.queue)
        
        return weighted_direction, avg_confidence
    
    def _refresh(self):
        """Remove stale signals from queue."""
        current_time = time.time()
        self.queue = [
            s for s in self.queue
            if self.validator.validate(s.signal, current_time).is_valid
        ]
    
    def clear(self):
        """Clear the queue."""
        self.queue = []
    
    def __len__(self) -> int:
        return len(self.queue)
