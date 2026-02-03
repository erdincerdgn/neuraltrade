"""
Regime Stability Filter with Hysteresis
Author: Erdinc Erdogan
Purpose: Prevents rapid regime oscillation (whipsaw) using minimum stay duration, probability lead thresholds, and cooldown periods.
References:
- Hysteresis in Signal Processing
- Regime Transition Stabilization
- Whipsaw Prevention Techniques
Usage:
    filter = RegimeStabilityFilter(min_stay_duration=5, lead_threshold=0.15)
    stable_state = filter.filter(raw_state, state_probs)
"""

# ============================================================================
# REGIME STABILITY FILTER - Hysteresis Mechanism
# Prevents rapid state oscillation (whipsaw) during regime transitions
# ============================================================================

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class RegimeTransition:
    """Record of a regime transition event."""
    timestamp: datetime
    from_state: int
    to_state: int
    confidence: float
    probability_lead: float


class RegimeStabilityFilter:
    """
    Hysteresis-based regime stability filter to prevent whipsaw trading.
    
    Implements two stabilization mechanisms:
    1. Minimum Stay Duration: Regime must persist for N periods before switching
    2. Probability Lead Threshold: New regime must have significant probability advantage
    
    Mathematical Foundation:
    - Transition allowed if: P(new_state) - P(current_state) > lead_threshold
    - AND: time_in_current_state >= min_stay_duration
    
    Usage:
        filter = RegimeStabilityFilter(min_stay_duration=5, lead_threshold=0.15)
        stable_state = filter.filter(raw_state, state_probs)
    """
    
    def __init__(
        self,
        n_states: int = 8,
        min_stay_duration: int = 5,
        lead_threshold: float = 0.15,
        cooldown_periods: int = 3,
        max_transitions_per_hour: int = 4,
        enable_adaptive_threshold: bool = True
    ):
        self.n_states = n_states
        self.min_stay_duration = min_stay_duration
        self.lead_threshold = lead_threshold
        self.cooldown_periods = cooldown_periods
        self.max_transitions_per_hour = max_transitions_per_hour
        self.enable_adaptive_threshold = enable_adaptive_threshold
        
        # State tracking
        self.current_state: Optional[int] = None
        self.time_in_state: int = 0
        self.cooldown_remaining: int = 0
        self.transition_history: deque = deque(maxlen=100)
        self.hourly_transitions: deque = deque(maxlen=60)  # Track last 60 periods
        
        # Adaptive threshold state
        self.volatility_ema: float = 0.0
        self.volatility_alpha: float = 0.1
        
    def filter(
        self, 
        raw_state: int, 
        state_probs: np.ndarray,
        volatility: Optional[float] = None
    ) -> Tuple[int, dict]:
        """
        Apply stability filter to raw regime detection.
        
        Args:
            raw_state: Raw detected state from HMM/Neural-HMM
            state_probs: Full probability distribution over states
            volatility: Optional current volatility for adaptive threshold
            
        Returns:
            Tuple of (filtered_state, diagnostics_dict)
        """
        state_probs = self._validate_probs(state_probs)
        
        # Initialize on first call
        if self.current_state is None:
            self.current_state = raw_state
            self.time_in_state = 1
            return raw_state, self._create_diagnostics("INITIALIZED", raw_state, state_probs)
        
        # Update adaptive threshold if enabled
        if self.enable_adaptive_threshold and volatility is not None:
            self._update_adaptive_threshold(volatility)
        
        # Increment time in current state
        self.time_in_state += 1
        
        # Decrement cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        
        # Check if transition is requested
        if raw_state == self.current_state:
            return self.current_state, self._create_diagnostics("STABLE", raw_state, state_probs)
        
        # Evaluate transition conditions
        transition_allowed, reason = self._evaluate_transition(raw_state, state_probs)
        
        if transition_allowed:
            # Record transition
            self._record_transition(raw_state, state_probs)
            
            # Execute transition
            old_state = self.current_state
            self.current_state = raw_state
            self.time_in_state = 1
            self.cooldown_remaining = self.cooldown_periods
            
            return raw_state, self._create_diagnostics(
                f"TRANSITION: {old_state} → {raw_state}", raw_state, state_probs
            )
        else:
            # Block transition
            return self.current_state, self._create_diagnostics(
                f"BLOCKED: {reason}", raw_state, state_probs
            )
    
    def _evaluate_transition(self, new_state: int, probs: np.ndarray) -> Tuple[bool, str]:
        """Evaluate if transition should be allowed."""
        
        # Check 1: Cooldown period
        if self.cooldown_remaining > 0:
            return False, f"COOLDOWN ({self.cooldown_remaining} periods remaining)"
        
        # Check 2: Minimum stay duration
        if self.time_in_state < self.min_stay_duration:
            return False, f"MIN_STAY ({self.time_in_state}/{self.min_stay_duration})"
        
        # Check 3: Probability lead threshold
        current_prob = probs[self.current_state]
        new_prob = probs[new_state]
        lead = new_prob - current_prob
        
        effective_threshold = self._get_effective_threshold()
        
        if lead < effective_threshold:
            return False, f"LEAD_THRESHOLD ({lead:.3f} < {effective_threshold:.3f})"
        
        # Check 4: Hourly transition limit
        recent_transitions = sum(1 for _ in self.hourly_transitions)
        if recent_transitions >= self.max_transitions_per_hour:
            return False, f"HOURLY_LIMIT ({recent_transitions}/{self.max_transitions_per_hour})"
        
        return True, "APPROVED"
    
    def _get_effective_threshold(self) -> float:
        """Get adaptive threshold based on volatility."""
        if not self.enable_adaptive_threshold:
            return self.lead_threshold
        
        # Higher volatility = higher threshold (more conservative)
        vol_multiplier = 1.0 + self.volatility_ema
        return self.lead_threshold * np.clip(vol_multiplier, 0.5, 2.0)
    
    def _update_adaptive_threshold(self, volatility: float):
        """Update volatility EMA for adaptive threshold."""
        normalized_vol = np.clip(volatility / 0.20, 0, 3)  # Normalize to ~20% baseline
        self.volatility_ema = (
            self.volatility_alpha * normalized_vol + 
            (1 - self.volatility_alpha) * self.volatility_ema
        )
    
    def _record_transition(self, new_state: int, probs: np.ndarray):
        """Record transition for history tracking."""
        transition = RegimeTransition(
            timestamp=datetime.now(),
            from_state=self.current_state,
            to_state=new_state,
            confidence=probs[new_state],
            probability_lead=probs[new_state] - probs[self.current_state]
        )
        self.transition_history.append(transition)
        self.hourly_transitions.append(1)
    
    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Validate and normalize probability array."""
        probs = np.asarray(probs)
        if len(probs) != self.n_states:
            return np.ones(self.n_states) / self.n_states
        probs = np.maximum(probs, 1e-10)
        return probs / probs.sum()
    
    def _create_diagnostics(self, action: str, raw_state: int, probs: np.ndarray) -> dict:
        """Create diagnostics dictionary."""
        return {
            "action": action,
            "raw_state": raw_state,
            "filtered_state": self.current_state,
            "time_in_state": self.time_in_state,
            "cooldown_remaining": self.cooldown_remaining,
            "effective_threshold": self._get_effective_threshold(),
            "current_prob": float(probs[self.current_state]) if self.current_state is not None else 0,
            "raw_prob": float(probs[raw_state]),
            "total_transitions": len(self.transition_history)
        }
    
    def reset(self):
        """Reset filter state."""
        self.current_state = None
        self.time_in_state = 0
        self.cooldown_remaining = 0
        self.transition_history.clear()
        self.hourly_transitions.clear()
        self.volatility_ema = 0.0
    
    def get_transition_rate(self, lookback: int = 60) -> float:
        """Get recent transition rate (transitions per period)."""
        recent = list(self.transition_history)[-lookback:]
        return len(recent) / max(lookback, 1)
