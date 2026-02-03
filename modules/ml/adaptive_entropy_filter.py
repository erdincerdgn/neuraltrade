"""
Adaptive Entropy Filter for Regime Uncertainty
Author: Erdinc Erdogan
Purpose: Uses Shannon entropy of regime probabilities to detect uncertainty and dynamically adjust position sizing during regime transitions.
References:
- Shannon Entropy for Uncertainty Quantification
- Adaptive Position Sizing
- Regime Transition Risk Management
Usage:
    filter = AdaptiveEntropyFilter(n_states=8)
    state = filter.update(regime_probs)
    if state.deleverage_signal: reduce_positions(state.position_scale)
"""

# ============================================================================
# ADAPTIVE ENTROPY FILTER - Dynamic Regime Uncertainty Gating
# Implements entropy-based position scaling during regime uncertainty
# Phase 7C: Beyond Tier-1 Enhancement
# ============================================================================

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import deque
from enum import IntEnum


class EntropyLevel(IntEnum):
    """Entropy classification levels."""
    LOW = 0           # Clear regime, full trading
    MODERATE = 1      # Some uncertainty, reduced sizing
    HIGH = 2          # High uncertainty, minimal trading
    CRITICAL = 3      # Extreme uncertainty, halt new positions


@dataclass
class EntropyState:
    """Current entropy filter state."""
    entropy: float
    entropy_gradient: float
    level: EntropyLevel
    position_scale: float
    deleverage_signal: bool
    max_entropy: float
    dynamic_threshold: float
    confidence: float


class AdaptiveEntropyFilter:
    """
    Adaptive Entropy Filter for regime uncertainty management.
    
    Uses Shannon entropy of regime probabilities to detect uncertainty
    and dynamically adjust position sizing. When entropy exceeds
    adaptive thresholds, the system de-leverages to protect capital.
    
    Mathematical Foundation:
    H(P) = -Σ p_i * log(p_i)  (Shannon Entropy)
    
    Max entropy for N states = log(N) (uniform distribution)
    Normalized entropy = H(P) / log(N)
    
    Position scaling:
    scale = max(0, 1 - (normalized_entropy - threshold) / (1 - threshold))
    
    Usage:
        filter = AdaptiveEntropyFilter(n_states=8)
        state = filter.update(regime_probs)
        if state.deleverage_signal:
            reduce_positions(state.position_scale)
    """
    
    def __init__(
        self,
        n_states: int = 8,
        base_threshold: float = 0.6,
        critical_threshold: float = 0.85,
        gradient_sensitivity: float = 0.5,
        ema_alpha: float = 0.1,
        lookback_window: int = 50,
        min_scale: float = 0.1,
        adaptive_threshold: bool = True
    ):
        self.n_states = n_states
        self.max_entropy = np.log(n_states)
        self.base_threshold = base_threshold
        self.critical_threshold = critical_threshold
        self.gradient_sensitivity = gradient_sensitivity
        self.ema_alpha = ema_alpha
        self.lookback_window = lookback_window
        self.min_scale = min_scale
        self.adaptive_threshold = adaptive_threshold
        
        # State
        self.entropy_history: deque = deque(maxlen=lookback_window)
        self.ema_entropy: float = 0.0
        self.prev_entropy: Optional[float] = None
        self.dynamic_threshold: float = base_threshold
        self.last_state: Optional[EntropyState] = None
        
    def update(
        self,
        regime_probs: np.ndarray,
        volatility: Optional[float] = None
    ) -> EntropyState:
        """
        Update entropy filter with new regime probabilities.
        
        Args:
            regime_probs: Probability distribution over regimes
            volatility: Optional current volatility for adaptive threshold
            
        Returns:
            EntropyState with current filter state
        """
        # Validate and normalize probabilities
        probs = self._validate_probs(regime_probs)
        
        # Calculate Shannon entropy
        entropy = self._compute_entropy(probs)
        normalized_entropy = entropy / self.max_entropy
        
        # Update EMA
        self.ema_entropy = (
            self.ema_alpha * normalized_entropy +
            (1 - self.ema_alpha) * self.ema_entropy
        )
        
        # Calculate entropy gradient
        gradient = 0.0
        if self.prev_entropy is not None:
            gradient = normalized_entropy - self.prev_entropy
        self.prev_entropy = normalized_entropy
        
        # Update history
        self.entropy_history.append(normalized_entropy)
        
        # Update adaptive threshold
        if self.adaptive_threshold:
            self._update_dynamic_threshold(volatility)
        
        # Classify entropy level
        level = self._classify_level(self.ema_entropy, gradient)
        
        # Calculate position scale
        position_scale = self._calculate_scale(self.ema_entropy, gradient)
        
        # Determine deleverage signal
        deleverage_signal = self._should_deleverage(level, gradient)
        
        # Calculate confidence (inverse of entropy)
        confidence = 1.0 - self.ema_entropy
        
        state = EntropyState(
            entropy=normalized_entropy,
            entropy_gradient=gradient,
            level=level,
            position_scale=position_scale,
            deleverage_signal=deleverage_signal,
            max_entropy=self.max_entropy,
            dynamic_threshold=self.dynamic_threshold,
            confidence=confidence
        )
        
        self.last_state = state
        return state
    
    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute Shannon entropy."""
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))
    
    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Validate and normalize probability array."""
        probs = np.asarray(probs)
        probs = np.maximum(probs, 1e-10)
        return probs / probs.sum()
    
    def _update_dynamic_threshold(self, volatility: Optional[float]):
        """Update threshold based on recent entropy and volatility."""
        if len(self.entropy_history) < 10:
            return
        
        # Use rolling statistics
        recent = list(self.entropy_history)[-20:]
        mean_entropy = np.mean(recent)
        std_entropy = np.std(recent)
        
        # Adaptive threshold: higher when entropy is stable, lower when volatile
        stability_factor = 1.0 / (1.0 + std_entropy * 5)
        
        # Volatility adjustment
        vol_factor = 1.0
        if volatility is not None:
            vol_factor = 1.0 - np.clip((volatility - 0.20) / 0.40, 0, 0.3)
        
        self.dynamic_threshold = np.clip(
            self.base_threshold * stability_factor * vol_factor,
            0.4,  # Minimum threshold
            0.8   # Maximum threshold
        )
    
    def _classify_level(self, entropy: float, gradient: float) -> EntropyLevel:
        """Classify entropy into discrete levels."""
        # Adjust for gradient (rising entropy is worse)
        adjusted_entropy = entropy + gradient * self.gradient_sensitivity
        
        if adjusted_entropy >= self.critical_threshold:
            return EntropyLevel.CRITICAL
        elif adjusted_entropy >= self.dynamic_threshold + 0.15:
            return EntropyLevel.HIGH
        elif adjusted_entropy >= self.dynamic_threshold:
            return EntropyLevel.MODERATE
        else:
            return EntropyLevel.LOW
    
    def _calculate_scale(self, entropy: float, gradient: float) -> float:
        """Calculate position scaling factor."""
        # Adjust for gradient
        adjusted_entropy = entropy + gradient * self.gradient_sensitivity
        
        if adjusted_entropy <= self.dynamic_threshold:
            return 1.0
        
        if adjusted_entropy >= self.critical_threshold:
            return self.min_scale
        
        # Linear interpolation between threshold and critical
        range_size = self.critical_threshold - self.dynamic_threshold
        excess = adjusted_entropy - self.dynamic_threshold
        reduction = excess / range_size
        
        scale = 1.0 - reduction * (1.0 - self.min_scale)
        return np.clip(scale, self.min_scale, 1.0)
    
    def _should_deleverage(self, level: EntropyLevel, gradient: float) -> bool:
        """Determine if aggressive deleveraging is needed."""
        # Deleverage on high/critical entropy
        if level >= EntropyLevel.HIGH:
            return True
        
        # Deleverage on rapidly rising entropy
        if gradient > 0.1 and level >= EntropyLevel.MODERATE:
            return True
        
        return False
    
    def get_orchestrator_signal(self) -> Dict:
        """
        Get entropy signal for main orchestrator integration.
        
        Returns:
            Dict with entropy gating signals
        """
        if self.last_state is None:
            return {
                "entropy_level": 0,
                "position_scale": 1.0,
                "deleverage": False,
                "confidence": 1.0
            }
        
        return {
            "entropy_level": int(self.last_state.level),
            "entropy_value": self.last_state.entropy,
            "entropy_gradient": self.last_state.entropy_gradient,
            "position_scale": self.last_state.position_scale,
            "deleverage": self.last_state.deleverage_signal,
            "confidence": self.last_state.confidence,
            "dynamic_threshold": self.last_state.dynamic_threshold
        }
    
    def reset(self):
        """Reset filter state."""
        self.entropy_history.clear()
        self.ema_entropy = 0.0
        self.prev_entropy = None
        self.dynamic_threshold = self.base_threshold
        self.last_state = None


class EntropyGatedOrchestrator:
    """
    Entropy-gated orchestrator wrapper for main trading system.
    
    Wraps the main orchestrator to apply entropy-based position gating
    before any trading decisions are executed.
    """
    
    def __init__(
        self,
        entropy_filter: AdaptiveEntropyFilter,
        emergency_deleverage_rate: float = 0.2
    ):
        self.entropy_filter = entropy_filter
        self.emergency_deleverage_rate = emergency_deleverage_rate
        self.is_deleveraging: bool = False
        self.deleverage_start_time: Optional[float] = None
        
    def gate_position(
        self,
        proposed_position: float,
        regime_probs: np.ndarray,
        volatility: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Apply entropy gating to proposed position.
        
        Args:
            proposed_position: Original position size
            regime_probs: Current regime probabilities
            volatility: Current volatility
            
        Returns:
            Tuple of (gated_position, diagnostics)
        """
        state = self.entropy_filter.update(regime_probs, volatility)
        
        # Apply scaling
        gated_position = proposed_position * state.position_scale
        
        # Handle deleverage signal
        if state.deleverage_signal and not self.is_deleveraging:
            self.is_deleveraging = True
            self.deleverage_start_time = datetime.now().timestamp()
        elif not state.deleverage_signal and state.level <= EntropyLevel.LOW:
            self.is_deleveraging = False
            self.deleverage_start_time = None
        
        diagnostics = {
            "original_position": proposed_position,
            "gated_position": gated_position,
            "scale_applied": state.position_scale,
            "entropy_level": state.level.name,
            "is_deleveraging": self.is_deleveraging,
            **self.entropy_filter.get_orchestrator_signal()
        }
        
        return gated_position, diagnostics
    
    def get_emergency_reduce_amount(self, current_exposure: float) -> float:
        """Get amount to reduce in emergency deleverage mode."""
        if not self.is_deleveraging:
            return 0.0
        
        return current_exposure * self.emergency_deleverage_rate
