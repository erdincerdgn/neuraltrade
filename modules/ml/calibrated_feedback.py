"""
Calibrated Execution Feedback with Logarithmic Penalties
Author: Erdinc Erdogan
Purpose: Implements regime-aware logarithmic penalty scale for agent signal quality based on execution slippage with EMA smoothing and recovery mechanisms.
References:
- Logarithmic Penalty Functions
- Execution Quality Feedback Loops
- Regime-Adaptive Threshold Models
Usage:
    feedback = CalibratedFeedbackLoop(min_penalty_factor=0.5, ema_alpha=0.1)
    penalty = feedback.calculate_penalty(execution_result, regime="HIGH_VOL")
"""

# ============================================================================
# CALIBRATED EXECUTION FEEDBACK
# Logarithmic penalty scale to prevent over-optimization
# Phase 8C: Interconnect Stabilization - Task 3
# ============================================================================

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque
from enum import IntEnum
import time


class ExecutionQuality(IntEnum):
    EXCELLENT = 4
    GOOD = 3
    ACCEPTABLE = 2
    POOR = 1
    FAILED = 0


@dataclass
class ExecutionResult:
    order_id: str
    agent_type: str
    signal_id: str
    expected_price: float
    executed_price: float
    expected_quantity: float
    executed_quantity: float
    slippage_bps: float
    market_impact_bps: float
    execution_time_ms: float
    timestamp: float
    quality: ExecutionQuality
    regime: str = "NORMAL"


@dataclass
class CalibratedPenalty:
    """Calibrated penalty with regime awareness."""
    agent_type: str
    penalty_factor: float
    raw_penalty: float
    regime_adjustment: float
    recovery_bonus: float
    slippage_score: float
    quality_score: float
    consecutive_good: int


class CalibratedFeedbackLoop:
    """
    Calibrated Execution Feedback with Logarithmic Penalties.
    
    Key Improvements:
    1. Logarithmic penalty scale (prevents over-penalization)
    2. Raised min_penalty_factor to 0.5 (fixes FO-001)
    3. EMA smoothing for penalty updates (fixes FO-002)
    4. Regime-aware slippage thresholds
    5. Recovery mechanism for good executions
    
    Mathematical Foundation:
    penalty = 1 - log(1 + slippage/scale) / log(1 + max_slippage/scale)
    
    This creates a diminishing penalty curve:
    - Small slippage: minimal penalty
    - Large slippage: significant but bounded penalty
    """
    
    # Regime-specific slippage thresholds
    REGIME_SLIPPAGE_TARGETS = {
        "LOW_VOL": 5.0,
        "NORMAL": 10.0,
        "HIGH_VOL": 20.0,
        "CRISIS": 40.0,
    }
    
    def __init__(
        self,
        lookback_window: int = 200,
        min_penalty_factor: float = 0.5,  # Raised from 0.3 (fixes FO-001)
        max_penalty_factor: float = 1.0,
        log_scale: float = 10.0,
        ema_alpha: float = 0.1,  # EMA smoothing (fixes FO-002)
        recovery_rate: float = 0.05,
        recovery_threshold: int = 5,
        regime_adaptive: bool = True
    ):
        self.lookback_window = lookback_window
        self.min_penalty_factor = max(min_penalty_factor, 0.5)  # Hard floor
        self.max_penalty_factor = max_penalty_factor
        self.log_scale = log_scale
        self.ema_alpha = ema_alpha
        self.recovery_rate = recovery_rate
        self.recovery_threshold = recovery_threshold
        self.regime_adaptive = regime_adaptive
        
        # Per-agent tracking
        self.slippage_history: Dict[str, deque] = {}
        self.quality_history: Dict[str, deque] = {}
        self.cached_penalties: Dict[str, CalibratedPenalty] = {}
        self.consecutive_good: Dict[str, int] = {}
        
        # Current regime
        self.current_regime: str = "NORMAL"
        
        # Statistics
        self.total_executions: int = 0
        
    def record_execution(self, result: ExecutionResult):
        """Record execution and update penalties with EMA smoothing."""
        agent = result.agent_type
        
        # Initialize if needed
        if agent not in self.slippage_history:
            self._init_agent(agent)
        
        # Record metrics
        self.slippage_history[agent].append(result.slippage_bps)
        self.quality_history[agent].append(result.quality)
        
        # Update regime
        if result.regime:
            self.current_regime = result.regime
        
        # Track consecutive good executions
        if result.quality >= ExecutionQuality.GOOD:
            self.consecutive_good[agent] += 1
        else:
            self.consecutive_good[agent] = 0
        
        # Update penalty with EMA smoothing
        self._update_penalty_ema(agent, result)
        
        self.total_executions += 1
    
    def get_penalty_factor(self, agent_type: str) -> float:
        """Get current penalty factor for an agent."""
        if agent_type in self.cached_penalties:
            return self.cached_penalties[agent_type].penalty_factor
        return 1.0
    
    def get_calibrated_penalty(self, agent_type: str) -> CalibratedPenalty:
        """Get full calibrated penalty details."""
        if agent_type in self.cached_penalties:
            return self.cached_penalties[agent_type]
        return CalibratedPenalty(
            agent_type=agent_type,
            penalty_factor=1.0,
            raw_penalty=1.0,
            regime_adjustment=0.0,
            recovery_bonus=0.0,
            slippage_score=1.0,
            quality_score=1.0,
            consecutive_good=0
        )
    
    def set_regime(self, regime: str):
        """Update current volatility regime."""
        if regime in self.REGIME_SLIPPAGE_TARGETS:
            self.current_regime = regime
    
    def _init_agent(self, agent: str):
        """Initialize tracking for a new agent."""
        self.slippage_history[agent] = deque(maxlen=self.lookback_window)
        self.quality_history[agent] = deque(maxlen=self.lookback_window)
        self.consecutive_good[agent] = 0
        self.cached_penalties[agent] = CalibratedPenalty(
            agent_type=agent,
            penalty_factor=1.0,
            raw_penalty=1.0,
            regime_adjustment=0.0,
            recovery_bonus=0.0,
            slippage_score=1.0,
            quality_score=1.0,
            consecutive_good=0
        )
    
    def _update_penalty_ema(self, agent: str, result: ExecutionResult):
        """Update penalty using EMA smoothing."""
        # Get regime-specific target
        target_slippage = self.REGIME_SLIPPAGE_TARGETS.get(
            self.current_regime, 10.0
        )
        
        # Calculate raw logarithmic penalty
        raw_penalty = self._calculate_log_penalty(
            result.slippage_bps, target_slippage
        )
        
        # Calculate regime adjustment
        regime_adjustment = 0.0
        if self.regime_adaptive and self.current_regime in ["HIGH_VOL", "CRISIS"]:
            # Be more lenient in high-vol regimes
            regime_adjustment = 0.1 if self.current_regime == "HIGH_VOL" else 0.2
        
        # Calculate recovery bonus
        recovery_bonus = 0.0
        if self.consecutive_good[agent] >= self.recovery_threshold:
            recovery_bonus = min(
                self.recovery_rate * (self.consecutive_good[agent] - self.recovery_threshold + 1),
                0.2  # Cap at 20% bonus
            )
        
        # Calculate new penalty factor
        new_penalty = raw_penalty + regime_adjustment + recovery_bonus
        new_penalty = np.clip(new_penalty, self.min_penalty_factor, self.max_penalty_factor)
        
        # Apply EMA smoothing (fixes FO-002)
        old_penalty = self.cached_penalties[agent].penalty_factor
        smoothed_penalty = self.ema_alpha * new_penalty + (1 - self.ema_alpha) * old_penalty
        
        # Calculate scores
        slippage_score = self._calculate_slippage_score(agent, target_slippage)
        quality_score = self._calculate_quality_score(agent)
        
        # Update cached penalty
        self.cached_penalties[agent] = CalibratedPenalty(
            agent_type=agent,
            penalty_factor=smoothed_penalty,
            raw_penalty=raw_penalty,
            regime_adjustment=regime_adjustment,
            recovery_bonus=recovery_bonus,
            slippage_score=slippage_score,
            quality_score=quality_score,
            consecutive_good=self.consecutive_good[agent]
        )
    
    def _calculate_log_penalty(self, slippage: float, target: float) -> float:
        """
        Calculate logarithmic penalty.
        
        penalty = 1 - log(1 + slippage/scale) / log(1 + max_slippage/scale)
        
        This creates a diminishing penalty curve.
        """
        if slippage <= 0:
            return 1.0  # No penalty for price improvement
        
        max_slippage = target * 5  # 5x target is maximum
        
        # Logarithmic scaling
        log_slippage = np.log(1 + slippage / self.log_scale)
        log_max = np.log(1 + max_slippage / self.log_scale)
        
        penalty = 1.0 - (log_slippage / log_max)
        
        return np.clip(penalty, self.min_penalty_factor, 1.0)
    
    def _calculate_slippage_score(self, agent: str, target: float) -> float:
        """Calculate rolling slippage score."""
        if len(self.slippage_history[agent]) < 5:
            return 1.0
        
        slippages = list(self.slippage_history[agent])
        avg_slippage = np.mean(slippages)
        
        if avg_slippage <= target:
            return 1.0
        else:
            excess = avg_slippage - target
            return max(0.3, 1.0 - excess / (target * 3))
    
    def _calculate_quality_score(self, agent: str) -> float:
        """Calculate rolling quality score."""
        if len(self.quality_history[agent]) < 5:
            return 1.0
        
        qualities = list(self.quality_history[agent])
        avg_quality = np.mean(qualities)
        
        # Normalize to [0, 1]
        return avg_quality / 4.0
    
    def get_statistics(self) -> Dict:
        """Get feedback loop statistics."""
        return {
            "total_executions": self.total_executions,
            "current_regime": self.current_regime,
            "agents_tracked": len(self.slippage_history),
            "penalties": {
                agent: {
                    "factor": p.penalty_factor,
                    "consecutive_good": p.consecutive_good,
                    "recovery_bonus": p.recovery_bonus
                }
                for agent, p in self.cached_penalties.items()
            }
        }
    
    def reset(self):
        """Reset all tracking."""
        self.slippage_history.clear()
        self.quality_history.clear()
        self.cached_penalties.clear()
        self.consecutive_good.clear()
        self.total_executions = 0


class AdaptiveExecutionOptimizer:
    """
    Adaptive Execution Optimizer with calibrated feedback.
    
    Uses calibrated penalties to adjust execution parameters
    without making agents overly timid.
    """
    
    def __init__(self, feedback_loop: CalibratedFeedbackLoop):
        self.feedback_loop = feedback_loop
        
    def optimize_execution(
        self,
        agent_type: str,
        signal_confidence: float,
        target_quantity: float,
        volatility: float,
        spread_bps: float
    ) -> Dict:
        """Optimize execution with calibrated penalties."""
        penalty = self.feedback_loop.get_calibrated_penalty(agent_type)
        
        # Adjust confidence (but not too aggressively)
        adjusted_confidence = signal_confidence * penalty.penalty_factor
        
        # Ensure minimum confidence floor
        adjusted_confidence = max(adjusted_confidence, 0.3)
        
        # Adjust quantity based on penalty
        adjusted_quantity = target_quantity * penalty.penalty_factor
        
        # Determine execution style
        if penalty.quality_score > 0.7:
            style = "NORMAL"
            participation = 0.10
        elif penalty.quality_score > 0.4:
            style = "CAUTIOUS"
            participation = 0.07
        else:
            style = "PASSIVE"
            participation = 0.05
        
        return {
            "agent_type": agent_type,
            "original_confidence": signal_confidence,
            "adjusted_confidence": adjusted_confidence,
            "penalty_factor": penalty.penalty_factor,
            "target_quantity": target_quantity,
            "adjusted_quantity": adjusted_quantity,
            "execution_style": style,
            "participation_rate": participation,
            "consecutive_good": penalty.consecutive_good,
            "recovery_bonus": penalty.recovery_bonus,
        }
