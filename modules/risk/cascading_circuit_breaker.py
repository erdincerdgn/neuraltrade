"""
Cascading Circuit Breaker with Tiered Risk Intervention
Author: Erdinc Erdogan
Purpose: Implements multi-level circuit breaker (NORMAL→CAUTION→WARNING→DELEVERAGE→HALT) for gradual position reduction before full trading halt.
References:
- Tiered Risk Intervention Systems
- Gradual Deleveraging Protocols
- Trading Halt Mechanisms
Usage:
    breaker = CascadingCircuitBreaker()
    state = breaker.evaluate(drawdown=0.08, volatility=0.25)
    if state.level >= InterventionLevel.WARNING: reduce_positions(state.position_scale)
"""

# ============================================================================
# CASCADING CIRCUIT BREAKER - Tiered Risk Intervention System
# Implements gradual position reduction before full trading halt
# ============================================================================

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from collections import deque


class InterventionLevel(IntEnum):
    """Circuit breaker intervention levels."""
    NORMAL = 0          # Normal trading
    CAUTION = 1         # Increased monitoring
    WARNING = 2         # Reduce new positions
    DELEVERAGE = 3      # Actively reduce exposure
    HALT = 4            # Full trading halt


@dataclass
class InterventionConfig:
    """Configuration for each intervention level."""
    level: InterventionLevel
    position_scale: float       # Multiplier for new positions (1.0 = normal)
    max_exposure: float         # Maximum portfolio exposure allowed
    force_reduce: bool          # Whether to actively reduce positions
    reduce_rate: float          # Rate of position reduction per period
    allow_new_trades: bool      # Whether new trades are allowed
    description: str


@dataclass
class BreakerState:
    """Current state of the circuit breaker."""
    level: InterventionLevel
    triggered_at: Optional[datetime]
    trigger_reason: str
    drawdown: float
    volatility: float
    var_breach: float
    consecutive_losses: int
    time_in_level: int


class CascadingCircuitBreaker:
    """
    Multi-tiered circuit breaker with gradual intervention.
    
    Intervention Levels:
    1. NORMAL: Full trading capacity
    2. CAUTION: Increased monitoring, slight position reduction
    3. WARNING: Significant position reduction, no new longs in crisis
    4. DELEVERAGE: Active position reduction, hedging only
    5. HALT: Complete trading halt, emergency liquidation
    """
    
    # Default intervention configurations
    DEFAULT_CONFIGS = {
        InterventionLevel.NORMAL: InterventionConfig(
            level=InterventionLevel.NORMAL,
            position_scale=1.0,
            max_exposure=1.0,
            force_reduce=False,
            reduce_rate=0.0,
            allow_new_trades=True,
            description="Normal trading operations"
        ),
        InterventionLevel.CAUTION: InterventionConfig(
            level=InterventionLevel.CAUTION,
            position_scale=0.75,
            max_exposure=0.9,
            force_reduce=False,
            reduce_rate=0.0,
            allow_new_trades=True,
            description="Elevated risk - reduced position sizing"
        ),
        InterventionLevel.WARNING: InterventionConfig(
            level=InterventionLevel.WARNING,
            position_scale=0.5,
            max_exposure=0.7,
            force_reduce=False,
            reduce_rate=0.0,
            allow_new_trades=True,
            description="High risk - significant position reduction"
        ),
        InterventionLevel.DELEVERAGE: InterventionConfig(
            level=InterventionLevel.DELEVERAGE,
            position_scale=0.0,
            max_exposure=0.5,
            force_reduce=True,
            reduce_rate=0.1,
            allow_new_trades=False,
            description="Critical risk - active deleveraging"
        ),
        InterventionLevel.HALT: InterventionConfig(
            level=InterventionLevel.HALT,
            position_scale=0.0,
            max_exposure=0.0,
            force_reduce=True,
            reduce_rate=0.25,
            allow_new_trades=False,
            description="Emergency - trading halted"
        ),
    }
    
    def __init__(
        self,
        drawdown_caution: float = 0.03,
        drawdown_warning: float = 0.05,
        drawdown_deleverage: float = 0.08,
        drawdown_halt: float = 0.12,
        vol_caution: float = 0.25,
        vol_warning: float = 0.40,
        vol_deleverage: float = 0.60,
        vol_halt: float = 0.80,
        var_breach_warning: float = 1.5,
        var_breach_deleverage: float = 2.0,
        var_breach_halt: float = 3.0,
        loss_streak_warning: int = 5,
        loss_streak_deleverage: int = 8,
        loss_streak_halt: int = 12,
        recovery_periods: int = 10,
        min_confidence_for_recovery: float = 0.6
    ):
        self.drawdown_thresholds = {
            InterventionLevel.CAUTION: drawdown_caution,
            InterventionLevel.WARNING: drawdown_warning,
            InterventionLevel.DELEVERAGE: drawdown_deleverage,
            InterventionLevel.HALT: drawdown_halt,
        }
        self.vol_thresholds = {
            InterventionLevel.CAUTION: vol_caution,
            InterventionLevel.WARNING: vol_warning,
            InterventionLevel.DELEVERAGE: vol_deleverage,
            InterventionLevel.HALT: vol_halt,
        }
        self.var_breach_thresholds = {
            InterventionLevel.WARNING: var_breach_warning,
            InterventionLevel.DELEVERAGE: var_breach_deleverage,
            InterventionLevel.HALT: var_breach_halt,
        }
        self.loss_streak_thresholds = {
            InterventionLevel.WARNING: loss_streak_warning,
            InterventionLevel.DELEVERAGE: loss_streak_deleverage,
            InterventionLevel.HALT: loss_streak_halt,
        }
        
        self.recovery_periods = recovery_periods
        self.min_confidence_for_recovery = min_confidence_for_recovery
        
        self.current_level = InterventionLevel.NORMAL
        self.time_in_level = 0
        self.triggered_at: Optional[datetime] = None
        self.trigger_reason = ""
        self.history: deque = deque(maxlen=1000)
        self.configs = self.DEFAULT_CONFIGS.copy()
    
    def evaluate(
        self,
        drawdown: float,
        volatility: float,
        var_breach: float = 0.0,
        consecutive_losses: int = 0,
        regime_confidence: float = 1.0
    ) -> Tuple[InterventionConfig, BreakerState]:
        levels = []
        reasons = []
        
        dd_level = self._check_threshold(drawdown, self.drawdown_thresholds)
        if dd_level > InterventionLevel.NORMAL:
            levels.append(dd_level)
            reasons.append(f"Drawdown: {drawdown:.1%}")
        
        vol_level = self._check_threshold(volatility, self.vol_thresholds)
        if vol_level > InterventionLevel.NORMAL:
            levels.append(vol_level)
            reasons.append(f"Volatility: {volatility:.1%}")
        
        var_level = self._check_threshold(var_breach, self.var_breach_thresholds)
        if var_level > InterventionLevel.NORMAL:
            levels.append(var_level)
            reasons.append(f"VaR Breach: {var_breach:.1f}x")
        
        streak_level = self._check_threshold(consecutive_losses, self.loss_streak_thresholds)
        if streak_level > InterventionLevel.NORMAL:
            levels.append(streak_level)
            reasons.append(f"Loss Streak: {consecutive_losses}")
        
        if levels:
            required_level = max(levels)
            trigger_reason = " | ".join(reasons)
        else:
            required_level = InterventionLevel.NORMAL
            trigger_reason = "Normal conditions"
        
        if regime_confidence < 0.3 and required_level < InterventionLevel.WARNING:
            required_level = InterventionLevel.WARNING
            trigger_reason += " | Low regime confidence"
        
        new_level = self._handle_transition(required_level, regime_confidence)
        
        if new_level != self.current_level:
            self.current_level = new_level
            self.time_in_level = 0
            self.triggered_at = datetime.now()
            self.trigger_reason = trigger_reason
        else:
            self.time_in_level += 1
        
        state = BreakerState(
            level=self.current_level,
            triggered_at=self.triggered_at,
            trigger_reason=self.trigger_reason,
            drawdown=drawdown,
            volatility=volatility,
            var_breach=var_breach,
            consecutive_losses=consecutive_losses,
            time_in_level=self.time_in_level
        )
        
        self.history.append(state)
        return self.configs[self.current_level], state
    
    def _check_threshold(self, value: float, thresholds: Dict) -> InterventionLevel:
        result = InterventionLevel.NORMAL
        for level, threshold in sorted(thresholds.items(), key=lambda x: int(x[0])):
            if value >= threshold:
                result = level
        return result
    
    def _handle_transition(self, required_level: InterventionLevel, regime_confidence: float) -> InterventionLevel:
        current_value = int(self.current_level)
        required_value = int(required_level)
        
        if required_value > current_value:
            return required_level
        
        if required_value < current_value:
            if self.time_in_level < self.recovery_periods:
                return self.current_level
            if regime_confidence < self.min_confidence_for_recovery:
                return self.current_level
            return InterventionLevel(current_value - 1)
        
        return self.current_level
    
    def get_position_scale(self) -> float:
        return self.configs[self.current_level].position_scale
    
    def reset(self):
        self.current_level = InterventionLevel.NORMAL
        self.time_in_level = 0
        self.triggered_at = None
        self.trigger_reason = ""
        self.history.clear()
