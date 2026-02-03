"""
Adaptive Position Sizer with Entropy and Sharpe Integration
Author: Erdinc Erdogan
Purpose: Dynamically adjusts position sizes based on regime entropy, agent Sharpe ratios, volatility regime, and drawdown levels for optimal capital deployment.
References:
- Kelly Criterion for Position Sizing
- Shannon Entropy for Uncertainty Gating
- Volatility-Scaled Position Sizing
Usage:
    sizer = AdaptivePositionSizer(base_size=10000)
    result = sizer.calculate_size(signal_confidence=0.8, entropy=0.3, sharpe=1.5)
"""

# ============================================================================
# ADAPTIVE PORTFOLIO MANAGER
# Links to SharpeWeightedConsensus and Entropy for dynamic position sizing
# Phase 8B: The Neural Interconnect - Task 2
# ============================================================================

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time


class LeverageMode(IntEnum):
    """Leverage mode classification."""
    CONSERVATIVE = 0    # Low leverage, high margin
    NORMAL = 1          # Standard leverage
    AGGRESSIVE = 2      # High leverage, low margin
    DEFENSIVE = 3       # Deleveraging mode


class PositionSizeMethod(IntEnum):
    """Position sizing method."""
    FIXED = 0           # Fixed position size
    VOLATILITY_SCALED = 1  # Scale by inverse volatility
    KELLY = 2           # Kelly criterion
    ENTROPY_GATED = 3   # Scale by regime certainty
    SHARPE_WEIGHTED = 4 # Scale by agent Sharpe


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    base_size: float
    adjusted_size: float
    leverage: float
    margin_required: float
    confidence_factor: float
    entropy_factor: float
    sharpe_factor: float
    method: PositionSizeMethod
    reasoning: str


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_equity: float
    used_margin: float
    available_margin: float
    current_leverage: float
    max_leverage: float
    positions: Dict[str, float]
    unrealized_pnl: float
    realized_pnl: float


class AdaptivePositionSizer:
    """
    Adaptive Position Sizer with Entropy and Sharpe Integration.
    
    Dynamically adjusts position sizes based on:
    1. Regime entropy (uncertainty)
    2. Agent Sharpe ratios (performance)
    3. Volatility regime
    4. Current drawdown
    
    Mathematical Foundation:
    size = base_size × entropy_factor × sharpe_factor × vol_factor × dd_factor
    
    Where:
    - entropy_factor = 1 - (entropy - threshold) / (1 - threshold) if entropy > threshold
    - sharpe_factor = sigmoid(sharpe / temperature)
    - vol_factor = target_vol / current_vol
    - dd_factor = 1 - (drawdown / max_drawdown)^2
    
    Usage:
        sizer = AdaptivePositionSizer(base_size=10000)
        result = sizer.calculate_size(signal_confidence, entropy, sharpe, volatility)
    """
    
    def __init__(
        self,
        base_size: float = 10000.0,
        max_size: float = 100000.0,
        min_size: float = 1000.0,
        target_volatility: float = 0.15,
        entropy_threshold: float = 0.6,
        sharpe_temperature: float = 1.5,
        max_drawdown_threshold: float = 0.15,
        kelly_fraction: float = 0.25,
        enable_kelly: bool = True
    ):
        self.base_size = base_size
        self.max_size = max_size
        self.min_size = min_size
        self.target_volatility = target_volatility
        self.entropy_threshold = entropy_threshold
        self.sharpe_temperature = sharpe_temperature
        self.max_drawdown_threshold = max_drawdown_threshold
        self.kelly_fraction = kelly_fraction
        self.enable_kelly = enable_kelly
        
        # State
        self.current_drawdown: float = 0.0
        self.current_volatility: float = 0.15
        self.last_result: Optional[PositionSizeResult] = None
        
    def calculate_size(
        self,
        signal_confidence: float,
        entropy: float,
        agent_sharpe: float,
        volatility: float,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
        drawdown: float = 0.0
    ) -> PositionSizeResult:
        """
        Calculate adaptive position size.
        
        Args:
            signal_confidence: Signal confidence [0, 1]
            entropy: Regime entropy [0, 1]
            agent_sharpe: Agent's rolling Sharpe ratio
            volatility: Current annualized volatility
            win_rate: Historical win rate
            avg_win_loss_ratio: Average win/loss ratio
            drawdown: Current drawdown
            
        Returns:
            PositionSizeResult with adjusted size
        """
        self.current_drawdown = drawdown
        self.current_volatility = volatility
        
        # Start with base size scaled by confidence
        base = self.base_size * signal_confidence
        
        # Calculate adjustment factors
        entropy_factor = self._calculate_entropy_factor(entropy)
        sharpe_factor = self._calculate_sharpe_factor(agent_sharpe)
        vol_factor = self._calculate_volatility_factor(volatility)
        dd_factor = self._calculate_drawdown_factor(drawdown)
        
        # Kelly criterion (optional)
        kelly_size = base
        if self.enable_kelly and win_rate > 0 and avg_win_loss_ratio > 0:
            kelly_size = self._calculate_kelly_size(
                base, win_rate, avg_win_loss_ratio
            )
        
        # Combine factors
        adjusted_size = base * entropy_factor * sharpe_factor * vol_factor * dd_factor
        
        # Blend with Kelly if enabled
        if self.enable_kelly:
            adjusted_size = 0.7 * adjusted_size + 0.3 * kelly_size
        
        # Clip to bounds
        adjusted_size = np.clip(adjusted_size, self.min_size, self.max_size)
        
        # Calculate leverage and margin
        leverage = adjusted_size / self.base_size
        margin_required = adjusted_size * 0.1  # 10% margin requirement
        
        # Determine method used
        method = PositionSizeMethod.ENTROPY_GATED if entropy > self.entropy_threshold else PositionSizeMethod.SHARPE_WEIGHTED
        
        # Build reasoning
        reasoning = self._build_reasoning(
            entropy_factor, sharpe_factor, vol_factor, dd_factor
        )
        
        result = PositionSizeResult(
            base_size=base,
            adjusted_size=adjusted_size,
            leverage=leverage,
            margin_required=margin_required,
            confidence_factor=signal_confidence,
            entropy_factor=entropy_factor,
            sharpe_factor=sharpe_factor,
            method=method,
            reasoning=reasoning
        )
        
        self.last_result = result
        return result
    
    def _calculate_entropy_factor(self, entropy: float) -> float:
        """Calculate entropy-based scaling factor."""
        if entropy <= self.entropy_threshold:
            return 1.0
        
        # Linear reduction above threshold
        excess = entropy - self.entropy_threshold
        max_excess = 1.0 - self.entropy_threshold
        reduction = excess / max_excess
        
        return max(0.2, 1.0 - reduction * 0.8)
    
    def _calculate_sharpe_factor(self, sharpe: float) -> float:
        """Calculate Sharpe-based scaling factor."""
        # Sigmoid transformation
        factor = 1.0 / (1.0 + np.exp(-sharpe / self.sharpe_temperature))
        
        # Scale to [0.5, 1.5] range
        return 0.5 + factor
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """Calculate volatility-based scaling factor."""
        if volatility <= 0:
            return 1.0
        
        # Inverse volatility scaling
        factor = self.target_volatility / volatility
        
        # Clip to reasonable range
        return np.clip(factor, 0.3, 2.0)
    
    def _calculate_drawdown_factor(self, drawdown: float) -> float:
        """Calculate drawdown-based scaling factor."""
        if drawdown <= 0:
            return 1.0
        
        # Quadratic reduction
        ratio = drawdown / self.max_drawdown_threshold
        factor = 1.0 - ratio ** 2
        
        return max(0.1, factor)
    
    def _calculate_kelly_size(
        self,
        base: float,
        win_rate: float,
        avg_win_loss_ratio: float
    ) -> float:
        """Calculate Kelly criterion position size."""
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = 1 - p, b = win/loss ratio
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio
        
        kelly_fraction = (p * b - q) / b
        
        # Apply fractional Kelly
        kelly_fraction *= self.kelly_fraction
        
        # Clip to reasonable range
        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.5)
        
        return base * (1 + kelly_fraction)
    
    def _build_reasoning(
        self,
        entropy_factor: float,
        sharpe_factor: float,
        vol_factor: float,
        dd_factor: float
    ) -> str:
        """Build human-readable reasoning."""
        parts = []
        
        if entropy_factor < 0.8:
            parts.append(f"Entropy reduction: {entropy_factor:.0%}")
        if sharpe_factor > 1.2:
            parts.append(f"Sharpe boost: {sharpe_factor:.0%}")
        elif sharpe_factor < 0.8:
            parts.append(f"Sharpe penalty: {sharpe_factor:.0%}")
        if vol_factor < 0.8:
            parts.append(f"High vol reduction: {vol_factor:.0%}")
        if dd_factor < 0.8:
            parts.append(f"Drawdown reduction: {dd_factor:.0%}")
        
        return " | ".join(parts) if parts else "Standard sizing"


class DynamicLeverageManager:
    """
    Dynamic Leverage Manager with Regime Awareness.
    
    Adjusts leverage based on:
    1. Regime entropy
    2. Volatility regime
    3. Drawdown level
    4. Agent consensus confidence
    """
    
    # Leverage limits by mode
    LEVERAGE_LIMITS = {
        LeverageMode.CONSERVATIVE: (1.0, 2.0),
        LeverageMode.NORMAL: (2.0, 4.0),
        LeverageMode.AGGRESSIVE: (4.0, 6.0),
        LeverageMode.DEFENSIVE: (0.5, 1.0),
    }
    
    def __init__(
        self,
        base_leverage: float = 2.0,
        max_leverage: float = 5.0,
        min_leverage: float = 0.5,
        entropy_threshold: float = 0.7,
        drawdown_threshold: float = 0.10,
        volatility_threshold: float = 0.30
    ):
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.entropy_threshold = entropy_threshold
        self.drawdown_threshold = drawdown_threshold
        self.volatility_threshold = volatility_threshold
        
        # State
        self.current_mode: LeverageMode = LeverageMode.NORMAL
        self.current_leverage: float = base_leverage
        
    def calculate_leverage(
        self,
        consensus_confidence: float,
        entropy: float,
        volatility: float,
        drawdown: float,
        regime: str = "NEUTRAL"
    ) -> Tuple[float, LeverageMode]:
        """
        Calculate dynamic leverage.
        
        Returns:
            Tuple of (leverage, mode)
        """
        # Determine mode based on conditions
        mode = self._determine_mode(entropy, volatility, drawdown)
        
        # Get leverage limits for mode
        min_lev, max_lev = self.LEVERAGE_LIMITS[mode]
        
        # Calculate base leverage from confidence
        base = self.base_leverage * consensus_confidence
        
        # Apply entropy adjustment
        if entropy > self.entropy_threshold:
            entropy_factor = 1.0 - (entropy - self.entropy_threshold) / (1.0 - self.entropy_threshold)
            base *= entropy_factor
        
        # Apply volatility adjustment
        if volatility > self.volatility_threshold:
            vol_factor = self.volatility_threshold / volatility
            base *= vol_factor
        
        # Apply drawdown adjustment
        if drawdown > 0:
            dd_factor = 1.0 - (drawdown / self.drawdown_threshold) ** 2
            base *= max(0.3, dd_factor)
        
        # Clip to mode limits
        leverage = np.clip(base, min_lev, max_lev)
        
        # Also clip to global limits
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
        
        self.current_mode = mode
        self.current_leverage = leverage
        
        return leverage, mode
    
    def _determine_mode(
        self,
        entropy: float,
        volatility: float,
        drawdown: float
    ) -> LeverageMode:
        """Determine leverage mode based on conditions."""
        # Defensive mode triggers
        if drawdown > self.drawdown_threshold:
            return LeverageMode.DEFENSIVE
        if entropy > 0.85:
            return LeverageMode.DEFENSIVE
        if volatility > self.volatility_threshold * 1.5:
            return LeverageMode.DEFENSIVE
        
        # Conservative mode triggers
        if entropy > self.entropy_threshold:
            return LeverageMode.CONSERVATIVE
        if volatility > self.volatility_threshold:
            return LeverageMode.CONSERVATIVE
        if drawdown > self.drawdown_threshold * 0.5:
            return LeverageMode.CONSERVATIVE
        
        # Aggressive mode triggers
        if entropy < 0.3 and volatility < self.volatility_threshold * 0.5:
            return LeverageMode.AGGRESSIVE
        
        return LeverageMode.NORMAL
    
    def get_margin_requirement(self, position_value: float) -> float:
        """Calculate margin requirement based on current leverage."""
        if self.current_leverage <= 0:
            return position_value
        return position_value / self.current_leverage


class AdaptivePortfolioManager:
    """
    Adaptive Portfolio Manager with Full Neural Interconnect.
    
    Integrates:
    - AdaptivePositionSizer
    - DynamicLeverageManager
    - Entropy gating
    - Sharpe-weighted sizing
    - Execution feedback
    """
    
    def __init__(
        self,
        initial_equity: float = 1000000.0,
        base_position_size: float = 10000.0,
        max_position_size: float = 100000.0,
        max_portfolio_leverage: float = 5.0
    ):
        self.position_sizer = AdaptivePositionSizer(
            base_size=base_position_size,
            max_size=max_position_size
        )
        self.leverage_manager = DynamicLeverageManager(
            max_leverage=max_portfolio_leverage
        )
        
        # Portfolio state
        self.equity = initial_equity
        self.positions: Dict[str, float] = {}
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.peak_equity: float = initial_equity
        
    def calculate_position(
        self,
        symbol: str,
        signal_direction: float,
        signal_confidence: float,
        agent_sharpe: float,
        entropy: float,
        volatility: float,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5
    ) -> Dict:
        """
        Calculate position size with full adaptive logic.
        
        Returns dict with position details.
        """
        # Calculate drawdown
        drawdown = max(0, (self.peak_equity - self.equity) / self.peak_equity)
        
        # Get position size
        size_result = self.position_sizer.calculate_size(
            signal_confidence=signal_confidence,
            entropy=entropy,
            agent_sharpe=agent_sharpe,
            volatility=volatility,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio,
            drawdown=drawdown
        )
        
        # Get leverage
        leverage, mode = self.leverage_manager.calculate_leverage(
            consensus_confidence=signal_confidence,
            entropy=entropy,
            volatility=volatility,
            drawdown=drawdown
        )
        
        # Apply direction
        position_size = size_result.adjusted_size * np.sign(signal_direction)
        
        return {
            "symbol": symbol,
            "direction": "LONG" if signal_direction > 0 else "SHORT",
            "size": abs(position_size),
            "signed_size": position_size,
            "leverage": leverage,
            "leverage_mode": mode.name,
            "margin_required": size_result.margin_required,
            "entropy_factor": size_result.entropy_factor,
            "sharpe_factor": size_result.sharpe_factor,
            "confidence": signal_confidence,
            "reasoning": size_result.reasoning,
            "drawdown": drawdown
        }
    
    def update_equity(self, pnl: float):
        """Update equity and track peak."""
        self.equity += pnl
        self.realized_pnl += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
    
    def get_state(self) -> PortfolioState:
        """Get current portfolio state."""
        used_margin = sum(abs(p) * 0.1 for p in self.positions.values())
        
        return PortfolioState(
            total_equity=self.equity,
            used_margin=used_margin,
            available_margin=self.equity - used_margin,
            current_leverage=self.leverage_manager.current_leverage,
            max_leverage=self.leverage_manager.max_leverage,
            positions=self.positions.copy(),
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl
        )
