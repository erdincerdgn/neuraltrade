"""
Adaptive Exposure Manager - Volatility-Aware Position Limits
Author: Erdinc Erdogan
Purpose: Dynamically adjusts exposure limits based on volatility and chaos metrics,
ensuring position sizes scale appropriately with market conditions.
References:
- Volatility Scaling in Risk Management
- Adaptive Position Sizing Algorithms
- Chaos-Based Risk Adjustment
Usage:
    manager = AdaptiveExposureManager(config)
    limits = manager.get_adjusted_limits(volatility=0.03, chaos_factor=0.7)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

@dataclass
class AdaptiveExposureConfig:
    base_single_asset: float = 0.10      # 10% base
    base_portfolio: float = 0.50          # 50% base
    base_correlated: float = 0.25         # 25% base
    vol_scaling_factor: float = 2.0       # How much vol affects limits
    chaos_scaling_factor: float = 1.5     # How much chaos affects limits
    min_limit_multiplier: float = 0.25    # Never go below 25% of base
    max_limit_multiplier: float = 1.5     # Never exceed 150% of base

class AdaptiveExposureManager:
    """Volatility-aware exposure limits with O(1) computation."""
    
    CORRELATION_MATRIX = {
        ("BTCUSD", "ETHUSD"): 0.85, ("BTCUSD", "SOLUSD"): 0.78,
        ("ETHUSD", "SOLUSD"): 0.82, ("BTCUSD", "XAUUSD"): 0.15,
        ("EURUSD", "GBPUSD"): 0.72,
    }
    
    def __init__(self, config: Optional[AdaptiveExposureConfig] = None):
        self.config = config or AdaptiveExposureConfig()
        self._positions: Dict[str, float] = {}
        self._portfolio_value: float = 100_000.0
        self._current_volatility: float = 0.02  # 2% default
        self._current_chaos: float = 0.0        # 0-1 scale
        self._cached_limits: Dict[str, float] = {}
        self._limits_dirty: bool = True
    
    def update_market_conditions(self, volatility: float, chaos_index: float):
        """Update volatility and chaos metrics (triggers limit recalc)."""
        self._current_volatility = max(0.001, volatility)
        self._current_chaos = max(0.0, min(1.0, chaos_index))
        self._limits_dirty = True
    
    def _compute_adaptive_multiplier(self) -> float:
        """Compute limit multiplier based on market conditions."""
        vol_factor = 0.02 / self._current_volatility
        vol_adjustment = vol_factor ** (1 / self.config.vol_scaling_factor)
        
        chaos_adjustment = 1.0 - (self._current_chaos * 0.5)
        
        multiplier = vol_adjustment * chaos_adjustment
        
        return max(self.config.min_limit_multiplier, 
                   min(self.config.max_limit_multiplier, multiplier))
    
    def get_current_limits(self) -> Dict[str, float]:
        """Get current adaptive limits (cached for O(1) access)."""
        if self._limits_dirty:
            mult = self._compute_adaptive_multiplier()
            self._cached_limits = {
                "single_asset": self.config.base_single_asset * mult,
                "portfolio": self.config.base_portfolio * mult,
                "correlated": self.config.base_correlated * mult,
                "multiplier": mult,
                "volatility": self._current_volatility,
                "chaos": self._current_chaos
            }
            self._limits_dirty = False
        return self._cached_limits
    
    def check_exposure(self, symbol: str, new_position_value: float) -> Tuple[bool, Optional[str], str]:
        """O(1) exposure check with adaptive limits."""
        limits = self.get_current_limits()
        
        single_pct = abs(new_position_value) / self._portfolio_value
        if single_pct > limits["single_asset"]:
            return False, "AEX-001", f"Single {single_pct:.1%} > adaptive {limits['single_asset']:.1%}"
        
        temp_pos = self._positions.copy()
        temp_pos[symbol] = new_position_value
        total_pct = sum(abs(p) for p in temp_pos.values()) / self._portfolio_value
        if total_pct > limits["portfolio"]:
            return False, "AEX-002", f"Portfolio {total_pct:.1%} > adaptive {limits['portfolio']:.1%}"
        
        return True, None, "OK"
    
    def update_position(self, symbol: str, value: float):
        self._positions[symbol] = value
