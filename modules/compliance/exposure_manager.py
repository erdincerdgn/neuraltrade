"""
Exposure Manager - Cross-Asset Correlation-Aware Limits
Author: Erdinc Erdogan
Purpose: Manages portfolio exposure with cross-asset correlation awareness, grouping
correlated assets together for combined limit enforcement.
References:
- Portfolio Correlation Matrix Analysis
- VaR-Based Exposure Management
- Correlation-Adjusted Position Limits
Usage:
    manager = ExposureManager(limit=ExposureLimit())
    result = manager.check_exposure(symbol="BTCUSD", size=1000, current_positions=portfolio)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ExposureLimit:
    max_single_asset: float = 0.1      # 10% max per asset
    max_portfolio: float = 0.5          # 50% max total
    max_correlated_group: float = 0.25  # 25% max for correlated assets
    correlation_threshold: float = 0.7  # Assets with corr > 0.7 are grouped

class ExposureManager:
    """Manages exposure with cross-asset correlation awareness."""
    
    CORRELATION_MATRIX = {
        ("BTCUSD", "ETHUSD"): 0.85,
        ("BTCUSD", "SOLUSD"): 0.78,
        ("ETHUSD", "SOLUSD"): 0.82,
        ("BTCUSD", "XAUUSD"): 0.15,
        ("EURUSD", "GBPUSD"): 0.72,
    }
    
    def __init__(self, limits: Optional[ExposureLimit] = None):
        self.limits = limits or ExposureLimit()
        self._positions: Dict[str, float] = {}
        self._portfolio_value: float = 100000.0
    
    def get_correlation(self, asset1: str, asset2: str) -> float:
        key = tuple(sorted([asset1, asset2]))
        return self.CORRELATION_MATRIX.get(key, 0.0)
    
    def get_correlated_exposure(self, symbol: str) -> float:
        """Calculate total exposure including correlated assets."""
        total = abs(self._positions.get(symbol, 0.0))
        for other, pos in self._positions.items():
            if other != symbol:
                corr = self.get_correlation(symbol, other)
                if corr >= self.limits.correlation_threshold:
                    total += abs(pos) * corr
        return total / self._portfolio_value
    
    def check_exposure(self, symbol: str, new_position: float) -> tuple:
        """Check if new position violates exposure limits."""
        single = abs(new_position) / self._portfolio_value
        if single > self.limits.max_single_asset:
            return False, "EXP-001", f"Single asset {single:.1%} > {self.limits.max_single_asset:.1%}"
        
        temp_pos = self._positions.copy()
        temp_pos[symbol] = new_position
        total = sum(abs(p) for p in temp_pos.values()) / self._portfolio_value
        if total > self.limits.max_portfolio:
            return False, "EXP-002", f"Portfolio {total:.1%} > {self.limits.max_portfolio:.1%}"
        
        corr_exp = self.get_correlated_exposure(symbol)
        if corr_exp > self.limits.max_correlated_group:
            return False, "EXP-003", f"Correlated {corr_exp:.1%} > {self.limits.max_correlated_group:.1%}"
        
        return True, None, "OK"
    
    def update_position(self, symbol: str, position: float):
        self._positions[symbol] = position
