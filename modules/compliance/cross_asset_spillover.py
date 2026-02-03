"""
Cross-Asset Spillover Manager - Correlated Limit Adjustment
Author: Erdinc Erdogan
Purpose: Manages real-time correlated asset limit adjustments when exposure breaches occur,
preventing concentrated risk in highly correlated asset groups.
References:
- Cross-Asset Correlation in Portfolio Risk Management
- Spillover Effects in Financial Markets
- Dynamic Limit Management Systems
Usage:
    spillover = CrossAssetSpillover(config=SpilloverConfig())
    adjusted_limits = spillover.apply_spillover(symbol="BTCUSD", breach_amount=10000)
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set
import math

@dataclass
class SpilloverConfig:
    correlation_threshold: float = 0.70  # Trigger spillover above 70%
    spillover_factor: float = 0.50       # Reduce correlated by 50% of breach
    cascade_depth: int = 2               # Max cascade levels

class CrossAssetSpillover:
    """Real-time correlated asset limit adjustment."""
    
    CORRELATIONS = {
        ("BTCUSD", "ETHUSD"): 0.85, ("BTCUSD", "SOLUSD"): 0.78,
        ("ETHUSD", "SOLUSD"): 0.82, ("BTCUSD", "BNBUSD"): 0.75,
        ("ETHUSD", "BNBUSD"): 0.80, ("XAUUSD", "XAGUSD"): 0.90,
    }
    
    def __init__(self, config: Optional[SpilloverConfig] = None):
        self.config = config or SpilloverConfig()
        self._base_limits: Dict[str, float] = {}
        self._active_limits: Dict[str, float] = {}
        self._breach_history: Dict[str, float] = {}
    
    def set_base_limit(self, symbol: str, limit: float):
        self._base_limits[symbol] = limit
        self._active_limits[symbol] = limit
    
    def get_correlation(self, sym1: str, sym2: str) -> float:
        key = (sym1, sym2) if (sym1, sym2) in self.CORRELATIONS else (sym2, sym1)
        return self.CORRELATIONS.get(key, 0.0)
    
    def get_correlated_assets(self, symbol: str) -> Set[str]:
        correlated = set()
        for (s1, s2), corr in self.CORRELATIONS.items():
            if corr >= self.config.correlation_threshold:
                if s1 == symbol: correlated.add(s2)
                elif s2 == symbol: correlated.add(s1)
        return correlated
    
    def trigger_spillover(self, symbol: str, breach_pct: float) -> Dict[str, float]:
        """Cascade limit reduction to correlated assets."""
        adjustments = {}
        self._breach_history[symbol] = breach_pct
        
        correlated = self.get_correlated_assets(symbol)
        for asset in correlated:
            if asset not in self._base_limits:
                continue
            corr = self.get_correlation(symbol, asset)
            reduction = breach_pct * self.config.spillover_factor * corr
            new_limit = self._base_limits[asset] * (1 - reduction)
            self._active_limits[asset] = max(0.01, new_limit)
            adjustments[asset] = self._active_limits[asset]
        
        return adjustments
    
    def get_active_limit(self, symbol: str) -> float:
        return self._active_limits.get(symbol, self._base_limits.get(symbol, 1.0))
    
    def reset_limits(self):
        self._active_limits = self._base_limits.copy()
        self._breach_history.clear()
