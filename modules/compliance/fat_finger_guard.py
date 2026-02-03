"""
Fat-Finger Protection - Catastrophic Order Error Prevention
Author: Erdinc Erdogan
Purpose: Prevents catastrophic order errors through multi-layer validation including size,
price deviation, and notional value checks before order submission.
References:
- Fat-Finger Error Prevention in Trading Systems
- Order Validation Best Practices
- Price Band Circuit Breakers
Usage:
    guard = FatFingerGuard(config=FatFingerConfig())
    result = guard.validate_order(order_size=100000, price=50000, symbol="BTCUSD")
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import time

class RejectionReason(Enum):
    SIZE_TOO_LARGE = "FF-001"
    SIZE_TOO_SMALL = "FF-002"
    PRICE_DEVIATION = "FF-003"
    NOTIONAL_EXCEEDED = "FF-004"
    PRICE_BAND_BREACH = "FF-005"

@dataclass
class FatFingerConfig:
    max_order_size_usd: float = 100_000
    min_order_size_usd: float = 10
    max_price_deviation_pct: float = 0.05  # 5% from mid
    max_notional_per_order: float = 500_000
    price_band_multiplier: float = 0.10  # 10% circuit breaker band

class FatFingerGuard:
    """Multi-layer fat-finger protection with O(1) validation."""
    
    def __init__(self, config: Optional[FatFingerConfig] = None):
        self.config = config or FatFingerConfig()
        self._mid_prices: Dict[str, float] = {}
        self._price_bands: Dict[str, Tuple[float, float]] = {}
        self._last_update: Dict[str, float] = {}
    
    def update_mid_price(self, symbol: str, mid_price: float):
        """Update reference mid-price for deviation checks."""
        self._mid_prices[symbol] = mid_price
        band = self.config.price_band_multiplier
        self._price_bands[symbol] = (mid_price * (1 - band), mid_price * (1 + band))
        self._last_update[symbol] = time.time()
    
    def validate_order(self, symbol: str, side: str, quantity: float, 
                       price: float) -> Tuple[bool, Optional[str], str]:
        """O(1) multi-layer validation pipeline."""
        
        notional = quantity * price
        
        # Layer 1: Notional size check
        if notional > self.config.max_notional_per_order:
            return False, RejectionReason.NOTIONAL_EXCEEDED.value,                    f"Notional ${notional:,.0f} > ${self.config.max_notional_per_order:,.0f}"
        
        if notional > self.config.max_order_size_usd:
            return False, RejectionReason.SIZE_TOO_LARGE.value,                    f"Size ${notional:,.0f} > ${self.config.max_order_size_usd:,.0f}"
        
        if notional < self.config.min_order_size_usd:
            return False, RejectionReason.SIZE_TOO_SMALL.value,                    f"Size ${notional:,.2f} < ${self.config.min_order_size_usd:,.0f}"
        
        # Layer 2: Price deviation from mid
        mid = self._mid_prices.get(symbol)
        if mid and mid > 0:
            deviation = abs(price - mid) / mid
            if deviation > self.config.max_price_deviation_pct:
                return False, RejectionReason.PRICE_DEVIATION.value,                        f"Price deviation {deviation:.1%} > {self.config.max_price_deviation_pct:.1%}"
        
        # Layer 3: Price band (circuit breaker style)
        if symbol in self._price_bands:
            low, high = self._price_bands[symbol]
            if price < low or price > high:
                return False, RejectionReason.PRICE_BAND_BREACH.value,                        f"Price ${price:,.2f} outside band [${low:,.2f}, ${high:,.2f}]"
        
        return True, None, "OK"
    
    def get_safe_price_range(self, symbol: str) -> Tuple[float, float]:
        """Get acceptable price range for symbol."""
        return self._price_bands.get(symbol, (0.0, float('inf')))
