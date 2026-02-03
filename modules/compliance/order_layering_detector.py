"""
Order Layering Detector - Coordinated Spoofing Detection
Author: Erdinc Erdogan
Purpose: Detects coordinated order layering and spoofing patterns where multiple orders
are placed and rapidly cancelled to manipulate market prices.
References:
- CFTC Anti-Spoofing Regulations
- Order Book Manipulation Detection
- High-Cancel-Rate Pattern Analysis
Usage:
    detector = OrderLayeringDetector(config=LayeringConfig())
    detector.record_order(oid="001", aid="agent_01", sym="BTCUSD", ...)
    is_layering = detector.check_layering("BTCUSD")
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set

@dataclass
class LayeringConfig:
    window_ms: int = 2000
    min_layers: int = 3
    cancel_ratio_threshold: float = 0.80
    price_cluster_pct: float = 0.005

class OrderLayeringDetector:
    __slots__ = ('config', '_orders', '_cancels', '_agent_orders')
    
    def __init__(self, config: Optional[LayeringConfig] = None):
        self.config = config or LayeringConfig()
        self._orders: Dict[str, deque] = {}
        self._cancels: Dict[str, int] = {}
        self._agent_orders: Dict[str, Set[str]] = {}
    
    def record_order(self, oid: str, aid: str, sym: str, side: str, price: float, qty: float, ts: int):
        if sym not in self._orders:
            self._orders[sym] = deque(maxlen=500)
        self._orders[sym].append((ts, oid, aid, side, price, qty))
        self._agent_orders.setdefault(aid, set()).add(oid)
    
    def record_cancel(self, oid: str, sym: str, ts: int):
        self._cancels[sym] = self._cancels.get(sym, 0) + 1
    
    def detect_layering(self, sym: str, ts: int) -> Tuple[bool, Optional[str]]:
        if sym not in self._orders:
            return False, None
        cutoff = ts - self.config.window_ms
        recent = [r for r in self._orders[sym] if r[0] >= cutoff]
        if len(recent) < self.config.min_layers:
            return False, None
        prices = [r[4] for r in recent]
        mid = sum(prices) / len(prices)
        clustered = sum(1 for p in prices if abs(p - mid) / mid < self.config.price_cluster_pct)
        if clustered >= self.config.min_layers:
            agents = set(r[2] for r in recent)
            if len(agents) > 1:
                return True, f"LY-001: {len(agents)} agents, {clustered} clustered"
        return False, None
