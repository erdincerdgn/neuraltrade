"""
Manipulation Detector - Wash Trading and Spoofing Detection
Author: Erdinc Erdogan
Purpose: Detects market manipulation patterns including wash trading and spoofing in
multi-agent order flows to prevent regulatory violations.
References:
- SEC Market Manipulation Rules
- Wash Trading Detection Algorithms
- Spoofing Pattern Recognition
Usage:
    detector = ManipulationDetector()
    is_wash = detector.check_wash_trade(new_order, recent_orders)
    is_spoof = detector.check_spoofing(order_book_history)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone
from collections import deque

@dataclass
class Order:
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    timestamp: float
    agent_id: str

class ManipulationDetector:
    """Detects market manipulation patterns."""
    
    WASH_TRADE_WINDOW_SEC = 60  # 60-second window
    SPOOF_CANCEL_RATIO = 0.8   # 80% cancel ratio = spoofing
    
    def __init__(self):
        self._recent_orders: Dict[str, deque] = {}  # symbol -> orders
        self._cancel_counts: Dict[str, int] = {}
        self._fill_counts: Dict[str, int] = {}
    
    def check_wash_trade(self, order: Order) -> bool:
        """Check if order creates wash trade pattern."""
        symbol = order.symbol
        if symbol not in self._recent_orders:
            self._recent_orders[symbol] = deque(maxlen=100)
        
        now = order.timestamp
        opposite_side = "SELL" if order.side == "BUY" else "BUY"
        
        for prev in self._recent_orders[symbol]:
            if now - prev.timestamp > self.WASH_TRADE_WINDOW_SEC:
                continue
            if prev.side == opposite_side and abs(prev.price - order.price) / order.price < 0.001:
                if prev.agent_id == order.agent_id or self._same_entity(prev.agent_id, order.agent_id):
                    return True  # Wash trade detected
        
        self._recent_orders[symbol].append(order)
        return False
    
    def check_spoofing(self, agent_id: str) -> bool:
        """Check if agent exhibits spoofing behavior."""
        cancels = self._cancel_counts.get(agent_id, 0)
        fills = self._fill_counts.get(agent_id, 0)
        total = cancels + fills
        if total < 10:
            return False
        return (cancels / total) > self.SPOOF_CANCEL_RATIO
    
    def record_cancel(self, agent_id: str):
        self._cancel_counts[agent_id] = self._cancel_counts.get(agent_id, 0) + 1
    
    def record_fill(self, agent_id: str):
        self._fill_counts[agent_id] = self._fill_counts.get(agent_id, 0) + 1
    
    def _same_entity(self, agent1: str, agent2: str) -> bool:
        """Check if two agents belong to same entity."""
        return agent1.split("_")[0] == agent2.split("_")[0]
