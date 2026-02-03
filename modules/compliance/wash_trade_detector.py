"""
Optimized Wash Trade Detector - O(1) Hash-Based Detection
Author: Erdinc Erdogan
Purpose: Provides O(1) wash trade detection using rolling hash windows for real-time
identification of self-trading patterns in high-frequency environments.
References:
- Wash Trading Detection Algorithms
- Hash-Based Pattern Matching
- Rolling Window Analytics
Usage:
    detector = OptimizedWashTradeDetector(config=WashTradeConfig())
    is_wash = detector.check_order(new_order, agent_id="agent_01")
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set
import hashlib
import time

@dataclass
class WashTradeConfig:
    time_window_ms: int = 5000       # 5 second window (expanded from 1s)
    price_tolerance_pct: float = 0.001  # 0.1% price match
    size_tolerance_pct: float = 0.05    # 5% size match
    max_self_trades_per_window: int = 2
    hash_bucket_count: int = 1000    # For O(1) lookup

class OptimizedWashTradeDetector:
    """O(1) wash trade detection using hash buckets."""
    
    def __init__(self, config: Optional[WashTradeConfig] = None):
        self.config = config or WashTradeConfig()
        self._trade_buckets: Dict[str, deque] = {}
        self._agent_trade_hashes: Dict[str, Set[str]] = {}
        self._violation_count: Dict[str, int] = {}
    
    def _compute_trade_hash(self, symbol: str, price: float, size: float) -> str:
        """Compute fuzzy hash for trade matching."""
        price_bucket = int(price / (price * self.config.price_tolerance_pct))
        size_bucket = int(size / (size * self.config.size_tolerance_pct)) if size > 0 else 0
        key = f"{symbol}:{price_bucket}:{size_bucket}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def check_wash_trade(self, agent_id: str, symbol: str, side: str,
                         price: float, size: float, timestamp_ms: int) -> Tuple[bool, Optional[str], str]:
        """O(1) wash trade detection."""
        trade_hash = self._compute_trade_hash(symbol, price, size)
        bucket_key = f"{symbol}:{trade_hash}"
        
        if bucket_key not in self._trade_buckets:
            self._trade_buckets[bucket_key] = deque(maxlen=100)
        
        bucket = self._trade_buckets[bucket_key]
        cutoff = timestamp_ms - self.config.time_window_ms
        
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()
        
        opposite = "SELL" if side == "BUY" else "BUY"
        matches = sum(1 for ts, s, aid in bucket if s == opposite)
        
        if matches >= self.config.max_self_trades_per_window:
            self._violation_count[agent_id] = self._violation_count.get(agent_id, 0) + 1
            return False, "WT-001", f"Wash trade pattern: {matches} opposite matches in {self.config.time_window_ms}ms"
        
        bucket.append((timestamp_ms, side, agent_id))
        return True, None, "OK"
    
    def get_violation_count(self, agent_id: str) -> int:
        return self._violation_count.get(agent_id, 0)
    
    def reset_violations(self, agent_id: str):
        self._violation_count[agent_id] = 0
