"""
Circuit Breaker - Multi-Level Flash Crash Protection
Author: Erdinc Erdogan
Purpose: Implements multi-level circuit breakers for flash crash protection with automatic
recovery, providing trading halts at configurable price movement thresholds.
References:
- NYSE/NASDAQ Circuit Breaker Rules
- Flash Crash Protection Mechanisms
- State Machine Pattern for Trading Controls
Usage:
    breaker = CircuitBreaker(config=BreakerConfig())
    state = breaker.check_price_move(current_price=100.0, previous_price=95.0)
    if state == BreakerState.OPEN: halt_trading()
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
from collections import deque

class BreakerState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Trading halted
    HALF_OPEN = "HALF_OPEN"  # Testing recovery

@dataclass
class BreakerConfig:
    level1_threshold: float = 0.05   # 5% move -> slow mode
    level2_threshold: float = 0.10   # 10% move -> halt 5min
    level3_threshold: float = 0.20   # 20% move -> halt 15min
    window_seconds: int = 60         # Price window
    cooldown_seconds: int = 300      # Recovery cooldown

class CircuitBreaker:
    """Multi-level circuit breaker for flash crash protection."""
    
    def __init__(self, config: Optional[BreakerConfig] = None):
        self.config = config or BreakerConfig()
        self._states: Dict[str, BreakerState] = {}
        self._price_history: Dict[str, deque] = {}
        self._trip_times: Dict[str, float] = {}
        self._halt_durations: Dict[str, int] = {}
    
    def record_price(self, symbol: str, price: float, timestamp: float):
        """Record price tick and check for breaker trip."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=1000)
            self._states[symbol] = BreakerState.CLOSED
        
        self._price_history[symbol].append((timestamp, price))
        self._cleanup_old_prices(symbol, timestamp)
        return self._check_trip(symbol, price, timestamp)
    
    def _cleanup_old_prices(self, symbol: str, now: float):
        while self._price_history[symbol]:
            if now - self._price_history[symbol][0][0] > self.config.window_seconds:
                self._price_history[symbol].popleft()
            else:
                break
    
    def _check_trip(self, symbol: str, current: float, now: float) -> tuple:
        if not self._price_history[symbol]:
            return BreakerState.CLOSED, 0, 0.0
        
        prices = [p[1] for p in self._price_history[symbol]]
        ref_price = prices[0]
        move = abs(current - ref_price) / ref_price if ref_price > 0 else 0
        
        if move >= self.config.level3_threshold:
            self._trip(symbol, now, 900, 3)
        elif move >= self.config.level2_threshold:
            self._trip(symbol, now, 300, 2)
        elif move >= self.config.level1_threshold:
            self._trip(symbol, now, 60, 1)
        
        return self._states.get(symbol, BreakerState.CLOSED), self._get_level(symbol), move
    
    def _trip(self, symbol: str, now: float, duration: int, level: int):
        self._states[symbol] = BreakerState.OPEN
        self._trip_times[symbol] = now
        self._halt_durations[symbol] = duration
    
    def _get_level(self, symbol: str) -> int:
        dur = self._halt_durations.get(symbol, 0)
        if dur >= 900: return 3
        if dur >= 300: return 2
        if dur >= 60: return 1
        return 0
    
    def can_trade(self, symbol: str) -> tuple:
        state = self._states.get(symbol, BreakerState.CLOSED)
        if state == BreakerState.CLOSED:
            return True, "OK"
        now = time.time()
        trip_time = self._trip_times.get(symbol, 0)
        halt_dur = self._halt_durations.get(symbol, 0)
        if now - trip_time >= halt_dur:
            self._states[symbol] = BreakerState.CLOSED
            return True, "RECOVERED"
        remaining = int(halt_dur - (now - trip_time))
        return False, f"HALTED ({remaining}s remaining)"
    
    def get_status(self, symbol: str) -> Dict:
        return {"symbol": symbol, "state": self._states.get(symbol, BreakerState.CLOSED).value,
                "level": self._get_level(symbol)}
