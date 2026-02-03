"""
Order Book Imbalance (OBI) Engine
Author: Erdinc Erdogan
Purpose: Calculates volume imbalance between bid/ask levels using price-weighted depth analysis to detect micro-trends and institutional positioning.
References:
- Order Flow Imbalance (Cont et al., 2014)
- Price-Weighted Depth Analysis
- Microstructure-Based Trading Signals
Usage:
    engine = OrderBookImbalanceEngine(n_levels=10)
    result = engine.calculate(order_book_snapshot)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum


class OBISignal(IntEnum):
    """Order book imbalance signal strength."""
    STRONG_SELL = -2
    WEAK_SELL = -1
    NEUTRAL = 0
    WEAK_BUY = 1
    STRONG_BUY = 2


@dataclass
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    timestamp: float
    bids: List[OrderBookLevel]  # Sorted descending by price
    asks: List[OrderBookLevel]  # Sorted ascending by price
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0


@dataclass
class OBIResult:
    """Result of OBI calculation."""
    imbalance: float              # Raw imbalance [-1, 1]
    signal: OBISignal             # Discretized signal
    bid_depth: float              # Total bid volume
    ask_depth: float              # Total ask volume
    weighted_imbalance: float     # Price-weighted imbalance
    momentum: float               # Imbalance momentum (rate of change)
    confidence: float             # Signal confidence [0, 1]
    micro_trend: str              # "BULLISH", "BEARISH", "NEUTRAL"


class OrderBookImbalanceEngine:
    """
    Order Book Imbalance (OBI) Engine for micro-trend detection.
    
    Calculates multiple imbalance metrics:
    1. Volume Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    2. Weighted Imbalance: Price-distance weighted volume imbalance
    3. Depth Imbalance: Cumulative depth at N levels
    4. Order Flow Imbalance: Rate of change in imbalance
    
    Mathematical Foundation:
    OBI = Σ(bid_qty_i * w_i) - Σ(ask_qty_i * w_i) / Σ(bid_qty_i * w_i) + Σ(ask_qty_i * w_i)
    
    Where w_i = exp(-λ * distance_from_mid_i) for price-weighted version
    
    Usage:
        engine = OrderBookImbalanceEngine(n_levels=10)
        result = engine.calculate(order_book_snapshot)
        if result.signal >= OBISignal.WEAK_BUY:
            # Bullish micro-trend detected
    """
    
    def __init__(
        self,
        n_levels: int = 10,
        decay_factor: float = 0.5,
        strong_threshold: float = 0.6,
        weak_threshold: float = 0.2,
        momentum_window: int = 20,
        ema_alpha: float = 0.1,
        min_depth_for_signal: float = 1000.0
    ):
        self.n_levels = n_levels
        self.decay_factor = decay_factor
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.momentum_window = momentum_window
        self.ema_alpha = ema_alpha
        self.min_depth_for_signal = min_depth_for_signal
        
        # State
        self.imbalance_history: deque = deque(maxlen=momentum_window)
        self.ema_imbalance: float = 0.0
        self.last_result: Optional[OBIResult] = None
        
    def calculate(self, snapshot: OrderBookSnapshot) -> OBIResult:
        """
        Calculate order book imbalance from snapshot.
        
        Args:
            snapshot: OrderBookSnapshot with bids and asks
            
        Returns:
            OBIResult with all imbalance metrics
        """
        # Extract top N levels
        bids = snapshot.bids[:self.n_levels]
        asks = snapshot.asks[:self.n_levels]
        
        if not bids or not asks:
            return self._empty_result()
        
        mid_price = snapshot.mid_price
        
        # Calculate raw volume imbalance
        bid_depth = sum(level.quantity for level in bids)
        ask_depth = sum(level.quantity for level in asks)
        total_depth = bid_depth + ask_depth
        
        if total_depth < 1e-10:
            return self._empty_result()
        
        raw_imbalance = (bid_depth - ask_depth) / total_depth
        
        # Calculate price-weighted imbalance
        weighted_imbalance = self._calculate_weighted_imbalance(
            bids, asks, mid_price
        )
        
        # Update EMA
        self.ema_imbalance = (
            self.ema_alpha * raw_imbalance + 
            (1 - self.ema_alpha) * self.ema_imbalance
        )
        
        # Calculate momentum
        self.imbalance_history.append(raw_imbalance)
        momentum = self._calculate_momentum()
        
        # Determine signal
        signal = self._classify_signal(self.ema_imbalance)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            raw_imbalance, weighted_imbalance, total_depth
        )
        
        # Determine micro-trend
        micro_trend = self._determine_micro_trend(signal, momentum, confidence)
        
        result = OBIResult(
            imbalance=raw_imbalance,
            signal=signal,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            weighted_imbalance=weighted_imbalance,
            momentum=momentum,
            confidence=confidence,
            micro_trend=micro_trend
        )
        
        self.last_result = result
        return result
    
    def _calculate_weighted_imbalance(
        self,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        mid_price: float
    ) -> float:
        """Calculate price-distance weighted imbalance."""
        if mid_price <= 0:
            return 0.0
        
        weighted_bid = 0.0
        weighted_ask = 0.0
        
        for level in bids:
            distance = abs(level.price - mid_price) / mid_price
            weight = np.exp(-self.decay_factor * distance * 100)  # Scale distance
            weighted_bid += level.quantity * weight
        
        for level in asks:
            distance = abs(level.price - mid_price) / mid_price
            weight = np.exp(-self.decay_factor * distance * 100)
            weighted_ask += level.quantity * weight
        
        total = weighted_bid + weighted_ask
        if total < 1e-10:
            return 0.0
        
        return (weighted_bid - weighted_ask) / total
    
    def _calculate_momentum(self) -> float:
        """Calculate imbalance momentum (rate of change)."""
        if len(self.imbalance_history) < 2:
            return 0.0
        
        history = list(self.imbalance_history)
        
        # Simple linear regression slope
        n = len(history)
        x = np.arange(n)
        y = np.array(history)
        
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def _classify_signal(self, imbalance: float) -> OBISignal:
        """Classify imbalance into discrete signal."""
        if imbalance >= self.strong_threshold:
            return OBISignal.STRONG_BUY
        elif imbalance >= self.weak_threshold:
            return OBISignal.WEAK_BUY
        elif imbalance <= -self.strong_threshold:
            return OBISignal.STRONG_SELL
        elif imbalance <= -self.weak_threshold:
            return OBISignal.WEAK_SELL
        else:
            return OBISignal.NEUTRAL
    
    def _calculate_confidence(
        self,
        raw_imbalance: float,
        weighted_imbalance: float,
        total_depth: float
    ) -> float:
        """Calculate signal confidence."""
        # Agreement between raw and weighted
        agreement = 1.0 - abs(raw_imbalance - weighted_imbalance)
        
        # Depth factor (more depth = more confidence)
        depth_factor = min(1.0, total_depth / self.min_depth_for_signal)
        
        # Magnitude factor (stronger signal = more confidence)
        magnitude = abs(raw_imbalance)
        
        confidence = agreement * depth_factor * (0.5 + 0.5 * magnitude)
        return np.clip(confidence, 0.0, 1.0)
    
    def _determine_micro_trend(
        self,
        signal: OBISignal,
        momentum: float,
        confidence: float
    ) -> str:
        """Determine micro-trend direction."""
        if confidence < 0.3:
            return "NEUTRAL"
        
        if signal >= OBISignal.WEAK_BUY and momentum > 0:
            return "BULLISH"
        elif signal <= OBISignal.WEAK_SELL and momentum < 0:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _empty_result(self) -> OBIResult:
        """Return empty result for invalid input."""
        return OBIResult(
            imbalance=0.0,
            signal=OBISignal.NEUTRAL,
            bid_depth=0.0,
            ask_depth=0.0,
            weighted_imbalance=0.0,
            momentum=0.0,
            confidence=0.0,
            micro_trend="NEUTRAL"
        )
    
    def get_swarm_signal(self) -> Dict:
        """
        Get OBI signal formatted for Swarm integration.
        
        Returns:
            Dict with signal data for swarm agents
        """
        if self.last_result is None:
            return {
                "obi_signal": 0,
                "obi_confidence": 0.0,
                "obi_micro_trend": "NEUTRAL",
                "obi_momentum": 0.0
            }
        
        return {
            "obi_signal": int(self.last_result.signal),
            "obi_confidence": self.last_result.confidence,
            "obi_micro_trend": self.last_result.micro_trend,
            "obi_momentum": self.last_result.momentum,
            "obi_imbalance": self.last_result.imbalance,
            "obi_weighted": self.last_result.weighted_imbalance
        }
    
    def reset(self):
        """Reset engine state."""
        self.imbalance_history.clear()
        self.ema_imbalance = 0.0
        self.last_result = None


class MultiAssetOBIEngine:
    """
    Multi-asset OBI engine for cross-market analysis.
    
    Tracks OBI across multiple instruments and detects
    cross-asset divergences that may signal regime changes.
    """
    
    def __init__(self, symbols: List[str], **kwargs):
        self.symbols = symbols
        self.engines = {sym: OrderBookImbalanceEngine(**kwargs) for sym in symbols}
        self.cross_correlation_window = kwargs.get('correlation_window', 50)
        self.imbalance_matrix: Dict[str, deque] = {
            sym: deque(maxlen=self.cross_correlation_window) for sym in symbols
        }
    
    def update(self, symbol: str, snapshot: OrderBookSnapshot) -> OBIResult:
        """Update OBI for a specific symbol."""
        if symbol not in self.engines:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        result = self.engines[symbol].calculate(snapshot)
        self.imbalance_matrix[symbol].append(result.imbalance)
        return result
    
    def get_cross_asset_divergence(self) -> float:
        """
        Calculate cross-asset OBI divergence.
        
        High divergence may indicate regime instability.
        """
        if len(self.symbols) < 2:
            return 0.0
        
        imbalances = []
        for sym in self.symbols:
            if self.imbalance_matrix[sym]:
                imbalances.append(list(self.imbalance_matrix[sym])[-1])
        
        if len(imbalances) < 2:
            return 0.0
        
        # Standard deviation of imbalances across assets
        return np.std(imbalances)
    
    def get_aggregate_signal(self) -> OBISignal:
        """Get consensus OBI signal across all assets."""
        signals = []
        for sym in self.symbols:
            result = self.engines[sym].last_result
            if result:
                signals.append(int(result.signal))
        
        if not signals:
            return OBISignal.NEUTRAL
        
        avg_signal = np.mean(signals)
        return OBISignal(int(np.round(avg_signal)))
