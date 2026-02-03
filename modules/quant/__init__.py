"""
NeuralTrade Quant Module
Author: Erdinc Erdogan
"""

from .backtest import *
from .backtest_v2 import *
from .paper_trading import *

# NEW: Institutional Backtest Engine
from .backtest_engine import (
    BacktestEngine,
    BacktestMode,
    OrderSide,
    OrderType,PositionStatus,
    Order,
    Trade,
    Position,
    PerformanceMetrics,
    DrawdownAnalysis,
    WalkForwardResult,
    MonteCarloResult,
    BacktestResult,
)

__all__ = [
    'Backtest', 'BacktestV2', 'PaperTrading',
    # NEW
    'BacktestEngine', 'BacktestMode', 'OrderSide', 'OrderType',
    'PositionStatus', 'Order', 'Trade', 'Position', 'PerformanceMetrics',
    'DrawdownAnalysis', 'WalkForwardResult', 'MonteCarloResult', 'BacktestResult',
]