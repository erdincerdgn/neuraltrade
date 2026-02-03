"""
NeuralTrade Execution Module
Author: Erdinc Erdogan
"""

from .calibrated_feedback import *
from .execution_feedback import *
from .execution_optimizer import *
from .executor import *
from .obi_engine import *
from .slippage import *
from .slippage_hardened import *
from .smart_order import *
from .analyzer import *
from .dark_pool import *

# NEW: Institutional Execution Engine
from .execution_engine import (
    ExecutionEngine,
    ExecutionAlgorithm,
    OrderSide,
    OrderType,
    ExecutionSchedule,
    OrderSlice,
    ExecutionResult,
    OptimalTrajectory,
    MarketImpactEstimate,
    TransactionCostAnalysis,
    VolumeProfile,
)

__all__ = [
    'CalibratedFeedback', 'ExecutionFeedback', 'ExecutionOptimizer',
    'Executor', 'OBIEngine', 'SlippageModel', 'SlippageModelHardened',
    'SmartOrderRouter', 'Analyzer', 'DarkPoolRouter',
    # NEW
    'ExecutionEngine', 'ExecutionAlgorithm', 'OrderSide', 'OrderType',
    'ExecutionSchedule', 'OrderSlice', 'ExecutionResult',
    'OptimalTrajectory', 'MarketImpactEstimate', 'TransactionCostAnalysis',
    'VolumeProfile',
]