"""
NeuralTrade Risk Module
Author: Erdinc Erdogan
"""

from .cascading_circuit_breaker import *
from .circuit_breaker import *
from .cvar_engine import *
from .expected_shortfall import *
from .risk_engine import *
from .risk_engine_hardened import *
from .var_models import *

# NEW: Correlation Engine
from .correlation_engine import (
    CorrelationEngine,
    CorrelationMethod,
    CopulaType,
    CorrelationMatrix,
    DynamicCorrelation,
    DCCGARCHResult,
    CopulaResult,
    ShrinkageResult,
    RMTCleanedMatrix,
    CorrelationStressTest,
    HierarchicalCluster,
    CorrelationRegime,
)

# NEW: Risk Manager
from .risk_manager import (
    RiskManager,
    PositionSizingMethod,
    StopLossType,
    RiskLimitType,
    PositionSize,
    StopLossLevel,
    RiskLimit,PortfolioRiskMetrics,
    RiskAdjustment,
    KellyCriterionResult,
    OptimalFResult,
    DrawdownControl,
)

__all__ = [
    'CascadingCircuitBreaker', 'CircuitBreaker', 'CVaREngine',
    'ExpectedShortfall', 'RiskEngine', 'RiskEngineHardened', 'VaRModels',
    # NEW: Correlation Engine
    'CorrelationEngine', 'CorrelationMethod', 'CopulaType', 'CorrelationMatrix',
    'DynamicCorrelation', 'DCCGARCHResult', 'CopulaResult', 'ShrinkageResult',
    'RMTCleanedMatrix', 'CorrelationStressTest', 'HierarchicalCluster',
    'CorrelationRegime',
    # NEW: Risk Manager
    'RiskManager', 'PositionSizingMethod', 'StopLossType', 'RiskLimitType',
    'PositionSize', 'StopLossLevel', 'RiskLimit', 'PortfolioRiskMetrics',
    'RiskAdjustment', 'KellyCriterionResult', 'OptimalFResult', 'DrawdownControl',
]