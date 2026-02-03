"""
NeuralTrade Generated Protocol Buffer Modules
==============================================
Auto-generated Python modules from ai_service.proto v2.0

Generated Files:
    - ai_service_pb2.py: Message classes and enums
    - ai_service_pb2_grpc.py: gRPC service stubs and servicers

Regenerate Command:
    python -m grpc_tools.protoc \
        -I./proto \
        --python_out=./proto/generated \
        --grpc_python_out=./proto/generated \
        ./proto/ai_service.proto

Usage:
    # Import message classes
    from proto.generated.ai_service_pb2 import (
        SignalRequest,
        SignalResponse,
        VolSurfaceRequest,
        VolSurfaceResponse,
        GreeksRequest,
        GreeksResponse,
        RiskMetricsRequest,
        RiskMetricsResponse,PortfolioOptRequest,
        PortfolioOptResponse,
    )
    
    # Import service stubs
    from proto.generated.ai_service_pb2_grpc import (
        AIServiceStub,
        AIServiceServicer,
        add_AIServiceServicer_to_server,
    )
    
    # Import enums
    from proto.generated.ai_service_pb2 import (
        OptionType,
        SignalAction,
        MarketRegime,VolatilityModel,
        OptimizationMethod,
    )

Version: 2.0.0
"""

from __future__ import absolute_import

__version__ = "2.0.0"

# Import generated modules
try:
    from . import ai_service_pb2
    from . import ai_service_pb2_grpc
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import generated proto modules: {e}. "
        "Please regenerate using: python -m grpc_tools.protoc ..."
    )
    ai_service_pb2 = None
    ai_service_pb2_grpc = None

# =============================================================================
# MESSAGE CLASSES
# =============================================================================

# Signal & Prediction
SignalRequest = getattr(ai_service_pb2, 'SignalRequest', None)
SignalResponse = getattr(ai_service_pb2, 'SignalResponse', None)
SignalStreamRequest = getattr(ai_service_pb2, 'SignalStreamRequest', None)
SignalContributor = getattr(ai_service_pb2, 'SignalContributor', None)

# Model Management
ModelSelectionRequest = getattr(ai_service_pb2, 'ModelSelectionRequest', None)
ModelSelectionResponse = getattr(ai_service_pb2, 'ModelSelectionResponse', None)
ListModelsRequest = getattr(ai_service_pb2, 'ListModelsRequest', None)
ListModelsResponse = getattr(ai_service_pb2, 'ListModelsResponse', None)
ModelInfo = getattr(ai_service_pb2, 'ModelInfo', None)
ModelPerformance = getattr(ai_service_pb2, 'ModelPerformance', None)
ModelMetricsRequest = getattr(ai_service_pb2, 'ModelMetricsRequest', None)
ModelMetricsResponse = getattr(ai_service_pb2, 'ModelMetricsResponse', None)
PredictionRecord = getattr(ai_service_pb2, 'PredictionRecord', None)
ConfusionMatrix = getattr(ai_service_pb2, 'ConfusionMatrix', None)

# Strategy Routing
StrategyRequest = getattr(ai_service_pb2, 'StrategyRequest', None)
StrategyResponse = getattr(ai_service_pb2, 'StrategyResponse', None)
ListStrategiesRequest = getattr(ai_service_pb2, 'ListStrategiesRequest', None)
ListStrategiesResponse = getattr(ai_service_pb2, 'ListStrategiesResponse', None)
StrategyInfo = getattr(ai_service_pb2, 'StrategyInfo', None)
StrategyPerformance = getattr(ai_service_pb2, 'StrategyPerformance', None)
BacktestRequest = getattr(ai_service_pb2, 'BacktestRequest', None)
BacktestResponse = getattr(ai_service_pb2, 'BacktestResponse', None)
RiskLimits = getattr(ai_service_pb2, 'RiskLimits', None)
TradeRecord = getattr(ai_service_pb2, 'TradeRecord', None)
EquityCurve = getattr(ai_service_pb2, 'EquityCurve', None)
DrawdownAnalysis = getattr(ai_service_pb2, 'DrawdownAnalysis', None)
DrawdownPeriod = getattr(ai_service_pb2, 'DrawdownPeriod', None)

# Volatility Surface (NEW v2.0)
VolSurfaceRequest = getattr(ai_service_pb2, 'VolSurfaceRequest', None)
VolSurfaceResponse = getattr(ai_service_pb2, 'VolSurfaceResponse', None)
VolSurfacePoint = getattr(ai_service_pb2, 'VolSurfacePoint', None)
CalibratedSmile = getattr(ai_service_pb2, 'CalibratedSmile', None)
SABRParams = getattr(ai_service_pb2, 'SABRParams', None)
SVIParams = getattr(ai_service_pb2, 'SVIParams', None)
ImpliedVolRequest = getattr(ai_service_pb2, 'ImpliedVolRequest', None)
ImpliedVolResponse = getattr(ai_service_pb2, 'ImpliedVolResponse', None)
LocalVolRequest = getattr(ai_service_pb2, 'LocalVolRequest', None)
LocalVolResponse = getattr(ai_service_pb2, 'LocalVolResponse', None)
SkewMetricsRequest = getattr(ai_service_pb2, 'SkewMetricsRequest', None)
SkewMetricsResponse = getattr(ai_service_pb2, 'SkewMetricsResponse', None)
ArbitrageCheckRequest = getattr(ai_service_pb2, 'ArbitrageCheckRequest', None)
ArbitrageCheckResponse = getattr(ai_service_pb2, 'ArbitrageCheckResponse', None)
ArbitragePoint = getattr(ai_service_pb2, 'ArbitragePoint', None)

# Options & Greeks (NEW v2.0)
GreeksRequest = getattr(ai_service_pb2, 'GreeksRequest', None)
GreeksResponse = getattr(ai_service_pb2, 'GreeksResponse', None)
OptionPricingRequest = getattr(ai_service_pb2, 'OptionPricingRequest', None)
OptionPricingResponse = getattr(ai_service_pb2, 'OptionPricingResponse', None)
OptionsChainRequest = getattr(ai_service_pb2, 'OptionsChainRequest', None)
OptionsChainResponse = getattr(ai_service_pb2, 'OptionsChainResponse', None)
OptionChainExpiry = getattr(ai_service_pb2, 'OptionChainExpiry', None)
OptionChainStrike = getattr(ai_service_pb2, 'OptionChainStrike', None)

# Risk Management (NEW v2.0)
RiskMetricsRequest = getattr(ai_service_pb2, 'RiskMetricsRequest', None)
RiskMetricsResponse = getattr(ai_service_pb2, 'RiskMetricsResponse', None)
VaRRequest = getattr(ai_service_pb2, 'VaRRequest', None)
VaRResponse = getattr(ai_service_pb2, 'VaRResponse', None)
CVaRRequest = getattr(ai_service_pb2, 'CVaRRequest', None)
CVaRResponse = getattr(ai_service_pb2, 'CVaRResponse', None)
StressTestRequest = getattr(ai_service_pb2, 'StressTestRequest', None)
StressTestResponse = getattr(ai_service_pb2, 'StressTestResponse', None)
StressScenario = getattr(ai_service_pb2, 'StressScenario', None)
StressTestResult = getattr(ai_service_pb2, 'StressTestResult', None)

# Portfolio Optimization (NEW v2.0)
PortfolioOptRequest = getattr(ai_service_pb2, 'PortfolioOptRequest', None)
PortfolioOptResponse = getattr(ai_service_pb2, 'PortfolioOptResponse', None)
PortfolioConstraints = getattr(ai_service_pb2, 'PortfolioConstraints', None)
EfficientFrontier = getattr(ai_service_pb2, 'EfficientFrontier', None)
HRPRequest = getattr(ai_service_pb2, 'HRPRequest', None)
HRPResponse = getattr(ai_service_pb2, 'HRPResponse', None)
ClusterInfo = getattr(ai_service_pb2, 'ClusterInfo', None)
BlackLittermanRequest = getattr(ai_service_pb2, 'BlackLittermanRequest', None)
BlackLittermanResponse = getattr(ai_service_pb2, 'BlackLittermanResponse', None)
ViewInfo = getattr(ai_service_pb2, 'ViewInfo', None)

# Market Analysis
RegimeDetectionRequest = getattr(ai_service_pb2, 'RegimeDetectionRequest', None)
RegimeDetectionResponse = getattr(ai_service_pb2, 'RegimeDetectionResponse', None)
RegimeProbability = getattr(ai_service_pb2, 'RegimeProbability', None)
RegimeTransitionMatrix = getattr(ai_service_pb2, 'RegimeTransitionMatrix', None)
SentimentRequest = getattr(ai_service_pb2, 'SentimentRequest', None)
SentimentResponse = getattr(ai_service_pb2, 'SentimentResponse', None)
SentimentItem = getattr(ai_service_pb2, 'SentimentItem', None)
SentimentTrend = getattr(ai_service_pb2, 'SentimentTrend', None)
MicrostructureRequest = getattr(ai_service_pb2, 'MicrostructureRequest', None)
MicrostructureResponse = getattr(ai_service_pb2, 'MicrostructureResponse', None)
ToxicityMetrics = getattr(ai_service_pb2, 'ToxicityMetrics', None)

# Health & Monitoring
HealthRequest = getattr(ai_service_pb2, 'HealthRequest', None)
HealthResponse = getattr(ai_service_pb2, 'HealthResponse', None)
SystemResources = getattr(ai_service_pb2, 'SystemResources', None)
MetricsRequest = getattr(ai_service_pb2, 'MetricsRequest', None)
MetricsResponse = getattr(ai_service_pb2, 'MetricsResponse', None)
LatencyBucket = getattr(ai_service_pb2, 'LatencyBucket', None)

# =============================================================================
# ENUMS
# =============================================================================

OptionType = getattr(ai_service_pb2, 'OptionType', None)
SignalAction = getattr(ai_service_pb2, 'SignalAction', None)
MarketRegime = getattr(ai_service_pb2, 'MarketRegime', None)
ModelType = getattr(ai_service_pb2, 'ModelType', None)
VolatilityModel = getattr(ai_service_pb2, 'VolatilityModel', None)
RiskProfile = getattr(ai_service_pb2, 'RiskProfile', None)
ExecutionType = getattr(ai_service_pb2, 'ExecutionType', None)
OptimizationMethod = getattr(ai_service_pb2, 'OptimizationMethod', None)

# =============================================================================
# SERVICE STUBS & SERVICERS
# =============================================================================

AIServiceStub = getattr(ai_service_pb2_grpc, 'AIServiceStub', None)
AIServiceServicer = getattr(ai_service_pb2_grpc, 'AIServiceServicer', None)
add_AIServiceServicer_to_server = getattr(ai_service_pb2_grpc, 'add_AIServiceServicer_to_server', None)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Modules
    "ai_service_pb2",
    "ai_service_pb2_grpc",
    
    # Service
    "AIServiceStub",
    "AIServiceServicer",
    "add_AIServiceServicer_to_server",
    
    # Enums
    "OptionType",
    "SignalAction",
    "MarketRegime",
    "ModelType",
    "VolatilityModel",
    "RiskProfile",
    "ExecutionType",
    "OptimizationMethod",
    
    # Signal Messages
    "SignalRequest",
    "SignalResponse",
    "SignalStreamRequest",
    "SignalContributor",
    
    # Model Messages
    "ModelSelectionRequest",
    "ModelSelectionResponse",
    "ListModelsRequest",
    "ListModelsResponse",
    "ModelInfo",
    "ModelPerformance",
    "ModelMetricsRequest",
    "ModelMetricsResponse",
    "PredictionRecord",
    "ConfusionMatrix",
    
    # Strategy Messages
    "StrategyRequest",
    "StrategyResponse",
    "ListStrategiesRequest",
    "ListStrategiesResponse",
    "StrategyInfo",
    "StrategyPerformance",
    "BacktestRequest",
    "BacktestResponse",
    "RiskLimits",
    "TradeRecord",
    "EquityCurve",
    "DrawdownAnalysis",
    "DrawdownPeriod",
    
    # Volatility Surface Messages
    "VolSurfaceRequest",
    "VolSurfaceResponse",
    "VolSurfacePoint",
    "CalibratedSmile",
    "SABRParams",
    "SVIParams",
    "ImpliedVolRequest",
    "ImpliedVolResponse",
    "LocalVolRequest",
    "LocalVolResponse",
    "SkewMetricsRequest",
    "SkewMetricsResponse",
    "ArbitrageCheckRequest",
    "ArbitrageCheckResponse",
    "ArbitragePoint",
    
    # Greeks Messages
    "GreeksRequest",
    "GreeksResponse",
    "OptionPricingRequest",
    "OptionPricingResponse",
    "OptionsChainRequest",
    "OptionsChainResponse",
    "OptionChainExpiry",
    "OptionChainStrike",
    
    # Risk Messages
    "RiskMetricsRequest",
    "RiskMetricsResponse",
    "VaRRequest",
    "VaRResponse",
    "CVaRRequest",
    "CVaRResponse",
    "StressTestRequest",
    "StressTestResponse",
    "StressScenario",
    "StressTestResult",
    
    # Portfolio Messages
    "PortfolioOptRequest",
    "PortfolioOptResponse",
    "PortfolioConstraints",
    "EfficientFrontier",
    "HRPRequest",
    "HRPResponse",
    "ClusterInfo",
    "BlackLittermanRequest",
    "BlackLittermanResponse",
    "ViewInfo",
    
    # Market Analysis Messages
    "RegimeDetectionRequest",
    "RegimeDetectionResponse",
    "RegimeProbability",
    "RegimeTransitionMatrix",
    "SentimentRequest",
    "SentimentResponse",
    "SentimentItem",
    "SentimentTrend",
    "MicrostructureRequest",
    "MicrostructureResponse",
    "ToxicityMetrics",
    
    # Health Messages
    "HealthRequest",
    "HealthResponse",
    "SystemResources",
    "MetricsRequest",
    "MetricsResponse",
    "LatencyBucket",
]