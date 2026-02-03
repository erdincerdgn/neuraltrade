"""
NeuralTrade Backend Protocol Buffer Definitions
===============================================
gRPC service definitions for NeuralTrade Backend (Go) communication.

This module provides the proto definitions used by the Go backend
to communicate with the Python AI Engine via gRPC.

Package Structure:
    backend-be/
    └── proto/
        ├── __init__.py (this file)
        ├── ai_service.proto
        └── generated/
            ├── ai_service.pb.go
            └── ai_service_grpc.pb.go

Go Generation Command:
    protoc --go_out=./proto/generated \
           --go_opt=paths=source_relative \
           --go-grpc_out=./proto/generated \
           --go-grpc_opt=paths=source_relative \
           ./proto/ai_service.proto

Python Client Usage:
    import grpc
    from proto.generated import ai_service_pb2, ai_service_pb2_grpc
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = ai_service_pb2_grpc.AIServiceStub(channel)
    
    # Call volatility surface calibration
    request = ai_service_pb2.VolSurfaceRequest(
        symbol="AAPL",
        spot_price=150.0,
        risk_free_rate=0.05,
        model=ai_service_pb2.VOL_MODEL_SABR,)
    response = stub.CalibrateVolatilitySurface(request)

Version: 2.0.0
Author: Senior Quant Developer
"""

__version__ = "2.0.0"
__author__ = "NeuralTrade Quant Team"

# Proto file configuration
PROTO_FILE = "ai_service.proto"
PACKAGE_NAME = "neuraltrade.ai.v2"

# gRPC Service Configuration
GRPC_CONFIG = {
    "service_name": "neuraltrade.ai.v2.AIService",
    "default_port": 50051,
    "max_message_size": 100 * 1024 * 1024,  # 100MB
    "keepalive_time_ms": 30000,
    "keepalive_timeout_ms": 10000,
    "max_concurrent_streams": 1000,
    "initial_window_size": 1024 * 1024,  # 1MB
}

# Service Methods Registry
SERVICE_METHODS = {
    # Signal & Prediction
    "PredictSignal": {
        "request": "SignalRequest",
        "response": "SignalResponse",
        "streaming": False,
        "description": "Generate trading signal prediction",
    },
    "StreamSignals": {
        "request": "SignalStreamRequest",
        "response": "SignalResponse",
        "streaming": True,
        "description": "Stream real-time trading signals",
    },
    # Model Management
    "SelectModel": {
        "request": "ModelSelectionRequest",
        "response": "ModelSelectionResponse",
        "streaming": False,
        "description": "Select optimal model for market conditions",
    },
    "ListModels": {
        "request": "ListModelsRequest",
        "response": "ListModelsResponse",
        "streaming": False,
        "description": "List available ML models",
    },
    "GetModelMetrics": {
        "request": "ModelMetricsRequest",
        "response": "ModelMetricsResponse",
        "streaming": False,
        "description": "Get model performance metrics",
    },
    
    # Strategy Routing
    "RouteStrategy": {
        "request": "StrategyRequest",
        "response": "StrategyResponse",
        "streaming": False,
        "description": "Route signal to appropriate strategy",
    },
    "ListStrategies": {
        "request": "ListStrategiesRequest",
        "response": "ListStrategiesResponse",
        "streaming": False,
        "description": "List available trading strategies",
    },
    "BacktestStrategy": {
        "request": "BacktestRequest",
        "response": "BacktestResponse",
        "streaming": False,
        "description": "Backtest a trading strategy",
    },
    
    # Volatility Surface (NEW v2.0)
    "CalibrateVolatilitySurface": {
        "request": "VolSurfaceRequest",
        "response": "VolSurfaceResponse",
        "streaming": False,
        "description": "Calibrate volatility surface (SABR/SVI/Dupire)",
    },
    "GetImpliedVolatility": {
        "request": "ImpliedVolRequest",
        "response": "ImpliedVolResponse",
        "streaming": False,
        "description": "Get implied volatility for strike/expiry",
    },
    "GetLocalVolatility": {
        "request": "LocalVolRequest",
        "response": "LocalVolResponse",
        "streaming": False,
        "description": "Get Dupire local volatility surface",
    },
    "GetSkewMetrics": {
        "request": "SkewMetricsRequest",
        "response": "SkewMetricsResponse",
        "streaming": False,
        "description": "Get volatility skew metrics",
    },
    "CheckSurfaceArbitrage": {
        "request": "ArbitrageCheckRequest",
        "response": "ArbitrageCheckResponse",
        "streaming": False,
        "description": "Check for calendar/butterfly arbitrage",
    },
    
    # Options & Greeks (NEW v2.0)
    "CalculateGreeks": {
        "request": "GreeksRequest",
        "response": "GreeksResponse",
        "streaming": False,
        "description": "Calculate option Greeks (delta, gamma, etc.)",
    },
    "PriceOption": {
        "request": "OptionPricingRequest",
        "response": "OptionPricingResponse",
        "streaming": False,
        "description": "Price an option with Greeks",
    },
    "GetOptionsChain": {
        "request": "OptionsChainRequest",
        "response": "OptionsChainResponse",
        "streaming": False,
        "description": "Get full options chain with pricing",
    },
    
    # Risk Management (NEW v2.0)
    "CalculateRiskMetrics": {
        "request": "RiskMetricsRequest",
        "response": "RiskMetricsResponse",
        "streaming": False,
        "description": "Calculate comprehensive risk metrics",
    },
    "CalculateVaR": {
        "request": "VaRRequest",
        "response": "VaRResponse",
        "streaming": False,
        "description": "Calculate Value at Risk",
    },
    "CalculateCVaR": {
        "request": "CVaRRequest",
        "response": "CVaRResponse",
        "streaming": False,
        "description": "Calculate Conditional VaR (Expected Shortfall)",
    },
    "StressTest": {
        "request": "StressTestRequest",
        "response": "StressTestResponse",
        "streaming": False,
        "description": "Run portfolio stress tests",
    },
    
    # Portfolio Optimization (NEW v2.0)
    "OptimizePortfolio": {
        "request": "PortfolioOptRequest",
        "response": "PortfolioOptResponse",
        "streaming": False,
        "description": "Optimize portfolio weights",
    },
    "CalculateHRP": {
        "request": "HRPRequest",
        "response": "HRPResponse",
        "streaming": False,
        "description": "Hierarchical Risk Parity optimization",
    },
    "BlackLitterman": {
        "request": "BlackLittermanRequest",
        "response": "BlackLittermanResponse",
        "streaming": False,
        "description": "Black-Litterman portfolio optimization",
    },
    
    # Market Analysis
    "DetectRegime": {
        "request": "RegimeDetectionRequest",
        "response": "RegimeDetectionResponse",
        "streaming": False,
        "description": "Detect current market regime",
    },
    "AnalyzeSentiment": {
        "request": "SentimentRequest",
        "response": "SentimentResponse",
        "streaming": False,
        "description": "Analyze market sentiment",
    },
    "GetMarketMicrostructure": {
        "request": "MicrostructureRequest",
        "response": "MicrostructureResponse",
        "streaming": False,
        "description": "Get market microstructure metrics",
    },
    
    # Health & Monitoring
    "HealthCheck": {
        "request": "HealthRequest",
        "response": "HealthResponse",
        "streaming": False,
        "description": "Service health check",
    },
    "GetServiceMetrics": {
        "request": "MetricsRequest",
        "response": "MetricsResponse",
        "streaming": False,
        "description": "Get service performance metrics",
    },
}

# Enum Definitions for Reference
ENUMS = {
    "OptionType": {
        "OPTION_TYPE_UNSPECIFIED": 0,
        "OPTION_TYPE_CALL": 1,
        "OPTION_TYPE_PUT": 2,
    },
    "SignalAction": {
        "SIGNAL_ACTION_UNSPECIFIED": 0,
        "SIGNAL_ACTION_BUY": 1,
        "SIGNAL_ACTION_SELL": 2,
        "SIGNAL_ACTION_HOLD": 3,
        "SIGNAL_ACTION_CLOSE": 4,
    },
    "MarketRegime": {
        "MARKET_REGIME_UNSPECIFIED": 0,
        "MARKET_REGIME_TRENDING_UP": 1,
        "MARKET_REGIME_TRENDING_DOWN": 2,
        "MARKET_REGIME_RANGING": 3,
        "MARKET_REGIME_VOLATILE": 4,
        "MARKET_REGIME_CRISIS": 5,
    },
    "ModelType": {
        "MODEL_TYPE_UNSPECIFIED": 0,
        "MODEL_TYPE_LSTM": 1,
        "MODEL_TYPE_TRANSFORMER": 2,
        "MODEL_TYPE_DRL": 3,
        "MODEL_TYPE_ENSEMBLE": 4,
        "MODEL_TYPE_XGB": 5,
        "MODEL_TYPE_LIGHTGBM": 6,},
    "VolatilityModel": {
        "VOL_MODEL_UNSPECIFIED": 0,
        "VOL_MODEL_SABR": 1,
        "VOL_MODEL_SVI": 2,
        "VOL_MODEL_DUPIRE": 3,"VOL_MODEL_HESTON": 4,
    },
    "RiskProfile": {
        "RISK_PROFILE_UNSPECIFIED": 0,
        "RISK_PROFILE_CONSERVATIVE": 1,
        "RISK_PROFILE_MODERATE": 2,
        "RISK_PROFILE_AGGRESSIVE": 3,
    },
    "ExecutionType": {
        "EXECUTION_TYPE_UNSPECIFIED": 0,
        "EXECUTION_TYPE_AGGRESSIVE": 1,
        "EXECUTION_TYPE_NEUTRAL": 2,
        "EXECUTION_TYPE_CONSERVATIVE": 3,
        "EXECUTION_TYPE_TWAP": 4,
        "EXECUTION_TYPE_VWAP": 5,
    },
    "OptimizationMethod": {
        "OPT_METHOD_UNSPECIFIED": 0,
        "OPT_METHOD_MEAN_VARIANCE": 1,
        "OPT_METHOD_BLACK_LITTERMAN": 2,
        "OPT_METHOD_HRP": 3,
        "OPT_METHOD_RISK_PARITY": 4,"OPT_METHOD_MAX_SHARPE": 5,
        "OPT_METHOD_MIN_VARIANCE": 6,
        "OPT_METHOD_MAX_DIVERSIFICATION": 7,
    },
}


def get_method_info(method_name: str) -> dict:
    """Get method information by name."""
    return SERVICE_METHODS.get(method_name, {})


def list_streaming_methods() -> list:
    """List all streaming methods."""
    return [name for name, info in SERVICE_METHODS.items() if info.get("streaming")]


def list_unary_methods() -> list:
    """List all unary (non-streaming) methods."""
    return [name for name, info in SERVICE_METHODS.items() if not info.get("streaming")]


def get_enum_value(enum_name: str, value_name: str) -> int:
    """Get enum integer value by name."""
    enum = ENUMS.get(enum_name, {})
    return enum.get(value_name, 0)


def list_methods_by_category() -> dict:
    """Group methods by category."""
    categories = {
        "Signal & Prediction": ["PredictSignal", "StreamSignals"],
        "Model Management": ["SelectModel", "ListModels", "GetModelMetrics"],
        "Strategy Routing": ["RouteStrategy", "ListStrategies", "BacktestStrategy"],
        "Volatility Surface": ["CalibrateVolatilitySurface", "GetImpliedVolatility", "GetLocalVolatility", "GetSkewMetrics", "CheckSurfaceArbitrage"],
        "Options & Greeks": ["CalculateGreeks", "PriceOption", "GetOptionsChain"],
        "Risk Management": ["CalculateRiskMetrics", "CalculateVaR", "CalculateCVaR", "StressTest"],
        "Portfolio Optimization": ["OptimizePortfolio", "CalculateHRP", "BlackLitterman"],
        "Market Analysis": ["DetectRegime", "AnalyzeSentiment", "GetMarketMicrostructure"],
        "Health & Monitoring": ["HealthCheck", "GetServiceMetrics"],
    }
    return categories


__all__ = [
    "__version__",
    "__author__",
    "PROTO_FILE",
    "PACKAGE_NAME",
    "GRPC_CONFIG",
    "SERVICE_METHODS",
    "ENUMS",
    "get_method_info",
    "list_streaming_methods",
    "list_unary_methods",
    "get_enum_value",
    "list_methods_by_category",
]