"""
NeuralTrade Protocol Buffer Definitions
=======================================
gRPC service definitions for NeuralTrade AI Engine communication.

Package Structure:
    proto/
    ├── __init__.py (this file)
    ├── ai_service.proto
    └── generated/
        ├── __init__.py
        ├── ai_service_pb2.py
        └── ai_service_pb2_grpc.py

Usage:
    from proto.generated import ai_service_pb2, ai_service_pb2_grpc

Version: 2.0.0
Author: Senior Quant Developer
"""

__version__ = "2.0.0"
__author__ = "NeuralTrade Quant Team"

# Proto file path for reference
PROTO_FILE = "ai_service.proto"

# Service configuration
SERVICE_NAME = "neuraltrade.ai.v2.AIService"
PACKAGE_NAME = "neuraltrade.ai.v2"

# gRPC default settings
DEFAULT_GRPC_PORT = 50051
DEFAULT_MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB
DEFAULT_KEEPALIVE_TIME_MS = 30000
DEFAULT_KEEPALIVE_TIMEOUT_MS = 10000

__all__ = [
    "__version__",
    "__author__",
    "PROTO_FILE",
    "SERVICE_NAME",
    "PACKAGE_NAME",
    "DEFAULT_GRPC_PORT",
    "DEFAULT_MAX_MESSAGE_SIZE",
    "DEFAULT_KEEPALIVE_TIME_MS",
    "DEFAULT_KEEPALIVE_TIMEOUT_MS",
]