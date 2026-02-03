"""
NeuralTrade Infrastructure Module
Author: Erdinc Erdogan
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

# Core modules - should always work
try:
    from .accelerator import *
except ImportError as e:
    _logger.debug(f"Accelerator not available: {e}")

try:
    from .audit import *
except ImportError as e:
    _logger.debug(f"Audit not available: {e}")

try:
    from .cloud_connector import *
except ImportError as e:
    _logger.debug(f"CloudConnector not available: {e}")

try:
    from .cloud_orchestrator import *
except ImportError as e:
    _logger.debug(f"CloudOrchestrator not available: {e}")

try:
    from .distributed_chaos import *
except ImportError as e:
    _logger.debug(f"DistributedChaos not available: {e}")

try:
    from .docker_monitor import *
except ImportError as e:
    _logger.debug(f"DockerMonitor not available: {e}")

try:
    from .fast_serializer import *
except ImportError as e:
    _logger.debug(f"FastSerializer not available: {e}")

try:
    from .gpu_job_manager import *
except ImportError as e:
    _logger.debug(f"GPUJobManager not available: {e}")

try:
    from .latency import *
except ImportError as e:
    _logger.debug(f"Latency not available: {e}")

try:
    from .metrics import *
except ImportError as e:
    _logger.debug(f"Metrics not available: {e}")

# Optional: MLflow (requires mlflow package)
try:
    from .mlflow_config import *
except ImportError as e:
    _logger.debug(f"MLflowConfig not available: {e}")

try:
    from .multicloud import *
except ImportError as e:
    _logger.debug(f"MultiCloud not available: {e}")

try:
    from .ptp import *
except ImportError as e:
    _logger.debug(f"PTP not available: {e}")

try:
    from .bridge import *
except ImportError as e:
    _logger.debug(f"Bridge not available: {e}")

try:
    from .resilient_connection import *
except ImportError as e:
    _logger.debug(f"ResilientConnection not available: {e}")

try:
    from .rust_engine import *
except ImportError as e:
    _logger.debug(f"RustEngine not available: {e}")

try:
    from .secure_credentials import *
except ImportError as e:
    _logger.debug(f"SecureCredentials not available: {e}")

# Optional: SignalPublisher (requires redis package)
try:
    from .signal_publisher import *
except ImportError as e:
    _logger.debug(f"SignalPublisher not available: {e}")

try:
    from .stash_manager import *
except ImportError as e:
    _logger.debug(f"StashManager not available: {e}")

__all__ = [
    'Accelerator', 'InfrastructureAudit', 'CloudConnector',
    'CloudOrchestrator', 'MultiCloud', 'DistributedChaos', 'DockerMonitor',
    'FastSerializer', 'GPUJobManager', 'LatencyMonitor', 'MetricsCollector',
    'MLflowConfig', 'PTP', 'Bridge', 'ResilientConnection',
    'RustEngine', 'SecureCredentials', 'SignalPublisher', 'StashManager',
]