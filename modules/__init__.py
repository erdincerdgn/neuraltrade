"""
NeuralTrade Modules
Author: Erdinc Erdogan
"""

__version__ = '1.0.0'
__author__ = 'Erdinc Erdogan'
import logging as _logging

_logger = _logging.getLogger(__name__)

# Import each module with graceful fallback
_modules = [
    'agents', 'alpha', 'causal', 'chaos', 'compliance', 'core', 'data',
    'execution', 'infrastructure', 'intelligence', 'ml', 'models',
    'monitors', 'options', 'portfolio', 'quant', 'risk', 'security',
    'signals', 'tests', 'legacy',
]

for _module in _modules:
    try:
        exec(f"from . import {_module}")
    except ImportError as e:
        _logger.debug(f"Module {_module} not available: {e}")

__all__ = [
    'agents', 'alpha', 'causal', 'chaos', 'compliance', 'core', 'data',
    'execution', 'infrastructure', 'intelligence', 'ml', 'models',
    'monitors', 'options', 'portfolio', 'quant', 'risk', 'security',
    'signals', 'tests', 'legacy',
]