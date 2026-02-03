"""
NeuralTrade Signals Module
Author: Erdinc Erdogan
"""

from .signal_generator import (
    SignalGenerator,
    SignalType,
    SignalStrength,
    TrendDirection,
    Signal,
    SignalBundle,
    TechnicalIndicators,
    StatisticalSignals,
    MomentumSignals,
    VolatilitySignals,
)

__all__ = [
    'SignalGenerator', 'SignalType', 'SignalStrength', 'TrendDirection',
    'Signal', 'SignalBundle', 'TechnicalIndicators', 'StatisticalSignals',
    'MomentumSignals', 'VolatilitySignals',
]