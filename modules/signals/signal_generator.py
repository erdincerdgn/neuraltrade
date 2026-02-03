"""
Comprehensive Signal Generator Engine
Author: Erdinc Erdogan
Purpose: Generates trading signals from 10+ technical indicators (SMA, EMA, RSI, MACD, Bollinger, Stochastic, ATR, ADX, OBV, Ichimoku) and statistical features.
References:
- Technical Analysis Indicators
- Mean Reversion Signals (Z-Score, Half-Life)
- Momentum and Trend-Following Signals
- Hurst Exponent for Mean Reversion Detection
Usage:
    generator = SignalGenerator(prices)
    bundle = generator.generate_all_signals()
    recommendation = bundle.recommendation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class SignalType(Enum):
    """Signal type classification"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    STATISTICAL = "statistical"
    COMPOSITE = "composite"


class SignalStrength(Enum):
    """Signal strength levels"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class TrendDirection(Enum):
    """Trend direction"""
    STRONG_UPTREND = 2
    UPTREND = 1
    SIDEWAYS = 0
    DOWNTREND = -1
    STRONG_DOWNTREND = -2


@dataclass
class Signal:
    """Individual trading signal"""
    name: str
    value: float
    signal_type: str
    strength: int
    timestamp: Optional[float] = None
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SignalBundle:
    """Collection of signals for a single asset"""
    asset_name: str
    signals: List[Signal]
    composite_score: float
    recommendation: str
    trend_direction: int
    volatility_regime: str
    timestamp: Optional[float] = None


@dataclass
class TechnicalIndicators:
    """Complete set of technical indicators"""
    sma: Dict[int, np.ndarray]
    ema: Dict[int, np.ndarray]
    rsi: np.ndarray
    macd: Dict[str, np.ndarray]
    bollinger: Dict[str, np.ndarray]
    stochastic: Dict[str, np.ndarray]
    atr: np.ndarray
    adx: np.ndarray
    obv: np.ndarray


@dataclass
class StatisticalSignals:
    """Statistical trading signals"""
    z_score: np.ndarray
    hurst_exponent: float
    half_life: float
    mean_reversion_speed: float
    trend_strength: float
    regime: str


@dataclass
class MomentumSignals:
    """Momentum-based signals"""
    roc: Dict[int, np.ndarray]
    momentum: np.ndarray
    tsi: np.ndarray
    williams_r: np.ndarray
    cci: np.ndarray
    ultimate_oscillator: np.ndarray


@dataclass
class VolatilitySignals:
    """Volatility-based signals"""
    atr: np.ndarray
    atr_percent: np.ndarray
    bollinger_width: np.ndarray
    keltner_width: np.ndarray
    volatility_regime: str
    volatility_percentile: float


class SignalGenerator(BaseModule):
    """
    Institutional-Grade Signal Generator.
    
    Implements comprehensive technical analysis and statistical signals
    for systematic trading strategies.
    
    Mathematical Framework:
    ----------------------
    
    Exponential Moving Average (EMA):
        EMA_t = α × P_t + (1-α) × EMA_{t-1}
        where α = 2/(n+1)
    
    Relative Strength Index (RSI):
        RSI = 100 - 100/(1 + RS)
        RS = Average Gain / Average Loss
    
    MACD:
        MACD Line = EMA_12 - EMA_26
        Signal Line = EMA_9(MACD Line)
        Histogram = MACD Line - Signal Line
    
    Bollinger Bands:
        Middle = SMA_n
        Upper = SMA_n + k × σ_n
        Lower = SMA_n - k × σ_n%B = (Price - Lower) / (Upper - Lower)
    
    Stochastic Oscillator:
        %K = (Close - Low_n) / (High_n - Low_n) × 100
        %D = SMA_3(%K)
    
    Average True Range (ATR):
        TR = max(High-Low, |High-Close_{t-1}|, |Low-Close_{t-1}|)
        ATR = EMA_n(TR)
    
    Average Directional Index (ADX):
        +DM = High_t - High_{t-1} (if positive and > -DM)
        -DM = Low_{t-1} - Low_t (if positive and > +DM)
        +DI = 100 × EMA(+DM) / ATR
        -DI = 100 × EMA(-DM) / ATR
        DX = |+DI - -DI| / (+DI + -DI) × 100
        ADX = EMA_n(DX)
    
    Z-Score (Mean Reversion):
        Z = (P_t - μ) / σ
        where μ = rolling mean, σ = rolling std
    
    Hurst Exponent:
        E[R/S] ~ c × n^H
        H< 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
    
    Ornstein-Uhlenbeck Half-Life:
        dX = θ(μ - X)dt + σdW
        Half-life = ln(2) / θ
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.rsi_period: int = self.config.get('rsi_period', 14)
        self.macd_fast: int = self.config.get('macd_fast', 12)
        self.macd_slow: int = self.config.get('macd_slow', 26)
        self.macd_signal: int = self.config.get('macd_signal', 9)
        self.bb_period: int = self.config.get('bb_period', 20)
        self.bb_std: float = self.config.get('bb_std', 2.0)
        self.atr_period: int = self.config.get('atr_period', 14)
        self.adx_period: int = self.config.get('adx_period', 14)
    
    #=========================================================================
    # MOVING AVERAGES
    # =========================================================================
    
    def compute_sma(
        self,
        prices: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Simple Moving Average.
        
        SMA_t = (1/n) × Σ P_{t-i} for i=0 to n-1
        """
        prices = np.asarray(prices)
        sma = np.full(len(prices), np.nan)
        
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma
    
    def compute_ema(
        self,
        prices: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Exponential Moving Average.
        
        EMA_t = α × P_t + (1-α) × EMA_{t-1}
        α = 2 / (n + 1)
        """
        prices = np.asarray(prices)
        alpha = 2.0 / (period + 1)
        
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def compute_wma(
        self,
        prices: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Weighted Moving Average.
        
        WMA_t = Σ(w_i × P_{t-i}) / Σw_i
        where w_i = n - i
        """
        prices = np.asarray(prices)
        weights = np.arange(1, period + 1)
        
        wma = np.full(len(prices), np.nan)
        
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            wma[i] = np.sum(weights * window) / np.sum(weights)
        
        return wma
    
    def compute_dema(
        self,
        prices: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Double Exponential Moving Average.
        
        DEMA = 2 × EMA - EMA(EMA)
        """
        ema1 = self.compute_ema(prices, period)
        ema2 = self.compute_ema(ema1[~np.isnan(ema1)], period)
        
        dema = np.full(len(prices), np.nan)
        start_idx = 2 * (period - 1)
        
        if start_idx < len(prices):
            valid_len = min(len(ema2), len(prices) - start_idx)
            dema[start_idx:start_idx + valid_len] = 2 * ema1[start_idx:start_idx + valid_len] - ema2[:valid_len]
        
        return dema
    
    def compute_tema(
        self,
        prices: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Triple Exponential Moving Average.
        
        TEMA = 3×EMA - 3×EMA(EMA) + EMA(EMA(EMA))
        """
        ema1 = self.compute_ema(prices, period)
        ema2 = self.compute_ema(ema1[~np.isnan(ema1)], period)
        ema3 = self.compute_ema(ema2[~np.isnan(ema2)], period)
        
        tema = np.full(len(prices), np.nan)
        start_idx = 3 * (period - 1)
        
        if start_idx < len(prices) and len(ema3) > 0:
            valid_len = min(len(ema3), len(prices) - start_idx)
            tema[start_idx:start_idx + valid_len] = (
                3 * ema1[start_idx:start_idx + valid_len] -
                3 * ema2[start_idx - period + 1:start_idx - period + 1 + valid_len] +
                ema3[:valid_len]
            )
        
        return tema
    
    # =========================================================================
    # RSI (RELATIVE STRENGTH INDEX)
    # =========================================================================
    
    def compute_rsi(
        self,
        prices: np.ndarray,
        period: Optional[int] = None
    ) -> np.ndarray:
        """
        Relative Strength Index.
        
        RSI = 100 - 100/(1 + RS)
        RS = Average Gain / Average Loss
        """
        prices = np.asarray(prices)
        period = period or self.rsi_period
        
        # Price changes
        delta = np.diff(prices)
        
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        rsi = np.full(len(prices), np.nan)
        
        # Initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - 100 / (1 + rs)
        
        # Smoothed averages
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi
    
    def compute_stochastic_rsi(
        self,
        prices: np.ndarray,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Stochastic RSI.
        
        StochRSI = (RSI - RSI_Low) / (RSI_High - RSI_Low)
        """
        rsi = self.compute_rsi(prices, rsi_period)
        
        stoch_rsi = np.full(len(prices), np.nan)
        
        for i in range(rsi_period + stoch_period - 1, len(prices)):
            rsi_window = rsi[i - stoch_period + 1:i + 1]
            rsi_high = np.nanmax(rsi_window)
            rsi_low = np.nanmin(rsi_window)
            
            if rsi_high - rsi_low > 0:
                stoch_rsi[i] = (rsi[i] - rsi_low) / (rsi_high - rsi_low) * 100
            else:
                stoch_rsi[i] = 50
        
        # Smooth %K and %D
        k = self.compute_sma(stoch_rsi, k_smooth)
        d = self.compute_sma(k, d_smooth)
        
        return {'stoch_rsi': stoch_rsi, 'k': k, 'd': d}
    
    # =========================================================================
    # MACD
    # =========================================================================
    
    def compute_macd(
        self,
        prices: np.ndarray,
        fast_period: Optional[int] = None,
        slow_period: Optional[int] = None,
        signal_period: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Moving Average Convergence Divergence.
        
        MACD Line = EMA_fast - EMA_slow
        Signal Line = EMA(MACD Line)
        Histogram = MACD Line - Signal Line
        """
        prices = np.asarray(prices)
        fast = fast_period or self.macd_fast
        slow = slow_period or self.macd_slow
        signal = signal_period or self.macd_signal
        ema_fast = self.compute_ema(prices, fast)
        ema_slow = self.compute_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        # Signal line
        valid_macd = macd_line[~np.isnan(macd_line)]
        signal_line_valid = self.compute_ema(valid_macd, signal)
        
        signal_line = np.full(len(prices), np.nan)
        start_idx = slow -1 + signal - 1
        if start_idx < len(prices):
            valid_len = min(len(signal_line_valid), len(prices) - start_idx)
            signal_line[start_idx:start_idx + valid_len] = signal_line_valid[:valid_len]
        
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    # =========================================================================
    # BOLLINGER BANDS
    # =========================================================================
    
    def compute_bollinger_bands(
        self,
        prices: np.ndarray,
        period: Optional[int] = None,
        num_std: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Bollinger Bands.
        
        Middle = SMA_n
        Upper = SMA_n + k × σ_n
        Lower = SMA_n - k × σ_n
        %B = (Price - Lower) / (Upper - Lower)
        Bandwidth = (Upper - Lower) / Middle
        """
        prices = np.asarray(prices)
        period = period or self.bb_period
        num_std = num_std or self.bb_std
        
        middle = self.compute_sma(prices, period)
        
        std = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1], ddof=1)
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        # %B
        percent_b = (prices - lower) / (upper - lower)
        
        # Bandwidth
        bandwidth = (upper - lower) / middle
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'percent_b': percent_b,
            'bandwidth': bandwidth,
            'std': std
        }
    
    # =========================================================================
    # STOCHASTIC OSCILLATOR
    # =========================================================================
    
    def compute_stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Stochastic Oscillator.
        
        %K = (Close - Low_n) / (High_n - Low_n) × 100
        %D = SMA_3(%K)
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        
        k = np.full(len(close), np.nan)
        
        for i in range(k_period - 1, len(close)):
            highest_high = np.max(high[i - k_period + 1:i + 1])
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            
            if highest_high - lowest_low > 0:
                k[i] = (close[i] - lowest_low) / (highest_high - lowest_low) * 100
            else:
                k[i] = 50
        
        d = self.compute_sma(k, d_period)
        
        return {'k': k, 'd': d}
    
    # =========================================================================
    # ATR (AVERAGE TRUE RANGE)
    # =========================================================================
    
    def compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: Optional[int] = None
    ) -> np.ndarray:
        """
        Average True Range.
        
        TR = max(High-Low, |High-Close_{t-1}|, |Low-Close_{t-1}|)
        ATR = EMA_n(TR)
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        period = period or self.atr_period
        
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        
        atr = self.compute_ema(tr, period)
        
        return atr
    
    def compute_atr_percent(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: Optional[int] = None
    ) -> np.ndarray:
        """ATR as percentage of price."""
        atr = self.compute_atr(high, low, close, period)
        return atr / close * 100
    
    # =========================================================================
    # ADX (AVERAGE DIRECTIONAL INDEX)
    # =========================================================================
    
    def compute_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Average Directional Index.
        
        +DM = High_t - High_{t-1} (if positive and > -DM)
        -DM = Low_{t-1} - Low_t (if positive and > +DM)
        +DI = 100 × EMA(+DM) / ATR
        -DI = 100 × EMA(-DM) / ATR
        DX = |+DI - -DI| / (+DI + -DI) × 100
        ADX = EMA_n(DX)
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        period = period or self.adx_period
        
        n = len(close)
        
        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # ATR
        atr = self.compute_atr(high, low, close, period)
        
        # Smoothed DM
        plus_dm_smooth = self.compute_ema(plus_dm, period)
        minus_dm_smooth = self.compute_ema(minus_dm, period)
        
        # Directional Indicators
        plus_di = np.where(atr > 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr > 0, 100 * minus_dm_smooth / atr, 0)
        
        # DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0)
        
        # ADX
        adx = self.compute_ema(dx, period)
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'dx': dx
        }
    
    # =========================================================================
    # OBV (ON-BALANCE VOLUME)
    # =========================================================================
    
    def compute_obv(
        self,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        On-Balance Volume.
        
        OBV_t = OBV_{t-1} + Volume_t (if Close_t > Close_{t-1})
        OBV_t = OBV_{t-1} - Volume_t (if Close_t < Close_{t-1})
        OBV_t = OBV_{t-1} (if Close_t = Close_{t-1})
        """
        close = np.asarray(close)
        volume = np.asarray(volume)
        
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        
        return obv
    
    def compute_obv_signal(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        signal_period: int = 20
    ) -> Dict[str, np.ndarray]:
        """OBV with signal line."""
        obv = self.compute_obv(close, volume)
        obv_signal = self.compute_ema(obv, signal_period)
        obv_histogram = obv - obv_signal
        
        return {
            'obv': obv,
            'signal': obv_signal,
            'histogram': obv_histogram
        }
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    def compute_roc(
        self,
        prices: np.ndarray,
        period: int = 10
    ) -> np.ndarray:
        """
        Rate of Change.
        
        ROC = (Price_t - Price_{t-n}) / Price_{t-n} × 100
        """
        prices = np.asarray(prices)
        roc = np.full(len(prices), np.nan)
        
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                roc[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100
        
        return roc
    
    def compute_momentum(
        self,
        prices: np.ndarray,
        period: int = 10
    ) -> np.ndarray:
        """
        Momentum.
        
        Momentum = Price_t - Price_{t-n}
        """
        prices = np.asarray(prices)
        momentum = np.full(len(prices), np.nan)
        
        for i in range(period, len(prices)):
            momentum[i] = prices[i] - prices[i - period]
        
        return momentum
    
    def compute_williams_r(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """
        Williams %R.
        
        %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        
        williams_r = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            highest = np.max(high[i - period + 1:i + 1])
            lowest = np.min(low[i - period + 1:i + 1])
            
            if highest - lowest > 0:
                williams_r[i] = (highest - close[i]) / (highest - lowest) * -100
            else:
                williams_r[i] = -50
        
        return williams_r
    
    def compute_cci(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """
        Commodity Channel Index.
        
        TP = (High + Low + Close) / 3
        CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation)
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        
        tp = (high + low + close) / 3
        tp_sma = self.compute_sma(tp, period)
        cci = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            tp_window = tp[i - period + 1:i + 1]
            mean_dev = np.mean(np.abs(tp_window - tp_sma[i]))
            
            if mean_dev > 0:
                cci[i] = (tp[i] - tp_sma[i]) / (0.015 * mean_dev)
            else:
                cci[i] = 0
        
        return cci
    
    def compute_ultimate_oscillator(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> np.ndarray:
        """
        Ultimate Oscillator.
        
        BP = Close - min(Low, Close_{t-1})
        TR = max(High, Close_{t-1}) - min(Low, Close_{t-1})
        UO = 100 × [(4×Avg1 +2×Avg2 + Avg3) / 7]
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        
        n = len(close)
        bp = np.zeros(n)
        tr = np.zeros(n)
        
        for i in range(1, n):
            bp[i] = close[i] - min(low[i], close[i - 1])
            tr[i] = max(high[i], close[i - 1]) - min(low[i], close[i - 1])
        
        uo = np.full(n, np.nan)
        
        for i in range(period3, n):
            avg1 = np.sum(bp[i - period1 + 1:i + 1]) / np.sum(tr[i - period1 + 1:i + 1]) if np.sum(tr[i - period1 + 1:i + 1]) > 0 else 0
            avg2 = np.sum(bp[i - period2 + 1:i + 1]) / np.sum(tr[i - period2 + 1:i + 1]) if np.sum(tr[i - period2 + 1:i + 1]) > 0 else 0
            avg3 = np.sum(bp[i - period3 + 1:i + 1]) / np.sum(tr[i - period3 + 1:i + 1]) if np.sum(tr[i - period3 + 1:i + 1]) > 0 else 0
            uo[i] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        
        return uo
    
    # =========================================================================
    # STATISTICAL SIGNALS
    # =========================================================================
    
    def compute_z_score(
        self,
        prices: np.ndarray,
        lookback: int = 20
    ) -> np.ndarray:
        """
        Z-Score for mean reversion.
        
        Z = (P_t - μ) / σ
        """
        prices = np.asarray(prices)
        z_score = np.full(len(prices), np.nan)
        
        for i in range(lookback - 1, len(prices)):
            window = prices[i - lookback + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            
            if std > 0:
                z_score[i] = (prices[i] - mean) / std
            else:
                z_score[i] = 0
        
        return z_score
    
    def compute_hurst_exponent(
        self,
        prices: np.ndarray,
        max_lag: int = 100
    ) -> float:
        """
        Hurst Exponent using R/S analysis.
        
        H< 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        prices = np.asarray(prices)
        returns = np.diff(np.log(prices))
        
        lags = range(2, min(max_lag, len(returns) // 2))
        rs_values = []
        
        for lag in lags:
            rs_list = []
            for start in range(0, len(returns) - lag, lag):
                segment = returns[start:start + lag]
                mean_seg = np.mean(segment)
                cumdev = np.cumsum(segment - mean_seg)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(segment, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            
            if rs_list:
                rs_values.append((lag, np.mean(rs_list)))
        
        if len(rs_values) < 2:
            return 0.5
        
        log_lags = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
        
        return float(np.clip(slope, 0, 1))
    
    def compute_half_life(
        self,
        prices: np.ndarray
    ) -> float:
        """
        Half-life of mean reversion (Ornstein-Uhlenbeck).
        
        dX = θ(μ - X)dt + σdW
        Half-life = ln(2) / θ
        """
        prices = np.asarray(prices)
        log_prices = np.log(prices)
        
        # Lag regression
        y = np.diff(log_prices)
        x = log_prices[:-1] - np.mean(log_prices[:-1])
        
        # OLS: y = α + β*x + ε
        x_with_const = np.column_stack([np.ones(len(x)), x])
        
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            theta = -beta[1]
            
            if theta > 0:
                half_life = np.log(2) / theta
                return float(min(half_life, 252))  # Cap at 1 year
            else:
                return float('inf')
        except:
            return float('inf')
    
    def compute_statistical_signals(
        self,
        prices: np.ndarray,
        lookback: int = 60
    ) -> StatisticalSignals:
        """Compute comprehensive statistical signals."""
        z_score = self.compute_z_score(prices, lookback)
        hurst = self.compute_hurst_exponent(prices)
        half_life = self.compute_half_life(prices)
        
        # Mean reversion speed
        if half_life < float('inf') and half_life > 0:
            mr_speed = np.log(2) / half_life
        else:
            mr_speed = 0
        
        # Trend strength based on Hurst
        trend_strength = abs(hurst - 0.5) * 2
        
        # Regime classification
        if hurst < 0.4:
            regime = "strong_mean_reversion"
        elif hurst < 0.5:
            regime = "mean_reversion"
        elif hurst < 0.6:
            regime = "random_walk"
        elif hurst < 0.7:
            regime = "trending"
        else:
            regime = "strong_trending"
        
        return StatisticalSignals(
            z_score=z_score,
            hurst_exponent=float(hurst),
            half_life=float(half_life),
            mean_reversion_speed=float(mr_speed),
            trend_strength=float(trend_strength),
            regime=regime
        )
    
    # =========================================================================
    # KELTNER CHANNELS
    # =========================================================================
    
    def compute_keltner_channels(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Keltner Channels.
        
        Middle = EMA(Close)
        Upper = EMA + multiplier × ATR
        Lower = EMA - multiplier × ATR
        """
        middle = self.compute_ema(close, ema_period)
        atr = self.compute_atr(high, low, close, atr_period)
        
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': (upper - lower) / middle
        }
    
    # =========================================================================
    # ICHIMOKU CLOUD
    # =========================================================================
    
    def compute_ichimoku(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52
    ) -> Dict[str, np.ndarray]:
        """
        Ichimoku Cloud.
        
        Tenkan-sen = (Highest High + Lowest Low) / 2 over 9 periods
        Kijun-sen = (Highest High + Lowest Low) / 2 over 26 periods
        Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
        Senkou Span B = (Highest High + Lowest Low) / 2 over 52 periods, shifted 26 periods ahead
        Chikou Span = Close, shifted 26 periods back
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        n = len(close)
        
        def donchian_mid(h, l, period):
            result = np.full(len(h), np.nan)
            for i in range(period - 1, len(h)):
                result[i] = (np.max(h[i - period + 1:i + 1]) + np.min(l[i - period + 1:i + 1])) / 2
            return result
        
        tenkan = donchian_mid(high, low, tenkan_period)
        kijun = donchian_mid(high, low, kijun_period)
        # Senkou Span A (shifted forward)
        senkou_a = np.full(n + kijun_period, np.nan)
        senkou_a_base = (tenkan + kijun) / 2
        senkou_a[kijun_period:n + kijun_period] = senkou_a_base
        senkou_a = senkou_a[:n]
        
        # Senkou Span B (shifted forward)
        senkou_b_base = donchian_mid(high, low, senkou_b_period)
        senkou_b = np.full(n + kijun_period, np.nan)
        senkou_b[kijun_period:n + kijun_period] = senkou_b_base
        senkou_b = senkou_b[:n]
        
        # Chikou Span (shifted back)
        chikou = np.full(n, np.nan)
        chikou[:n - kijun_period] = close[kijun_period:]
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def generate_ma_crossover_signal(
        self,
        prices: np.ndarray,
        fast_period: int = 10,
        slow_period: int = 30
    ) -> Signal:
        """Generate MA crossover signal."""
        fast_ma = self.compute_ema(prices, fast_period)
        slow_ma = self.compute_ema(prices, slow_period)
        
        current_fast = fast_ma[-1]
        current_slow = slow_ma[-1]
        prev_fast = fast_ma[-2] if len(fast_ma) > 1 else current_fast
        prev_slow = slow_ma[-2] if len(slow_ma) > 1 else current_slow
        
        # Crossover detection
        if prev_fast <= prev_slow and current_fast > current_slow:
            strength = SignalStrength.BUY.value
            value = 1.0
        elif prev_fast >= prev_slow and current_fast < current_slow:
            strength = SignalStrength.SELL.value
            value = -1.0
        else:
            diff_pct = (current_fast - current_slow) / current_slow * 100
            value = np.clip(diff_pct / 2, -1, 1)
            if value > 0.5:
                strength = SignalStrength.BUY.value
            elif value < -0.5:
                strength = SignalStrength.SELL.value
            else:
                strength = SignalStrength.NEUTRAL.value
        
        return Signal(
            name="MA_Crossover",
            value=float(value),
            signal_type=SignalType.TREND_FOLLOWING.value,
            strength=strength,
            confidence=abs(value),
            metadata={'fast_period': fast_period, 'slow_period': slow_period}
        )
    
    def generate_rsi_signal(
        self,
        prices: np.ndarray,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30
    ) -> Signal:
        """Generate RSI signal."""
        rsi = self.compute_rsi(prices, period)
        current_rsi = rsi[-1]
        
        if np.isnan(current_rsi):
            return Signal(
                name="RSI",
                value=0.0,
                signal_type=SignalType.MEAN_REVERSION.value,
                strength=SignalStrength.NEUTRAL.value,
                confidence=0.0
            )
        
        # Normalize to -1 to 1
        if current_rsi >= overbought:
            value = -((current_rsi - overbought) / (100 - overbought))
            strength = SignalStrength.SELL.value if current_rsi > 80 else SignalStrength.SELL.value
        elif current_rsi <= oversold:
            value = (oversold - current_rsi) / oversold
            strength = SignalStrength.BUY.value if current_rsi < 20 else SignalStrength.BUY.value
        else:
            value = (50 - current_rsi) / 50 * 0.5
            strength = SignalStrength.NEUTRAL.value
        
        return Signal(
            name="RSI",
            value=float(value),
            signal_type=SignalType.MEAN_REVERSION.value,
            strength=strength,
            confidence=abs(value),
            metadata={'rsi': float(current_rsi), 'period': period}
        )
    
    def generate_macd_signal(
        self,
        prices: np.ndarray
    ) -> Signal:
        """Generate MACD signal."""
        macd_data = self.compute_macd(prices)
        
        macd_line = macd_data['macd'][-1]
        signal_line = macd_data['signal'][-1]
        histogram = macd_data['histogram'][-1]
        
        if np.isnan(macd_line) or np.isnan(signal_line):
            return Signal(
                name="MACD",
                value=0.0,
                signal_type=SignalType.MOMENTUM.value,
                strength=SignalStrength.NEUTRAL.value,
                confidence=0.0
            )
        
        prev_histogram = macd_data['histogram'][-2] if len(macd_data['histogram']) > 1 else histogram
        # Signal based on histogram and crossover
        if prev_histogram < 0 and histogram > 0:
            value = 1.0
            strength = SignalStrength.BUY.value
        elif prev_histogram > 0 and histogram < 0:
            value = -1.0
            strength = SignalStrength.SELL.value
        else:
            # Normalize histogram
            hist_std = np.nanstd(macd_data['histogram'])
            if hist_std > 0:
                value = np.clip(histogram / (2 * hist_std), -1, 1)
            else:
                value = 0
            if value > 0.3:
                strength = SignalStrength.BUY.value
            elif value < -0.3:
                strength = SignalStrength.SELL.value
            else:
                strength = SignalStrength.NEUTRAL.value
        
        return Signal(
            name="MACD",
            value=float(value),
            signal_type=SignalType.MOMENTUM.value,
            strength=strength,
            confidence=abs(value),
            metadata={'macd': float(macd_line), 'signal': float(signal_line), 'histogram': float(histogram)}
        )
    
    def generate_bollinger_signal(
        self,
        prices: np.ndarray
    ) -> Signal:
        """Generate Bollinger Bands signal."""
        bb = self.compute_bollinger_bands(prices)
        
        percent_b = bb['percent_b'][-1]
        bandwidth = bb['bandwidth'][-1]
        
        if np.isnan(percent_b):
            return Signal(
                name="Bollinger",
                value=0.0,
                signal_type=SignalType.MEAN_REVERSION.value,
                strength=SignalStrength.NEUTRAL.value,
                confidence=0.0
            )
        
        # Mean reversion signal
        if percent_b > 1.0:
            value = -(percent_b - 1.0)
            strength = SignalStrength.SELL.value
        elif percent_b < 0.0:
            value = -percent_b
            strength = SignalStrength.BUY.value
        else:
            value = (0.5 - percent_b) * 2
            if abs(value) > 0.5:
                strength = SignalStrength.BUY.value if value > 0 else SignalStrength.SELL.value
            else:
                strength = SignalStrength.NEUTRAL.value
        
        value = np.clip(value, -1, 1)
        
        return Signal(
            name="Bollinger",
            value=float(value),
            signal_type=SignalType.MEAN_REVERSION.value,
            strength=strength,
            confidence=abs(value),
            metadata={'percent_b': float(percent_b), 'bandwidth': float(bandwidth)}
        )
    
    def generate_composite_signal(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> SignalBundle:
        """Generate composite signal from multiple indicators."""
        if high is None:
            high = prices
        if low is None:
            low = prices
        
        # Default weights
        if weights is None:
            weights = {
                'ma_crossover': 0.20,
                'rsi': 0.20,
                'macd': 0.20,
                'bollinger': 0.15,
                'stochastic': 0.15,
                'adx': 0.10
            }
        
        signals = []
        
        # MA Crossover
        ma_signal = self.generate_ma_crossover_signal(prices)
        signals.append(ma_signal)
        
        # RSI
        rsi_signal = self.generate_rsi_signal(prices)
        signals.append(rsi_signal)
        
        # MACD
        macd_signal = self.generate_macd_signal(prices)
        signals.append(macd_signal)
        
        # Bollinger
        bb_signal = self.generate_bollinger_signal(prices)
        signals.append(bb_signal)
        
        # Stochastic
        stoch = self.compute_stochastic(high, low, prices)
        stoch_k = stoch['k'][-1]
        if not np.isnan(stoch_k):
            if stoch_k > 80:
                stoch_value = -(stoch_k - 80) / 20
                stoch_strength = SignalStrength.SELL.value
            elif stoch_k < 20:
                stoch_value = (20 - stoch_k) / 20
                stoch_strength = SignalStrength.BUY.value
            else:
                stoch_value = (50 - stoch_k) / 50
                stoch_strength = SignalStrength.NEUTRAL.value
        else:
            stoch_value = 0
            stoch_strength = SignalStrength.NEUTRAL.value
        
        stoch_signal = Signal(
            name="Stochastic",
            value=float(stoch_value),
            signal_type=SignalType.MOMENTUM.value,
            strength=stoch_strength,
            confidence=abs(stoch_value)
        )
        signals.append(stoch_signal)
        
        # ADX for trend strength
        adx_data = self.compute_adx(high, low, prices)
        adx_value = adx_data['adx'][-1]
        plus_di = adx_data['plus_di'][-1]
        minus_di = adx_data['minus_di'][-1]
        
        if not np.isnan(adx_value):
            if adx_value > 25:
                if plus_di > minus_di:
                    adx_signal_value = adx_value / 100
                    adx_strength = SignalStrength.BUY.value
                else:
                    adx_signal_value = -adx_value / 100
                    adx_strength = SignalStrength.SELL.value
            else:
                adx_signal_value = 0
                adx_strength = SignalStrength.NEUTRAL.value
        else:
            adx_signal_value = 0
            adx_strength = SignalStrength.NEUTRAL.value
        
        adx_signal = Signal(
            name="ADX",
            value=float(adx_signal_value),
            signal_type=SignalType.TREND_FOLLOWING.value,
            strength=adx_strength,
            confidence=abs(adx_signal_value),
            metadata={'adx': float(adx_value) if not np.isnan(adx_value) else 0}
        )
        signals.append(adx_signal)
        
        # Composite score
        signal_map = {s.name: s for s in signals}
        composite_score = 0
        total_weight = 0
        
        weight_map = {
            'MA_Crossover': weights.get('ma_crossover', 0.2),
            'RSI': weights.get('rsi', 0.2),
            'MACD': weights.get('macd', 0.2),
            'Bollinger': weights.get('bollinger', 0.15),
            'Stochastic': weights.get('stochastic', 0.15),
            'ADX': weights.get('adx', 0.1)
        }
        
        for name, weight in weight_map.items():
            if name in signal_map:
                composite_score += signal_map[name].value * weight
                total_weight += weight
        
        if total_weight > 0:
            composite_score /= total_weight
        
        # Recommendation
        if composite_score > 0.5:
            recommendation = "STRONG_BUY"
        elif composite_score > 0.2:
            recommendation = "BUY"
        elif composite_score < -0.5:
            recommendation = "STRONG_SELL"
        elif composite_score < -0.2:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        # Trend direction
        if adx_value > 25 and plus_di > minus_di:
            trend = TrendDirection.UPTREND.value
        elif adx_value > 25 and minus_di > plus_di:
            trend = TrendDirection.DOWNTREND.value
        else:
            trend = TrendDirection.SIDEWAYS.value
        
        # Volatility regime
        bb_data = self.compute_bollinger_bands(prices)
        current_bandwidth = bb_data['bandwidth'][-1]
        avg_bandwidth = np.nanmean(bb_data['bandwidth'])
        
        if current_bandwidth > avg_bandwidth * 1.5:
            vol_regime = "high"
        elif current_bandwidth < avg_bandwidth * 0.5:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        return SignalBundle(
            asset_name="Asset",
            signals=signals,
            composite_score=float(composite_score),
            recommendation=recommendation,
            trend_direction=trend,
            volatility_regime=vol_regime
        )