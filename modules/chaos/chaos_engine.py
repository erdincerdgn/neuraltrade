"""
Integrated Chaos Engine - High-Dimensional Non-Linear Dynamics
Author: Erdinc Erdogan
Purpose: Combines DFA, Lyapunov exponent, and MSE analysis for institutional-grade chaos
detection and market regime classification in financial time series.
References:
- Detrended Fluctuation Analysis (DFA): Peng et al. (1994)
- Lyapunov Exponent: Rosenstein et al. (1993)
- Multiscale Entropy: Costa et al. (2002)
Usage:
    engine = ChaosEngine()
    metrics = engine.analyze(price_series)
    if metrics.is_chaotic: reduce_position_size(metrics.position_size_factor)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ChaosMetrics:
    hurst: float
    lyapunov: float
    complexity_index: float
    regime: str
    is_chaotic: bool
    position_size_factor: float
    is_valid: bool

class ChaosGate:
    """Links chaos metrics to position sizing."""
    
    def __init__(self, lyapunov_threshold: float = 0.5, hurst_trend: float = 0.6, hurst_mr: float = 0.4):
        self.lyapunov_threshold = lyapunov_threshold
        self.hurst_trend = hurst_trend
        self.hurst_mr = hurst_mr
    
    def compute_position_factor(self, hurst: float, lyapunov: float, complexity: float) -> float:
        """Entropy-based position sizing."""
        base_factor = 1.0
        
        # High chaos = reduce position
        if lyapunov > self.lyapunov_threshold:
            base_factor *= 0.5
        
        # Strong trend = increase position
        if hurst > self.hurst_trend:
            base_factor *= 1.2
        # Mean-reverting = moderate position
        elif hurst < self.hurst_mr:
            base_factor *= 0.8
        
        # High complexity = reduce position
        if complexity > 20:
            base_factor *= 0.7
        
        return np.clip(base_factor, 0.1, 1.5)

class IntegratedChaosEngine:
    """Main chaos engine combining all metrics."""
    
    EPSILON = 1e-10
    
    def __init__(self):
        self.gate = ChaosGate()
    
    def _dfa_hurst(self, data: np.ndarray) -> float:
        """Simplified DFA for speed."""
        n = len(data)
        if n < 50:
            return 0.5
        
        profile = np.cumsum(data - np.mean(data))
        windows = [10, 20, 50, 100, min(200, n//4)]
        windows = [w for w in windows if w < n//2]
        
        if len(windows) < 3:
            return 0.5
        
        fluctuations = []
        for w in windows:
            n_seg = n // w
            f_list = []
            for i in range(n_seg):
                seg = profile[i*w:(i+1)*w]
                x = np.arange(w)
                trend = np.polyval(np.polyfit(x, seg, 1), x)
                f_list.append(np.sqrt(np.mean((seg - trend)**2) + self.EPSILON))
            fluctuations.append(np.mean(f_list))
        
        log_w = np.log(windows)
        log_f = np.log(np.array(fluctuations) + self.EPSILON)
        return np.polyfit(log_w, log_f, 1)[0]
    
    def _fast_lyapunov(self, data: np.ndarray) -> float:
        """Fast Lyapunov approximation."""
        n = len(data)
        if n < 100:
            return 0.0
        
        # Simple divergence estimate
        divergences = []
        for i in range(min(50, n-20)):
            d0 = abs(data[i] - data[i+1]) + self.EPSILON
            d10 = abs(data[i+10] - data[i+11]) + self.EPSILON
            divergences.append(np.log(d10/d0) / 10)
        
        return np.mean(divergences) if divergences else 0.0
    
    def _sample_entropy(self, data: np.ndarray) -> float:
        """Fast sample entropy."""
        n = len(data)
        if n < 50:
            return 0.0
        
        r = 0.2 * np.std(data)
        m = 2
        
        def count_matches(template_len):
            count = 0
            for i in range(n - template_len - 1):
                for j in range(i + 1, n - template_len):
                    if np.max(np.abs(data[i:i+template_len] - data[j:j+template_len])) < r:
                        count += 1
            return count + self.EPSILON
        
        A = count_matches(m + 1)
        B = count_matches(m)
        return -np.log(A / B)
    
    def compute(self, data: np.ndarray) -> ChaosMetrics:
        """Compute all chaos metrics."""
        if len(data) < 100:
            return ChaosMetrics(0.5, 0.0, 0.0, "INSUFFICIENT_DATA", False, 1.0, False)
        
        hurst = self._dfa_hurst(data)
        lyapunov = self._fast_lyapunov(data)
        complexity = self._sample_entropy(data)
        
        # Regime classification
        if hurst < 0.4:
            regime = "MEAN_REVERTING"
        elif hurst > 0.6:
            regime = "TRENDING"
        else:
            regime = "RANDOM_WALK"
        
        is_chaotic = lyapunov > 0.05
        position_factor = self.gate.compute_position_factor(hurst, lyapunov, complexity)
        
        return ChaosMetrics(hurst, lyapunov, complexity, regime, is_chaotic, position_factor, True)
