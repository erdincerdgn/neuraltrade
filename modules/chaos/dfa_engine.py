"""
Detrended Fluctuation Analysis Engine - Hurst Exponent Calculator
Author: Erdinc Erdogan
Purpose: Computes the Hurst exponent using DFA for robust long-range dependence detection
in non-stationary financial time series data.
References:
- Peng et al. (1994) "Mosaic Organization of DNA Nucleotides"
- Hurst (1951) "Long-term Storage Capacity of Reservoirs"
- DFA for Financial Time Series Analysis
Usage:
    dfa = DFAEngine(min_window=4, max_window=None)
    result = dfa.compute(price_series)
    print(f"Hurst: {result.hurst:.3f}, Regime: {result.regime}")
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class DFAResult:
    hurst: float
    alpha: float
    r_squared: float
    is_valid: bool
    regime: str

class DFAEngine:
    EPSILON = 1e-10
    
    def __init__(self, min_window: int = 4, max_window: int = None, n_windows: int = 20, order: int = 1):
        self.min_window = min_window
        self.max_window = max_window
        self.n_windows = n_windows
        self.order = order
    
    def _cumulative_sum(self, data: np.ndarray) -> np.ndarray:
        return np.cumsum(data - np.mean(data))
    
    def _detrend_segment(self, segment: np.ndarray) -> np.ndarray:
        x = np.arange(len(segment))
        coeffs = np.polyfit(x, segment, self.order)
        return segment - np.polyval(coeffs, x)
    
    def _compute_fluctuation(self, profile: np.ndarray, window: int) -> float:
        n_segments = len(profile) // window
        if n_segments < 2:
            return np.nan
        fluctuations = []
        for i in range(n_segments):
            segment = profile[i * window:(i + 1) * window]
            detrended = self._detrend_segment(segment)
            fluctuations.append(np.sqrt(np.mean(detrended**2) + self.EPSILON))
        return np.mean(fluctuations)
    
    def compute(self, data: np.ndarray) -> DFAResult:
        n = len(data)
        if n < self.min_window * 4:
            return DFAResult(0.5, 0.5, 0.0, False, "INSUFFICIENT_DATA")
        
        max_win = self.max_window or n // 4
        windows = np.unique(np.logspace(np.log10(self.min_window), np.log10(max_win), self.n_windows).astype(int))
        profile = self._cumulative_sum(data)
        
        fluctuations, valid_windows = [], []
        for w in windows:
            f = self._compute_fluctuation(profile, w)
            if not np.isnan(f) and f > self.EPSILON:
                fluctuations.append(f)
                valid_windows.append(w)
        
        if len(valid_windows) < 3:
            return DFAResult(0.5, 0.5, 0.0, False, "INSUFFICIENT_WINDOWS")
        
        log_w, log_f = np.log(valid_windows), np.log(fluctuations)
        coeffs = np.polyfit(log_w, log_f, 1)
        hurst = coeffs[0]
        
        predicted = np.polyval(coeffs, log_w)
        ss_res = np.sum((log_f - predicted)**2)
        ss_tot = np.sum((log_f - np.mean(log_f))**2) + self.EPSILON
        r_squared = 1 - ss_res / ss_tot
        
        regime = "MEAN_REVERTING" if hurst < 0.4 else "TRENDING" if hurst > 0.6 else "RANDOM_WALK"
        return DFAResult(hurst, hurst, r_squared, True, regime)
