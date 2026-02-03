"""
Robust DFA Engine - Flash Crash Resistant Hurst Computation
Author: Erdinc Erdogan
Purpose: Implements robust DFA with outlier detection and gap handling for accurate Hurst
exponent estimation during extreme market events like flash crashes.
References:
- Peng et al. (1994) "Mosaic Organization of DNA Nucleotides"
- Robust Statistics for Financial Time Series
- Flash Crash Detection and Handling
Usage:
    robust_dfa = RobustDFAEngine(outlier_threshold=5.0, gap_threshold=0.1)
    result = robust_dfa.compute(price_series)
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class RobustDFAResult:
    hurst: float
    alpha: float
    r_squared: float
    gaps_detected: int
    outliers_removed: int
    is_valid: bool
    regime: str

class RobustDFAEngine:
    EPSILON = 1e-10
    
    def __init__(self, min_window: int = 4, max_window: int = None, n_windows: int = 20,
                 gap_threshold: float = 5.0, outlier_threshold: float = 10.0):
        self.min_window = min_window
        self.max_window = max_window
        self.n_windows = n_windows
        self.gap_threshold = gap_threshold  # Sigma threshold for gap detection
        self.outlier_threshold = outlier_threshold  # Sigma threshold for outliers
    
    def _detect_and_fill_gaps(self, data: np.ndarray) -> tuple:
        """Detect flash crash gaps and interpolate."""
        returns = np.diff(data)
        std_ret = np.std(returns) + self.EPSILON
        
        # Detect gaps (returns > threshold * sigma)
        gap_mask = np.abs(returns) > self.gap_threshold * std_ret
        gaps_detected = np.sum(gap_mask)
        
        if gaps_detected > 0:
            # Linear interpolation for gaps
            data_clean = data.copy()
            gap_indices = np.where(gap_mask)[0] + 1
            
            for idx in gap_indices:
                if idx > 0 and idx < len(data) - 1:
                    data_clean[idx] = (data_clean[idx - 1] + data_clean[idx + 1]) / 2
            
            return data_clean, gaps_detected
        
        return data, 0
    
    def _winsorize(self, data: np.ndarray) -> tuple:
        """Winsorize extreme outliers."""
        mean = np.mean(data)
        std = np.std(data) + self.EPSILON
        
        lower = mean - self.outlier_threshold * std
        upper = mean + self.outlier_threshold * std
        
        outliers = np.sum((data < lower) | (data > upper))
        data_clean = np.clip(data, lower, upper)
        
        return data_clean, outliers
    
    def _cumulative_sum(self, data: np.ndarray) -> np.ndarray:
        return np.cumsum(data - np.mean(data))
    
    def _detrend_segment_safe(self, segment: np.ndarray) -> np.ndarray:
        """Safe detrending that handles edge cases."""
        n = len(segment)
        if n < 2:
            return np.zeros(n)
        
        x = np.arange(n)
        
        try:
            # Use robust linear regression
            coeffs = np.polyfit(x, segment, 1)
            if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
                return segment - np.mean(segment)
            trend = np.polyval(coeffs, x)
            return segment - trend
        except:
            return segment - np.mean(segment)
    
    def _compute_fluctuation(self, profile: np.ndarray, window: int) -> float:
        n_segments = len(profile) // window
        if n_segments < 2:
            return np.nan
        
        fluctuations = []
        for i in range(n_segments):
            segment = profile[i * window:(i + 1) * window]
            detrended = self._detrend_segment_safe(segment)
            rms = np.sqrt(np.mean(detrended**2) + self.EPSILON)
            
            if not np.isnan(rms) and not np.isinf(rms):
                fluctuations.append(rms)
        
        return np.mean(fluctuations) if fluctuations else np.nan
    
    def compute(self, data: np.ndarray) -> RobustDFAResult:
        n = len(data)
        if n < self.min_window * 4:
            return RobustDFAResult(0.5, 0.5, 0.0, 0, 0, False, "INSUFFICIENT_DATA")
        
        # Step 1: Detect and fill gaps
        data_clean, gaps_detected = self._detect_and_fill_gaps(data)
        
        # Step 2: Winsorize outliers
        data_clean, outliers_removed = self._winsorize(data_clean)
        
        # Step 3: Standard DFA
        max_win = self.max_window or n // 4
        windows = np.unique(np.logspace(np.log10(self.min_window), np.log10(max_win), self.n_windows).astype(int))
        profile = self._cumulative_sum(data_clean)
        
        fluctuations, valid_windows = [], []
        for w in windows:
            f = self._compute_fluctuation(profile, w)
            if not np.isnan(f) and f > self.EPSILON:
                fluctuations.append(f)
                valid_windows.append(w)
        
        if len(valid_windows) < 3:
            return RobustDFAResult(0.5, 0.5, 0.0, gaps_detected, outliers_removed, False, "INSUFFICIENT_WINDOWS")
        
        log_w, log_f = np.log(valid_windows), np.log(fluctuations)
        coeffs = np.polyfit(log_w, log_f, 1)
        hurst = coeffs[0]
        
        # Bound hurst to reasonable range
        hurst = np.clip(hurst, 0.0, 2.0)
        
        predicted = np.polyval(coeffs, log_w)
        ss_res = np.sum((log_f - predicted)**2)
        ss_tot = np.sum((log_f - np.mean(log_f))**2) + self.EPSILON
        r_squared = max(0, 1 - ss_res / ss_tot)
        
        regime = "MEAN_REVERTING" if hurst < 0.4 else "TRENDING" if hurst > 0.6 else "RANDOM_WALK"
        
        return RobustDFAResult(hurst, hurst, r_squared, gaps_detected, outliers_removed, True, regime)
