"""
Multiscale Entropy Engine - Complexity Analysis Across Timescales
Author: Erdinc Erdogan
Purpose: Computes multiscale entropy to capture market complexity across multiple timescales,
enabling regime detection and adaptive trading strategy selection.
References:
- Costa et al. (2002) "Multiscale Entropy Analysis of Complex Physiologic Time Series"
- Sample Entropy: Richman & Moorman (2000)
- Complexity Theory in Financial Markets
Usage:
    mse = MSEEngine(max_scale=20, tolerance=0.15)
    result = mse.compute(price_series)
    print(f"Complexity Index: {result.complexity_index:.3f}")
"""

import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class MSEResult:
    mse_values: List[float]
    complexity_index: float
    scales: List[int]
    is_valid: bool

class MSEEngine:
    EPSILON = 1e-10
    
    def __init__(self, max_scale: int = 20, m: int = 2, r_factor: float = 0.15):
        self.max_scale = max_scale
        self.m = m  # Embedding dimension
        self.r_factor = r_factor  # Tolerance factor (r = r_factor * std)
    
    def _coarse_grain(self, data: np.ndarray, scale: int) -> np.ndarray:
        n = len(data) // scale
        if n == 0:
            return np.array([])
        coarse = np.zeros(n)
        for i in range(n):
            coarse[i] = np.mean(data[i * scale:(i + 1) * scale])
        return coarse
    
    def _sample_entropy(self, data: np.ndarray, m: int, r: float) -> float:
        n = len(data)
        if n < m + 2:
            return np.nan
        
        def count_matches(template_len):
            count = 0
            templates = np.array([data[i:i + template_len] for i in range(n - template_len)])
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count
        
        A = count_matches(m + 1)
        B = count_matches(m)
        
        if B == 0 or A == 0:
            return np.nan
        
        return -np.log((A + self.EPSILON) / (B + self.EPSILON))
    
    def compute(self, data: np.ndarray) -> MSEResult:
        if len(data) < self.max_scale * 10:
            return MSEResult([], 0.0, [], False)
        
        r = self.r_factor * np.std(data)
        mse_values = []
        scales = []
        
        for scale in range(1, self.max_scale + 1):
            coarse = self._coarse_grain(data, scale)
            if len(coarse) < self.m + 10:
                break
            se = self._sample_entropy(coarse, self.m, r)
            if not np.isnan(se):
                mse_values.append(se)
                scales.append(scale)
        
        if len(mse_values) < 3:
            return MSEResult([], 0.0, [], False)
        
        # Complexity index = area under MSE curve
        complexity_index = np.trapz(mse_values, scales)
        
        return MSEResult(mse_values, complexity_index, scales, True)
