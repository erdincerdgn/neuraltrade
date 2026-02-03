"""
High-Precision Arithmetic for Chaos Calculations
Author: Erdinc Erdogan
Purpose: Implements high-precision arithmetic operations using Kahan summation and Neumaier's
algorithm to minimize floating-point precision loss in chaos calculations.
References:
- Kahan (1965) "Pracniques: Further Remarks on Reducing Truncation Errors"
- Neumaier (1974) "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen"
- IEEE 754 Floating-Point Arithmetic
Usage:
    result = PrecisionMath.kahan_sum(values)
    mean, var = PrecisionMath.stable_mean_var(values)
"""

import numpy as np
from typing import Tuple

class PrecisionMath:
    """High-precision arithmetic operations."""
    
    EPSILON = np.finfo(np.float64).eps
    
    @staticmethod
    def kahan_sum(values: np.ndarray) -> float:
        """Kahan summation algorithm - reduces floating-point error."""
        total = 0.0
        compensation = 0.0
        for val in values:
            y = val - compensation
            t = total + y
            compensation = (t - total) - y
            total = t
        return total
    
    @staticmethod
    def neumaier_sum(values: np.ndarray) -> float:
        """Neumaier's improved Kahan summation."""
        total = 0.0
        compensation = 0.0
        for val in values:
            t = total + val
            if abs(total) >= abs(val):
                compensation += (total - t) + val
            else:
                compensation += (val - t) + total
            total = t
        return total + compensation
    
    @staticmethod
    def precise_norm(vec: np.ndarray) -> float:
        """Precise L2 norm using compensated summation."""
        squared = vec ** 2
        return np.sqrt(PrecisionMath.neumaier_sum(squared))
    
    @staticmethod
    def precise_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Precise Euclidean distance."""
        diff = a - b
        return PrecisionMath.precise_norm(diff)
    
    @staticmethod
    def safe_log(x: float, epsilon: float = 1e-300) -> float:
        """Safe logarithm that never returns -inf."""
        return np.log(max(x, epsilon))
    
    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """Safe division that never returns inf or nan."""
        if abs(b) < PrecisionMath.EPSILON:
            return default
        result = a / b
        if np.isnan(result) or np.isinf(result):
            return default
        return result
