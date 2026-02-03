"""
Robust Conditional Independence Test - Adaptive Regularization CI Testing
Author: Erdinc Erdogan
Purpose: Implements robust conditional independence testing with adaptive regularization,
caching, and numerical stability for reliable causal structure learning.
References:
- Partial Correlation and Conditional Independence
- Ridge Regularization for Numerical Stability
- Fisher's Z-Transform for Correlation Testing
Usage:
    ci_test = RobustConditionalIndependenceTest(method="adaptive")
    is_independent, z_stat, p_value = ci_test.test(data, x_idx, y_idx, cond_set)
"""

import numpy as np
from scipy import stats
from scipy.linalg import inv
from typing import Tuple, List, Dict


class RobustConditionalIndependenceTest:
    """Robust CI testing with adaptive regularization and caching."""
    
    def __init__(self, alpha: float = 0.05, method: str = "adaptive",
                 min_regularization: float = 1e-6, regularization_scale: float = 0.01):
        self.alpha = alpha
        self.method = method
        self.min_regularization = min_regularization
        self.regularization_scale = regularization_scale
        self._cache: Dict[tuple, Tuple[bool, float, float]] = {}
        
    def test(self, data: np.ndarray, x_idx: int, y_idx: int,
             conditioning_set: List[int]) -> Tuple[bool, float, float]:
        cache_key = (x_idx, y_idx, tuple(sorted(conditioning_set)))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        n_samples = data.shape[0]
        if self.method == "robust":
            result = self._robust_test(data, x_idx, y_idx, conditioning_set, n_samples)
        else:
            result = self._adaptive_test(data, x_idx, y_idx, conditioning_set, n_samples)
        
        self._cache[cache_key] = result
        return result
    
    def _compute_adaptive_regularization(self, matrix: np.ndarray) -> float:
        try:
            cond = np.linalg.cond(matrix)
            if np.isinf(cond) or np.isnan(cond):
                return 0.1
            return max(self.min_regularization, self.regularization_scale / cond)
        except:
            return 0.1
    
    def _adaptive_test(self, data, x_idx, y_idx, conditioning_set, n_samples):
        if len(conditioning_set) == 0:
            partial_corr = np.corrcoef(data[:, x_idx], data[:, y_idx])[0, 1]
        else:
            all_vars = [x_idx, y_idx] + list(conditioning_set)
            corr_matrix = np.corrcoef(data[:, all_vars].T)
            reg_lambda = self._compute_adaptive_regularization(corr_matrix)
            regularized = corr_matrix + reg_lambda * np.eye(len(all_vars))
            try:
                precision = inv(regularized)
                partial_corr = -precision[0, 1] / np.sqrt(abs(precision[0, 0] * precision[1, 1]) + 1e-10)
            except:
                partial_corr = 0.0
        return self._fisher_z_test(partial_corr, n_samples, len(conditioning_set))
    
    def _robust_test(self, data, x_idx, y_idx, conditioning_set, n_samples):
        if len(conditioning_set) == 0:
            partial_corr, _ = stats.spearmanr(data[:, x_idx], data[:, y_idx])
        else:
            cond_vars = list(conditioning_set)
            x_ranks = stats.rankdata(data[:, x_idx])
            y_ranks = stats.rankdata(data[:, y_idx])
            cond_ranks = np.column_stack([stats.rankdata(data[:, i]) for i in cond_vars])
            X_design = np.column_stack([np.ones(n_samples), cond_ranks])
            try:
                resid_x = x_ranks - X_design @ np.linalg.lstsq(X_design, x_ranks, rcond=None)[0]
                resid_y = y_ranks - X_design @ np.linalg.lstsq(X_design, y_ranks, rcond=None)[0]
                partial_corr, _ = stats.spearmanr(resid_x, resid_y)
            except:
                partial_corr = 0.0
        return self._fisher_z_test(partial_corr, n_samples, len(conditioning_set))
    
    def _fisher_z_test(self, partial_corr, n_samples, cond_size):
        partial_corr = np.clip(partial_corr, -0.9999, 0.9999)
        if np.isnan(partial_corr):
            return True, 0.0, 1.0
        z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr + 1e-10))
        se = 1.0 / np.sqrt(max(1, n_samples - cond_size - 3))
        p_value = 2 * (1 - stats.norm.cdf(abs(z) / se))
        return p_value > self.alpha, abs(z) / se, p_value
    
    def clear_cache(self):
        self._cache.clear()
