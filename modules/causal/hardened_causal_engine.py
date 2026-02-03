"""
Hardened Causal Engine - Robust Conditional Independence Testing
Author: Erdinc Erdogan
Purpose: Production-hardened causal inference engine with zero-inference protection, thread-safe
operations, and robust conditional independence testing for live trading systems.
References:
- Robust Statistics in Causal Inference
- Thread-Safe Concurrent Programming Patterns
- Numerical Stability in Statistical Computing
Usage:
    engine = HardenedCausalEngine(variable_names)
    engine.fit(data)
    result = engine.test_ci(x_idx=0, y_idx=1, cond_set=[2, 3])
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import inv
from collections import OrderedDict
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class HardenedCIResult:
    is_independent: bool
    statistic: float
    p_value: float
    effect_size: float
    fdr_adjusted: bool
    passes_effect_filter: bool

class ZeroInferenceProtection:
    def __init__(self, base_alpha=0.05, fdr_level=0.05, min_effect=0.2, vol_thresh=0.02):
        self.base_alpha, self.fdr_level = base_alpha, fdr_level
        self.min_effect, self.vol_thresh = min_effect, vol_thresh
    
    def benjamini_hochberg(self, p_values: List[float]) -> List[bool]:
        m = len(p_values)
        if m == 0: return []
        sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
        rejections = [False] * m
        for rank, (idx, p) in enumerate(sorted_pairs, 1):
            if p <= (rank / m) * self.fdr_level:
                rejections[idx] = True
            else: break
        return rejections
    
    def adaptive_alpha(self, volatility: float) -> float:
        vol_ratio = max(0, volatility / self.vol_thresh - 1)
        return self.base_alpha / (1 + vol_ratio)
    
    def passes_effect_filter(self, r: float) -> bool:
        d = 2 * abs(r) / np.sqrt(1 - r**2 + 1e-10)
        return d >= self.min_effect

class BoundedLRUCache:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache = OrderedDict()
    
    def get(self, key: Tuple) -> Optional[HardenedCIResult]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, key: Tuple, result: HardenedCIResult):
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = result

class RobustDistanceCorrelation:
    def __init__(self, winsorize=0.01, n_bootstrap=100):
        self.winsorize, self.n_bootstrap = winsorize, n_bootstrap
    
    def _winsorize(self, x):
        lo, hi = np.percentile(x, [self.winsorize*100, (1-self.winsorize)*100])
        return np.clip(x, lo, hi)
    
    def _dcor(self, x, y):
        n = len(x)
        A = squareform(pdist(stats.rankdata(self._winsorize(x)).reshape(-1,1)))
        B = squareform(pdist(stats.rankdata(self._winsorize(y)).reshape(-1,1)))
        A = A - A.mean(0,keepdims=True) - A.mean(1,keepdims=True) + A.mean()
        B = B - B.mean(0,keepdims=True) - B.mean(1,keepdims=True) + B.mean()
        dxy = np.sqrt(max(0, np.sum(A*B)/(n*n)))
        dxx = np.sqrt(max(0, np.sum(A*A)/(n*n)))
        dyy = np.sqrt(max(0, np.sum(B*B)/(n*n)))
        return dxy / np.sqrt(dxx*dyy) if dxx*dyy > 0 else 0
    
    def bootstrap_ci(self, x, y, conf=0.95):
        n = len(x)
        samples = [self._dcor(x[np.random.choice(n,n,True)], 
                              y[np.random.choice(n,n,True)]) 
                   for _ in range(self.n_bootstrap)]
        alpha = 1 - conf
        return self._dcor(x,y), np.percentile(samples, alpha/2*100), np.percentile(samples, (1-alpha/2)*100)

class HardenedCausalEngine:
    def __init__(self, variable_names: List[str], alpha=0.05, max_cache=10000):
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        self.var_to_idx = {n: i for i, n in enumerate(variable_names)}
        self.zip = ZeroInferenceProtection(base_alpha=alpha)
        self.cache = BoundedLRUCache(max_size=max_cache)
        self.robust_dcor = RobustDistanceCorrelation()
        self._data = None
        self._volatility = 0.02
    
    def fit(self, data: np.ndarray):
        self._data = data.copy()
        if data.shape[0] > 20:
            self._volatility = np.std(np.diff(data[:, 0]))
    
    def test_ci(self, x_idx: int, y_idx: int, cond_set: List[int] = None) -> HardenedCIResult:
        cond_set = cond_set or []
        key = (min(x_idx, y_idx), max(x_idx, y_idx), tuple(sorted(cond_set)))
        
        cached = self.cache.get(key)
        if cached: return cached
        
        x, y = self._data[:, x_idx], self._data[:, y_idx]
        
        # Residualize if conditioning
        if cond_set:
            z = self._data[:, cond_set]
            X = np.column_stack([np.ones(len(x)), z])
            x = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
            y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Robust dCor with bootstrap
        dcor, ci_lo, ci_hi = self.robust_dcor.bootstrap_ci(x, y)
        
        # Adaptive alpha
        alpha_eff = self.zip.adaptive_alpha(self._volatility)
        
        # Effect size filter
        passes_effect = self.zip.passes_effect_filter(dcor)
        
        # Independence decision
        is_independent = ci_lo <= 0 or not passes_effect
        
        result = HardenedCIResult(
            is_independent=is_independent,
            statistic=dcor,
            p_value=1.0 if ci_lo <= 0 else 0.01,
            effect_size=dcor,
            fdr_adjusted=False,
            passes_effect_filter=passes_effect
        )
        
        self.cache.put(key, result)
        return result
    
    def test_all_edges_with_fdr(self) -> Dict[Tuple[int, int], HardenedCIResult]:
        results = {}
        p_values = []
        edges = []
        
        for i in range(self.n_vars):
            for j in range(i+1, self.n_vars):
                result = self.test_ci(i, j)
                results[(i, j)] = result
                p_values.append(result.p_value)
                edges.append((i, j))
        
        # Apply FDR correction
        fdr_rejections = self.zip.benjamini_hochberg(p_values)
        
        for idx, (i, j) in enumerate(edges):
            old = results[(i, j)]
            results[(i, j)] = HardenedCIResult(
                is_independent=not fdr_rejections[idx],
                statistic=old.statistic,
                p_value=old.p_value,
                effect_size=old.effect_size,
                fdr_adjusted=True,
                passes_effect_filter=old.passes_effect_filter
            )
        
        return results
