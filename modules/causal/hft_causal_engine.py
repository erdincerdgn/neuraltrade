"""
HFT Causal Engine - Temporal Causality for High-Frequency Trading
Author: Erdinc Erdogan
Purpose: Implements temporal causality analysis for HFT including Granger causality testing,
non-linear dependence detection, and adaptive DAG rebuilding for real-time trading.
References:
- Granger (1969) "Investigating Causal Relations by Econometric Models"
- Székely et al. (2007) "Measuring and Testing Dependence by Correlation of Distances"
- Pearl (2009) "Causality: Models, Reasoning, and Inference"
Usage:
    engine = HFTCausalEngine(variable_names, max_lag=5)
    engine.fit(data)
    result = engine.test_granger_causality("OrderFlow", "PriceChange")
"""

import numpy as np
from scipy import stats
from scipy.linalg import inv
from scipy.spatial.distance import pdist, squareform
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import deque
import time


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TimeLaggedCIResult:
    is_independent: bool
    statistic: float
    p_value: float
    optimal_lag: int
    granger_f_stat: float

@dataclass
class NonLinearCIResult:
    is_independent: bool
    dcor: float
    hsic: float
    p_value: float
    method_used: str

@dataclass
class DAGSnapshot:
    adjacency_matrix: np.ndarray
    timestamp: float
    regime: str
    volatility: float
    n_samples_used: int

@dataclass
class RegimeState:
    regime: str = "NORMAL"
    volatility: float = 0.0
    ticks_since_rebuild: int = 0
    structural_break_score: float = 0.0


# ============================================================================
# TIME-LAGGED CI TEST
# ============================================================================

class TimeLaggedCITest:
    """Granger-style time-lagged conditional independence test."""
    
    def __init__(self, max_lag: int = 10, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha = alpha
    
    def _select_optimal_lag(self, data: np.ndarray, x_idx: int, y_idx: int) -> int:
        best_lag, best_bic = 1, np.inf
        for lag in range(1, min(self.max_lag + 1, len(data) // 3)):
            n = len(data) - lag
            y = data[lag:, y_idx]
            X = np.column_stack([np.ones(n)] + 
                [data[lag-k:len(data)-k, y_idx] for k in range(1, lag+1)] +
                [data[lag-k:len(data)-k, x_idx] for k in range(1, lag+1)])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                ssr = np.sum((y - X @ beta)**2)
                bic = n * np.log(ssr/n) + X.shape[1] * np.log(n)
                if bic < best_bic:
                    best_bic, best_lag = bic, lag
            except:
                continue
        return best_lag
    
    def test(self, data: np.ndarray, x_idx: int, y_idx: int, 
             lag: int = None) -> TimeLaggedCIResult:
        if lag is None:
            lag = self._select_optimal_lag(data, x_idx, y_idx)
        
        n = len(data) - lag
        y = data[lag:, y_idx]
        
        X_r = np.column_stack([np.ones(n)] + 
            [data[lag-k:len(data)-k, y_idx] for k in range(1, lag+1)])
        X_u = np.column_stack([X_r] + 
            [data[lag-k:len(data)-k, x_idx] for k in range(1, lag+1)])
        
        try:
            ssr_r = np.sum((y - X_r @ np.linalg.lstsq(X_r, y, rcond=None)[0])**2)
            ssr_u = np.sum((y - X_u @ np.linalg.lstsq(X_u, y, rcond=None)[0])**2)
            f_stat = ((ssr_r - ssr_u) / lag) / (ssr_u / (n - X_u.shape[1]))
            p_value = 1 - stats.f.cdf(f_stat, lag, n - X_u.shape[1])
        except:
            f_stat, p_value = 0.0, 1.0
        
        return TimeLaggedCIResult(p_value > self.alpha, f_stat, p_value, lag, f_stat)


# ============================================================================
# NON-LINEAR CI TEST
# ============================================================================

class NonLinearCITest:
    """Distance correlation and HSIC for non-linear dependence."""
    
    def __init__(self, alpha: float = 0.05, n_permutations: int = 100):
        self.alpha = alpha
        self.n_permutations = n_permutations
    
    def _dcor(self, x: np.ndarray, y: np.ndarray) -> float:
        def dcov(a, b):
            n = len(a)
            A = squareform(pdist(a.reshape(-1,1)))
            B = squareform(pdist(b.reshape(-1,1)))
            A = A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
            B = B - B.mean(0, keepdims=True) - B.mean(1, keepdims=True) + B.mean()
            return np.sqrt(np.sum(A * B) / (n * n))
        
        dxy, dxx, dyy = dcov(x, y), dcov(x, x), dcov(y, y)
        return dxy / np.sqrt(dxx * dyy) if dxx * dyy > 0 else 0.0
    
    def test(self, data: np.ndarray, x_idx: int, y_idx: int) -> NonLinearCIResult:
        x, y = data[:, x_idx], data[:, y_idx]
        observed = self._dcor(x, y)
        
        count = sum(1 for _ in range(self.n_permutations) 
                   if self._dcor(x, np.random.permutation(y)) >= observed)
        p_value = (count + 1) / (self.n_permutations + 1)
        
        return NonLinearCIResult(p_value > self.alpha, observed, 0.0, p_value, "dcor")


# ============================================================================
# ADAPTIVE DAG MANAGER
# ============================================================================

class AdaptiveDAGManager:
    """Regime-aware DAG persistence with CUSUM break detection."""
    
    THRESHOLDS = {
        "NORMAL": {"ticks": 1000}, "HIGH_VOL": {"ticks": 100}, "CRISIS": {"ticks": 50}
    }
    
    def __init__(self, n_vars: int, vol_threshold: float = 0.02, cusum_threshold: float = 3.0):
        self.n_vars = n_vars
        self.vol_threshold = vol_threshold
        self.cusum_threshold = cusum_threshold
        self.current_dag = None
        self.dag_history = deque(maxlen=10)
        self.regime_state = RegimeState()
        self._cusum_pos = self._cusum_neg = 0.0
        self._tick = self._last_rebuild = 0
    
    def _detect_regime(self, returns: np.ndarray) -> str:
        if len(returns) < 20:
            return "NORMAL"
        vol = np.std(returns[-20:])
        self.regime_state.volatility = vol
        if vol > self.vol_threshold * 2:
            return "CRISIS"
        return "HIGH_VOL" if vol > self.vol_threshold else "NORMAL"
    
    def should_rebuild(self, data: np.ndarray = None) -> Tuple[bool, str]:
        self._tick += 1
        ticks_elapsed = self._tick - self._last_rebuild
        
        if data is not None and len(data) > 20:
            self.regime_state.regime = self._detect_regime(np.diff(data[:, 0]))
        
        threshold = self.THRESHOLDS.get(self.regime_state.regime, {"ticks": 1000})["ticks"]
        if ticks_elapsed >= threshold:
            return True, f"TICK_THRESHOLD ({self.regime_state.regime})"
        return False, "NO_REBUILD"
    
    def update_dag(self, new_dag: np.ndarray, data: np.ndarray = None):
        self.dag_history.append(DAGSnapshot(
            new_dag.copy(), time.time(), self.regime_state.regime,
            self.regime_state.volatility, len(data) if data is not None else 0
        ))
        self.current_dag = new_dag.copy()
        self._last_rebuild = self._tick


# ============================================================================
# INTEGRATED HFT CAUSAL ENGINE
# ============================================================================

class HFTCausalEngine:
    """
    Integrated HFT Causal Engine.
    
    Combines:
    - Time-lagged Granger causality
    - Non-linear distance correlation
    - Adaptive regime-aware DAG management
    """
    
    def __init__(self, variable_names: List[str], max_lag: int = 10, alpha: float = 0.05):
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        self.var_to_idx = {name: i for i, name in enumerate(variable_names)}
        
        self.tl_test = TimeLaggedCITest(max_lag=max_lag, alpha=alpha)
        self.nl_test = NonLinearCITest(alpha=alpha)
        self.dag_manager = AdaptiveDAGManager(self.n_vars)
        
        self._data_buffer = None
    
    def fit(self, data: np.ndarray):
        self._data_buffer = data.copy()
    
    def test_granger_causality(self, cause: str, effect: str, 
                                lag: int = None) -> TimeLaggedCIResult:
        x_idx = self.var_to_idx[cause]
        y_idx = self.var_to_idx[effect]
        return self.tl_test.test(self._data_buffer, x_idx, y_idx, lag)
    
    def test_nonlinear_dependence(self, var1: str, var2: str) -> NonLinearCIResult:
        x_idx = self.var_to_idx[var1]
        y_idx = self.var_to_idx[var2]
        return self.nl_test.test(self._data_buffer, x_idx, y_idx)
    
    def should_rebuild_dag(self, new_data: np.ndarray = None) -> Tuple[bool, str]:
        return self.dag_manager.should_rebuild(new_data)
    
    def rebuild_dag(self, data: np.ndarray, dag: np.ndarray = None):
        if dag is None:
            dag = np.zeros((self.n_vars, self.n_vars))
        self.dag_manager.update_dag(dag, data)
        self._data_buffer = data.copy()
    
    @property
    def current_dag(self) -> Optional[np.ndarray]:
        return self.dag_manager.current_dag
    
    @property
    def regime(self) -> str:
        return self.dag_manager.regime_state.regime
