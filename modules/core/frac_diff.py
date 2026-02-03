"""
Fractional Differentiation Module - Memory-Preserving Stationarity
Author: Erdinc Erdogan
Purpose: Implements fractional differentiation to achieve stationarity while preserving
memory in financial time series, based on López de Prado's methodology.
References:
- López de Prado (2018) "Advances in Financial Machine Learning"
- Fractional Calculus: (1-B)^d × X_t = Σ w_k × X_{t-k}
- ADF Test for Optimal d Selection
Usage:
    frac_diff = FractionalDifferentiator()
    stationary_series = frac_diff.fit_transform(price_series, d=0.5)
    optimal_d = frac_diff.find_optimal_d(price_series)
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# FRACTIONAL DIFFERENTIATION RESULT
# ============================================================================

@dataclass
class FracDiffResult:
    """Result of fractional differentiation."""
    series: np.ndarray              # Fractionally differenced series
    d: float                        # Differentiation order
    weights: np.ndarray             # FFD weights used
    adf_statistic: float            # ADF test statistic
    adf_pvalue: float               # ADF p-value
    is_stationary: bool             # Stationarity flag
    memory_preserved: float         # Memory preservation ratio (0-1)
    correlation_with_original: float # Correlation with original
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimalDResult:
    """Result of optimal d search."""
    optimal_d: float                # Optimal differentiation order
    d_values_tested: np.ndarray     # All d values tested
    adf_statistics: np.ndarray      # ADF stats for each d
    adf_pvalues: np.ndarray         # ADF p-values for each d
    correlations: np.ndarray        # Correlations for each d
    memory_scores: np.ndarray       # Memory preservation scores
    first_stationary_d: float       # First d achieving stationarity


# ============================================================================
# WEIGHT CALCULATION
# ============================================================================

def get_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate fractional differentiation weights.
    
    Formula:
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k  for k >= 1
    
    Args:
        d: Differentiation order (0 < d < 1)
        size: Maximum number of weights
        threshold: Minimum weight magnitude to include
        
    Returns:
        Array of weights
    """
    weights = [1.0]
    k = 1
    
    while k < size:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    
    return np.array(weights)


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate Fixed-width window Fractional Differentiation (FFD) weights.
    
    FFD uses a fixed window size determined by weight threshold,
    making it more practical for real-time applications.
    
    Args:
        d: Differentiation order
        threshold: Weight cutoff threshold
        
    Returns:
        Array of FFD weights
    """
    weights = [1.0]
    k = 1
    
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        
        # Safety limit
        if k > 10000:
            break
    
    return np.array(weights)[::-1]  # Reverse for convolution


# ============================================================================
# FRACTIONAL DIFFERENTIATION
# ============================================================================

def frac_diff(series: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Apply fractional differentiation to a time series.
    
    Mathematical Definition:
        (1 - B)^d * X_t = Σ_{k=0}^{∞} w_k * X_{t-k}
    
    Args:
        series: Input time series
        d: Differentiation order (0 < d < 1)
        threshold: Weight cutoff threshold
        
    Returns:
        Fractionally differenced series
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    
    # Apply convolution
    result = np.zeros(len(series))
    
    for t in range(width - 1, len(series)):
        result[t] = np.dot(weights, series[t - width + 1:t + 1])
    
    # Set initial values to NaN (insufficient history)
    result[:width - 1] = np.nan
    
    return result


def frac_diff_fixed_window(series: np.ndarray, d: float, 
                           window: int = 100) -> np.ndarray:
    """
    Apply fractional differentiation with fixed window size.
    
    More memory-efficient for long series.
    
    Args:
        series: Input time series
        d: Differentiation order
        window: Fixed window size
        
    Returns:
        Fractionally differenced series
    """
    weights = get_weights(d, window)
    weights = weights[::-1]  # Reverse for convolution
    
    result = np.zeros(len(series))
    
    for t in range(len(weights) - 1, len(series)):
        result[t] = np.dot(weights, series[t - len(weights) + 1:t + 1])
    
    result[:len(weights) - 1] = np.nan
    
    return result


# ============================================================================
# ADF TEST (SIMPLIFIED)
# ============================================================================

def adf_test(series: np.ndarray, max_lag: int = None) -> Tuple[float, float]:
    """
    Simplified Augmented Dickey-Fuller test for stationarity.
    
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    
    Args:
        series: Time series to test
        max_lag: Maximum lag for ADF regression
        
    Returns:
        Tuple of (ADF statistic, p-value)
    """
    # Remove NaN values
    series = series[~np.isnan(series)]
    n = len(series)
    
    if n < 20:
        return 0.0, 1.0  # Not enough data
    
    if max_lag is None:
        max_lag = int(np.floor(np.power(n - 1, 1/3)))
    
    # First difference
    diff = np.diff(series)
    
    # Lagged level
    lagged = series[:-1]
    
    # Simple OLS regression: Δy_t = α + β*y_{t-1} + ε_t
    X = np.column_stack([np.ones(len(lagged)), lagged])
    y = diff
    
    try:
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Residuals
        residuals = y - X @ beta
        
        # Standard error of β
        sigma2 = np.sum(residuals**2) / (len(y) - 2)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(var_beta[1, 1])
        
        # ADF statistic
        adf_stat = beta[1] / se_beta
        
        # Approximate p-value using MacKinnon critical values
        # Critical values for n=100: 1%: -3.51, 5%: -2.89, 10%: -2.58
        if adf_stat < -3.51:
            p_value = 0.01
        elif adf_stat < -2.89:
            p_value = 0.05
        elif adf_stat < -2.58:
            p_value = 0.10
        else:
            # Approximate p-value
            p_value = min(1.0, norm.cdf(adf_stat + 2.5))
        
        return adf_stat, p_value
        
    except Exception:
        return 0.0, 1.0


# ============================================================================
# OPTIMAL D SEARCH
# ============================================================================

def find_optimal_d(series: np.ndarray, 
                   d_range: Tuple[float, float] = (0.0, 1.0),
                   num_points: int = 20,
                   significance: float = 0.05,
                   threshold: float = 1e-5) -> OptimalDResult:
    """
    Find optimal differentiation order d.
    
    Strategy:
    1. Test d values from 0 to 1
    2. Find minimum d that achieves stationarity (ADF p-value < significance)
    3. Balance stationarity with memory preservation
    
    Args:
        series: Input time series
        d_range: Range of d values to test
        num_points: Number of d values to test
        significance: ADF significance level
        threshold: Weight threshold for FFD
        
    Returns:
        OptimalDResult with optimal d and diagnostics
    """
    d_values = np.linspace(d_range[0], d_range[1], num_points)
    adf_stats = np.zeros(num_points)
    adf_pvals = np.zeros(num_points)
    correlations = np.zeros(num_points)
    memory_scores = np.zeros(num_points)
    
    original = series[~np.isnan(series)]
    
    first_stationary_d = 1.0
    found_stationary = False
    
    for i, d in enumerate(d_values):
        if d == 0:
            diff_series = series.copy()
        else:
            diff_series = frac_diff(series, d, threshold)
        
        # Remove NaN
        valid_idx = ~np.isnan(diff_series)
        diff_valid = diff_series[valid_idx]
        orig_valid = original[valid_idx[:len(original)]] if len(valid_idx) <= len(original) else original
        
        if len(diff_valid) < 20:
            adf_stats[i] = 0
            adf_pvals[i] = 1.0
            correlations[i] = 0
            memory_scores[i] = 0
            continue
        
        # ADF test
        adf_stat, adf_pval = adf_test(diff_valid)
        adf_stats[i] = adf_stat
        adf_pvals[i] = adf_pval
        
        # Correlation with original (memory preservation)
        min_len = min(len(diff_valid), len(orig_valid))
        if min_len > 10:
            corr = np.corrcoef(diff_valid[-min_len:], orig_valid[-min_len:])[0, 1]
            correlations[i] = corr if not np.isnan(corr) else 0
        
        # Memory score (higher is better)
        memory_scores[i] = correlations[i] * (1 - d)
        
        # Track first stationary d
        if adf_pval < significance and not found_stationary:
            first_stationary_d = d
            found_stationary = True
    
    # Find optimal d: minimum d that achieves stationarity
    stationary_mask = adf_pvals < significance
    if np.any(stationary_mask):
        # Among stationary options, pick the one with highest memory
        stationary_indices = np.where(stationary_mask)[0]
        best_idx = stationary_indices[np.argmax(memory_scores[stationary_indices])]
        optimal_d = d_values[best_idx]
    else:
        # No stationary solution found, use d=1
        optimal_d = 1.0
    
    return OptimalDResult(
        optimal_d=optimal_d,
        d_values_tested=d_values,
        adf_statistics=adf_stats,
        adf_pvalues=adf_pvals,
        correlations=correlations,
        memory_scores=memory_scores,
        first_stationary_d=first_stationary_d
    )


# ============================================================================
# FRACTIONAL DIFFERENTIATOR CLASS
# ============================================================================

class FractionalDifferentiator:
    """
    Fractional Differentiation for Financial Time Series.
    
    Preserves memory while achieving stationarity, following
    Marcos López de Prado's methodology.
    
    Key Benefits:
    - Preserves predictive information (long memory)
    - Achieves stationarity for ML models
    - Optimal d balances both objectives
    
    Usage:
        fd = FractionalDifferentiator()
        result = fd.fit_transform(prices)
        
        # Or with automatic d selection
        fd = FractionalDifferentiator(auto_d=True)
        result = fd.fit_transform(prices)
    """
    
    def __init__(self, 
                 d: Optional[float] = None,
                 auto_d: bool = True,
                 threshold: float = 1e-5,
                 significance: float = 0.05):
        """
        Initialize differentiator.
        
        Args:
            d: Fixed differentiation order (if not auto)
            auto_d: Automatically find optimal d
            threshold: Weight cutoff threshold
            significance: ADF significance level for auto_d
        """
        self.d = d
        self.auto_d = auto_d
        self.threshold = threshold
        self.significance = significance
        
        self.optimal_d_result: Optional[OptimalDResult] = None
        self.weights: Optional[np.ndarray] = None
        self._fitted = False
    
    def fit(self, series: np.ndarray) -> 'FractionalDifferentiator':
        """
        Fit the differentiator (find optimal d if auto_d=True).
        
        Args:
            series: Input time series
            
        Returns:
            self
        """
        if self.auto_d:
            self.optimal_d_result = find_optimal_d(
                series, 
                significance=self.significance,
                threshold=self.threshold
            )
            self.d = self.optimal_d_result.optimal_d
        
        # Compute weights
        self.weights = get_weights_ffd(self.d, self.threshold)
        self._fitted = True
        
        return self
    
    def transform(self, series: np.ndarray) -> FracDiffResult:
        """
        Apply fractional differentiation.
        
        Args:
            series: Input time series
            
        Returns:
            FracDiffResult with differenced series and diagnostics
        """
        if not self._fitted:
            self.fit(series)
        
        # Apply differentiation
        diff_series = frac_diff(series, self.d, self.threshold)
        
        # Compute diagnostics
        valid_idx = ~np.isnan(diff_series)
        diff_valid = diff_series[valid_idx]
        
        adf_stat, adf_pval = adf_test(diff_valid)
        
        # Correlation with original
        orig_valid = series[valid_idx]
        min_len = min(len(diff_valid), len(orig_valid))
        corr = np.corrcoef(diff_valid[-min_len:], orig_valid[-min_len:])[0, 1]
        
        # Memory preservation (1 - d gives rough estimate)
        memory = 1 - self.d
        
        return FracDiffResult(
            series=diff_series,
            d=self.d,
            weights=self.weights,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pval,
            is_stationary=adf_pval < self.significance,
            memory_preserved=memory,
            correlation_with_original=corr if not np.isnan(corr) else 0
        )
    
    def fit_transform(self, series: np.ndarray) -> FracDiffResult:
        """Fit and transform in one step."""
        self.fit(series)
        return self.transform(series)
    
    def generate_report(self) -> str:
        """Generate diagnostics report."""
        if not self._fitted:
            return "Differentiator not fitted yet."
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              FRACTIONAL DIFFERENTIATION REPORT                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 DIFFERENTIATION PARAMETERS                                               ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  Optimal d:           {self.d:.4f}                                           ║
║  Weight Threshold:    {self.threshold:.0e}                                   ║
║  Number of Weights:   {len(self.weights)}                                    ║
║                                                                              ║
║  📈 INTERPRETATION                                                           ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  d = 0.0: Original series (non-stationary, full memory)                      ║
║  d = 0.5: Half-differenced (balanced)                                        ║
║  d = 1.0: First difference (stationary, no memory)                           ║
║                                                                              ║
║  Current d = {self.d:.2f}: {'Preserves ' + str(int((1-self.d)*100)) + '% of memory'}                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report


# ============================================================================
# FEATURE ENGINEERING WITH FRAC DIFF
# ============================================================================

class FracDiffFeatureEngine:
    """
    Feature engineering using fractional differentiation.
    
    Applies optimal fractional differentiation to multiple features
    while preserving memory for ML models.
    """
    
    def __init__(self, 
                 features: List[str] = None,
                 auto_d: bool = True,
                 significance: float = 0.05):
        """
        Initialize feature engine.
        
        Args:
            features: List of feature names to transform
            auto_d: Automatically find optimal d for each feature
            significance: ADF significance level
        """
        self.features = features or []
        self.auto_d = auto_d
        self.significance = significance
        
        self.differentiators: Dict[str, FractionalDifferentiator] = {}
        self.optimal_ds: Dict[str, float] = {}
    
    def fit_transform_dict(self, data: Dict[str, np.ndarray]) -> Dict[str, FracDiffResult]:
        """
        Fit and transform multiple features.
        
        Args:
            data: Dictionary of feature_name -> series
            
        Returns:
            Dictionary of feature_name -> FracDiffResult
        """
        results = {}
        
        for name, series in data.items():
            fd = FractionalDifferentiator(auto_d=self.auto_d, significance=self.significance)
            result = fd.fit_transform(series)
            
            self.differentiators[name] = fd
            self.optimal_ds[name] = fd.d
            results[name] = result
        
        return results
    
    def get_optimal_ds(self) -> Dict[str, float]:
        """Get optimal d values for all features."""
        return self.optimal_ds
    
    def generate_summary(self) -> str:
        """Generate summary of all features."""
        summary = "\n📊 FRACTIONAL DIFFERENTIATION SUMMARY\n"
        summary += "=" * 50 + "\n"
        
        for name, d in self.optimal_ds.items():
            memory = int((1 - d) * 100)
            summary += f"  {name}: d={d:.3f} (memory: {memory}%)\n"
        
        return summary
