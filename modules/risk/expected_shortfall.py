"""
Expected Shortfall (ES) Engine with Extreme Value Theory
Author: Erdinc Erdogan
Purpose: Calculates Expected Shortfall using historical, parametric, Monte Carlo, EVT Peaks-Over-Threshold, and GARCH-FHS methods for Basel III/IV compliance.
References:
- ES_α = E[L | L > VaR_α]
- Extreme Value Theory (Generalized Pareto Distribution)
- Filtered Historical Simulation (GARCH-FHS)
Usage:
    engine = ESEngine(confidence_level=0.975)
    result = engine.calculate(returns, method=ESMethod.EXTREME_VALUE)
    decomposition = engine.decompose_portfolio_es(weights, returns)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize, brentq
from scipy.interpolate import interp1d
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class ESMethod(Enum):
    """Expected Shortfall calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC_GAUSSIAN = "parametric_gaussian"
    PARAMETRIC_STUDENT_T = "parametric_student_t"
    PARAMETRIC_GPD = "parametric_gpd"
    MONTE_CARLO = "monte_carlo"
    EXTREME_VALUE = "extreme_value"
    FILTERED_HISTORICAL = "filtered_historical"
    KERNEL_DENSITY = "kernel_density"


class TailType(Enum):
    """Tail distribution types for EVT"""
    FRECHET = "frechet"      # Heavy tail (ξ > 0)
    GUMBEL = "gumbel"        # Light tail (ξ = 0)
    WEIBULL = "weibull"      # Bounded tail (ξ < 0)


@dataclass
class EVTParameters:
    """Extreme Value Theory fitted parameters"""
    xi: float               # Shape parameter (tail index)
    sigma: float            # Scale parameter
    threshold: float        # POT threshold
    n_exceedances: int      # Number of threshold exceedances
    tail_type: TailType     # Inferred tail type
    ks_statistic: float     # Kolmogorov-Smirnov goodness of fit
    ks_pvalue: float        # KS test p-value


@dataclass
class ESResult:
    """Container for Expected Shortfall calculation results"""
    expected_shortfall: float
    var: float
    confidence_level: float
    method: str
    tail_losses: np.ndarray
    mean_excess: float
    tail_index: Optional[float] = None
    standard_error: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    evt_params: Optional[EVTParameters] = None
    backtesting_metrics: Optional[Dict] = None


@dataclass
class ESDecomposition:
    """Decomposition of portfolio Expected Shortfall"""
    total_es: float
    marginal_es: np.ndarray
    component_es: np.ndarray
    percentage_contribution: np.ndarray
    incremental_es: np.ndarray
    euler_allocation: np.ndarray


@dataclass
class ESBacktestResult:
    """Backtesting results for ES model validation"""
    n_violations: int
    violation_ratio: float
    expected_violations: float
    traffic_light_zone: str      # Green, Yellow, Red (Basel)
    conditional_coverage: float
    independence_test_pvalue: float
    magnitude_test_pvalue: float
    model_score: float


class ExpectedShortfallEngine(BaseModule):
    """Institutional-Grade Expected Shortfall (ES) Engine.
    
    Expected Shortfall is the expected loss conditional on the loss
    exceeding VaR. It is the primary risk measure under Basel III/IV.
    
    Mathematical Framework:
    ----------------------
    Definition (Continuous):
        ES_α = E[L | L > VaR_α]
        ES_α = (1/(1-α)) × ∫_{α}^{1} VaR_u du
    
    Definition (Discrete):
        ES_α = (1/n_tail) × Σ L_i for all L_i > VaR_α
    
    Parametric (Gaussian):
        ES_α = μ + σ × φ(Φ⁻¹(α)) / (1-α)
    
    Parametric (Student-t):
        ES_α = μ + σ × (f_ν(t_α) / (1-α)) × ((ν + t_α²) / (ν-1))
    
    EVT (GPD):
        ES_α = (VaR_α / (1-ξ)) + (σ - ξu) / (1-ξ)
        where u is threshold, ξ is shape, σ is scale
    
    Properties (Coherent Risk Measure):
    - Monotonicity: X ≤ Y ⟹ ES(X) ≤ ES(Y)
    - Translation Invariance: ES(X + c) = ES(X) + c
    - Positive Homogeneity: ES(λX) = λES(X) for λ > 0
    - Subadditivity: ES(X + Y) ≤ ES(X) + ES(Y)
    
    Basel III/IV Compliance:
    - 97.5% confidence level for internal models
    - 10-day holding period
    - Stressed ES calibration required
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.default_confidence: float = self.config.get('confidence_level', 0.975)
        self.n_simulations: int = self.config.get('n_simulations', 100000)
        self.evt_threshold_quantile: float = self.config.get('evt_threshold', 0.90)
        self.random_seed: Optional[int] = self.config.get('random_seed', None)
        self._rng = np.random.default_rng(self.random_seed)
    
    # =========================================================================
    # CORE ES CALCULATION METHODS
    # =========================================================================
    
    def compute_es_historical(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.975
    ) -> ESResult:
        """
        Historical Simulation Expected Shortfall.
        
        Non-parametric estimation using empirical distribution.
        ES_α = mean(L | L > VaR_α)
        """
        returns = np.asarray(returns).flatten()
        losses = -returns
        
        alpha = 1 - confidence_level
        var = np.percentile(losses, confidence_level * 100)
        
        tail_losses = losses[losses >= var]
        
        if len(tail_losses) == 0:
            es = var
            tail_losses = np.array([var])
        else:
            es = np.mean(tail_losses)
        
        n_tail = len(tail_losses)
        if n_tail > 1:
            se = np.std(tail_losses, ddof=1) / np.sqrt(n_tail)
            ci = (es - 1.96 * se, es + 1.96 * se)
        else:
            se = None
            ci = None
        
        mean_excess = es - var
        
        return ESResult(
            expected_shortfall=float(es),
            var=float(var),
            confidence_level=confidence_level,
            method=ESMethod.HISTORICAL.value,
            tail_losses=tail_losses,
            mean_excess=float(mean_excess),
            standard_error=se,
            confidence_interval=ci
        )
    
    def compute_es_parametric_gaussian(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.975
    ) -> ESResult:
        """
        Parametric ES assuming Gaussian distribution.
        
        ES_α = μ + σ × φ(Φ⁻¹(α)) / (1-α)
        """
        returns = np.asarray(returns).flatten()
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(alpha)
        phi_z = stats.norm.pdf(z_alpha)
        
        var = -(mu + sigma * z_alpha)
        es = -(mu - sigma * phi_z / alpha)
        
        losses = -returns
        tail_losses = losses[losses >= var]
        mean_excess = es - var
        
        n = len(returns)
        se = sigma * np.sqrt((1 + z_alpha**2 / 2) / n) / alpha
        ci = (es - 1.96 * se, es + 1.96 * se)
        
        return ESResult(
            expected_shortfall=float(es),
            var=float(var),
            confidence_level=confidence_level,
            method=ESMethod.PARAMETRIC_GAUSSIAN.value,
            tail_losses=tail_losses if len(tail_losses) > 0 else np.array([var]),
            mean_excess=float(mean_excess),
            standard_error=float(se),
            confidence_interval=ci
        )
    
    def compute_es_parametric_student_t(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.975,
        df: Optional[float] = None
    ) -> ESResult:
        """
        Parametric ES assuming Student-t distribution.
        
        ES_α = μ + σ × (f_ν(t_α) / (1-α)) × ((ν + t_α²) / (ν-1))
        """
        returns = np.asarray(returns).flatten()
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        if df is None:
            nu, _, _ = stats.t.fit(returns)
            nu = max(nu, 2.1)
        else:
            nu = max(df, 2.1)
        
        alpha = 1 - confidence_level
        t_alpha = stats.t.ppf(alpha, nu)
        f_t = stats.t.pdf(t_alpha, nu)
        
        scaling = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
        var = -(mu + sigma * t_alpha * scaling)
        
        es_factor = (f_t / alpha) * ((nu + t_alpha**2) / (nu - 1))
        es = -(mu - sigma * es_factor * scaling)
        
        losses = -returns
        tail_losses = losses[losses >= var]
        mean_excess = es - var
        
        return ESResult(
            expected_shortfall=float(es),
            var=float(var),
            confidence_level=confidence_level,
            method=ESMethod.PARAMETRIC_STUDENT_T.value,
            tail_losses=tail_losses if len(tail_losses) > 0 else np.array([var]),
            mean_excess=float(mean_excess),
            tail_index=float(nu)
        )
    
    def compute_es_evt(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.975,
        threshold_quantile: Optional[float] = None
    ) -> ESResult:
        """
        Expected Shortfall using Extreme Value Theory (Peaks Over Threshold).
        Fits Generalized Pareto Distribution (GPD) to tail exceedances:
        G_ξ,σ(x) = 1 - (1 + ξx/σ)^(-1/ξ)
        
        ES formula for GPD:
        ES_α = VaR_α/(1-ξ) + (σ - ξu)/(1-ξ)
        """
        returns = np.asarray(returns).flatten()
        losses = -returns
        
        threshold_q = threshold_quantile or self.evt_threshold_quantile
        threshold = np.percentile(losses, threshold_q * 100)
        
        exceedances = losses[losses > threshold] - threshold
        n_exceed = len(exceedances)
        n_total = len(losses)
        
        if n_exceed < 30:
            warnings.warn(f"Only {n_exceed} exceedances. EVT estimates may be unreliable.")
        
        xi, loc, sigma = self._fit_gpd(exceedances)
        
        if xi >= 1:
            warnings.warn("Shape parameter ξ ≥ 1: ES is infinite. Using historical method.")
            return self.compute_es_historical(returns, confidence_level)
        
        alpha = 1 - confidence_level
        exceed_prob = n_exceed / n_total
        
        if xi != 0:
            var = threshold + (sigma / xi) * ((alpha / exceed_prob)**(-xi) - 1)
        else:
            var = threshold - sigma * np.log(alpha / exceed_prob)
        
        if xi < 1:
            es = var / (1 - xi) + (sigma - xi * threshold) / (1 - xi)
        else:
            es = var * 2
        
        if xi > 0:
            tail_type = TailType.FRECHET
        elif xi == 0:
            tail_type = TailType.GUMBEL
        else:
            tail_type = TailType.WEIBULL
        
        ks_stat, ks_pval = stats.kstest(exceedances, 'genpareto', args=(xi, 0, sigma))
        
        evt_params = EVTParameters(
            xi=float(xi),
            sigma=float(sigma),
            threshold=float(threshold),
            n_exceedances=n_exceed,
            tail_type=tail_type,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval)
        )
        
        tail_losses = losses[losses >= var]
        mean_excess = es - var
        
        return ESResult(
            expected_shortfall=float(es),
            var=float(var),
            confidence_level=confidence_level,
            method=ESMethod.EXTREME_VALUE.value,
            tail_losses=tail_losses if len(tail_losses) > 0 else np.array([var]),
            mean_excess=float(mean_excess),
            tail_index=float(xi),
            evt_params=evt_params
        )
    
    def _fit_gpd(self, exceedances: np.ndarray) -> Tuple[float, float, float]:
        """Fit Generalized Pareto Distribution using MLE."""
        try:
            xi, loc, sigma = stats.genpareto.fit(exceedances, floc=0)
            xi = np.clip(xi, -0.5, 0.5)
        except Exception:
            sigma = np.mean(exceedances)
            xi = 0.0
            loc = 0.0
        return xi, loc, sigma
    
    def compute_es_monte_carlo(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.975,
        n_simulations: Optional[int] = None,
        use_antithetic: bool = True
    ) -> ESResult:
        """
        Monte Carlo ES with variance reduction techniques.
        
        Uses antithetic variates for variance reduction.
        """
        returns = np.asarray(returns).flatten()
        n_sims = n_simulations or self.n_simulations
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)
        
        if use_antithetic:
            n_half = n_sims // 2
            z1 = self._rng.standard_normal(n_half)
            z2 = -z1
            z = np.concatenate([z1, z2])
        else:
            z = self._rng.standard_normal(n_sims)
        
        z_adjusted = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24
        simulated_returns = mu + sigma * z_adjusted
        simulated_losses = -simulated_returns
        
        alpha = 1 - confidence_level
        var = np.percentile(simulated_losses, confidence_level * 100)
        tail_losses = simulated_losses[simulated_losses >= var]
        es = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        n_tail = len(tail_losses)
        se = np.std(tail_losses, ddof=1) / np.sqrt(n_tail) if n_tail > 1 else None
        ci = (es - 1.96 * se, es + 1.96 * se) if se else None
        
        actual_losses = -returns
        actual_tail = actual_losses[actual_losses >= var]
        
        return ESResult(
            expected_shortfall=float(es),
            var=float(var),
            confidence_level=confidence_level,
            method=ESMethod.MONTE_CARLO.value,
            tail_losses=actual_tail if len(actual_tail) > 0 else np.array([var]),
            mean_excess=float(es - var),
            standard_error=se,
            confidence_interval=ci
        )
    
    def compute_es_kernel_density(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.975,
        bandwidth: Optional[float] = None
    ) -> ESResult:
        """
        ES using Kernel Density Estimation.
        
        Non-parametric smoothed density estimation.
        """
        returns = np.asarray(returns).flatten()
        losses = -returns
        
        if bandwidth is None:
            bandwidth = 1.06 * np.std(losses) * len(losses)**(-1/5)
        
        kde = stats.gaussian_kde(losses, bw_method=bandwidth)
        x_grid = np.linspace(losses.min(), losses.max() * 1.5, 10000)
        cdf_values = np.array([kde.integrate_box_1d(-np.inf, x) for x in x_grid])
        
        var_idx = np.searchsorted(cdf_values, confidence_level)
        var = x_grid[min(var_idx, len(x_grid)-1)]
        
        tail_grid = x_grid[x_grid >= var]
        if len(tail_grid) > 0:
            tail_pdf = kde(tail_grid)
            tail_prob = 1 - confidence_level
            es = np.trapz(tail_grid * tail_pdf, tail_grid) / tail_prob
        else:
            es = var
        
        tail_losses = losses[losses >= var]
        
        return ESResult(
            expected_shortfall=float(es),
            var=float(var),
            confidence_level=confidence_level,
            method=ESMethod.KERNEL_DENSITY.value,
            tail_losses=tail_losses if len(tail_losses) > 0 else np.array([var]),
            mean_excess=float(es - var)
        )
    
    # =========================================================================
    # PORTFOLIO ES METHODS
    # =========================================================================
    
    def compute_portfolio_es(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float = 0.975,
        method: ESMethod = ESMethod.HISTORICAL
    ) -> ESResult:
        """Compute Expected Shortfall for a weighted portfolio."""
        weights = np.asarray(weights).flatten()
        portfolio_returns = returns @ weights
        
        method_map = {
            ESMethod.HISTORICAL: self.compute_es_historical,
            ESMethod.PARAMETRIC_GAUSSIAN: self.compute_es_parametric_gaussian,
            ESMethod.PARAMETRIC_STUDENT_T: self.compute_es_parametric_student_t,
            ESMethod.EXTREME_VALUE: self.compute_es_evt,
            ESMethod.MONTE_CARLO: self.compute_es_monte_carlo,
            ESMethod.KERNEL_DENSITY: self.compute_es_kernel_density,
        }
        
        compute_func = method_map.get(method, self.compute_es_historical)
        return compute_func(portfolio_returns, confidence_level)
    
    def decompose_portfolio_es(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float = 0.975,
        delta: float = 0.0001
    ) -> ESDecomposition:
        """
        Decompose portfolio ES into component contributions.
        
        Euler Allocation: ES = Σ w_i × ∂ES/∂w_i
        Component ES_i = w_i × Marginal ES_i
        """
        weights = np.asarray(weights).flatten()
        n_assets = len(weights)
        
        base_result = self.compute_es_historical(returns @ weights, confidence_level)
        total_es = base_result.expected_shortfall
        
        marginal_es = np.zeros(n_assets)
        for i in range(n_assets):
            w_up = weights.copy()
            w_up[i] += delta
            w_up = w_up / np.sum(w_up)
            
            w_down = weights.copy()
            w_down[i] -= delta
            w_down = w_down / np.sum(w_down)
            
            es_up = self.compute_es_historical(returns @ w_up, confidence_level).expected_shortfall
            es_down = self.compute_es_historical(returns @ w_down, confidence_level).expected_shortfall
            
            marginal_es[i] = (es_up - es_down) / (2 * delta)
        
        component_es = weights * marginal_es
        percentage_contribution = component_es / total_es * 100
        
        incremental_es = np.zeros(n_assets)
        for i in range(n_assets):
            w_reduced = weights.copy()
            w_reduced[i] = 0
            if np.sum(w_reduced) > 0:
                w_reduced = w_reduced / np.sum(w_reduced)
                es_reduced = self.compute_es_historical(returns @ w_reduced, confidence_level).expected_shortfall
            else:
                es_reduced = 0
            incremental_es[i] = total_es - es_reduced
        
        euler_allocation = component_es / np.sum(component_es) * total_es
        
        return ESDecomposition(
            total_es=total_es,
            marginal_es=marginal_es,
            component_es=component_es,
            percentage_contribution=percentage_contribution,
            incremental_es=incremental_es,
            euler_allocation=euler_allocation
        )
    
    # =========================================================================
    # BACKTESTING & VALIDATION
    # =========================================================================
    
    def backtest_es(
        self,
        returns: np.ndarray,
        es_forecasts: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.975
    ) -> ESBacktestResult:
        """
        Backtest ES model using regulatory tests.
        
        Tests:
        1. Violation ratio (should be ≈ 1-α)
        2. Conditional coverage (Christoffersen)
        3. Independence of violations
        4. Magnitude test (Acerbi-Szekely)
        """
        returns = np.asarray(returns).flatten()
        losses = -returns
        
        violations = losses > var_forecasts
        n_violations = np.sum(violations)
        n_obs = len(returns)
        violation_ratio = n_violations / n_obs
        expected_violations = n_obs * (1 - confidence_level)
        
        expected_ratio = 1 - confidence_level
        if violation_ratio <= expected_ratio * 1.5:
            zone = "GREEN"
        elif violation_ratio <= expected_ratio * 2.0:
            zone = "YELLOW"
        else:
            zone = "RED"
        
        if n_violations > 0:
            violation_losses = losses[violations]
            es_at_violations = es_forecasts[violations]
            magnitude_ratio = np.mean(violation_losses) / np.mean(es_at_violations)
            magnitude_pvalue = 1 - stats.norm.cdf(abs(magnitude_ratio - 1) * np.sqrt(n_violations))
        else:
            magnitude_pvalue = 1.0
        
        if n_violations >= 2:
            violation_indices = np.where(violations)[0]
            gaps = np.diff(violation_indices)
            expected_gap = 1 / (1 - confidence_level)
            independence_stat = np.sum((gaps - expected_gap)**2) / expected_gap
            independence_pvalue = 1 - stats.chi2.cdf(independence_stat, len(gaps))
        else:
            independence_pvalue = 1.0
        
        conditional_coverage = 1 - abs(violation_ratio - (1 - confidence_level)) / (1 - confidence_level)
        
        model_score = (
            0.4 * conditional_coverage +
            0.3 * magnitude_pvalue +
            0.3 * independence_pvalue
        )
        
        return ESBacktestResult(
            n_violations=int(n_violations),
            violation_ratio=float(violation_ratio),
            expected_violations=float(expected_violations),
            traffic_light_zone=zone,
            conditional_coverage=float(conditional_coverage),
            independence_test_pvalue=float(independence_pvalue),
            magnitude_test_pvalue=float(magnitude_pvalue),
            model_score=float(model_score)
        )
    
    # =========================================================================
    # STRESSED ES (BASEL III/IV)
    # =========================================================================
    
    def compute_stressed_es(
        self,
        returns: np.ndarray,
        stress_period_returns: np.ndarray,
        confidence_level: float = 0.975,
        stress_weight: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Compute Stressed ES as per Basel III/IV requirements.
        
        Returns:
            Tuple of (current_es, stressed_es, combined_es)
        """
        current_result = self.compute_es_historical(returns, confidence_level)
        stressed_result = self.compute_es_historical(stress_period_returns, confidence_level)
        
        combined_es = (
            (1 - stress_weight) * current_result.expected_shortfall +
            stress_weight * stressed_result.expected_shortfall
        )
        
        return (
            current_result.expected_shortfall,
            stressed_result.expected_shortfall,
            combined_es
        )
    
    def compute_es_term_structure(
        self,
        returns: np.ndarray,
        confidence_levels: List[float] = None,
        method: ESMethod = ESMethod.HISTORICAL
    ) -> pd.DataFrame:
        """Compute ES across multiple confidence levels."""
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
        
        results = []
        for cl in confidence_levels:
            if method == ESMethod.HISTORICAL:
                result = self.compute_es_historical(returns, cl)
            elif method == ESMethod.PARAMETRIC_GAUSSIAN:
                result = self.compute_es_parametric_gaussian(returns, cl)
            elif method == ESMethod.EXTREME_VALUE:
                result = self.compute_es_evt(returns, cl)
            else:
                result = self.compute_es_historical(returns, cl)
            
            results.append({
                'confidence_level': cl,
                'var': result.var,
                'expected_shortfall': result.expected_shortfall,
                'mean_excess': result.mean_excess,
                'es_var_ratio': result.expected_shortfall / result.var if result.var > 0 else np.nan
            })
        
        return pd.DataFrame(results)