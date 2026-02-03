"""
Value at Risk (VaR) Models Engine
Author: Erdinc Erdogan
Purpose: Implements 10 VaR models including Historical, EWMA, GARCH, Cornish-Fisher, EVT, Monte Carlo, and Filtered Historical Simulation with backtesting.
References:
- Historical Simulation (HS)
- EWMA (RiskMetrics, 1996)
- GARCH(1,1) (Bollerslev, 1986)
- Extreme Value Theory (EVT)
Usage:
    engine = VaRModelsEngine(returns)
    result = engine.calculate(confidence=0.99, model=VaRModel.GARCH)
    backtest = engine.backtest(var_series, realized_returns)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class VaRModel(Enum):
    """Available VaR calculation models"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    EWMA = "ewma"
    GARCH = "garch"
    CORNISH_FISHER = "cornish_fisher"
    EVT = "evt"
    MONTE_CARLO = "monte_carlo"
    FILTERED_HISTORICAL = "filtered_historical"
    WEIGHTED_HISTORICAL = "weighted_historical"
    HULL_WHITE = "hull_white"


class HoldingPeriodScaling(Enum):
    """Methods for scaling VaR to different holding periods"""
    SQUARE_ROOT = "square_root"           # √T scaling (IID assumption)
    ACTUAL = "actual"                      # Use actual multi-day returns
    AUTOCORRELATION_ADJUSTED = "autocorr"  # Adjust for serial correlation


@dataclass
class VaRResult:
    """Container for VaR calculation results"""
    var: float
    confidence_level: float
    holding_period: int
    model: str
    volatility: float
    returns_used: int
    timestamp: Optional[str] = None
    model_params: Optional[Dict] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class VaRBacktestResult:
    """Backtesting results for VaR model validation"""
    n_observations: int
    n_violations: int
    violation_ratio: float
    expected_violations: float
    kupiec_statistic: float
    kupiec_pvalue: float
    christoffersen_statistic: float
    christoffersen_pvalue: float
    traffic_light_zone: str
    model_rejected: bool
    violation_dates: Optional[List] = None


@dataclass
class GARCHParams:
    """GARCH(1,1) model parameters"""
    omega: float      # Constant term
    alpha: float      # ARCH coefficient (lagged squared return)
    beta: float       # GARCH coefficient (lagged variance)
    persistence: float  # α + β (should be < 1 for stationarity)
    unconditional_var: float  # ω / (1 - α - β)
    log_likelihood: float


@dataclass
class EWMAParams:
    """EWMA model parameters"""
    lambda_param: float  # Decay factor (typically 0.94 for daily)
    half_life: float     # ln(0.5) / ln(λ)
    effective_window: int  # Approximate number of effective observations


class VaRModelsEngine(BaseModule):
    """
    Institutional-Grade Value at Risk (VaR) Models Engine.
    
    Implements multiple VaR methodologies for comprehensive risk measurement.
    
    Mathematical Framework:
    ----------------------
    VaR Definition:
        P(L > VaR_α) = 1 - α
        VaR_α = inf{x : P(L ≤ x) ≥ α}
    
    Historical Simulation:
        VaR_α = -Quantile(returns, 1-α)
    
    Parametric (Gaussian):
        VaR_α = μ + σ × Φ⁻¹(1-α)
    
    EWMA (RiskMetrics):
        σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}
        VaR_α = σ_t × Φ⁻¹(α)
    
    GARCH(1,1):
        σ²_t = ω + αr²_{t-1} + βσ²_{t-1}
        VaR_α = σ_t × Φ⁻¹(α)
    
    Cornish-Fisher:
        z_CF = z + (z²-1)×S/6 + (z³-3z)×K/24 - (2z³-5z)×S²/36
        VaR_α = μ + σ × z_CF
    
    Holding Period Scaling:
        VaR_T = VaR_1 × √T  (under IID assumption)
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.default_confidence: float = self.config.get('confidence_level', 0.99)
        self.default_holding_period: int = self.config.get('holding_period', 1)
        self.ewma_lambda: float = self.config.get('ewma_lambda', 0.94)
        self.n_simulations: int = self.config.get('n_simulations', 10000)
        self.random_seed: Optional[int] = self.config.get('random_seed', None)
        self._rng = np.random.default_rng(self.random_seed)
        self._garch_params: Optional[GARCHParams] = None
    
    # =========================================================================
    # HISTORICAL SIMULATION METHODS
    # =========================================================================
    
    def compute_var_historical(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Historical Simulation VaR.
        
        Non-parametric method using empirical distribution.
        VaR_α = -Quantile(returns, 1-α)
        """
        returns = np.asarray(returns).flatten()
        
        if holding_period > 1:
            multi_period_returns = self._compute_multi_period_returns(returns, holding_period)
        else:
            multi_period_returns = returns
        
        alpha = 1 - confidence_level
        var = -np.percentile(multi_period_returns, alpha * 100)
        volatility = np.std(returns, ddof=1)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.HISTORICAL.value,
            volatility=float(volatility),
            returns_used=len(returns)
        )
    
    def compute_var_weighted_historical(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        decay_factor: float = 0.98,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Weighted Historical Simulation (WHS) VaR.
        
        More recent observations receive higher weights.
        w_i = λ^(n-i) × (1-λ) / (1-λ^n)
        """
        returns = np.asarray(returns).flatten()
        n = len(returns)
        
        weights = np.array([decay_factor**(n-1-i) for i in range(n)])
        weights = weights / np.sum(weights)
        
        sorted_indices = np.argsort(returns)
        sorted_returns = returns[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumulative_weights = np.cumsum(sorted_weights)
        alpha = 1 - confidence_level
        var_idx = np.searchsorted(cumulative_weights, alpha)
        var = -sorted_returns[min(var_idx, n-1)]
        
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        volatility = np.std(returns, ddof=1)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.WEIGHTED_HISTORICAL.value,
            volatility=float(volatility),
            returns_used=n,
            model_params={'decay_factor': decay_factor}
        )
    
    def compute_var_hull_white(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Hull-White Age-Weighted Historical Simulation.
        
        Combines volatility updating with historical simulation.
        Scales historical returns by ratio of current to historical volatility.
        """
        returns = np.asarray(returns).flatten()
        n = len(returns)
        
        ewma_vol = self._compute_ewma_volatility(returns)
        current_vol = ewma_vol[-1]
        
        historical_vols = np.zeros(n)
        for i in range(n):
            window = returns[max(0, i-20):i+1]
            historical_vols[i] = np.std(window, ddof=1) if len(window) > 1 else current_vol
        
        historical_vols = np.maximum(historical_vols, 1e-10)
        scaled_returns = returns * (current_vol / historical_vols)
        
        alpha = 1 - confidence_level
        var = -np.percentile(scaled_returns, alpha * 100)
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.HULL_WHITE.value,
            volatility=float(current_vol),
            returns_used=n,
            model_params={'current_volatility': float(current_vol)}
        )
    
    # =========================================================================
    # PARAMETRIC METHODS
    # =========================================================================
    
    def compute_var_parametric(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1,
        distribution: str = "gaussian"
    ) -> VaRResult:
        """
        Parametric (Variance-Covariance) VaR.
        
        Gaussian: VaR_α = μ + σ × Φ⁻¹(1-α)
        Student-t: VaR_α = μ + σ × t_ν⁻¹(1-α) × √((ν-2)/ν)
        """
        returns = np.asarray(returns).flatten()
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        alpha = 1 - confidence_level
        if distribution == "gaussian":
            z_score = stats.norm.ppf(alpha)
            var = -(mu + sigma * z_score)
        elif distribution == "student_t":
            nu, _, _ = stats.t.fit(returns)
            nu = max(nu, 2.1)
            t_score = stats.t.ppf(alpha, nu)
            scaling = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
            var = -(mu + sigma * t_score * scaling)
        else:
            z_score = stats.norm.ppf(alpha)
            var = -(mu + sigma * z_score)
        
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.PARAMETRIC.value,
            volatility=float(sigma),
            returns_used=len(returns),
            model_params={'distribution': distribution, 'mean': float(mu)}
        )
    
    def compute_var_cornish_fisher(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Cornish-Fisher VaR with skewness and kurtosis adjustment.
        
        z_CF = z + (z²-1)×S/6 + (z³-3z)×K/24 - (2z³-5z)×S²/36
        VaR_α = μ + σ × z_CF
        """
        returns = np.asarray(returns).flatten()
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)
        
        alpha = 1 - confidence_level
        z = stats.norm.ppf(alpha)
        
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = -(mu + sigma * z_cf)
        
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.CORNISH_FISHER.value,
            volatility=float(sigma),
            returns_used=len(returns),
            model_params={'skewness': float(skew), 'kurtosis': float(kurt)}
        )
    
    # =========================================================================
    # VOLATILITY MODEL METHODS (EWMA, GARCH)
    # =========================================================================
    
    def compute_var_ewma(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        lambda_param: Optional[float] = None,
        holding_period: int = 1
    ) -> VaRResult:
        """
        EWMA (Exponentially Weighted Moving Average) VaR.
        
        RiskMetrics methodology:
        σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}
        VaR_α = σ_t × Φ⁻¹(α)
        """
        returns = np.asarray(returns).flatten()
        lam = lambda_param or self.ewma_lambda
        
        ewma_vol = self._compute_ewma_volatility(returns, lam)
        current_vol = ewma_vol[-1]
        
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        var = -current_vol * z_score
        
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        half_life = np.log(0.5) / np.log(lam)
        
        ewma_params = EWMAParams(
            lambda_param=lam,
            half_life=float(half_life),
            effective_window=int(1 / (1 - lam))
        )
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.EWMA.value,
            volatility=float(current_vol),
            returns_used=len(returns),
            model_params={
                'lambda': lam,
                'half_life': float(half_life),
                'current_volatility': float(current_vol)
            }
        )
    
    def _compute_ewma_volatility(
        self,
        returns: np.ndarray,
        lambda_param: Optional[float] = None
    ) -> np.ndarray:
        """Compute EWMA volatility series."""
        lam = lambda_param or self.ewma_lambda
        n = len(returns)
        
        variance = np.zeros(n)
        variance[0] = returns[0]**2
        
        for t in range(1, n):
            variance[t] = lam * variance[t-1] + (1 - lam) * returns[t-1]**2
        
        return np.sqrt(variance)
    
    def compute_var_garch(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1
    ) -> VaRResult:
        """
        GARCH(1,1) VaR.
        
        σ²_t = ω + αr²_{t-1} + βσ²_{t-1}
        VaR_α = σ_t × Φ⁻¹(α)
        """
        returns = np.asarray(returns).flatten()
        
        garch_params = self._fit_garch(returns)
        self._garch_params = garch_params
        
        garch_vol = self._compute_garch_volatility(returns, garch_params)
        current_vol = garch_vol[-1]
        
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        var = -current_vol * z_score
        
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.GARCH.value,
            volatility=float(current_vol),
            returns_used=len(returns),
            model_params={
                'omega': garch_params.omega,
                'alpha': garch_params.alpha,
                'beta': garch_params.beta,
                'persistence': garch_params.persistence
            }
        )
    
    def _fit_garch(self, returns: np.ndarray) -> GARCHParams:
        """Fit GARCH(1,1) model using MLE."""
        returns = np.asarray(returns).flatten()
        T = len(returns)
        sample_var = np.var(returns, ddof=1)
        
        def neg_log_likelihood(params):
            omega, alpha, beta = params
            
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            sigma2 = np.zeros(T)
            sigma2[0] = sample_var
            
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
                sigma2[t] = max(sigma2[t], 1e-10)
            
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)
            return -ll
        
        initial_omega = sample_var * 0.05
        initial_alpha = 0.05
        initial_beta = 0.90
        bounds = [(1e-10, None), (1e-10, 0.5), (0.5, 0.9999)]
        constraints = {'type': 'ineq', 'fun': lambda x: 0.9999 - x[1] - x[2]}
        
        result = minimize(
            neg_log_likelihood,
            [initial_omega, initial_alpha, initial_beta],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        omega, alpha, beta = result.x
        persistence = alpha + beta
        unconditional_var = omega / (1 - persistence) if persistence < 1 else sample_var
        
        return GARCHParams(
            omega=float(omega),
            alpha=float(alpha),
            beta=float(beta),
            persistence=float(persistence),
            unconditional_var=float(unconditional_var),
            log_likelihood=float(-result.fun)
        )
    
    def _compute_garch_volatility(
        self,
        returns: np.ndarray,
        params: GARCHParams
    ) -> np.ndarray:
        """Compute GARCH(1,1) volatility series."""
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = params.unconditional_var
        
        for t in range(1, T):
            sigma2[t] = (params.omega +
                        params.alpha * returns[t-1]**2 +
                        params.beta * sigma2[t-1])
        return np.sqrt(sigma2)
    
    # =========================================================================
    # EXTREME VALUE THEORY
    # =========================================================================
    
    def compute_var_evt(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        threshold_quantile: float = 0.90,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Extreme Value Theory VaR using Peaks Over Threshold (POT).
        
        Fits Generalized Pareto Distribution to tail exceedances.
        VaR_α = u + (σ/ξ) × [(n/N_u × (1-α))^{-ξ} - 1]
        """
        returns = np.asarray(returns).flatten()
        losses = -returns
        
        threshold = np.percentile(losses, threshold_quantile * 100)
        exceedances = losses[losses > threshold] - threshold
        n_exceed = len(exceedances)
        n_total = len(losses)
        
        try:
            xi, _, sigma = stats.genpareto.fit(exceedances, floc=0)
            xi = np.clip(xi, -0.5, 0.5)
        except Exception:
            sigma = np.mean(exceedances)
            xi = 0.0
        
        alpha = 1 - confidence_level
        exceed_prob = n_exceed / n_total
        
        if xi != 0:
            var = threshold + (sigma / xi) * ((alpha / exceed_prob)**(-xi) - 1)
        else:
            var = threshold - sigma * np.log(alpha / exceed_prob)
        
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        volatility = np.std(returns, ddof=1)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.EVT.value,
            volatility=float(volatility),
            returns_used=len(returns),
            model_params={
                'xi': float(xi),
                'sigma': float(sigma),
                'threshold': float(threshold),
                'n_exceedances': n_exceed
            }
        )
    
    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================
    
    def compute_var_monte_carlo(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1,
        n_simulations: Optional[int] = None
    ) -> VaRResult:
        """
        Monte Carlo Simulation VaR.
        
        Simulates future returns from fitted distribution.
        """
        returns = np.asarray(returns).flatten()
        n_sims = n_simulations or self.n_simulations
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)
        
        z = self._rng.standard_normal(n_sims)
        z_adjusted = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24
        
        if holding_period > 1:
            simulated_returns = np.zeros(n_sims)
            for _ in range(holding_period):
                z = self._rng.standard_normal(n_sims)
                z_adjusted = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24
                simulated_returns += mu + sigma * z_adjusted
        else:
            simulated_returns = mu + sigma * z_adjusted
        
        alpha = 1 - confidence_level
        var = -np.percentile(simulated_returns, alpha * 100)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.MONTE_CARLO.value,
            volatility=float(sigma),
            returns_used=len(returns),
            model_params={'n_simulations': n_sims}
        )
    
    def compute_var_filtered_historical(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1
    ) -> VaRResult:
        """
        Filtered Historical Simulation (FHS) VaR.
        
        Combines GARCH volatility with historical simulation.
        Standardizes returns by GARCH volatility, then rescales.
        """
        returns = np.asarray(returns).flatten()
        
        garch_params = self._fit_garch(returns)
        garch_vol = self._compute_garch_volatility(returns, garch_params)
        current_vol = garch_vol[-1]
        
        garch_vol_safe = np.maximum(garch_vol, 1e-10)
        standardized_returns = returns / garch_vol_safe
        
        scaled_returns = standardized_returns * current_vol
        
        alpha = 1 - confidence_level
        var = -np.percentile(scaled_returns, alpha * 100)
        
        if holding_period > 1:
            var = var * np.sqrt(holding_period)
        
        return VaRResult(
            var=float(var),
            confidence_level=confidence_level,
            holding_period=holding_period,
            model=VaRModel.FILTERED_HISTORICAL.value,
            volatility=float(current_vol),
            returns_used=len(returns),
            model_params={
                'garch_omega': garch_params.omega,
                'garch_alpha': garch_params.alpha,
                'garch_beta': garch_params.beta
            }
        )
    
    # =========================================================================
    # PORTFOLIO VAR
    # =========================================================================
    
    def compute_portfolio_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        model: VaRModel = VaRModel.HISTORICAL,
        holding_period: int = 1
    ) -> VaRResult:
        """Compute VaR for a weighted portfolio."""
        weights = np.asarray(weights).flatten()
        portfolio_returns = returns @ weights
        
        model_map = {
            VaRModel.HISTORICAL: self.compute_var_historical,
            VaRModel.PARAMETRIC: self.compute_var_parametric,
            VaRModel.EWMA: self.compute_var_ewma,
            VaRModel.GARCH: self.compute_var_garch,
            VaRModel.CORNISH_FISHER: self.compute_var_cornish_fisher,
            VaRModel.EVT: self.compute_var_evt,
            VaRModel.MONTE_CARLO: self.compute_var_monte_carlo,
            VaRModel.FILTERED_HISTORICAL: self.compute_var_filtered_historical,}
        
        compute_func = model_map.get(model, self.compute_var_historical)
        return compute_func(portfolio_returns, confidence_level, holding_period)
    
    def compute_component_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        delta: float = 0.0001
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Component VaR decomposition.
        
        Returns:
            Tuple of (total_var, marginal_var, component_var)
        """
        weights = np.asarray(weights).flatten()
        n_assets = len(weights)
        
        base_result = self.compute_var_historical(returns @ weights, confidence_level)
        total_var = base_result.var
        
        marginal_var = np.zeros(n_assets)
        for i in range(n_assets):
            w_up = weights.copy()
            w_up[i] += delta
            w_up = w_up / np.sum(w_up)
            
            w_down = weights.copy()
            w_down[i] -= delta
            w_down = w_down / np.sum(w_down)
            
            var_up = self.compute_var_historical(returns @ w_up, confidence_level).var
            var_down = self.compute_var_historical(returns @ w_down, confidence_level).var
            
            marginal_var[i] = (var_up - var_down) / (2 * delta)
        
        component_var = weights * marginal_var
        
        return total_var, marginal_var, component_var
    
    # =========================================================================
    # BACKTESTING
    # =========================================================================
    
    def backtest_var(
        self,
        returns: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.99
    ) -> VaRBacktestResult:
        """
        Backtest VaR model using regulatory tests.
        
        Tests:
        1. Kupiec Test (Unconditional Coverage)
        2. Christoffersen Test (Independence)
        3. Basel Traffic Light System
        """
        returns = np.asarray(returns).flatten()
        losses = -returns
        
        violations = losses > var_forecasts
        n_violations = np.sum(violations)
        n_obs = len(returns)
        violation_ratio = n_violations / n_obs
        expected_ratio = 1 - confidence_level
        expected_violations = n_obs * expected_ratio
        
        p = expected_ratio
        if n_violations == 0:
            kupiec_stat = 2 * n_obs * np.log(1 - p)
        elif n_violations == n_obs:
            kupiec_stat = 2 * n_obs * np.log(p)
        else:
            kupiec_stat = 2 * (
                n_violations * np.log(violation_ratio / p) +
                (n_obs - n_violations) * np.log((1 - violation_ratio) / (1 - p))
            )
        kupiec_pvalue = 1 - stats.chi2.cdf(abs(kupiec_stat), 1)
        
        n_00, n_01, n_10, n_11 = 0, 0, 0, 0
        for t in range(1, n_obs):
            if not violations[t-1] and not violations[t]:
                n_00 += 1
            elif not violations[t-1] and violations[t]:
                n_01 += 1
            elif violations[t-1] and not violations[t]:
                n_10 += 1
            else:
                n_11 += 1
        
        if (n_00 + n_01) > 0 and (n_10 + n_11) > 0:
            pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
            pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
            pi = (n_01 + n_11) / (n_obs - 1)
            
            if 0 < pi < 1 and 0 < pi_01 < 1 and 0 < pi_11 < 1:
                ll_ind = (n_00 * np.log(1 - pi_01) + n_01 * np.log(pi_01) +
                         n_10 * np.log(1 - pi_11) + n_11 * np.log(pi_11))
                ll_null = (n_00 + n_10) * np.log(1 - pi) + (n_01 + n_11) * np.log(pi)
                christoffersen_stat = 2 * (ll_ind - ll_null)
            else:
                christoffersen_stat = 0
        else:
            christoffersen_stat = 0
        
        christoffersen_pvalue = 1 - stats.chi2.cdf(abs(christoffersen_stat), 1)
        
        if n_violations <= expected_violations * 1.5:
            zone = "GREEN"
        elif n_violations <= expected_violations * 2.0:
            zone = "YELLOW"
        else:
            zone = "RED"
        
        model_rejected = kupiec_pvalue < 0.05 or christoffersen_pvalue < 0.05
        
        violation_dates = list(np.where(violations)[0])
        
        return VaRBacktestResult(
            n_observations=n_obs,
            n_violations=int(n_violations),
            violation_ratio=float(violation_ratio),
            expected_violations=float(expected_violations),
            kupiec_statistic=float(kupiec_stat),
            kupiec_pvalue=float(kupiec_pvalue),
            christoffersen_statistic=float(christoffersen_stat),
            christoffersen_pvalue=float(christoffersen_pvalue),
            traffic_light_zone=zone,
            model_rejected=model_rejected,
            violation_dates=violation_dates
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _compute_multi_period_returns(
        self,
        returns: np.ndarray,
        holding_period: int
    ) -> np.ndarray:
        """Compute overlapping multi-period returns."""
        n = len(returns)
        if holding_period >= n:
            return np.array([np.sum(returns)])
        
        multi_period = np.zeros(n - holding_period + 1)
        for i in range(len(multi_period)):
            multi_period[i] = np.sum(returns[i:i + holding_period])
        
        return multi_period
    
    def compare_models(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
        holding_period: int = 1
    ) -> pd.DataFrame:
        """Compare VaR estimates across all models."""
        results = []
        
        models = [
            ('Historical', self.compute_var_historical),
            ('Parametric', self.compute_var_parametric),
            ('EWMA', self.compute_var_ewma),
            ('GARCH', self.compute_var_garch),
            ('Cornish-Fisher', self.compute_var_cornish_fisher),
            ('EVT', self.compute_var_evt),
            ('Monte Carlo', self.compute_var_monte_carlo),
            ('Filtered Historical', self.compute_var_filtered_historical),
            ('Weighted Historical', self.compute_var_weighted_historical),
            ('Hull-White', self.compute_var_hull_white),
        ]
        
        for name, func in models:
            try:
                result = func(returns, confidence_level, holding_period)
                results.append({
                    'Model': name,
                    'VaR': result.var,
                    'Volatility': result.volatility,
                    'VaR/Vol Ratio': result.var / result.volatility if result.volatility > 0 else np.nan
                })
            except Exception as e:
                results.append({
                    'Model': name,
                    'VaR': np.nan,
                    'Volatility': np.nan,
                    'VaR/Vol Ratio': np.nan
                })
        
        return pd.DataFrame(results)