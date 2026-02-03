"""
Conditional Value at Risk (CVaR) Engine
Author: Erdinc Erdogan
Purpose: Calculates CVaR/Expected Shortfall using historical, parametric (Gaussian/Student-t/Cornish-Fisher), Monte Carlo, and Filtered Historical Simulation methods.
References:
- CVaR_α = E[L | L > VaR_α]
- Coherent Risk Measures (Artzner et al., 1999)
- Basel III/IV Compliance
Usage:
    engine = CVaREngine(confidence_level=0.99)
    result = engine.calculate(returns, method=CVaRMethod.HISTORICAL)
    marginal = engine.marginal_cvar(portfolio_weights)
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


class CVaRMethod(Enum):
    """CVaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC_GAUSSIAN = "parametric_gaussian"
    PARAMETRIC_STUDENT_T = "parametric_student_t"
    CORNISH_FISHER = "cornish_fisher"
    MONTE_CARLO = "monte_carlo"
    FILTERED_HISTORICAL = "filtered_historical"


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    CORNISH_FISHER = "cornish_fisher"
    MONTE_CARLO = "monte_carlo"


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    max_drawdown: float
    volatility: float
    skewness: float
    kurtosis: float
    tail_ratio: float
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None


@dataclass
class CVaRResult:
    """Container for CVaR calculation results"""
    cvar: float
    var: float
    confidence_level: float
    method: str
    tail_losses: np.ndarray
    expected_tail_loss: float
    tail_probability: float
    marginal_cvar: Optional[np.ndarray] = None
    component_cvar: Optional[np.ndarray] = None
    incremental_cvar: Optional[Dict[int, float]] = None


@dataclass
class StressTestResult:
    """Container for stress test results"""
    scenario_name: str
    portfolio_loss: float
    var_breach: bool
    cvar_breach: bool
    worst_asset: int
    worst_asset_loss: float
    correlation_shift: float


class CVaREngine(BaseModule):
    """
    Institutional-Grade Conditional Value at Risk (CVaR) Engine.
    CVaR (Expected Shortfall) is a coherent risk measure that quantifies
    the expected loss given that the loss exceeds VaR.
    
    Mathematical Framework:
    ----------------------
    VaR_α = inf{x : P(L > x) ≤ 1 - α}
    CVaR_α = E[L | L > VaR_α] = (1/(1-α)) × ∫_{α}^{1} VaR_u du
    
    For continuous distributions:
    CVaR_α = E[L | L > VaR_α]
    
    Properties (Coherent Risk Measure):
    - Monotonicity: If X ≤ Y, then ρ(X) ≤ ρ(Y)
    - Translation Invariance: ρ(X + c) = ρ(X) + c
    - Positive Homogeneity: ρ(λX) = λρ(X) for λ > 0
    - Subadditivity: ρ(X + Y) ≤ ρ(X) + ρ(Y)
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.default_confidence: float = self.config.get('confidence_level', 0.95)
        self.n_simulations: int = self.config.get('n_simulations', 10000)
        self.random_seed: Optional[int] = self.config.get('random_seed', None)
        self._rng = np.random.default_rng(self.random_seed)
        
    def compute_var_historical(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Historical VaR using empirical quantile.
        
        VaR_α = -Quantile(returns, 1-α)
        """
        alpha = 1 - confidence_level
        var = -np.percentile(returns, alpha * 100)
        return float(var)
    
    def compute_var_parametric(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        distribution: str = "gaussian"
    ) -> float:
        """
        Parametric VaR assuming specified distribution.
        
        Gaussian: VaR_α = μ + σ × Φ^(-1)(1-α)
        Student-t: VaR_α = μ + σ × t_ν^(-1)(1-α) × √((ν-2)/ν)
        """
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
            var = self.compute_var_historical(returns, confidence_level)
        
        return float(var)
    
    def compute_var_cornish_fisher(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Cornish-Fisher VaR with skewness and kurtosis adjustment.
        
        z_CF = z + (z² - 1)×S/6 + (z³ - 3z)×(K-3)/24 - (2z³ - 5z)×S²/36
        VaR_α = μ - σ × z_CF
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)
        
        alpha = 1 - confidence_level
        z = stats.norm.ppf(alpha)
        
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = -(mu + sigma * z_cf)
        return float(var)
    
    def compute_var_monte_carlo(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: Optional[int] = None
    ) -> float:
        """
        Monte Carlo VaR using bootstrapped returns.
        """
        n_sims = n_simulations or self.n_simulations
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        simulated = self._rng.normal(mu, sigma, n_sims)
        
        alpha = 1 - confidence_level
        var = -np.percentile(simulated, alpha * 100)
        return float(var)
    
    def compute_cvar_historical(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> CVaRResult:
        """
        Historical CVaR (Expected Shortfall).
        
        CVaR_α = E[L | L > VaR_α] = mean of losses exceeding VaR
        """
        var = self.compute_var_historical(returns, confidence_level)
        losses = -returns
        tail_losses = losses[losses > var]
        
        if len(tail_losses) == 0:
            cvar = var
            tail_losses = np.array([var])
        else:
            cvar = np.mean(tail_losses)
        
        return CVaRResult(
            cvar=float(cvar),
            var=float(var),
            confidence_level=confidence_level,
            method=CVaRMethod.HISTORICAL.value,
            tail_losses=tail_losses,
            expected_tail_loss=float(cvar),
            tail_probability=len(tail_losses) / len(returns)
        )
    
    def compute_cvar_parametric(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        distribution: str = "gaussian"
    ) -> CVaRResult:
        """
        Parametric CVaR for Gaussian distribution.
        
        CVaR_α = μ + σ × φ(Φ^(-1)(α)) / (1-α)
        
        Where φ is PDF and Φ is CDF of standard normal.
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        alpha = 1 - confidence_level
        
        if distribution == "gaussian":
            z_alpha = stats.norm.ppf(alpha)
            pdf_z = stats.norm.pdf(z_alpha)
            cvar = -(mu + sigma * pdf_z / alpha)
            var = -(mu + sigma * z_alpha)
        elif distribution == "student_t":
            nu, _, _ = stats.t.fit(returns)
            nu = max(nu, 2.1)
            t_alpha = stats.t.ppf(alpha, nu)
            pdf_t = stats.t.pdf(t_alpha, nu)
            scaling = (nu + t_alpha**2) / (nu - 1)
            cvar = -(mu + sigma * pdf_t * scaling / alpha)
            var = -(mu + sigma * t_alpha * np.sqrt((nu-2)/nu))
        else:
            return self.compute_cvar_historical(returns, confidence_level)
        
        losses = -returns
        tail_losses = losses[losses > var]
        
        return CVaRResult(
            cvar=float(cvar),
            var=float(var),
            confidence_level=confidence_level,
            method=f"parametric_{distribution}",
            tail_losses=tail_losses if len(tail_losses) > 0 else np.array([var]),
            expected_tail_loss=float(cvar),
            tail_probability=alpha
        )
    
    def compute_cvar_cornish_fisher(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> CVaRResult:
        """
        Cornish-Fisher CVaR with higher moment adjustments.
        """
        var = self.compute_var_cornish_fisher(returns, confidence_level)
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)
        
        alpha = 1 - confidence_level
        z = stats.norm.ppf(alpha)
        pdf_z = stats.norm.pdf(z)
        
        es_adjustment = (1 +(z**2 - 1) * skew / 6 +
                        (z**3 - 3*z) * kurt / 24)
        
        cvar = -(mu + sigma * pdf_z * es_adjustment / alpha)
        
        losses = -returns
        tail_losses = losses[losses > var]
        
        return CVaRResult(
            cvar=float(cvar),
            var=float(var),
            confidence_level=confidence_level,
            method=CVaRMethod.CORNISH_FISHER.value,
            tail_losses=tail_losses if len(tail_losses) > 0 else np.array([var]),
            expected_tail_loss=float(cvar),
            tail_probability=alpha
        )
    
    def compute_cvar_monte_carlo(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        n_simulations: Optional[int] = None
    ) -> CVaRResult:
        """
        Monte Carlo CVaR using simulated scenarios.
        """
        n_sims = n_simulations or self.n_simulations
        
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        simulated_returns = self._rng.normal(mu, sigma, n_sims)
        simulated_losses = -simulated_returns
        
        alpha = 1 - confidence_level
        var = np.percentile(simulated_losses, confidence_level * 100)
        tail_losses = simulated_losses[simulated_losses > var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return CVaRResult(
            cvar=float(cvar),
            var=float(var),
            confidence_level=confidence_level,
            method=CVaRMethod.MONTE_CARLO.value,
            tail_losses=tail_losses,
            expected_tail_loss=float(cvar),
            tail_probability=len(tail_losses) / n_sims
        )
    
    def compute_portfolio_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        method: CVaRMethod = CVaRMethod.HISTORICAL
    ) -> CVaRResult:
        """Compute CVaR for a weighted portfolio.
        """
        weights = np.asarray(weights).flatten()
        portfolio_returns = returns @ weights
        
        method_map = {
            CVaRMethod.HISTORICAL: self.compute_cvar_historical,
            CVaRMethod.PARAMETRIC_GAUSSIAN: lambda r, c: self.compute_cvar_parametric(r, c, "gaussian"),
            CVaRMethod.PARAMETRIC_STUDENT_T: lambda r, c: self.compute_cvar_parametric(r, c, "student_t"),
            CVaRMethod.CORNISH_FISHER: self.compute_cvar_cornish_fisher,
            CVaRMethod.MONTE_CARLO: self.compute_cvar_monte_carlo,
        }
        
        compute_func = method_map.get(method, self.compute_cvar_historical)
        result = compute_func(portfolio_returns, confidence_level)
        
        marginal_cvar = self._compute_marginal_cvar(returns, weights, confidence_level)
        component_cvar = marginal_cvar * weights
        
        result.marginal_cvar = marginal_cvar
        result.component_cvar = component_cvar
        
        return result
    
    def _compute_marginal_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float,
        delta: float = 0.0001
    ) -> np.ndarray:
        """
        Compute marginal CVaR via finite differences.
        ∂CVaR/∂w_i ≈ (CVaR(w + δe_i) - CVaR(w - δe_i)) / (2δ)
        """
        n_assets = len(weights)
        marginal = np.zeros(n_assets)
        
        for i in range(n_assets):
            weights_up = weights.copy()
            weights_up[i] += delta
            weights_up = weights_up / np.sum(weights_up)
            
            weights_down = weights.copy()
            weights_down[i] -= delta
            weights_down = weights_down / np.sum(weights_down)
            
            cvar_up = self.compute_cvar_historical(returns @ weights_up, confidence_level).cvar
            cvar_down = self.compute_cvar_historical(returns @ weights_down, confidence_level).cvar
            
            marginal[i] = (cvar_up - cvar_down) / (2 * delta)
        
        return marginal
    
    def compute_incremental_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[int, float]:
        """
        Compute incremental CVaR for each asset.
        Incremental CVaR_i = CVaR(portfolio) - CVaR(portfolio without asset i)
        """
        n_assets = len(weights)
        base_cvar = self.compute_portfolio_cvar(returns, weights, confidence_level).cvar
        
        incremental = {}
        for i in range(n_assets):
            reduced_weights = weights.copy()
            reduced_weights[i] = 0
            if np.sum(reduced_weights) > 0:
                reduced_weights = reduced_weights / np.sum(reduced_weights)
                reduced_cvar = self.compute_portfolio_cvar(returns, reduced_weights, confidence_level).cvar
            else:
                reduced_cvar = 0
            
            incremental[i] = base_cvar - reduced_cvar
        
        return incremental
    
    def compute_comprehensive_risk_metrics(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> RiskMetrics:
        """
        Compute comprehensive suite of risk metrics.
        """
        returns = np.asarray(returns).flatten()
        
        var_95 = self.compute_var_historical(returns, 0.95)
        var_99 = self.compute_var_historical(returns, 0.99)
        cvar_95_result = self.compute_cvar_historical(returns, 0.95)
        cvar_99_result = self.compute_cvar_historical(returns, 0.99)
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = -np.min(drawdowns)
        
        volatility = np.std(returns, ddof=1) * np.sqrt(252)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)
        
        upper_tail = np.percentile(returns, 95)
        lower_tail = np.percentile(returns, 5)
        tail_ratio = abs(upper_tail / lower_tail) if lower_tail != 0 else np.inf
        
        annual_return = np.mean(returns) * 252
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else np.inf
        
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino_ratio = (annual_return - risk_free_rate) / downside_std
        
        threshold = risk_free_rate / 252
        gains = np.sum(returns[returns > threshold] - threshold)
        losses = np.sum(threshold - returns[returns <= threshold])
        omega_ratio = gains / losses if losses > 0 else np.inf
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95_result.cvar,
            cvar_99=cvar_99_result.cvar,
            expected_shortfall_95=cvar_95_result.cvar,
            expected_shortfall_99=cvar_99_result.cvar,
            max_drawdown=max_drawdown,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            omega_ratio=omega_ratio
        )
    
    def stress_test_portfolio(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        scenarios: Dict[str, Dict],
        confidence_level: float = 0.95
    ) -> List[StressTestResult]:
        """
        Run stress tests on portfolio under various scenarios.
        """
        base_result = self.compute_portfolio_cvar(returns, weights, confidence_level)
        results = []
        
        for scenario_name, scenario_params in scenarios.items():
            shock_vector = scenario_params.get('shocks', np.zeros(len(weights)))
            correlation_shift = scenario_params.get('correlation_shift', 0.0)
            volatility_multiplier = scenario_params.get('volatility_multiplier', 1.0)
            
            stressed_returns = returns.copy()
            stressed_returns = stressed_returns * volatility_multiplier
            stressed_returns = stressed_returns + shock_vector
            
            portfolio_loss = -np.sum(weights * shock_vector)
            
            worst_asset = np.argmin(shock_vector)
            worst_asset_loss = -shock_vector[worst_asset]
            
            results.append(StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss=portfolio_loss,
                var_breach=portfolio_loss > base_result.var,
                cvar_breach=portfolio_loss > base_result.cvar,
                worst_asset=worst_asset,
                worst_asset_loss=worst_asset_loss,
                correlation_shift=correlation_shift
            ))
        
        return results
    
    def optimize_cvar_portfolio(
        self,
        returns: np.ndarray,
        target_return: Optional[float] = None,
        confidence_level: float = 0.95,
        constraints: Optional[Dict] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize portfolio weights to minimize CVaR.
        
        min CVaR(w)
        s.t. w'μ ≥ target_return
             Σw = 1
             w ≥ 0 (if long_only)
        """
        n_assets = returns.shape[1]
        constraints_list = constraints or {}
        
        def objective(w):
            portfolio_returns = returns @ w
            cvar_result = self.compute_cvar_historical(portfolio_returns, confidence_level)
            return cvar_result.cvar
        
        bounds = [(0, 1) for _ in range(n_assets)] if constraints_list.get('long_only', True) else None
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            mean_returns = np.mean(returns, axis=0)
            cons.append({'type': 'ineq', 'fun': lambda w: w @ mean_returns - target_return})
        
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        optimal_weights = result.x
        optimal_cvar = result.fun
        
        return optimal_weights, optimal_cvar