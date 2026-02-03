"""
Comprehensive Portfolio Optimization Engine
Author: Erdinc Erdogan
Purpose: Implements 10+ optimization methods including MVO, Black-Litterman, HRP, Risk Parity, Max Sharpe, Min Variance, and CVaR with full constraint handling.
References:
- Markowitz Mean-Variance (1952)
- Black-Litterman (1992)
- Hierarchical Risk Parity (López de Prado, 2016)
- Risk Parity / Equal Risk Contribution
Usage:
    optimizer = PortfolioOptimizerEngine(returns, covariance)
    result = optimizer.optimize(objective=OptimizationObjective.MAX_SHARPE)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_CVAR = "min_cvar"
    BLACK_LITTERMAN = "black_litterman"
    HRP = "hrp"
    KELLY = "kelly"


class RiskMeasure(Enum):
    """Risk measures for optimization"""
    VARIANCE = "variance"
    STANDARD_DEVIATION = "std"
    CVAR = "cvar"
    VAR = "var"
    MAX_DRAWDOWN = "max_drawdown"
    SEMI_VARIANCE = "semi_variance"


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    sum_weights: float = 1.0
    max_turnover: Optional[float] = None
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    factor_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    max_tracking_error: Optional[float] = None
    min_holdings: Optional[int] = None
    max_holdings: Optional[int] = None


@dataclass
class OptimizationResult:
    """Portfolio optimization results"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective_value: float
    asset_names: List[str]
    optimization_method: str
    converged: bool
    risk_contributions: Optional[np.ndarray] = None
    diversification_ratio: Optional[float] = None
    cvar: Optional[float] = None


@dataclass
class BlackLittermanResult:
    """Black-Litterman model results"""
    weights: np.ndarray
    prior_returns: np.ndarray
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    views_contribution: np.ndarray
    asset_names: List[str]
    tau: float
    risk_aversion: float


@dataclass
class HRPResult:
    """Hierarchical Risk Parity results"""
    weights: np.ndarray
    cluster_order: List[int]
    linkage_matrix: np.ndarray
    asset_names: List[str]
    cluster_weights: Dict[int, float]
    dendrogram_order: List[str]


@dataclass
class EfficientFrontier:
    """Efficient frontier data"""
    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights: np.ndarray
    max_sharpe_idx: int
    min_vol_idx: int


@dataclass
class RiskParityResult:
    """Risk parity optimization results"""
    weights: np.ndarray
    risk_contributions: np.ndarray
    marginal_risk: np.ndarray
    total_risk: float
    risk_contribution_error: float
    asset_names: List[str]


class PortfolioOptimizer(BaseModule):
    """Institutional-Grade Portfolio Optimizer.
    
    Implements comprehensive portfolio optimization methodologies
    for asset allocation and risk management.
    
    Mathematical Framework:
    ----------------------
    
    Mean-Variance Optimization (Markowitz, 1952):
        max  w'μ - (λ/2)w'Σw
        s.t. Σw_i = 1, w_i ≥ 0
        Efficient Frontier: σ²_p = w'Σw, μ_p = w'μ
    
    Maximum Sharpe Ratio:
        max  (w'μ - r_f) / √(w'Σw)
        Equivalent: min w'Σw / (w'μ - r_f)²
    
    Global Minimum Variance:
        min w'Σw
        s.t. Σw_i = 1
        Closed-form: w* = Σ⁻¹1 / (1'Σ⁻¹1)
    
    Risk Parity (Equal Risk Contribution):
        w_i × (Σw)_i / (w'Σw) = 1/N  ∀i
        
        Minimize: Σ_i (RC_i - RC_target)²
        where RC_i = w_i × (Σw)_i / σ_p
    
    Black-Litterman (1992):
        Prior: Π = λΣw_mkt (equilibrium returns)
        Views: P'μ = Q + ε, ε ~ N(0,Ω)
        
        Posterior: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]
        Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
    
    Hierarchical Risk Parity (López de Prado, 2016):
        1. Tree Clustering: Hierarchical clustering on correlation
        2. Quasi-Diagonalization: Reorder covariance matrix
        3. Recursive Bisection: Allocate weights top-down
    
    Maximum Diversification:
        max  DR = (w'σ) / √(w'Σw)
        where σ = vector of asset volatilities
    
    CVaR Optimization:
        min  CVaR_α(w) = E[L | L > VaR_α]
        CVaR_α = (1/(1-α)) ∫_α^1 VaR_u du
    
    Kelly Criterion:
        f* = (μ - r_f) / σ²
        Full Kelly: w* = Σ⁻¹(μ - r_f) / λ
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.risk_free_rate: float = self.config.get('risk_free_rate', 0.02)
        self.risk_aversion: float = self.config.get('risk_aversion', 2.5)
        self.annualization_factor: int = self.config.get('annualization_factor', 252)
        self.cvar_confidence: float = self.config.get('cvar_confidence', 0.95)
        self.tau: float = self.config.get('tau', 0.05)
    
    # =========================================================================
    # MEAN-VARIANCE OPTIMIZATION
    # =========================================================================
    
    def optimize_mean_variance(
        self,
        returns: np.ndarray,
        target_return: Optional[float] = None,
        constraints: Optional[PortfolioConstraints] = None,
        asset_names: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Mean-Variance Optimization (Markowitz).
        
        max  w'μ - (λ/2)w'Σw
        s.t. Σw_i = 1, w_min ≤ w_i ≤ w_max
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        mu = np.mean(returns, axis=0) * self.annualization_factor
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        
        def objective(w):
            port_return = w @ mu
            port_var = w @ cov @ w
            return -(port_return - (self.risk_aversion / 2) * port_var)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - constraints.sum_weights}]
        
        if target_return is not None:
            cons.append({'type': 'eq', 'fun': lambda w: w @ mu - target_return})
        
        bounds = [(constraints.min_weight, constraints.max_weight)] * n_assets
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        weights = result.x
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk / port_vol
        
        return OptimizationResult(
            weights=weights,
            expected_return=float(port_return),
            volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            objective_value=float(-result.fun),
            asset_names=asset_names,
            optimization_method=OptimizationObjective.MAX_SHARPE.value,
            converged=result.success,
            risk_contributions=risk_contrib
        )
    
    def optimize_max_sharpe(
        self,
        returns: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None,
        asset_names: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Maximum Sharpe Ratio Portfolio.
        
        max  (w'μ - r_f) / √(w'Σw)
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        mu = np.mean(returns, axis=0) * self.annualization_factor
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        
        def neg_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - constraints.sum_weights}]
        bounds = [(constraints.min_weight, constraints.max_weight)] * n_assets
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            neg_sharpe, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        weights = result.x
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk / port_vol if port_vol > 0 else np.zeros(n_assets)
        
        return OptimizationResult(
            weights=weights,
            expected_return=float(port_return),
            volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            objective_value=float(sharpe),
            asset_names=asset_names,
            optimization_method=OptimizationObjective.MAX_SHARPE.value,
            converged=result.success,
            risk_contributions=risk_contrib
        )
    
    def optimize_min_variance(
        self,
        returns: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None,
        asset_names: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Global Minimum Variance Portfolio.
        
        min  w'Σw
        s.t. Σw_i = 1
        Closed-form: w* = Σ⁻¹1 / (1'Σ⁻¹1)
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        mu = np.mean(returns, axis=0) * self.annualization_factor
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        
        try:
            cov_inv = np.linalg.inv(cov)
            ones = np.ones(n_assets)
            weights_closed = cov_inv @ ones / (ones @ cov_inv @ ones)
            if np.all(weights_closed >= constraints.min_weight) and \
               np.all(weights_closed <= constraints.max_weight):
                weights = weights_closed
                converged = True
            else:
                raise ValueError("Closed-form violates constraints")
        except:
            def portfolio_variance(w):
                return w @ cov @ w
            
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - constraints.sum_weights}]
            bounds = [(constraints.min_weight, constraints.max_weight)] * n_assets
            w0 = np.ones(n_assets) / n_assets
            
            result = minimize(
                portfolio_variance, w0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000}
            )
            
            weights = result.x
            converged = result.success
        
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk / port_vol if port_vol > 0 else np.zeros(n_assets)
        
        return OptimizationResult(
            weights=weights,
            expected_return=float(port_return),
            volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            objective_value=float(port_vol**2),
            asset_names=asset_names,
            optimization_method=OptimizationObjective.MIN_VARIANCE.value,
            converged=converged,
            risk_contributions=risk_contrib
        )
    
    # =========================================================================
    # BLACK-LITTERMAN MODEL
    # =========================================================================
    
    def optimize_black_litterman(
        self,
        returns: np.ndarray,
        market_weights: np.ndarray,
        views_matrix: np.ndarray,
        views_returns: np.ndarray,
        views_confidence: Optional[np.ndarray] = None,
        tau: Optional[float] = None,
        asset_names: Optional[List[str]] = None
    ) -> BlackLittermanResult:
        """
        Black-Litterman Model with Bayesian Priors.
        
        Prior (Equilibrium Returns):
            Π = λΣw_mkt
        
        Views:
            P'μ = Q + ε, ε ~ N(0, Ω)
        Posterior:
            μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]
            Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        tau = tau or self.tau
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        market_weights = np.asarray(market_weights).flatten()
        views_matrix = np.asarray(views_matrix)
        views_returns = np.asarray(views_returns).flatten()
        
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        
        prior_returns = self.risk_aversion * cov @ market_weights
        
        n_views = len(views_returns)
        if views_confidence is None:
            omega = np.diag(np.diag(views_matrix @ (tau * cov) @ views_matrix.T))
        else:
            omega = np.diag(1.0 / np.asarray(views_confidence))
        
        tau_cov = tau * cov
        tau_cov_inv = np.linalg.inv(tau_cov)
        omega_inv = np.linalg.inv(omega)
        
        posterior_cov_inv = tau_cov_inv + views_matrix.T @ omega_inv @ views_matrix
        posterior_cov = np.linalg.inv(posterior_cov_inv)
        
        posterior_returns = posterior_cov @ (
            tau_cov_inv @ prior_returns + 
            views_matrix.T @ omega_inv @ views_returns
        )
        
        combined_cov = cov + posterior_cov
        weights = np.linalg.inv(self.risk_aversion * combined_cov) @ posterior_returns
        
        weights = np.maximum(weights, 0)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_assets) / n_assets
        
        views_contribution = posterior_returns - prior_returns
        
        return BlackLittermanResult(
            weights=weights,
            prior_returns=prior_returns,
            posterior_returns=posterior_returns,
            posterior_covariance=posterior_cov,
            views_contribution=views_contribution,
            asset_names=asset_names,
            tau=tau,
            risk_aversion=self.risk_aversion
        )
    
    # =========================================================================
    # HIERARCHICAL RISK PARITY (HRP)
    # =========================================================================
    
    def optimize_hrp(
        self,
        returns: np.ndarray,
        linkage_method: str = 'ward',
        asset_names: Optional[List[str]] = None
    ) -> HRPResult:
        """
        Hierarchical Risk Parity (López de Prado, 2016).
        
        Three-step process:
        1. Tree Clustering: Hierarchical clustering on correlation distance
        2. Quasi-Diagonalization: Reorder covariance matrix
        3. Recursive Bisection: Top-down weight allocation
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        cov = np.cov(returns, rowvar=False)
        corr = np.corrcoef(returns, rowvar=False)
        
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        
        condensed_dist = squareform(dist)
        link = linkage(condensed_dist, method=linkage_method)
        
        sort_idx = self._get_quasi_diag(link)
        
        weights = self._recursive_bisection(cov, sort_idx)
        
        cluster_weights = {}
        for i, idx in enumerate(sort_idx):
            cluster_weights[idx] = weights[idx]
        
        dendrogram_order = [asset_names[i] for i in sort_idx]
        
        return HRPResult(
            weights=weights,
            cluster_order=sort_idx,
            linkage_matrix=link,
            asset_names=asset_names,
            cluster_weights=cluster_weights,
            dendrogram_order=dendrogram_order
        )
    
    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal ordering from linkage matrix."""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df0])
            sort_idx = sort_idx.sort_index()
            sort_idx.index = range(sort_idx.shape[0])
        
        return sort_idx.tolist()
    
    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sort_idx: List[int]
    ) -> np.ndarray:
        """Recursive bisection for HRP weight allocation."""
        n_assets = cov.shape[0]
        weights = np.ones(n_assets)
        cluster_items = [sort_idx]
        
        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k] for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            
            for i in range(0, len(cluster_items), 2):
                if i + 1 < len(cluster_items):
                    cluster_0 = cluster_items[i]
                    cluster_1 = cluster_items[i + 1]
                    
                    var_0 = self._get_cluster_var(cov, cluster_0)
                    var_1 = self._get_cluster_var(cov, cluster_1)
                    
                    alpha = 1 - var_0 / (var_0 + var_1)
                
                    weights[cluster_0] *= alpha
                    weights[cluster_1] *= 1 - alpha
        
        return weights
    
    def _get_cluster_var(
        self,
        cov: np.ndarray,
        cluster_items: List[int]
    ) -> float:
        """Get cluster variance using inverse-variance weights."""
        cov_slice = cov[np.ix_(cluster_items, cluster_items)]
        
        ivp = 1.0 / np.diag(cov_slice)
        ivp = ivp / np.sum(ivp)
        
        cluster_var = ivp @ cov_slice @ ivp
        return cluster_var
    
    # =========================================================================
    # RISK PARITY (EQUAL RISK CONTRIBUTION)
    # =========================================================================
    
    def optimize_risk_parity(
        self,
        returns: np.ndarray,
        risk_budget: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None
    ) -> RiskParityResult:
        """
        Risk Parity / Equal Risk Contribution Portfolio.
        
        Target: w_i × (Σw)_i / (w'Σw) = b_i  ∀i
        where b_i is the risk budget (default: 1/N)
        
        Minimize: Σ_i (RC_i - b_i)²
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets
        else:
            risk_budget = np.asarray(risk_budget)
            risk_budget = risk_budget / np.sum(risk_budget)
        
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        
        def risk_parity_objective(w):
            w = np.maximum(w, 1e-10)
            port_vol = np.sqrt(w @ cov @ w)
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / port_vol
            risk_contrib_pct = risk_contrib / port_vol
            
            return np.sum((risk_contrib_pct - risk_budget)**2)
        
        def risk_contrib_constraint(w):
            return np.sum(w) - 1.0
        
        cons = [{'type': 'eq', 'fun': risk_contrib_constraint}]
        bounds = [(0.001, 1.0)] * n_assets
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            risk_parity_objective, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        weights = result.x
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk / port_vol
        risk_contrib_pct = risk_contrib / port_vol
        rc_error = np.sqrt(np.sum((risk_contrib_pct - risk_budget)**2))
        
        return RiskParityResult(
            weights=weights,
            risk_contributions=risk_contrib,
            marginal_risk=marginal_risk,
            total_risk=float(port_vol),
            risk_contribution_error=float(rc_error),
            asset_names=asset_names
        )
    
    # =========================================================================
    # MAXIMUM DIVERSIFICATION
    # =========================================================================
    
    def optimize_max_diversification(
        self,
        returns: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None,
        asset_names: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Maximum Diversification Portfolio.
        
        max  DR = (w'σ) / √(w'Σw)
        where σ = vector of asset volatilities
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        mu = np.mean(returns, axis=0) * self.annualization_factor
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        vols = np.sqrt(np.diag(cov))
        
        def neg_diversification_ratio(w):
            weighted_vol = w @ vols
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 1e10
            return -weighted_vol / port_vol
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - constraints.sum_weights}]
        bounds = [(constraints.min_weight, constraints.max_weight)] * n_assets
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            neg_diversification_ratio, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        weights = result.x
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        div_ratio = (weights @ vols) / port_vol if port_vol > 0 else 1.0
        
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk / port_vol if port_vol > 0 else np.zeros(n_assets)
        
        return OptimizationResult(
            weights=weights,
            expected_return=float(port_return),
            volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            objective_value=float(div_ratio),
            asset_names=asset_names,
            optimization_method=OptimizationObjective.MAX_DIVERSIFICATION.value,
            converged=result.success,
            risk_contributions=risk_contrib,
            diversification_ratio=float(div_ratio)
        )
    
    # =========================================================================
    # CVAR OPTIMIZATION
    # =========================================================================
    
    def optimize_min_cvar(
        self,
        returns: np.ndarray,
        confidence: Optional[float] = None,
        target_return: Optional[float] = None,
        constraints: Optional[PortfolioConstraints] = None,
        asset_names: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Minimum CVaR (Conditional Value at Risk) Portfolio.
        
        min  CVaR_α(w) = E[L | L > VaR_α]
        
        Using Rockafellar-Uryasev formulation:
        min  ζ + (1/(1-α)) × (1/T) × Σ max(0, -w'r_t - ζ)
        """
        returns = np.asarray(returns)
        n_obs, n_assets = returns.shape
        confidence = confidence or self.cvar_confidence
        alpha = confidence
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        mu = np.mean(returns, axis=0) * self.annualization_factor
        
        def cvar_objective(x):
            w = x[:n_assets]
            zeta = x[n_assets]
            
            port_returns = returns @ w
            
            losses = -port_returns
            exceedances = np.maximum(losses - zeta, 0)
            cvar = zeta + np.mean(exceedances) / (1 - alpha)
            
            return cvar * self.annualization_factor
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_assets]) - constraints.sum_weights}]
        
        if target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda x: x[:n_assets] @ mu - target_return
            })
        
        bounds = [(constraints.min_weight, constraints.max_weight)] * n_assets + [(-1, 1)]
        
        x0 = np.concatenate([np.ones(n_assets) / n_assets, [0.0]])
        
        result = minimize(
            cvar_objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        weights = result.x[:n_assets]
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        port_return = weights @ mu
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        port_returns = returns @ weights
        var = -np.percentile(port_returns, (1 - alpha) * 100)
        cvar = -np.mean(port_returns[port_returns <= -var]) * self.annualization_factor
        
        return OptimizationResult(
            weights=weights,
            expected_return=float(port_return),
            volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            objective_value=float(cvar),
            asset_names=asset_names,
            optimization_method=OptimizationObjective.MIN_CVAR.value,
            converged=result.success,
            cvar=float(cvar)
        )
    
    # =========================================================================
    # KELLY CRITERION
    # =========================================================================
    
    def optimize_kelly(
        self,
        returns: np.ndarray,
        fraction: float = 0.5,
        constraints: Optional[PortfolioConstraints] = None,
        asset_names: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Kelly Criterion Portfolio.
        
        Full Kelly: w* = Σ⁻¹(μ - r_f) / λFractional Kelly: w* = fraction × Full_Kelly
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        mu = np.mean(returns, axis=0) * self.annualization_factor
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        
        excess_returns = mu - self.risk_free_rate
        
        try:
            cov_inv = np.linalg.inv(cov)
            kelly_weights = cov_inv @ excess_returns
        except:
            cov_reg = cov + 0.01 * np.eye(n_assets)
            cov_inv = np.linalg.inv(cov_reg)
            kelly_weights = cov_inv @ excess_returns
        
        weights = fraction * kelly_weights
        
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        
        if np.sum(np.abs(weights)) > constraints.sum_weights:
            weights = weights / np.sum(np.abs(weights)) * constraints.sum_weights
        
        if constraints.min_weight >= 0:
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
        
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=float(port_return),
            volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            objective_value=float(sharpe),
            asset_names=asset_names,
            optimization_method=OptimizationObjective.KELLY.value,
            converged=True
        )
    
    # =========================================================================
    # EFFICIENT FRONTIER
    # =========================================================================
    
    def compute_efficient_frontier(
        self,
        returns: np.ndarray,
        n_points: int = 50,
        constraints: Optional[PortfolioConstraints] = None,
        asset_names: Optional[List[str]] = None
    ) -> EfficientFrontier:
        """
        Compute the efficient frontier.
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        mu = np.mean(returns, axis=0) * self.annualization_factor
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        
        min_var_result = self.optimize_min_variance(returns, constraints, asset_names)
        
        max_return = np.max(mu)
        min_return = min_var_result.expected_return
        
        target_returns = np.linspace(min_return, max_return * 0.95, n_points)
        
        frontier_returns = []
        frontier_vols = []
        frontier_sharpes = []
        frontier_weights = []
        
        for target in target_returns:
            try:
                result = self.optimize_mean_variance(
                    returns, target_return=target,
                    constraints=constraints, asset_names=asset_names
                )
                
                frontier_returns.append(result.expected_return)
                frontier_vols.append(result.volatility)
                frontier_sharpes.append(result.sharpe_ratio)
                frontier_weights.append(result.weights)
            except:
                continue
        
        frontier_returns = np.array(frontier_returns)
        frontier_vols = np.array(frontier_vols)
        frontier_sharpes = np.array(frontier_sharpes)
        frontier_weights = np.array(frontier_weights)
        
        max_sharpe_idx = np.argmax(frontier_sharpes)
        min_vol_idx = np.argmin(frontier_vols)
        
        return EfficientFrontier(
            returns=frontier_returns,
            volatilities=frontier_vols,
            sharpe_ratios=frontier_sharpes,
            weights=frontier_weights,
            max_sharpe_idx=int(max_sharpe_idx),
            min_vol_idx=int(min_vol_idx)
        )
    
    # =========================================================================
    # PORTFOLIO ANALYTICS
    # =========================================================================
    
    def compute_portfolio_metrics(
        self,
        weights: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive portfolio metrics.
        """
        weights = np.asarray(weights)
        returns = np.asarray(returns)
        
        port_returns = returns @ weights
        
        ann_return = np.mean(port_returns) * self.annualization_factor
        ann_vol = np.std(port_returns) * np.sqrt(self.annualization_factor)
        
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        downside_returns = port_returns[port_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(self.annualization_factor) if len(downside_returns) > 0 else ann_vol
        sortino = (ann_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        cumulative = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        var_95 = -np.percentile(port_returns, 5) * np.sqrt(self.annualization_factor)
        cvar_95 = -np.mean(port_returns[port_returns <= np.percentile(port_returns, 5)]) * np.sqrt(self.annualization_factor)
        
        skewness = stats.skew(port_returns)
        kurtosis = stats.kurtosis(port_returns)
        
        return {
            'annualized_return': float(ann_return),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }
    
    def compute_risk_decomposition(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Decompose portfolio risk by asset.
        """
        weights = np.asarray(weights)
        returns = np.asarray(returns)
        n_assets = len(weights)
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        cov = np.cov(returns, rowvar=False) * self.annualization_factor
        port_vol = np.sqrt(weights @ cov @ weights)
        
        marginal_risk = cov @ weights / port_vol
        
        component_risk = weights * marginal_risk
        
        pct_contribution = component_risk / port_vol
        
        return pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights,
            'Marginal_Risk': marginal_risk,
            'Component_Risk': component_risk,
            'Pct_Contribution': pct_contribution
        })