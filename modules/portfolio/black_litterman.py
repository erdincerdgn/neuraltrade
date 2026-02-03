"""
Black-Litterman Portfolio Allocation Engine
Author: Erdinc Erdogan
Purpose: Combines CAPM equilibrium returns with investor views using Bayesian inference to generate posterior expected returns and optimal portfolio weights.
References:
- Black-Litterman Model (1992)
- Bayesian Portfolio Optimization
- E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)Q]
Usage:
    engine = BlackLittermanEngine(config={'tau': 0.025})
    result = engine.optimize(covariance, market_weights, views)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class ViewType(Enum):
    """Types of investor views in Black-Litterman framework"""
    ABSOLUTE = "absolute"      # Asset i will return X%
    RELATIVE = "relative"      # Asset i will outperform asset j by X%


@dataclass
class InvestorView:
    """
    Represents a single investor view for Black-Litterman model.
    """
    view_type: ViewType
    assets: List[int]
    weights: List[float]
    expected_return: float
    confidence: float = 0.5
    
    def __post_init__(self):
        if self.view_type == ViewType.RELATIVE:
            if abs(sum(self.weights)) > 1e-10:
                raise ValueError("Relative view weights must sum to zero")
        if not0 < self.confidence <= 1:
            raise ValueError("Confidence must be in (0, 1]")


@dataclass
class BlackLittermanResult:
    """Container for Black-Litterman optimization results"""
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    optimal_weights: np.ndarray
    equilibrium_returns: np.ndarray
    risk_aversion: float
    tau: float
    sharpe_ratio: float
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    view_contributions: Optional[Dict] = None


class BlackLittermanEngine(BaseModule):
    """
    Institutional-Grade Black-Litterman Portfolio Allocation Engine.
    
    Mathematical Framework:
    ----------------------
    Prior (CAPM Equilibrium): π = δΣw_mkt
    Posterior Expected Returns: E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)Q]
    Posterior Covariance: Σ_posterior = Σ + [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.tau: float = self.config.get('tau', 0.025)
        self.risk_free_rate: float = self.config.get('risk_free_rate', 0.02)
        self.default_risk_aversion: float = self.config.get('risk_aversion', 2.5)
        self._posterior_returns: Optional[np.ndarray] = None
        self._posterior_cov: Optional[np.ndarray] = None
        
    def compute_equilibrium_returns(
        self,
        covariance_matrix: np.ndarray,
        market_weights: np.ndarray,
        risk_aversion: Optional[float] = None
    ) -> np.ndarray:
        """Reverse optimization: π = δΣw_mkt"""
        delta = risk_aversion or self.default_risk_aversion
        market_weights = np.asarray(market_weights).flatten()
        return delta * covariance_matrix @ market_weights
    
    def estimate_risk_aversion(
        self,
        market_return: float,
        market_variance: float,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """Estimate δ = (E[R_mkt] - R_f) / σ²_mkt"""
        rf = risk_free_rate or self.risk_free_rate
        if market_variance <= 0:
            raise ValueError("Market variance must be positive")
        return (market_return - rf) / market_variance
    
    def construct_view_matrices(
        self,
        views: List[InvestorView],
        n_assets: int,
        covariance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct P, Q,Ω matrices using Idzorek's method"""
        k = len(views)
        P = np.zeros((k, n_assets))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)
        
        for i, view in enumerate(views):
            for asset_idx, weight in zip(view.assets, view.weights):
                if asset_idx >= n_assets:
                    raise ValueError(f"Asset index {asset_idx} out of bounds")
                P[i, asset_idx] = weight
            Q[i] = view.expected_return
            p_row = P[i:i+1, :]
            view_variance = float(p_row @ (self.tau * covariance_matrix) @ p_row.T)
            alpha = (1.0 - view.confidence) / view.confidence
            omega_diag[i] = alpha * view_variance
        
        return P, Q, np.diag(omega_diag)
    
    def compute_posterior(
        self,
        equilibrium_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,Omega: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Black-Litterman Master Formula"""
        tau_sigma = self.tau * covariance_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        
        posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
        M = np.linalg.inv(posterior_precision)
        posterior_returns = M @ (tau_sigma_inv @ equilibrium_returns + P.T @ omega_inv @ Q)
        posterior_covariance = covariance_matrix + M
        
        self._posterior_returns = posterior_returns
        self._posterior_cov = posterior_covariance
        return posterior_returns, posterior_covariance
    
    def compute_posterior_no_views(
        self,
        equilibrium_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure CAPM prior (no views)"""
        return equilibrium_returns, covariance_matrix + self.tau * covariance_matrix
    
    def optimize_weights(
        self,
        posterior_returns: np.ndarray,
        posterior_covariance: np.ndarray,
        risk_aversion: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Mean-variance optimization: w* = (1/δ) × Σ^(-1) × E[R]"""
        delta = risk_aversion or self.default_risk_aversion
        constraints = constraints or {}
        
        cov_inv = np.linalg.inv(posterior_covariance)
        raw_weights = (1.0 / delta) * cov_inv @ posterior_returns
        
        if constraints.get('long_only', False):
            raw_weights = np.maximum(raw_weights, 0)
        if (max_w := constraints.get('max_weight')) is not None:
            raw_weights = np.minimum(raw_weights, max_w)
        if (min_w := constraints.get('min_weight')) is not None:
            raw_weights = np.maximum(raw_weights, min_w)
        if constraints.get('fully_invested', True):
            raw_weights = raw_weights / np.sum(raw_weights)
        
        return raw_weights
    
    def compute_view_contributions(
        self,
        equilibrium_returns: np.ndarray,
        posterior_returns: np.ndarray,
        P: np.ndarray,
        views: List[InvestorView]
    ) -> Dict[int, Dict]:
        """Decompose view contributions to posterior"""
        return_diff = posterior_returns - equilibrium_returns
        contributions = {}
        for i, view in enumerate(views):
            affected = np.where(np.abs(P[i, :]) > 1e-10)[0]
            contributions[i] = {
                'view_type': view.view_type.value,
                'confidence': view.confidence,
                'expected_return': view.expected_return,
                'affected_assets': affected.tolist(),
                'return_impact': return_diff[affected].tolist()
            }
        return contributions
    
    def fit(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        market_weights: np.ndarray,
        views: Optional[List[InvestorView]] = None,
        risk_aversion: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> BlackLittermanResult:
        """Full Black-Litterman optimization pipeline"""
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
        
        n_assets = returns.shape[1]
        delta = risk_aversion or self.default_risk_aversion
        covariance_matrix = np.cov(returns, rowvar=False)
        equilibrium_returns = self.compute_equilibrium_returns(covariance_matrix, market_weights, delta)
        
        if views and len(views) > 0:
            P, Q, Omega = self.construct_view_matrices(views, n_assets, covariance_matrix)
            posterior_returns, posterior_cov = self.compute_posterior(
                equilibrium_returns, covariance_matrix, P, Q, Omega
            )
            view_contributions = self.compute_view_contributions(
                equilibrium_returns, posterior_returns, P, views
            )
        else:
            posterior_returns, posterior_cov = self.compute_posterior_no_views(
                equilibrium_returns, covariance_matrix
            )
            view_contributions = None
        
        optimal_weights = self.optimize_weights(posterior_returns, posterior_cov, delta, constraints)
        
        portfolio_return = optimal_weights @ posterior_returns
        portfolio_vol = np.sqrt(optimal_weights @ posterior_cov @ optimal_weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return BlackLittermanResult(
            posterior_returns=posterior_returns,
            posterior_covariance=posterior_cov,
            optimal_weights=optimal_weights,
            equilibrium_returns=equilibrium_returns,
            risk_aversion=delta,
            tau=self.tau,
            sharpe_ratio=sharpe_ratio,
            view_contributions=view_contributions
        )
    
    def sensitivity_analysis(
        self,
        returns: np.ndarray,
        market_weights: np.ndarray,
        views: List[InvestorView],
        confidence_range: Tuple[float, float] = (0.1, 0.9),
        n_points: int = 10
    ) -> pd.DataFrame:
        """Analyze weight sensitivity to confidence levels"""
        confidences = np.linspace(confidence_range[0], confidence_range[1], n_points)
        results = []
        for conf in confidences:
            adjusted_views = [
                InvestorView(v.view_type, v.assets, v.weights, v.expected_return, conf)
                for v in views
            ]
            result = self.fit(returns, market_weights, adjusted_views)
            results.append({'confidence': conf, **{f'w_{i}': w for i, w in enumerate(result.optimal_weights)}})
        return pd.DataFrame(results)