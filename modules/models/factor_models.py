"""
Factor Models Engine
Author: Erdinc Erdogan
Purpose: Implements CAPM, Fama-French 3/5-Factor, Carhart 4-Factor, APT, and PCA-based statistical factor models for risk decomposition and alpha generation.
References:
- CAPM (Sharpe, 1964)
- Fama-French Factor Models (1992, 2015)
- Carhart Momentum Factor (1997)
- Arbitrage Pricing Theory (Ross, 1976)
Usage:
    engine = FactorModelEngine(model_type=FactorModelType.FAMA_FRENCH_5)
    result = engine.fit(returns, factor_data)
    exposures = engine.get_factor_exposures()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import svd
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class FactorModelType(Enum):
    """Factor model types"""
    CAPM = "capm"
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    CARHART_4 = "carhart_4"
    APT = "apt"
    PCA = "pca"
    FUNDAMENTAL = "fundamental"
    MACROECONOMIC = "macroeconomic"


class RegressionMethod(Enum):
    """Regression methods for factor estimation"""
    OLS = "ols"
    WLS = "wls"
    ROBUST = "robust"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


@dataclass
class FactorExposure:
    """Factor exposure (beta) results"""
    factor_name: str
    beta: float
    t_statistic: float
    p_value: float
    std_error: float
    confidence_interval: Tuple[float, float]


@dataclass
class FactorModelResult:
    """Complete factor model estimation results"""
    model_type: str
    alpha: float
    alpha_t_stat: float
    alpha_p_value: float
    factor_exposures: List[FactorExposure]
    r_squared: float
    adjusted_r_squared: float
    residual_std: float
    f_statistic: float
    f_p_value: float
    durbin_watson: float
    n_observations: int
    factor_names: List[str]
    betas: np.ndarray


@dataclass
class PCAFactorResult:
    """PCA factor analysis results"""
    n_factors: int
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    factor_loadings: np.ndarray
    factor_scores: np.ndarray
    factor_returns: np.ndarray
    asset_names: List[str]


@dataclass
class FactorRiskDecomposition:
    """Factor-based risk decomposition"""
    total_variance: float
    systematic_variance: float
    idiosyncratic_variance: float
    factor_contributions: Dict[str, float]
    r_squared: float
    tracking_error: float


@dataclass
class FactorMimickingPortfolio:
    """Factor mimicking portfolio weights"""
    factor_name: str
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    asset_names: List[str]


@dataclass
class APTResult:
    """Arbitrage Pricing Theory results"""
    factor_risk_premia: np.ndarray
    factor_betas: np.ndarray
    expected_returns: np.ndarray
    pricing_errors: np.ndarray
    r_squared: float
    factor_names: List[str]


@dataclass
class BarraRiskModel:
    """Barra-style risk model results"""
    factor_covariance: np.ndarray
    specific_variance: np.ndarray
    factor_exposures: np.ndarray
    total_risk: float
    factor_risk: float
    specific_risk: float
    factor_names: List[str]
    asset_names: List[str]


class FactorModelsEngine(BaseModule):
    """Institutional-Grade Factor Models Engine.Implements comprehensive factor-based asset pricing and risk models
    for portfolio management and alpha generation.
    
    Mathematical Framework:
    ----------------------
    
    CAPM (Capital Asset Pricing Model):
        E[R_i] - R_f = β_i × (E[R_m] - R_f)
        R_i,t - R_f,t = α_i + β_i × (R_m,t - R_f,t) + ε_i,t
    
    Fama-French Three-Factor Model:
        R_i,t - R_f,t = α_i + β_mkt × MKT_t + β_smb × SMB_t + β_hml × HML_t + ε_i,t
        Where:
        - MKT = Market excess return (R_m - R_f)
        - SMB = Small Minus Big (size factor)
        - HML = High Minus Low (value factor)
    Fama-French Five-Factor Model:
        R_i,t - R_f,t = α_i + β_mkt × MKT_t + β_smb × SMB_t + β_hml × HML_t+ β_rmw × RMW_t + β_cma × CMA_t + ε_i,t
        
        Additional factors:
        - RMW = Robust Minus Weak (profitability)
        - CMA = Conservative Minus Aggressive (investment)
    
    Carhart Four-Factor Model:
        FF3 + β_mom × MOM_t
        - MOM = Momentum factor (winners minus losers)
    Arbitrage Pricing Theory (APT):
        E[R_i] = R_f + Σ_k β_i,k × λ_k
        
        Where λ_k is the risk premium for factor k
    
    PCA Factor Model:
        R = F × Λ' + ε
        
        Where F = factor scores, Λ = factor loadings
        Factors extracted via eigendecomposition of covariance matrix
    
    Barra Risk Model:
        Σ = X × F × X' + Δ
        
        Where:
        - X = factor exposures matrix
        - F = factor covariance matrix
        - Δ = diagonal specific variance matrix
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.risk_free_rate: float = self.config.get('risk_free_rate', 0.02)
        self.annualization_factor: int = self.config.get('annualization_factor', 252)
        self.confidence_level: float = self.config.get('confidence_level', 0.95)
        self.min_observations: int = self.config.get('min_observations', 36)
    
    # =========================================================================
    # CAPM (CAPITAL ASSET PRICING MODEL)
    # =========================================================================
    
    def estimate_capm(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> FactorModelResult:
        """
        Estimate CAPM (Capital Asset Pricing Model).
        
        R_i,t - R_f,t = α + β × (R_m,t - R_f,t) + ε_t
        """
        asset_returns = np.asarray(asset_returns).flatten()
        market_returns = np.asarray(market_returns).flatten()
        
        n_obs = len(asset_returns)
        
        if risk_free_rate is None:
            rf = self.risk_free_rate / self.annualization_factor
            risk_free_rate = np.full(n_obs, rf)
        else:
            risk_free_rate = np.asarray(risk_free_rate).flatten()
        
        # Excess returns
        excess_asset = asset_returns - risk_free_rate
        excess_market = market_returns - risk_free_rate
        
        # OLS regression
        X = np.column_stack([np.ones(n_obs), excess_market])
        y = excess_asset
        
        # Beta estimation: (X'X)^(-1) X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y
        
        alpha = betas[0]
        beta_market = betas[1]
        
        # Residuals and statistics
        y_hat = X @ betas
        residuals = y - y_hat
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - 2)
        
        # Standard errors
        mse = ss_res / (n_obs - 2)
        var_betas = mse * np.diag(XtX_inv)
        se_betas = np.sqrt(var_betas)
        
        # T-statistics and p-values
        t_stats = betas / se_betas
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - 2))
        
        # Confidence intervals
        t_crit = stats.t.ppf(1 - (1 - self.confidence_level) / 2, n_obs - 2)
        
        # F-statistic
        f_stat = (r_squared /1) / ((1 - r_squared) / (n_obs - 2))
        f_p_value = 1 - stats.f.cdf(f_stat, 1, n_obs - 2)
        
        # Durbin-Watson
        dw = np.sum(np.diff(residuals)**2) / ss_res
        
        # Factor exposure
        market_exposure = FactorExposure(
            factor_name="MKT",
            beta=float(beta_market),
            t_statistic=float(t_stats[1]),
            p_value=float(p_values[1]),
            std_error=float(se_betas[1]),
            confidence_interval=(
                float(beta_market - t_crit * se_betas[1]),
                float(beta_market + t_crit * se_betas[1])
            )
        )
        
        return FactorModelResult(
            model_type=FactorModelType.CAPM.value,
            alpha=float(alpha) * self.annualization_factor,
            alpha_t_stat=float(t_stats[0]),
            alpha_p_value=float(p_values[0]),
            factor_exposures=[market_exposure],
            r_squared=float(r_squared),
            adjusted_r_squared=float(adj_r_squared),
            residual_std=float(np.sqrt(mse) * np.sqrt(self.annualization_factor)),
            f_statistic=float(f_stat),
            f_p_value=float(f_p_value),
            durbin_watson=float(dw),
            n_observations=n_obs,
            factor_names=["MKT"],
            betas=np.array([beta_market])
        )
    
    # =========================================================================
    # FAMA-FRENCH THREE-FACTOR MODEL
    # =========================================================================
    
    def estimate_fama_french_3(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        smb_returns: np.ndarray,
        hml_returns: np.ndarray,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> FactorModelResult:
        """
        Estimate Fama-French Three-Factor Model.
        
        R_i,t - R_f,t = α + β_mkt×MKT + β_smb×SMB + β_hml×HML + ε_t
        """
        asset_returns = np.asarray(asset_returns).flatten()
        market_returns = np.asarray(market_returns).flatten()
        smb_returns = np.asarray(smb_returns).flatten()
        hml_returns = np.asarray(hml_returns).flatten()
        
        n_obs = len(asset_returns)
        
        if risk_free_rate is None:
            rf = self.risk_free_rate / self.annualization_factor
            risk_free_rate = np.full(n_obs, rf)
        else:
            risk_free_rate = np.asarray(risk_free_rate).flatten()
        
        # Excess returns
        excess_asset = asset_returns - risk_free_rate
        excess_market = market_returns - risk_free_rate
        
        # Design matrix
        X = np.column_stack([np.ones(n_obs), excess_market, smb_returns, hml_returns])
        y = excess_asset
        
        # OLS estimation
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y
        
        alpha = betas[0]
        beta_mkt = betas[1]
        beta_smb = betas[2]
        beta_hml = betas[3]
        
        # Residuals
        y_hat = X @ betas
        residuals = y - y_hat
        
        # Statistics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        k = 3# Number of factors
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1)
        
        mse = ss_res / (n_obs - k - 1)
        var_betas = mse * np.diag(XtX_inv)
        se_betas = np.sqrt(var_betas)
        
        t_stats = betas / se_betas
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - k - 1))
        
        t_crit = stats.t.ppf(1 - (1 - self.confidence_level) / 2, n_obs - k - 1)
        
        f_stat = (r_squared / k) / ((1 - r_squared) / (n_obs - k - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, k, n_obs - k - 1)
        dw = np.sum(np.diff(residuals)**2) / ss_res
        
        # Factor exposures
        factor_names = ["MKT", "SMB", "HML"]
        factor_betas = [beta_mkt, beta_smb, beta_hml]
        
        exposures = []
        for i, (name, beta) in enumerate(zip(factor_names, factor_betas)):
            idx = i + 1
            exposures.append(FactorExposure(
                factor_name=name,
                beta=float(beta),
                t_statistic=float(t_stats[idx]),
                p_value=float(p_values[idx]),
                std_error=float(se_betas[idx]),
                confidence_interval=(
                    float(beta - t_crit * se_betas[idx]),
                    float(beta + t_crit * se_betas[idx])
                )
            ))
        
        return FactorModelResult(
            model_type=FactorModelType.FAMA_FRENCH_3.value,
            alpha=float(alpha) * self.annualization_factor,
            alpha_t_stat=float(t_stats[0]),
            alpha_p_value=float(p_values[0]),
            factor_exposures=exposures,
            r_squared=float(r_squared),
            adjusted_r_squared=float(adj_r_squared),
            residual_std=float(np.sqrt(mse) * np.sqrt(self.annualization_factor)),
            f_statistic=float(f_stat),
            f_p_value=float(f_p_value),
            durbin_watson=float(dw),
            n_observations=n_obs,
            factor_names=factor_names,
            betas=np.array(factor_betas)
        )
    
    # =========================================================================
    # FAMA-FRENCH FIVE-FACTOR MODEL
    # =========================================================================
    
    def estimate_fama_french_5(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        smb_returns: np.ndarray,
        hml_returns: np.ndarray,
        rmw_returns: np.ndarray,
        cma_returns: np.ndarray,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> FactorModelResult:
        """
        Estimate Fama-French Five-Factor Model.
        
        R_i,t - R_f,t = α + β_mkt×MKT + β_smb×SMB + β_hml×HML 
                      + β_rmw×RMW + β_cma×CMA + ε_t
        """
        asset_returns = np.asarray(asset_returns).flatten()
        n_obs = len(asset_returns)
        
        if risk_free_rate is None:
            rf = self.risk_free_rate / self.annualization_factor
            risk_free_rate = np.full(n_obs, rf)
        else:
            risk_free_rate = np.asarray(risk_free_rate).flatten()
        
        excess_asset = asset_returns - risk_free_rate
        excess_market = np.asarray(market_returns).flatten() - risk_free_rate
        
        # Design matrix
        X = np.column_stack([
            np.ones(n_obs),
            excess_market,
            np.asarray(smb_returns).flatten(),
            np.asarray(hml_returns).flatten(),
            np.asarray(rmw_returns).flatten(),
            np.asarray(cma_returns).flatten()
        ])
        y = excess_asset
        
        # OLS estimation
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y
        
        alpha = betas[0]
        factor_betas = betas[1:]
        
        # Residuals and statistics
        y_hat = X @ betas
        residuals = y - y_hat
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        k = 5
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1)
        
        mse = ss_res / (n_obs - k - 1)
        var_betas = mse * np.diag(XtX_inv)
        se_betas = np.sqrt(var_betas)
        
        t_stats = betas / se_betas
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - k - 1))
        
        t_crit = stats.t.ppf(1 - (1 - self.confidence_level) / 2, n_obs - k - 1)
        
        f_stat = (r_squared / k) / ((1 - r_squared) / (n_obs - k - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, k, n_obs - k - 1)
        
        dw = np.sum(np.diff(residuals)**2) / ss_res
        
        # Factor exposures
        factor_names = ["MKT", "SMB", "HML", "RMW", "CMA"]
        
        exposures = []
        for i, name in enumerate(factor_names):
            idx = i + 1
            exposures.append(FactorExposure(
                factor_name=name,
                beta=float(factor_betas[i]),
                t_statistic=float(t_stats[idx]),
                p_value=float(p_values[idx]),
                std_error=float(se_betas[idx]),
                confidence_interval=(
                    float(factor_betas[i] - t_crit * se_betas[idx]),
                    float(factor_betas[i] + t_crit * se_betas[idx])
                )
            ))
        
        return FactorModelResult(
            model_type=FactorModelType.FAMA_FRENCH_5.value,
            alpha=float(alpha) * self.annualization_factor,
            alpha_t_stat=float(t_stats[0]),
            alpha_p_value=float(p_values[0]),
            factor_exposures=exposures,
            r_squared=float(r_squared),
            adjusted_r_squared=float(adj_r_squared),
            residual_std=float(np.sqrt(mse) * np.sqrt(self.annualization_factor)),
            f_statistic=float(f_stat),
            f_p_value=float(f_p_value),
            durbin_watson=float(dw),
            n_observations=n_obs,
            factor_names=factor_names,
            betas=factor_betas
        )
    
    # =========================================================================
    # CARHART FOUR-FACTOR MODEL
    # =========================================================================
    
    def estimate_carhart_4(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        smb_returns: np.ndarray,
        hml_returns: np.ndarray,
        mom_returns: np.ndarray,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> FactorModelResult:
        """
        Estimate Carhart Four-Factor Model.
        
        R_i,t - R_f,t = α + β_mkt×MKT + β_smb×SMB + β_hml×HML + β_mom×MOM + ε_t
        """
        asset_returns = np.asarray(asset_returns).flatten()
        n_obs = len(asset_returns)
        
        if risk_free_rate is None:
            rf = self.risk_free_rate / self.annualization_factor
            risk_free_rate = np.full(n_obs, rf)
        else:
            risk_free_rate = np.asarray(risk_free_rate).flatten()
        
        excess_asset = asset_returns - risk_free_rate
        excess_market = np.asarray(market_returns).flatten() - risk_free_rate
        
        X = np.column_stack([
            np.ones(n_obs),
            excess_market,
            np.asarray(smb_returns).flatten(),
            np.asarray(hml_returns).flatten(),
            np.asarray(mom_returns).flatten()
        ])
        y = excess_asset
        
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y
        
        alpha = betas[0]
        factor_betas = betas[1:]
        
        y_hat = X @ betas
        residuals = y - y_hat
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        k = 4
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1)
        
        mse = ss_res / (n_obs - k - 1)
        var_betas = mse * np.diag(XtX_inv)
        se_betas = np.sqrt(var_betas)
        
        t_stats = betas / se_betas
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - k - 1))
        
        t_crit = stats.t.ppf(1 - (1 - self.confidence_level) / 2, n_obs - k - 1)
        
        f_stat = (r_squared / k) / ((1 - r_squared) / (n_obs - k - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, k, n_obs - k - 1)
        
        dw = np.sum(np.diff(residuals)**2) / ss_res
        
        factor_names = ["MKT", "SMB", "HML", "MOM"]
        
        exposures = []
        for i, name in enumerate(factor_names):
            idx = i + 1
            exposures.append(FactorExposure(
                factor_name=name,
                beta=float(factor_betas[i]),
                t_statistic=float(t_stats[idx]),
                p_value=float(p_values[idx]),
                std_error=float(se_betas[idx]),
                confidence_interval=(
                    float(factor_betas[i] - t_crit * se_betas[idx]),
                    float(factor_betas[i] + t_crit * se_betas[idx])
                )
            ))
        
        return FactorModelResult(
            model_type=FactorModelType.CARHART_4.value,
            alpha=float(alpha) * self.annualization_factor,
            alpha_t_stat=float(t_stats[0]),
            alpha_p_value=float(p_values[0]),
            factor_exposures=exposures,
            r_squared=float(r_squared),
            adjusted_r_squared=float(adj_r_squared),
            residual_std=float(np.sqrt(mse) * np.sqrt(self.annualization_factor)),
            f_statistic=float(f_stat),
            f_p_value=float(f_p_value),
            durbin_watson=float(dw),
            n_observations=n_obs,
            factor_names=factor_names,
            betas=factor_betas
        )
    
    # =========================================================================
    # PCA STATISTICAL FACTOR MODEL
    # =========================================================================
    
    def estimate_pca_factors(
        self,
        returns: np.ndarray,
        n_factors: Optional[int] = None,
        variance_threshold: float = 0.90,
        asset_names: Optional[List[str]] = None
    ) -> PCAFactorResult:
        """
        Extract statistical factors using Principal Component Analysis.
        
        R = F × Λ' +ε
        
        Where:
        - F = factor scores (T × K)
        - Λ = factor loadings (N × K)
        """
        returns = np.asarray(returns)
        n_obs, n_assets = returns.shape
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        # Standardize returns
        returns_mean = np.mean(returns, axis=0)
        returns_std = np.std(returns, axis=0)
        returns_standardized = (returns - returns_mean) / (returns_std + 1e-10)
        
        # Covariance matrix
        cov_matrix = np.cov(returns_standardized, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Explained variance
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Determine number of factors
        if n_factors is None:
            n_factors = np.searchsorted(cumulative_variance_ratio, variance_threshold) + 1
            n_factors = min(n_factors, n_assets)
        
        # Factor loadings (eigenvectors scaled by sqrt of eigenvalues)
        factor_loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])
        
        # Factor scores
        factor_scores = returns_standardized @ eigenvectors[:, :n_factors]
        
        # Factor returns (unstandardized)
        factor_returns = factor_scores * returns_std.mean()
        
        return PCAFactorResult(
            n_factors=n_factors,
            eigenvalues=eigenvalues,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance_ratio=cumulative_variance_ratio,
            factor_loadings=factor_loadings,
            factor_scores=factor_scores,
            factor_returns=factor_returns,
            asset_names=asset_names
        )
    
    # =========================================================================
    # ARBITRAGE PRICING THEORY (APT)
    # =========================================================================
    
    def estimate_apt(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray,
        factor_names: Optional[List[str]] = None
    ) -> APTResult:
        """
        Estimate Arbitrage Pricing Theory model.
        
        E[R_i] = R_f + Σ_k β_i,k × λ_k
        
        Two-pass regression:
        1. Time-series: Estimate betas for each asset
        2. Cross-sectional: Estimate factor risk premia
        """
        returns = np.asarray(returns)
        factor_returns = np.asarray(factor_returns)
        
        n_obs, n_assets = returns.shape
        n_factors = factor_returns.shape[1]
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        # First pass: Time-series regressions
        betas = np.zeros((n_assets, n_factors))
        
        for i in range(n_assets):
            X = np.column_stack([np.ones(n_obs), factor_returns])
            y = returns[:, i]
            
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            betas[i, :] = coeffs[1:]
        
        # Second pass: Cross-sectional regression
        mean_returns = np.mean(returns, axis=0)
        
        X_cross = np.column_stack([np.ones(n_assets), betas])
        lambdas = np.linalg.lstsq(X_cross, mean_returns, rcond=None)[0]
        
        risk_free_implied = lambdas[0]
        factor_risk_premia = lambdas[1:]
        
        # Expected returns from model
        expected_returns = risk_free_implied + betas @ factor_risk_premia
        
        # Pricing errors
        pricing_errors = mean_returns - expected_returns
        
        # R-squared
        ss_res = np.sum(pricing_errors**2)
        ss_tot = np.sum((mean_returns - np.mean(mean_returns))**2)
        r_squared = 1 - ss_res / ss_tot
        
        return APTResult(
            factor_risk_premia=factor_risk_premia * self.annualization_factor,
            factor_betas=betas,
            expected_returns=expected_returns * self.annualization_factor,
            pricing_errors=pricing_errors * self.annualization_factor,
            r_squared=float(r_squared),
            factor_names=factor_names
        )
    
    # =========================================================================
    # BARRA-STYLE RISK MODEL
    # =========================================================================
    
    def estimate_barra_risk_model(
        self,
        returns: np.ndarray,
        factor_exposures: np.ndarray,
        factor_names: Optional[List[str]] = None,
        asset_names: Optional[List[str]] = None
    ) -> BarraRiskModel:
        """
        Estimate Barra-style multi-factor risk model.
        Σ = X × F × X' + Δ
        
        Where:
        - X = factor exposures (N × K)
        - F = factor covariance (K × K)
        - Δ = specific variance (diagonal)
        """
        returns = np.asarray(returns)
        factor_exposures = np.asarray(factor_exposures)
        
        n_obs, n_assets = returns.shape
        n_factors = factor_exposures.shape[1]
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        # Estimate factor returns via cross-sectional regression
        factor_returns = np.zeros((n_obs, n_factors))
        residuals = np.zeros((n_obs, n_assets))
        
        for t in range(n_obs):
            X = factor_exposures
            y = returns[t, :]
            
            # WLS or OLS
            f_t = np.linalg.lstsq(X, y, rcond=None)[0]
            factor_returns[t, :] = f_t
            residuals[t, :] = y - X @ f_t
        
        # Factor covariance matrix
        factor_cov = np.cov(factor_returns, rowvar=False) * self.annualization_factor
        
        # Specific variance (diagonal)
        specific_var = np.var(residuals, axis=0) * self.annualization_factor
        
        # Total covariance
        systematic_cov = factor_exposures @ factor_cov @ factor_exposures.T
        total_cov = systematic_cov + np.diag(specific_var)
        
        # Risk decomposition (equal-weighted portfolio)
        w = np.ones(n_assets) / n_assets
        
        total_var = w @ total_cov @ w
        factor_var = w @ systematic_cov @ w
        specific_var_port = w @ np.diag(specific_var) @ w
        
        total_risk = np.sqrt(total_var)
        factor_risk = np.sqrt(factor_var)
        specific_risk = np.sqrt(specific_var_port)
        
        return BarraRiskModel(
            factor_covariance=factor_cov,
            specific_variance=specific_var,
            factor_exposures=factor_exposures,
            total_risk=float(total_risk),
            factor_risk=float(factor_risk),
            specific_risk=float(specific_risk),
            factor_names=factor_names,
            asset_names=asset_names
        )
    
    # =========================================================================
    # FACTOR RISK DECOMPOSITION
    # =========================================================================
    
    def decompose_factor_risk(
        self,
        weights: np.ndarray,
        factor_betas: np.ndarray,
        factor_covariance: np.ndarray,
        specific_variance: np.ndarray,
        factor_names: Optional[List[str]] = None
    ) -> FactorRiskDecomposition:
        """
        Decompose portfolio risk into factor and specific components.
        
        σ²_p = w'XFX'w + w'Δw
        """
        weights = np.asarray(weights)
        factor_betas = np.asarray(factor_betas)
        factor_covariance = np.asarray(factor_covariance)
        specific_variance = np.asarray(specific_variance)
        
        n_factors = factor_covariance.shape[0]
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        # Portfolio factor exposures
        port_betas = weights @ factor_betas
        
        # Systematic variance
        systematic_var = port_betas @ factor_covariance @ port_betas
        
        # Idiosyncratic variance
        idio_var = weights @ (specific_variance * weights)
        
        # Total variance
        total_var = systematic_var + idio_var
        
        # Factor contributions
        factor_contributions = {}
        for i, name in enumerate(factor_names):
            # Marginal contribution of factor i
            factor_var_i = port_betas[i]**2 * factor_covariance[i, i]
            
            # Cross terms
            cross_terms = 0
            for j in range(n_factors):
                if i != j:
                    cross_terms += port_betas[i] * port_betas[j] * factor_covariance[i, j]
            
            factor_contributions[name] = float(factor_var_i + cross_terms)
        
        # R-squared (systematic / total)
        r_squared = systematic_var / total_var if total_var > 0 else 0
        
        # Tracking error (sqrt of idiosyncratic variance)
        tracking_error = np.sqrt(idio_var)
        
        return FactorRiskDecomposition(
            total_variance=float(total_var),
            systematic_variance=float(systematic_var),
            idiosyncratic_variance=float(idio_var),
            factor_contributions=factor_contributions,
            r_squared=float(r_squared),
            tracking_error=float(tracking_error)
        )
    
    # =========================================================================
    # FACTOR MIMICKING PORTFOLIOS
    # =========================================================================
    
    def construct_factor_mimicking_portfolio(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray,
        factor_name: str = "Factor",
        asset_names: Optional[List[str]] = None
    ) -> FactorMimickingPortfolio:
        """
        Construct factor mimicking portfolio.
        
        Minimize tracking error to factor while being dollar-neutral.
        """
        returns = np.asarray(returns)
        factor_returns = np.asarray(factor_returns).flatten()
        
        n_obs, n_assets = returns.shape
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        # Regression: factor = w'R
        # Minimize ||f - R'w||²
        def objective(w):
            port_returns = returns @ w
            tracking_error = np.sum((port_returns - factor_returns)**2)
            return tracking_error
        
        # Constraints: sum of weights = 0(dollar neutral)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)}]
        
        # Bounds
        bounds = [(-1, 1)] * n_assets
        
        # Initial guess
        w0 = np.zeros(n_assets)
        
        result = minimize(
            objective, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        weights = result.x
        
        # Normalize to unit exposure
        port_returns = returns @ weights
        beta_to_factor = np.cov(port_returns, factor_returns)[0, 1] / np.var(factor_returns)
        
        if abs(beta_to_factor) > 1e-6:
            weights = weights / beta_to_factor
        
        # Portfolio metrics
        port_returns_final = returns @ weights
        expected_return = np.mean(port_returns_final) * self.annualization_factor
        volatility = np.std(port_returns_final) * np.sqrt(self.annualization_factor)
        sharpe = expected_return / volatility if volatility > 0 else 0
        
        return FactorMimickingPortfolio(
            factor_name=factor_name,
            weights=weights,
            expected_return=float(expected_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe),
            asset_names=asset_names
        )
    
    # =========================================================================
    # GENERIC MULTI-FACTOR REGRESSION
    # =========================================================================
    
    def estimate_multi_factor(
        self,
        asset_returns: np.ndarray,
        factor_returns: np.ndarray,
        factor_names: Optional[List[str]] = None,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> FactorModelResult:
        """
        Estimate generic multi-factor model.
        
        R_i,t - R_f,t = α + Σ_k β_k × F_k,t + ε_t
        """
        asset_returns = np.asarray(asset_returns).flatten()
        factor_returns = np.asarray(factor_returns)
        
        if factor_returns.ndim == 1:
            factor_returns = factor_returns.reshape(-1, 1)
        
        n_obs = len(asset_returns)
        n_factors = factor_returns.shape[1]
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        if risk_free_rate is None:
            rf = self.risk_free_rate / self.annualization_factor
            risk_free_rate = np.full(n_obs, rf)
        else:
            risk_free_rate = np.asarray(risk_free_rate).flatten()
        
        excess_asset = asset_returns - risk_free_rate
        
        X = np.column_stack([np.ones(n_obs), factor_returns])
        y = excess_asset
        
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y
        
        alpha = betas[0]
        factor_betas = betas[1:]
        
        y_hat = X @ betas
        residuals = y - y_hat
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        k = n_factors
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k - 1)
        
        mse = ss_res / (n_obs - k - 1)
        var_betas = mse * np.diag(XtX_inv)
        se_betas = np.sqrt(var_betas)
        
        t_stats = betas / se_betas
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - k - 1))
        
        t_crit = stats.t.ppf(1 - (1 - self.confidence_level) / 2, n_obs - k - 1)
        
        f_stat = (r_squared / k) / ((1 - r_squared) / (n_obs - k - 1)) if k > 0 else 0
        f_p_value = 1 - stats.f.cdf(f_stat, k, n_obs - k - 1) if k > 0 else 1
        
        dw = np.sum(np.diff(residuals)**2) / ss_res
        
        exposures = []
        for i, name in enumerate(factor_names):
            idx = i + 1
            exposures.append(FactorExposure(
                factor_name=name,
                beta=float(factor_betas[i]),
                t_statistic=float(t_stats[idx]),
                p_value=float(p_values[idx]),
                std_error=float(se_betas[idx]),
                confidence_interval=(
                    float(factor_betas[i] - t_crit * se_betas[idx]),
                    float(factor_betas[i] + t_crit * se_betas[idx])
                )
            ))
        
        return FactorModelResult(
            model_type="multi_factor",
            alpha=float(alpha) * self.annualization_factor,
            alpha_t_stat=float(t_stats[0]),
            alpha_p_value=float(p_values[0]),
            factor_exposures=exposures,
            r_squared=float(r_squared),
            adjusted_r_squared=float(adj_r_squared),
            residual_std=float(np.sqrt(mse) * np.sqrt(self.annualization_factor)),
            f_statistic=float(f_stat),
            f_p_value=float(f_p_value),
            durbin_watson=float(dw),
            n_observations=n_obs,
            factor_names=factor_names,
            betas=factor_betas
        )
    
    # =========================================================================
    # ROLLING FACTOR ESTIMATION
    # =========================================================================
    
    def estimate_rolling_betas(
        self,
        asset_returns: np.ndarray,
        factor_returns: np.ndarray,
        window: int = 60,
        factor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Estimate rolling factor betas.
        """
        asset_returns = np.asarray(asset_returns).flatten()
        factor_returns = np.asarray(factor_returns)
        
        if factor_returns.ndim == 1:
            factor_returns = factor_returns.reshape(-1, 1)
        
        n_obs = len(asset_returns)
        n_factors = factor_returns.shape[1]
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        # Initialize results
        rolling_betas = np.full((n_obs, n_factors), np.nan)
        rolling_alpha = np.full(n_obs, np.nan)
        rolling_r2 = np.full(n_obs, np.nan)
        
        for t in range(window - 1, n_obs):
            start_idx = t - window + 1
            
            y = asset_returns[start_idx:t + 1]
            X = np.column_stack([np.ones(window), factor_returns[start_idx:t + 1]])
            
            try:
                betas = np.linalg.lstsq(X, y, rcond=None)[0]
                rolling_alpha[t] = betas[0]
                rolling_betas[t, :] = betas[1:]
                y_hat = X @ betas
                ss_res = np.sum((y - y_hat)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                rolling_r2[t] = 1 - ss_res / ss_tot
            except:
                continue
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'Alpha': rolling_alpha * self.annualization_factor,
            'R_Squared': rolling_r2
        })
        
        for i, name in enumerate(factor_names):
            result_df[f'Beta_{name}'] = rolling_betas[:, i]
        
        return result_df