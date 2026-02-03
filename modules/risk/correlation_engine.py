"""
Dynamic Correlation Engine
Author: Erdinc Erdogan
Purpose: Computes static, rolling, EWMA, and DCC-GARCH correlations with copula-based dependence modeling and Random Matrix Theory denoising.
References:
- DCC-GARCH (Engle, 2002)
- Copula Theory (Gaussian, Clayton, Gumbel)
- Ledoit-Wolf Shrinkage
- Random Matrix Theory Denoising
Usage:
    engine = CorrelationEngine()
    result = engine.compute(returns, method=CorrelationMethod.DCC_GARCH)
    cleaned = engine.apply_rmt_cleaning(result.correlation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats, linalg
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class CorrelationMethod(Enum):
    """Correlation calculation methods"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    EWMA = "ewma"
    DCC_GARCH = "dcc_garch"
    ROLLING = "rolling"


class CopulaType(Enum):
    """Copula types for dependence modeling"""
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"


class ShrinkageMethod(Enum):
    """Covariance shrinkage methods"""
    LEDOIT_WOLF = "ledoit_wolf"
    ORACLE_APPROXIMATING = "oracle_approximating"
    CONSTANT_CORRELATION = "constant_correlation"


@dataclass
class CorrelationMatrix:
    """Container for correlation matrix results"""
    correlation: np.ndarray
    method: str
    asset_names: List[str]
    timestamp: Optional[str] = None
    is_positive_definite: bool = True
    condition_number: float = 0.0
    effective_rank: int = 0


@dataclass
class DynamicCorrelation:
    """Dynamic correlation time series"""
    correlations: np.ndarray
    timestamps: np.ndarray
    asset_pair: Tuple[str, str]
    method: str
    half_life: Optional[float] = None


@dataclass
class CopulaResult:
    """Copula fitting results"""
    copula_type: str
    parameters: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    tail_dependence_lower: float
    tail_dependence_upper: float


@dataclass
class CorrelationStressResult:
    """Correlation stress test results"""
    base_correlation: np.ndarray
    stressed_correlation: np.ndarray
    scenario_name: str
    correlation_shift: float
    eigenvalue_impact: np.ndarray
    portfolio_vol_change: float


@dataclass
class RMTCleaningResult:
    """Random Matrix Theory cleaning results"""
    original_correlation: np.ndarray
    cleaned_correlation: np.ndarray
    eigenvalues_original: np.ndarray
    eigenvalues_cleaned: np.ndarray
    noise_eigenvalues: int
    signal_eigenvalues: int
    marchenko_pastur_bound: float


@dataclass
class CorrelationRegimeResult:
    """Correlation regime detection results"""
    current_regime: str
    regime_history: np.ndarray
    transition_matrix: np.ndarray
    regime_correlations: Dict[str, np.ndarray]
    regime_probabilities: np.ndarray


class CorrelationEngine(BaseModule):
    """
    Institutional-Grade Correlation Engine.
    Implements comprehensive correlation analysis for portfolio risk management,
    including dynamic correlations, copulas, and stress testing.
    
    Mathematical Framework:
    ----------------------
    
    Pearson Correlation:
        ρ_XY = Cov(X,Y) / (σ_X × σ_Y)
        ρ_XY = Σ(x_i - x̄)(y_i - ȳ) / √[Σ(x_i - x̄)² × Σ(y_i - ȳ)²]
    
    Spearman Rank Correlation:
        ρ_s = 1 - 6×Σd_i² / (n×(n²-1))
        where d_i = rank(x_i) - rank(y_i)
    
    Kendall Tau:
        τ = (C - D) / (n×(n-1)/2)
        where C = concordant pairs, D = discordant pairs
    
    EWMA Correlation (RiskMetrics):
        σ²_x,t = λ×σ²_x,t-1 + (1-λ)×r²_x,t
        Cov_xy,t = λ×Cov_xy,t-1 + (1-λ)×r_x,t×r_y,t
        ρ_xy,t = Cov_xy,t / (σ_x,t × σ_y,t)
    
    DCC-GARCH (Engle, 2002):
        Q_t = (1-a-b)×Q̄ + a×ε_t-1×ε'_t-1 + b×Q_t-1
        R_t = diag(Q_t)^(-1/2) × Q_t × diag(Q_t)^(-1/2)
    
    Gaussian Copula:
        C(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))
        where Φ_ρ is bivariate normal CDF with correlation ρ
    
    Clayton Copula (Lower Tail Dependence):
        C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
        λ_L = 2^(-1/θ) (lower tail dependence)
    
    Ledoit-Wolf Shrinkage:
        Σ_shrunk = α×F + (1-α)×S
        where F = structured estimator, S = sample covariance
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.ewma_lambda: float = self.config.get('ewma_lambda', 0.94)
        self.dcc_a: float = self.config.get('dcc_a', 0.05)
        self.dcc_b: float = self.config.get('dcc_b', 0.93)
        self.min_observations: int = self.config.get('min_observations', 30)
    
    # =========================================================================
    # STATIC CORRELATION METHODS
    # =========================================================================
    
    def compute_correlation_matrix(
        self,
        returns: np.ndarray,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        asset_names: Optional[List[str]] = None
    ) -> CorrelationMatrix:
        """
        Compute correlation matrix using specified method.
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        if method == CorrelationMethod.PEARSON:
            corr_matrix = np.corrcoef(returns, rowvar=False)
        elif method == CorrelationMethod.SPEARMAN:
            corr_matrix, _ = stats.spearmanr(returns)
            if n_assets == 2:
                corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
        elif method == CorrelationMethod.KENDALL:
            corr_matrix = np.zeros((n_assets, n_assets))
            for i in range(n_assets):
                for j in range(i, n_assets):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        tau, _ = stats.kendalltau(returns[:, i], returns[:, j])
                        corr_matrix[i, j] = tau
                        corr_matrix[j, i] = tau
        else:
            corr_matrix = np.corrcoef(returns, rowvar=False)
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        is_pd = np.all(eigenvalues > 0)
        
        # Condition number
        cond_number = np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)
        
        # Effective rank
        eigenvalues_normalized = eigenvalues / np.sum(eigenvalues)
        entropy = -np.sum(eigenvalues_normalized * np.log(eigenvalues_normalized + 1e-10))
        effective_rank = int(np.exp(entropy))
        
        return CorrelationMatrix(
            correlation=corr_matrix,
            method=method.value,
            asset_names=asset_names,
            is_positive_definite=is_pd,
            condition_number=float(cond_number),
            effective_rank=effective_rank
        )
    
    def compute_covariance_matrix(
        self,
        returns: np.ndarray,
        annualization_factor: int = 252
    ) -> np.ndarray:
        """
        Compute annualized covariance matrix.
        """
        returns = np.asarray(returns)
        cov_matrix = np.cov(returns, rowvar=False) * annualization_factor
        return cov_matrix
    
    # =========================================================================
    # DYNAMIC CORRELATION METHODS
    # =========================================================================
    
    def compute_rolling_correlation(
        self,
        returns: np.ndarray,
        window: int = 60,
        asset_i: int = 0,
        asset_j: int = 1,
        asset_names: Optional[List[str]] = None
    ) -> DynamicCorrelation:
        """
        Compute rolling window correlation between two assets.
        """
        returns = np.asarray(returns)
        n_obs = len(returns)
        
        if asset_names is None:
            asset_names = [f"Asset_{asset_i}", f"Asset_{asset_j}"]
        
        correlations = np.full(n_obs, np.nan)
        
        for t in range(window - 1, n_obs):
            window_returns = returns[t - window + 1:t + 1]
            corr = np.corrcoef(window_returns[:, asset_i], window_returns[:, asset_j])[0, 1]
            correlations[t] = corr
        
        return DynamicCorrelation(
            correlations=correlations,
            timestamps=np.arange(n_obs),
            asset_pair=(asset_names[0], asset_names[1]),
            method=CorrelationMethod.ROLLING.value,
            half_life=None
        )
    
    def compute_ewma_correlation(
        self,
        returns: np.ndarray,
        lambda_param: Optional[float] = None,
        asset_i: int = 0,
        asset_j: int = 1,
        asset_names: Optional[List[str]] = None
    ) -> DynamicCorrelation:
        """
        Compute EWMA (Exponentially Weighted Moving Average) correlation.
        
        σ²_t = λ×σ²_{t-1} + (1-λ)×r²_t
        Cov_t = λ×Cov_{t-1} + (1-λ)×r_x,t×r_y,t
        ρ_t = Cov_t / (σ_x,t × σ_y,t)
        """
        returns = np.asarray(returns)
        lam = lambda_param or self.ewma_lambda
        n_obs = len(returns)
        
        if asset_names is None:
            asset_names = [f"Asset_{asset_i}", f"Asset_{asset_j}"]
        
        r_x = returns[:, asset_i]
        r_y = returns[:, asset_j]
        
        # Initialize
        var_x = np.zeros(n_obs)
        var_y = np.zeros(n_obs)
        cov_xy = np.zeros(n_obs)
        
        var_x[0] = r_x[0]**2
        var_y[0] = r_y[0]**2
        cov_xy[0] = r_x[0] * r_y[0]
        
        # EWMA recursion
        for t in range(1, n_obs):
            var_x[t] = lam * var_x[t-1] + (1 - lam) * r_x[t-1]**2
            var_y[t] = lam * var_y[t-1] + (1 - lam) * r_y[t-1]**2
            cov_xy[t] = lam * cov_xy[t-1] + (1 - lam) * r_x[t-1] * r_y[t-1]
        
        # Correlation
        correlations = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y) + 1e-10)
        correlations = np.clip(correlations, -1, 1)
        
        # Half-life
        half_life = np.log(0.5) / np.log(lam)
        
        return DynamicCorrelation(
            correlations=correlations,
            timestamps=np.arange(n_obs),
            asset_pair=(asset_names[0], asset_names[1]),
            method=CorrelationMethod.EWMA.value,
            half_life=float(half_life)
        )
    
    def compute_ewma_correlation_matrix(
        self,
        returns: np.ndarray,
        lambda_param: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute full EWMA correlation matrix at final time point.
        """
        returns = np.asarray(returns)
        lam = lambda_param or self.ewma_lambda
        n_obs, n_assets = returns.shape
        
        # Initialize covariance matrix
        cov_matrix = np.outer(returns[0], returns[0])
        
        # EWMA recursion
        for t in range(1, n_obs):
            cov_matrix = lam * cov_matrix + (1 - lam) * np.outer(returns[t-1], returns[t-1])
        
        # Convert to correlation
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        np.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix
    
    def compute_dcc_garch(
        self,
        returns: np.ndarray,
        a: Optional[float] = None,
        b: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute DCC-GARCH dynamic correlations.
        
        Q_t = (1-a-b)×Q̄ + a×ε_{t-1}×ε'_{t-1} + b×Q_{t-1}
        R_t = diag(Q_t)^{-1/2} × Q_t × diag(Q_t)^{-1/2}
        """
        returns = np.asarray(returns)
        a = a or self.dcc_a
        b = b or self.dcc_b
        n_obs, n_assets = returns.shape
        
        # Step 1: Standardize returns using GARCH(1,1) volatilities
        standardized = np.zeros_like(returns)
        for i in range(n_assets):
            vol = self._compute_garch_volatility(returns[:, i])
            standardized[:, i] = returns[:, i] / (vol + 1e-10)
        
        # Step 2: Unconditional correlation matrix
        Q_bar = np.corrcoef(standardized, rowvar=False)
        
        # Step 3: DCC recursion
        Q_t = Q_bar.copy()
        R_series = np.zeros((n_obs, n_assets, n_assets))
        R_series[0] = Q_bar
        
        for t in range(1, n_obs):
            eps_outer = np.outer(standardized[t-1], standardized[t-1])
            Q_t = (1 - a - b) * Q_bar + a * eps_outer + b * Q_t
            # Normalize to correlation
            Q_diag_sqrt = np.sqrt(np.diag(Q_t))
            R_t = Q_t / np.outer(Q_diag_sqrt, Q_diag_sqrt)
            np.fill_diagonal(R_t, 1.0)
            R_series[t] = R_t
        
        return R_series, standardized
    
    def _compute_garch_volatility(
        self,
        returns: np.ndarray,
        omega: float = 0.00001,
        alpha: float = 0.05,
        beta: float = 0.90
    ) -> np.ndarray:
        """Compute GARCH(1,1) volatility series."""
        n = len(returns)
        var = np.zeros(n)
        var[0] = np.var(returns)
        
        for t in range(1, n):
            var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
        
        return np.sqrt(var)
    
    # =========================================================================
    # COPULA METHODS
    # =========================================================================
    
    def fit_gaussian_copula(
        self,
        returns: np.ndarray,
        asset_i: int = 0,
        asset_j: int = 1
    ) -> CopulaResult:
        """
        Fit Gaussian copula to bivariate returns.
        
        C(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))
        """
        u = self._empirical_cdf(returns[:, asset_i])
        v = self._empirical_cdf(returns[:, asset_j])
        
        # Transform to normal
        z_u = stats.norm.ppf(np.clip(u, 0.001, 0.999))
        z_v = stats.norm.ppf(np.clip(v, 0.001, 0.999))
        
        # MLE for correlation
        rho = np.corrcoef(z_u, z_v)[0, 1]
        
        # Log-likelihood
        n = len(u)
        ll = -0.5 * n * np.log(1 - rho**2) - (rho**2 / (2 * (1 - rho**2))) * np.sum(z_u**2 + z_v**2 - 2 * rho * z_u * z_v)
        
        # AIC, BIC
        k = 1  # One parameter
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        
        return CopulaResult(
            copula_type=CopulaType.GAUSSIAN.value,
            parameters={'rho': float(rho)},
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic),
            tail_dependence_lower=0.0,
            tail_dependence_upper=0.0
        )
    
    def fit_clayton_copula(
        self,
        returns: np.ndarray,
        asset_i: int = 0,
        asset_j: int = 1
    ) -> CopulaResult:
        """
        Fit Clayton copula (lower tail dependence).
        
        C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
        λ_L = 2^{-1/θ}
        """
        u = self._empirical_cdf(returns[:, asset_i])
        v = self._empirical_cdf(returns[:, asset_j])
        
        u = np.clip(u, 0.001, 0.999)
        v = np.clip(v, 0.001, 0.999)
        
        def neg_log_likelihood(theta):
            if theta <= 0:
                return 1e10
            
            term1 = (1 + theta) * np.sum(np.log(u * v))
            term2 = -(2 + 1/theta) * np.sum(np.log(u**(-theta) + v**(-theta) - 1))
            term3 = len(u) * np.log(1 + theta)
            
            return -(term1 + term2 + term3)
        
        result = minimize(neg_log_likelihood, x0=1.0, method='L-BFGS-B', bounds=[(0.01, 20)])
        theta = result.x[0]
        
        ll = -result.fun
        n = len(u)
        k = 1
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        
        # Lower tail dependence
        lambda_L = 2**(-1/theta)
        
        return CopulaResult(
            copula_type=CopulaType.CLAYTON.value,
            parameters={'theta': float(theta)},
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic),
            tail_dependence_lower=float(lambda_L),
            tail_dependence_upper=0.0
        )
    
    def fit_gumbel_copula(
        self,
        returns: np.ndarray,
        asset_i: int = 0,
        asset_j: int = 1
    ) -> CopulaResult:
        """
        Fit Gumbel copula (upper tail dependence).
        
        C(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ})
        λ_U = 2 - 2^{1/θ}
        """
        u = self._empirical_cdf(returns[:, asset_i])
        v = self._empirical_cdf(returns[:, asset_j])
        
        u = np.clip(u, 0.001, 0.999)
        v = np.clip(v, 0.001, 0.999)
        
        def neg_log_likelihood(theta):
            if theta < 1:
                return 1e10
            
            neg_ln_u = -np.log(u)
            neg_ln_v = -np.log(v)
            
            A = (neg_ln_u**theta + neg_ln_v**theta)**(1/theta)
            
            term1 = -np.sum(A)
            term2 = (theta - 1) * np.sum(np.log(neg_ln_u) + np.log(neg_ln_v))
            term3 = (1/theta - 2) * np.sum(np.log(neg_ln_u**theta + neg_ln_v**theta))
            term4 = np.sum(np.log(A + theta - 1))
            
            return -(term1 + term2 + term3 + term4)
        
        result = minimize(neg_log_likelihood, x0=2.0, method='L-BFGS-B', bounds=[(1.01, 20)])
        theta = result.x[0]
        
        ll = -result.fun
        n = len(u)
        k = 1
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        
        # Upper tail dependence
        lambda_U = 2 - 2**(1/theta)
        
        return CopulaResult(
            copula_type=CopulaType.GUMBEL.value,
            parameters={'theta': float(theta)},
            log_likelihood=float(ll),
            aic=float(aic),
            bic=float(bic),
            tail_dependence_lower=0.0,
            tail_dependence_upper=float(lambda_U)
        )
    
    def _empirical_cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute empirical CDF (probability integral transform)."""
        n = len(x)
        ranks = stats.rankdata(x)
        return ranks / (n + 1)
    
    # =========================================================================
    # SHRINKAGE ESTIMATORS
    # =========================================================================
    
    def compute_ledoit_wolf_shrinkage(
        self,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix.
        
        Σ_shrunk = α×F + (1-α)×S
        where F = μ×I (scaled identity), S = sample covariance
        """
        returns = np.asarray(returns)
        n, p = returns.shape
        
        # Sample covariance
        X = returns - np.mean(returns, axis=0)
        S = X.T @ X / n
        
        # Shrinkage target: scaled identity
        mu = np.trace(S) / p
        F = mu * np.eye(p)
        
        # Compute optimal shrinkage intensity
        delta = S - F
        
        # Frobenius norms
        delta_sq = np.sum(delta**2)
        # Estimate shrinkage intensity
        X2 = X**2
        sample_var = np.sum(X2.T @ X2) / n - np.sum(S**2)
        
        kappa = (sample_var / n) / delta_sq if delta_sq > 0 else 0
        shrinkage = max(0, min(1, kappa))
        
        # Shrunk covariance
        cov_shrunk = shrinkage * F + (1 - shrinkage) * S
        
        return cov_shrunk, float(shrinkage)
    
    def compute_constant_correlation_shrinkage(
        self,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Shrinkage toward constant correlation matrix.
        """
        returns = np.asarray(returns)
        n, p = returns.shape
        
        # Sample covariance and correlation
        S = np.cov(returns, rowvar=False)
        std_devs = np.sqrt(np.diag(S))
        R = S / np.outer(std_devs, std_devs)
        
        # Average correlation (excluding diagonal)
        mask = ~np.eye(p, dtype=bool)
        rho_bar = np.mean(R[mask])
        
        # Constant correlation target
        F_corr = rho_bar * np.ones((p, p))
        np.fill_diagonal(F_corr, 1.0)
        F = np.outer(std_devs, std_devs) * F_corr
        
        # Compute shrinkage intensity (simplified)
        delta = S - F
        delta_sq = np.sum(delta**2)
        
        X = returns - np.mean(returns, axis=0)
        X2 = X**2
        sample_var = np.sum(X2.T @ X2) / n - np.sum(S**2)
        
        kappa = (sample_var / n) / delta_sq if delta_sq > 0 else 0
        shrinkage = max(0, min(1, kappa))
        
        cov_shrunk = shrinkage * F + (1 - shrinkage) * S
        
        return cov_shrunk, float(shrinkage)
    
    # =========================================================================
    # RANDOM MATRIX THEORY (RMT) CLEANING
    # =========================================================================
    
    def clean_correlation_rmt(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> RMTCleaningResult:
        """
        Clean correlation matrix using Random Matrix Theory.
        
        Removes noise eigenvalues below Marchenko-Pastur bound.
        """
        returns = np.asarray(returns)
        T, N = returns.shape
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(N)]
        
        # Sample correlation matrix
        corr_matrix = np.corrcoef(returns, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Marchenko-Pastur bounds
        q = T / N
        lambda_plus = (1 + 1/np.sqrt(q))**2
        lambda_minus = (1 - 1/np.sqrt(q))**2
        
        # Identify signal vs noise eigenvalues
        signal_mask = eigenvalues > lambda_plus
        noise_mask = ~signal_mask
        
        n_signal = np.sum(signal_mask)
        n_noise = np.sum(noise_mask)
        
        # Clean eigenvalues: replace noise with average
        eigenvalues_cleaned = eigenvalues.copy()
        if n_noise > 0:
            avg_noise = np.mean(eigenvalues[noise_mask])
            eigenvalues_cleaned[noise_mask] = avg_noise
        
        # Reconstruct correlation matrix
        corr_cleaned = eigenvectors @ np.diag(eigenvalues_cleaned) @ eigenvectors.T
        
        # Ensure valid correlation matrix
        np.fill_diagonal(corr_cleaned, 1.0)
        corr_cleaned = np.clip(corr_cleaned, -1, 1)
        
        return RMTCleaningResult(
            original_correlation=corr_matrix,
            cleaned_correlation=corr_cleaned,
            eigenvalues_original=eigenvalues,
            eigenvalues_cleaned=eigenvalues_cleaned,
            noise_eigenvalues=int(n_noise),
            signal_eigenvalues=int(n_signal),
            marchenko_pastur_bound=float(lambda_plus)
        )
    
    # =========================================================================
    # CORRELATION STRESS TESTING
    # =========================================================================
    
    def stress_correlation_matrix(
        self,
        correlation: np.ndarray,
        stress_factor: float = 0.5,
        scenario_name: str = "Crisis"
    ) -> CorrelationStressResult:
        """
        Stress correlation matrix toward higher correlations.
        
        ρ_stressed = ρ + stress_factor × (1 - |ρ|) × sign(ρ)
        """
        corr_stressed = correlation.copy()
        n = correlation.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                rho = correlation[i, j]
                if rho >= 0:
                    rho_stressed = rho + stress_factor * (1 - rho)
                else:
                    rho_stressed = rho - stress_factor * (1 + rho)
                
                corr_stressed[i, j] = rho_stressed
                corr_stressed[j, i] = rho_stressed
        
        # Ensure positive definiteness
        corr_stressed = self._nearest_positive_definite(corr_stressed)
        
        # Eigenvalue analysis
        eig_original = np.linalg.eigvalsh(correlation)
        eig_stressed = np.linalg.eigvalsh(corr_stressed)
        
        # Portfolio volatility impact (equal weight)
        weights = np.ones(n) / n
        vol_original = np.sqrt(weights @ correlation @ weights)
        vol_stressed = np.sqrt(weights @ corr_stressed @ weights)
        vol_change = (vol_stressed - vol_original) / vol_original
        
        return CorrelationStressResult(
            base_correlation=correlation,
            stressed_correlation=corr_stressed,
            scenario_name=scenario_name,
            correlation_shift=float(stress_factor),
            eigenvalue_impact=eig_stressed - eig_original,
            portfolio_vol_change=float(vol_change)
        )
    
    def _nearest_positive_definite(
        self,
        matrix: np.ndarray,
        epsilon: float = 1e-8
    ) -> np.ndarray:
        """Find nearest positive definite matrix."""
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Clip negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, epsilon)
        
        # Reconstruct
        matrix_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Ensure symmetry and unit diagonal
        matrix_pd = (matrix_pd + matrix_pd.T) / 2
        d = np.sqrt(np.diag(matrix_pd))
        matrix_pd = matrix_pd / np.outer(d, d)
        np.fill_diagonal(matrix_pd, 1.0)
        
        return matrix_pd
    
    # =========================================================================
    # HIERARCHICAL CLUSTERING
    # =========================================================================
    
    def compute_hierarchical_correlation(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        method: str = 'ward'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compute hierarchical clustering of correlation matrix.
        
        Returns reordered correlation matrix based on clustering.
        """
        returns = np.asarray(returns)
        n_assets = returns.shape[1]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        # Correlation matrix
        corr_matrix = np.corrcoef(returns, rowvar=False)
        
        # Distance matrix (1 - correlation)
        dist_matrix = np.sqrt(2* (1 - corr_matrix))
        np.fill_diagonal(dist_matrix, 0)
        
        # Hierarchical clustering
        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method=method)
        
        # Get optimal ordering
        from scipy.cluster.hierarchy import leaves_list
        order = leaves_list(linkage_matrix)
        
        # Reorder correlation matrix
        corr_reordered = corr_matrix[np.ix_(order, order)]
        names_reordered = [asset_names[i] for i in order]
        
        return corr_reordered, linkage_matrix, names_reordered
    
    # =========================================================================
    # CORRELATION REGIME DETECTION
    # =========================================================================
    
    def detect_correlation_regimes(
        self,
        returns: np.ndarray,
        n_regimes: int = 2,
        window: int = 60
    ) -> CorrelationRegimeResult:
        """
        Detect correlation regimes using rolling correlation clustering.
        """
        returns = np.asarray(returns)
        n_obs = returns.shape[0]
        n_assets = returns.shape[1]
        
        # Compute rolling correlations
        rolling_corrs = []
        for t in range(window - 1, n_obs):
            window_returns = returns[t - window + 1:t + 1]
            corr = np.corrcoef(window_returns, rowvar=False)
            # Extract upper triangle
            upper_tri = corr[np.triu_indices(n_assets, k=1)]
            rolling_corrs.append(upper_tri)
        
        rolling_corrs = np.array(rolling_corrs)
        
        # K-means clustering
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(rolling_corrs, n_regimes, minit='++')
        
        # Pad labels for initial window
        full_labels = np.full(n_obs, -1)
        full_labels[window - 1:] = labels
        
        # Compute regime-specific correlations
        regime_correlations = {}
        for regime in range(n_regimes):
            regime_mask = full_labels == regime
            if np.sum(regime_mask) > window:
                regime_returns = returns[regime_mask]
                regime_corr = np.corrcoef(regime_returns, rowvar=False)
                regime_correlations[f"Regime_{regime}"] = regime_corr
        
        # Transition matrix
        transitions = np.zeros((n_regimes, n_regimes))
        valid_labels = full_labels[full_labels >= 0]
        for i in range(len(valid_labels) - 1):
            transitions[valid_labels[i], valid_labels[i + 1]] += 1
        
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = transitions / np.maximum(row_sums, 1)
        
        # Regime probabilities
        regime_probs = np.bincount(valid_labels, minlength=n_regimes) / len(valid_labels)
        
        # Current regime
        current_regime = f"Regime_{full_labels[-1]}" if full_labels[-1] >= 0 else "Unknown"
        
        return CorrelationRegimeResult(
            current_regime=current_regime,
            regime_history=full_labels,
            transition_matrix=transition_matrix,
            regime_correlations=regime_correlations,
            regime_probabilities=regime_probs
        )