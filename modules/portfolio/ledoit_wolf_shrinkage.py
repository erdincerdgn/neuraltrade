"""
Ledoit-Wolf Shrinkage Estimator for Covariance Matrices
Author: Erdinc Erdogan
Purpose: Shrinks sample covariance toward scaled identity target to handle singular/ill-conditioned matrices and reduce estimation error.
References:
- Ledoit & Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix"
- Optimal Shrinkage Intensity Estimation
- Σ_shrunk = δ * F + (1 - δ) * S
Usage:
    shrinkage = LedoitWolfShrinkage()
    result = shrinkage.estimate(returns)
    robust_cov = result.covariance
"""

# ============================================================================
# LEDOIT-WOLF SHRINKAGE ESTIMATOR - Robust Covariance Estimation
# Handles singular/ill-conditioned covariance matrices
# ============================================================================

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ShrinkageResult:
    """Result of shrinkage estimation."""
    covariance: np.ndarray
    shrinkage_intensity: float
    was_singular: bool
    condition_number: float
    method_used: str


class LedoitWolfShrinkage:
    """
    Ledoit-Wolf Shrinkage Estimator for covariance matrices.
    
    Shrinks sample covariance toward a structured target (scaled identity)
    to improve conditioning and reduce estimation error.
    
    Mathematical Foundation:
    Σ_shrunk = δ * F + (1 - δ) * S
    
    Where:
    - S = sample covariance
    - F = shrinkage target (scaled identity)
    - δ = optimal shrinkage intensity
    
    The optimal δ minimizes expected loss under Frobenius norm.
    
    Usage:
        shrinkage = LedoitWolfShrinkage()
        result = shrinkage.estimate(returns)
        cov = result.covariance
    """
    
    def __init__(
        self,
        min_shrinkage: float = 0.0,
        max_shrinkage: float = 1.0,
        fallback_shrinkage: float = 0.5,
        condition_threshold: float = 1e10,
        eigenvalue_floor: float = 1e-8
    ):
        self.min_shrinkage = min_shrinkage
        self.max_shrinkage = max_shrinkage
        self.fallback_shrinkage = fallback_shrinkage
        self.condition_threshold = condition_threshold
        self.eigenvalue_floor = eigenvalue_floor
    
    def estimate(
        self, 
        returns: np.ndarray,
        sample_cov: Optional[np.ndarray] = None
    ) -> ShrinkageResult:
        """
        Estimate shrunk covariance matrix.
        
        Args:
            returns: T x N matrix of returns (T observations, N assets)
            sample_cov: Optional pre-computed sample covariance
            
        Returns:
            ShrinkageResult with shrunk covariance and diagnostics
        """
        returns = np.asarray(returns)
        
        # Handle 1D input
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        T, N = returns.shape
        
        # Compute sample covariance if not provided
        if sample_cov is None:
            sample_cov = np.cov(returns, rowvar=False)
            if sample_cov.ndim == 0:
                sample_cov = np.array([[sample_cov]])
        
        # Check for singularity
        was_singular = False
        try:
            eigenvalues = np.linalg.eigvalsh(sample_cov)
            condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-15)
            if eigenvalues.min() < self.eigenvalue_floor or condition_number > self.condition_threshold:
                was_singular = True
        except np.linalg.LinAlgError:
            was_singular = True
            condition_number = np.inf
        
        # Compute optimal shrinkage intensity
        if was_singular:
            # Use fallback for singular matrices
            shrinkage_intensity = self.fallback_shrinkage
            method_used = "fallback_singular"
        else:
            shrinkage_intensity = self._compute_optimal_shrinkage(returns, sample_cov)
            method_used = "ledoit_wolf"
        
        # Clip shrinkage intensity
        shrinkage_intensity = np.clip(
            shrinkage_intensity, 
            self.min_shrinkage, 
            self.max_shrinkage
        )
        
        # Compute shrinkage target (scaled identity)
        mu = np.trace(sample_cov) / N
        target = mu * np.eye(N)
        
        # Apply shrinkage
        shrunk_cov = (
            shrinkage_intensity * target + 
            (1 - shrinkage_intensity) * sample_cov
        )
        
        # Ensure positive definiteness
        shrunk_cov = self._ensure_positive_definite(shrunk_cov)
        
        # Recompute condition number
        try:
            eigenvalues = np.linalg.eigvalsh(shrunk_cov)
            final_condition = eigenvalues.max() / max(eigenvalues.min(), 1e-15)
        except:
            final_condition = condition_number
        
        return ShrinkageResult(
            covariance=shrunk_cov,
            shrinkage_intensity=shrinkage_intensity,
            was_singular=was_singular,
            condition_number=final_condition,
            method_used=method_used
        )
    
    def _compute_optimal_shrinkage(
        self, 
        returns: np.ndarray, 
        sample_cov: np.ndarray
    ) -> float:
        """
        Compute optimal Ledoit-Wolf shrinkage intensity.
        
        Based on Ledoit & Wolf (2004) "A Well-Conditioned Estimator 
        for Large-Dimensional Covariance Matrices"
        """
        T, N = returns.shape
        
        # Demean returns
        returns_centered = returns - returns.mean(axis=0)
        
        # Compute shrinkage target scale
        mu = np.trace(sample_cov) / N
        
        # Compute delta (sum of squared off-diagonal elements)
        delta = np.sum((sample_cov - mu * np.eye(N)) ** 2)
        
        # Compute beta (estimation error)
        # This is the key Ledoit-Wolf formula
        beta = 0.0
        for t in range(T):
            x_t = returns_centered[t:t+1].T  # Column vector
            outer = x_t @ x_t.T
            beta += np.sum((outer - sample_cov) ** 2)
        beta /= T ** 2
        
        # Compute gamma
        gamma = np.sum((sample_cov - mu * np.eye(N)) ** 2)
        
        # Optimal shrinkage intensity
        kappa = (beta - gamma / T) / delta if delta > 0 else 0
        shrinkage = max(0, min(1, kappa))
        
        return shrinkage
    
    def _ensure_positive_definite(self, cov: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Floor eigenvalues
            eigenvalues = np.maximum(eigenvalues, self.eigenvalue_floor)
            
            # Reconstruct
            cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # Ensure symmetry
            cov = (cov + cov.T) / 2
            
        except np.linalg.LinAlgError:
            # Fallback: add ridge
            cov = cov + self.eigenvalue_floor * np.eye(cov.shape[0])
        
        return cov
    
    def shrink_existing(
        self, 
        sample_cov: np.ndarray,
        shrinkage_intensity: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply shrinkage to an existing covariance matrix.
        
        Args:
            sample_cov: Sample covariance matrix
            shrinkage_intensity: Optional manual shrinkage (uses fallback if None)
            
        Returns:
            Shrunk covariance matrix
        """
        N = sample_cov.shape[0]
        
        if shrinkage_intensity is None:
            shrinkage_intensity = self.fallback_shrinkage
        
        # Compute target
        mu = np.trace(sample_cov) / N
        target = mu * np.eye(N)
        
        # Apply shrinkage
        shrunk = (
            shrinkage_intensity * target + 
            (1 - shrinkage_intensity) * sample_cov
        )
        
        return self._ensure_positive_definite(shrunk)
