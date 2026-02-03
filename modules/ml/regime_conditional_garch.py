"""
Regime-Conditional GARCH Volatility Model
Author: Erdinc Erdogan
Purpose: Extends GARCH(1,1) with regime-dependent parameters blended using soft clustering probabilities for volatility forecasting.
References:
- GARCH(1,1) (Bollerslev, 1986)
- Regime-Switching GARCH
- Soft Clustering Parameter Blending
Usage:
    model = RegimeConditionalGARCH(n_regimes=5)
    result = model.forecast(returns, regime_probs, horizon=5)
"""

# ============================================================================
# REGIME-CONDITIONAL GARCH - Volatility Regime Coupling
# Variance targets shift based on soft clustering regime outputs
# Phase 7C: Beyond Tier-1 Enhancement
# ============================================================================

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import IntEnum


class VolatilityRegime(IntEnum):
    """Volatility regime classification."""
    LOW = 0
    NORMAL = 1
    ELEVATED = 2
    HIGH = 3
    CRISIS = 4


@dataclass
class RegimeGARCHParams:
    """GARCH parameters for a specific regime."""
    omega: float      # Long-run variance constant
    alpha: float      # ARCH coefficient (shock sensitivity)
    beta: float       # GARCH coefficient (persistence)
    target_vol: float # Target annualized volatility
    
    @property
    def persistence(self) -> float:
        return self.alpha + self.beta
    
    @property
    def long_run_variance(self) -> float:
        if self.persistence >= 1:
            return self.omega / 0.01  # Fallback
        return self.omega / (1 - self.persistence)


@dataclass
class RegimeGARCHResult:
    """Result of regime-conditional GARCH forecast."""
    variance: float
    volatility: float
    regime: VolatilityRegime
    regime_weight: float
    blended_params: RegimeGARCHParams
    forecast_horizon: int
    confidence_interval: Tuple[float, float]


class RegimeConditionalGARCH:
    """
    Regime-Conditional GARCH for volatility forecasting.
    
    Extends standard GARCH(1,1) with regime-dependent parameters.
    Parameters are blended based on soft clustering regime probabilities.
    
    Mathematical Foundation:
    σ²_t = Σ_k w_k * [ω_k + α_k * ε²_{t-1} + β_k * σ²_{t-1}]
    
    Where w_k are regime probabilities from soft clustering.
    
    Key Innovation:
    - Different GARCH parameters for each regime
    - Smooth blending based on regime probabilities
    - Regime-specific variance targets
    
    Usage:
        garch = RegimeConditionalGARCH()
        result = garch.forecast(returns, regime_probs)
    """
    
    # Default regime-specific parameters
    DEFAULT_REGIME_PARAMS = {
        VolatilityRegime.LOW: RegimeGARCHParams(
            omega=0.000001, alpha=0.05, beta=0.90, target_vol=0.10
        ),
        VolatilityRegime.NORMAL: RegimeGARCHParams(
            omega=0.000005, alpha=0.08, beta=0.88, target_vol=0.18
        ),
        VolatilityRegime.ELEVATED: RegimeGARCHParams(
            omega=0.000015, alpha=0.12, beta=0.85, target_vol=0.28
        ),
        VolatilityRegime.HIGH: RegimeGARCHParams(
            omega=0.000040, alpha=0.18, beta=0.78, target_vol=0.45
        ),
        VolatilityRegime.CRISIS: RegimeGARCHParams(
            omega=0.000100, alpha=0.25, beta=0.70, target_vol=0.80
        ),
    }
    
    def __init__(
        self,
        regime_params: Optional[Dict[VolatilityRegime, RegimeGARCHParams]] = None,
        annualization: float = 252,
        min_variance: float = 1e-8,
        max_variance: float = 0.1,
        regime_mapping: Optional[Dict[int, VolatilityRegime]] = None
    ):
        self.regime_params = regime_params or self.DEFAULT_REGIME_PARAMS
        self.annualization = annualization
        self.min_variance = min_variance
        self.max_variance = max_variance
        
        # Default mapping from 8-state HMM to 5 vol regimes
        self.regime_mapping = regime_mapping or {
            0: VolatilityRegime.LOW,
            1: VolatilityRegime.LOW,
            2: VolatilityRegime.NORMAL,
            3: VolatilityRegime.NORMAL,
            4: VolatilityRegime.ELEVATED,
            5: VolatilityRegime.ELEVATED,
            6: VolatilityRegime.HIGH,
            7: VolatilityRegime.CRISIS,
        }
        
        # State
        self.current_variance: float = 0.0002  # ~22% annualized
        self.last_return: float = 0.0
        self.last_result: Optional[RegimeGARCHResult] = None
        
    def forecast(
        self,
        returns: np.ndarray,
        regime_probs: np.ndarray,
        horizon: int = 1
    ) -> RegimeGARCHResult:
        """
        Generate regime-conditional volatility forecast.
        
        Args:
            returns: Recent returns array
            regime_probs: Soft clustering regime probabilities
            horizon: Forecast horizon in periods
            
        Returns:
            RegimeGARCHResult with forecast and diagnostics
        """
        returns = np.asarray(returns)
        regime_probs = self._validate_probs(regime_probs)
        
        # Map regime probs to volatility regimes
        vol_regime_probs = self._map_to_vol_regimes(regime_probs)
        
        # Blend GARCH parameters
        blended_params = self._blend_parameters(vol_regime_probs)
        
        # Get last return for ARCH term
        last_return = returns[-1] if len(returns) > 0 else 0.0
        
        # Update variance using blended GARCH
        new_variance = self._update_variance(
            blended_params, last_return, self.current_variance
        )
        
        # Multi-step forecast
        forecast_variance = self._multi_step_forecast(
            blended_params, new_variance, horizon
        )
        
        # Determine dominant regime
        dominant_regime = self._get_dominant_regime(vol_regime_probs)
        dominant_weight = vol_regime_probs[dominant_regime]
        
        # Calculate confidence interval
        ci = self._calculate_confidence_interval(forecast_variance, horizon)
        
        # Update state
        self.current_variance = new_variance
        self.last_return = last_return
        
        result = RegimeGARCHResult(
            variance=forecast_variance,
            volatility=np.sqrt(forecast_variance * self.annualization),
            regime=dominant_regime,
            regime_weight=dominant_weight,
            blended_params=blended_params,
            forecast_horizon=horizon,
            confidence_interval=ci
        )
        
        self.last_result = result
        return result
    
    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Validate and normalize probabilities."""
        probs = np.asarray(probs)
        probs = np.maximum(probs, 1e-10)
        return probs / probs.sum()
    
    def _map_to_vol_regimes(self, regime_probs: np.ndarray) -> Dict[VolatilityRegime, float]:
        """Map HMM regime probs to volatility regime probs."""
        vol_probs = {regime: 0.0 for regime in VolatilityRegime}
        
        for i, prob in enumerate(regime_probs):
            if i in self.regime_mapping:
                vol_regime = self.regime_mapping[i]
                vol_probs[vol_regime] += prob
        
        # Normalize
        total = sum(vol_probs.values())
        if total > 0:
            vol_probs = {k: v / total for k, v in vol_probs.items()}
        
        return vol_probs
    
    def _blend_parameters(
        self, 
        vol_regime_probs: Dict[VolatilityRegime, float]
    ) -> RegimeGARCHParams:
        """Blend GARCH parameters based on regime probabilities."""
        omega = 0.0
        alpha = 0.0
        beta = 0.0
        target_vol = 0.0
        
        for regime, prob in vol_regime_probs.items():
            params = self.regime_params[regime]
            omega += prob * params.omega
            alpha += prob * params.alpha
            beta += prob * params.beta
            target_vol += prob * params.target_vol
        
        return RegimeGARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            target_vol=target_vol
        )
    
    def _update_variance(
        self,
        params: RegimeGARCHParams,
        last_return: float,
        prev_variance: float
    ) -> float:
        """Update variance using GARCH(1,1) equation."""
        shock = last_return ** 2
        
        new_variance = (
            params.omega +
            params.alpha * shock +
            params.beta * prev_variance
        )
        
        return np.clip(new_variance, self.min_variance, self.max_variance)
    
    def _multi_step_forecast(
        self,
        params: RegimeGARCHParams,
        current_variance: float,
        horizon: int
    ) -> float:
        """Generate multi-step variance forecast."""
        if horizon == 1:
            return current_variance
        
        # For GARCH, multi-step forecast converges to long-run variance
        long_run = params.long_run_variance
        persistence = params.persistence
        
        # Weighted average of current and long-run
        forecast = long_run + (persistence ** horizon) * (current_variance - long_run)
        
        return np.clip(forecast, self.min_variance, self.max_variance)
    
    def _get_dominant_regime(
        self, 
        vol_regime_probs: Dict[VolatilityRegime, float]
    ) -> VolatilityRegime:
        """Get regime with highest probability."""
        return max(vol_regime_probs, key=vol_regime_probs.get)
    
    def _calculate_confidence_interval(
        self,
        variance: float,
        horizon: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for volatility forecast."""
        vol = np.sqrt(variance * self.annualization)
        
        # Approximate CI using chi-squared distribution
        # Wider for longer horizons
        uncertainty = 0.1 + 0.02 * horizon
        
        lower = vol * (1 - uncertainty)
        upper = vol * (1 + uncertainty)
        
        return (max(0.01, lower), min(2.0, upper))
    
    def get_regime_vol_targets(self) -> Dict[str, float]:
        """Get volatility targets for each regime."""
        return {
            regime.name: params.target_vol
            for regime, params in self.regime_params.items()
        }
    
    def reset(self):
        """Reset GARCH state."""
        self.current_variance = 0.0002
        self.last_return = 0.0
        self.last_result = None
