"""
Adaptive Momentum System - Volatility-Adaptive EMA Smoother
Author: Erdinc Erdogan
Purpose: Smooths regime probabilities using an adaptive exponential moving average that adjusts
its momentum based on volatility, entropy gradients, and market conditions.
References:
- Exponential Moving Average (EMA) Algorithms
- Volatility-Adaptive Smoothing Techniques
- Information Entropy in Financial Time Series
Usage:
    momentum = AdaptiveMomentumEMA(base_alpha=0.1, min_alpha=0.01, max_alpha=0.5)
    smoothed_probs = momentum.update(raw_probabilities, volatility=0.02)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdaptiveMomentumState:
    """State container for adaptive momentum system."""
    ema_probs: np.ndarray
    momentum: float
    volatility_estimate: float
    entropy_gradient: float
    smoothing_factor: float


class AdaptiveMomentumEMA:
    """
    Adaptive Exponential Moving Average for regime probabilities.
    
    Key Innovation: Momentum adapts to market conditions
    - High volatility → Lower momentum (faster response)
    - Low volatility → Higher momentum (more smoothing)
    - High entropy gradient → Lower momentum (regime changing)
    
    Mathematical Foundation:
    α_adaptive = α_base * (1 - vol_factor) * (1 - entropy_factor)
    
    EMA_t = α_adaptive * P_t + (1 - α_adaptive) * EMA_{t-1}
    
    Usage:
        ema = AdaptiveMomentumEMA(n_states=8)
        smoothed_probs = ema.update(raw_probs, volatility=0.25)
    """
    
    def __init__(
        self,
        n_states: int = 8,
        base_momentum: float = 0.9,
        min_momentum: float = 0.5,
        max_momentum: float = 0.98,
        vol_sensitivity: float = 0.5,
        entropy_sensitivity: float = 0.3,
        baseline_volatility: float = 0.20
    ):
        self.n_states = n_states
        self.base_momentum = base_momentum
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.vol_sensitivity = vol_sensitivity
        self.entropy_sensitivity = entropy_sensitivity
        self.baseline_volatility = baseline_volatility
        
        # State
        self.ema_probs: Optional[np.ndarray] = None
        self.prev_entropy: Optional[float] = None
        self.vol_ema: float = baseline_volatility
        self.current_momentum: float = base_momentum
        
    def update(
        self,
        raw_probs: np.ndarray,
        volatility: Optional[float] = None,
        returns: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, AdaptiveMomentumState]:
        """
        Update EMA with adaptive momentum.
        
        Args:
            raw_probs: Raw state probabilities from detector
            volatility: Optional explicit volatility
            returns: Optional returns for volatility estimation
            
        Returns:
            Tuple of (smoothed_probs, state)
        """
        raw_probs = self._validate_probs(raw_probs)
        
        # Estimate volatility if not provided
        if volatility is None and returns is not None:
            volatility = np.std(returns) * np.sqrt(252)
        elif volatility is None:
            volatility = self.vol_ema
        
        # Update volatility EMA
        self.vol_ema = 0.1 * volatility + 0.9 * self.vol_ema
        
        # Calculate entropy and gradient
        entropy = self._compute_entropy(raw_probs)
        entropy_gradient = 0.0
        if self.prev_entropy is not None:
            entropy_gradient = abs(entropy - self.prev_entropy)
        self.prev_entropy = entropy
        
        # Calculate adaptive momentum
        self.current_momentum = self._compute_adaptive_momentum(
            volatility, entropy_gradient
        )
        
        # Initialize or update EMA
        if self.ema_probs is None:
            self.ema_probs = raw_probs.copy()
        else:
            alpha = 1.0 - self.current_momentum
            self.ema_probs = alpha * raw_probs + self.current_momentum * self.ema_probs
            self.ema_probs = self._validate_probs(self.ema_probs)
        
        # Create state object
        state = AdaptiveMomentumState(
            ema_probs=self.ema_probs.copy(),
            momentum=self.current_momentum,
            volatility_estimate=self.vol_ema,
            entropy_gradient=entropy_gradient,
            smoothing_factor=1.0 - self.current_momentum
        )
        
        return self.ema_probs.copy(), state
    
    def _compute_adaptive_momentum(
        self, 
        volatility: float, 
        entropy_gradient: float
    ) -> float:
        """
        Compute adaptive momentum based on market conditions.
        
        Lower momentum (faster response) when:
        - Volatility is high
        - Entropy is changing rapidly (regime shift)
        """
        # Volatility factor: high vol → lower momentum
        vol_ratio = volatility / self.baseline_volatility
        vol_factor = np.clip(vol_ratio - 1.0, 0, 2) * self.vol_sensitivity
        
        # Entropy factor: high gradient → lower momentum
        entropy_factor = np.clip(entropy_gradient * 5, 0, 1) * self.entropy_sensitivity
        
        # Compute adaptive momentum
        momentum = self.base_momentum * (1 - vol_factor) * (1 - entropy_factor)
        
        return np.clip(momentum, self.min_momentum, self.max_momentum)
    
    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute Shannon entropy."""
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))
    
    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Validate and normalize probabilities."""
        probs = np.asarray(probs)
        probs = np.maximum(probs, 1e-10)
        return probs / probs.sum()
    
    def reset(self):
        """Reset EMA state."""
        self.ema_probs = None
        self.prev_entropy = None
        self.vol_ema = self.baseline_volatility
        self.current_momentum = self.base_momentum
    
    def get_effective_smoothing(self) -> float:
        """Get current effective smoothing factor (1 - momentum)."""
        return 1.0 - self.current_momentum
