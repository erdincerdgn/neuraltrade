"""
Soft Clustering Module - Probabilistic Regime Weighting
Author: Erdinc Erdogan
Purpose: Implements probabilistic regime weighting using Shannon entropy for confidence
scoring and weighted Kelly/CVaR calculations for regime-aware position sizing.
References:
- Shannon Entropy: H(γ) = -Σ γ_t(k) × log(γ_t(k))
- Kelly Criterion with Confidence Adjustment
- Probability-Weighted CVaR Thresholds
Usage:
    clusterer = SoftClusterer(n_regimes=3)
    weights = clusterer.get_regime_weights(observation)
    kelly_adj = clusterer.weighted_kelly_multiplier(base_kelly=0.25)
"""
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# NUMERICAL SAFETY FUNCTIONS
# ============================================================================

def safe_log(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Safe logarithm preventing NaN/Inf."""
    return np.log(np.clip(x, eps, 1.0))

def safe_normalize(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Safely normalize probability distribution."""
    probs = np.maximum(probs, eps)
    return probs / probs.sum()

def validate_probs(probs: np.ndarray, n_states: int) -> np.ndarray:
    """Validate and fix probability array."""
    if probs is None or len(probs) != n_states:
        return np.ones(n_states) / n_states
    probs = np.nan_to_num(probs, nan=1.0/n_states, posinf=1.0, neginf=0.0)
    return safe_normalize(probs)


# ============================================================================
# SHANNON ENTROPY CALCULATOR
# ============================================================================

class ShannonEntropyCalculator:
    """
    Shannon Entropy for Regime Confidence Scoring.
    
    H(γ_t) = -Σ_{k=1}^K γ_t(k) · log(γ_t(k))
    
    Properties:
    - H = 0: Maximum certainty (one state has P=1)
    - H = log(K): Maximum uncertainty (uniform distribution)
    
    Confidence Score:
    C_t = 1 - H(γ_t) / log(K)
    """
    
    def __init__(self, n_states: int = 8):
        self.n_states = n_states
        self.max_entropy = np.log(n_states)  # H_max = log(K)
    
    def compute_entropy(self, probs: np.ndarray) -> float:
        """
        Compute Shannon entropy: H(γ) = -Σ_k γ_k · log(γ_k)
        """
        probs = validate_probs(probs, self.n_states)
        log_probs = safe_log(probs)
        entropy = -np.sum(probs * log_probs)
        return float(np.clip(entropy, 0, self.max_entropy))
    
    def compute_confidence(self, probs: np.ndarray) -> float:
        """
        Compute confidence score: C = 1 - H(γ) / H_max
        
        Returns [0, 1]: 1 = certain, 0 = uncertain
        """
        entropy = self.compute_entropy(probs)
        confidence = 1.0 - entropy / self.max_entropy
        return float(np.clip(confidence, 0.0, 1.0))
    
    def compute_effective_states(self, probs: np.ndarray) -> float:
        """Perplexity = exp(H) = effective number of active states."""
        return float(np.exp(self.compute_entropy(probs)))


# ============================================================================
# REGIME PARAMETER BLENDER
# ============================================================================

class RegimeParameterBlender:
    """
    Soft blending of regime-specific parameters.
    
    param_soft = Σ_k γ_t(k) · param_k
    """
    
    # Per-state parameters (8 regimes)
    VOLATILITIES = np.array([0.10, 0.15, 0.12, 0.08, 0.15, 0.25, 0.40, 0.60])
    CVAR_95 = np.array([0.020, 0.035, 0.025, 0.012, 0.028, 0.050, 0.080, 0.120])
    KELLY_SCALERS = np.array([1.00, 0.50, 0.70, 1.20, 1.00, 0.60, 0.30, 0.10])
    RISK_MULTIPLIERS = np.array([1.0, 1.5, 1.2, 0.8, 1.0, 1.8, 2.5, 4.0])
    
    def __init__(self, n_states: int = 8):
        self.n_states = n_states
    
    def blend(self, probs: np.ndarray, values: np.ndarray) -> float:
        """Generic soft blending: Σ_k γ_k · value_k"""
        probs = validate_probs(probs, self.n_states)
        return float(np.dot(probs, values))
    
    def blend_volatility(self, probs: np.ndarray) -> float:
        return self.blend(probs, self.VOLATILITIES)
    
    def blend_cvar(self, probs: np.ndarray) -> float:
        return self.blend(probs, self.CVAR_95)
    
    def blend_kelly_scaler(self, probs: np.ndarray) -> float:
        return self.blend(probs, self.KELLY_SCALERS)
    
    def blend_risk_multiplier(self, probs: np.ndarray) -> float:
        return self.blend(probs, self.RISK_MULTIPLIERS)


# ============================================================================
# WEIGHTED KELLY MULTIPLIER
# ============================================================================

class WeightedKellyCalculator:
    """
    Confidence-adjusted Kelly fraction calculator.
    
    weighted_kelly_multiplier = confidence · regime_scaler
    
    Final Kelly: f_adj = f_base · weighted_kelly_multiplier
    """
    
    def __init__(self, n_states: int = 8, max_multiplier: float = 1.2):
        self.n_states = n_states
        self.max_multiplier = max_multiplier
        self.entropy_calc = ShannonEntropyCalculator(n_states)
        self.blender = RegimeParameterBlender(n_states)
    
    def compute(self, state_probs: np.ndarray) -> float:
        """
        Compute weighted_kelly_multiplier for orchestrator.
        
        Returns multiplier in [0, max_multiplier]
        """
        probs = validate_probs(state_probs, self.n_states)
        confidence = self.entropy_calc.compute_confidence(probs)
        regime_scaler = self.blender.blend_kelly_scaler(probs)
        
        multiplier = confidence * regime_scaler
        return float(np.clip(multiplier, 0, self.max_multiplier))


# ============================================================================
# WEIGHTED CVAR THRESHOLD
# ============================================================================

class WeightedCVaRCalculator:
    """
    Probability-weighted CVaR threshold calculator.
    
    weighted_cvar_threshold = Σ_k γ_k · CVaR_k · uncertainty_penalty
    
    Increases CVaR (more conservative) when regime is uncertain.
    """
    
    def __init__(self, n_states: int = 8, uncertainty_factor: float = 0.5):
        self.n_states = n_states
        self.uncertainty_factor = uncertainty_factor
        self.entropy_calc = ShannonEntropyCalculator(n_states)
        self.blender = RegimeParameterBlender(n_states)
    
    def compute(self, state_probs: np.ndarray) -> float:
        """
        Compute weighted_cvar_threshold for orchestrator.
        
        Returns CVaR threshold with uncertainty penalty.
        """
        probs = validate_probs(state_probs, self.n_states)
        base_cvar = self.blender.blend_cvar(probs)
        
        # Apply uncertainty penalty
        confidence = self.entropy_calc.compute_confidence(probs)
        uncertainty = 1.0 - confidence
        penalty = 1.0 + uncertainty * self.uncertainty_factor
        
        return float(base_cvar * penalty)


# ============================================================================
# SOFT REGIME RESULT (EXPORT STRUCTURE)
# ============================================================================

@dataclass
class SoftRegimeResult:
    """
    Complete soft regime analysis result.
    
    KEY EXPORTS FOR ORCHESTRATOR:
    - weighted_kelly_multiplier: Use to scale Kelly fraction
    - weighted_cvar_threshold: Use for risk limits
    - confidence_score: Use for signal filtering
    """
    weighted_kelly_multiplier: float
    weighted_cvar_threshold: float
    confidence_score: float
    
    state_probabilities: np.ndarray
    dominant_state: int
    dominant_state_name: str
    effective_states: float
    soft_volatility: float
    risk_multiplier: float
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Export for orchestrator integration."""
        return {
            "weighted_kelly_multiplier": self.weighted_kelly_multiplier,
            "weighted_cvar_threshold": self.weighted_cvar_threshold,
            "confidence_score": self.confidence_score,
            "dominant_state": self.dominant_state,
            "dominant_state_name": self.dominant_state_name,
            "effective_states": self.effective_states,
            "soft_volatility": self.soft_volatility,
            "risk_multiplier": self.risk_multiplier
        }
    
    def is_valid(self) -> bool:
        """Check for NaN/Inf values."""
        vals = [self.weighted_kelly_multiplier, self.weighted_cvar_threshold, 
                self.confidence_score, self.soft_volatility]
        return not any(np.isnan(v) or np.isinf(v) for v in vals)


# ============================================================================
# SOFT REGIME DETECTOR (MAIN CLASS)
# ============================================================================

class SoftRegimeDetector:
    """
    Soft Regime Detection with Probabilistic Weighting.
    
    Main interface for Phase 6 soft clustering.
    Exports weighted_kelly_multiplier and weighted_cvar_threshold.
    
    Usage:
        detector = SoftRegimeDetector()
        result = detector.analyze(state_probs)
        
        # Use in orchestrator
        kelly_mult = result.weighted_kelly_multiplier
        cvar_thresh = result.weighted_cvar_threshold
        confidence = result.confidence_score
    """
    
    STATE_NAMES = [
        "BULL_TREND", "BEAR_TREND", "SIDEWAYS", "LOW_VOL",
        "NORMAL_VOL", "HIGH_VOL", "EXTREME_VOL", "CRISIS"
    ]
    
    def __init__(self, n_states: int = 8, ema_momentum: float = 0.9):
        self.n_states = n_states
        self.ema_momentum = ema_momentum
        
        self.entropy_calc = ShannonEntropyCalculator(n_states)
        self.blender = RegimeParameterBlender(n_states)
        self.kelly_calc = WeightedKellyCalculator(n_states)
        self.cvar_calc = WeightedCVaRCalculator(n_states)
        
        self.ema_probs = None
    
    def analyze(self, state_probs: np.ndarray, smooth: bool = True) -> SoftRegimeResult:
        """
        Perform complete soft regime analysis.
        
        Args:
            state_probs: Raw state probabilities from HMM/Neural-HMM
            smooth: Apply EMA smoothing
            
        Returns:
            SoftRegimeResult with all exports
        """
        probs = validate_probs(state_probs, self.n_states)
        
        # Apply EMA smoothing
        if smooth and self.ema_probs is not None:
            probs = self.ema_momentum * self.ema_probs + (1 - self.ema_momentum) * probs
            probs = safe_normalize(probs)
        self.ema_probs = probs.copy()
        
        # Core metrics
        confidence = self.entropy_calc.compute_confidence(probs)
        effective_states = self.entropy_calc.compute_effective_states(probs)
        
        # KEY EXPORTS
        weighted_kelly = self.kelly_calc.compute(probs)
        weighted_cvar = self.cvar_calc.compute(probs)
        
        # Additional metrics
        soft_vol = self.blender.blend_volatility(probs)
        risk_mult = self.blender.blend_risk_multiplier(probs)
        dominant = int(np.argmax(probs))
        
        return SoftRegimeResult(
            weighted_kelly_multiplier=weighted_kelly,
            weighted_cvar_threshold=weighted_cvar,
            confidence_score=confidence,
            state_probabilities=probs,
            dominant_state=dominant,
            dominant_state_name=self.STATE_NAMES[dominant],
            effective_states=effective_states,
            soft_volatility=soft_vol,
            risk_multiplier=risk_mult
        )
    
    def get_weighted_kelly_multiplier(self, state_probs: np.ndarray) -> float:
        """Quick access for orchestrator."""
        return self.kelly_calc.compute(state_probs)
    
    def get_weighted_cvar_threshold(self, state_probs: np.ndarray) -> float:
        """Quick access for orchestrator."""
        return self.cvar_calc.compute(state_probs)
    
    def get_confidence_score(self, state_probs: np.ndarray) -> float:
        """Quick access for orchestrator."""
        return self.entropy_calc.compute_confidence(state_probs)
    
    def reset(self):
        """Reset EMA state."""
        self.ema_probs = None


# ============================================================================
# ORCHESTRATOR INTEGRATION HELPER
# ============================================================================

class SoftClusteringBridge:
    """
    Bridge class for main_orchestrator.py integration.
    
    Replaces discrete regime calls with soft probability weights.
    """
    
    def __init__(self, n_states: int = 8):
        self.detector = SoftRegimeDetector(n_states)
        self.last_result: Optional[SoftRegimeResult] = None
    
    def update(self, state_probs: np.ndarray) -> Dict:
        """
        Update and return soft parameters for orchestrator.
        
        Returns:
            Dict with weighted_kelly_multiplier, weighted_cvar_threshold, confidence_score
        """
        result = self.detector.analyze(state_probs)
        self.last_result = result
        
        if not result.is_valid():
            return {
                "weighted_kelly_multiplier": 0.5,
                "weighted_cvar_threshold": 0.03,
                "confidence_score": 0.5,
                "valid": False
            }
        
        return {
            "weighted_kelly_multiplier": result.weighted_kelly_multiplier,
            "weighted_cvar_threshold": result.weighted_cvar_threshold,
            "confidence_score": result.confidence_score,
            "dominant_state": result.dominant_state_name,
            "risk_multiplier": result.risk_multiplier,
            "valid": True
        }
    
    def get_position_scale(self, state_probs: np.ndarray) -> float:
        """Get position scaling factor."""
        result = self.detector.analyze(state_probs, smooth=False)
        return result.weighted_kelly_multiplier * result.confidence_score
    
    def get_risk_limit(self, portfolio_value: float, state_probs: np.ndarray) -> float:
        """Get dollar risk limit."""
        cvar = self.detector.get_weighted_cvar_threshold(state_probs)
        return portfolio_value * cvar
