"""
Adaptive Bayesian Priors Engine - Regime-Linked Prior System
Author: Erdinc Erdogan
Purpose: Links Bayesian priors to soft-clustering regime outputs, dynamically adjusting agent
weights based on market regime confidence and entropy levels.
References:
- Bayesian Inference: P(Agent|Regime) ∝ P(Regime|Agent) × P(Agent)
- Regime-Conditional Probability Models
- Online Bayesian Learning
Usage:
    priors = AdaptiveBayesianPriors(update_rate=0.1)
    weights = priors.get_agent_weights(regime=RegimeType.BULL, entropy=0.3)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum


class AgentType(IntEnum):
    """Agent type classification."""
    BULL = 0
    BEAR = 1
    NEUTRAL = 2


class RegimeType(IntEnum):
    """Market regime classification."""
    STRONG_BULL = 0
    BULL = 1
    WEAK_BULL = 2
    NEUTRAL = 3
    WEAK_BEAR = 4
    BEAR = 5
    STRONG_BEAR = 6
    CRISIS = 7


@dataclass
class PriorState:
    """Current state of Bayesian priors."""
    bull_prior: float
    bear_prior: float
    neutral_prior: float
    regime: RegimeType
    regime_confidence: float
    update_rate: float
    last_update_time: float


@dataclass
class AgentWeight:
    """Weight assignment for an agent."""
    agent_type: AgentType
    base_weight: float
    regime_adjusted_weight: float
    performance_weight: float
    final_weight: float
    confidence: float


class AdaptiveBayesianPriors:
    """
    Adaptive Bayesian Priors Engine for Agent Weighting.
    
    Replaces static 0.33/0.33/0.33 priors with regime-conditional
    dynamic priors that adapt to market conditions.
    
    Mathematical Foundation:
    P(Agent|Regime) ∝ P(Regime|Agent) × P(Agent)
    
    Prior Update Rule:
    π_t = α × π_regime + (1-α) × π_{t-1}
    
    Where:
    - π_regime = regime-conditional prior from lookup table
    - α = update_rate (adaptive based on regime confidence)
    
    Key Innovations:
    1. Regime-Conditional Priors: Different base priors per regime
    2. Confidence-Scaled Updates: Higher confidence = faster adaptation
    3. Performance Integration: Historical accuracy affects priors
    4. Entropy Gating: High entropy reduces all priors toward uniform
    
    Usage:
        priors = AdaptiveBayesianPriors()
        weights = priors.get_agent_weights(regime_probs, agent_performance)
    """
    
    # Regime-conditional prior lookup table
    # Format: {regime: (bull_prior, bear_prior, neutral_prior)}
    REGIME_PRIORS = {
        RegimeType.STRONG_BULL:  (0.60, 0.15, 0.25),
        RegimeType.BULL:         (0.50, 0.20, 0.30),
        RegimeType.WEAK_BULL:    (0.40, 0.25, 0.35),
        RegimeType.NEUTRAL:      (0.33, 0.33, 0.34),
        RegimeType.WEAK_BEAR:    (0.25, 0.40, 0.35),
        RegimeType.BEAR:         (0.20, 0.50, 0.30),
        RegimeType.STRONG_BEAR:  (0.15, 0.60, 0.25),
        RegimeType.CRISIS:       (0.10, 0.55, 0.35),  # Favor bear but keep neutral high
    }
    
    def __init__(
        self,
        n_regimes: int = 8,
        base_update_rate: float = 0.15,
        min_update_rate: float = 0.05,
        max_update_rate: float = 0.40,
        entropy_threshold: float = 0.7,
        performance_weight: float = 0.3,
        lookback_window: int = 100,
        min_prior: float = 0.05,
        max_prior: float = 0.80
    ):
        self.n_regimes = n_regimes
        self.base_update_rate = base_update_rate
        self.min_update_rate = min_update_rate
        self.max_update_rate = max_update_rate
        self.entropy_threshold = entropy_threshold
        self.performance_weight = performance_weight
        self.lookback_window = lookback_window
        self.min_prior = min_prior
        self.max_prior = max_prior
        
        # Current priors (start uniform)
        self.bull_prior: float = 0.33
        self.bear_prior: float = 0.33
        self.neutral_prior: float = 0.34
        
        # Performance tracking
        self.bull_performance: deque = deque(maxlen=lookback_window)
        self.bear_performance: deque = deque(maxlen=lookback_window)
        self.neutral_performance: deque = deque(maxlen=lookback_window)
        
        # State tracking
        self.current_regime: RegimeType = RegimeType.NEUTRAL
        self.regime_confidence: float = 0.5
        self.last_update_time: float = 0.0
        self.update_count: int = 0
        
    def update(
        self,
        regime_probs: np.ndarray,
        timestamp: Optional[float] = None,
        entropy: Optional[float] = None
    ) -> PriorState:
        """
        Update priors based on regime probabilities.
        
        Args:
            regime_probs: Soft clustering regime probabilities (8 states)
            timestamp: Current timestamp
            entropy: Optional entropy value for gating
            
        Returns:
            PriorState with updated priors
        """
        regime_probs = self._validate_probs(regime_probs)
        
        # Determine dominant regime
        dominant_regime_idx = np.argmax(regime_probs)
        self.current_regime = RegimeType(dominant_regime_idx)
        self.regime_confidence = regime_probs[dominant_regime_idx]
        
        # Get regime-conditional priors
        target_bull, target_bear, target_neutral = self.REGIME_PRIORS[self.current_regime]
        
        # Calculate adaptive update rate
        update_rate = self._calculate_update_rate(entropy)
        
        # Apply entropy gating (high entropy -> move toward uniform)
        if entropy is not None and entropy > self.entropy_threshold:
            entropy_factor = (entropy - self.entropy_threshold) / (1.0 - self.entropy_threshold)
            target_bull = target_bull * (1 - entropy_factor) + 0.33 * entropy_factor
            target_bear = target_bear * (1 - entropy_factor) + 0.33 * entropy_factor
            target_neutral = target_neutral * (1 - entropy_factor) + 0.34 * entropy_factor
        
        # Exponential moving average update
        self.bull_prior = update_rate * target_bull + (1 - update_rate) * self.bull_prior
        self.bear_prior = update_rate * target_bear + (1 - update_rate) * self.bear_prior
        self.neutral_prior = update_rate * target_neutral + (1 - update_rate) * self.neutral_prior
        
        # Normalize and clip
        self._normalize_priors()
        
        # Update timestamp
        if timestamp is not None:
            self.last_update_time = timestamp
        self.update_count += 1
        
        return PriorState(
            bull_prior=self.bull_prior,
            bear_prior=self.bear_prior,
            neutral_prior=self.neutral_prior,
            regime=self.current_regime,
            regime_confidence=self.regime_confidence,
            update_rate=update_rate,
            last_update_time=self.last_update_time
        )
    
    def get_agent_weights(
        self,
        regime_probs: np.ndarray,
        agent_sharpe: Optional[Dict[AgentType, float]] = None,
        entropy: Optional[float] = None
    ) -> Dict[AgentType, AgentWeight]:
        """
        Get final agent weights combining priors and performance.
        
        Args:
            regime_probs: Current regime probabilities
            agent_sharpe: Rolling Sharpe ratios per agent
            entropy: Current entropy for gating
            
        Returns:
            Dict mapping AgentType to AgentWeight
        """
        # Update priors first
        state = self.update(regime_probs, entropy=entropy)
        
        # Base weights from priors
        base_weights = {
            AgentType.BULL: state.bull_prior,
            AgentType.BEAR: state.bear_prior,
            AgentType.NEUTRAL: state.neutral_prior,
        }
        
        # Calculate performance weights
        perf_weights = self._calculate_performance_weights(agent_sharpe)
        
        # Combine: (1 - perf_weight) * prior + perf_weight * performance
        final_weights = {}
        for agent_type in AgentType:
            base = base_weights[agent_type]
            perf = perf_weights.get(agent_type, base)
            
            # Blend prior and performance
            final = (1 - self.performance_weight) * base + self.performance_weight * perf
            
            # Apply regime confidence scaling
            # Higher confidence = trust the regime-adjusted weight more
            regime_adjusted = base * state.regime_confidence + 0.33 * (1 - state.regime_confidence)
            
            final_weights[agent_type] = AgentWeight(
                agent_type=agent_type,
                base_weight=base,
                regime_adjusted_weight=regime_adjusted,
                performance_weight=perf,
                final_weight=final,
                confidence=state.regime_confidence
            )
        
        # Normalize final weights
        total = sum(w.final_weight for w in final_weights.values())
        if total > 0:
            for agent_type in final_weights:
                final_weights[agent_type].final_weight /= total
        
        return final_weights
    
    def record_performance(
        self,
        agent_type: AgentType,
        return_value: float,
        was_correct: bool
    ):
        """Record agent performance for adaptive weighting."""
        if agent_type == AgentType.BULL:
            self.bull_performance.append((return_value, was_correct))
        elif agent_type == AgentType.BEAR:
            self.bear_performance.append((return_value, was_correct))
        else:
            self.neutral_performance.append((return_value, was_correct))
    
    def _calculate_update_rate(self, entropy: Optional[float]) -> float:
        """Calculate adaptive update rate based on regime confidence and entropy."""
        # Base rate scaled by regime confidence
        rate = self.base_update_rate * (0.5 + 0.5 * self.regime_confidence)
        
        # Entropy adjustment: high entropy = slower updates (more uncertain)
        if entropy is not None:
            entropy_factor = 1.0 - entropy * 0.5  # Range [0.5, 1.0]
            rate *= entropy_factor
        
        return np.clip(rate, self.min_update_rate, self.max_update_rate)
    
    def _calculate_performance_weights(
        self,
        agent_sharpe: Optional[Dict[AgentType, float]]
    ) -> Dict[AgentType, float]:
        """Calculate performance-based weights from Sharpe ratios."""
        if agent_sharpe is None:
            return {AgentType.BULL: 0.33, AgentType.BEAR: 0.33, AgentType.NEUTRAL: 0.34}
        
        # Convert Sharpe to weights using softmax
        sharpe_values = np.array([
            agent_sharpe.get(AgentType.BULL, 0.0),
            agent_sharpe.get(AgentType.BEAR, 0.0),
            agent_sharpe.get(AgentType.NEUTRAL, 0.0),
        ])
        
        # Softmax with temperature
        temperature = 2.0  # Higher = more uniform
        exp_sharpe = np.exp(sharpe_values / temperature)
        weights = exp_sharpe / exp_sharpe.sum()
        
        return {
            AgentType.BULL: weights[0],
            AgentType.BEAR: weights[1],
            AgentType.NEUTRAL: weights[2],
        }
    
    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Validate and normalize probability array."""
        probs = np.asarray(probs)
        probs = np.maximum(probs, 1e-10)
        return probs / probs.sum()
    
    def _normalize_priors(self):
        """Normalize and clip priors."""
        # Clip to bounds
        self.bull_prior = np.clip(self.bull_prior, self.min_prior, self.max_prior)
        self.bear_prior = np.clip(self.bear_prior, self.min_prior, self.max_prior)
        self.neutral_prior = np.clip(self.neutral_prior, self.min_prior, self.max_prior)
        
        # Normalize to sum to 1
        total = self.bull_prior + self.bear_prior + self.neutral_prior
        self.bull_prior /= total
        self.bear_prior /= total
        self.neutral_prior /= total
    
    def get_state(self) -> Dict:
        """Get current state for monitoring."""
        return {
            "bull_prior": self.bull_prior,
            "bear_prior": self.bear_prior,
            "neutral_prior": self.neutral_prior,
            "regime": self.current_regime.name,
            "regime_confidence": self.regime_confidence,
            "update_count": self.update_count,
        }
    
    def reset(self):
        """Reset to uniform priors."""
        self.bull_prior = 0.33
        self.bear_prior = 0.33
        self.neutral_prior = 0.34
        self.current_regime = RegimeType.NEUTRAL
        self.regime_confidence = 0.5
        self.update_count = 0
        self.bull_performance.clear()
        self.bear_performance.clear()
        self.neutral_performance.clear()


class RegimePriorMapper:
    """
    Maps soft clustering outputs to agent priors.
    
    Provides a clean interface between the ML regime detection
    and the agent weighting system.
    """
    
    def __init__(self, priors_engine: AdaptiveBayesianPriors):
        self.priors_engine = priors_engine
        
    def map_regime_to_weights(
        self,
        soft_clustering_output: np.ndarray,
        agent_performance: Optional[Dict] = None,
        entropy: Optional[float] = None,
        obi_signal: Optional[float] = None
    ) -> Tuple[Dict[AgentType, float], Dict]:
        """
        Map soft clustering output to agent weights.
        
        Args:
            soft_clustering_output: 8-state probability distribution
            agent_performance: Dict with agent Sharpe ratios
            entropy: Current regime entropy
            obi_signal: Order book imbalance signal [-1, 1]
            
        Returns:
            Tuple of (weights_dict, diagnostics_dict)
        """
        # Get base weights from priors engine
        agent_weights = self.priors_engine.get_agent_weights(
            soft_clustering_output,
            agent_sharpe=agent_performance,
            entropy=entropy
        )
        
        # Extract final weights
        weights = {
            agent_type: aw.final_weight 
            for agent_type, aw in agent_weights.items()
        }
        
        # Apply OBI adjustment if available
        if obi_signal is not None:
            weights = self._apply_obi_adjustment(weights, obi_signal)
        
        # Build diagnostics
        diagnostics = {
            "regime": self.priors_engine.current_regime.name,
            "regime_confidence": self.priors_engine.regime_confidence,
            "raw_priors": {
                "bull": self.priors_engine.bull_prior,
                "bear": self.priors_engine.bear_prior,
                "neutral": self.priors_engine.neutral_prior,
            },
            "final_weights": weights,
            "obi_applied": obi_signal is not None,
        }
        
        return weights, diagnostics
    
    def _apply_obi_adjustment(
        self,
        weights: Dict[AgentType, float],
        obi_signal: float
    ) -> Dict[AgentType, float]:
        """Apply order book imbalance adjustment to weights."""
        # OBI > 0 = bullish pressure, OBI < 0 = bearish pressure
        obi_factor = 0.1  # Maximum adjustment
        
        adjustment = obi_signal * obi_factor
        
        weights[AgentType.BULL] += adjustment
        weights[AgentType.BEAR] -= adjustment
        
        # Renormalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
