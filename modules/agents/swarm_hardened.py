"""
Hardened Swarm Intelligence Engine - Production-Grade Multi-Agent System
Author: Erdinc Erdogan
Purpose: Production-hardened swarm intelligence implementation with signal validation, freshness
tracking, regime-adaptive consensus, and robust error handling for live trading environments.
References:
- Swarm Intelligence and Collective Decision Making
- Signal Decay and Freshness Models
- Regime-Adaptive Weighted Consensus Algorithms
Usage:
    swarm = HardenedSwarmEngine(consensus_method=ConsensusMethod.SHARPE_WEIGHTED)
    consensus = swarm.get_consensus(signals=[bull_sig, bear_sig], regime="NEUTRAL")
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time
import hashlib


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class AgentType(IntEnum):
    BULL = 0
    BEAR = 1
    NEUTRAL = 2


class RegimeType(IntEnum):
    STRONG_BULL = 0
    BULL = 1
    WEAK_BULL = 2
    NEUTRAL = 3
    WEAK_BEAR = 4
    BEAR = 5
    STRONG_BEAR = 6
    CRISIS = 7


class SignalStatus(IntEnum):
    FRESH = 0
    AGING = 1
    STALE = 2
    EXPIRED = 3


class ConsensusMethod(IntEnum):
    SIMPLE_AVERAGE = 0
    SHARPE_WEIGHTED = 1
    REGIME_ADAPTIVE = 2


@dataclass
class Signal:
    signal_id: str
    agent_type: str
    direction: float
    confidence: float
    timestamp: float
    features_hash: str = ""


@dataclass
class ValidatedSignal:
    signal: Signal
    status: SignalStatus
    age_seconds: float
    decay_factor: float
    adjusted_confidence: float
    adjusted_direction: float
    is_valid: bool


@dataclass
class SwarmConsensus:
    direction: float
    confidence: float
    bull_weight: float
    bear_weight: float
    neutral_weight: float
    regime: str
    entropy_gated: bool
    obi_adjusted: bool
    signals_used: int
    signals_rejected: int


# ============================================================================
# SIGNAL FRESHNESS VALIDATOR (Embedded)
# Implements: S_adj = S_raw × e^(-λ × Δt)
# ============================================================================

class SignalFreshnessValidator:
    """Signal freshness with exponential decay."""
    
    DEFAULT_TTL = {"BULL": 30.0, "BEAR": 30.0, "NEUTRAL": 45.0}
    REGIME_DECAY = {"LOW_VOL": 0.02, "NORMAL": 0.05, "HIGH_VOL": 0.10, "CRISIS": 0.20}
    
    def __init__(self, default_ttl: float = 30.0, decay_rate: float = 0.05):
        self.default_ttl = default_ttl
        self.decay_rate = decay_rate
        self.current_regime = "NORMAL"
        self.recent_hashes: deque = deque(maxlen=100)
        
    def validate(self, signal: Signal, current_time: float = None) -> ValidatedSignal:
        if current_time is None:
            current_time = time.time()
        
        age = current_time - signal.timestamp
        ttl = self.DEFAULT_TTL.get(signal.agent_type, self.default_ttl)
        
        # Check duplicate
        if signal.features_hash and signal.features_hash in self.recent_hashes:
            return ValidatedSignal(signal, SignalStatus.EXPIRED, age, 0, 0, 0, False)
        
        # Determine status
        if age > ttl:
            return ValidatedSignal(signal, SignalStatus.EXPIRED, age, 0, 0, 0, False)
        elif age > ttl * 0.8:
            status = SignalStatus.STALE
        elif age > ttl * 0.5:
            status = SignalStatus.AGING
        else:
            status = SignalStatus.FRESH
        
        # Calculate decay: S_adj = S_raw × e^(-λ × Δt)
        lambda_rate = self.REGIME_DECAY.get(self.current_regime, self.decay_rate)
        decay_factor = np.exp(-lambda_rate * age)
        
        adjusted_conf = signal.confidence * decay_factor
        adjusted_dir = signal.direction * decay_factor
        
        if signal.features_hash:
            self.recent_hashes.append(signal.features_hash)
        
        is_valid = status != SignalStatus.STALE and adjusted_conf > 0.1
        return ValidatedSignal(signal, status, age, decay_factor, adjusted_conf, adjusted_dir, is_valid)
    
    def set_regime(self, regime: str):
        if regime in self.REGIME_DECAY:
            self.current_regime = regime


# ============================================================================
# ADAPTIVE BAYESIAN PRIORS (Embedded)
# Links to soft_clustering regime outputs
# ============================================================================

class AdaptiveBayesianPriors:
    """Regime-conditional Bayesian priors for agent weighting."""
    
    REGIME_PRIORS = {
        RegimeType.STRONG_BULL:  (0.60, 0.15, 0.25),
        RegimeType.BULL:         (0.50, 0.20, 0.30),
        RegimeType.WEAK_BULL:    (0.40, 0.25, 0.35),
        RegimeType.NEUTRAL:      (0.33, 0.33, 0.34),
        RegimeType.WEAK_BEAR:    (0.25, 0.40, 0.35),
        RegimeType.BEAR:         (0.20, 0.50, 0.30),
        RegimeType.STRONG_BEAR:  (0.15, 0.60, 0.25),
        RegimeType.CRISIS:       (0.10, 0.55, 0.35),
    }
    
    def __init__(self, update_rate: float = 0.15, entropy_threshold: float = 0.7):
        self.update_rate = update_rate
        self.entropy_threshold = entropy_threshold
        self.bull_prior = 0.33
        self.bear_prior = 0.33
        self.neutral_prior = 0.34
        self.current_regime = RegimeType.NEUTRAL
        
    def update(self, regime_probs: np.ndarray, entropy: float = None) -> Dict[AgentType, float]:
        regime_probs = np.maximum(regime_probs, 1e-10)
        regime_probs = regime_probs / regime_probs.sum()
        
        # Determine dominant regime
        dominant_idx = np.argmax(regime_probs)
        self.current_regime = RegimeType(dominant_idx)
        confidence = regime_probs[dominant_idx]
        
        # Get target priors
        target_bull, target_bear, target_neutral = self.REGIME_PRIORS[self.current_regime]
        
        # Entropy gating: high entropy -> uniform priors
        if entropy is not None and entropy > self.entropy_threshold:
            factor = (entropy - self.entropy_threshold) / (1.0 - self.entropy_threshold)
            target_bull = target_bull * (1 - factor) + 0.33 * factor
            target_bear = target_bear * (1 - factor) + 0.33 * factor
            target_neutral = target_neutral * (1 - factor) + 0.34 * factor
        
        # Adaptive update rate based on confidence
        rate = self.update_rate * (0.5 + 0.5 * confidence)
        
        # EMA update
        self.bull_prior = rate * target_bull + (1 - rate) * self.bull_prior
        self.bear_prior = rate * target_bear + (1 - rate) * self.bear_prior
        self.neutral_prior = rate * target_neutral + (1 - rate) * self.neutral_prior
        
        # Normalize
        total = self.bull_prior + self.bear_prior + self.neutral_prior
        self.bull_prior /= total
        self.bear_prior /= total
        self.neutral_prior /= total
        
        return {
            AgentType.BULL: self.bull_prior,
            AgentType.BEAR: self.bear_prior,
            AgentType.NEUTRAL: self.neutral_prior
        }


# ============================================================================
# SHARPE-WEIGHTED CONSENSUS ENGINE
# Performance-proven agents have louder voice
# ============================================================================

class SharpeWeightedConsensus:
    """Sharpe-weighted consensus calculation."""
    
    def __init__(self, lookback: int = 100, temperature: float = 2.0):
        self.lookback = lookback
        self.temperature = temperature
        self.returns = {at: deque(maxlen=lookback) for at in AgentType}
        self.sharpe = {at: 0.0 for at in AgentType}
        
    def record(self, agent_type: AgentType, ret: float):
        self.returns[agent_type].append(ret)
        if len(self.returns[agent_type]) >= 10:
            arr = np.array(list(self.returns[agent_type]))
            mean_ret = np.mean(arr)
            std_ret = np.std(arr) + 1e-10
            self.sharpe[agent_type] = (mean_ret / std_ret) * np.sqrt(252)
    
    def get_weights(self, regime_weights: Dict[AgentType, float] = None) -> Dict[AgentType, float]:
        sharpe_arr = np.array([self.sharpe[at] for at in AgentType])
        
        if np.all(sharpe_arr == 0):
            perf_weights = {AgentType.BULL: 0.33, AgentType.BEAR: 0.33, AgentType.NEUTRAL: 0.34}
        else:
            exp_sharpe = np.exp(sharpe_arr / self.temperature)
            weights = exp_sharpe / exp_sharpe.sum()
            perf_weights = {AgentType(i): weights[i] for i in range(3)}
        
        if regime_weights:
            final = {}
            for at in AgentType:
                final[at] = 0.5 * perf_weights[at] + 0.5 * regime_weights.get(at, 0.33)
            total = sum(final.values())
            return {at: w / total for at, w in final.items()}
        
        return perf_weights


# ============================================================================
# HARDENED SWARM INTELLIGENCE (Main Class)
# ============================================================================

class HardenedSwarmIntelligence:
    """
    Hardened Swarm Intelligence with all Phase 8A fixes.
    
    Integrates:
    - AdaptiveBayesianPriors (linked to soft_clustering)
    - SignalFreshnessValidator (TTL + exponential decay)
    - SharpeWeightedConsensus (performance-based weighting)
    - Circuit breaker integration
    - OBI/Entropy signal integration
    """
    
    def __init__(
        self,
        signal_ttl: float = 30.0,
        circuit_breaker_threshold: float = 0.12,
        enable_circuit_breaker: bool = True
    ):
        self.priors = AdaptiveBayesianPriors()
        self.validator = SignalFreshnessValidator(default_ttl=signal_ttl)
        self.sharpe_engine = SharpeWeightedConsensus()
        
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.enable_circuit_breaker = enable_circuit_breaker
        self.is_halted = False
        self.halt_reason = None
        
    def get_consensus(
        self,
        bull_signal: Tuple[float, float],
        bear_signal: Tuple[float, float],
        neutral_signal: Tuple[float, float],
        regime_probs: np.ndarray,
        entropy: float = None,
        obi_signal: float = None,
        drawdown: float = None,
        timestamp: float = None
    ) -> SwarmConsensus:
        """Calculate hardened swarm consensus."""
        
        if timestamp is None:
            timestamp = time.time()
        
        # Circuit breaker check
        if self.enable_circuit_breaker and drawdown and drawdown >= self.circuit_breaker_threshold:
            self.is_halted = True
            self.halt_reason = f"Drawdown {drawdown:.1%}"
            return SwarmConsensus(0, 0, 0, 0, 0, "HALTED", False, False, 0, 0)
        
        # Create and validate signals
        signals = [
            Signal("bull", "BULL", bull_signal[0], bull_signal[1], timestamp),
            Signal("bear", "BEAR", bear_signal[0], bear_signal[1], timestamp),
            Signal("neutral", "NEUTRAL", neutral_signal[0], neutral_signal[1], timestamp),
        ]
        
        validated = [self.validator.validate(s, timestamp) for s in signals]
        valid = [v for v in validated if v.is_valid]
        rejected = len(validated) - len(valid)
        
        if not valid:
            return SwarmConsensus(0, 0, 0, 0, 0, "NO_SIGNAL", False, False, 0, rejected)
        
        # Get regime-adaptive priors
        regime_weights = self.priors.update(regime_probs, entropy)
        
        # Get Sharpe-weighted final weights
        final_weights = self.sharpe_engine.get_weights(regime_weights)
        
        # Calculate weighted consensus
        weighted_dir = 0.0
        weighted_conf = 0.0
        total_weight = 0.0
        
        for vs in valid:
            at = AgentType[vs.signal.agent_type]
            w = final_weights[at]
            weighted_dir += vs.adjusted_direction * w
            weighted_conf += vs.adjusted_confidence * w
            total_weight += w
        
        if total_weight > 0:
            weighted_dir /= total_weight
            weighted_conf /= total_weight
        
        # Entropy gating
        entropy_gated = False
        if entropy and entropy > 0.7:
            weighted_conf *= (1.0 - (entropy - 0.7) / 0.3)
            entropy_gated = True
        
        # OBI adjustment
        obi_adjusted = False
        if obi_signal is not None:
            if np.sign(obi_signal) == np.sign(weighted_dir):
                weighted_conf *= 1.0 + abs(obi_signal) * 0.2
            else:
                weighted_conf *= 1.0 - abs(obi_signal) * 0.1
            obi_adjusted = True
        
        weighted_dir = np.clip(weighted_dir, -1, 1)
        weighted_conf = np.clip(weighted_conf, 0, 1)
        
        return SwarmConsensus(
            direction=weighted_dir,
            confidence=weighted_conf,
            bull_weight=final_weights[AgentType.BULL],
            bear_weight=final_weights[AgentType.BEAR],
            neutral_weight=final_weights[AgentType.NEUTRAL],
            regime=self.priors.current_regime.name,
            entropy_gated=entropy_gated,
            obi_adjusted=obi_adjusted,
            signals_used=len(valid),
            signals_rejected=rejected
        )
    
    def record_outcome(self, agent_type: AgentType, ret: float):
        self.sharpe_engine.record(agent_type, ret)
    
    def set_regime(self, regime: str):
        self.validator.set_regime(regime)
    
    def reset_halt(self):
        self.is_halted = False
        self.halt_reason = None
    
    def get_diagnostics(self) -> Dict:
        return {
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "priors": {
                "bull": self.priors.bull_prior,
                "bear": self.priors.bear_prior,
                "neutral": self.priors.neutral_prior,
            },
            "sharpe": dict(self.sharpe_engine.sharpe),
            "regime": self.priors.current_regime.name,
        }


# Legacy compatibility alias
SwarmIntelligence = HardenedSwarmIntelligence
