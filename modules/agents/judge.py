"""
Sharpe-Weighted Judge Agent - Multi-Agent Decision Arbitrator
Author: Erdinc Erdogan
Purpose: Arbitrates between bull and bear agent signals using Sharpe-weighted voting, combining
performance-based weighting with regime-adaptive judgment for final trading decisions.
References:
- Sharpe Ratio and Risk-Adjusted Performance Metrics
- Bayesian Model Averaging
- Softmax Weighted Voting Systems
Usage:
    judge = SharpeWeightedJudge(temperature=1.0)
    result = judge.judge(bull_signal=bull_sig, bear_signal=bear_sig, regime="BULL")
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class AgentType(IntEnum):
    BULL = 0
    BEAR = 1
    NEUTRAL = 2


class JudgmentType(IntEnum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


class ConflictType(IntEnum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


@dataclass
class AgentSignal:
    agent_type: AgentType
    direction: float
    confidence: float
    timestamp: float
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.5


@dataclass
class JudgmentResult:
    judgment: JudgmentType
    direction: float
    confidence: float
    conflict_level: ConflictType
    bull_weight: float
    bear_weight: float
    neutral_weight: float
    regime: str
    entropy_adjusted: bool
    reasoning: str


# ============================================================================
# SHARPE-WEIGHTED JUDGE ENGINE
# ============================================================================

class SharpeWeightedJudge:
    """
    Sharpe-Weighted Judge with rolling performance metrics.
    
    Replaces simple averaging with performance-based weighting.
    High-IR agents have more voting power in final judgment.
    
    Mathematical Foundation:
    w_i = softmax(Sharpe_i / temperature)
    judgment = Σ w_i × signal_i
    
    Key Features:
    1. Rolling Sharpe/Sortino calculation
    2. Conflict detection and resolution
    3. Regime-aware judgment
    4. Entropy-based confidence scaling
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        temperature: float = 2.0,
        min_weight: float = 0.10,
        max_weight: float = 0.60,
        conflict_threshold: float = 0.5,
        entropy_threshold: float = 0.7
    ):
        self.lookback_window = lookback_window
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.conflict_threshold = conflict_threshold
        self.entropy_threshold = entropy_threshold
        
        # Performance tracking
        self.returns = {at: deque(maxlen=lookback_window) for at in AgentType}
        self.sharpe = {at: 0.0 for at in AgentType}
        self.sortino = {at: 0.0 for at in AgentType}
        self.win_rates = {at: 0.5 for at in AgentType}
        self.correct_calls = {at: deque(maxlen=lookback_window) for at in AgentType}
        
        # State
        self.current_regime = "NEUTRAL"
        self.last_judgment: Optional[JudgmentResult] = None
        
    def judge(
        self,
        bull_signal: AgentSignal,
        bear_signal: AgentSignal,
        neutral_signal: AgentSignal,
        regime_probs: np.ndarray = None,
        entropy: float = None,
        obi_signal: float = None
    ) -> JudgmentResult:
        """
        Make judgment based on agent signals with Sharpe weighting.
        
        Args:
            bull_signal: Signal from bull agent
            bear_signal: Signal from bear agent
            neutral_signal: Signal from neutral agent
            regime_probs: Soft clustering regime probabilities
            entropy: Current regime entropy
            obi_signal: Order book imbalance
            
        Returns:
            JudgmentResult with final decision
        """
        signals = [bull_signal, bear_signal, neutral_signal]
        
        # Calculate Sharpe-weighted weights
        weights = self._calculate_weights()
        
        # Detect conflicts
        conflict_level = self._detect_conflict(signals)
        
        # Calculate weighted consensus
        weighted_direction = 0.0
        weighted_confidence = 0.0
        
        for signal in signals:
            w = weights[signal.agent_type]
            weighted_direction += signal.direction * w
            weighted_confidence += signal.confidence * w
        
        # Apply conflict penalty
        if conflict_level >= ConflictType.MODERATE:
            weighted_confidence *= 0.7
        elif conflict_level == ConflictType.MILD:
            weighted_confidence *= 0.85
        
        # Apply entropy gating
        entropy_adjusted = False
        if entropy is not None and entropy > self.entropy_threshold:
            factor = 1.0 - (entropy - self.entropy_threshold) / (1.0 - self.entropy_threshold)
            weighted_confidence *= factor
            entropy_adjusted = True
        
        # Apply OBI confirmation/contradiction
        if obi_signal is not None:
            if np.sign(obi_signal) == np.sign(weighted_direction):
                weighted_confidence *= 1.1  # Confirmation boost
            else:
                weighted_confidence *= 0.9  # Contradiction penalty
        
        # Determine regime
        regime = "NEUTRAL"
        if regime_probs is not None:
            regime_idx = np.argmax(regime_probs)
            regime_names = ["STRONG_BULL", "BULL", "WEAK_BULL", "NEUTRAL", 
                          "WEAK_BEAR", "BEAR", "STRONG_BEAR", "CRISIS"]
            regime = regime_names[regime_idx] if regime_idx < len(regime_names) else "NEUTRAL"
        
        # Clip values
        weighted_direction = np.clip(weighted_direction, -1, 1)
        weighted_confidence = np.clip(weighted_confidence, 0, 1)
        
        # Determine judgment type
        judgment = self._direction_to_judgment(weighted_direction, weighted_confidence)
        
        # Build reasoning
        reasoning = self._build_reasoning(
            signals, weights, conflict_level, entropy_adjusted, regime
        )
        
        result = JudgmentResult(
            judgment=judgment,
            direction=weighted_direction,
            confidence=weighted_confidence,
            conflict_level=conflict_level,
            bull_weight=weights[AgentType.BULL],
            bear_weight=weights[AgentType.BEAR],
            neutral_weight=weights[AgentType.NEUTRAL],
            regime=regime,
            entropy_adjusted=entropy_adjusted,
            reasoning=reasoning
        )
        
        self.last_judgment = result
        return result
    
    def record_outcome(
        self,
        agent_type: AgentType,
        return_value: float,
        was_correct: bool
    ):
        """Record agent performance for weight calculation."""
        self.returns[agent_type].append(return_value)
        self.correct_calls[agent_type].append(1 if was_correct else 0)
        
        # Update Sharpe ratio
        if len(self.returns[agent_type]) >= 10:
            arr = np.array(list(self.returns[agent_type]))
            mean_ret = np.mean(arr)
            std_ret = np.std(arr) + 1e-10
            self.sharpe[agent_type] = (mean_ret / std_ret) * np.sqrt(252)
            
            # Sortino (downside deviation)
            downside = arr[arr < 0]
            if len(downside) > 0:
                downside_std = np.std(downside) + 1e-10
                self.sortino[agent_type] = (mean_ret / downside_std) * np.sqrt(252)
        
        # Update win rate
        if len(self.correct_calls[agent_type]) >= 10:
            self.win_rates[agent_type] = np.mean(list(self.correct_calls[agent_type]))
    
    def _calculate_weights(self) -> Dict[AgentType, float]:
        """Calculate Sharpe-weighted agent weights."""
        # Combine Sharpe and win rate
        combined_scores = {}
        for at in AgentType:
            # 70% Sharpe, 30% win rate
            combined_scores[at] = 0.7 * self.sharpe[at] + 0.3 * (self.win_rates[at] - 0.5) * 2
        
        scores_arr = np.array([combined_scores[at] for at in AgentType])
        
        # Handle all-zero case
        if np.all(scores_arr == 0):
            return {AgentType.BULL: 0.33, AgentType.BEAR: 0.33, AgentType.NEUTRAL: 0.34}
        
        # Softmax with temperature
        exp_scores = np.exp(scores_arr / self.temperature)
        weights = exp_scores / exp_scores.sum()
        
        # Clip to bounds
        result = {}
        for i, at in enumerate(AgentType):
            result[at] = np.clip(weights[i], self.min_weight, self.max_weight)
        
        # Renormalize
        total = sum(result.values())
        return {at: w / total for at, w in result.items()}
    
    def _detect_conflict(self, signals: List[AgentSignal]) -> ConflictType:
        """Detect level of conflict between agent signals."""
        directions = [s.direction for s in signals]
        
        # Check for opposing strong signals
        bull_dir = directions[0]
        bear_dir = directions[1]
        
        # Severe: Bull and Bear both confident but opposite
        if bull_dir > 0.5 and bear_dir < -0.5:
            return ConflictType.SEVERE
        if bull_dir < -0.5 and bear_dir > 0.5:
            return ConflictType.SEVERE
        
        # Moderate: Significant disagreement
        spread = max(directions) - min(directions)
        if spread > 1.0:
            return ConflictType.MODERATE
        elif spread > 0.5:
            return ConflictType.MILD
        
        return ConflictType.NONE
    
    def _direction_to_judgment(
        self, 
        direction: float, 
        confidence: float
    ) -> JudgmentType:
        """Convert direction and confidence to judgment type."""
        # Require minimum confidence for non-HOLD
        if confidence < 0.3:
            return JudgmentType.HOLD
        
        if direction > 0.6:
            return JudgmentType.STRONG_BUY
        elif direction > 0.2:
            return JudgmentType.BUY
        elif direction < -0.6:
            return JudgmentType.STRONG_SELL
        elif direction < -0.2:
            return JudgmentType.SELL
        else:
            return JudgmentType.HOLD
    
    def _build_reasoning(
        self,
        signals: List[AgentSignal],
        weights: Dict[AgentType, float],
        conflict: ConflictType,
        entropy_adjusted: bool,
        regime: str
    ) -> str:
        """Build human-readable reasoning for judgment."""
        parts = []
        
        # Weights
        parts.append(f"Weights: Bull={weights[AgentType.BULL]:.0%}, "
                    f"Bear={weights[AgentType.BEAR]:.0%}, "
                    f"Neutral={weights[AgentType.NEUTRAL]:.0%}")
        
        # Sharpe ratios
        parts.append(f"Sharpe: Bull={self.sharpe[AgentType.BULL]:.2f}, "
                    f"Bear={self.sharpe[AgentType.BEAR]:.2f}, "
                    f"Neutral={self.sharpe[AgentType.NEUTRAL]:.2f}")
        
        # Conflict
        if conflict != ConflictType.NONE:
            parts.append(f"Conflict: {conflict.name}")
        
        # Entropy
        if entropy_adjusted:
            parts.append("Entropy-adjusted")
        
        # Regime
        parts.append(f"Regime: {regime}")
        
        return " | ".join(parts)
    
    def get_diagnostics(self) -> Dict:
        """Get judge diagnostics for monitoring."""
        return {
            "sharpe_ratios": dict(self.sharpe),
            "sortino_ratios": dict(self.sortino),
            "win_rates": dict(self.win_rates),
            "current_weights": self._calculate_weights(),
            "last_judgment": {
                "type": self.last_judgment.judgment.name if self.last_judgment else None,
                "direction": self.last_judgment.direction if self.last_judgment else None,
                "confidence": self.last_judgment.confidence if self.last_judgment else None,
            }
        }


# ============================================================================
# HARDENED JUDGE (Main Class with Full Integration)
# ============================================================================

class HardenedJudge:
    """
    Hardened Judge Agent with all Phase 8A fixes.
    
    Integrates:
    - Sharpe-weighted consensus
    - Conflict detection and resolution
    - Regime awareness
    - Entropy gating
    - OBI signal integration
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        temperature: float = 2.0,
        enable_regime_awareness: bool = True,
        enable_entropy_gating: bool = True
    ):
        self.judge_engine = SharpeWeightedJudge(
            lookback_window=lookback_window,
            temperature=temperature
        )
        self.enable_regime_awareness = enable_regime_awareness
        self.enable_entropy_gating = enable_entropy_gating
        
    def make_judgment(
        self,
        bull_direction: float,
        bull_confidence: float,
        bear_direction: float,
        bear_confidence: float,
        neutral_direction: float,
        neutral_confidence: float,
        regime_probs: np.ndarray = None,
        entropy: float = None,
        obi_signal: float = None,
        timestamp: float = None
    ) -> JudgmentResult:
        """Make final judgment from agent signals."""
        
        if timestamp is None:
            timestamp = time.time()
        
        # Create agent signals
        bull_signal = AgentSignal(
            AgentType.BULL, bull_direction, bull_confidence, timestamp,
            self.judge_engine.sharpe[AgentType.BULL],
            self.judge_engine.sortino[AgentType.BULL],
            self.judge_engine.win_rates[AgentType.BULL]
        )
        
        bear_signal = AgentSignal(
            AgentType.BEAR, bear_direction, bear_confidence, timestamp,
            self.judge_engine.sharpe[AgentType.BEAR],
            self.judge_engine.sortino[AgentType.BEAR],
            self.judge_engine.win_rates[AgentType.BEAR]
        )
        
        neutral_signal = AgentSignal(
            AgentType.NEUTRAL, neutral_direction, neutral_confidence, timestamp,
            self.judge_engine.sharpe[AgentType.NEUTRAL],
            self.judge_engine.sortino[AgentType.NEUTRAL],
            self.judge_engine.win_rates[AgentType.NEUTRAL]
        )
        
        # Apply regime awareness
        if not self.enable_regime_awareness:
            regime_probs = None
        
        # Apply entropy gating
        if not self.enable_entropy_gating:
            entropy = None
        
        return self.judge_engine.judge(
            bull_signal, bear_signal, neutral_signal,
            regime_probs, entropy, obi_signal
        )
    
    def record_outcome(self, agent_type: AgentType, return_value: float, was_correct: bool):
        self.judge_engine.record_outcome(agent_type, return_value, was_correct)
    
    def get_diagnostics(self) -> Dict:
        return self.judge_engine.get_diagnostics()


# Legacy compatibility alias
JudgeAgent = HardenedJudge


# ============================================================================
# PHASE 8C: DEADLOCK RECOVERY INTEGRATION
# Prevents entropy-induced trading paralysis
# ============================================================================

from modules.agents.deadlock_recovery import (
    DeadlockDetector, DeadlockState, RecoveryAction,
    EntropyGateWithRecovery, DeadlockStatus
)

class DeadlockAwareJudge(HardenedJudge):
    """
    Judge with integrated deadlock detection and recovery.
    
    Fixes:
    - ED-003: Deadlock detection mechanism
    - ED-004: Forced execution override
    - ED-001: Raised entropy threshold to 0.8
    - ED-002: Global confidence floor of 0.15
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        temperature: float = 2.0,
        enable_regime_awareness: bool = True,
        enable_entropy_gating: bool = True,
        enable_deadlock_recovery: bool = True
    ):
        super().__init__(
            lookback_window=lookback_window,
            temperature=temperature,
            enable_regime_awareness=enable_regime_awareness,
            enable_entropy_gating=enable_entropy_gating
        )
        
        self.enable_deadlock_recovery = enable_deadlock_recovery
        self.entropy_gate = EntropyGateWithRecovery(
            base_threshold=0.8,  # Raised from 0.7 (fixes ED-001)
            min_confidence_floor=0.15  # Global floor (fixes ED-002)
        )
        
        # High-alpha bypass configuration
        self.alpha_bypass_threshold = 0.8
        self.bypass_entropy_limit = 0.85
        
    def make_judgment_with_recovery(
        self,
        bull_direction: float,
        bull_confidence: float,
        bear_direction: float,
        bear_confidence: float,
        neutral_direction: float,
        neutral_confidence: float,
        regime_probs: np.ndarray = None,
        entropy: float = None,
        obi_signal: float = None,
        alpha_score: float = 0.0,
        timestamp: float = None
    ) -> Tuple[JudgmentResult, DeadlockStatus]:
        """Make judgment with deadlock recovery."""
        
        if timestamp is None:
            timestamp = time.time()
        
        # Get base judgment
        result = self.make_judgment(
            bull_direction, bull_confidence,
            bear_direction, bear_confidence,
            neutral_direction, neutral_confidence,
            regime_probs, entropy, obi_signal, timestamp
        )
        
        # Apply entropy gate with recovery
        if self.enable_deadlock_recovery and entropy is not None:
            adjusted_conf, was_gated, deadlock_status = self.entropy_gate.apply_gate(
                confidence=result.confidence,
                entropy=entropy,
                alpha_score=alpha_score,
                signal_direction=result.direction
            )
            
            # Update result with adjusted confidence
            result = JudgmentResult(
                judgment=result.judgment,
                direction=result.direction,
                confidence=adjusted_conf,
                conflict_level=result.conflict_level,
                bull_weight=result.bull_weight,
                bear_weight=result.bear_weight,
                neutral_weight=result.neutral_weight,
                regime=result.regime,
                entropy_adjusted=was_gated,
                reasoning=result.reasoning + f" | Deadlock: {deadlock_status.state.name}"
            )
            
            return result, deadlock_status
        
        # No deadlock recovery - return dummy status
        return result, DeadlockStatus(
            state=DeadlockState.NORMAL,
            consecutive_blocks=0,
            time_since_last_trade=0,
            current_entropy_threshold=0.8,
            original_entropy_threshold=0.8,
            recovery_action=RecoveryAction.NONE,
            forced_execution_available=False,
            alert_message=""
        )
    
    def check_high_alpha_bypass(
        self,
        alpha_score: float,
        entropy: float,
        confidence: float
    ) -> bool:
        """Check if signal qualifies for high-alpha bypass."""
        return (
            alpha_score >= self.alpha_bypass_threshold and
            entropy < self.bypass_entropy_limit and
            confidence >= 0.5
        )
    
    def force_execution(self) -> bool:
        """Force execution override for deadlock scenarios."""
        return self.entropy_gate.deadlock_detector.force_execution_override()
    
    def record_trade(self):
        """Record successful trade to reset deadlock counters."""
        self.entropy_gate.record_trade()
    
    def get_deadlock_status(self) -> Dict:
        """Get current deadlock detection status."""
        return self.entropy_gate.deadlock_detector.get_statistics()


# Legacy alias with deadlock support
JudgeWithRecovery = DeadlockAwareJudge
