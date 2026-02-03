"""
Causal Gating Engine - Correlation vs Causation Filter
Author: Erdinc Erdogan
Purpose: Filters trading signals by distinguishing causal relationships from mere correlations,
integrating with judge agents to veto signals lacking causal support.
References:
- Pearl (2009) "Causality: Models, Reasoning, and Inference"
- Causal Inference in Financial Trading
- Spurious Correlation Detection Methods
Usage:
    gating = CausalGatingEngine(dag=causal_dag, threshold=0.1)
    result = gating.check_signal(signal, context)
    if result.decision == GatingDecision.ALLOW: execute_trade(signal)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
from enum import IntEnum
import time


class GatingDecision(IntEnum):
    """Causal gating decisions."""
    ALLOW = 0         # Signal has causal support
    VETO = 1          # Signal lacks causal support
    WARN = 2          # Weak causal support
    OVERRIDE = 3      # Manual override


@dataclass
class CausalGatingResult:
    """Result of causal gating check."""
    decision: GatingDecision
    signal_variable: str
    target_variable: str
    has_causal_link: bool
    causal_strength: float
    correlation: float
    spurious_probability: float
    reasoning: str
    timestamp: float


class CausalGate:
    """
    Causal Gate for Signal Validation.
    
    Filters signals based on causal relationships:
    - If signal is based on correlation without causation -> VETO
    - If signal has strong causal support -> ALLOW
    - If causal link is weak or uncertain -> WARN
    
    Integration with Judge:
    - Judge queries CausalGate before executing trades
    - Signals without causal backing are vetoed
    - Prevents trading on spurious correlations
    """
    
    def __init__(
        self,
        causal_discovery_engine=None,
        min_causal_strength: float = 0.3,
        max_spurious_prob: float = 0.5,
        enable_strict_mode: bool = True
    ):
        self.discovery_engine = causal_discovery_engine
        self.min_causal_strength = min_causal_strength
        self.max_spurious_prob = max_spurious_prob
        self.enable_strict_mode = enable_strict_mode
        
        # Gating history
        self.gating_history: List[CausalGatingResult] = []
        self.veto_count: int = 0
        self.allow_count: int = 0
        self.warn_count: int = 0
        
    def set_discovery_engine(self, engine):
        """Set the causal discovery engine."""
        self.discovery_engine = engine
        
    def check_signal(
        self,
        signal_variable: str,
        target_variable: str,
        signal_direction: float,
        signal_confidence: float,
        data: np.ndarray = None
    ) -> CausalGatingResult:
        """
        Check if a signal has causal support.
        
        Args:
            signal_variable: Variable the signal is based on
            target_variable: Variable being predicted
            signal_direction: Direction of signal (-1 to 1)
            signal_confidence: Confidence of signal (0 to 1)
            data: Optional data for correlation check
            
        Returns:
            CausalGatingResult with decision
        """
        if self.discovery_engine is None:
            # No causal engine - allow by default
            return CausalGatingResult(
                decision=GatingDecision.ALLOW,
                signal_variable=signal_variable,
                target_variable=target_variable,
                has_causal_link=True,
                causal_strength=1.0,
                correlation=0.0,
                spurious_probability=0.0,
                reasoning="No causal engine - allowing by default",
                timestamp=time.time()
            )
        
        # Check for causal link
        has_causal_link = self.discovery_engine.has_causal_link(
            signal_variable, target_variable
        )
        
        # Get causal path
        causal_path = self.discovery_engine.get_causal_path(
            signal_variable, target_variable
        )
        
        # Compute causal strength (based on path length and edge strengths)
        causal_strength = self._compute_causal_strength(
            signal_variable, target_variable, causal_path
        )
        
        # Compute correlation (if data provided)
        correlation = 0.0
        if data is not None and self.discovery_engine.current_dag is not None:
            try:
                var_names = self.discovery_engine.current_dag.variable_names
                if signal_variable in var_names and target_variable in var_names:
                    sig_idx = var_names.index(signal_variable)
                    tgt_idx = var_names.index(target_variable)
                    correlation = np.corrcoef(data[:, sig_idx], data[:, tgt_idx])[0, 1]
            except:
                correlation = 0.0
        
        # Compute spurious probability
        spurious_prob = self._compute_spurious_probability(
            has_causal_link, causal_strength, abs(correlation)
        )
        
        # Make decision
        decision, reasoning = self._make_decision(
            has_causal_link, causal_strength, spurious_prob, signal_confidence
        )
        
        result = CausalGatingResult(
            decision=decision,
            signal_variable=signal_variable,
            target_variable=target_variable,
            has_causal_link=has_causal_link,
            causal_strength=causal_strength,
            correlation=correlation,
            spurious_probability=spurious_prob,
            reasoning=reasoning,
            timestamp=time.time()
        )
        
        # Update statistics
        self.gating_history.append(result)
        if decision == GatingDecision.VETO:
            self.veto_count += 1
        elif decision == GatingDecision.ALLOW:
            self.allow_count += 1
        else:
            self.warn_count += 1
        
        return result
    
    def _compute_causal_strength(
        self,
        source: str,
        target: str,
        path: List[str]
    ) -> float:
        """Compute causal strength based on path."""
        if not path or len(path) < 2:
            return 0.0
        
        # Strength decreases with path length
        path_penalty = 1.0 / len(path)
        
        # Get edge strengths from DAG
        total_strength = 0.0
        if self.discovery_engine.current_dag is not None:
            edges = self.discovery_engine.current_dag.edges
            for i in range(len(path) - 1):
                for edge in edges:
                    if edge.source == path[i] and edge.target == path[i + 1]:
                        total_strength += edge.strength
                        break
            
            if len(path) > 1:
                total_strength /= (len(path) - 1)
        
        return total_strength * path_penalty
    
    def _compute_spurious_probability(
        self,
        has_causal_link: bool,
        causal_strength: float,
        correlation: float
    ) -> float:
        """
        Compute probability that correlation is spurious.
        
        High correlation + low causal strength = likely spurious
        """
        if has_causal_link and causal_strength > 0.5:
            return 0.1  # Low spurious probability
        
        if not has_causal_link and abs(correlation) > 0.5:
            return 0.9  # High spurious probability
        
        # Intermediate cases
        if has_causal_link:
            return max(0.1, 0.5 - causal_strength)
        else:
            return min(0.9, 0.5 + abs(correlation))
    
    def _make_decision(
        self,
        has_causal_link: bool,
        causal_strength: float,
        spurious_prob: float,
        signal_confidence: float
    ) -> Tuple[GatingDecision, str]:
        """Make gating decision based on causal analysis."""
        
        # Strong causal support
        if has_causal_link and causal_strength >= self.min_causal_strength:
            return GatingDecision.ALLOW, f"Causal link confirmed (strength={causal_strength:.2f})"
        
        # No causal link but high confidence signal
        if not has_causal_link and signal_confidence > 0.8:
            if self.enable_strict_mode:
                return GatingDecision.VETO, "No causal link - signal based on correlation only"
            else:
                return GatingDecision.WARN, "No causal link - proceed with caution"
        
        # High spurious probability
        if spurious_prob > self.max_spurious_prob:
            return GatingDecision.VETO, f"High spurious probability ({spurious_prob:.2f})"
        
        # Weak causal link
        if has_causal_link and causal_strength < self.min_causal_strength:
            return GatingDecision.WARN, f"Weak causal link (strength={causal_strength:.2f})"
        
        # Default: allow with warning
        return GatingDecision.WARN, "Uncertain causal relationship"
    
    def get_statistics(self) -> Dict:
        """Get gating statistics."""
        total = self.veto_count + self.allow_count + self.warn_count
        return {
            "total_checks": total,
            "veto_count": self.veto_count,
            "allow_count": self.allow_count,
            "warn_count": self.warn_count,
            "veto_rate": self.veto_count / max(total, 1),
            "allow_rate": self.allow_count / max(total, 1),
            "strict_mode": self.enable_strict_mode
        }
    
    def reset_statistics(self):
        """Reset gating statistics."""
        self.gating_history.clear()
        self.veto_count = 0
        self.allow_count = 0
        self.warn_count = 0


class CausalJudgeIntegration:
    """
    Integration layer between CausalGate and Judge.
    
    Provides causal veto capability to the Judge agent:
    - Intercepts signals before judgment
    - Applies causal gating
    - Modifies confidence based on causal support
    """
    
    def __init__(
        self,
        causal_gate: CausalGate,
        confidence_penalty: float = 0.3,
        enable_veto: bool = True
    ):
        self.causal_gate = causal_gate
        self.confidence_penalty = confidence_penalty
        self.enable_veto = enable_veto
        
        # Integration statistics
        self.signals_processed: int = 0
        self.signals_vetoed: int = 0
        self.confidence_adjustments: List[float] = []
        
    def process_signal(
        self,
        signal_variable: str,
        target_variable: str,
        signal_direction: float,
        signal_confidence: float,
        data: np.ndarray = None
    ) -> Tuple[float, float, bool, str]:
        """
        Process signal through causal gate.
        
        Args:
            signal_variable: Variable the signal is based on
            target_variable: Variable being predicted
            signal_direction: Direction of signal
            signal_confidence: Original confidence
            data: Market data for analysis
            
        Returns:
            Tuple of (adjusted_direction, adjusted_confidence, is_vetoed, reason)
        """
        self.signals_processed += 1
        
        # Check causal support
        result = self.causal_gate.check_signal(
            signal_variable, target_variable,
            signal_direction, signal_confidence, data
        )
        
        # Handle veto
        if result.decision == GatingDecision.VETO and self.enable_veto:
            self.signals_vetoed += 1
            return 0.0, 0.0, True, result.reasoning
        
        # Adjust confidence based on causal strength
        adjusted_confidence = signal_confidence
        
        if result.decision == GatingDecision.WARN:
            # Reduce confidence for weak causal support
            penalty = self.confidence_penalty * (1 - result.causal_strength)
            adjusted_confidence = signal_confidence * (1 - penalty)
            self.confidence_adjustments.append(penalty)
        
        elif result.decision == GatingDecision.ALLOW:
            # Boost confidence for strong causal support
            boost = 0.1 * result.causal_strength
            adjusted_confidence = min(1.0, signal_confidence * (1 + boost))
            self.confidence_adjustments.append(-boost)
        
        return signal_direction, adjusted_confidence, False, result.reasoning
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics."""
        return {
            "signals_processed": self.signals_processed,
            "signals_vetoed": self.signals_vetoed,
            "veto_rate": self.signals_vetoed / max(self.signals_processed, 1),
            "avg_confidence_adjustment": np.mean(self.confidence_adjustments) if self.confidence_adjustments else 0.0,
            "causal_gate_stats": self.causal_gate.get_statistics()
        }


class RegimeAwareCausalGate(CausalGate):
    """
    Regime-Aware Causal Gate.
    
    Extends CausalGate with regime-specific causal analysis:
    - Different causal structures per regime
    - Regime-conditional gating thresholds
    - Integration with soft_clustering
    """
    
    # Regime-specific thresholds
    REGIME_THRESHOLDS = {
        "BULL": {"min_strength": 0.2, "max_spurious": 0.6},
        "BEAR": {"min_strength": 0.4, "max_spurious": 0.4},
        "HIGH_VOL": {"min_strength": 0.5, "max_spurious": 0.3},
        "LOW_VOL": {"min_strength": 0.2, "max_spurious": 0.6},
        "CRISIS": {"min_strength": 0.6, "max_spurious": 0.2},
    }
    
    def __init__(
        self,
        causal_discovery_engine=None,
        regime_detector=None,
        default_regime: str = "NORMAL"
    ):
        super().__init__(causal_discovery_engine)
        self.regime_detector = regime_detector
        self.current_regime = default_regime
        self.regime_dag_cache: Dict[str, Any] = {}
        
    def set_regime(self, regime: str):
        """Set current market regime."""
        self.current_regime = regime
        
        # Update thresholds
        if regime in self.REGIME_THRESHOLDS:
            thresholds = self.REGIME_THRESHOLDS[regime]
            self.min_causal_strength = thresholds["min_strength"]
            self.max_spurious_prob = thresholds["max_spurious"]
    
    def check_signal_regime_aware(
        self,
        signal_variable: str,
        target_variable: str,
        signal_direction: float,
        signal_confidence: float,
        regime: str = None,
        data: np.ndarray = None
    ) -> CausalGatingResult:
        """Check signal with regime-specific thresholds."""
        if regime:
            self.set_regime(regime)
        
        return self.check_signal(
            signal_variable, target_variable,
            signal_direction, signal_confidence, data
        )
    
    def update_regime_dag(self, regime: str, data: np.ndarray):
        """Update DAG for specific regime."""
        if self.discovery_engine is not None:
            dag = self.discovery_engine.discover(data)
            self.regime_dag_cache[regime] = dag
