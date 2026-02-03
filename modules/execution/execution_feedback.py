"""
Execution Feedback Loop
Author: Erdinc Erdogan
Purpose: Closes the loop between execution outcomes and agent signal quality by tracking slippage, market impact, fill rates, and calculating agent penalties.
References:
- Execution Quality Metrics
- Agent Weighting via Sigmoid Penalty Functions
- Rolling Performance Attribution
Usage:
    feedback = ExecutionFeedbackLoop(lookback_window=200)
    feedback.record_execution(result)
    penalties = feedback.get_agent_penalties()
"""

# ============================================================================
# EXECUTION FEEDBACK LOOP
# Closes the loop between execution and agent signal quality
# Phase 8B: The Neural Interconnect - Task 1
# ============================================================================

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time


class ExecutionQuality(IntEnum):
    """Execution quality classification."""
    EXCELLENT = 4    # Better than expected (positive slippage)
    GOOD = 3         # Within tolerance
    ACCEPTABLE = 2   # Slight slippage
    POOR = 1         # Significant slippage
    FAILED = 0       # Execution failed or extreme slippage


@dataclass
class ExecutionResult:
    """Result of a trade execution."""
    order_id: str
    agent_type: str
    signal_id: str
    expected_price: float
    executed_price: float
    expected_quantity: float
    executed_quantity: float
    slippage_bps: float
    market_impact_bps: float
    execution_time_ms: float
    timestamp: float
    quality: ExecutionQuality
    venue: str = "PRIMARY"
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentPenalty:
    """Penalty/reward for an agent based on execution quality."""
    agent_type: str
    penalty_factor: float      # 0 to 1 (1 = no penalty, 0 = full penalty)
    slippage_score: float      # Rolling slippage score
    impact_score: float        # Rolling market impact score
    fill_rate: float           # Percentage of orders fully filled
    avg_execution_time: float  # Average execution time
    quality_score: float       # Overall quality score


class ExecutionFeedbackLoop:
    """
    Execution Feedback Loop for Agent Signal Quality.
    
    Closes the loop between execution outcomes and agent weighting.
    Agents that consistently produce high-slippage signals are penalized.
    
    Mathematical Foundation:
    penalty_factor = sigmoid(quality_score - threshold)
    quality_score = w1*slippage + w2*impact + w3*fill_rate + w4*timing
    
    Key Features:
    1. Rolling execution quality tracking per agent
    2. Slippage-based penalty calculation
    3. Market impact attribution
    4. Fill rate monitoring
    5. Execution timing analysis
    
    Usage:
        feedback = ExecutionFeedbackLoop()
        feedback.record_execution(result)
        penalties = feedback.get_agent_penalties()
    """
    
    # Quality thresholds (in basis points)
    SLIPPAGE_THRESHOLDS = {
        ExecutionQuality.EXCELLENT: -5,    # Negative = price improvement
        ExecutionQuality.GOOD: 5,
        ExecutionQuality.ACCEPTABLE: 15,
        ExecutionQuality.POOR: 30,
        ExecutionQuality.FAILED: float('inf'),
    }
    
    # Weights for quality score calculation
    QUALITY_WEIGHTS = {
        'slippage': 0.40,
        'impact': 0.25,
        'fill_rate': 0.20,
        'timing': 0.15,
    }
    
    def __init__(
        self,
        lookback_window: int = 200,
        penalty_sensitivity: float = 2.0,
        penalty_threshold: float = 0.5,
        min_penalty_factor: float = 0.3,
        max_penalty_factor: float = 1.0,
        slippage_target_bps: float = 10.0,
        impact_target_bps: float = 5.0,
        timing_target_ms: float = 100.0
    ):
        self.lookback_window = lookback_window
        self.penalty_sensitivity = penalty_sensitivity
        self.penalty_threshold = penalty_threshold
        self.min_penalty_factor = min_penalty_factor
        self.max_penalty_factor = max_penalty_factor
        self.slippage_target_bps = slippage_target_bps
        self.impact_target_bps = impact_target_bps
        self.timing_target_ms = timing_target_ms
        
        # Per-agent execution history
        self.execution_history: Dict[str, deque] = {}
        self.slippage_history: Dict[str, deque] = {}
        self.impact_history: Dict[str, deque] = {}
        self.fill_history: Dict[str, deque] = {}
        self.timing_history: Dict[str, deque] = {}
        
        # Cached penalties
        self.cached_penalties: Dict[str, AgentPenalty] = {}
        self.last_update_time: float = 0.0
        
        # Statistics
        self.total_executions: int = 0
        self.total_slippage_bps: float = 0.0
        
    def record_execution(self, result: ExecutionResult):
        """
        Record execution result and update agent penalties.
        
        Args:
            result: ExecutionResult from execution engine
        """
        agent = result.agent_type
        
        # Initialize history for new agents
        if agent not in self.execution_history:
            self._init_agent_history(agent)
        
        # Record metrics
        self.execution_history[agent].append(result)
        self.slippage_history[agent].append(result.slippage_bps)
        self.impact_history[agent].append(result.market_impact_bps)
        self.fill_history[agent].append(
            result.executed_quantity / max(result.expected_quantity, 1e-10)
        )
        self.timing_history[agent].append(result.execution_time_ms)
        
        # Update statistics
        self.total_executions += 1
        self.total_slippage_bps += result.slippage_bps
        
        # Recalculate penalty for this agent
        self._update_agent_penalty(agent)
        
    def get_agent_penalties(self) -> Dict[str, AgentPenalty]:
        """Get current penalties for all agents."""
        return self.cached_penalties.copy()
    
    def get_penalty_factor(self, agent_type: str) -> float:
        """Get penalty factor for a specific agent."""
        if agent_type in self.cached_penalties:
            return self.cached_penalties[agent_type].penalty_factor
        return 1.0  # No penalty for unknown agents
    
    def get_execution_feedback(self, agent_type: str) -> Dict:
        """
        Get detailed execution feedback for an agent.
        
        Returns dict suitable for agent learning/adjustment.
        """
        if agent_type not in self.cached_penalties:
            return {
                "penalty_factor": 1.0,
                "feedback": "NO_DATA",
                "recommendations": []
            }
        
        penalty = self.cached_penalties[agent_type]
        
        recommendations = []
        if penalty.slippage_score < 0.5:
            recommendations.append("REDUCE_AGGRESSION: High slippage detected")
        if penalty.impact_score < 0.5:
            recommendations.append("REDUCE_SIZE: High market impact detected")
        if penalty.fill_rate < 0.9:
            recommendations.append("IMPROVE_TIMING: Low fill rate detected")
        if penalty.avg_execution_time > self.timing_target_ms * 2:
            recommendations.append("OPTIMIZE_ROUTING: Slow execution detected")
        
        return {
            "penalty_factor": penalty.penalty_factor,
            "quality_score": penalty.quality_score,
            "slippage_score": penalty.slippage_score,
            "impact_score": penalty.impact_score,
            "fill_rate": penalty.fill_rate,
            "avg_execution_time": penalty.avg_execution_time,
            "feedback": "GOOD" if penalty.quality_score > 0.7 else "NEEDS_IMPROVEMENT",
            "recommendations": recommendations
        }
    
    def apply_penalty_to_signal(
        self,
        agent_type: str,
        signal_confidence: float
    ) -> float:
        """
        Apply execution penalty to signal confidence.
        
        Args:
            agent_type: Type of agent
            signal_confidence: Original signal confidence
            
        Returns:
            Adjusted confidence after penalty
        """
        penalty_factor = self.get_penalty_factor(agent_type)
        return signal_confidence * penalty_factor
    
    def _init_agent_history(self, agent: str):
        """Initialize history buffers for a new agent."""
        self.execution_history[agent] = deque(maxlen=self.lookback_window)
        self.slippage_history[agent] = deque(maxlen=self.lookback_window)
        self.impact_history[agent] = deque(maxlen=self.lookback_window)
        self.fill_history[agent] = deque(maxlen=self.lookback_window)
        self.timing_history[agent] = deque(maxlen=self.lookback_window)
    
    def _update_agent_penalty(self, agent: str):
        """Update penalty calculation for an agent."""
        if len(self.slippage_history[agent]) < 5:
            # Not enough data
            self.cached_penalties[agent] = AgentPenalty(
                agent_type=agent,
                penalty_factor=1.0,
                slippage_score=1.0,
                impact_score=1.0,
                fill_rate=1.0,
                avg_execution_time=0.0,
                quality_score=1.0
            )
            return
        
        # Calculate component scores
        slippage_score = self._calculate_slippage_score(agent)
        impact_score = self._calculate_impact_score(agent)
        fill_rate = self._calculate_fill_rate(agent)
        timing_score = self._calculate_timing_score(agent)
        avg_timing = np.mean(list(self.timing_history[agent]))
        
        # Weighted quality score
        quality_score = (
            self.QUALITY_WEIGHTS['slippage'] * slippage_score +
            self.QUALITY_WEIGHTS['impact'] * impact_score +
            self.QUALITY_WEIGHTS['fill_rate'] * fill_rate +
            self.QUALITY_WEIGHTS['timing'] * timing_score
        )
        
        # Convert to penalty factor using sigmoid
        penalty_factor = self._sigmoid(
            quality_score - self.penalty_threshold,
            self.penalty_sensitivity
        )
        
        # Clip to bounds
        penalty_factor = np.clip(
            penalty_factor,
            self.min_penalty_factor,
            self.max_penalty_factor
        )
        
        self.cached_penalties[agent] = AgentPenalty(
            agent_type=agent,
            penalty_factor=penalty_factor,
            slippage_score=slippage_score,
            impact_score=impact_score,
            fill_rate=fill_rate,
            avg_execution_time=avg_timing,
            quality_score=quality_score
        )
    
    def _calculate_slippage_score(self, agent: str) -> float:
        """Calculate slippage score (0 to 1, higher is better)."""
        slippages = list(self.slippage_history[agent])
        avg_slippage = np.mean(slippages)
        
        # Score based on target
        if avg_slippage <= 0:
            return 1.0  # Price improvement
        elif avg_slippage <= self.slippage_target_bps:
            return 1.0 - (avg_slippage / self.slippage_target_bps) * 0.3
        else:
            excess = avg_slippage - self.slippage_target_bps
            return max(0.0, 0.7 - excess / (self.slippage_target_bps * 3))
    
    def _calculate_impact_score(self, agent: str) -> float:
        """Calculate market impact score (0 to 1, higher is better)."""
        impacts = list(self.impact_history[agent])
        avg_impact = np.mean(impacts)
        
        if avg_impact <= self.impact_target_bps:
            return 1.0 - (avg_impact / self.impact_target_bps) * 0.2
        else:
            excess = avg_impact - self.impact_target_bps
            return max(0.0, 0.8 - excess / (self.impact_target_bps * 4))
    
    def _calculate_fill_rate(self, agent: str) -> float:
        """Calculate average fill rate."""
        fills = list(self.fill_history[agent])
        return np.mean(fills)
    
    def _calculate_timing_score(self, agent: str) -> float:
        """Calculate timing score (0 to 1, higher is better)."""
        timings = list(self.timing_history[agent])
        avg_timing = np.mean(timings)
        
        if avg_timing <= self.timing_target_ms:
            return 1.0
        else:
            excess = avg_timing - self.timing_target_ms
            return max(0.0, 1.0 - excess / (self.timing_target_ms * 5))
    
    def _sigmoid(self, x: float, sensitivity: float) -> float:
        """Sigmoid function for smooth penalty transition."""
        return 1.0 / (1.0 + np.exp(-sensitivity * x))
    
    def get_statistics(self) -> Dict:
        """Get overall execution statistics."""
        return {
            "total_executions": self.total_executions,
            "avg_slippage_bps": self.total_slippage_bps / max(self.total_executions, 1),
            "agents_tracked": len(self.execution_history),
            "penalties": {
                agent: penalty.penalty_factor 
                for agent, penalty in self.cached_penalties.items()
            }
        }
    
    def reset(self):
        """Reset all tracking data."""
        self.execution_history.clear()
        self.slippage_history.clear()
        self.impact_history.clear()
        self.fill_history.clear()
        self.timing_history.clear()
        self.cached_penalties.clear()
        self.total_executions = 0
        self.total_slippage_bps = 0.0


class ExecutionOptimizer:
    """
    Execution Optimizer with Agent Feedback Integration.
    
    Optimizes execution based on agent signal quality and
    provides feedback to improve future signals.
    """
    
    def __init__(
        self,
        feedback_loop: ExecutionFeedbackLoop,
        slippage_budget_bps: float = 20.0,
        urgency_factor: float = 1.0
    ):
        self.feedback_loop = feedback_loop
        self.slippage_budget_bps = slippage_budget_bps
        self.urgency_factor = urgency_factor
        
    def optimize_execution(
        self,
        agent_type: str,
        signal_direction: float,
        signal_confidence: float,
        target_quantity: float,
        current_price: float,
        volatility: float,
        spread_bps: float
    ) -> Dict:
        """
        Optimize execution parameters based on agent history.
        
        Returns execution plan with adjusted parameters.
        """
        # Get agent's execution feedback
        feedback = self.feedback_loop.get_execution_feedback(agent_type)
        penalty_factor = feedback["penalty_factor"]
        
        # Adjust confidence based on execution history
        adjusted_confidence = signal_confidence * penalty_factor
        
        # Determine execution style based on agent quality
        if feedback["quality_score"] > 0.8:
            execution_style = "AGGRESSIVE"
            participation_rate = 0.15
        elif feedback["quality_score"] > 0.5:
            execution_style = "NORMAL"
            participation_rate = 0.10
        else:
            execution_style = "PASSIVE"
            participation_rate = 0.05
        
        # Adjust quantity based on penalty
        adjusted_quantity = target_quantity * penalty_factor
        
        # Calculate expected slippage
        expected_slippage = spread_bps * 0.5 + volatility * 100 * participation_rate
        
        return {
            "agent_type": agent_type,
            "original_confidence": signal_confidence,
            "adjusted_confidence": adjusted_confidence,
            "penalty_factor": penalty_factor,
            "target_quantity": target_quantity,
            "adjusted_quantity": adjusted_quantity,
            "execution_style": execution_style,
            "participation_rate": participation_rate,
            "expected_slippage_bps": expected_slippage,
            "recommendations": feedback.get("recommendations", [])
        }
