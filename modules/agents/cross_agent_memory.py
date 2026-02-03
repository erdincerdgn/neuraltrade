"""
Cross-Agent Memory Buffer - Collective Learning System
Author: Erdinc Erdogan
Purpose: Implements a shared memory buffer for multi-agent trading systems, enabling collective
learning from successes, failures, and market regime changes across all trading agents.
References:
- Multi-Agent Reinforcement Learning (MARL)
- Experience Replay Buffers in Deep Learning
- Collective Intelligence and Swarm Learning
Usage:
    memory = CrossAgentMemory(max_size=10000, decay_rate=0.99)
    memory.add_entry(entry_id="001", memory_type=MemoryType.SUCCESS, agent_type="bull", ...)
    insights = memory.get_collective_insights()
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time
import hashlib


class MemoryType(IntEnum):
    """Type of memory entry."""
    SUCCESS = 0
    FAILURE = 1
    NEAR_MISS = 2
    REGIME_CHANGE = 3
    CIRCUIT_BREAKER = 4


class FailureCategory(IntEnum):
    """Category of failure for learning."""
    SLIPPAGE = 0
    WRONG_DIRECTION = 1
    TIMING = 2
    REGIME_MISMATCH = 3
    OVERCONFIDENCE = 4
    CORRELATION_BREAKDOWN = 5


@dataclass
class MemoryEntry:
    """Single memory entry in the shared buffer."""
    entry_id: str
    timestamp: float
    memory_type: MemoryType
    agent_type: str
    
    # Market context
    regime: str
    entropy: float
    volatility: float
    
    # Signal details
    signal_direction: float
    signal_confidence: float
    
    # Outcome
    realized_return: float
    slippage_bps: float
    was_correct: bool
    
    # Learning metadata
    failure_category: Optional[FailureCategory] = None
    features_snapshot: Dict = field(default_factory=dict)
    lessons: List[str] = field(default_factory=list)
    
    # Decay
    relevance_score: float = 1.0
    access_count: int = 0


@dataclass
class CollectiveInsight:
    """Aggregated insight from memory analysis."""
    insight_type: str
    confidence: float
    affected_agents: List[str]
    regime_context: str
    recommendation: str
    supporting_memories: int


class CrossAgentMemory:
    """
    Cross-Agent Memory Buffer for Collective Learning.
    
    Enables agents to learn from each other's failures and successes.
    Maintains a shared memory of recent trading outcomes with context.
    
    Key Features:
    1. Failure pattern detection
    2. Regime-specific memory retrieval
    3. Similarity-based memory search
    4. Collective insight generation
    5. Memory decay and pruning
    
    Mathematical Foundation:
    relevance = base_relevance × decay^(age) × similarity(context, query)
    
    Usage:
        memory = CrossAgentMemory()
        memory.record(entry)
        insights = memory.get_insights(current_context)
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        decay_rate: float = 0.995,
        similarity_threshold: float = 0.7,
        min_relevance: float = 0.1,
        failure_weight: float = 2.0,
        success_weight: float = 1.0
    ):
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        self.min_relevance = min_relevance
        self.failure_weight = failure_weight
        self.success_weight = success_weight
        
        # Memory storage
        self.memories: deque = deque(maxlen=max_entries)
        self.failure_index: Dict[FailureCategory, List[str]] = {
            cat: [] for cat in FailureCategory
        }
        self.regime_index: Dict[str, List[str]] = {}
        self.agent_index: Dict[str, List[str]] = {}
        
        # Statistics
        self.total_entries: int = 0
        self.total_failures: int = 0
        self.total_successes: int = 0
        
        # Cached insights
        self.cached_insights: List[CollectiveInsight] = []
        self.last_insight_update: float = 0.0
        
    def record(
        self,
        agent_type: str,
        signal_direction: float,
        signal_confidence: float,
        realized_return: float,
        slippage_bps: float,
        regime: str,
        entropy: float,
        volatility: float,
        features: Dict = None,
        timestamp: float = None
    ) -> MemoryEntry:
        """
        Record a trading outcome in shared memory.
        
        Args:
            agent_type: Type of agent that generated signal
            signal_direction: Signal direction [-1, 1]
            signal_confidence: Signal confidence [0, 1]
            realized_return: Actual return achieved
            slippage_bps: Execution slippage in bps
            regime: Current market regime
            entropy: Regime entropy
            volatility: Current volatility
            features: Optional feature snapshot
            timestamp: Optional timestamp
            
        Returns:
            MemoryEntry that was recorded
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Determine if correct
        was_correct = (signal_direction > 0 and realized_return > 0) or                      (signal_direction < 0 and realized_return < 0)
        
        # Determine memory type
        if was_correct and realized_return > 0:
            memory_type = MemoryType.SUCCESS
            self.total_successes += 1
        elif not was_correct:
            memory_type = MemoryType.FAILURE
            self.total_failures += 1
        else:
            memory_type = MemoryType.NEAR_MISS
        
        # Categorize failure
        failure_category = None
        lessons = []
        
        if memory_type == MemoryType.FAILURE:
            failure_category, lessons = self._categorize_failure(
                signal_direction, signal_confidence, realized_return,
                slippage_bps, entropy, volatility
            )
        
        # Create entry
        entry_id = self._generate_entry_id(timestamp, agent_type)
        
        entry = MemoryEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            memory_type=memory_type,
            agent_type=agent_type,
            regime=regime,
            entropy=entropy,
            volatility=volatility,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            realized_return=realized_return,
            slippage_bps=slippage_bps,
            was_correct=was_correct,
            failure_category=failure_category,
            features_snapshot=features or {},
            lessons=lessons
        )
        
        # Store and index
        self.memories.append(entry)
        self._index_entry(entry)
        self.total_entries += 1
        
        # Decay old memories
        self._apply_decay()
        
        return entry
    
    def query(
        self,
        regime: str = None,
        agent_type: str = None,
        failure_category: FailureCategory = None,
        min_relevance: float = None,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """
        Query memories with filters.
        
        Returns list of relevant memories.
        """
        if min_relevance is None:
            min_relevance = self.min_relevance
        
        results = []
        
        for entry in self.memories:
            # Apply filters
            if regime and entry.regime != regime:
                continue
            if agent_type and entry.agent_type != agent_type:
                continue
            if failure_category and entry.failure_category != failure_category:
                continue
            if entry.relevance_score < min_relevance:
                continue
            
            results.append(entry)
            entry.access_count += 1
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:limit]
    
    def get_similar_failures(
        self,
        current_regime: str,
        current_entropy: float,
        current_volatility: float,
        agent_type: str = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Find similar past failures for learning.
        
        Uses context similarity to find relevant failures.
        """
        failures = [
            e for e in self.memories 
            if e.memory_type == MemoryType.FAILURE
        ]
        
        if not failures:
            return []
        
        # Calculate similarity scores
        scored = []
        for entry in failures:
            similarity = self._calculate_similarity(
                entry, current_regime, current_entropy, current_volatility
            )
            
            if agent_type and entry.agent_type != agent_type:
                similarity *= 0.5  # Reduce but don't exclude
            
            if similarity >= self.similarity_threshold:
                scored.append((entry, similarity))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [e for e, _ in scored[:limit]]
    
    def get_collective_insights(
        self,
        current_regime: str,
        current_entropy: float,
        force_refresh: bool = False
    ) -> List[CollectiveInsight]:
        """
        Generate collective insights from memory analysis.
        
        Returns actionable insights for all agents.
        """
        current_time = time.time()
        
        # Use cache if recent
        if not force_refresh and (current_time - self.last_insight_update) < 60:
            return self.cached_insights
        
        insights = []
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns()
        for pattern, data in failure_patterns.items():
            if data['count'] >= 3:  # Minimum pattern threshold
                insights.append(CollectiveInsight(
                    insight_type=f"FAILURE_PATTERN_{pattern}",
                    confidence=min(1.0, data['count'] / 10),
                    affected_agents=data['agents'],
                    regime_context=data['common_regime'],
                    recommendation=data['recommendation'],
                    supporting_memories=data['count']
                ))
        
        # Analyze regime-specific performance
        regime_insights = self._analyze_regime_performance(current_regime)
        insights.extend(regime_insights)
        
        # Analyze entropy correlation
        if current_entropy > 0.7:
            high_entropy_failures = [
                e for e in self.memories
                if e.memory_type == MemoryType.FAILURE and e.entropy > 0.7
            ]
            if len(high_entropy_failures) >= 5:
                insights.append(CollectiveInsight(
                    insight_type="HIGH_ENTROPY_WARNING",
                    confidence=0.8,
                    affected_agents=list(set(e.agent_type for e in high_entropy_failures)),
                    regime_context=current_regime,
                    recommendation="REDUCE_CONFIDENCE: High entropy historically correlates with failures",
                    supporting_memories=len(high_entropy_failures)
                ))
        
        self.cached_insights = insights
        self.last_insight_update = current_time
        
        return insights
    
    def get_agent_lessons(self, agent_type: str) -> List[str]:
        """Get accumulated lessons for a specific agent."""
        lessons = []
        
        agent_failures = [
            e for e in self.memories
            if e.agent_type == agent_type and e.memory_type == MemoryType.FAILURE
        ]
        
        for entry in agent_failures[-20:]:  # Last 20 failures
            lessons.extend(entry.lessons)
        
        # Deduplicate and count
        lesson_counts = {}
        for lesson in lessons:
            lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1
        
        # Return most common lessons
        sorted_lessons = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)
        return [lesson for lesson, _ in sorted_lessons[:5]]
    
    def _categorize_failure(
        self,
        direction: float,
        confidence: float,
        realized_return: float,
        slippage: float,
        entropy: float,
        volatility: float
    ) -> Tuple[FailureCategory, List[str]]:
        """Categorize failure and generate lessons."""
        lessons = []
        
        # High slippage failure
        if slippage > 20:
            lessons.append("REDUCE_AGGRESSION: Slippage exceeded 20bps")
            return FailureCategory.SLIPPAGE, lessons
        
        # Wrong direction with high confidence
        if confidence > 0.7 and abs(realized_return) > 0.01:
            lessons.append("CALIBRATE_CONFIDENCE: High confidence signal was wrong")
            return FailureCategory.OVERCONFIDENCE, lessons
        
        # High entropy failure
        if entropy > 0.7:
            lessons.append("REDUCE_SIZE_HIGH_ENTROPY: Failed during regime uncertainty")
            return FailureCategory.REGIME_MISMATCH, lessons
        
        # High volatility failure
        if volatility > 0.30:
            lessons.append("REDUCE_SIZE_HIGH_VOL: Failed during high volatility")
            return FailureCategory.TIMING, lessons
        
        # Default: wrong direction
        lessons.append("REVIEW_SIGNAL_LOGIC: Direction prediction was incorrect")
        return FailureCategory.WRONG_DIRECTION, lessons
    
    def _calculate_similarity(
        self,
        entry: MemoryEntry,
        regime: str,
        entropy: float,
        volatility: float
    ) -> float:
        """Calculate context similarity score."""
        score = 0.0
        
        # Regime match
        if entry.regime == regime:
            score += 0.4
        
        # Entropy similarity
        entropy_diff = abs(entry.entropy - entropy)
        score += 0.3 * (1.0 - min(entropy_diff, 1.0))
        
        # Volatility similarity
        vol_diff = abs(entry.volatility - volatility) / max(volatility, 0.01)
        score += 0.3 * (1.0 - min(vol_diff, 1.0))
        
        return score
    
    def _analyze_failure_patterns(self) -> Dict:
        """Analyze common failure patterns."""
        patterns = {}
        
        for category in FailureCategory:
            failures = [
                e for e in self.memories
                if e.failure_category == category
            ]
            
            if failures:
                agents = list(set(e.agent_type for e in failures))
                regimes = [e.regime for e in failures]
                common_regime = max(set(regimes), key=regimes.count) if regimes else "UNKNOWN"
                
                patterns[category.name] = {
                    'count': len(failures),
                    'agents': agents,
                    'common_regime': common_regime,
                    'recommendation': self._get_pattern_recommendation(category)
                }
        
        return patterns
    
    def _analyze_regime_performance(self, current_regime: str) -> List[CollectiveInsight]:
        """Analyze performance in current regime."""
        insights = []
        
        regime_entries = [e for e in self.memories if e.regime == current_regime]
        
        if len(regime_entries) >= 10:
            success_rate = sum(1 for e in regime_entries if e.was_correct) / len(regime_entries)
            
            if success_rate < 0.4:
                insights.append(CollectiveInsight(
                    insight_type="LOW_REGIME_SUCCESS",
                    confidence=0.7,
                    affected_agents=list(set(e.agent_type for e in regime_entries)),
                    regime_context=current_regime,
                    recommendation=f"CAUTION: Only {success_rate:.0%} success rate in {current_regime} regime",
                    supporting_memories=len(regime_entries)
                ))
        
        return insights
    
    def _get_pattern_recommendation(self, category: FailureCategory) -> str:
        """Get recommendation for failure pattern."""
        recommendations = {
            FailureCategory.SLIPPAGE: "REDUCE_ORDER_SIZE: Use more passive execution",
            FailureCategory.WRONG_DIRECTION: "REVIEW_FEATURES: Check signal generation logic",
            FailureCategory.TIMING: "IMPROVE_ENTRY: Wait for better entry conditions",
            FailureCategory.REGIME_MISMATCH: "ADD_REGIME_FILTER: Avoid trading in uncertain regimes",
            FailureCategory.OVERCONFIDENCE: "CALIBRATE_CONFIDENCE: Reduce confidence scaling",
            FailureCategory.CORRELATION_BREAKDOWN: "DIVERSIFY_SIGNALS: Add uncorrelated features",
        }
        return recommendations.get(category, "REVIEW_STRATEGY")
    
    def _generate_entry_id(self, timestamp: float, agent_type: str) -> str:
        """Generate unique entry ID."""
        data = f"{timestamp}_{agent_type}_{self.total_entries}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _index_entry(self, entry: MemoryEntry):
        """Index entry for fast retrieval."""
        # Failure index
        if entry.failure_category is not None:
            self.failure_index[entry.failure_category].append(entry.entry_id)
        
        # Regime index
        if entry.regime not in self.regime_index:
            self.regime_index[entry.regime] = []
        self.regime_index[entry.regime].append(entry.entry_id)
        
        # Agent index
        if entry.agent_type not in self.agent_index:
            self.agent_index[entry.agent_type] = []
        self.agent_index[entry.agent_type].append(entry.entry_id)
    
    def _apply_decay(self):
        """Apply relevance decay to all memories."""
        for entry in self.memories:
            # Failures decay slower
            if entry.memory_type == MemoryType.FAILURE:
                entry.relevance_score *= (self.decay_rate ** 0.5)
            else:
                entry.relevance_score *= self.decay_rate
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        return {
            "total_entries": self.total_entries,
            "current_size": len(self.memories),
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": self.total_failures / max(self.total_entries, 1),
            "failure_categories": {
                cat.name: len(ids) for cat, ids in self.failure_index.items()
            },
            "regimes_tracked": list(self.regime_index.keys()),
            "agents_tracked": list(self.agent_index.keys()),
        }
    
    def reset(self):
        """Reset memory buffer."""
        self.memories.clear()
        self.failure_index = {cat: [] for cat in FailureCategory}
        self.regime_index.clear()
        self.agent_index.clear()
        self.total_entries = 0
        self.total_failures = 0
        self.total_successes = 0
        self.cached_insights.clear()
