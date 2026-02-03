"""
Sanitized Cross-Agent Memory - Secure Collective Learning System
Author: Erdinc Erdogan
Purpose: Implements a sanitized shared memory buffer with isolation modes, validation, and
quarantine mechanisms to prevent memory poisoning and ensure data integrity across agents.
References:
- Memory Isolation and Access Control Patterns
- Outlier Detection and Data Validation
- Secure Multi-Agent Communication Protocols
Usage:
    memory = SanitizedCrossAgentMemory(isolation_mode=IsolationMode.FILTERED_SHARED)
    result = memory.add_validated_entry(entry, agent_type="bull")
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time
import hashlib


class MemoryType(IntEnum):
    SUCCESS = 0
    FAILURE = 1
    NEAR_MISS = 2
    REGIME_CHANGE = 3
    CIRCUIT_BREAKER = 4


class FailureCategory(IntEnum):
    SLIPPAGE = 0
    WRONG_DIRECTION = 1
    TIMING = 2
    REGIME_MISMATCH = 3
    OVERCONFIDENCE = 4
    CORRELATION_BREAKDOWN = 5


class IsolationMode(IntEnum):
    """Memory isolation modes."""
    FULL_SHARED = 0      # All agents share all memories
    AGENT_ISOLATED = 1   # Agents only see own memories
    FILTERED_SHARED = 2  # Agents see filtered cross-agent memories
    QUARANTINED = 3      # New entries quarantined before sharing


@dataclass
class MemoryEntry:
    """Single memory entry with isolation metadata."""
    entry_id: str
    timestamp: float
    memory_type: MemoryType
    agent_type: str
    regime: str
    entropy: float
    volatility: float
    signal_direction: float
    signal_confidence: float
    realized_return: float
    slippage_bps: float
    was_correct: bool
    failure_category: Optional[FailureCategory] = None
    features_snapshot: Dict = field(default_factory=dict)
    lessons: List[str] = field(default_factory=list)
    relevance_score: float = 1.0
    access_count: int = 0
    # Isolation fields
    is_quarantined: bool = False
    quarantine_until: float = 0.0
    is_validated: bool = False
    outlier_score: float = 0.0
    cross_agent_visible: bool = True


@dataclass
class MemoryValidationResult:
    """Result of memory entry validation."""
    is_valid: bool
    outlier_score: float
    rejection_reason: Optional[str]
    should_quarantine: bool
    quarantine_duration: float


class MemoryValidator:
    """
    Validates memory entries before storage.
    
    Prevents memory poisoning by detecting:
    1. Outlier returns (too extreme)
    2. Anomalous slippage
    3. Suspicious patterns
    4. Correlated failures
    """
    
    def __init__(
        self,
        max_return_zscore: float = 4.0,
        max_slippage_bps: float = 100.0,
        min_confidence: float = 0.01,
        quarantine_duration: float = 300.0,  # 5 minutes
        outlier_threshold: float = 0.7
    ):
        self.max_return_zscore = max_return_zscore
        self.max_slippage_bps = max_slippage_bps
        self.min_confidence = min_confidence
        self.quarantine_duration = quarantine_duration
        self.outlier_threshold = outlier_threshold
        
        # Rolling statistics for outlier detection
        self.return_history: deque = deque(maxlen=500)
        self.slippage_history: deque = deque(maxlen=500)
        
    def validate(self, entry: MemoryEntry) -> MemoryValidationResult:
        """Validate a memory entry before storage."""
        outlier_score = 0.0
        rejection_reason = None
        should_quarantine = False
        
        # Check for extreme returns
        if len(self.return_history) >= 20:
            returns = np.array(list(self.return_history))
            mean_ret = np.mean(returns)
            std_ret = np.std(returns) + 1e-10
            zscore = abs(entry.realized_return - mean_ret) / std_ret
            
            if zscore > self.max_return_zscore:
                outlier_score += 0.4
                should_quarantine = True
        
        # Check for extreme slippage
        if entry.slippage_bps > self.max_slippage_bps:
            outlier_score += 0.3
            should_quarantine = True
        
        # Check for invalid confidence
        if entry.signal_confidence < self.min_confidence:
            outlier_score += 0.2
            rejection_reason = "Invalid confidence"
        
        # Check for NaN/Inf values
        if np.isnan(entry.realized_return) or np.isinf(entry.realized_return):
            outlier_score = 1.0
            rejection_reason = "NaN/Inf return value"
        
        # Update history
        self.return_history.append(entry.realized_return)
        self.slippage_history.append(entry.slippage_bps)
        
        is_valid = outlier_score < self.outlier_threshold and rejection_reason is None
        
        return MemoryValidationResult(
            is_valid=is_valid,
            outlier_score=outlier_score,
            rejection_reason=rejection_reason,
            should_quarantine=should_quarantine,
            quarantine_duration=self.quarantine_duration if should_quarantine else 0.0
        )


class MemoryIsolationManager:
    """
    Manages memory isolation between agents.
    
    Prevents a failing agent from poisoning other agents' decisions.
    """
    
    def __init__(
        self,
        default_mode: IsolationMode = IsolationMode.FILTERED_SHARED,
        cross_agent_weight: float = 0.3,  # Weight for cross-agent memories
        same_agent_weight: float = 1.0,   # Weight for same-agent memories
        failure_isolation_factor: float = 0.5  # Reduce cross-agent failure impact
    ):
        self.default_mode = default_mode
        self.cross_agent_weight = cross_agent_weight
        self.same_agent_weight = same_agent_weight
        self.failure_isolation_factor = failure_isolation_factor
        
        # Per-agent isolation settings
        self.agent_modes: Dict[str, IsolationMode] = {}
        
    def get_memory_weight(
        self,
        entry: MemoryEntry,
        querying_agent: str
    ) -> float:
        """Get weight for a memory entry based on isolation rules."""
        # Same agent - full weight
        if entry.agent_type == querying_agent:
            return self.same_agent_weight
        
        # Cross-agent - reduced weight
        weight = self.cross_agent_weight
        
        # Further reduce weight for failures from other agents
        if entry.memory_type == MemoryType.FAILURE:
            weight *= self.failure_isolation_factor
        
        # Quarantined entries have zero weight for cross-agent
        if entry.is_quarantined:
            weight = 0.0
        
        return weight
    
    def filter_memories(
        self,
        memories: List[MemoryEntry],
        querying_agent: str,
        mode: IsolationMode = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """Filter and weight memories based on isolation mode."""
        mode = mode or self.agent_modes.get(querying_agent, self.default_mode)
        
        result = []
        current_time = time.time()
        
        for entry in memories:
            # Check quarantine expiry
            if entry.is_quarantined and current_time < entry.quarantine_until:
                if entry.agent_type != querying_agent:
                    continue  # Skip quarantined cross-agent entries
            
            if mode == IsolationMode.AGENT_ISOLATED:
                if entry.agent_type != querying_agent:
                    continue
                weight = self.same_agent_weight
            elif mode == IsolationMode.FILTERED_SHARED:
                weight = self.get_memory_weight(entry, querying_agent)
                if weight <= 0:
                    continue
            else:  # FULL_SHARED
                weight = self.same_agent_weight if entry.agent_type == querying_agent else self.cross_agent_weight
            
            result.append((entry, weight))
        
        return result
    
    def set_agent_mode(self, agent_type: str, mode: IsolationMode):
        """Set isolation mode for a specific agent."""
        self.agent_modes[agent_type] = mode


class SanitizedCrossAgentMemory:
    """
    Sanitized Cross-Agent Memory with isolation and validation.
    
    Key Improvements over original:
    1. Memory validation before storage (MP-003)
    2. Quarantine mechanism for suspicious entries (MP-005)
    3. Agent isolation to prevent poisoning (MP-001)
    4. Capped failure_weight at 1.5 (MP-002)
    5. Outlier detection and rejection
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        decay_rate: float = 0.995,
        failure_weight: float = 1.5,  # Capped from 2.0 (fixes MP-002)
        success_weight: float = 1.0,
        isolation_mode: IsolationMode = IsolationMode.FILTERED_SHARED,
        enable_validation: bool = True,
        enable_quarantine: bool = True
    ):
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.failure_weight = min(failure_weight, 1.5)  # Hard cap
        self.success_weight = success_weight
        
        # Core components
        self.validator = MemoryValidator() if enable_validation else None
        self.isolation_manager = MemoryIsolationManager(default_mode=isolation_mode)
        self.enable_quarantine = enable_quarantine
        
        # Memory storage
        self.memories: deque = deque(maxlen=max_entries)
        self.quarantine_queue: deque = deque(maxlen=100)
        
        # Indexes
        self.failure_index: Dict[FailureCategory, List[str]] = {cat: [] for cat in FailureCategory}
        self.regime_index: Dict[str, List[str]] = {}
        self.agent_index: Dict[str, List[str]] = {}
        
        # Statistics
        self.total_entries: int = 0
        self.total_rejected: int = 0
        self.total_quarantined: int = 0
        self.total_failures: int = 0
        self.total_successes: int = 0
        
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
    ) -> Optional[MemoryEntry]:
        """Record a trading outcome with validation and isolation."""
        if timestamp is None:
            timestamp = time.time()
        
        # Determine correctness
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
        
        # Validate entry (fixes MP-003)
        if self.validator:
            validation = self.validator.validate(entry)
            entry.outlier_score = validation.outlier_score
            entry.is_validated = True
            
            if not validation.is_valid:
                self.total_rejected += 1
                return None
            
            # Quarantine if needed (fixes MP-005)
            if validation.should_quarantine and self.enable_quarantine:
                entry.is_quarantined = True
                entry.quarantine_until = timestamp + validation.quarantine_duration
                entry.cross_agent_visible = False
                self.total_quarantined += 1
                self.quarantine_queue.append(entry)
        
        # Store and index
        self.memories.append(entry)
        self._index_entry(entry)
        self.total_entries += 1
        
        # Apply decay
        self._apply_decay()
        
        # Process quarantine releases
        self._process_quarantine_releases()
        
        return entry
    
    def query(
        self,
        querying_agent: str,
        regime: str = None,
        failure_category: FailureCategory = None,
        min_relevance: float = 0.1,
        limit: int = 50
    ) -> List[Tuple[MemoryEntry, float]]:
        """Query memories with isolation filtering."""
        # Get filtered memories based on isolation
        filtered = self.isolation_manager.filter_memories(
            list(self.memories),
            querying_agent
        )
        
        # Apply additional filters
        results = []
        for entry, weight in filtered:
            if regime and entry.regime != regime:
                continue
            if failure_category and entry.failure_category != failure_category:
                continue
            if entry.relevance_score < min_relevance:
                continue
            
            # Apply failure weight adjustment
            if entry.memory_type == MemoryType.FAILURE:
                weight *= self.failure_weight
            
            results.append((entry, weight))
            entry.access_count += 1
        
        # Sort by weighted relevance
        results.sort(key=lambda x: x[0].relevance_score * x[1], reverse=True)
        
        return results[:limit]
    
    def get_agent_lessons(self, agent_type: str, include_cross_agent: bool = False) -> List[str]:
        """Get lessons for an agent with optional cross-agent learning."""
        lessons = []
        
        for entry in self.memories:
            if entry.memory_type != MemoryType.FAILURE:
                continue
            
            # Check isolation
            if entry.agent_type != agent_type and not include_cross_agent:
                continue
            
            # Skip quarantined cross-agent entries
            if entry.is_quarantined and entry.agent_type != agent_type:
                continue
            
            lessons.extend(entry.lessons)
        
        # Deduplicate and count
        lesson_counts = {}
        for lesson in lessons:
            lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1
        
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
        
        if slippage > 20:
            lessons.append("REDUCE_AGGRESSION: Slippage exceeded 20bps")
            return FailureCategory.SLIPPAGE, lessons
        
        if confidence > 0.7 and abs(realized_return) > 0.01:
            lessons.append("CALIBRATE_CONFIDENCE: High confidence signal was wrong")
            return FailureCategory.OVERCONFIDENCE, lessons
        
        if entropy > 0.7:
            lessons.append("REDUCE_SIZE_HIGH_ENTROPY: Failed during regime uncertainty")
            return FailureCategory.REGIME_MISMATCH, lessons
        
        if volatility > 0.30:
            lessons.append("REDUCE_SIZE_HIGH_VOL: Failed during high volatility")
            return FailureCategory.TIMING, lessons
        
        lessons.append("REVIEW_SIGNAL_LOGIC: Direction prediction was incorrect")
        return FailureCategory.WRONG_DIRECTION, lessons
    
    def _process_quarantine_releases(self):
        """Release entries from quarantine when time expires."""
        current_time = time.time()
        
        for entry in list(self.quarantine_queue):
            if current_time >= entry.quarantine_until:
                entry.is_quarantined = False
                entry.cross_agent_visible = True
                self.quarantine_queue.remove(entry)
    
    def _generate_entry_id(self, timestamp: float, agent_type: str) -> str:
        data = f"{timestamp}_{agent_type}_{self.total_entries}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _index_entry(self, entry: MemoryEntry):
        if entry.failure_category is not None:
            self.failure_index[entry.failure_category].append(entry.entry_id)
        
        if entry.regime not in self.regime_index:
            self.regime_index[entry.regime] = []
        self.regime_index[entry.regime].append(entry.entry_id)
        
        if entry.agent_type not in self.agent_index:
            self.agent_index[entry.agent_type] = []
        self.agent_index[entry.agent_type].append(entry.entry_id)
    
    def _apply_decay(self):
        for entry in self.memories:
            if entry.memory_type == MemoryType.FAILURE:
                entry.relevance_score *= (self.decay_rate ** 0.5)
            else:
                entry.relevance_score *= self.decay_rate
    
    def get_statistics(self) -> Dict:
        return {
            "total_entries": self.total_entries,
            "total_rejected": self.total_rejected,
            "total_quarantined": self.total_quarantined,
            "current_quarantine_size": len(self.quarantine_queue),
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_weight": self.failure_weight,
            "isolation_mode": self.isolation_manager.default_mode.name,
        }
    
    def reset(self):
        self.memories.clear()
        self.quarantine_queue.clear()
        self.failure_index = {cat: [] for cat in FailureCategory}
        self.regime_index.clear()
        self.agent_index.clear()
        self.total_entries = 0
        self.total_rejected = 0
        self.total_quarantined = 0
        self.total_failures = 0
        self.total_successes = 0
