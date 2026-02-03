"""
Deadlock Recovery Mechanism - Trading System Resilience
Author: Erdinc Erdogan
Purpose: Monitors trading frequency and detects entropy-induced paralysis, implementing automatic
recovery mechanisms to prevent silent halts and ensure continuous system operation.
References:
- Deadlock Detection Algorithms (Operating Systems Theory)
- Adaptive Threshold Management
- High-Frequency Trading System Resilience Patterns
Usage:
    detector = DeadlockDetector(block_threshold=5, recovery_factor=0.1)
    status = detector.check_and_recover(was_blocked=True, entropy=0.8, confidence=0.7)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum
import time


class DeadlockState(IntEnum):
    """Deadlock detection state."""
    NORMAL = 0
    WARNING = 1
    DEADLOCK = 2
    RECOVERY = 3
    FORCED_EXECUTION = 4


class RecoveryAction(IntEnum):
    """Recovery action types."""
    NONE = 0
    REDUCE_THRESHOLD = 1
    BYPASS_ENTROPY = 2
    FORCE_EXECUTION = 3
    EMERGENCY_OVERRIDE = 4


@dataclass
class DeadlockStatus:
    """Current deadlock status."""
    state: DeadlockState
    consecutive_blocks: int
    time_since_last_trade: float
    current_entropy_threshold: float
    original_entropy_threshold: float
    recovery_action: RecoveryAction
    forced_execution_available: bool
    alert_message: str


@dataclass
class HighAlphaSignal:
    """High-alpha signal that can bypass entropy gating."""
    signal_id: str
    direction: float
    confidence: float
    alpha_score: float
    entropy: float
    timestamp: float
    bypass_reason: str


class DeadlockDetector:
    """
    Deadlock Detection and Recovery System.
    
    Monitors trading frequency and detects entropy-induced paralysis.
    Implements automatic recovery mechanisms to prevent silent halts.
    
    Key Features:
    1. Trade frequency monitoring
    2. Consecutive block detection
    3. Adaptive entropy threshold reduction
    4. High-alpha signal bypass
    5. Forced execution path
    6. Alert system
    
    Mathematical Foundation:
    - Deadlock detected when: consecutive_blocks >= threshold
    - Recovery: entropy_threshold = original * (1 - recovery_factor * blocks/max_blocks)
    - Bypass: alpha_score > bypass_threshold AND entropy < bypass_entropy_limit
    
    Usage:
        detector = DeadlockDetector()
        status = detector.check_and_recover(was_blocked, entropy, confidence)
        if status.forced_execution_available:
            execute_trade()
    """
    
    def __init__(
        self,
        block_threshold: int = 5,
        time_threshold_seconds: float = 300.0,
        original_entropy_threshold: float = 0.7,
        min_entropy_threshold: float = 0.5,
        recovery_factor: float = 0.1,
        alpha_bypass_threshold: float = 0.8,
        bypass_entropy_limit: float = 0.85,
        max_recovery_blocks: int = 20,
        alert_callback: callable = None
    ):
        self.block_threshold = block_threshold
        self.time_threshold_seconds = time_threshold_seconds
        self.original_entropy_threshold = original_entropy_threshold
        self.min_entropy_threshold = min_entropy_threshold
        self.recovery_factor = recovery_factor
        self.alpha_bypass_threshold = alpha_bypass_threshold
        self.bypass_entropy_limit = bypass_entropy_limit
        self.max_recovery_blocks = max_recovery_blocks
        self.alert_callback = alert_callback
        
        # State tracking
        self.consecutive_blocks: int = 0
        self.last_trade_time: float = time.time()
        self.current_entropy_threshold: float = original_entropy_threshold
        self.state: DeadlockState = DeadlockState.NORMAL
        
        # History
        self.block_history: deque = deque(maxlen=100)
        self.recovery_history: deque = deque(maxlen=50)
        self.bypass_signals: deque = deque(maxlen=20)
        
        # Statistics
        self.total_blocks: int = 0
        self.total_recoveries: int = 0
        self.total_bypasses: int = 0
        self.total_forced_executions: int = 0
        
    def check_and_recover(
        self,
        was_blocked: bool,
        current_entropy: float,
        signal_confidence: float,
        alpha_score: float = 0.0,
        signal_direction: float = 0.0
    ) -> DeadlockStatus:
        """
        Check for deadlock and initiate recovery if needed.
        
        Args:
            was_blocked: Whether the last signal was blocked by entropy gating
            current_entropy: Current regime entropy
            signal_confidence: Signal confidence before gating
            alpha_score: Optional alpha score for bypass consideration
            signal_direction: Signal direction for bypass
            
        Returns:
            DeadlockStatus with current state and recovery actions
        """
        current_time = time.time()
        time_since_trade = current_time - self.last_trade_time
        
        if was_blocked:
            self.consecutive_blocks += 1
            self.total_blocks += 1
            self.block_history.append({
                'timestamp': current_time,
                'entropy': current_entropy,
                'confidence': signal_confidence
            })
        else:
            # Trade executed - reset counters
            self.consecutive_blocks = 0
            self.last_trade_time = current_time
            self.current_entropy_threshold = self.original_entropy_threshold
            self.state = DeadlockState.NORMAL
        
        # Determine state
        recovery_action = RecoveryAction.NONE
        forced_execution = False
        alert_message = ""
        
        # Check for deadlock conditions
        if self.consecutive_blocks >= self.block_threshold or            time_since_trade > self.time_threshold_seconds:
            
            if self.state == DeadlockState.NORMAL:
                self.state = DeadlockState.WARNING
                alert_message = f"WARNING: {self.consecutive_blocks} consecutive blocks"
            
            if self.consecutive_blocks >= self.block_threshold * 2:
                self.state = DeadlockState.DEADLOCK
                alert_message = f"DEADLOCK: {self.consecutive_blocks} blocks, {time_since_trade:.0f}s since trade"
                
                # Initiate recovery
                recovery_action = self._initiate_recovery()
        
        # Check for high-alpha bypass opportunity
        if was_blocked and self._can_bypass(alpha_score, current_entropy, signal_confidence):
            forced_execution = True
            recovery_action = RecoveryAction.FORCE_EXECUTION
            self.state = DeadlockState.FORCED_EXECUTION
            self.total_bypasses += 1
            
            self.bypass_signals.append(HighAlphaSignal(
                signal_id=f"bypass_{current_time}",
                direction=signal_direction,
                confidence=signal_confidence,
                alpha_score=alpha_score,
                entropy=current_entropy,
                timestamp=current_time,
                bypass_reason="HIGH_ALPHA_LOW_ENTROPY"
            ))
            
            alert_message = f"BYPASS: High-alpha signal (α={alpha_score:.2f}) bypassing entropy gate"
        
        # Send alert if callback provided
        if alert_message and self.alert_callback:
            self.alert_callback(alert_message, self.state)
        
        return DeadlockStatus(
            state=self.state,
            consecutive_blocks=self.consecutive_blocks,
            time_since_last_trade=time_since_trade,
            current_entropy_threshold=self.current_entropy_threshold,
            original_entropy_threshold=self.original_entropy_threshold,
            recovery_action=recovery_action,
            forced_execution_available=forced_execution,
            alert_message=alert_message
        )
    
    def _initiate_recovery(self) -> RecoveryAction:
        """Initiate deadlock recovery by reducing entropy threshold."""
        # Calculate reduction based on consecutive blocks
        reduction = self.recovery_factor * (self.consecutive_blocks / self.max_recovery_blocks)
        reduction = min(reduction, 0.3)  # Cap at 30% reduction
        
        new_threshold = self.original_entropy_threshold * (1 - reduction)
        new_threshold = max(new_threshold, self.min_entropy_threshold)
        
        if new_threshold < self.current_entropy_threshold:
            self.current_entropy_threshold = new_threshold
            self.state = DeadlockState.RECOVERY
            self.total_recoveries += 1
            
            self.recovery_history.append({
                'timestamp': time.time(),
                'blocks': self.consecutive_blocks,
                'old_threshold': self.original_entropy_threshold,
                'new_threshold': new_threshold
            })
            
            return RecoveryAction.REDUCE_THRESHOLD
        
        return RecoveryAction.NONE
    
    def _can_bypass(
        self,
        alpha_score: float,
        entropy: float,
        confidence: float
    ) -> bool:
        """Check if signal qualifies for entropy bypass."""
        # High alpha + not extreme entropy + decent confidence
        return (
            alpha_score >= self.alpha_bypass_threshold and
            entropy < self.bypass_entropy_limit and
            confidence >= 0.5
        )
    
    def get_adjusted_entropy_threshold(self) -> float:
        """Get current (possibly reduced) entropy threshold."""
        return self.current_entropy_threshold
    
    def force_execution_override(self) -> bool:
        """
        Manual override for forced execution.
        
        Returns True if override is granted.
        """
        if self.state in [DeadlockState.DEADLOCK, DeadlockState.RECOVERY]:
            self.total_forced_executions += 1
            self.consecutive_blocks = 0
            self.last_trade_time = time.time()
            self.state = DeadlockState.FORCED_EXECUTION
            return True
        return False
    
    def record_successful_trade(self):
        """Record that a trade was successfully executed."""
        self.consecutive_blocks = 0
        self.last_trade_time = time.time()
        self.current_entropy_threshold = self.original_entropy_threshold
        self.state = DeadlockState.NORMAL
    
    def get_statistics(self) -> Dict:
        """Get deadlock detection statistics."""
        return {
            "state": self.state.name,
            "consecutive_blocks": self.consecutive_blocks,
            "time_since_trade": time.time() - self.last_trade_time,
            "current_threshold": self.current_entropy_threshold,
            "original_threshold": self.original_entropy_threshold,
            "total_blocks": self.total_blocks,
            "total_recoveries": self.total_recoveries,
            "total_bypasses": self.total_bypasses,
            "total_forced": self.total_forced_executions,
        }
    
    def reset(self):
        """Reset detector to initial state."""
        self.consecutive_blocks = 0
        self.last_trade_time = time.time()
        self.current_entropy_threshold = self.original_entropy_threshold
        self.state = DeadlockState.NORMAL
        self.block_history.clear()
        self.recovery_history.clear()
        self.bypass_signals.clear()


class EntropyGateWithRecovery:
    """
    Entropy Gate with integrated deadlock recovery.
    
    Wraps the standard entropy gating logic with deadlock detection
    and automatic recovery mechanisms.
    """
    
    def __init__(
        self,
        base_threshold: float = 0.8,  # Raised from 0.7 (fixes ED-001)
        min_confidence_floor: float = 0.15,  # Global floor (fixes ED-002)
        deadlock_detector: DeadlockDetector = None
    ):
        self.base_threshold = base_threshold
        self.min_confidence_floor = min_confidence_floor
        self.deadlock_detector = deadlock_detector or DeadlockDetector(
            original_entropy_threshold=base_threshold
        )
        
    def apply_gate(
        self,
        confidence: float,
        entropy: float,
        alpha_score: float = 0.0,
        signal_direction: float = 0.0
    ) -> Tuple[float, bool, DeadlockStatus]:
        """
        Apply entropy gating with deadlock recovery.
        
        Args:
            confidence: Original signal confidence
            entropy: Current regime entropy
            alpha_score: Optional alpha score for bypass
            signal_direction: Signal direction
            
        Returns:
            Tuple of (adjusted_confidence, was_gated, deadlock_status)
        """
        # Get current (possibly reduced) threshold
        threshold = self.deadlock_detector.get_adjusted_entropy_threshold()
        
        # Check if gating should apply
        was_gated = False
        adjusted_confidence = confidence
        
        if entropy > threshold:
            # Calculate gating factor
            excess = entropy - threshold
            max_excess = 1.0 - threshold
            gating_factor = 1.0 - (excess / max_excess) * 0.7  # Max 70% reduction
            
            adjusted_confidence = confidence * gating_factor
            was_gated = True
        
        # Apply global confidence floor (fixes ED-002)
        adjusted_confidence = max(adjusted_confidence, self.min_confidence_floor)
        
        # Check deadlock status
        status = self.deadlock_detector.check_and_recover(
            was_blocked=(adjusted_confidence < 0.2),  # Effectively blocked
            current_entropy=entropy,
            signal_confidence=confidence,
            alpha_score=alpha_score,
            signal_direction=signal_direction
        )
        
        # If forced execution available, restore confidence
        if status.forced_execution_available:
            adjusted_confidence = max(confidence * 0.8, 0.5)  # Restore with small penalty
            was_gated = False
        
        return adjusted_confidence, was_gated, status
    
    def record_trade(self):
        """Record successful trade execution."""
        self.deadlock_detector.record_successful_trade()
