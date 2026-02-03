"""
Signal Validator - Numerical Integrity Checker
Author: Erdinc Erdogan
Purpose: Validates trading signals for numerical integrity, detecting NaN, infinity, and
out-of-range values to prevent logic collisions from safe-math guards.
References:
- IEEE 754 Floating Point Standard
- Numerical Stability in Financial Computing
- Defensive Programming Patterns
Usage:
    validator = SignalValidator(min_value=-1.0, max_value=1.0)
    result = validator.validate(signal_value=0.85, entropy=0.3)
    if result.is_valid: execute_trade(result.corrected_value)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import IntEnum
import time


class ValidationStatus(IntEnum):
    VALID = 0
    NAN_DETECTED = 1
    INF_DETECTED = 2
    OUT_OF_RANGE = 3
    ZERO_DENOMINATOR = 4
    ENTROPY_CASCADE = 5


@dataclass
class SignalValidationResult:
    """Result of signal validation."""
    is_valid: bool
    status: ValidationStatus
    original_value: float
    corrected_value: float
    correction_applied: bool
    warning_message: str


class SignalValidator:
    """
    Signal Validator with NaN/Inf Detection.
    
    Prevents logic collision by detecting invalid values
    at pipeline entry points instead of silently clipping.
    
    Key Features:
    1. NaN/Inf detection at critical points (fixes LC-003)
    2. Entropy cascade detection (fixes ED-005)
    3. Logging of guard activations
    4. Graceful degradation with warnings
    """
    
    def __init__(
        self,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        min_direction: float = -1.0,
        max_direction: float = 1.0,
        entropy_cascade_threshold: int = 3,
        enable_logging: bool = True
    ):
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.min_direction = min_direction
        self.max_direction = max_direction
        self.entropy_cascade_threshold = entropy_cascade_threshold
        self.enable_logging = enable_logging
        
        # Tracking
        self.nan_count: int = 0
        self.inf_count: int = 0
        self.range_violations: int = 0
        self.entropy_cascades: int = 0
        self.guard_activations: List[Dict] = []
        
    def validate_confidence(self, value: float, source: str = "") -> SignalValidationResult:
        """Validate confidence value."""
        return self._validate_value(
            value, self.min_confidence, self.max_confidence, "confidence", source
        )
    
    def validate_direction(self, value: float, source: str = "") -> SignalValidationResult:
        """Validate direction value."""
        return self._validate_value(
            value, self.min_direction, self.max_direction, "direction", source
        )
    
    def validate_entropy(self, value: float, source: str = "") -> SignalValidationResult:
        """Validate entropy value."""
        return self._validate_value(value, 0.0, 1.0, "entropy", source)
    
    def check_entropy_cascade(
        self,
        entropy_adjustments: List[float]
    ) -> Tuple[bool, float]:
        """
        Check for entropy cascade (multiple entropy gates compounding).
        
        Returns (is_cascade, combined_factor)
        """
        if len(entropy_adjustments) < self.entropy_cascade_threshold:
            return False, 1.0
        
        # Calculate combined effect
        combined = 1.0
        for adj in entropy_adjustments:
            combined *= adj
        
        # Cascade detected if combined effect is too severe
        is_cascade = combined < 0.3
        
        if is_cascade:
            self.entropy_cascades += 1
            if self.enable_logging:
                self.guard_activations.append({
                    'type': 'ENTROPY_CASCADE',
                    'timestamp': time.time(),
                    'adjustments': entropy_adjustments,
                    'combined_factor': combined
                })
        
        return is_cascade, combined
    
    def consolidate_entropy_gates(
        self,
        confidence: float,
        entropy: float,
        entropy_threshold: float = 0.8,
        max_reduction: float = 0.5
    ) -> float:
        """
        Consolidate multiple entropy gates into single adjustment.
        
        This prevents the cascade effect from 59+ entropy checks.
        """
        if entropy <= entropy_threshold:
            return confidence
        
        # Single consolidated reduction
        excess = entropy - entropy_threshold
        max_excess = 1.0 - entropy_threshold
        reduction_factor = 1.0 - (excess / max_excess) * max_reduction
        
        return confidence * max(reduction_factor, 0.5)
    
    def _validate_value(
        self,
        value: float,
        min_val: float,
        max_val: float,
        value_type: str,
        source: str
    ) -> SignalValidationResult:
        """Core validation logic."""
        original = value
        corrected = value
        correction_applied = False
        status = ValidationStatus.VALID
        warning = ""
        
        # Check NaN
        if np.isnan(value):
            status = ValidationStatus.NAN_DETECTED
            corrected = (min_val + max_val) / 2  # Default to midpoint
            correction_applied = True
            warning = f"NaN detected in {value_type} from {source}"
            self.nan_count += 1
        
        # Check Inf
        elif np.isinf(value):
            status = ValidationStatus.INF_DETECTED
            corrected = max_val if value > 0 else min_val
            correction_applied = True
            warning = f"Inf detected in {value_type} from {source}"
            self.inf_count += 1
        
        # Check range
        elif value < min_val or value > max_val:
            status = ValidationStatus.OUT_OF_RANGE
            corrected = np.clip(value, min_val, max_val)
            correction_applied = True
            warning = f"Range violation in {value_type}: {value:.4f} from {source}"
            self.range_violations += 1
        
        # Log if correction applied
        if correction_applied and self.enable_logging:
            self.guard_activations.append({
                'type': status.name,
                'timestamp': time.time(),
                'value_type': value_type,
                'source': source,
                'original': original,
                'corrected': corrected
            })
        
        return SignalValidationResult(
            is_valid=(status == ValidationStatus.VALID),
            status=status,
            original_value=original,
            corrected_value=corrected,
            correction_applied=correction_applied,
            warning_message=warning
        )
    
    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        return {
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "range_violations": self.range_violations,
            "entropy_cascades": self.entropy_cascades,
            "total_corrections": self.nan_count + self.inf_count + self.range_violations,
            "recent_activations": self.guard_activations[-10:] if self.guard_activations else []
        }
    
    def reset_statistics(self):
        """Reset statistics."""
        self.nan_count = 0
        self.inf_count = 0
        self.range_violations = 0
        self.entropy_cascades = 0
        self.guard_activations.clear()


class ConsolidatedEntropyGate:
    """
    Consolidated Entropy Gate to prevent cascade effects.
    
    Replaces 59+ individual entropy checks with single consolidated gate.
    """
    
    def __init__(
        self,
        base_threshold: float = 0.8,
        min_confidence_floor: float = 0.15,
        max_reduction: float = 0.5
    ):
        self.base_threshold = base_threshold
        self.min_confidence_floor = min_confidence_floor
        self.max_reduction = max_reduction
        self.validator = SignalValidator()
        
        # Track applications
        self.applications: int = 0
        self.reductions_applied: int = 0
        
    def apply(
        self,
        confidence: float,
        entropy: float
    ) -> Tuple[float, bool]:
        """
        Apply consolidated entropy gate.
        
        Returns (adjusted_confidence, was_reduced)
        """
        self.applications += 1
        
        # Validate inputs
        conf_result = self.validator.validate_confidence(confidence, "entropy_gate")
        ent_result = self.validator.validate_entropy(entropy, "entropy_gate")
        
        confidence = conf_result.corrected_value
        entropy = ent_result.corrected_value
        
        # Apply single consolidated gate
        if entropy <= self.base_threshold:
            return confidence, False
        
        # Calculate reduction
        excess = entropy - self.base_threshold
        max_excess = 1.0 - self.base_threshold
        reduction_factor = 1.0 - (excess / max_excess) * self.max_reduction
        
        adjusted = confidence * reduction_factor
        
        # Apply floor
        adjusted = max(adjusted, self.min_confidence_floor)
        
        self.reductions_applied += 1
        
        return adjusted, True
    
    def get_statistics(self) -> Dict:
        """Get gate statistics."""
        return {
            "applications": self.applications,
            "reductions_applied": self.reductions_applied,
            "reduction_rate": self.reductions_applied / max(self.applications, 1),
            "validator_stats": self.validator.get_statistics()
        }
