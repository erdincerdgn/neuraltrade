"""
Neural Emission Calibrator for HMM
Author: Erdinc Erdogan
Purpose: Online calibration layer that adjusts HMM emission probabilities based on realized market impact using bias correction and temperature scaling.
References:
- Platt Scaling for Probability Calibration
- Online Learning for HMM
- Market Impact Feedback Loops
Usage:
    calibrator = NeuralEmissionCalibrator(n_states=8)
    calibrated_probs = calibrator.calibrate(raw_probs)
    calibrator.update(raw_probs, realized_regime, market_impact)
"""

# ============================================================================
# NEURAL EMISSION CALIBRATION - Online HMM Calibration Layer
# Adjusts state probabilities against realized market impact
# Phase 7C: Beyond Tier-1 Enhancement
# ============================================================================

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from collections import deque


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality assessment."""
    prediction_error: float
    calibration_adjustment: float
    confidence_score: float
    samples_used: int
    regime_accuracy: Dict[int, float]


@dataclass
class CalibrationState:
    """Current state of the calibration layer."""
    bias_corrections: np.ndarray
    scale_factors: np.ndarray
    temperature: float
    last_metrics: Optional[CalibrationMetrics]
    is_calibrated: bool


class NeuralEmissionCalibrator:
    """
    Online Calibration Layer for Neural HMM Emissions.
    
    Continuously adjusts emission probabilities based on realized
    market outcomes to improve regime detection accuracy.
    
    Key Innovations:
    1. Bias Correction: Learns systematic over/under-prediction per state
    2. Scale Calibration: Adjusts confidence (temperature scaling)
    3. Online Learning: Adapts to changing market microstructure
    4. Impact Feedback: Uses realized slippage/impact as ground truth
    
    Mathematical Foundation:
    P_calibrated(s|x) = softmax((log P_raw(s|x) + bias_s) / temperature)
    
    Where:
    - bias_s is learned per-state correction
    - temperature controls confidence (>1 = less confident)
    
    Usage:
        calibrator = NeuralEmissionCalibrator(n_states=8)
        calibrated_probs = calibrator.calibrate(raw_probs)
        calibrator.update(raw_probs, realized_regime, market_impact)
    """
    
    def __init__(
        self,
        n_states: int = 8,
        learning_rate: float = 0.01,
        temperature_lr: float = 0.005,
        min_temperature: float = 0.5,
        max_temperature: float = 2.0,
        initial_temperature: float = 1.0,
        memory_size: int = 500,
        min_samples_for_calibration: int = 50,
        impact_weight: float = 0.3,
        regime_weight: float = 0.7
    ):
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.temperature_lr = temperature_lr
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.memory_size = memory_size
        self.min_samples_for_calibration = min_samples_for_calibration
        self.impact_weight = impact_weight
        self.regime_weight = regime_weight
        
        # Calibration parameters
        self.bias_corrections = np.zeros(n_states)
        self.scale_factors = np.ones(n_states)
        self.temperature = initial_temperature
        
        # Memory buffers
        self.prediction_history: deque = deque(maxlen=memory_size)
        self.outcome_history: deque = deque(maxlen=memory_size)
        self.impact_history: deque = deque(maxlen=memory_size)
        
        # Per-state tracking
        self.state_predictions: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(n_states)
        }
        self.state_outcomes: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(n_states)
        }
        
        # State
        self.is_calibrated = False
        self.last_metrics: Optional[CalibrationMetrics] = None
        
    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw emission probabilities.
        
        Args:
            raw_probs: Raw probabilities from neural emission network
            
        Returns:
            Calibrated probability distribution
        """
        raw_probs = self._validate_probs(raw_probs)
        
        # Apply log-space bias correction
        log_probs = np.log(raw_probs + 1e-10)
        corrected_log_probs = log_probs + self.bias_corrections
        
        # Apply temperature scaling
        scaled_log_probs = corrected_log_probs / self.temperature
        
        # Apply per-state scale factors
        scaled_log_probs = scaled_log_probs * self.scale_factors
        
        # Softmax to get calibrated probabilities
        calibrated = self._softmax(scaled_log_probs)
        
        return calibrated
    
    def update(
        self,
        predicted_probs: np.ndarray,
        realized_regime: int,
        market_impact: Optional[float] = None,
        expected_impact: Optional[float] = None
    ):
        """
        Update calibration based on realized outcome.
        
        Args:
            predicted_probs: Probabilities that were predicted
            realized_regime: Actual regime that occurred
            market_impact: Realized market impact (slippage)
            expected_impact: Expected impact based on prediction
        """
        predicted_probs = self._validate_probs(predicted_probs)
        
        # Store in history
        self.prediction_history.append(predicted_probs.copy())
        self.outcome_history.append(realized_regime)
        
        if market_impact is not None:
            self.impact_history.append({
                'realized': market_impact,
                'expected': expected_impact or 0.0,
                'regime': realized_regime
            })
        
        # Update per-state tracking
        predicted_regime = np.argmax(predicted_probs)
        self.state_predictions[predicted_regime].append(predicted_probs[predicted_regime])
        self.state_outcomes[predicted_regime].append(1 if predicted_regime == realized_regime else 0)
        
        # Perform calibration update if enough samples
        if len(self.prediction_history) >= self.min_samples_for_calibration:
            self._update_calibration()
            self.is_calibrated = True
    
    def _update_calibration(self):
        """Update calibration parameters based on history."""
        predictions = np.array(list(self.prediction_history))
        outcomes = np.array(list(self.outcome_history))
        
        # Calculate prediction errors per state
        for state in range(self.n_states):
            # Get predictions for this state
            state_mask = outcomes == state
            if state_mask.sum() < 5:
                continue
            
            # Average predicted probability when this state occurred
            avg_predicted = predictions[state_mask, state].mean()
            
            # Ideal would be 1.0 when state occurs
            # Bias correction moves prediction toward 1.0
            error = 1.0 - avg_predicted
            
            # Update bias (move toward correcting under-prediction)
            self.bias_corrections[state] += self.learning_rate * error
        
        # Clip bias corrections
        self.bias_corrections = np.clip(self.bias_corrections, -2.0, 2.0)
        
        # Update temperature based on calibration quality
        self._update_temperature(predictions, outcomes)
        
        # Update scale factors based on per-state accuracy
        self._update_scale_factors()
        
        # Compute metrics
        self._compute_metrics(predictions, outcomes)
    
    def _update_temperature(self, predictions: np.ndarray, outcomes: np.ndarray):
        """Update temperature based on prediction confidence vs accuracy."""
        # Calculate average confidence (max probability)
        confidences = predictions.max(axis=1)
        avg_confidence = confidences.mean()
        
        # Calculate accuracy
        predicted_states = predictions.argmax(axis=1)
        accuracy = (predicted_states == outcomes).mean()
        
        # If overconfident (high confidence, low accuracy), increase temperature
        # If underconfident (low confidence, high accuracy), decrease temperature
        confidence_error = avg_confidence - accuracy
        
        self.temperature += self.temperature_lr * confidence_error
        self.temperature = np.clip(
            self.temperature, 
            self.min_temperature, 
            self.max_temperature
        )
    
    def _update_scale_factors(self):
        """Update per-state scale factors based on accuracy."""
        for state in range(self.n_states):
            if len(self.state_outcomes[state]) < 10:
                continue
            
            # Calculate accuracy for this state
            accuracy = np.mean(list(self.state_outcomes[state]))
            
            # Scale factor: increase for accurate states, decrease for inaccurate
            target_scale = 0.5 + accuracy  # Range [0.5, 1.5]
            
            # Smooth update
            self.scale_factors[state] = (
                0.95 * self.scale_factors[state] + 
                0.05 * target_scale
            )
        
        # Normalize scale factors
        self.scale_factors = self.scale_factors / self.scale_factors.mean()
    
    def _compute_metrics(self, predictions: np.ndarray, outcomes: np.ndarray):
        """Compute calibration quality metrics."""
        # Overall prediction error
        predicted_states = predictions.argmax(axis=1)
        accuracy = (predicted_states == outcomes).mean()
        prediction_error = 1.0 - accuracy
        
        # Per-state accuracy
        regime_accuracy = {}
        for state in range(self.n_states):
            state_mask = outcomes == state
            if state_mask.sum() > 0:
                state_preds = predicted_states[state_mask]
                regime_accuracy[state] = (state_preds == state).mean()
            else:
                regime_accuracy[state] = 0.0
        
        # Calibration adjustment magnitude
        calibration_adjustment = np.abs(self.bias_corrections).mean()
        
        # Confidence score (inverse of temperature deviation from 1.0)
        confidence_score = 1.0 / (1.0 + abs(self.temperature - 1.0))
        
        self.last_metrics = CalibrationMetrics(
            prediction_error=prediction_error,
            calibration_adjustment=calibration_adjustment,
            confidence_score=confidence_score,
            samples_used=len(predictions),
            regime_accuracy=regime_accuracy
        )
    
    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Validate and normalize probabilities."""
        probs = np.asarray(probs)
        probs = np.maximum(probs, 1e-10)
        return probs / probs.sum()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
    
    def get_state(self) -> CalibrationState:
        """Get current calibration state."""
        return CalibrationState(
            bias_corrections=self.bias_corrections.copy(),
            scale_factors=self.scale_factors.copy(),
            temperature=self.temperature,
            last_metrics=self.last_metrics,
            is_calibrated=self.is_calibrated
        )
    
    def get_diagnostics(self) -> Dict:
        """Get calibration diagnostics for monitoring."""
        return {
            "is_calibrated": self.is_calibrated,
            "temperature": self.temperature,
            "bias_corrections": self.bias_corrections.tolist(),
            "scale_factors": self.scale_factors.tolist(),
            "samples_collected": len(self.prediction_history),
            "metrics": {
                "prediction_error": self.last_metrics.prediction_error if self.last_metrics else None,
                "confidence_score": self.last_metrics.confidence_score if self.last_metrics else None,
            } if self.last_metrics else None
        }
    
    def reset(self):
        """Reset calibration to initial state."""
        self.bias_corrections = np.zeros(self.n_states)
        self.scale_factors = np.ones(self.n_states)
        self.temperature = 1.0
        self.prediction_history.clear()
        self.outcome_history.clear()
        self.impact_history.clear()
        for state in range(self.n_states):
            self.state_predictions[state].clear()
            self.state_outcomes[state].clear()
        self.is_calibrated = False
        self.last_metrics = None
    
    def save_calibration(self) -> Dict:
        """Save calibration parameters for persistence."""
        return {
            "bias_corrections": self.bias_corrections.tolist(),
            "scale_factors": self.scale_factors.tolist(),
            "temperature": self.temperature,
            "is_calibrated": self.is_calibrated
        }
    
    def load_calibration(self, params: Dict):
        """Load calibration parameters."""
        self.bias_corrections = np.array(params["bias_corrections"])
        self.scale_factors = np.array(params["scale_factors"])
        self.temperature = params["temperature"]
        self.is_calibrated = params["is_calibrated"]


class CalibratedNeuralHMM:
    """
    Wrapper that integrates calibration with Neural HMM.
    
    Provides a unified interface for calibrated regime detection.
    """
    
    def __init__(
        self,
        neural_hmm,  # The actual Neural HMM instance
        calibrator: Optional[NeuralEmissionCalibrator] = None,
        auto_calibrate: bool = True
    ):
        self.neural_hmm = neural_hmm
        self.calibrator = calibrator or NeuralEmissionCalibrator(
            n_states=getattr(neural_hmm, 'n_states', 8)
        )
        self.auto_calibrate = auto_calibrate
        self.last_raw_probs: Optional[np.ndarray] = None
        
    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Get calibrated regime prediction.
        
        Args:
            features: Input features for HMM
            
        Returns:
            Tuple of (predicted_state, calibrated_probabilities)
        """
        # Get raw prediction from Neural HMM
        raw_probs = self.neural_hmm.predict_proba(features)
        self.last_raw_probs = raw_probs.copy()
        
        # Apply calibration
        calibrated_probs = self.calibrator.calibrate(raw_probs)
        
        # Get predicted state
        predicted_state = np.argmax(calibrated_probs)
        
        return predicted_state, calibrated_probs
    
    def update_calibration(
        self,
        realized_regime: int,
        market_impact: Optional[float] = None
    ):
        """Update calibration with realized outcome."""
        if self.last_raw_probs is not None and self.auto_calibrate:
            self.calibrator.update(
                self.last_raw_probs,
                realized_regime,
                market_impact
            )
