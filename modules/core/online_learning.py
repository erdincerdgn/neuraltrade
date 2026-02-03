"""
Online Learning Module - Adaptive Baum-Welch Algorithm
Author: Erdinc Erdogan
Purpose: Implements online Baum-Welch algorithm with exponential forgetting for real-time
HMM parameter adaptation and adaptive learning rate decay.
References:
- Baum-Welch Algorithm: Baum et al. (1970)
- Online EM with Exponential Forgetting: Cappé & Moulines (2009)
- Stochastic Approximation: Robbins & Monro (1951)
Usage:
    learner = OnlineBaumWelch(n_states=3, forgetting_factor=0.95)
    learner.update(observation)
    new_params = learner.get_parameters()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ONLINE LEARNING CONFIGURATION
# ============================================================================

@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    n_states: int = 8                    # Number of HMM states
    forgetting_factor: float = 0.995     # λ: exponential forgetting (0.99-0.999)
    initial_learning_rate: float = 0.01  # η_0: initial learning rate
    learning_rate_decay: float = 0.5     # κ: decay exponent
    learning_rate_tau: float = 1000.0    # τ: decay time constant
    min_learning_rate: float = 0.001     # Minimum learning rate
    regularization: float = 1e-6         # Regularization for stability
    update_frequency: int = 1            # Update every N observations


@dataclass
class OnlineLearningState:
    """State of online learning algorithm."""
    n_updates: int = 0
    current_learning_rate: float = 0.01
    transition_sufficient_stats: Optional[np.ndarray] = None
    emission_sufficient_stats: Optional[Dict] = None
    gamma_sum: Optional[np.ndarray] = None
    last_update_time: datetime = field(default_factory=datetime.now)


# ============================================================================
# SUFFICIENT STATISTICS ACCUMULATOR
# ============================================================================

class SufficientStatistics:
    """
    Accumulator for HMM sufficient statistics with exponential forgetting.
    
    For transition matrix A:
        ξ_t(i,j) = P(S_{t-1}=i, S_t=j | O_{1:T})
        A_ij ∝ Σ_t ξ_t(i,j)
    
    For emission parameters:
        γ_t(k) = P(S_t=k | O_{1:T})
        μ_k = Σ_t γ_t(k) · O_t / Σ_t γ_t(k)
        σ²_k = Σ_t γ_t(k) · (O_t - μ_k)² / Σ_t γ_t(k)
    """
    
    def __init__(self, n_states: int, forgetting_factor: float = 0.995):
        self.n_states = n_states
        self.lambda_ = forgetting_factor
        
        # Transition sufficient statistics: Σ ξ_t(i,j)
        self.xi_sum = np.ones((n_states, n_states)) / n_states
        
        # State occupation: Σ γ_t(k)
        self.gamma_sum = np.ones(n_states)
        
        # Emission sufficient statistics (for Gaussian)
        self.obs_sum = np.zeros(n_states)      # Σ γ_t(k) · O_t
        self.obs_sq_sum = np.zeros(n_states)   # Σ γ_t(k) · O_t²
        
        self.n_samples = 0
    
    def update(self, gamma: np.ndarray, xi: np.ndarray, observation: float):
        """
        Update sufficient statistics with new observation.
        
        Args:
            gamma: State probabilities γ_t(k) for current time
            xi: Transition probabilities ξ_t(i,j) 
            observation: Current observation O_t
        """
        # Apply exponential forgetting
        self.xi_sum = self.lambda_ * self.xi_sum + xi
        self.gamma_sum = self.lambda_ * self.gamma_sum + gamma
        self.obs_sum = self.lambda_ * self.obs_sum + gamma * observation
        self.obs_sq_sum = self.lambda_ * self.obs_sq_sum + gamma * observation**2
        
        self.n_samples += 1
    
    def get_transition_estimate(self) -> np.ndarray:
        """Get current transition matrix estimate."""
        # A_ij = ξ_sum(i,j) / Σ_j ξ_sum(i,j)
        row_sums = self.xi_sum.sum(axis=1, keepdims=True)
        A = self.xi_sum / (row_sums + 1e-10)
        return A
    
    def get_emission_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current emission parameter estimates (mean, std)."""
        # μ_k = obs_sum(k) / gamma_sum(k)
        mu = self.obs_sum / (self.gamma_sum + 1e-10)
        
        # σ²_k = obs_sq_sum(k) / gamma_sum(k) - μ_k²
        var = self.obs_sq_sum / (self.gamma_sum + 1e-10) - mu**2
        sigma = np.sqrt(np.maximum(var, 1e-6))
        
        return mu, sigma
    
    def reset(self):
        """Reset sufficient statistics."""
        self.xi_sum = np.ones((self.n_states, self.n_states)) / self.n_states
        self.gamma_sum = np.ones(self.n_states)
        self.obs_sum = np.zeros(self.n_states)
        self.obs_sq_sum = np.zeros(self.n_states)
        self.n_samples = 0


# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

class LearningRateScheduler:
    """
    Adaptive learning rate scheduler.
    
    η_t = max(η_min, η_0 · (1 + t/τ)^{-κ})
    
    This ensures:
    - Fast initial learning
    - Gradual convergence
    - Stability in the long run
    """
    
    def __init__(self, 
                 initial_rate: float = 0.01,
                 decay_exponent: float = 0.5,
                 tau: float = 1000.0,
                 min_rate: float = 0.001):
        self.eta_0 = initial_rate
        self.kappa = decay_exponent
        self.tau = tau
        self.eta_min = min_rate
        self.t = 0
    
    def get_rate(self) -> float:
        """Get current learning rate."""
        eta = self.eta_0 * np.power(1 + self.t / self.tau, -self.kappa)
        return max(eta, self.eta_min)
    
    def step(self):
        """Increment time step."""
        self.t += 1
    
    def reset(self):
        """Reset scheduler."""
        self.t = 0


# ============================================================================
# ONLINE BAUM-WELCH ALGORITHM
# ============================================================================

class OnlineBaumWelch:
    """
    Online Baum-Welch Algorithm for HMM Parameter Estimation.
    
    Standard Baum-Welch requires full sequence for E-step and M-step.
    Online version processes observations one at a time using:
    
    1. Forward-Backward on sliding window
    2. Exponential forgetting of sufficient statistics
    3. Incremental parameter updates
    
    Mathematical Details:
    ---------------------
    E-Step (per observation):
        α_t(j) = [Σ_i α_{t-1}(i) · A_ij] · B_j(O_t)
        β_t(i) = Σ_j A_ij · B_j(O_{t+1}) · β_{t+1}(j)
        γ_t(i) = α_t(i) · β_t(i) / P(O)
        ξ_t(i,j) = α_t(i) · A_ij · B_j(O_{t+1}) · β_{t+1}(j) / P(O)
    
    M-Step (online):
        A_ij^{new} = (1-η) · A_ij^{old} + η · [ξ_sum(i,j) / γ_sum(i)]
    
    Usage:
        obw = OnlineBaumWelch(config)
        for observation in stream:
            obw.update(observation, current_state_probs)
            new_params = obw.get_parameters()
    """
    
    def __init__(self, config: OnlineLearningConfig = None):
        self.config = config or OnlineLearningConfig()
        
        # Initialize components
        self.stats = SufficientStatistics(
            self.config.n_states, 
            self.config.forgetting_factor
        )
        self.lr_scheduler = LearningRateScheduler(
            self.config.initial_learning_rate,
            self.config.learning_rate_decay,
            self.config.learning_rate_tau,
            self.config.min_learning_rate
        )
        
        # Current parameter estimates
        self.transition_matrix = np.ones((self.config.n_states, self.config.n_states))
        self.transition_matrix /= self.config.n_states
        
        self.emission_means = np.zeros(self.config.n_states)
        self.emission_stds = np.ones(self.config.n_states) * 0.02
        
        # State tracking
        self.state = OnlineLearningState()
        self.prev_gamma = None
        self.observation_buffer = []
    
    def compute_xi(self, gamma_prev: np.ndarray, gamma_curr: np.ndarray,
                   observation: float) -> np.ndarray:
        """
        Compute transition posterior ξ_t(i,j).
        
        ξ_t(i,j) ∝ γ_{t-1}(i) · A_ij · B_j(O_t) · [γ_t(j) / Σ_k A_ik · B_k(O_t)]
        
        Simplified approximation for online setting.
        """
        # Emission probabilities
        z = (observation - self.emission_means) / (self.emission_stds + 1e-10)
        log_emission = -0.5 * z**2 - np.log(self.emission_stds + 1e-10)
        emission = np.exp(log_emission - log_emission.max())
        emission /= emission.sum() + 1e-10
        
        # ξ_t(i,j) ≈ γ_{t-1}(i) · A_ij · emission_j · γ_t(j)
        xi = np.outer(gamma_prev, emission * gamma_curr)
        xi *= self.transition_matrix
        
        # Normalize
        xi /= xi.sum() + 1e-10
        
        return xi
    
    def update(self, observation: float, gamma: np.ndarray) -> Dict:
        """
        Update parameters with new observation.
        
        Args:
            observation: Current observation O_t
            gamma: Current state probabilities γ_t(k)
            
        Returns:
            Dictionary with update statistics
        """
        # Compute transition posterior if we have previous state
        if self.prev_gamma is not None:
            xi = self.compute_xi(self.prev_gamma, gamma, observation)
        else:
            # First observation - use uniform xi
            xi = np.ones((self.config.n_states, self.config.n_states))
            xi /= self.config.n_states**2
        
        # Update sufficient statistics
        self.stats.update(gamma, xi, observation)
        
        # Get current learning rate
        eta = self.lr_scheduler.get_rate()
        
        # Update parameters (M-step)
        if self.state.n_updates % self.config.update_frequency == 0:
            self._update_parameters(eta)
        
        # Update state
        self.prev_gamma = gamma.copy()
        self.state.n_updates += 1
        self.state.current_learning_rate = eta
        self.lr_scheduler.step()
        
        return {
            "n_updates": self.state.n_updates,
            "learning_rate": eta,
            "xi_norm": np.linalg.norm(xi),
            "gamma_entropy": -np.sum(gamma * np.log(gamma + 1e-10))
        }
    
    def _update_parameters(self, eta: float):
        """
        Update HMM parameters using online M-step.
        
        θ_new = (1 - η) · θ_old + η · θ_estimated
        """
        # Get estimates from sufficient statistics
        A_new = self.stats.get_transition_estimate()
        mu_new, sigma_new = self.stats.get_emission_params()
        
        # Blend with current parameters
        self.transition_matrix = (1 - eta) * self.transition_matrix + eta * A_new
        self.emission_means = (1 - eta) * self.emission_means + eta * mu_new
        self.emission_stds = (1 - eta) * self.emission_stds + eta * sigma_new
        
        # Ensure valid transition matrix
        self.transition_matrix = np.maximum(self.transition_matrix, self.config.regularization)
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Ensure valid emission parameters
        self.emission_stds = np.maximum(self.emission_stds, 0.001)
    
    def get_parameters(self) -> Dict:
        """Get current parameter estimates."""
        return {
            "transition_matrix": self.transition_matrix.copy(),
            "emission_means": self.emission_means.copy(),
            "emission_stds": self.emission_stds.copy(),
            "n_updates": self.state.n_updates,
            "learning_rate": self.state.current_learning_rate
        }
    
    def set_parameters(self, transition: np.ndarray = None,
                       means: np.ndarray = None, stds: np.ndarray = None):
        """Set parameters externally."""
        if transition is not None:
            self.transition_matrix = transition.copy()
        if means is not None:
            self.emission_means = means.copy()
        if stds is not None:
            self.emission_stds = stds.copy()
    
    def reset(self):
        """Reset online learner."""
        self.stats.reset()
        self.lr_scheduler.reset()
        self.state = OnlineLearningState()
        self.prev_gamma = None


# ============================================================================
# ADAPTIVE FORGETTING FACTOR
# ============================================================================

class AdaptiveForgettingFactor:
    """
    Adaptive forgetting factor based on regime change detection.
    
    When regime changes are detected, decrease λ to forget old data faster.
    When regime is stable, increase λ to use more history.
    
    λ_t = λ_base + (λ_max - λ_base) · stability_score
    """
    
    def __init__(self, 
                 lambda_base: float = 0.98,
                 lambda_max: float = 0.999,
                 stability_window: int = 20):
        self.lambda_base = lambda_base
        self.lambda_max = lambda_max
        self.stability_window = stability_window
        self.state_history = []
    
    def update(self, current_state: int) -> float:
        """
        Update and return adaptive forgetting factor.
        
        Args:
            current_state: Current most likely state
            
        Returns:
            Adaptive forgetting factor λ_t
        """
        self.state_history.append(current_state)
        
        if len(self.state_history) > self.stability_window:
            self.state_history.pop(0)
        
        # Compute stability score (fraction of time in same state)
        if len(self.state_history) >= 2:
            transitions = sum(1 for i in range(1, len(self.state_history)) 
                            if self.state_history[i] != self.state_history[i-1])
            stability = 1 - transitions / (len(self.state_history) - 1)
        else:
            stability = 0.5
        
        # Adaptive lambda
        lambda_t = self.lambda_base + (self.lambda_max - self.lambda_base) * stability
        
        return lambda_t
    
    def reset(self):
        """Reset history."""
        self.state_history = []


# ============================================================================
# ONLINE HMM LEARNER (UNIFIED INTERFACE)
# ============================================================================

class OnlineHMMLearner:
    """
    Unified Online HMM Learning System.
    
    Combines:
    - Online Baum-Welch for parameter updates
    - Adaptive forgetting factor
    - Learning rate scheduling
    
    Usage:
        learner = OnlineHMMLearner(n_states=8)
        
        for observation, state_probs in data_stream:
            learner.update(observation, state_probs)
            params = learner.get_parameters()
    """
    
    def __init__(self, n_states: int = 8, config: OnlineLearningConfig = None):
        self.config = config or OnlineLearningConfig(n_states=n_states)
        self.n_states = n_states
        
        self.baum_welch = OnlineBaumWelch(self.config)
        self.adaptive_lambda = AdaptiveForgettingFactor()
        
        self.update_count = 0
    
    def update(self, observation: float, state_probs: np.ndarray) -> Dict:
        """
        Update HMM parameters with new observation.
        
        Args:
            observation: Current observation
            state_probs: Current state probabilities from inference
            
        Returns:
            Update statistics
        """
        # Get adaptive forgetting factor
        current_state = int(np.argmax(state_probs))
        lambda_t = self.adaptive_lambda.update(current_state)
        
        # Update forgetting factor in sufficient statistics
        self.baum_welch.stats.lambda_ = lambda_t
        
        # Perform online update
        stats = self.baum_welch.update(observation, state_probs)
        stats["forgetting_factor"] = lambda_t
        
        self.update_count += 1
        return stats
    
    def get_parameters(self) -> Dict:
        """Get current HMM parameters."""
        return self.baum_welch.get_parameters()
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get current transition matrix."""
        return self.baum_welch.transition_matrix.copy()
    
    def get_emission_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get emission parameters (means, stds)."""
        return (self.baum_welch.emission_means.copy(),
                self.baum_welch.emission_stds.copy())
    
    def reset(self):
        """Reset learner."""
        self.baum_welch.reset()
        self.adaptive_lambda.reset()
        self.update_count = 0
