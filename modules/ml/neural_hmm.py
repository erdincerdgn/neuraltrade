"""
Neural HMM (LSTM-HMM Hybrid) for Regime Detection
Author: Erdinc Erdogan
Purpose: Combines LSTM temporal encoding with HMM state decoding using neural emission networks for enhanced regime detection accuracy.
References:
- Neural HMM (Tran et al., 2016)
- LSTM Encoder-Decoder
- Hybrid Viterbi Inference
Usage:
    model = NeuralHMM(n_states=8, lstm_hidden=64)
    model.fit(returns_sequence)
    regime = model.decode(new_observations)
"""
import numpy as np
from modules.core.safe_math import (
    safe_log, safe_exp, safe_sqrt, safe_divide, 
    validate_array, validate_probability
)
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class Activation:
    """Activation functions for neural networks."""
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + safe_exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(safe_exp(np.clip(x, -500, 500)))


# ============================================================================
# LSTM CELL
# ============================================================================

class LSTMCell:
    """
    LSTM Cell implementation in NumPy.
    
    Gates:
    - f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  (forget)
    - i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  (input)
    - c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  (candidate)
    - o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  (output)
    - c_t = f_t * c_{t-1} + i_t * c̃_t
    - h_t = o_t * tanh(c_t)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W = np.random.randn(4 * hidden_dim, input_dim + hidden_dim) * scale
        self.b = np.zeros(4 * hidden_dim)
        self.b[:hidden_dim] = 1.0  # Forget gate bias
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, 
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        combined = np.concatenate([h_prev, x], axis=-1)
        gates = combined @ self.W.T + self.b
        
        f = Activation.sigmoid(gates[..., :self.hidden_dim])
        i = Activation.sigmoid(gates[..., self.hidden_dim:2*self.hidden_dim])
        c_tilde = Activation.tanh(gates[..., 2*self.hidden_dim:3*self.hidden_dim])
        o = Activation.sigmoid(gates[..., 3*self.hidden_dim:])
        
        c = f * c_prev + i * c_tilde
        h = o * Activation.tanh(c)
        return h, c
    
    def init_hidden(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros((batch_size, self.hidden_dim)), np.zeros((batch_size, self.hidden_dim))


# ============================================================================
# LSTM ENCODER
# ============================================================================

class LSTMEncoder:
    """LSTM Encoder for temporal feature extraction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layers = []
        for i in range(num_layers):
            layer_input = input_dim if i == 0 else hidden_dim
            self.layers.append(LSTMCell(layer_input, hidden_dim))
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        
        seq_len, batch_size, _ = x.shape
        states = [layer.init_hidden(batch_size) for layer in self.layers]
        
        outputs = []
        for t in range(seq_len):
            layer_input = x[t]
            for i, layer in enumerate(self.layers):
                h, c = states[i]
                h_new, c_new = layer.forward(layer_input, h, c)
                states[i] = (h_new, c_new)
                layer_input = h_new
            outputs.append(layer_input)
        
        return np.stack(outputs, axis=0), states
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        _, states = self.forward(x)
        return states[-1][0]


# ============================================================================
# NEURAL EMISSION NETWORK
# ============================================================================

class NeuralEmissionNetwork:
    """
    Neural network for emission probability computation.
    
    P(O_t | S_t=k, h_t) = N(O_t; μ_k(h_t), σ_k(h_t)^2)
    """
    
    def __init__(self, hidden_dim: int, n_states: int = 8, emission_hidden: int = 32):
        self.hidden_dim = hidden_dim
        self.n_states = n_states
        
        scale = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(emission_hidden, hidden_dim) * scale
        self.b1 = np.zeros(emission_hidden)
        self.W_mu = np.random.randn(n_states, emission_hidden) * scale * 0.1
        self.b_mu = np.zeros(n_states)
        self.W_logstd = np.random.randn(n_states, emission_hidden) * scale * 0.1
        self.b_logstd = np.ones(n_states) * (-2.0)
    
    def forward(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hidden = Activation.relu(h @ self.W1.T + self.b1)
        mu = hidden @ self.W_mu.T + self.b_mu
        log_std = hidden @ self.W_logstd.T + self.b_logstd
        sigma = Activation.softplus(log_std) + 0.001
        return mu, sigma
    
    def log_prob(self, observation: float, h: np.ndarray) -> np.ndarray:
        mu, sigma = self.forward(h)
        z = (observation - mu) / sigma
        return -0.5 * safe_log(2 * np.pi) - safe_log(sigma) - 0.5 * z * z


# ============================================================================
# NEURAL HMM CONFIGURATION
# ============================================================================

@dataclass
class NeuralHMMConfig:
    input_dim: int = 5
    hidden_dim: int = 64
    n_states: int = 8
    n_lstm_layers: int = 2
    emission_hidden: int = 32
    sequence_length: int = 20


@dataclass
class NeuralHMMResult:
    current_state: int
    state_probabilities: np.ndarray
    confidence: float
    hidden_state: np.ndarray
    emission_params: Dict[int, Tuple[float, float]]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# NEURAL HMM MODEL
# ============================================================================

class NeuralHMM:
    """
    Neural Hidden Markov Model (LSTM-HMM Hybrid).
    
    Architecture:
        Input → LSTM Encoder → Neural Emissions → HMM Inference
    """
    
    DEFAULT_TRANSITION = np.array([
        [0.85, 0.02, 0.05, 0.03, 0.03, 0.01, 0.005, 0.005],
        [0.02, 0.82, 0.05, 0.01, 0.03, 0.04, 0.02, 0.01],
        [0.08, 0.08, 0.70, 0.04, 0.06, 0.03, 0.005, 0.005],
        [0.10, 0.02, 0.08, 0.65, 0.12, 0.02, 0.005, 0.005],
        [0.05, 0.05, 0.10, 0.08, 0.60, 0.08, 0.03, 0.01],
        [0.03, 0.07, 0.05, 0.02, 0.10, 0.55, 0.12, 0.06],
        [0.02, 0.08, 0.03, 0.01, 0.05, 0.15, 0.50, 0.16],
        [0.02, 0.10, 0.03, 0.01, 0.04, 0.10, 0.20, 0.50],
    ])
    
    DEFAULT_INITIAL = np.array([0.10, 0.10, 0.15, 0.10, 0.35, 0.10, 0.05, 0.05])
    
    STATE_NAMES = ["BULL", "BEAR", "SIDEWAYS", "LOW_VOL", "NORMAL", "HIGH_VOL", "EXTREME", "CRISIS"]
    
    def __init__(self, config: NeuralHMMConfig = None):
        self.config = config or NeuralHMMConfig()
        
        self.encoder = LSTMEncoder(
            self.config.input_dim, self.config.hidden_dim, self.config.n_lstm_layers
        )
        self.emission_net = NeuralEmissionNetwork(
            self.config.hidden_dim, self.config.n_states, self.config.emission_hidden
        )
        
        self.transition_matrix = self.DEFAULT_TRANSITION.copy()
        self.initial_distribution = self.DEFAULT_INITIAL.copy()
        self.current_state_probs = self.initial_distribution.copy()
        self.hidden_state = None
    
    def encode_sequence(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        
        if len(observations) < self.config.sequence_length:
            pad_len = self.config.sequence_length - len(observations)
            padding = np.zeros((pad_len, observations.shape[1]))
            observations = np.vstack([padding, observations])
        
        observations = observations[-self.config.sequence_length:]
        _, states = self.encoder.forward(observations)
        self.hidden_state = states[-1][0].squeeze()
        return self.hidden_state
    
    def forward_step(self, observation: float, hidden: np.ndarray) -> np.ndarray:
        log_emission = self.emission_net.log_prob(observation, hidden)
        log_alpha_prev = safe_log(self.current_state_probs + 1e-10)
        log_A = safe_log(self.transition_matrix + 1e-10)
        
        log_alpha = np.zeros(self.config.n_states)
        for j in range(self.config.n_states):
            log_alpha[j] = np.logaddexp.reduce(log_alpha_prev + log_A[:, j]) + log_emission[j]
        
        log_alpha = log_alpha - np.logaddexp.reduce(log_alpha)
        self.current_state_probs = safe_exp(log_alpha)
        return self.current_state_probs
    
    def infer(self, observations: np.ndarray, current_return: float = None) -> NeuralHMMResult:
        hidden = self.encode_sequence(observations)
        
        if current_return is None:
            current_return = observations[-1, 0] if observations.ndim > 1 else observations[-1]
        
        state_probs = self.forward_step(current_return, hidden)
        mu, sigma = self.emission_net.forward(hidden)
        emission_params = {i: (float(mu[i]), float(sigma[i])) for i in range(self.config.n_states)}
        
        entropy = -np.sum(state_probs * safe_log(state_probs + 1e-10))
        confidence = 1 - entropy / safe_log(self.config.n_states)
        
        return NeuralHMMResult(
            current_state=int(np.argmax(state_probs)),
            state_probabilities=state_probs,
            confidence=confidence,
            hidden_state=hidden,
            emission_params=emission_params
        )
    
    def viterbi_decode(self, observations: np.ndarray) -> List[int]:
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        
        T = len(observations)
        outputs, _ = self.encoder.forward(observations)
        
        log_delta = np.zeros((T, self.config.n_states))
        psi = np.zeros((T, self.config.n_states), dtype=int)
        
        hidden_0 = outputs[0].squeeze()
        log_delta[0] = safe_log(self.initial_distribution + 1e-10) + self.emission_net.log_prob(observations[0, 0], hidden_0)
        
        log_A = safe_log(self.transition_matrix + 1e-10)
        for t in range(1, T):
            hidden_t = outputs[t].squeeze()
            log_emission_t = self.emission_net.log_prob(observations[t, 0], hidden_t)
            for j in range(self.config.n_states):
                candidates = log_delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                log_delta[t, j] = candidates[psi[t, j]] + log_emission_t[j]
        
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(log_delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states.tolist()
    
    def reset_state(self):
        self.current_state_probs = self.initial_distribution.copy()
        self.hidden_state = None
    
    def get_state_name(self, idx: int) -> str:
        return self.STATE_NAMES[idx]
