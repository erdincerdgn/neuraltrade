"""
Institutional-Grade Market Regime Detection
Author: Erdinc Erdogan
Purpose: 8-state HMM with hybrid Gaussian/Student-t emissions, Viterbi decoding, and hysteresis logic for robust regime classification.
References:
- Hidden Markov Models (Rabiner, 1989)
- Student-t Distribution for Fat Tails
- Regime-Switching Models (Hamilton, 1989)
Usage:
    detector = RegimeDetector(n_states=8)
    detector.fit(returns)
    regime = detector.predict(new_observations)
"""
import numpy as np
from scipy import stats
from scipy.special import gammaln
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque
import threading


# ============================================================================
# EXTENDED MARKET REGIME ENUM
# ============================================================================

class MarketRegime(Enum):
    """
    Extended 8-state market regime classification.
    
    Trend-based states:
    - BULL_TREND: Sustained upward momentum
    - BEAR_TREND: Sustained downward momentum
    - SIDEWAYS: Range-bound, low directional bias
    
    Volatility-based states:
    - LOW_VOLATILITY: Below 25th percentile
    - NORMAL_VOLATILITY: 25th-75th percentile
    - HIGH_VOLATILITY: 75th-95th percentile
    - EXTREME_VOLATILITY: Above 95th percentile
    - CRISIS: Tail events, extreme stress
    """
    # Trend-based
    BULL_TREND = auto()
    BEAR_TREND = auto()
    SIDEWAYS = auto()
    # Volatility-based
    LOW_VOLATILITY = auto()
    NORMAL_VOLATILITY = auto()
    HIGH_VOLATILITY = auto()
    EXTREME_VOLATILITY = auto()
    CRISIS = auto()


# ============================================================================
# REGIME PARAMETERS
# ============================================================================

@dataclass
class EmissionParams:
    """Parameters for emission distribution."""
    distribution: str  # "gaussian" or "student_t"
    mu: float          # Mean
    sigma: float       # Standard deviation
    nu: Optional[float] = None  # Degrees of freedom (for Student's t)


@dataclass
class RegimeConfig:
    """Configuration for a market regime."""
    regime: MarketRegime
    emission: EmissionParams
    cvar_multiplier: float
    kelly_multiplier: float
    description: str


# Default regime configurations
REGIME_CONFIGS: Dict[MarketRegime, RegimeConfig] = {
    MarketRegime.BULL_TREND: RegimeConfig(
        regime=MarketRegime.BULL_TREND,
        emission=EmissionParams("gaussian", mu=0.0008, sigma=0.010, nu=None),
        cvar_multiplier=1.20,
        kelly_multiplier=2.00,  # Full Kelly
        description="Sustained upward momentum, favorable conditions"
    ),
    MarketRegime.BEAR_TREND: RegimeConfig(
        regime=MarketRegime.BEAR_TREND,
        emission=EmissionParams("student_t", mu=-0.0010, sigma=0.015, nu=5.0),
        cvar_multiplier=0.70,
        kelly_multiplier=0.25,  # Quarter Kelly
        description="Sustained downward momentum, defensive positioning"
    ),
    MarketRegime.SIDEWAYS: RegimeConfig(
        regime=MarketRegime.SIDEWAYS,
        emission=EmissionParams("gaussian", mu=0.0000, sigma=0.008, nu=None),
        cvar_multiplier=1.00,
        kelly_multiplier=0.50,  # Half Kelly
        description="Range-bound market, neutral positioning"
    ),
    MarketRegime.LOW_VOLATILITY: RegimeConfig(
        regime=MarketRegime.LOW_VOLATILITY,
        emission=EmissionParams("gaussian", mu=0.0003, sigma=0.006, nu=None),
        cvar_multiplier=1.30,
        kelly_multiplier=1.50,
        description="Calm market, can increase exposure"
    ),
    MarketRegime.NORMAL_VOLATILITY: RegimeConfig(
        regime=MarketRegime.NORMAL_VOLATILITY,
        emission=EmissionParams("gaussian", mu=0.0002, sigma=0.012, nu=None),
        cvar_multiplier=1.00,
        kelly_multiplier=0.50,  # Half Kelly (default)
        description="Normal market conditions"
    ),
    MarketRegime.HIGH_VOLATILITY: RegimeConfig(
        regime=MarketRegime.HIGH_VOLATILITY,
        emission=EmissionParams("student_t", mu=0.0000, sigma=0.025, nu=3.0),
        cvar_multiplier=0.80,  # Tighten by 20%
        kelly_multiplier=0.50,  # Half Kelly
        description="Elevated volatility, reduce exposure"
    ),
    MarketRegime.EXTREME_VOLATILITY: RegimeConfig(
        regime=MarketRegime.EXTREME_VOLATILITY,
        emission=EmissionParams("student_t", mu=-0.0005, sigma=0.035, nu=2.5),
        cvar_multiplier=0.60,
        kelly_multiplier=0.25,  # Quarter Kelly
        description="Extreme volatility, defensive mode"
    ),
    MarketRegime.CRISIS: RegimeConfig(
        regime=MarketRegime.CRISIS,
        emission=EmissionParams("student_t", mu=-0.0020, sigma=0.040, nu=2.5),
        cvar_multiplier=0.50,  # Tighten by 50%
        kelly_multiplier=0.10,  # Tenth Kelly
        description="Crisis conditions, maximum protection"
    ),
}


# ============================================================================
# TRANSITION MATRIX
# ============================================================================

# 8x8 Transition matrix for regime states
# Order: BULL, BEAR, SIDE, LOW_VOL, NORM_VOL, HIGH_VOL, EXT_VOL, CRISIS
TRANSITION_MATRIX = np.array([
    # BULL   BEAR   SIDE   LOW_V  NORM_V HIGH_V EXT_V  CRISIS
    [0.85,  0.02,  0.05,  0.03,  0.03,  0.01,  0.005, 0.005],  # From BULL
    [0.02,  0.82,  0.05,  0.01,  0.03,  0.04,  0.02,  0.01 ],  # From BEAR
    [0.08,  0.08,  0.70,  0.04,  0.06,  0.03,  0.005, 0.005],  # From SIDEWAYS
    [0.10,  0.02,  0.08,  0.65,  0.12,  0.02,  0.005, 0.005],  # From LOW_VOL
    [0.05,  0.05,  0.10,  0.08,  0.60,  0.08,  0.03,  0.01 ],  # From NORM_VOL
    [0.03,  0.07,  0.05,  0.02,  0.10,  0.55,  0.12,  0.06 ],  # From HIGH_VOL
    [0.02,  0.08,  0.03,  0.01,  0.05,  0.15,  0.50,  0.16 ],  # From EXT_VOL
    [0.02,  0.10,  0.03,  0.01,  0.04,  0.10,  0.20,  0.50 ],  # From CRISIS
])

# Initial state distribution (start in NORMAL_VOLATILITY)
INITIAL_DISTRIBUTION = np.array([0.10, 0.10, 0.15, 0.10, 0.35, 0.10, 0.05, 0.05])


# ============================================================================
# HYBRID EMISSION MODEL
# ============================================================================

class HybridEmissionModel:
    """
    Hybrid Gaussian + Student's t emission model.
    
    Uses Gaussian for stable regimes (Bull, Sideways, Low/Normal Vol)
    Uses Student's t for volatile regimes (Bear, High Vol, Extreme Vol, Crisis)
    
    Student's t provides fat tails for better crisis modeling:
    - nu=5: Moderate fat tails (Bear)
    - nu=3: Heavy fat tails (High Vol)
    - nu=2.5: Extreme fat tails (Crisis, Extreme Vol)
    """
    
    def __init__(self, regime_configs: Dict[MarketRegime, RegimeConfig] = None):
        self.configs = regime_configs or REGIME_CONFIGS
        self._regime_order = list(MarketRegime)
    
    def log_probability(self, observation: float, regime: MarketRegime) -> float:
        """
        Calculate log probability of observation given regime.
        
        Args:
            observation: Observed return value
            regime: Market regime state
            
        Returns:
            Log probability
        """
        config = self.configs[regime]
        emission = config.emission
        
        if emission.distribution == "gaussian":
            return self._gaussian_log_prob(observation, emission.mu, emission.sigma)
        else:  # student_t
            return self._student_t_log_prob(
                observation, emission.mu, emission.sigma, emission.nu
            )
    
    def _gaussian_log_prob(self, x: float, mu: float, sigma: float) -> float:
        """Gaussian log probability."""
        z = (x - mu) / sigma
        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * z * z
    
    def _student_t_log_prob(self, x: float, mu: float, sigma: float, nu: float) -> float:
        """Student's t log probability with location and scale."""
        z = (x - mu) / sigma
        log_prob = (
            gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi) - np.log(sigma)
            - ((nu + 1) / 2) * np.log(1 + z * z / nu)
        )
        return log_prob
    
    def emission_matrix(self, observations: np.ndarray) -> np.ndarray:
        """
        Calculate emission probability matrix for all observations and states.
        
        Args:
            observations: Array of observed returns
            
        Returns:
            Matrix of shape (n_observations, n_states) with log probabilities
        """
        n_obs = len(observations)
        n_states = len(self._regime_order)
        log_probs = np.zeros((n_obs, n_states))
        
        for t, obs in enumerate(observations):
            for s, regime in enumerate(self._regime_order):
                log_probs[t, s] = self.log_probability(obs, regime)
        
        return log_probs


# ============================================================================
# VITERBI ALGORITHM
# ============================================================================

class ViterbiDecoder:
    """
    Viterbi algorithm for finding most likely state sequence.
    
    Implements dynamic programming to find:
    argmax P(S1, S2, ..., ST | O1, O2, ..., OT)
    """
    
    def __init__(self, 
                 transition_matrix: np.ndarray,
                 initial_distribution: np.ndarray,
                 emission_model: HybridEmissionModel):
        self.A = transition_matrix
        self.pi = initial_distribution
        self.emission = emission_model
        self.n_states = len(initial_distribution)
        self._regime_order = list(MarketRegime)
    
    def decode(self, observations: np.ndarray) -> Tuple[List[MarketRegime], np.ndarray]:
        """
        Decode most likely state sequence using Viterbi algorithm.
        
        Args:
            observations: Array of observed returns
            
        Returns:
            Tuple of (state_sequence, state_probabilities)
        """
        T = len(observations)
        
        # Get emission log probabilities
        log_B = self.emission.emission_matrix(observations)
        
        # Initialize
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initial step
        log_delta[0] = np.log(self.pi + 1e-10) + log_B[0]
        
        # Forward pass
        log_A = np.log(self.A + 1e-10)
        for t in range(1, T):
            for j in range(self.n_states):
                candidates = log_delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                log_delta[t, j] = candidates[psi[t, j]] + log_B[t, j]
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(log_delta[T-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        # Convert to regime enums
        regime_sequence = [self._regime_order[s] for s in states]
        
        # Calculate state probabilities (normalized)
        state_probs = np.exp(log_delta - np.max(log_delta, axis=1, keepdims=True))
        state_probs = state_probs / (state_probs.sum(axis=1, keepdims=True) + 1e-10)
        
        return regime_sequence, state_probs


# ============================================================================
# FORWARD-BACKWARD ALGORITHM
# ============================================================================

class ForwardBackward:
    """
    Forward-Backward algorithm for state probability estimation.
    
    Calculates P(St = i | O1, ..., OT) for all t and i.
    """
    
    def __init__(self,
                 transition_matrix: np.ndarray,
                 initial_distribution: np.ndarray,
                 emission_model: HybridEmissionModel):
        self.A = transition_matrix
        self.pi = initial_distribution
        self.emission = emission_model
        self.n_states = len(initial_distribution)
    
    def compute(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute state probabilities using forward-backward.
        
        Returns:
            gamma: Array of shape (T, n_states) with P(St=i | observations)
        """
        T = len(observations)
        log_B = self.emission.emission_matrix(observations)
        B = np.exp(log_B - np.max(log_B, axis=1, keepdims=True))
        
        # Forward pass
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.pi * B[0]
        alpha[0] /= alpha[0].sum() + 1e-10
        
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            alpha[t] /= alpha[t].sum() + 1e-10
        
        # Backward pass
        beta = np.zeros((T, self.n_states))
        beta[T-1] = 1.0
        
        for t in range(T-2, -1, -1):
            beta[t] = self.A @ (B[t+1] * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-10
        
        # State probabilities
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
        
        return gamma


# ============================================================================
# REGIME TRANSITION MANAGER (HYSTERESIS)
# ============================================================================

@dataclass
class RegimeTransition:
    """Record of a regime transition."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    confidence: float
    confirmation_days: int


class RegimeTransitionManager:
    """
    Manages regime transitions with hysteresis to prevent whipsaws.
    
    Confirmation Requirements:
    - Bull Trend: 5 days at >70% probability
    - Bear Trend: 3 days at >70% probability
    - High Vol/Crisis: 1 day at >80% probability
    - Exit High Vol: 5 days at <50% probability
    """
    
    CONFIRMATION_RULES = {
        MarketRegime.BULL_TREND: {"days": 5, "threshold": 0.70},
        MarketRegime.BEAR_TREND: {"days": 3, "threshold": 0.70},
        MarketRegime.SIDEWAYS: {"days": 4, "threshold": 0.65},
        MarketRegime.LOW_VOLATILITY: {"days": 3, "threshold": 0.65},
        MarketRegime.NORMAL_VOLATILITY: {"days": 2, "threshold": 0.60},
        MarketRegime.HIGH_VOLATILITY: {"days": 1, "threshold": 0.80},
        MarketRegime.EXTREME_VOLATILITY: {"days": 1, "threshold": 0.80},
        MarketRegime.CRISIS: {"days": 1, "threshold": 0.80},
    }
    
    STAY_THRESHOLD = 0.40  # Threshold to stay in current regime
    
    def __init__(self):
        self.current_regime = MarketRegime.NORMAL_VOLATILITY
        self.candidate_regime: Optional[MarketRegime] = None
        self.candidate_days: int = 0
        self.candidate_probs: List[float] = []
        self.transition_history: List[RegimeTransition] = []
        self._lock = threading.Lock()
    
    def update(self, 
               state_probabilities: np.ndarray,
               regime_order: List[MarketRegime]) -> Tuple[MarketRegime, float, bool]:
        """
        Update regime with hysteresis logic.
        
        Args:
            state_probabilities: Probabilities for each state
            regime_order: Order of regimes in probability array
            
        Returns:
            Tuple of (current_regime, confidence, regime_changed)
        """
        with self._lock:
            # Find most likely regime
            max_idx = np.argmax(state_probabilities)
            max_prob = state_probabilities[max_idx]
            likely_regime = regime_order[max_idx]
            
            # Get current regime probability
            current_idx = regime_order.index(self.current_regime)
            current_prob = state_probabilities[current_idx]
            
            regime_changed = False
            
            # Check if we should stay in current regime
            if current_prob >= self.STAY_THRESHOLD:
                # Reset candidate if different
                if self.candidate_regime != likely_regime:
                    self.candidate_regime = likely_regime
                    self.candidate_days = 0
                    self.candidate_probs = []
            
            # Check candidate regime
            if likely_regime != self.current_regime:
                rules = self.CONFIRMATION_RULES[likely_regime]
                
                if self.candidate_regime == likely_regime:
                    self.candidate_days += 1
                    self.candidate_probs.append(max_prob)
                else:
                    self.candidate_regime = likely_regime
                    self.candidate_days = 1
                    self.candidate_probs = [max_prob]
                
                # Check confirmation
                avg_prob = np.mean(self.candidate_probs)
                if (self.candidate_days >= rules["days"] and 
                    avg_prob >= rules["threshold"]):
                    # Transition confirmed
                    transition = RegimeTransition(
                        from_regime=self.current_regime,
                        to_regime=likely_regime,
                        timestamp=datetime.now(),
                        confidence=avg_prob,
                        confirmation_days=self.candidate_days
                    )
                    self.transition_history.append(transition)
                    self.current_regime = likely_regime
                    self.candidate_regime = None
                    self.candidate_days = 0
                    self.candidate_probs = []
                    regime_changed = True
            else:
                # Already in likely regime
                self.candidate_regime = None
                self.candidate_days = 0
                self.candidate_probs = []
            
            return self.current_regime, max_prob, regime_changed
    
    def force_regime(self, regime: MarketRegime, reason: str = "Manual override"):
        """Force a regime change (for emergency situations)."""
        with self._lock:
            transition = RegimeTransition(
                from_regime=self.current_regime,
                to_regime=regime,
                timestamp=datetime.now(),
                confidence=1.0,
                confirmation_days=0
            )
            self.transition_history.append(transition)
            self.current_regime = regime
            self.candidate_regime = None
            self.candidate_days = 0
            self.candidate_probs = []


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class RegimeFeatureExtractor:
    """
    Extracts features for regime detection.
    
    Features:
    - Returns (log returns)
    - Volatility (EWMA)
    - Trend (SMA crossover)
    - Momentum (RSI-based)
    """
    
    def __init__(self, 
                 ewma_lambda: float = 0.94,
                 short_window: int = 20,
                 long_window: int = 50):
        self.ewma_lambda = ewma_lambda
        self.short_window = short_window
        self.long_window = long_window
    
    def extract(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from price series.
        
        Args:
            prices: Array of prices
            
        Returns:
            Dictionary of feature arrays
        """
        # Log returns
        returns = np.diff(np.log(prices))
        
        # EWMA volatility
        volatility = self._ewma_volatility(returns)
        
        # Trend (SMA crossover)
        trend = self._sma_trend(prices)
        
        # Momentum (simplified RSI)
        momentum = self._momentum(returns)
        
        return {
            "returns": returns,
            "volatility": volatility,
            "trend": trend[1:],  # Align with returns
            "momentum": momentum
        }
    
    def _ewma_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Calculate EWMA volatility."""
        n = len(returns)
        vol = np.zeros(n)
        vol[0] = returns[0] ** 2
        
        for t in range(1, n):
            vol[t] = self.ewma_lambda * vol[t-1] + (1 - self.ewma_lambda) * returns[t-1] ** 2
        
        return np.sqrt(vol)
    
    def _sma_trend(self, prices: np.ndarray) -> np.ndarray:
        """Calculate SMA trend indicator."""
        n = len(prices)
        trend = np.zeros(n)
        
        for t in range(self.long_window, n):
            sma_short = np.mean(prices[t-self.short_window:t])
            sma_long = np.mean(prices[t-self.long_window:t])
            trend[t] = (sma_short - sma_long) / sma_long
        
        return trend
    
    def _momentum(self, returns: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate momentum indicator."""
        n = len(returns)
        momentum = np.zeros(n)
        
        for t in range(window, n):
            gains = np.sum(np.maximum(returns[t-window:t], 0))
            losses = np.sum(np.abs(np.minimum(returns[t-window:t], 0)))
            
            if losses > 0:
                rs = gains / losses
                momentum[t] = (rs - 1) / (rs + 1)  # Normalized to [-1, 1]
            else:
                momentum[t] = 1.0
        
        return momentum


# ============================================================================
# MAIN REGIME DETECTOR
# ============================================================================

@dataclass
class RegimeDetectionResult:
    """Result of regime detection."""
    current_regime: MarketRegime
    confidence: float
    regime_changed: bool
    state_probabilities: Dict[MarketRegime, float]
    cvar_multiplier: float
    kelly_multiplier: float
    regime_description: str
    timestamp: datetime


class MarketRegimeDetector:
    """
    Institutional-Grade Market Regime Detector.
    
    Combines:
    - Hidden Markov Model with 8 states
    - Hybrid Gaussian + Student's t emissions
    - Viterbi decoding for state sequence
    - Forward-Backward for state probabilities
    - Hysteresis logic for transition confirmation
    
    Usage:
        detector = MarketRegimeDetector()
        result = detector.detect(returns_array)
        
        # Get adaptive parameters
        cvar_mult = result.cvar_multiplier
        kelly_mult = result.kelly_multiplier
    """
    
    def __init__(self,
                 transition_matrix: np.ndarray = None,
                 initial_distribution: np.ndarray = None,
                 regime_configs: Dict[MarketRegime, RegimeConfig] = None):
        """
        Initialize regime detector.
        
        Args:
            transition_matrix: Custom transition matrix (8x8)
            initial_distribution: Custom initial distribution
            regime_configs: Custom regime configurations
        """
        self.A = transition_matrix if transition_matrix is not None else TRANSITION_MATRIX
        self.pi = initial_distribution if initial_distribution is not None else INITIAL_DISTRIBUTION
        self.configs = regime_configs or REGIME_CONFIGS
        
        self._regime_order = list(MarketRegime)
        
        # Initialize components
        self.emission_model = HybridEmissionModel(self.configs)
        self.viterbi = ViterbiDecoder(self.A, self.pi, self.emission_model)
        self.forward_backward = ForwardBackward(self.A, self.pi, self.emission_model)
        self.transition_manager = RegimeTransitionManager()
        self.feature_extractor = RegimeFeatureExtractor()
        
        # History
        self.detection_history: List[RegimeDetectionResult] = []
        self._lock = threading.Lock()
    
    def detect(self, 
               returns: np.ndarray,
               use_viterbi: bool = True) -> RegimeDetectionResult:
        """
        Detect current market regime.
        
        Args:
            returns: Array of recent returns (at least 20 observations)
            use_viterbi: Use Viterbi (True) or Forward-Backward (False)
            
        Returns:
            RegimeDetectionResult with regime and adaptive parameters
        """
        with self._lock:
            if len(returns) < 20:
                # Not enough data, return default
                return self._default_result()
            
            # Get state probabilities
            if use_viterbi:
                _, state_probs = self.viterbi.decode(returns)
                current_probs = state_probs[-1]
            else:
                gamma = self.forward_backward.compute(returns)
                current_probs = gamma[-1]
            
            # Update with hysteresis
            regime, confidence, changed = self.transition_manager.update(
                current_probs, self._regime_order
            )
            
            # Get regime config
            config = self.configs[regime]
            
            # Build probability dict
            prob_dict = {
                r: current_probs[i] 
                for i, r in enumerate(self._regime_order)
            }
            
            result = RegimeDetectionResult(
                current_regime=regime,
                confidence=confidence,
                regime_changed=changed,
                state_probabilities=prob_dict,
                cvar_multiplier=config.cvar_multiplier,
                kelly_multiplier=config.kelly_multiplier,
                regime_description=config.description,
                timestamp=datetime.now()
            )
            
            self.detection_history.append(result)
            return result
    
    def detect_from_prices(self, prices: np.ndarray) -> RegimeDetectionResult:
        """
        Detect regime from price series.
        
        Args:
            prices: Array of prices
            
        Returns:
            RegimeDetectionResult
        """
        if len(prices) < 21:
            return self._default_result()
        
        returns = np.diff(np.log(prices))
        return self.detect(returns)
    
    def _default_result(self) -> RegimeDetectionResult:
        """Return default result when insufficient data."""
        config = self.configs[MarketRegime.NORMAL_VOLATILITY]
        return RegimeDetectionResult(
            current_regime=MarketRegime.NORMAL_VOLATILITY,
            confidence=0.5,
            regime_changed=False,
            state_probabilities={r: 1/8 for r in MarketRegime},
            cvar_multiplier=config.cvar_multiplier,
            kelly_multiplier=config.kelly_multiplier,
            regime_description=config.description,
            timestamp=datetime.now()
        )
    
    def get_current_regime(self) -> MarketRegime:
        """Get current regime without new detection."""
        return self.transition_manager.current_regime
    
    def get_adaptive_parameters(self) -> Dict[str, float]:
        """Get current adaptive parameters."""
        regime = self.transition_manager.current_regime
        config = self.configs[regime]
        return {
            "regime": regime.name,
            "cvar_multiplier": config.cvar_multiplier,
            "kelly_multiplier": config.kelly_multiplier
        }
    
    def force_crisis_mode(self):
        """Force crisis mode (emergency)."""
        self.transition_manager.force_regime(
            MarketRegime.CRISIS, 
            "Emergency crisis mode activated"
        )
    
    def generate_report(self) -> str:
        """Generate regime detection report."""
        regime = self.transition_manager.current_regime
        config = self.configs[regime]
        
        # Get recent history
        recent = self.detection_history[-10:] if self.detection_history else []
        
        report = f"""
<regime_detection_v3>
MARKET REGIME DETECTION REPORT
══════════════════════════════════════════════════════════════

📊 CURRENT REGIME: {regime.name}
   Description: {config.description}
   
📈 ADAPTIVE PARAMETERS:
   • CVaR Multiplier: {config.cvar_multiplier:.2f}x
   • Kelly Multiplier: {config.kelly_multiplier:.2f}x
   
📉 EMISSION MODEL:
   • Distribution: {config.emission.distribution}
   • Mean (μ): {config.emission.mu*100:.3f}% daily
   • Volatility (σ): {config.emission.sigma*100:.2f}% daily
   • Degrees of Freedom (ν): {config.emission.nu or 'N/A (Gaussian)'}

🔄 TRANSITION HISTORY (Last 5):
"""
        for t in self.transition_manager.transition_history[-5:]:
            report += f"   • {t.from_regime.name} → {t.to_regime.name} "
            report += f"(conf: {t.confidence:.1%}, days: {t.confirmation_days})\n"
        
        report += """
</regime_detection_v3>
"""
        return report


# ============================================================================
# ADAPTIVE RISK MANAGER
# ============================================================================

class AdaptiveRiskManager:
    """
    Manages risk parameters based on detected regime.
    
    Integrates with:
    - RiskEngine: Adjusts CVaR thresholds
    - PortfolioOptimizer: Adjusts Kelly fraction
    - CircuitBreaker: Adjusts trigger thresholds
    """
    
    def __init__(self, regime_detector: MarketRegimeDetector):
        self.detector = regime_detector
        self.base_cvar_threshold = 0.03  # 3% base CVaR limit
        self.base_kelly_fraction = 0.125  # 12.5% base Kelly (Half of 25%)
        self.base_max_drawdown = 0.05  # 5% base MDD limit
    
    def get_adjusted_cvar_threshold(self) -> float:
        """Get regime-adjusted CVaR threshold."""
        params = self.detector.get_adaptive_parameters()
        return self.base_cvar_threshold * params["cvar_multiplier"]
    
    def get_adjusted_kelly_fraction(self) -> float:
        """Get regime-adjusted Kelly fraction."""
        params = self.detector.get_adaptive_parameters()
        adjusted = self.base_kelly_fraction * params["kelly_multiplier"]
        return min(adjusted, 0.25)  # Cap at 25% (Full Kelly max)
    
    def get_adjusted_max_drawdown(self) -> float:
        """Get regime-adjusted max drawdown threshold."""
        params = self.detector.get_adaptive_parameters()
        return self.base_max_drawdown * params["cvar_multiplier"]
    
    def get_all_adjusted_parameters(self) -> Dict[str, float]:
        """Get all adjusted risk parameters."""
        params = self.detector.get_adaptive_parameters()
        return {
            "regime": params["regime"],
            "cvar_threshold": self.get_adjusted_cvar_threshold(),
            "kelly_fraction": self.get_adjusted_kelly_fraction(),
            "max_drawdown": self.get_adjusted_max_drawdown(),
            "cvar_multiplier": params["cvar_multiplier"],
            "kelly_multiplier": params["kelly_multiplier"]
        }
