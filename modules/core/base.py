"""
Institutional-Grade Base Classes - Core Abstractions and Protocols
Author: Erdinc Erdogan
Purpose: Provides institutional-grade base classes, enums, and protocols including Kelly
Criterion position sizing, CVaR-based risk tiers, and Bayesian confidence intervals.
References:
- Kelly Criterion: J.L. Kelly Jr. (1956) "A New Interpretation of Information Rate"
- Conditional Value at Risk (CVaR) Theory
- Hidden Markov Model Regime Classification
Usage:
    from modules.core.base import BaseOrchestrator, RiskTier, MarketRegime
    class MyOrchestrator(BaseOrchestrator): ...
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Protocol, runtime_checkable
import numpy as np
from datetime import datetime


# ============================================================================
# STATISTICAL CONSTANTS (RiskMetrics & Institutional Standards)
# ============================================================================

class StatisticalConstants:
    """
    Institutional-grade statistical constants.
    Based on RiskMetrics, Basel III, and HFT standards.
    """
    # Exponential decay factors (RiskMetrics standard)
    EWMA_LAMBDA_DAILY: float = 0.94      # Daily volatility decay
    EWMA_LAMBDA_WEEKLY: float = 0.97     # Weekly decay
    EWMA_LAMBDA_MONTHLY: float = 0.99    # Monthly decay
    
    # Confidence intervals for VaR/CVaR
    CONFIDENCE_90: float = 0.90
    CONFIDENCE_95: float = 0.95
    CONFIDENCE_99: float = 0.99
    CONFIDENCE_999: float = 0.999        # Extreme tail risk
    
    # Z-scores for confidence levels
    Z_SCORE_90: float = 1.645
    Z_SCORE_95: float = 1.960
    Z_SCORE_99: float = 2.576
    Z_SCORE_999: float = 3.291
    
    # Kelly Criterion constraints
    KELLY_FRACTION_MAX: float = 0.25     # Max 25% of capital per trade
    KELLY_FRACTION_MIN: float = 0.01     # Min 1% position
    HALF_KELLY: float = 0.5              # Conservative Kelly multiplier
    QUARTER_KELLY: float = 0.25          # Ultra-conservative
    
    # Risk-free rate (annualized, update periodically)
    RISK_FREE_RATE: float = 0.05         # 5% (current Fed funds approximation)
    
    # Trading days per year
    TRADING_DAYS_YEAR: int = 252
    TRADING_HOURS_DAY: int = 6.5
    
    # Bayesian prior parameters
    PRIOR_ALPHA: float = 1.0             # Beta distribution alpha
    PRIOR_BETA: float = 1.0              # Beta distribution beta (uniform prior)
    
    # Shannon entropy thresholds
    ENTROPY_HIGH_INFO: float = 0.9       # High information content
    ENTROPY_LOW_INFO: float = 0.3        # Low information content
    
    # Minimum samples for statistical significance
    MIN_SAMPLES_SIGNIFICANCE: int = 30   # Central Limit Theorem threshold
    MIN_TRADES_BACKTEST: int = 100       # Minimum trades for valid backtest


# ============================================================================
# MARKET REGIME ENUMS (Hidden Markov Model States)
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
    # Trend-based (NEW)
    BULL_TREND = auto()
    BEAR_TREND = auto()
    SIDEWAYS = auto()
    # Volatility-based
    LOW_VOLATILITY = auto()
    NORMAL_VOLATILITY = auto()
    HIGH_VOLATILITY = auto()
    EXTREME_VOLATILITY = auto()
    CRISIS = auto()

class RiskTier(Enum):
    """
    CVaR-based risk classification.
    
    Mathematical basis:
    CVaR (Expected Shortfall) = E[X | X ≤ VaR_α]
    
    Tiers based on potential portfolio drawdown.
    """
    MINIMAL = auto()      # CVaR < 1% daily
    LOW = auto()          # 1% ≤ CVaR < 2%
    MODERATE = auto()     # 2% ≤ CVaR < 5%
    HIGH = auto()         # 5% ≤ CVaR < 10%
    EXTREME = auto()      # CVaR ≥ 10%
    CATASTROPHIC = auto() # CVaR ≥ 20% (tail event)


class PositionAction(Enum):
    """
    Continuous position sizing actions.
    
    Mathematical basis:
    Kelly Criterion: f* = (p*b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio
    """
    FULL_LONG = auto()       # 100% of Kelly allocation
    THREE_QUARTER_LONG = auto()  # 75% Kelly
    HALF_LONG = auto()       # 50% Kelly (conservative)
    QUARTER_LONG = auto()    # 25% Kelly (ultra-conservative)
    NEUTRAL = auto()         # 0% - flat position
    QUARTER_SHORT = auto()   # -25% Kelly
    HALF_SHORT = auto()      # -50% Kelly
    THREE_QUARTER_SHORT = auto()  # -75% Kelly
    FULL_SHORT = auto()      # -100% Kelly


# ============================================================================
# LEGACY ENUMS (Backward Compatibility)
# ============================================================================

class QueryType(Enum):
    """Query type classification - Extended."""
    TRADE = "trade"
    TECHNICAL = "tech"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    MACRO = "macro"
    GENERAL = "general"


class QueryComplexity(Enum):
    """Query complexity level - Extended with computational metrics."""
    SIMPLE = "simple"           # O(1) - instant response
    MODERATE = "moderate"       # O(n) - single pass analysis
    COMPLEX = "complex"         # O(n²) - multi-factor analysis
    INTENSIVE = "intensive"     # O(n³) - optimization required
    HPC_REQUIRED = "hpc"        # Requires distributed compute


class AlertUrgency(Enum):
    """Alert urgency level - Risk-adjusted."""
    LOW = "low"                 # Informational
    MEDIUM = "medium"           # Action within hours
    HIGH = "high"               # Action within minutes
    CRITICAL = "critical"       # Immediate action required
    EMERGENCY = "emergency"     # Circuit breaker triggered


class TradeAction(Enum):
    """Trade action - Legacy compatibility."""
    BUY = "AL"
    SELL = "SAT"
    HOLD = "BEKLE"


# ============================================================================
# PROTOCOL INTERFACES (Type Safety)
# ============================================================================

@runtime_checkable
class Analyzable(Protocol):
    """Protocol for analyzable entities."""
    def analyze(self, ticker: str, data: Dict) -> Dict: ...


@runtime_checkable
class Scorable(Protocol):
    """Protocol for entities that produce scores."""
    def calculate_score(self) -> float: ...


@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable components."""
    def configure(self, config: Dict) -> None: ...


# ============================================================================
# DATA CLASSES (Immutable State)
# ============================================================================

@dataclass(frozen=True)
class ConfidenceInterval:
    """Immutable confidence interval."""
    lower: float
    upper: float
    confidence_level: float
    
    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper
    
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class BayesianPosterior:
    """Bayesian posterior distribution parameters."""
    alpha: float  # Successes + prior_alpha
    beta: float   # Failures + prior_beta
    
    @property
    def mean(self) -> float:
        """Posterior mean: α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """Posterior variance: αβ / ((α+β)²(α+β+1))"""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))
    
    @property
    def mode(self) -> float:
        """Posterior mode (MAP estimate)"""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return self.mean


@dataclass
class AgentDecision:
    """Structured agent decision with full metadata."""
    action: PositionAction
    confidence: float
    risk_tier: RiskTier
    kelly_fraction: float
    arguments: List[Dict] = field(default_factory=list)
    entropy_score: float = 0.0
    bayesian_posterior: Optional[BayesianPosterior] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action.name,
            "confidence": self.confidence,
            "risk_tier": self.risk_tier.name,
            "kelly_fraction": self.kelly_fraction,
            "entropy_score": self.entropy_score,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    All agents must implement:
    - analyze(): Core analysis method
    - calculate_confidence(): Bayesian confidence calculation
    - get_entropy_score(): Information-theoretic quality metric
    """
    
    def __init__(self, name: str, bias: str = "NEUTRAL", llm=None):
        self.name = name
        self.bias = bias
        self.llm = llm
        self.confidence: float = 0.0
        self.arguments: List[Dict] = []
        self._prior = BayesianPosterior(
            alpha=StatisticalConstants.PRIOR_ALPHA,
            beta=StatisticalConstants.PRIOR_BETA
        )
    
    @abstractmethod
    def analyze(self, ticker: str, price_data: Dict, 
                news: List[str] = None, technicals: Dict = None) -> Dict:
        """
        Perform analysis on the given ticker.
        
        Returns:
            Dict with keys: recommendation, confidence, arguments, etc.
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, signals: List[Dict]) -> float:
        """
        Calculate Bayesian confidence from signals.
        
        Uses Beta-Binomial conjugate prior:
        P(θ|data) ∝ P(data|θ) * P(θ)
        """
        pass
    
    def get_entropy_score(self) -> float:
        """
        Calculate Shannon entropy of arguments.
        
        H(X) = -Σ p(x) * log₂(p(x))
        
        Higher entropy = more diverse/informative arguments.
        """
        if not self.arguments:
            return 0.0
        
        # Count argument types
        arg_types = [arg.get("indicator", "unknown") for arg in self.arguments]
        unique_types = set(arg_types)
        
        if len(unique_types) <= 1:
            return 0.0
        
        # Calculate probabilities
        probs = [arg_types.count(t) / len(arg_types) for t in unique_types]
        
        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(unique_types))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def update_posterior(self, successes: int, failures: int) -> BayesianPosterior:
        """
        Update Bayesian posterior with new observations.
        
        Posterior: Beta(α + successes, β + failures)
        """
        new_alpha = self._prior.alpha + successes
        new_beta = self._prior.beta + failures
        self._prior = BayesianPosterior(alpha=new_alpha, beta=new_beta)
        return self._prior
    
    def debate_opening(self) -> str:
        """Generate debate opening statement."""
        return f"{self.name}: Analysis pending."


class BaseOrchestrator(ABC):
    """
    Abstract base class for swarm orchestrators.
    
    Implements:
    - Bayesian Model Averaging for consensus
    - Shannon entropy weighted voting
    - Async execution patterns
    """
    
    def __init__(self, name: str = "Orchestrator"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.history: List[Dict] = []
        self.total_decisions: int = 0
    
    @abstractmethod
    async def run_debate_async(self, ticker: str, price_data: Dict,
                                technicals: Dict = None, news: List[str] = None,
                                market_context: Dict = None) -> Dict:
        """
        Run async debate among agents.
        
        Returns:
            Consensus decision with Bayesian model averaging.
        """
        pass
    
    @abstractmethod
    def calculate_consensus(self, agent_decisions: List[AgentDecision]) -> AgentDecision:
        """
        Calculate Bayesian consensus from agent decisions.
        
        P(Action|D) = Σₖ P(Action|Mₖ,D) * P(Mₖ|D)
        """
        pass
    
    def calculate_ewma_confidence(self, decay: float = None) -> float:
        """
        Calculate exponentially weighted moving average confidence.
        
        C_ewm(t) = α * c_t + (1-α) * C_ewm(t-1)
        """
        if not self.history:
            return 0.0
        
        lambda_decay = decay or StatisticalConstants.EWMA_LAMBDA_DAILY
        alpha = 1 - lambda_decay
        
        ewma = self.history[0].get("confidence", 0.5)
        for record in self.history[1:]:
            conf = record.get("confidence", 0.5)
            ewma = alpha * conf + lambda_decay * ewma
        
        return ewma
    
    def get_statistics(self) -> Dict:
        """Get orchestrator statistics."""
        return {
            "total_decisions": self.total_decisions,
            "ewma_confidence": self.calculate_ewma_confidence(),
            "agent_count": len(self.agents)
        }


class BaseJudge(ABC):
    """
    Abstract base class for judge agents.
    
    Implements information-theoretic scoring and risk-adjusted decisions.
    """
    
    def __init__(self, name: str = "Judge", llm=None):
        self.name = name
        self.llm = llm
        self.risk_premium: float = 1.2  # 20% risk premium on bear arguments
    
    @abstractmethod
    def evaluate_debate(self, bull_analysis: Dict, bear_analysis: Dict,
                        market_context: Dict = None) -> Dict:
        """Evaluate debate between bull and bear agents."""
        pass
    
    def calculate_entropy_weighted_score(self, arguments: List[Dict], 
                                          base_confidence: float) -> float:
        """
        Calculate Shannon entropy weighted score.
        
        S = Σᵢ wᵢ * H(pᵢ) * sign(pᵢ - 0.5)
        
        Higher entropy arguments get more weight.
        """
        if not arguments:
            return 0.0
        
        total_score = 0.0
        
        for arg in arguments:
            # Get argument weight
            weight = arg.get("weight", 0.5)
            
            # Calculate binary entropy for this argument's confidence
            p = arg.get("confidence", 0.5)
            p = max(0.001, min(0.999, p))  # Avoid log(0)
            
            # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            
            # Direction: positive for bullish, negative for bearish
            direction = 1 if arg.get("sentiment", "").upper() == "BULLISH" else -1
            
            total_score += weight * entropy * direction
        
        return total_score * base_confidence


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """
    Calculate Kelly Criterion optimal fraction.
    
    f* = (p * b - q) / b = (p * (b + 1) - 1) / b
    
    Args:
        win_prob: Probability of winning (0-1)
        win_loss_ratio: Average win / Average loss
    
    Returns:
        Optimal fraction of capital to risk
    """
    if win_loss_ratio <= 0:
        return 0.0
    
    q = 1 - win_prob
    kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
    
    # Apply constraints
    kelly = max(StatisticalConstants.KELLY_FRACTION_MIN, kelly)
    kelly = min(StatisticalConstants.KELLY_FRACTION_MAX, kelly)
    
    return kelly


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    CVaR_α = E[X | X ≤ VaR_α]
    
    Args:
        returns: Array of returns
        confidence: Confidence level (e.g., 0.95)
    
    Returns:
        CVaR value (positive number representing potential loss)
    """
    if len(returns) == 0:
        return 0.0
    
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return abs(var_threshold)
    
    return abs(np.mean(tail_returns))


def classify_risk_tier(cvar: float) -> RiskTier:
    """Classify risk tier based on CVaR."""
    if cvar >= 0.20:
        return RiskTier.CATASTROPHIC
    elif cvar >= 0.10:
        return RiskTier.EXTREME
    elif cvar >= 0.05:
        return RiskTier.HIGH
    elif cvar >= 0.02:
        return RiskTier.MODERATE
    elif cvar >= 0.01:
        return RiskTier.LOW
    else:
        return RiskTier.MINIMAL


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = None) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    SR = (E[Rp] - Rf) / σp * √252
    """
    if len(returns) < 2:
        return 0.0
    
    rf = risk_free_rate or StatisticalConstants.RISK_FREE_RATE
    rf_daily = rf / StatisticalConstants.TRADING_DAYS_YEAR
    
    excess_returns = returns - rf_daily
    
    if np.std(returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(returns)
    
    # Annualize
    return sharpe * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)


# ============================================================================
# REGIME ADAPTIVE MULTIPLIERS (Phase 3)
# ============================================================================

REGIME_CVAR_MULTIPLIERS = {
    MarketRegime.BULL_TREND: 1.20,        # Relax 20%
    MarketRegime.BEAR_TREND: 0.70,        # Tighten 30%
    MarketRegime.SIDEWAYS: 1.00,          # Neutral
    MarketRegime.LOW_VOLATILITY: 1.30,    # Relax 30%
    MarketRegime.NORMAL_VOLATILITY: 1.00, # Neutral
    MarketRegime.HIGH_VOLATILITY: 0.80,   # Tighten 20%
    MarketRegime.EXTREME_VOLATILITY: 0.60,# Tighten 40%
    MarketRegime.CRISIS: 0.50,            # Tighten 50%
}

REGIME_KELLY_MULTIPLIERS = {
    MarketRegime.BULL_TREND: 2.00,        # Full Kelly
    MarketRegime.BEAR_TREND: 0.25,        # Quarter Kelly
    MarketRegime.SIDEWAYS: 0.50,          # Half Kelly
    MarketRegime.LOW_VOLATILITY: 1.50,    # 1.5x Kelly
    MarketRegime.NORMAL_VOLATILITY: 0.50, # Half Kelly (default)
    MarketRegime.HIGH_VOLATILITY: 0.50,   # Half Kelly
    MarketRegime.EXTREME_VOLATILITY: 0.25,# Quarter Kelly
    MarketRegime.CRISIS: 0.10,            # Tenth Kelly
}


def get_regime_adjusted_cvar(base_cvar: float, regime: MarketRegime) -> float:
    """Get CVaR threshold adjusted for market regime."""
    multiplier = REGIME_CVAR_MULTIPLIERS.get(regime, 1.0)
    return base_cvar * multiplier


def get_regime_adjusted_kelly(base_kelly: float, regime: MarketRegime) -> float:
    """Get Kelly fraction adjusted for market regime."""
    multiplier = REGIME_KELLY_MULTIPLIERS.get(regime, 0.5)
    adjusted = base_kelly * multiplier
    return min(adjusted, StatisticalConstants.KELLY_FRACTION_MAX)
