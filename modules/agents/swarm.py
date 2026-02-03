"""
Swarm Orchestrator - Institutional-Grade Multi-Agent Trading System
Author: Erdinc Erdogan
Purpose: Orchestrates bull, bear, and judge agents using Bayesian Model Averaging, Shannon Entropy
weighted voting, and Kelly Criterion position sizing for institutional trading decisions.
References:
- Bayesian Model Averaging: P(Action|D) = Σₖ P(Action|Mₖ,D) × P(Mₖ|D)
- Shannon Entropy and Information Theory
- Kelly Criterion Position Sizing: f* = (p×b - q) / b
Usage:
    swarm = SwarmOrchestrator(llm=ollama_llm)
    result = await swarm.debate(ticker="AAPL", market_data=data)
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from colorama import Fore, Style

# Core imports (institutional base classes)
from ..core.base import (
    BaseOrchestrator, BaseAgent, BaseJudge,
    StatisticalConstants, MarketRegime, RiskTier, PositionAction,
    AgentDecision, BayesianPosterior,
    calculate_kelly_fraction, calculate_cvar, classify_risk_tier
)

# Agent imports
from .bull import BullAgent
from .bear import BearAgent
from .judge import JudgeAgent


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DebateResult:
    """Structured debate result with full statistical metadata."""
    ticker: str
    consensus_action: str
    consensus_confidence: float
    kelly_fraction: float
    risk_tier: RiskTier
    market_regime: MarketRegime
    bull_analysis: Dict
    bear_analysis: Dict
    judge_verdict: Dict
    bayesian_posterior: BayesianPosterior
    entropy_scores: Dict[str, float]
    ewma_confidence: float
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "swarm_decision": self.consensus_action,
            "confidence": self.consensus_confidence,
            "kelly_fraction": self.kelly_fraction,
            "risk_tier": self.risk_tier.name,
            "market_regime": self.market_regime.name if self.market_regime else "UNKNOWN",
            "bull_analysis": self.bull_analysis,
            "bear_analysis": self.bear_analysis,
            "judge_verdict": self.judge_verdict,
            "bayesian_mean": self.bayesian_posterior.mean,
            "entropy_scores": self.entropy_scores,
            "ewma_confidence": self.ewma_confidence,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# SWARM ORCHESTRATOR (Institutional-Grade)
# ============================================================================

class SwarmOrchestrator(BaseOrchestrator):
    """
    Institutional-Grade Swarm Orchestrator.
    
    Implements:
    1. Bayesian Model Averaging: P(Action|D) = Σₖ P(Action|Mₖ,D) * P(Mₖ|D)
    2. Shannon Entropy Weighted Voting: S = Σᵢ wᵢ * H(pᵢ)
    3. EWMA Confidence Tracking: C_ewm(t) = α*c_t + (1-α)*C_ewm(t-1)
    4. Kelly Criterion Position Sizing: f* = (p*b - q) / b
    5. CVaR-based Risk Classification
    
    Architecture:
    - SwarmOrchestrator (Coordinator)
      ├── BullAgent (Bullish analysis)
      ├── BearAgent (Risk analysis)
      ├── JudgeAgent (Final verdict)
      └── SpecialistAgents (Extensible)
    """
    
    def __init__(self, llm=None, parallel: bool = True):
        """
        Initialize Swarm Orchestrator.
        
        Args:
            llm: LangChain LLM instance (distributed to all agents)
            parallel: Run agents in parallel (async)
        """
        super().__init__(name="🧬 SWARM ORCHESTRATOR")
        
        self.llm = llm
        self.parallel = parallel
        
        # Core agents
        self.bull = BullAgent(llm=llm)
        self.bear = BearAgent(llm=llm)

        self.bull = BullAgent(llm=llm)
        self.bear = BearAgent(llm=llm)
        
        # JudgeAgent initialization with fallback
        try:
            self.judge = JudgeAgent(llm=llm)
        except TypeError:
            # HardenedJudge doesn't accept llm - use default init
            self.judge = JudgeAgent()
            # Inject llm if the attribute exists
            if hasattr(self.judge, 'llm'):
                self.judge.llm = llm
            elif hasattr(self.judge, 'set_llm'):
                self.judge.set_llm(llm)
        
        # Register agents
        self.agents = {
            "bull": self.bull,
            "bear": self.bear,
            "judge": self.judge
        }
        
        # Specialist agents (extensible)
        self.specialist_agents: Dict[str, BaseAgent] = {}
        
        # Statistics tracking
        self.total_debates = 0
        self.decisions = {"AL": 0, "SAT": 0, "BEKLE": 0}
        
        # Bayesian tracking
        self._cumulative_posterior = BayesianPosterior(
            alpha=StatisticalConstants.PRIOR_ALPHA,
            beta=StatisticalConstants.PRIOR_BETA
        )
        
        # Performance tracking for Sharpe calculation
        self._returns_history: List[float] = []
    
    # ========================================================================
    # CORE DEBATE METHODS
    # ========================================================================
    
    async def run_debate_async(self, 
                                ticker: str,
                                price_data: Dict,
                                technicals: Dict = None,
                                news: List[str] = None,
                                market_context: Dict = None) -> Dict:
        """
        Run full async debate cycle with Bayesian consensus.
        
        Process:
        1. Parallel agent analysis (Bull & Bear)
        2. Shannon entropy calculation for argument quality
        3. Judge evaluation with entropy-weighted scoring
        4. Bayesian Model Averaging for consensus
        5. Kelly Criterion position sizing
        6. CVaR risk classification
        
        Returns:
            DebateResult as dictionary
        """
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}🧬 BAYESIAN SWARM DEBATE - {ticker}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}", flush=True)
        
        start_time = datetime.now()
        
        # Step 1: Parallel agent analysis
        if self.parallel:
            bull_result, bear_result = await self._run_parallel_analysis_async(
                ticker, price_data, technicals, news
            )
        else:
            bull_result = await self._run_agent_async(
                self.bull, ticker, price_data, news, technicals
            )
            bear_result = await self._run_agent_async(
                self.bear, ticker, price_data, news, technicals
            )
        
        # Step 2: Calculate Shannon entropy scores
        entropy_scores = self._calculate_entropy_scores(bull_result, bear_result)
        
        # Step 3: Display debate openings
        print(f"\n{Fore.GREEN}--- BAYESIAN DEBATE INITIATED ---{Style.RESET_ALL}", flush=True)
        print(f"\n{self.bull.debate_opening()}", flush=True)
        print(f"\n{self.bear.debate_opening()}", flush=True)
        print(f"\n📊 Entropy Scores: Bull={entropy_scores['bull']:.3f}, Bear={entropy_scores['bear']:.3f}", flush=True)
        
        # Step 4: Judge evaluation with entropy weighting
        verdict = self.judge.evaluate_debate(
            bull_result, bear_result, market_context
        )
        
        # Step 5: Bayesian consensus calculation
        consensus = self._calculate_bayesian_consensus(
            bull_result, bear_result, verdict, entropy_scores
        )
        
        # Step 6: Kelly Criterion position sizing
        kelly_fraction = self._calculate_kelly_position(consensus)
        
        # Step 7: CVaR risk classification
        risk_tier = self._classify_risk(bull_result, bear_result, market_context)
        
        # Step 8: Detect market regime
        market_regime = self._detect_market_regime(price_data, technicals)
        
        # Update statistics
        self.total_debates += 1
        self.decisions[consensus["action"]] = self.decisions.get(consensus["action"], 0) + 1
        
        # Update Bayesian posterior
        self._update_cumulative_posterior(consensus)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Build result
        result = DebateResult(
            ticker=ticker,
            consensus_action=consensus["action"],
            consensus_confidence=consensus["confidence"],
            kelly_fraction=kelly_fraction,
            risk_tier=risk_tier,
            market_regime=market_regime,
            bull_analysis=bull_result,
            bear_analysis=bear_result,
            judge_verdict=verdict,
            bayesian_posterior=self._cumulative_posterior,
            entropy_scores=entropy_scores,
            ewma_confidence=self.calculate_ewma_confidence(),
            duration_seconds=duration
        )
        
        # Store in history
        self.history.append(result.to_dict())
        
        print(f"\n{Fore.CYAN}⏱️ Debate duration: {duration:.2f}s{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}📈 Kelly Fraction: {kelly_fraction:.2%}{Style.RESET_ALL}", flush=True)
        print(f"{Fore.CYAN}⚠️ Risk Tier: {risk_tier.name}{Style.RESET_ALL}", flush=True)
        
        return result.to_dict()
    
    def run_debate(self,
                   ticker: str,
                   price_data: Dict,
                   technicals: Dict = None,
                   news: List[str] = None,
                   market_context: Dict = None) -> Dict:
        """
        Synchronous wrapper for async debate.
        Compatible with existing codebase.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Nested event loop handling
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(
                self.run_debate_async(ticker, price_data, technicals, news, market_context)
            )
        except RuntimeError:
            return asyncio.run(
                self.run_debate_async(ticker, price_data, technicals, news, market_context)
            )
    
    # ========================================================================
    # ASYNC EXECUTION
    # ========================================================================
    
    async def _run_parallel_analysis_async(self,
                                            ticker: str,
                                            price_data: Dict,
                                            technicals: Dict,
                                            news: List[str]) -> Tuple[Dict, Dict]:
        """
        Run Bull and Bear analysis in parallel using native asyncio.
        
        Replaces ThreadPoolExecutor with asyncio.gather for better performance.
        """
        # Create async tasks
        bull_task = self._run_agent_async(self.bull, ticker, price_data, news, technicals)
        bear_task = self._run_agent_async(self.bear, ticker, price_data, news, technicals)
        
        # Run in parallel
        results = await asyncio.gather(bull_task, bear_task, return_exceptions=True)
        
        # Handle exceptions
        bull_result = results[0] if not isinstance(results[0], Exception) else self._empty_analysis("bull")
        bear_result = results[1] if not isinstance(results[1], Exception) else self._empty_analysis("bear")
        
        return bull_result, bear_result
    
    async def _run_agent_async(self,
                                agent,
                                ticker: str,
                                price_data: Dict,
                                news: List[str],
                                technicals: Dict) -> Dict:
        """
        Run single agent analysis asynchronously.
        
        Wraps synchronous agent.analyze() in executor for non-blocking execution.
        """
        loop = asyncio.get_event_loop()
        
        # Run synchronous analyze in thread pool
        result = await loop.run_in_executor(
            None,  # Default executor
            lambda: agent.analyze(ticker, price_data, news, technicals)
        )
        
        return result
    
    def _empty_analysis(self, agent_type: str) -> Dict:
        """Return empty analysis for error cases."""
        return {
            "agent": agent_type,
            "recommendation": "BEKLE",
            "confidence": 0.0,
            "arguments": [],
            "error": True
        }
    
    # ========================================================================
    # BAYESIAN CONSENSUS
    # ========================================================================
    
    def calculate_consensus(self, agent_decisions: List[AgentDecision]) -> AgentDecision:
        """
        Calculate Bayesian Model Averaging consensus.
        
        P(Action|D) = Σₖ P(Action|Mₖ,D) * P(Mₖ|D)
        
        Where:
        - Mₖ = Agent k's model
        - D = Observed data
        - P(Mₖ|D) ∝ P(D|Mₖ) * P(Mₖ) (agent's posterior probability)
        """
        if not agent_decisions:
            return AgentDecision(
                action=PositionAction.NEUTRAL,
                confidence=0.0,
                risk_tier=RiskTier.MODERATE,
                kelly_fraction=0.0
            )
        
        # Calculate model weights (normalized posteriors)
        total_weight = sum(d.confidence for d in agent_decisions)
        if total_weight == 0:
            weights = [1.0 / len(agent_decisions)] * len(agent_decisions)
        else:
            weights = [d.confidence / total_weight for d in agent_decisions]
        
        # Weighted average confidence
        consensus_confidence = sum(w * d.confidence for w, d in zip(weights, agent_decisions))
        
        # Majority voting with weights
        action_scores = {}
        for w, d in zip(weights, agent_decisions):
            action = d.action
            action_scores[action] = action_scores.get(action, 0) + w
        
        consensus_action = max(action_scores, key=action_scores.get)
        
        # Aggregate risk tier (conservative - take highest risk)
        risk_tiers = [d.risk_tier.value for d in agent_decisions]
        consensus_risk = RiskTier(max(risk_tiers))
        
        # Kelly fraction (weighted average)
        consensus_kelly = sum(w * d.kelly_fraction for w, d in zip(weights, agent_decisions))
        
        return AgentDecision(
            action=consensus_action,
            confidence=consensus_confidence,
            risk_tier=consensus_risk,
            kelly_fraction=consensus_kelly
        )
    
    def _calculate_bayesian_consensus(self,
                                       bull_result: Dict,
                                       bear_result: Dict,
                                       verdict: Dict,
                                       entropy_scores: Dict) -> Dict:
        """
        Calculate Bayesian consensus with entropy weighting.
        
        Combines:
        1. Agent confidences (Bull, Bear)
        2. Judge verdict
        3. Shannon entropy weights
        """
        # Extract confidences
        bull_conf = bull_result.get("confidence", 0.5)
        bear_conf = bear_result.get("confidence", 0.5)
        judge_conf = verdict.get("confidence", 0.5)
        
        # Entropy-weighted confidences
        bull_entropy = entropy_scores.get("bull", 0.5)
        bear_entropy = entropy_scores.get("bear", 0.5)
        
        # Higher entropy = more informative = higher weight
        bull_weight = bull_conf * (1 + bull_entropy)
        bear_weight = bear_conf * (1 + bear_entropy) * 1.2  # Risk premium
        judge_weight = judge_conf * 1.5  # Judge has higher authority
        
        total_weight = bull_weight + bear_weight + judge_weight
        
        # Normalize
        bull_ratio = bull_weight / total_weight if total_weight > 0 else 0.33
        bear_ratio = bear_weight / total_weight if total_weight > 0 else 0.33
        
        # Determine action
        final_action = verdict.get("final_decision", "BEKLE")
        
        # Calculate consensus confidence using Bayesian averaging
        consensus_confidence = (
            bull_ratio * bull_conf +
            bear_ratio * (1 - bear_conf) +  # Invert bear for bullish consensus
            (1 - bull_ratio - bear_ratio) * judge_conf
        )
        
        # Clamp to [0, 1]
        consensus_confidence = max(0.0, min(1.0, consensus_confidence))
        
        return {
            "action": final_action,
            "confidence": consensus_confidence,
            "bull_weight": bull_ratio,
            "bear_weight": bear_ratio,
            "judge_weight": 1 - bull_ratio - bear_ratio
        }
    
    # ========================================================================
    # SHANNON ENTROPY
    # ========================================================================
    
    def _calculate_entropy_scores(self,
                                   bull_result: Dict,
                                   bear_result: Dict) -> Dict[str, float]:
        """
        Calculate Shannon entropy for each agent's arguments.
        
        H(X) = -Σ p(x) * log₂(p(x))
        
        Higher entropy = more diverse arguments = more informative.
        """
        def calculate_argument_entropy(arguments: List[Dict]) -> float:
            if not arguments:
                return 0.0
            
            # Count argument types
            arg_types = [arg.get("indicator", "unknown") for arg in arguments]
            unique_types = set(arg_types)
            
            if len(unique_types) <= 1:
                return 0.0
            
            # Calculate probabilities
            n = len(arg_types)
            probs = [arg_types.count(t) / n for t in unique_types]
            
            # Shannon entropy
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            
            # Normalize to [0, 1]
            max_entropy = np.log2(len(unique_types))
            return entropy / max_entropy if max_entropy > 0 else 0.0
        
        bull_args = bull_result.get("arguments", [])
        bear_args = bear_result.get("arguments", [])
        
        return {
            "bull": calculate_argument_entropy(bull_args),
            "bear": calculate_argument_entropy(bear_args),
            "combined": (calculate_argument_entropy(bull_args) + 
                        calculate_argument_entropy(bear_args)) / 2
        }
    
    # ========================================================================
    # KELLY CRITERION
    # ========================================================================
    
    def _calculate_kelly_position(self, consensus: Dict) -> float:
        """
        Calculate Kelly Criterion optimal position size.
        
        f* = (p * b - q) / b
        
        Where:
        - p = win probability (consensus confidence)
        - q = 1 - p
        - b = win/loss ratio (estimated from history)
        """
        win_prob = consensus.get("confidence", 0.5)
        
        # Estimate win/loss ratio from history
        if len(self.history) >= StatisticalConstants.MIN_SAMPLES_SIGNIFICANCE:
            wins = sum(1 for h in self.history if h.get("swarm_decision") == "AL" 
                      and h.get("confidence", 0) > 0.6)
            losses = len(self.history) - wins
            win_loss_ratio = (wins + 1) / (losses + 1)  # Laplace smoothing
        else:
            win_loss_ratio = 1.5  # Default assumption
        
        kelly = calculate_kelly_fraction(win_prob, win_loss_ratio)
        
        # Apply Half-Kelly for safety
        return kelly * StatisticalConstants.HALF_KELLY
    
    # ========================================================================
    # RISK CLASSIFICATION
    # ========================================================================
    
    def _classify_risk(self,
                       bull_result: Dict,
                       bear_result: Dict,
                       market_context: Dict = None) -> RiskTier:
        """
        Classify risk tier using CVaR methodology.
        
        Considers:
        1. Bear agent's risk assessment
        2. Market context (VIX, etc.)
        3. Historical volatility
        """
        # Base risk from bear analysis
        bear_conf = bear_result.get("confidence", 0.5)
        
        # Market context adjustments
        vix = 20  # Default
        if market_context:
            vix = market_context.get("vix", 20)
        
        # Estimate CVaR proxy
        # Higher bear confidence + higher VIX = higher risk
        cvar_proxy = bear_conf * (vix / 20) * 0.05  # Scale to percentage
        
        return classify_risk_tier(cvar_proxy)
    
    # ========================================================================
    # MARKET REGIME DETECTION
    # ========================================================================
    
    def _detect_market_regime(self,
                               price_data: Dict,
                               technicals: Dict = None) -> MarketRegime:
        """
        Detect current market regime.
        
        Uses:
        1. Volatility level
        2. Trend direction
        3. Technical indicators
        """
        tech = technicals or {}
        
        # Volatility-based regime
        volatility = tech.get("volatility", 0.02)
        
        if volatility >= 0.40:
            return MarketRegime.EXTREME_VOLATILITY
        elif volatility >= 0.25:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.15:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend-based regime
        trend = tech.get("trend", "").lower()
        rsi = tech.get("rsi", 50)
        
        if "bull" in trend or rsi < 30:
            return MarketRegime.WEAK_UPTREND
        elif "bear" in trend or rsi > 70:
            return MarketRegime.WEAK_DOWNTREND
        
        return MarketRegime.SIDEWAYS
    
    # ========================================================================
    # POSTERIOR UPDATES
    # ========================================================================
    
    def _update_cumulative_posterior(self, consensus: Dict):
        """
        Update cumulative Bayesian posterior.
        
        Posterior: Beta(α + successes, β + failures)
        """
        action = consensus.get("action", "BEKLE")
        confidence = consensus.get("confidence", 0.5)
        
        # Treat high-confidence decisions as "successes"
        if confidence > 0.6:
            new_alpha = self._cumulative_posterior.alpha + 1
            new_beta = self._cumulative_posterior.beta
        else:
            new_alpha = self._cumulative_posterior.alpha
            new_beta = self._cumulative_posterior.beta + 1
        
        self._cumulative_posterior = BayesianPosterior(
            alpha=new_alpha,
            beta=new_beta
        )
    
    # ========================================================================
    # SPECIALIST AGENTS
    # ========================================================================
    
    def add_specialist_agent(self, name: str, agent: BaseAgent):
        """
        Add specialist agent to the swarm.
        
        Examples: FED Watcher, Technical Chart Analyst, Sentiment Analyzer
        """
        self.specialist_agents[name] = agent
        self.agents[name] = agent
        print(f"{Fore.CYAN}➕ Specialist agent added: {name}{Style.RESET_ALL}", flush=True)
    
    def remove_specialist_agent(self, name: str):
        """Remove specialist agent from swarm."""
        if name in self.specialist_agents:
            del self.specialist_agents[name]
            del self.agents[name]
            print(f"{Fore.YELLOW}➖ Specialist agent removed: {name}{Style.RESET_ALL}", flush=True)
    
    # ========================================================================
    # STATISTICS & REPORTING
    # ========================================================================
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive swarm statistics.
        
        Includes:
        - Decision counts
        - Win rate (risk-adjusted)
        - EWMA confidence
        - Bayesian posterior
        - Sharpe ratio estimate
        """
        return {
            "total_debates": self.total_debates,
            "decisions": self.decisions,
            "risk_adjusted_win_rate": self._calculate_risk_adjusted_win_rate(),
            "ewma_confidence": self.calculate_ewma_confidence(),
            "bayesian_posterior_mean": self._cumulative_posterior.mean,
            "bayesian_posterior_variance": self._cumulative_posterior.variance,
            "agent_count": len(self.agents),
            "specialist_count": len(self.specialist_agents)
        }
    
    def _calculate_risk_adjusted_win_rate(self) -> float:
        """
        Calculate risk-adjusted win rate.
        
        Risk-Adjusted Win Rate = Σ(Rᵢ * wᵢ) / Σ(|Rᵢ| * wᵢ) * (σ_benchmark / σ_strategy)
        """
        if not self.history:
            return 0.0
        
        # Simple win rate for now (can be enhanced with actual returns)
        total = sum(self.decisions.values())
        if total == 0:
            return 0.0
        
        wins = self.decisions.get("AL", 0)
        
        # Apply confidence weighting
        weighted_wins = sum(
            h.get("confidence", 0.5) 
            for h in self.history 
            if h.get("swarm_decision") == "AL"
        )
        weighted_total = sum(h.get("confidence", 0.5) for h in self.history)
        
        if weighted_total == 0:
            return 0.0
        
        return (weighted_wins / weighted_total) * 100
    
    def generate_swarm_report(self, result: Dict) -> str:
        """Generate comprehensive swarm intelligence report."""
        report = f"""
<swarm_intelligence_v2>
🧬 BAYESIAN SWARM INTELLIGENCE REPORT - {result.get('ticker', 'N/A')}
══════════════════════════════════════════════════════════════════════

📊 CONSENSUS DECISION: {result.get('swarm_decision', 'N/A')}
📈 Confidence: {result.get('confidence', 0)*100:.1f}%
💰 Kelly Fraction: {result.get('kelly_fraction', 0)*100:.2f}%
⚠️ Risk Tier: {result.get('risk_tier', 'N/A')}
🌊 Market Regime: {result.get('market_regime', 'N/A')}

🔬 BAYESIAN ANALYSIS:
  • Posterior Mean: {result.get('bayesian_mean', 0):.4f}
  • EWMA Confidence: {result.get('ewma_confidence', 0):.4f}

📊 ENTROPY SCORES:
  • Bull Entropy: {result.get('entropy_scores', {}).get('bull', 0):.3f}
  • Bear Entropy: {result.get('entropy_scores', {}).get('bear', 0):.3f}

🐂 BULL ANALYSIS:
  • Recommendation: {result.get('bull_analysis', {}).get('recommendation', 'N/A')}
  • Confidence: {result.get('bull_analysis', {}).get('confidence', 0)*100:.0f}%
  • Arguments: {len(result.get('bull_analysis', {}).get('arguments', []))}

🐻 BEAR ANALYSIS:
  • Risk Level: {result.get('bear_analysis', {}).get('risk_level', 'N/A')}
  • Confidence: {result.get('bear_analysis', {}).get('confidence', 0)*100:.0f}%
  • Arguments: {len(result.get('bear_analysis', {}).get('arguments', []))}

⚖️ JUDGE VERDICT:
  • Bull Score: {result.get('judge_verdict', {}).get('bull_score', 0):.2f}
  • Bear Score: {result.get('judge_verdict', {}).get('bear_score', 0):.2f}
  • Reasoning: {result.get('judge_verdict', {}).get('reasoning', 'N/A')}

⏱️ Duration: {result.get('duration_seconds', 0):.2f}s
🕐 Timestamp: {result.get('timestamp', 'N/A')}

</swarm_intelligence_v2>
"""
        return report


# ============================================================================
# SPECIALIST AGENTS (Institutional Framework)
# ============================================================================

class FEDWatcherAgent(BaseAgent):
    """
    FED Watcher Specialist Agent.
    
    Monitors Federal Reserve policy and its market implications.
    """
    
    def __init__(self, llm=None):
        super().__init__(name="🏛️ FED WATCHER", bias="NEUTRAL", llm=llm)
    
    def analyze(self, ticker: str, price_data: Dict,
                news: List[str] = None, technicals: Dict = None) -> Dict:
        """Analyze FED policy impact."""
        self.arguments = []
        
        # Placeholder for FED analysis
        fed_stance = self._analyze_fed_stance()
        
        self.confidence = 0.6  # Moderate confidence
        
        return {
            "agent": self.name,
            "ticker": ticker,
            "fed_stance": fed_stance["stance"],
            "rate_outlook": fed_stance["rate_outlook"],
            "confidence": self.confidence,
            "arguments": self.arguments,
            "recommendation": self._fed_to_recommendation(fed_stance)
        }
    
    def calculate_confidence(self, signals: List[Dict]) -> float:
        """Calculate Bayesian confidence from FED signals."""
        if not signals:
            return 0.5
        
        # Count dovish vs hawkish signals
        dovish = sum(1 for s in signals if s.get("type") == "dovish")
        hawkish = sum(1 for s in signals if s.get("type") == "hawkish")
        
        # Bayesian update
        posterior = BayesianPosterior(
            alpha=self._prior.alpha + dovish,
            beta=self._prior.beta + hawkish
        )
        
        return posterior.mean
    
    def _analyze_fed_stance(self) -> Dict:
        """Analyze current FED stance."""
        return {
            "stance": "NEUTRAL",
            "rate_outlook": "HOLD",
            "reason": "Awaiting FOMC minutes"
        }
    
    def _fed_to_recommendation(self, fed_stance: Dict) -> str:
        """Convert FED stance to trading recommendation."""
        stance = fed_stance.get("stance", "NEUTRAL")
        if stance == "DOVISH":
            return "AL"
        elif stance == "HAWKISH":
            return "SAT"
        return "BEKLE"


class TechnicalChartAgent(BaseAgent):
    """
    Technical Chart Analyst Specialist Agent.
    
    Performs advanced chart pattern recognition and technical analysis.
    """
    
    def __init__(self, llm=None):
        super().__init__(name="📊 CHART ANALYST", bias="NEUTRAL", llm=llm)
    
    def analyze(self, ticker: str, price_data: Dict,
                news: List[str] = None, technicals: Dict = None) -> Dict:
        """Analyze chart patterns."""
        self.arguments = []
        
        # Chart pattern analysis
        pattern = self._detect_pattern(price_data, technicals)
        
        self.confidence = pattern.get("confidence", 0.5)
        
        return {
            "agent": self.name,
            "ticker": ticker,
            "pattern": pattern["name"],
            "breakout_level": pattern.get("breakout_level"),
            "target": pattern.get("target"),
            "confidence": self.confidence,
            "arguments": self.arguments,
            "recommendation": pattern.get("recommendation", "BEKLE")
        }
    
    def calculate_confidence(self, signals: List[Dict]) -> float:
        """Calculate confidence from technical signals."""
        if not signals:
            return 0.5
        
        # Weight signals by reliability
        total_weight = 0
        weighted_conf = 0
        
        for signal in signals:
            weight = signal.get("reliability", 0.5)
            conf = signal.get("confidence", 0.5)
            weighted_conf += weight * conf
            total_weight += weight
        
        return weighted_conf / total_weight if total_weight > 0 else 0.5
    
    def _detect_pattern(self, price_data: Dict, technicals: Dict) -> Dict:
        """Detect chart patterns."""
        # Placeholder for pattern detection
        return {
            "name": "Ascending Triangle",
            "breakout_level": price_data.get("close", 0) * 1.02,
            "target": price_data.get("close", 0) * 1.10,
            "confidence": 0.65,
            "recommendation": "AL"
        }
