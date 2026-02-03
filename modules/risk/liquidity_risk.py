"""
Liquidity Risk Engine
Author: Erdinc Erdogan
Purpose: Models bid-ask spreads, market impact (Kyle Lambda, Almgren-Chriss), Liquidity-Adjusted VaR, and Basel III LCR/NSFR ratios.
References:
- Kyle's Lambda (Linear Market Impact)
- Almgren-Chriss (Square Root Impact)
- Amihud Illiquidity Ratio
- Basel III LCR/NSFR
Usage:
    engine = LiquidityRiskEngine()
    impact = engine.calculate_market_impact(order_size, adv, model=MarketImpactModel.ALMGREN_CHRISS)
    lvar = engine.liquidity_adjusted_var(var, bid_ask_cost)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize, brentq
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class MarketImpactModel(Enum):
    """Market impact model types"""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    POWER_LAW = "power_law"
    KYLE_LAMBDA = "kyle_lambda"
    ALMGREN_CHRISS = "almgren_chriss"


class LiquidityTier(Enum):
    """Asset liquidity classification"""
    HIGHLY_LIQUID = "highly_liquid"
    LIQUID = "liquid"
    LESS_LIQUID = "less_liquid"
    ILLIQUID = "illiquid"


class HQLALevel(Enum):
    """High Quality Liquid Assets classification (Basel III)"""
    LEVEL_1 = "level_1"
    LEVEL_2A = "level_2a"
    LEVEL_2B = "level_2b"
    NON_HQLA = "non_hqla"


@dataclass
class BidAskMetrics:
    """Bid-ask spread analysis results"""
    bid_price: float
    ask_price: float
    mid_price: float
    absolute_spread: float
    relative_spread: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    half_spread_cost: float


@dataclass
class MarketImpactResult:
    """Market impact calculation results"""
    permanent_impact: float
    temporary_impact: float
    total_impact: float
    impact_cost: float
    impact_bps: float
    model: str
    execution_time: float
    participation_rate: float


@dataclass
class LiquidityAdjustedVaR:
    """Liquidity-adjusted VaR results"""
    base_var: float
    spread_cost: float
    market_impact: float
    liquidity_var: float
    liquidity_horizon: int
    confidence_level: float
    scaling_factor: float


@dataclass
class OptimalExecutionResult:
    """Optimal execution strategy results"""
    optimal_horizon: float
    optimal_trajectory: np.ndarray
    expected_cost: float
    cost_variance: float
    participation_rate: float
    num_slices: int
    urgency_parameter: float


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics"""
    amihud_illiquidity: float
    roll_spread: float
    turnover_ratio: float
    volume_volatility: float
    bid_ask_spread: float
    market_depth: float
    resilience: float
    liquidity_score: float
    liquidity_tier: LiquidityTier


@dataclass
class LCRResult:
    """Liquidity Coverage Ratio results"""
    hqla_total: float
    level_1_assets: float
    level_2a_assets: float
    level_2b_assets: float
    net_cash_outflows: float
    lcr_ratio: float
    compliant: bool
    buffer: float


@dataclass
class StressTestResult:
    """Liquidity stress test results"""
    scenario_name: str
    stressed_spread: float
    stressed_volume: float
    stressed_impact: float
    liquidation_cost: float
    time_to_liquidate: float
    survival_period: int


class LiquidityRiskEngine(BaseModule):
    """
    Institutional-Grade Liquidity Risk Engine.
    Implements comprehensive liquidity risk measurement and management
    for trading and portfolio management.
    
    Mathematical Framework:
    ----------------------
    Bid-Ask Spread Cost: LC = 0.5 × (Ask - Bid) × Position
    
    Market Impact (Kyle's Lambda):
        ΔP = λ × σ × √(V / ADV)
        where λ is Kyle's lambda, σ is volatility, V is trade size
    
    Market Impact (Almgren-Chriss):
        g(v) = γ × σ × (v / V_daily)^δ
        where δ ≈ 0.5 (square root law)
    
    Liquidity-Adjusted VaR:
        LVaR = VaR + 0.5 × Spread × Position + Market_Impact
    
    Optimal Execution (Almgren-Chriss):
        T* = √(X × σ² / (2 × η × λ))
        where X is position, η is temporary impact, λ is permanent impact
    
    Amihud Illiquidity:
        ILLIQ = (1/D) × Σ |r_t| / Volume_t
    
    Roll's Implied Spread:
        Spread = 2 × √(-Cov(Δp_t, Δp_{t-1}))
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.default_participation_rate: float = self.config.get('participation_rate', 0.10)
        self.kyle_lambda: float = self.config.get('kyle_lambda', 0.1)
        self.temporary_impact_coef: float = self.config.get('temporary_impact', 0.1)
        self.permanent_impact_coef: float = self.config.get('permanent_impact', 0.05)
        self.power_law_exponent: float = self.config.get('power_law_exponent', 0.5)
    
    # =========================================================================
    # BID-ASK SPREAD ANALYSIS
    # =========================================================================
    
    def analyze_bid_ask_spread(
        self,
        bid: float,
        ask: float,
        trade_price: Optional[float] = None,
        trade_direction: Optional[int] = None
    ) -> BidAskMetrics:
        """
        Comprehensive bid-ask spread analysis.
        
        Args:
            bid: Best bid price
            ask: Best ask price
            trade_price: Actual execution price (optional)
            trade_direction: 1 for buy, -1 for sell (optional)
        """
        mid_price = (bid + ask) / 2
        absolute_spread = ask - bid
        relative_spread = absolute_spread / mid_price
        if trade_price is not None and trade_direction is not None:
            effective_spread = 2 * abs(trade_price - mid_price)
            if trade_direction > 0:
                price_impact = trade_price - mid_price
            else:
                price_impact = mid_price - trade_price
            realized_spread = effective_spread - 2 * price_impact
        else:
            effective_spread = absolute_spread
            price_impact = 0.0
            realized_spread = absolute_spread
        
        half_spread_cost = 0.5 * absolute_spread
        
        return BidAskMetrics(
            bid_price=bid,
            ask_price=ask,
            mid_price=mid_price,
            absolute_spread=absolute_spread,
            relative_spread=relative_spread,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            price_impact=price_impact,
            half_spread_cost=half_spread_cost
        )
    
    def compute_spread_cost(
        self,
        position_value: float,
        bid_ask_spread: float
    ) -> float:
        """
        Compute transaction cost from bid-ask spread.
        
        Cost = 0.5 × Spread × Position_Value
        """
        return 0.5 * bid_ask_spread * position_value
    
    # =========================================================================
    # MARKET IMPACT MODELS
    # =========================================================================
    
    def compute_market_impact_linear(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        price: float,
        lambda_coef: Optional[float] = None
    ) -> MarketImpactResult:
        """
        Linear market impact model (Kyle's Lambda).
        
        ΔP/P = λ × σ × (V / ADV)
        """
        lam = lambda_coef or self.kyle_lambda
        participation = trade_size / daily_volume
        
        impact_pct = lam * volatility * participation
        impact_cost = impact_pct * trade_size * price
        
        permanent = 0.5 * impact_pct
        temporary = 0.5 * impact_pct
        
        return MarketImpactResult(
            permanent_impact=permanent,
            temporary_impact=temporary,
            total_impact=impact_pct,
            impact_cost=impact_cost,
            impact_bps=impact_pct * 10000,
            model=MarketImpactModel.LINEAR.value,
            execution_time=1.0,
            participation_rate=participation
        )
    
    def compute_market_impact_square_root(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        price: float,
        gamma_coef: Optional[float] = None,
        execution_days: float = 1.0
    ) -> MarketImpactResult:
        """
        Square root market impact model (Almgren-Chriss).
        
        ΔP/P = γ × σ × √(V / (ADV × T))
        """
        gamma = gamma_coef or self.temporary_impact_coef
        participation = trade_size / (daily_volume * execution_days)
        
        impact_pct = gamma * volatility * np.sqrt(participation)
        impact_cost = impact_pct * trade_size * price
        
        permanent = self.permanent_impact_coef * volatility * (trade_size / daily_volume)
        temporary = impact_pct - permanent
        
        return MarketImpactResult(
            permanent_impact=permanent,
            temporary_impact=max(temporary, 0),
            total_impact=impact_pct,
            impact_cost=impact_cost,
            impact_bps=impact_pct * 10000,
            model=MarketImpactModel.SQUARE_ROOT.value,
            execution_time=execution_days,
            participation_rate=participation
        )
    
    def compute_market_impact_power_law(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        price: float,
        exponent: Optional[float] = None
    ) -> MarketImpactResult:
        """
        Power law market impact model.
        
        ΔP/P = α × σ × (V / ADV)^δ
        """
        delta = exponent or self.power_law_exponent
        participation = trade_size / daily_volume
        
        impact_pct = volatility * (participation ** delta)
        impact_cost = impact_pct * trade_size * price
        
        permanent = 0.4 * impact_pct
        temporary = 0.6 * impact_pct
        
        return MarketImpactResult(
            permanent_impact=permanent,
            temporary_impact=temporary,
            total_impact=impact_pct,
            impact_cost=impact_cost,
            impact_bps=impact_pct * 10000,
            model=MarketImpactModel.POWER_LAW.value,
            execution_time=1.0,
            participation_rate=participation
        )
    
    def compute_almgren_chriss_impact(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        price: float,
        execution_time: float,
        eta: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> MarketImpactResult:
        """
        Full Almgren-Chriss market impact model.
        
        Total Cost = Permanent Impact + Temporary Impact + Timing Risk
        Permanent: g(v) = γ × v
        Temporary: h(v) = η × sign(v) × |v|^δ
        """
        eta = eta or self.temporary_impact_coef
        gamma = gamma or self.permanent_impact_coef
        
        n_periods = max(int(execution_time), 1)
        trade_rate = trade_size / n_periods
        
        permanent_impact = gamma * trade_size / daily_volume * volatility
        temporary_impact = eta * volatility * np.sqrt(trade_rate / daily_volume) * n_periods
        
        total_impact = permanent_impact + temporary_impact
        impact_cost = total_impact * price * trade_size
        
        return MarketImpactResult(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            total_impact=total_impact,
            impact_cost=impact_cost,
            impact_bps=total_impact * 10000,
            model=MarketImpactModel.ALMGREN_CHRISS.value,
            execution_time=execution_time,
            participation_rate=trade_size / (daily_volume * execution_time)
        )
    
    # =========================================================================
    # LIQUIDITY-ADJUSTED VAR
    # =========================================================================
    
    def compute_liquidity_adjusted_var(
        self,
        returns: np.ndarray,
        position_value: float,
        bid_ask_spread: float,
        daily_volume: float,
        volatility: float,
        confidence_level: float = 0.99,
        liquidity_horizon: int = 10,
        impact_model: MarketImpactModel = MarketImpactModel.SQUARE_ROOT
    ) -> LiquidityAdjustedVaR:
        """
        Compute Liquidity-Adjusted VaR (LVaR).
        
        LVaR = VaR × √T + Spread_Cost + Market_Impact
        
        Where T is the liquidity horizon (time to liquidate).
        """
        returns = np.asarray(returns).flatten()
        
        alpha = 1 - confidence_level
        base_var = -np.percentile(returns, alpha * 100) * position_value
        
        scaling_factor = np.sqrt(liquidity_horizon)
        scaled_var = base_var * scaling_factor
        
        spread_cost = self.compute_spread_cost(position_value, bid_ask_spread)
        
        if impact_model == MarketImpactModel.SQUARE_ROOT:
            impact_result = self.compute_market_impact_square_root(
                position_value, daily_volume * position_value, volatility, 1.0,
                execution_days=liquidity_horizon
            )
        else:
            impact_result = self.compute_market_impact_linear(
                position_value, daily_volume * position_value, volatility, 1.0
            )
        
        market_impact = impact_result.impact_cost
        
        liquidity_var = scaled_var + spread_cost + market_impact
        
        return LiquidityAdjustedVaR(
            base_var=base_var,
            spread_cost=spread_cost,
            market_impact=market_impact,
            liquidity_var=liquidity_var,
            liquidity_horizon=liquidity_horizon,
            confidence_level=confidence_level,
            scaling_factor=scaling_factor
        )
    
    # =========================================================================
    # OPTIMAL EXECUTION
    # =========================================================================
    
    def compute_optimal_execution(
        self,
        position_size: float,
        daily_volume: float,
        volatility: float,
        price: float,
        risk_aversion: float = 1e-6,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
        max_participation: float = 0.25
    ) -> OptimalExecutionResult:
        """
        Compute optimal execution strategy using Almgren-Chriss framework.
        
        Optimal Horizon: T* = √(X × σ² / (2 × η × λ))
        Minimizes: E[Cost] + λ × Var[Cost]
        """
        eta = eta or self.temporary_impact_coef
        gamma = gamma or self.permanent_impact_coef
        
        position_value = position_size * price
        
        if risk_aversion > 0 and eta > 0:
            optimal_T = np.sqrt(position_size * volatility**2 / (2 * eta * risk_aversion))
            optimal_T = max(1, min(optimal_T, position_size / (daily_volume * max_participation)))
        else:
            optimal_T = position_size / (daily_volume * max_participation)
        
        n_slices = max(int(np.ceil(optimal_T)), 1)
        
        kappa = np.sqrt(risk_aversion * volatility**2 / eta) if eta > 0 else 0.1
        
        trajectory = np.zeros(n_slices + 1)
        trajectory[0] = position_size
        
        for j in range(1, n_slices + 1):
            if kappa * optimal_T > 0:
                trajectory[j] = position_size * np.sinh(kappa * (optimal_T - j)) / np.sinh(kappa * optimal_T)
            else:
                trajectory[j] = position_size * (1 - j / n_slices)
        
        trade_sizes = -np.diff(trajectory)
        
        permanent_cost = gamma * volatility * position_size**2 / (2 * daily_volume)
        temporary_cost = eta * volatility * np.sum(trade_sizes**2) / np.sqrt(daily_volume * optimal_T)
        expected_cost = (permanent_cost + temporary_cost) * price
        
        cost_variance = volatility**2 * position_value**2 * optimal_T / n_slices
        
        participation_rate = position_size / (daily_volume * optimal_T)
        
        return OptimalExecutionResult(
            optimal_horizon=optimal_T,
            optimal_trajectory=trajectory,
            expected_cost=expected_cost,
            cost_variance=cost_variance,
            participation_rate=participation_rate,
            num_slices=n_slices,
            urgency_parameter=risk_aversion
        )
    
    def compute_twap_schedule(
        self,
        position_size: float,
        execution_periods: int
    ) -> np.ndarray:
        """Time-Weighted Average Price (TWAP) execution schedule."""
        trade_size = position_size / execution_periods
        return np.full(execution_periods, trade_size)
    
    def compute_vwap_schedule(
        self,
        position_size: float,
        volume_profile: np.ndarray
    ) -> np.ndarray:
        """Volume-Weighted Average Price (VWAP) execution schedule."""
        volume_weights = volume_profile / np.sum(volume_profile)
        return position_size * volume_weights
    
    # =========================================================================
    # LIQUIDITY METRICS
    # =========================================================================
    
    def compute_amihud_illiquidity(
        self,
        returns: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """
        Amihud Illiquidity Ratio.
        ILLIQ = (1/D) × Σ |r_t| / Volume_t
        
        Higher values indicate less liquid assets.
        """
        returns = np.asarray(returns).flatten()
        volumes = np.asarray(volumes).flatten()
        
        volumes = np.maximum(volumes, 1e-10)
        
        daily_illiq = np.abs(returns) / volumes
        amihud = np.mean(daily_illiq)
        
        return float(amihud)
    
    def compute_roll_spread(
        self,
        prices: np.ndarray
    ) -> float:
        """
        Roll's Implied Spread Estimator.
        
        Spread = 2 × √(-Cov(Δp_t, Δp_{t-1}))
        
        Based on serial covariance of price changes.
        """
        prices = np.asarray(prices).flatten()
        price_changes = np.diff(prices)
        
        if len(price_changes) < 2:
            return 0.0
        
        autocovariance = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
        
        if autocovariance < 0:
            roll_spread = 2 * np.sqrt(-autocovariance)
        else:
            roll_spread = 0.0
        
        return float(roll_spread)
    
    def compute_turnover_ratio(
        self,
        volumes: np.ndarray,
        shares_outstanding: float
    ) -> float:
        """
        Turnover Ratio - Average daily volume / shares outstanding.
        """
        avg_volume = np.mean(volumes)
        return float(avg_volume / shares_outstanding)
    
    def compute_comprehensive_liquidity_metrics(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bid_ask_spreads: np.ndarray,
        shares_outstanding: float
    ) -> LiquidityMetrics:
        """
        Compute comprehensive liquidity metrics for an asset.
        """
        prices = np.asarray(prices).flatten()
        volumes = np.asarray(volumes).flatten()
        bid_ask_spreads = np.asarray(bid_ask_spreads).flatten()
        
        returns = np.diff(prices) / prices[:-1]
        
        amihud = self.compute_amihud_illiquidity(returns, volumes[1:])
        roll = self.compute_roll_spread(prices)
        turnover = self.compute_turnover_ratio(volumes, shares_outstanding)
        volume_vol = np.std(volumes) / np.mean(volumes)
        avg_spread = np.mean(bid_ask_spreads)
        market_depth = np.mean(volumes) * np.mean(prices)
        
        if len(bid_ask_spreads) > 10:
            spread_shocks = bid_ask_spreads > np.percentile(bid_ask_spreads, 90)
            if np.any(spread_shocks):
                shock_indices = np.where(spread_shocks)[0]
                recovery_times = []
                normal_spread = np.median(bid_ask_spreads)
                for idx in shock_indices:
                    for j in range(idx + 1, min(idx + 10, len(bid_ask_spreads))):
                        if bid_ask_spreads[j] <= normal_spread * 1.1:
                            recovery_times.append(j - idx)
                            break
                resilience = 1 / np.mean(recovery_times) if recovery_times else 0.5
            else:
                resilience = 1.0
        else:
            resilience = 0.5
        
        amihud_score = max(0, 1 - amihud * 1e6)
        spread_score = max(0, 1 - avg_spread * 100)
        turnover_score = min(1, turnover * 10)
        volume_score = min(1, market_depth / 1e8)
        
        liquidity_score = (
            0.25 * amihud_score +
            0.25 * spread_score +
            0.25 * turnover_score +
            0.25 * volume_score
        )
        
        if liquidity_score >= 0.8:
            tier = LiquidityTier.HIGHLY_LIQUID
        elif liquidity_score >= 0.6:
            tier = LiquidityTier.LIQUID
        elif liquidity_score >= 0.4:
            tier = LiquidityTier.LESS_LIQUID
        else:
            tier = LiquidityTier.ILLIQUID
        
        return LiquidityMetrics(
            amihud_illiquidity=amihud,
            roll_spread=roll,
            turnover_ratio=turnover,
            volume_volatility=volume_vol,
            bid_ask_spread=avg_spread,
            market_depth=market_depth,
            resilience=resilience,
            liquidity_score=liquidity_score,
            liquidity_tier=tier
        )
    
    # =========================================================================
    # REGULATORY LIQUIDITY RATIOS
    # =========================================================================
    
    def compute_lcr(
        self,
        level_1_assets: float,
        level_2a_assets: float,
        level_2b_assets: float,
        cash_outflows: float,
        cash_inflows: float
    ) -> LCRResult:
        """
        Compute Liquidity Coverage Ratio (LCR) per Basel III.
        
        LCR = HQLA / Net Cash Outflows ≥ 100%
        HQLA = Level 1 + 0.85 × Level 2A + 0.50 × Level 2B
        Level 2 assets capped at 40% of HQLA
        Level 2B assets capped at 15% of HQLA
        """
        level_2a_adjusted = 0.85 * level_2a_assets
        level_2b_adjusted = 0.50 * level_2b_assets
        
        hqla_uncapped = level_1_assets + level_2a_adjusted + level_2b_adjusted
        
        max_level_2 = level_1_assets * (40 / 60)
        max_level_2b = hqla_uncapped * 0.15
        
        level_2_total = min(level_2a_adjusted + level_2b_adjusted, max_level_2)
        level_2b_final = min(level_2b_adjusted, max_level_2b)
        level_2a_final = level_2_total - level_2b_final
        
        hqla_total = level_1_assets + level_2a_final + level_2b_final
        
        net_outflows = max(cash_outflows - min(cash_inflows, 0.75 * cash_outflows), 0.01)
        
        lcr_ratio = hqla_total / net_outflows
        
        compliant = lcr_ratio >= 1.0
        buffer = hqla_total - net_outflows
        
        return LCRResult(
            hqla_total=hqla_total,
            level_1_assets=level_1_assets,
            level_2a_assets=level_2a_final,
            level_2b_assets=level_2b_final,
            net_cash_outflows=net_outflows,
            lcr_ratio=lcr_ratio,
            compliant=compliant,
            buffer=buffer
        )
    
    # =========================================================================
    # LIQUIDITY STRESS TESTING
    # =========================================================================
    
    def run_liquidity_stress_test(
        self,
        position_value: float,
        normal_spread: float,
        normal_volume: float,
        volatility: float,
        scenarios: Dict[str, Dict]
    ) -> List[StressTestResult]:
        """
        Run liquidity stress tests under various scenarios.
        """
        results = []
        
        for scenario_name, params in scenarios.items():
            spread_multiplier = params.get('spread_multiplier', 1.0)
            volume_multiplier = params.get('volume_multiplier', 1.0)
            volatility_multiplier = params.get('volatility_multiplier', 1.0)
            
            stressed_spread = normal_spread * spread_multiplier
            stressed_volume = normal_volume * volume_multiplier
            stressed_vol = volatility * volatility_multiplier
            
            impact_result = self.compute_market_impact_square_root(
                position_value,
                stressed_volume,
                stressed_vol,
                1.0,
                execution_days=1.0
            )
            
            spread_cost = self.compute_spread_cost(position_value, stressed_spread)
            liquidation_cost = spread_cost + impact_result.impact_cost
            
            max_daily_liquidation = stressed_volume * 0.25
            time_to_liquidate = position_value / max_daily_liquidation if max_daily_liquidation > 0 else float('inf')
            
            survival_period = int(30 / spread_multiplier)
            
            results.append(StressTestResult(
                scenario_name=scenario_name,
                stressed_spread=stressed_spread,
                stressed_volume=stressed_volume,
                stressed_impact=impact_result.total_impact,
                liquidation_cost=liquidation_cost,
                time_to_liquidate=time_to_liquidate,
                survival_period=survival_period
            ))
        
        return results
    
    def classify_hqla(
        self,
        asset_type: str,
        credit_rating: str,
        market_cap: float
    ) -> HQLALevel:
        """
        Classify asset into HQLA levels per Basel III.
        """
        level_1_types = ['cash', 'central_bank_reserves', 'sovereign_debt_aaa']
        level_2a_types = ['sovereign_debt_aa', 'covered_bonds_aa', 'corporate_bonds_aa']
        level_2b_types = ['corporate_bonds_a', 'equity_index', 'rmbs_aa']
        
        if asset_type.lower() in level_1_types:
            return HQLALevel.LEVEL_1
        elif asset_type.lower() in level_2a_types or credit_rating in ['AAA', 'AA+', 'AA', 'AA-']:
            return HQLALevel.LEVEL_2A
        elif asset_type.lower() in level_2b_types or credit_rating in ['A+', 'A', 'A-']:
            return HQLALevel.LEVEL_2B
        else:
            return HQLALevel.NON_HQLA