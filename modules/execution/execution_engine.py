"""
Institutional Execution Engine
Author: Erdinc Erdogan
Purpose: Provides institutional-grade execution algorithms including TWAP, VWAP, Implementation Shortfall, Almgren-Chriss optimal trajectory, and liquidity-seeking strategies.
References:
- Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
- TWAP/VWAP Execution Benchmarks
- Implementation Shortfall Minimization
Usage:
    engine = ExecutionEngine(algorithm=ExecutionAlgorithm.VWAP)
    schedule = engine.create_schedule(quantity=10000, duration_minutes=60)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class ExecutionAlgorithm(Enum):
    """Execution algorithm types"""
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ALMGREN_CHRISS = "almgren_chriss"
    POV = "pov"
    ICEBERG = "iceberg"
    ADAPTIVE = "adaptive"
    LIQUIDITY_SEEKING = "liquidity_seeking"
    ARRIVAL_PRICE = "arrival_price"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    PEGGED = "pegged"


class ExecutionUrgency(Enum):
    """Execution urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OrderSlice:
    """Individual order slice in execution schedule"""
    slice_id: int
    timestamp: float
    quantity: float
    price_limit: Optional[float]
    order_type: str
    participation_rate: float
    cumulative_quantity: float
    remaining_quantity: float


@dataclass
class ExecutionSchedule:
    """Complete execution schedule"""
    algorithm: str
    total_quantity: float
    side: str
    slices: List[OrderSlice]
    expected_duration: float
    expected_cost: float
    expected_market_impact: float
    participation_rate: float
    n_slices: int


@dataclass
class ExecutionResult:
    """Execution performance results"""
    algorithm: str
    total_quantity: float
    executed_quantity: float
    average_price: float
    arrival_price: float
    vwap_benchmark: float
    twap_benchmark: float
    implementation_shortfall: float
    implementation_shortfall_bps: float
    market_impact: float
    timing_cost: float
    execution_time: float
    fill_rate: float
    slippage: float
    slippage_bps: float


@dataclass
class MarketImpactEstimate:
    """Market impact estimation"""
    permanent_impact: float
    temporary_impact: float
    total_impact: float
    impact_cost: float
    impact_bps: float
    decay_time: float
    model: str


@dataclass
class OptimalTrajectory:
    """Almgren-Chriss optimal execution trajectory"""
    time_points: np.ndarray
    inventory_trajectory: np.ndarray
    trade_schedule: np.ndarray
    expected_cost: float
    cost_variance: float
    efficient_frontier_lambda: float
    urgency_parameter: float


@dataclass
class TransactionCostAnalysis:
    """Transaction Cost Analysis (TCA) results"""
    total_cost: float
    total_cost_bps: float
    commission_cost: float
    spread_cost: float
    market_impact_cost: float
    timing_cost: float
    opportunity_cost: float
    delay_cost: float
    price_improvement: float


@dataclass
class VolumeProfile:
    """Intraday volume profile"""
    time_buckets: np.ndarray
    volume_weights: np.ndarray
    cumulative_weights: np.ndarray
    peak_periods: List[int]
    low_periods: List[int]


class ExecutionEngine(BaseModule):
    """
    Institutional-Grade Execution Engine.
    
    Implements comprehensive execution algorithms for optimal trade execution,
    minimizing market impact and transaction costs.
    
    Mathematical Framework:
    ----------------------TWAP (Time-Weighted Average Price):
        Trade equal quantities at regular intervals
        q_i = Q / N  for i = 1, ..., N
        Benchmark: TWAP = (1/N) × Σ P_i
    
    VWAP (Volume-Weighted Average Price):
        Trade proportional to expected volume
        q_i = Q × (V_i / Σ V_j)
        
        Benchmark: VWAP = Σ(P_i × V_i) / Σ V_i
    
    Implementation Shortfall (Perold, 1988):
        IS = (Execution Price - Decision Price) × Quantity
        IS = Market Impact + Timing Cost + Opportunity Cost
        
        Components:
        - Delay Cost: Price move from decision to first execution
        - Trading Cost: Price move during execution
        - Opportunity Cost: Unexecuted portion
    
    Almgren-Chriss Optimal Execution (2000):
        Minimize: E[Cost] + λ × Var[Cost]
        
        Cost = Permanent Impact + Temporary Impact + Timing Risk
        
        Permanent Impact: g(v) = γ × v
        Temporary Impact: h(v) = η × sign(v) × |v|^δ
        Optimal Trajectory:
        x_j = X × sinh(κ(T-t_j)) / sinh(κT)
        
        where κ = √(λσ²/η)
    
    Percentage of Volume (POV):
        q_i = ρ × V_i
        
        where ρ is target participation rate
    
    Market Impact Models:
        Linear: ΔP = λ × σ × (Q/ADV)
        Square Root: ΔP = γ × σ × √(Q/ADV)
        Power Law: ΔP = α × σ × (Q/ADV)^δ
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.default_participation_rate: float = self.config.get('participation_rate', 0.10)
        self.temporary_impact_coef: float = self.config.get('temporary_impact', 0.1)
        self.permanent_impact_coef: float = self.config.get('permanent_impact', 0.05)
        self.risk_aversion: float = self.config.get('risk_aversion', 1e-6)
        self.trading_hours: float = self.config.get('trading_hours', 6.5)
        self.commission_rate: float = self.config.get('commission_rate', 0.0001)
    
    #=========================================================================
    # TWAP (TIME-WEIGHTED AVERAGE PRICE)
    # =========================================================================
    
    def generate_twap_schedule(
        self,
        total_quantity: float,
        duration_minutes: float,
        interval_minutes: float = 5.0,
        side: OrderSide = OrderSide.BUY,
        start_time: float = 0.0
    ) -> ExecutionSchedule:
        """
        Generate TWAP execution schedule.
        
        Trade equal quantities at regular intervals.
        q_i = Q / N for all i
        """
        n_slices = max(1, int(duration_minutes / interval_minutes))
        quantity_per_slice = total_quantity / n_slices
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_slices):
            timestamp = start_time + i * interval_minutes
            cumulative += quantity_per_slice
            
            slices.append(OrderSlice(
                slice_id=i,
                timestamp=timestamp,
                quantity=quantity_per_slice,
                price_limit=None,
                order_type=OrderType.MARKET.value,
                participation_rate=1.0 / n_slices,
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.TWAP.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=duration_minutes,
            expected_cost=0.0,
            expected_market_impact=0.0,
            participation_rate=1.0 / n_slices,
            n_slices=n_slices
        )
    
    def calculate_twap_benchmark(
        self,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate TWAP benchmark price.
        TWAP = (1/N) × Σ P_i
        """
        prices = np.asarray(prices)
        return float(np.mean(prices))
    
    # =========================================================================
    # VWAP (VOLUME-WEIGHTED AVERAGE PRICE)
    # =========================================================================
    
    def generate_vwap_schedule(
        self,
        total_quantity: float,
        volume_profile: np.ndarray,
        duration_minutes: float,
        side: OrderSide = OrderSide.BUY,
        start_time: float = 0.0
    ) -> ExecutionSchedule:
        """
        Generate VWAP execution schedule.
        
        Trade proportional to expected volume.
        q_i = Q × (V_i / Σ V_j)
        """
        volume_profile = np.asarray(volume_profile)
        n_slices = len(volume_profile)
        
        # Normalize volume weights
        volume_weights = volume_profile / np.sum(volume_profile)
        
        # Calculate quantities per slice
        quantities = total_quantity * volume_weights
        
        interval_minutes = duration_minutes / n_slices
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_slices):
            timestamp = start_time + i * interval_minutes
            cumulative += quantities[i]
            
            slices.append(OrderSlice(
                slice_id=i,
                timestamp=timestamp,
                quantity=float(quantities[i]),
                price_limit=None,
                order_type=OrderType.MARKET.value,
                participation_rate=float(volume_weights[i]),
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.VWAP.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=duration_minutes,
            expected_cost=0.0,
            expected_market_impact=0.0,
            participation_rate=float(np.max(volume_weights)),
            n_slices=n_slices
        )
    
    def calculate_vwap_benchmark(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """
        Calculate VWAP benchmark price.
        
        VWAP = Σ(P_i × V_i) / Σ V_i
        """
        prices = np.asarray(prices)
        volumes = np.asarray(volumes)
        
        total_value = np.sum(prices * volumes)
        total_volume = np.sum(volumes)
        
        return float(total_value / total_volume) if total_volume > 0 else float(np.mean(prices))
    
    def estimate_volume_profile(
        self,
        historical_volumes: np.ndarray,
        n_buckets: int = 78
    ) -> VolumeProfile:
        """
        Estimate intraday volume profile from historical data.
        
        Typically78 buckets for 5-minute intervals in 6.5-hour trading day.
        """
        historical_volumes = np.asarray(historical_volumes)
        
        if historical_volumes.ndim == 1:
            # Single day - reshape to buckets
            n_obs = len(historical_volumes)
            bucket_size = max(1, n_obs // n_buckets)
            
            volume_by_bucket = []
            for i in range(n_buckets):
                start_idx = i * bucket_size
                end_idx = min((i + 1) * bucket_size, n_obs)
                volume_by_bucket.append(np.sum(historical_volumes[start_idx:end_idx]))
            
            avg_volume = np.array(volume_by_bucket)
        else:
            # Multiple days - average across days
            avg_volume = np.mean(historical_volumes, axis=0)
        
        # Normalize
        volume_weights = avg_volume / np.sum(avg_volume)
        cumulative_weights = np.cumsum(volume_weights)
        
        # Identify peak and low periods
        threshold_high = np.percentile(volume_weights, 75)
        threshold_low = np.percentile(volume_weights, 25)
        
        peak_periods = list(np.where(volume_weights >= threshold_high)[0])
        low_periods = list(np.where(volume_weights <= threshold_low)[0])
        
        return VolumeProfile(
            time_buckets=np.arange(n_buckets),
            volume_weights=volume_weights,
            cumulative_weights=cumulative_weights,
            peak_periods=peak_periods,
            low_periods=low_periods
        )
    
    # =========================================================================
    # IMPLEMENTATION SHORTFALL
    # =========================================================================
    
    def generate_is_schedule(
        self,
        total_quantity: float,
        arrival_price: float,
        volatility: float,
        daily_volume: float,
        duration_minutes: float,
        side: OrderSide = OrderSide.BUY,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    ) -> ExecutionSchedule:
        """
        Generate Implementation Shortfall minimizing schedule.
        
        Balances market impact vs timing risk.
        """
        # Urgency parameters
        urgency_params = {
            ExecutionUrgency.LOW: {'aggression': 0.3, 'front_load': 0.4},
            ExecutionUrgency.MEDIUM: {'aggression': 0.5, 'front_load': 0.5},
            ExecutionUrgency.HIGH: {'aggression': 0.7, 'front_load': 0.6},
            ExecutionUrgency.CRITICAL: {'aggression': 0.9, 'front_load': 0.8}
        }
        
        params = urgency_params[urgency]
        aggression = params['aggression']
        front_load = params['front_load']
        
        # Number of slices based on duration
        n_slices = max(1, int(duration_minutes / 5))
        interval_minutes = duration_minutes / n_slices
        
        # Generate front-loaded schedule
        weights = np.array([front_load ** i for i in range(n_slices)])
        weights = weights / np.sum(weights)
        
        quantities = total_quantity * weights
        
        # Estimate market impact
        participation = total_quantity / (daily_volume * duration_minutes / (self.trading_hours * 60))
        expected_impact = self._estimate_market_impact_simple(
            total_quantity, daily_volume, volatility, arrival_price
        )
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_slices):
            timestamp = i * interval_minutes
            cumulative += quantities[i]
            
            slices.append(OrderSlice(
                slice_id=i,
                timestamp=timestamp,
                quantity=float(quantities[i]),
                price_limit=None,
                order_type=OrderType.MARKET.value,
                participation_rate=float(weights[i]),
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=duration_minutes,
            expected_cost=expected_impact.impact_cost,
            expected_market_impact=expected_impact.total_impact,
            participation_rate=float(participation),
            n_slices=n_slices
        )
    
    def calculate_implementation_shortfall(
        self,
        decision_price: float,
        execution_prices: np.ndarray,
        execution_quantities: np.ndarray,
        side: OrderSide = OrderSide.BUY
    ) -> float:
        """
        Calculate Implementation Shortfall.
        
        IS = (Average Execution Price - Decision Price) × Total Quantity
        
        For buys: positive IS means we paid more than decision price
        For sells: positive IS means we received less than decision price
        """
        execution_prices = np.asarray(execution_prices)
        execution_quantities = np.asarray(execution_quantities)
        
        total_quantity = np.sum(execution_quantities)
        avg_execution_price = np.sum(execution_prices * execution_quantities) / total_quantity
        
        if side == OrderSide.BUY:
            shortfall = (avg_execution_price - decision_price) * total_quantity
        else:
            shortfall = (decision_price - avg_execution_price) * total_quantity
        
        return float(shortfall)
    
    def decompose_implementation_shortfall(
        self,
        decision_price: float,
        arrival_price: float,
        execution_prices: np.ndarray,
        execution_quantities: np.ndarray,
        final_price: float,
        total_intended_quantity: float,
        side: OrderSide = OrderSide.BUY
    ) -> Dict[str, float]:
        """
        Decompose Implementation Shortfall into components.
        
        IS = Delay Cost + Trading Cost + Opportunity Cost
        """
        execution_prices = np.asarray(execution_prices)
        execution_quantities = np.asarray(execution_quantities)
        
        executed_quantity = np.sum(execution_quantities)
        unexecuted_quantity = total_intended_quantity - executed_quantity
        
        avg_execution_price = np.sum(execution_prices * execution_quantities) / executed_quantity
        
        sign = 1 if side == OrderSide.BUY else -1
        
        # Delay cost: price move from decision to arrival
        delay_cost = sign * (arrival_price - decision_price) * total_intended_quantity
        
        # Trading cost: price move during execution
        trading_cost = sign * (avg_execution_price - arrival_price) * executed_quantity
        
        # Opportunity cost: unexecuted portion
        opportunity_cost = sign * (final_price - decision_price) * unexecuted_quantity
        
        # Total IS
        total_is = delay_cost + trading_cost + opportunity_cost
        
        # In basis points
        total_is_bps = (total_is / (decision_price * total_intended_quantity)) * 10000
        
        return {
            'total_is': float(total_is),
            'total_is_bps': float(total_is_bps),
            'delay_cost': float(delay_cost),
            'trading_cost': float(trading_cost),
            'opportunity_cost': float(opportunity_cost),
            'executed_quantity': float(executed_quantity),
            'fill_rate': float(executed_quantity / total_intended_quantity)
        }
    
    # =========================================================================
    # ALMGREN-CHRISS OPTIMAL EXECUTION
    # =========================================================================
    
    def generate_almgren_chriss_trajectory(
        self,
        total_quantity: float,
        volatility: float,
        daily_volume: float,
        price: float,
        duration_days: float = 1.0,
        risk_aversion: Optional[float] = None,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
        n_periods: int = 20
    ) -> OptimalTrajectory:
        """
        Generate Almgren-Chriss optimal execution trajectory.
        
        Minimize: E[Cost] + λ × Var[Cost]
        Optimal trajectory:
        x_j = X × sinh(κ(T-t_j)) / sinh(κT)
        
        where κ = √(λσ²/η)
        """
        risk_aversion = risk_aversion or self.risk_aversion
        eta = eta or self.temporary_impact_coef
        gamma = gamma or self.permanent_impact_coef
        
        # Time grid
        T = duration_days
        dt = T / n_periods
        time_points = np.linspace(0, T, n_periods + 1)
        
        # Kappa parameter
        if eta > 0 and risk_aversion > 0:
            kappa = np.sqrt(risk_aversion * volatility**2 / eta)
        else:
            kappa = 0.1
        
        # Optimal inventory trajectory
        # x_j = X × sinh(κ(T-t_j)) / sinh(κT)
        if kappa * T > 1e-6:
            inventory = total_quantity * np.sinh(kappa * (T - time_points)) / np.sinh(kappa * T)
        else:
            # Linear trajectory for small kappa
            inventory = total_quantity * (1 - time_points / T)
        
        # Trade schedule (negative of inventory changes)
        trade_schedule = -np.diff(inventory)
        
        # Expected cost calculation
        # E[Cost] = 0.5 × γ × X² + η × Σ n_j²
        permanent_cost = 0.5 * gamma * total_quantity**2 / daily_volume * volatility
        temporary_cost = eta * np.sum(trade_schedule**2) / np.sqrt(daily_volume * dt) * volatility
        
        expected_cost = (permanent_cost + temporary_cost) * price
        
        # Cost variance
        # Var[Cost] = σ² × Σ x_j² × dt
        cost_variance = volatility**2 * np.sum(inventory[:-1]**2) * dt * price**2
        
        return OptimalTrajectory(
            time_points=time_points,
            inventory_trajectory=inventory,
            trade_schedule=trade_schedule,
            expected_cost=float(expected_cost),
            cost_variance=float(cost_variance),
            efficient_frontier_lambda=float(risk_aversion),
            urgency_parameter=float(kappa)
        )
    
    def compute_efficient_frontier(
        self,
        total_quantity: float,
        volatility: float,
        daily_volume: float,
        price: float,
        duration_days: float = 1.0,
        n_points: int = 20
    ) -> pd.DataFrame:
        """
        Compute execution efficient frontier.
        
        Trade-off between expected cost and cost variance.
        """
        # Range of risk aversion parameters
        lambdas = np.logspace(-8, -4, n_points)
        
        results = []
        for lam in lambdas:
            trajectory = self.generate_almgren_chriss_trajectory(
                total_quantity, volatility, daily_volume, price,
                duration_days, risk_aversion=lam
            )
            
            results.append({
                'risk_aversion': lam,
                'expected_cost': trajectory.expected_cost,
                'cost_std': np.sqrt(trajectory.cost_variance),
                'urgency': trajectory.urgency_parameter
            })
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # PERCENTAGE OF VOLUME (POV)
    # =========================================================================
    
    def generate_pov_schedule(
        self,
        total_quantity: float,
        target_participation: float,
        expected_volumes: np.ndarray,
        interval_minutes: float = 5.0,
        side: OrderSide = OrderSide.BUY,
        max_participation: float = 0.25
    ) -> ExecutionSchedule:
        """
        Generate Percentage of Volume (POV) schedule.
        
        q_i = min(ρ × V_i, max_participation × V_i)
        """
        expected_volumes = np.asarray(expected_volumes)
        n_slices = len(expected_volumes)
        
        # Cap participation rate
        participation = min(target_participation, max_participation)
        
        # Calculate quantities
        quantities = participation * expected_volumes
        
        # Ensure we don't exceed total quantity
        cumsum = np.cumsum(quantities)
        if cumsum[-1] > total_quantity:
            # Scale down
            scale = total_quantity / cumsum[-1]
            quantities = quantities * scale
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_slices):
            if cumulative >= total_quantity:
                break
            qty = min(quantities[i], total_quantity - cumulative)
            timestamp = i * interval_minutes
            cumulative += qty
            
            slices.append(OrderSlice(
                slice_id=i,
                timestamp=timestamp,
                quantity=float(qty),
                price_limit=None,
                order_type=OrderType.MARKET.value,
                participation_rate=float(participation),
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        duration = len(slices) * interval_minutes
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.POV.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=duration,
            expected_cost=0.0,
            expected_market_impact=0.0,
            participation_rate=float(participation),
            n_slices=len(slices)
        )
    
    # =========================================================================
    # ICEBERG ORDERS
    # =========================================================================
    
    def generate_iceberg_schedule(
        self,
        total_quantity: float,
        display_quantity: float,
        price_limit: float,
        side: OrderSide = OrderSide.BUY,
        refresh_interval_seconds: float = 30.0
    ) -> ExecutionSchedule:
        """
        Generate Iceberg order schedule.
        
        Show only display_quantity at a time, refresh when filled.
        """
        n_slices = int(np.ceil(total_quantity / display_quantity))
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_slices):
            qty = min(display_quantity, total_quantity - cumulative)
            timestamp = i * refresh_interval_seconds / 60.0  # Convert to minutes
            cumulative += qty
            
            slices.append(OrderSlice(
                slice_id=i,
                timestamp=timestamp,
                quantity=float(qty),
                price_limit=price_limit,
                order_type=OrderType.ICEBERG.value,
                participation_rate=float(display_quantity / total_quantity),
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.ICEBERG.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=n_slices * refresh_interval_seconds / 60.0,
            expected_cost=0.0,
            expected_market_impact=0.0,
            participation_rate=float(display_quantity / total_quantity),
            n_slices=n_slices
        )
    
    # =========================================================================
    # ADAPTIVE EXECUTION
    # =========================================================================
    
    def generate_adaptive_schedule(
        self,
        total_quantity: float,
        arrival_price: float,
        current_price: float,
        volatility: float,
        daily_volume: float,
        duration_minutes: float,
        side: OrderSide = OrderSide.BUY,
        aggression_factor: float = 0.5
    ) -> ExecutionSchedule:
        """
        Generate adaptive execution schedule.
        
        Adjusts aggressiveness based on price movement relative to arrival.
        """
        n_slices = max(1, int(duration_minutes / 5))
        interval_minutes = duration_minutes / n_slices
        
        # Calculate price deviation
        price_move = (current_price - arrival_price) / arrival_price
        
        # Adjust aggression based on price move
        if side == OrderSide.BUY:
            # More aggressive if price falling, less if rising
            adjusted_aggression = aggression_factor * (1 - price_move * 10)
        else:
            # More aggressive if price rising, less if falling
            adjusted_aggression = aggression_factor * (1 + price_move * 10)
        
        adjusted_aggression = np.clip(adjusted_aggression, 0.1, 0.9)
        
        # Generate schedule with adjusted front-loading
        weights = np.array([adjusted_aggression ** i for i in range(n_slices)])
        weights = weights / np.sum(weights)
        
        quantities = total_quantity * weights
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_slices):
            timestamp = i * interval_minutes
            cumulative += quantities[i]
            
            slices.append(OrderSlice(
                slice_id=i,
                timestamp=timestamp,
                quantity=float(quantities[i]),
                price_limit=None,
                order_type=OrderType.MARKET.value,
                participation_rate=float(weights[i]),
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        expected_impact = self._estimate_market_impact_simple(
            total_quantity, daily_volume, volatility, arrival_price
        )
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.ADAPTIVE.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=duration_minutes,
            expected_cost=expected_impact.impact_cost,
            expected_market_impact=expected_impact.total_impact,
            participation_rate=float(adjusted_aggression),
            n_slices=n_slices
        )
    
    # =========================================================================
    # MARKET IMPACT MODELS
    # =========================================================================
    
    def estimate_market_impact_linear(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        price: float,
        lambda_coef: Optional[float] = None
    ) -> MarketImpactEstimate:
        """
        Linear market impact model.
        
        ΔP/P = λ × σ × (Q / ADV)
        """
        lam = lambda_coef or (self.temporary_impact_coef + self.permanent_impact_coef)
        
        participation = quantity / daily_volume
        impact_pct = lam * volatility * participation
        
        permanent = self.permanent_impact_coef * volatility * participation
        temporary = self.temporary_impact_coef * volatility * participation
        
        impact_cost = impact_pct * quantity * price
        
        return MarketImpactEstimate(
            permanent_impact=float(permanent),
            temporary_impact=float(temporary),
            total_impact=float(impact_pct),
            impact_cost=float(impact_cost),
            impact_bps=float(impact_pct * 10000),
            decay_time=1.0,
            model="linear"
        )
    
    def estimate_market_impact_square_root(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        price: float,
        gamma_coef: Optional[float] = None
    ) -> MarketImpactEstimate:
        """
        Square root market impact model.
        
        ΔP/P = γ × σ × √(Q / ADV)
        """
        gamma = gamma_coef or self.temporary_impact_coef
        
        participation = quantity / daily_volume
        impact_pct = gamma * volatility * np.sqrt(participation)
        
        permanent = 0.4 * impact_pct
        temporary = 0.6 * impact_pct
        
        impact_cost = impact_pct * quantity * price
        
        return MarketImpactEstimate(
            permanent_impact=float(permanent),
            temporary_impact=float(temporary),
            total_impact=float(impact_pct),
            impact_cost=float(impact_cost),
            impact_bps=float(impact_pct * 10000),
            decay_time=0.5,
            model="square_root"
        )
    
    def estimate_market_impact_power_law(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        price: float,
        alpha: float = 0.1,
        delta: float = 0.5
    ) -> MarketImpactEstimate:
        """
        Power law market impact model.
        
        ΔP/P = α × σ × (Q / ADV)^δ
        """
        participation = quantity / daily_volume
        impact_pct = alpha * volatility * (participation ** delta)
        
        permanent = 0.3 * impact_pct
        temporary = 0.7 * impact_pct
        
        impact_cost = impact_pct * quantity * price
        
        return MarketImpactEstimate(
            permanent_impact=float(permanent),
            temporary_impact=float(temporary),
            total_impact=float(impact_pct),
            impact_cost=float(impact_cost),
            impact_bps=float(impact_pct * 10000),
            decay_time=0.3,
            model="power_law"
        )
    
    def _estimate_market_impact_simple(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        price: float
    ) -> MarketImpactEstimate:
        """Simple market impact estimate using square root model."""
        return self.estimate_market_impact_square_root(
            quantity, daily_volume, volatility, price
        )
    
    # =========================================================================
    # TRANSACTION COST ANALYSIS (TCA)
    # =========================================================================
    
    def analyze_execution(
        self,
        execution_prices: np.ndarray,
        execution_quantities: np.ndarray,
        execution_timestamps: np.ndarray,
        market_prices: np.ndarray,
        market_volumes: np.ndarray,
        decision_price: float,
        arrival_price: float,
        side: OrderSide = OrderSide.BUY
    ) -> ExecutionResult:
        """
        Comprehensive execution analysis.
        """
        execution_prices = np.asarray(execution_prices)
        execution_quantities = np.asarray(execution_quantities)
        market_prices = np.asarray(market_prices)
        market_volumes = np.asarray(market_volumes)
        
        total_quantity = np.sum(execution_quantities)
        avg_price = np.sum(execution_prices * execution_quantities) / total_quantity
        
        # Benchmarks
        vwap_benchmark = self.calculate_vwap_benchmark(market_prices, market_volumes)
        twap_benchmark = self.calculate_twap_benchmark(market_prices)
        
        # Implementation shortfall
        sign = 1 if side == OrderSide.BUY else -1
        is_value = sign * (avg_price - decision_price) * total_quantity
        is_bps = sign * (avg_price - decision_price) / decision_price * 10000
        
        # Market impact (vs arrival)
        market_impact = sign * (avg_price - arrival_price) / arrival_price
        
        # Timing cost
        timing_cost = sign * (arrival_price - decision_price) / decision_price
        
        # Slippage vs VWAP
        slippage = sign * (avg_price - vwap_benchmark)
        slippage_bps = slippage / vwap_benchmark * 10000
        
        # Execution time
        execution_time = float(execution_timestamps[-1] - execution_timestamps[0])
        
        return ExecutionResult(
            algorithm="analyzed",
            total_quantity=float(total_quantity),
            executed_quantity=float(total_quantity),
            average_price=float(avg_price),
            arrival_price=float(arrival_price),
            vwap_benchmark=float(vwap_benchmark),
            twap_benchmark=float(twap_benchmark),
            implementation_shortfall=float(is_value),
            implementation_shortfall_bps=float(is_bps),
            market_impact=float(market_impact),
            timing_cost=float(timing_cost),
            execution_time=float(execution_time),
            fill_rate=1.0,
            slippage=float(slippage),
            slippage_bps=float(slippage_bps)
        )
    
    def compute_tca(
        self,
        execution_prices: np.ndarray,
        execution_quantities: np.ndarray,
        decision_price: float,
        arrival_price: float,
        close_price: float,
        bid_ask_spread: float,
        side: OrderSide = OrderSide.BUY
    ) -> TransactionCostAnalysis:
        """
        Compute detailed Transaction Cost Analysis.
        """
        execution_prices = np.asarray(execution_prices)
        execution_quantities = np.asarray(execution_quantities)
        
        total_quantity = np.sum(execution_quantities)
        total_value = np.sum(execution_prices * execution_quantities)
        avg_price = total_value / total_quantity
        
        sign = 1 if side == OrderSide.BUY else -1
        
        # Commission cost
        commission_cost = self.commission_rate * total_value
        
        # Spread cost (half spread)
        spread_cost = 0.5 * bid_ask_spread * total_value
        
        # Market impact (execution vs arrival)
        market_impact_cost = sign * (avg_price - arrival_price) * total_quantity
        
        # Timing cost (arrival vs decision)
        timing_cost = sign * (arrival_price - decision_price) * total_quantity
        
        # Delay cost (same as timing for simplicity)
        delay_cost = timing_cost
        
        # Opportunity cost (close vs decision for unexecuted)
        opportunity_cost = 0.0  # Assuming full execution
        
        # Total cost
        total_cost = commission_cost + spread_cost + market_impact_cost + timing_cost
        total_cost_bps = total_cost / total_value * 10000
        
        # Price improvement (if any)
        if side == OrderSide.BUY:
            price_improvement = max(0, arrival_price - avg_price) * total_quantity
        else:
            price_improvement = max(0, avg_price - arrival_price) * total_quantity
        
        return TransactionCostAnalysis(
            total_cost=float(total_cost),
            total_cost_bps=float(total_cost_bps),
            commission_cost=float(commission_cost),
            spread_cost=float(spread_cost),
            market_impact_cost=float(market_impact_cost),
            timing_cost=float(timing_cost),
            opportunity_cost=float(opportunity_cost),
            delay_cost=float(delay_cost),
            price_improvement=float(price_improvement)
        )
    
    # =========================================================================
    # LIQUIDITY-SEEKING ALGORITHM
    # =========================================================================
    
    def generate_liquidity_seeking_schedule(
        self,
        total_quantity: float,
        spread_threshold: float,
        volume_threshold: float,
        expected_spreads: np.ndarray,
        expected_volumes: np.ndarray,
        interval_minutes: float = 5.0,
        side: OrderSide = OrderSide.BUY
    ) -> ExecutionSchedule:
        """
        Generate liquidity-seeking execution schedule.
        
        Trade more aggressively when spreads are tight and volume is high.
        """
        expected_spreads = np.asarray(expected_spreads)
        expected_volumes = np.asarray(expected_volumes)
        n_periods = len(expected_spreads)
        
        # Liquidity score: high volume, low spread = good
        spread_score = 1 - (expected_spreads / np.max(expected_spreads))
        volume_score = expected_volumes / np.max(expected_volumes)
        
        liquidity_score = 0.5 * spread_score + 0.5 * volume_score
        
        # Identify good periods
        good_periods = (expected_spreads <= spread_threshold) & (expected_volumes >= volume_threshold)
        
        # Allocate more to good periods
        weights = liquidity_score.copy()
        weights[good_periods] *= 1.5
        weights = weights / np.sum(weights)
        
        quantities = total_quantity * weights
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_periods):
            if quantities[i] < 1e-6:
                continue
            timestamp = i * interval_minutes
            cumulative += quantities[i]
            
            slices.append(OrderSlice(
                slice_id=len(slices),
                timestamp=timestamp,
                quantity=float(quantities[i]),
                price_limit=None,
                order_type=OrderType.LIMIT.value if good_periods[i] else OrderType.MARKET.value,
                participation_rate=float(weights[i]),
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.LIQUIDITY_SEEKING.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=n_periods * interval_minutes,
            expected_cost=0.0,
            expected_market_impact=0.0,
            participation_rate=float(np.max(weights)),
            n_slices=len(slices)
        )
    
    # =========================================================================
    # ARRIVAL PRICE ALGORITHM
    # =========================================================================
    
    def generate_arrival_price_schedule(
        self,
        total_quantity: float,
        arrival_price: float,
        current_price: float,
        volatility: float,
        daily_volume: float,
        max_duration_minutes: float = 60.0,
        side: OrderSide = OrderSide.BUY
    ) -> ExecutionSchedule:
        """
        Generate Arrival Price algorithm schedule.
        
        Minimize deviation from arrival price benchmark.
        """
        # Calculate optimal duration based on quantity and volume
        participation_target = 0.10
        estimated_duration = (total_quantity / daily_volume) / participation_target * self.trading_hours * 60
        duration = min(estimated_duration, max_duration_minutes)
        
        n_slices = max(1, int(duration / 5))
        interval_minutes = duration / n_slices
        
        # Price deviation from arrival
        price_deviation = (current_price - arrival_price) / arrival_price
        
        # Adjust schedule based on price movement
        if side == OrderSide.BUY:
            # If price above arrival, be more aggressive
            urgency = 1.0 + price_deviation * 5
        else:
            # If price below arrival, be more aggressive
            urgency = 1.0 - price_deviation * 5
        
        urgency = np.clip(urgency, 0.5, 2.0)
        
        # Generate schedule
        base_weights = np.ones(n_slices)
        # Front-load based on urgency
        for i in range(n_slices):
            base_weights[i] = urgency ** (n_slices - i - 1)
        
        weights = base_weights / np.sum(base_weights)
        quantities = total_quantity * weights
        
        slices = []
        cumulative = 0.0
        
        for i in range(n_slices):
            timestamp = i * interval_minutes
            cumulative += quantities[i]
            
            slices.append(OrderSlice(
                slice_id=i,
                timestamp=timestamp,
                quantity=float(quantities[i]),
                price_limit=None,
                order_type=OrderType.MARKET.value,
                participation_rate=float(weights[i]),
                cumulative_quantity=cumulative,
                remaining_quantity=total_quantity - cumulative
            ))
        
        expected_impact = self._estimate_market_impact_simple(
            total_quantity, daily_volume, volatility, arrival_price
        )
        
        return ExecutionSchedule(
            algorithm=ExecutionAlgorithm.ARRIVAL_PRICE.value,
            total_quantity=total_quantity,
            side=side.value,
            slices=slices,
            expected_duration=duration,
            expected_cost=expected_impact.impact_cost,
            expected_market_impact=expected_impact.total_impact,
            participation_rate=float(participation_target),
            n_slices=n_slices
        )