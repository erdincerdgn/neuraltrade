"""
Execution Optimizer with TWAP/VWAP
Author: Erdinc Erdogan
Purpose: Optimizes order execution by slicing large orders into TWAP/VWAP schedules to minimize market impact and timing risk.
References:
- TWAP: Time-Weighted Average Price
- VWAP: Volume-Weighted Average Price
- Optimal Order Slicing Theory
Usage:
    optimizer = ExecutionOptimizer()
    plan = optimizer.create_vwap_plan(quantity=10000, volume_profile=profile, duration=60)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# EXECUTION ALGORITHM TYPES
# ============================================================================

class ExecutionAlgorithm(Enum):
    """Supported execution algorithms."""
    TWAP = auto()          # Time-Weighted Average Price
    VWAP = auto()          # Volume-Weighted Average Price
    IS = auto()            # Implementation Shortfall
    POV = auto()           # Percentage of Volume
    ARRIVAL = auto()       # Arrival Price


# ============================================================================
# EXECUTION SLICE
# ============================================================================

@dataclass
class ExecutionSlice:
    """Single execution slice."""
    slice_id: int
    quantity: float
    target_time: datetime
    price_limit: Optional[float] = None
    executed_quantity: float = 0.0
    executed_price: float = 0.0
    status: str = "PENDING"
    
    @property
    def fill_rate(self) -> float:
        """Percentage filled."""
        return self.executed_quantity / self.quantity if self.quantity > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            "slice_id": self.slice_id,
            "quantity": self.quantity,
            "target_time": self.target_time.isoformat(),
            "price_limit": self.price_limit,
            "executed_quantity": self.executed_quantity,
            "executed_price": self.executed_price,
            "status": self.status,
            "fill_rate": self.fill_rate
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan."""
    algorithm: ExecutionAlgorithm
    total_quantity: float
    slices: List[ExecutionSlice]
    start_time: datetime
    end_time: datetime
    expected_cost_bps: float
    volume_profile: Optional[np.ndarray] = None
    
    @property
    def n_slices(self) -> int:
        return len(self.slices)
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60
    
    @property
    def executed_quantity(self) -> float:
        return sum(s.executed_quantity for s in self.slices)
    
    @property
    def fill_rate(self) -> float:
        return self.executed_quantity / self.total_quantity if self.total_quantity > 0 else 0
    
    @property
    def vwap(self) -> float:
        """Volume-weighted average execution price."""
        total_value = sum(s.executed_quantity * s.executed_price for s in self.slices)
        total_qty = self.executed_quantity
        return total_value / total_qty if total_qty > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            "algorithm": self.algorithm.name,
            "total_quantity": self.total_quantity,
            "n_slices": self.n_slices,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "expected_cost_bps": self.expected_cost_bps,
            "fill_rate": self.fill_rate,
            "vwap": self.vwap
        }


# ============================================================================
# INTRADAY VOLUME PROFILE
# ============================================================================

class IntradayVolumeProfile:
    """
    Intraday volume profile for VWAP execution.
    
    Models the typical U-shaped volume pattern:
    - High volume at open
    - Low volume midday
    - High volume at close
    """
    
    # Default U-shaped profile (30-minute buckets, 13 buckets for 6.5 hour day)
    DEFAULT_PROFILE = np.array([
        0.12,  # 9:30-10:00 (high open)
        0.09,  # 10:00-10:30
        0.07,  # 10:30-11:00
        0.06,  # 11:00-11:30
        0.05,  # 11:30-12:00 (low midday)
        0.05,  # 12:00-12:30
        0.05,  # 12:30-13:00
        0.06,  # 13:00-13:30
        0.07,  # 13:30-14:00
        0.08,  # 14:00-14:30
        0.09,  # 14:30-15:00
        0.10,  # 15:00-15:30
        0.11,  # 15:30-16:00 (high close)
    ])
    
    def __init__(self, custom_profile: np.ndarray = None):
        """
        Initialize volume profile.
        
        Args:
            custom_profile: Custom volume profile (must sum to 1)
        """
        if custom_profile is not None:
            self.profile = custom_profile / custom_profile.sum()
        else:
            self.profile = self.DEFAULT_PROFILE / self.DEFAULT_PROFILE.sum()
        
        self.n_buckets = len(self.profile)
        self.bucket_duration_minutes = 390 / self.n_buckets
    
    def get_volume_fraction(self, start_minute: float, end_minute: float) -> float:
        """
        Get volume fraction for a time window.
        
        Args:
            start_minute: Start time in minutes from market open
            end_minute: End time in minutes from market open
            
        Returns:
            Fraction of daily volume in this window
        """
        start_bucket = int(start_minute / self.bucket_duration_minutes)
        end_bucket = int(end_minute / self.bucket_duration_minutes)
        
        start_bucket = max(0, min(start_bucket, self.n_buckets - 1))
        end_bucket = max(0, min(end_bucket, self.n_buckets - 1))
        
        return self.profile[start_bucket:end_bucket + 1].sum()
    
    def get_slice_weights(self, n_slices: int, 
                          start_minute: float = 0, 
                          end_minute: float = 390) -> np.ndarray:
        """
        Get VWAP slice weights based on volume profile.
        
        Args:
            n_slices: Number of execution slices
            start_minute: Start time in minutes from open
            end_minute: End time in minutes from open
            
        Returns:
            Array of weights for each slice
        """
        duration = end_minute - start_minute
        slice_duration = duration / n_slices
        
        weights = np.zeros(n_slices)
        for i in range(n_slices):
            slice_start = start_minute + i * slice_duration
            slice_end = slice_start + slice_duration
            weights[i] = self.get_volume_fraction(slice_start, slice_end)
        
        # Normalize
        return weights / weights.sum()


# ============================================================================
# TWAP EXECUTOR
# ============================================================================

class TWAPExecutor:
    """
    Time-Weighted Average Price (TWAP) Executor.
    
    Splits order into equal-sized slices executed at regular intervals.
    
    Formula:
        slice_size = Q / n_slices
        interval = T / n_slices
    
    Benefits:
    - Simple and predictable
    - Minimizes timing risk
    - Good for low-urgency orders
    
    Usage:
        twap = TWAPExecutor()
        plan = twap.create_plan(quantity, duration_minutes, n_slices)
    """
    
    def __init__(self, randomize: bool = True, randomize_pct: float = 0.1):
        """
        Initialize TWAP executor.
        
        Args:
            randomize: Add randomization to slice sizes/times
            randomize_pct: Maximum randomization percentage
        """
        self.randomize = randomize
        self.randomize_pct = randomize_pct
    
    def create_plan(self,
                    quantity: float,
                    duration_minutes: float,
                    n_slices: int = 10,
                    start_time: datetime = None,
                    price: float = None,
                    volatility: float = 0.02) -> ExecutionPlan:
        """
        Create TWAP execution plan.
        
        Args:
            quantity: Total quantity to execute
            duration_minutes: Execution duration
            n_slices: Number of slices
            start_time: Execution start time
            price: Current price (for cost estimation)
            volatility: Daily volatility
            
        Returns:
            ExecutionPlan with TWAP slices
        """
        if start_time is None:
            start_time = datetime.now()
        
        # Calculate slice parameters
        base_slice_size = quantity / n_slices
        interval_minutes = duration_minutes / n_slices
        
        slices = []
        for i in range(n_slices):
            # Apply randomization if enabled
            if self.randomize:
                size_factor = 1 + np.random.uniform(-self.randomize_pct, self.randomize_pct)
                time_factor = 1 + np.random.uniform(-self.randomize_pct/2, self.randomize_pct/2)
            else:
                size_factor = 1.0
                time_factor = 1.0
            
            slice_size = base_slice_size * size_factor
            slice_time = start_time + timedelta(minutes=i * interval_minutes * time_factor)
            
            slices.append(ExecutionSlice(
                slice_id=i,
                quantity=slice_size,
                target_time=slice_time
            ))
        
        # Normalize slice sizes to match total quantity
        total_planned = sum(s.quantity for s in slices)
        for s in slices:
            s.quantity = s.quantity * quantity / total_planned
        
        # Estimate execution cost
        expected_cost_bps = self._estimate_cost(quantity, duration_minutes, n_slices, volatility)
        
        return ExecutionPlan(
            algorithm=ExecutionAlgorithm.TWAP,
            total_quantity=quantity,
            slices=slices,
            start_time=start_time,
            end_time=start_time + timedelta(minutes=duration_minutes),
            expected_cost_bps=expected_cost_bps
        )
    
    def _estimate_cost(self, quantity: float, duration: float, 
                       n_slices: int, volatility: float) -> float:
        """Estimate execution cost in basis points."""
        # Timing risk increases with duration
        timing_risk = volatility * np.sqrt(duration / 390) * 10000 * 0.5
        
        # Impact cost decreases with more slices
        impact_cost = 5.0 / np.sqrt(n_slices)
        
        return timing_risk + impact_cost
    
    def optimal_n_slices(self, quantity: float, adv: float, 
                         duration_minutes: float, volatility: float) -> int:
        """
        Calculate optimal number of slices.
        
        Balances market impact vs timing risk.
        """
        participation = quantity / adv
        
        # More slices for larger orders
        base_slices = 10
        size_factor = np.sqrt(participation / 0.01)
        
        # More slices for longer duration
        time_factor = np.sqrt(duration_minutes / 60)
        
        optimal = int(base_slices * size_factor * time_factor)
        return max(5, min(optimal, 100))


# ============================================================================
# VWAP EXECUTOR
# ============================================================================

class VWAPExecutor:
    """
    Volume-Weighted Average Price (VWAP) Executor.
    
    Splits order according to expected volume profile.
    
    Formula:
        slice_size[i] = Q * volume_profile[i] / sum(volume_profile)
    
    Benefits:
    - Tracks market VWAP benchmark
    - Reduces market impact by trading with volume
    - Good for benchmark-sensitive orders
    
    Usage:
        vwap = VWAPExecutor()
        plan = vwap.create_plan(quantity, duration_minutes, n_slices)
    """
    
    def __init__(self, volume_profile: IntradayVolumeProfile = None):
        """
        Initialize VWAP executor.
        
        Args:
            volume_profile: Custom intraday volume profile
        """
        self.volume_profile = volume_profile or IntradayVolumeProfile()
    
    def create_plan(self,
                    quantity: float,
                    duration_minutes: float,
                    n_slices: int = 10,
                    start_time: datetime = None,
                    start_minute: float = 0,
                    price: float = None,
                    volatility: float = 0.02) -> ExecutionPlan:
        """
        Create VWAP execution plan.
        
        Args:
            quantity: Total quantity to execute
            duration_minutes: Execution duration
            n_slices: Number of slices
            start_time: Execution start time
            start_minute: Minutes from market open
            price: Current price
            volatility: Daily volatility
            
        Returns:
            ExecutionPlan with VWAP slices
        """
        if start_time is None:
            start_time = datetime.now()
        
        end_minute = start_minute + duration_minutes
        
        # Get volume-weighted slice sizes
        weights = self.volume_profile.get_slice_weights(
            n_slices, start_minute, end_minute
        )
        
        interval_minutes = duration_minutes / n_slices
        
        slices = []
        for i in range(n_slices):
            slice_size = quantity * weights[i]
            slice_time = start_time + timedelta(minutes=i * interval_minutes)
            
            slices.append(ExecutionSlice(
                slice_id=i,
                quantity=slice_size,
                target_time=slice_time
            ))
        
        # Estimate execution cost
        expected_cost_bps = self._estimate_cost(quantity, duration_minutes, n_slices, volatility)
        
        return ExecutionPlan(
            algorithm=ExecutionAlgorithm.VWAP,
            total_quantity=quantity,
            slices=slices,
            start_time=start_time,
            end_time=start_time + timedelta(minutes=duration_minutes),
            expected_cost_bps=expected_cost_bps,
            volume_profile=weights
        )
    
    def _estimate_cost(self, quantity: float, duration: float,
                       n_slices: int, volatility: float) -> float:
        """Estimate execution cost in basis points."""
        # VWAP typically has lower impact than TWAP
        timing_risk = volatility * np.sqrt(duration / 390) * 10000 * 0.4
        impact_cost = 4.0 / np.sqrt(n_slices)
        
        return timing_risk + impact_cost


# ============================================================================
# EXECUTION OPTIMIZER
# ============================================================================

class ExecutionOptimizer:
    """
    Unified Execution Optimizer.
    
    Selects optimal execution algorithm and parameters based on:
    - Order size relative to ADV
    - Urgency (time constraint)
    - Volatility
    - Benchmark requirements
    
    Usage:
        optimizer = ExecutionOptimizer()
        plan = optimizer.optimize(quantity, price, adv, volatility, urgency)
    """
    
    def __init__(self):
        """Initialize execution optimizer."""
        self.twap = TWAPExecutor()
        self.vwap = VWAPExecutor()
    
    def optimize(self,
                 quantity: float,
                 price: float,
                 adv: float,
                 volatility: float,
                 urgency: str = "MEDIUM",
                 benchmark: str = None,
                 start_time: datetime = None) -> ExecutionPlan:
        """
        Create optimized execution plan.
        
        Args:
            quantity: Order quantity
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
            urgency: "LOW", "MEDIUM", "HIGH"
            benchmark: "VWAP", "TWAP", or None
            start_time: Execution start time
            
        Returns:
            Optimized ExecutionPlan
        """
        participation = quantity / adv
        
        # Determine optimal duration based on urgency
        if urgency == "HIGH":
            duration_minutes = min(30, 390 * participation / 0.2)
        elif urgency == "LOW":
            duration_minutes = min(390, 390 * participation / 0.05)
        else:  # MEDIUM
            duration_minutes = min(180, 390 * participation / 0.1)
        
        # Determine optimal number of slices
        n_slices = self._optimal_slices(participation, duration_minutes, volatility)
        
        # Select algorithm
        if benchmark == "VWAP":
            algorithm = ExecutionAlgorithm.VWAP
        elif benchmark == "TWAP":
            algorithm = ExecutionAlgorithm.TWAP
        else:
            # Auto-select based on order characteristics
            algorithm = self._select_algorithm(participation, volatility, urgency)
        
        # Create plan
        if algorithm == ExecutionAlgorithm.VWAP:
            plan = self.vwap.create_plan(
                quantity, duration_minutes, n_slices, start_time, 
                price=price, volatility=volatility
            )
        else:
            plan = self.twap.create_plan(
                quantity, duration_minutes, n_slices, start_time,
                price=price, volatility=volatility
            )
        
        return plan
    
    def _optimal_slices(self, participation: float, duration: float, 
                        volatility: float) -> int:
        """Calculate optimal number of slices."""
        # Base: 1 slice per 10 minutes
        base_slices = max(5, int(duration / 10))
        
        # Adjust for participation rate
        if participation > 0.05:
            base_slices = int(base_slices * 1.5)
        
        # Adjust for volatility
        if volatility > 0.03:
            base_slices = int(base_slices * 1.2)
        
        return min(base_slices, 50)
    
    def _select_algorithm(self, participation: float, volatility: float,
                          urgency: str) -> ExecutionAlgorithm:
        """Auto-select best algorithm."""
        # VWAP for larger orders (better benchmark tracking)
        if participation > 0.03:
            return ExecutionAlgorithm.VWAP
        
        # TWAP for high urgency (simpler, faster)
        if urgency == "HIGH":
            return ExecutionAlgorithm.TWAP
        
        # VWAP for volatile markets
        if volatility > 0.025:
            return ExecutionAlgorithm.VWAP
        
        # Default to TWAP
        return ExecutionAlgorithm.TWAP
    
    def estimate_total_cost(self,
                            quantity: float,
                            price: float,
                            adv: float,
                            volatility: float,
                            plan: ExecutionPlan) -> Dict:
        """
        Estimate total execution cost.
        
        Returns breakdown of:
        - Market impact
        - Timing risk
        - Spread cost
        - Total cost
        """
        notional = quantity * price
        participation = quantity / adv
        
        # Market impact (Almgren-Chriss style)
        impact_bps = 10 * volatility * np.sqrt(participation) * 10000
        
        # Timing risk
        timing_bps = volatility * np.sqrt(plan.duration_minutes / 390) * 10000 * 0.5
        
        # Spread cost (assume 2 bps half-spread)
        spread_bps = 2.0
        
        total_bps = impact_bps + timing_bps + spread_bps
        
        return {
            "market_impact_bps": impact_bps,
            "timing_risk_bps": timing_bps,
            "spread_cost_bps": spread_bps,
            "total_cost_bps": total_bps,
            "total_cost_dollars": notional * total_bps / 10000
        }
    
    def generate_report(self, plan: ExecutionPlan, quantity: float, price: float) -> str:
        """Generate execution plan report."""
        notional = quantity * price
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EXECUTION OPTIMIZATION REPORT                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 ORDER DETAILS                                                            ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  Quantity: {quantity:>15,.0f} shares                                         ║
║  Price: ${price:>18,.2f}                                                     ║
║  Notional: ${notional:>16,.2f}                                               ║
║                                                                              ║
║  🎯 EXECUTION PLAN                                                           ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  Algorithm: {plan.algorithm.name:<10}                                        ║
║  Duration: {plan.duration_minutes:>6.1f} minutes                             ║
║  Slices: {plan.n_slices:>8}                                                  ║
║  Expected Cost: {plan.expected_cost_bps:>6.1f} bps                           ║
║                                                                              ║
║  📋 SLICE SCHEDULE                                                           ║
║  ────────────────────────────────────────────────────────────────────────    ║
"""
        for i, s in enumerate(plan.slices[:5]):  # Show first 5 slices
            report += f"║  Slice {i+1}: {s.quantity:>10,.0f} shares at {s.target_time.strftime('%H:%M:%S')}                    ║\n"
        
        if plan.n_slices > 5:
            report += f"║  ... and {plan.n_slices - 5} more slices                                            ║\n"
        
        report += """║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report
