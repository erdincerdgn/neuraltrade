"""
Hardened Almgren-Chriss Market Impact Model
Author: Erdinc Erdogan
Purpose: Numerically-safe implementation of Almgren-Chriss slippage model with safe math operations for production stability.
References:
- Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
- Numerical Stability in Financial Calculations
- Safe Math Operations for Edge Cases
Usage:
    model = AlmgrenChrissModelHardened(params=AlmgrenChrissParams())
    estimate = model.estimate_slippage(quantity=10000, price=100, volatility=0.02, adv=1_000_000)
"""
import numpy as np
from modules.core.safe_math import (
    safe_sqrt, safe_divide, safe_power, safe_log,
    validate_array
)
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# IMPACT TYPES
# ============================================================================

class ImpactType(Enum):
    """Types of market impact."""
    TEMPORARY = auto()     # Decays after execution
    PERMANENT = auto()     # Lasting price change
    TOTAL = auto()         # Sum of both


# ============================================================================
# SLIPPAGE RESULT
# ============================================================================

@dataclass
class SlippageEstimate:
    """Result of slippage estimation."""
    temporary_impact: float        # Temporary price impact (%)
    permanent_impact: float        # Permanent price impact (%)
    total_slippage: float          # Total slippage (%)
    slippage_cost: float           # Slippage cost in dollars
    participation_rate: float      # Order size / ADV
    execution_time_optimal: float  # Optimal execution time (minutes)
    confidence_interval: Tuple[float, float]  # 95% CI for slippage
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "temporary_impact_pct": self.temporary_impact * 100,
            "permanent_impact_pct": self.permanent_impact * 100,
            "total_slippage_pct": self.total_slippage * 100,
            "slippage_cost": self.slippage_cost,
            "participation_rate": self.participation_rate,
            "execution_time_optimal": self.execution_time_optimal,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class OptimalTrajectory:
    """Optimal execution trajectory from Almgren-Chriss."""
    time_points: np.ndarray        # Time points
    position_trajectory: np.ndarray # Remaining position at each time
    trade_rate: np.ndarray         # Trade rate at each time
    cumulative_cost: np.ndarray    # Cumulative expected cost
    expected_cost: float           # Total expected cost
    cost_variance: float           # Variance of cost
    risk_adjusted_cost: float      # E[Cost] + λ * Var[Cost]


# ============================================================================
# ALMGREN-CHRISS MODEL PARAMETERS
# ============================================================================

@dataclass
class AlmgrenChrissParams:
    """
    Parameters for Almgren-Chriss market impact model.
    
    Calibrated from empirical market microstructure research.
    """
    # Temporary impact parameters
    eta: float = 0.1              # Temporary impact coefficient
    beta: float = 0.6             # Temporary impact exponent (typically 0.5-0.7)
    
    # Permanent impact parameters
    gamma: float = 0.1            # Permanent impact coefficient
    
    # Risk aversion
    lambda_risk: float = 1e-6    # Risk aversion parameter
    
    # Market parameters
    daily_volume_fraction: float = 0.1  # Max participation rate
    
    def to_dict(self) -> Dict:
        return {
            "eta": self.eta,
            "beta": self.beta,
            "gamma": self.gamma,
            "lambda_risk": self.lambda_risk,
            "daily_volume_fraction": self.daily_volume_fraction
        }


# ============================================================================
# TEMPORARY IMPACT MODEL
# ============================================================================

class TemporaryImpactModel:
    """
    Temporary (transient) market impact model.
    
    Formula:
        ΔP_temp = η * σ * (v/V)^β
        
    Where:
        η = temporary impact coefficient
        σ = daily volatility
        v = trade rate (shares per unit time)
        V = average daily volume
        β = impact exponent (typically 0.5-0.7)
    
    The temporary impact decays after execution completes.
    """
    
    def __init__(self, eta: float = 0.1, beta: float = 0.6):
        """
        Initialize temporary impact model.
        
        Args:
            eta: Impact coefficient
            beta: Impact exponent
        """
        self.eta = eta
        self.beta = beta
    
    def calculate(self, 
                  trade_rate: float,
                  adv: float,
                  volatility: float,
                  price: float) -> float:
        """
        Calculate temporary price impact.
        
        Args:
            trade_rate: Shares traded per minute
            adv: Average daily volume
            volatility: Daily volatility
            price: Current price
            
        Returns:
            Temporary impact as fraction of price
        """
        # Normalize trade rate to daily volume
        minutes_per_day = 390
        daily_trade_rate = trade_rate * minutes_per_day
        participation = daily_trade_rate / adv
        
        # Temporary impact formula
        impact = self.eta * volatility * safe_power(participation, self.beta)
        
        return impact
    
    def calculate_cost(self,
                       quantity: float,
                       adv: float,
                       volatility: float,
                       price: float,
                       execution_time_minutes: float) -> float:
        """
        Calculate total temporary impact cost.
        
        Args:
            quantity: Total shares to trade
            adv: Average daily volume
            volatility: Daily volatility
            price: Current price
            execution_time_minutes: Execution duration
            
        Returns:
            Temporary impact cost in dollars
        """
        trade_rate = quantity / execution_time_minutes
        impact = self.calculate(trade_rate, adv, volatility, price)
        
        return quantity * price * impact


# ============================================================================
# PERMANENT IMPACT MODEL
# ============================================================================

class PermanentImpactModel:
    """
    Permanent market impact model.
    
    Formula:
        ΔP_perm = γ * σ * (Q/V)
        
    Where:
        γ = permanent impact coefficient
        σ = daily volatility
        Q = total order size
        V = average daily volume
    
    Permanent impact represents information leakage and
    does not decay after execution.
    """
    
    def __init__(self, gamma: float = 0.1):
        """
        Initialize permanent impact model.
        
        Args:
            gamma: Permanent impact coefficient
        """
        self.gamma = gamma
    
    def calculate(self,
                  quantity: float,
                  adv: float,
                  volatility: float) -> float:
        """
        Calculate permanent price impact.
        
        Args:
            quantity: Total order size
            adv: Average daily volume
            volatility: Daily volatility
            
        Returns:
            Permanent impact as fraction of price
        """
        participation = quantity / adv
        impact = self.gamma * volatility * participation
        
        return impact
    
    def calculate_cost(self,
                       quantity: float,
                       adv: float,
                       volatility: float,
                       price: float) -> float:
        """
        Calculate permanent impact cost.
        
        Args:
            quantity: Total order size
            adv: Average daily volume
            volatility: Daily volatility
            price: Current price
            
        Returns:
            Permanent impact cost in dollars
        """
        impact = self.calculate(quantity, adv, volatility)
        return quantity * price * impact


# ============================================================================
# ALMGREN-CHRISS OPTIMAL EXECUTION
# ============================================================================

class AlmgrenChrissModel:
    """
    Almgren-Chriss Optimal Execution Model.
    
    Finds the optimal trading trajectory that minimizes:
        E[Cost] + λ * Var[Cost]
    
    The optimal solution balances:
    - Trading quickly to reduce timing risk
    - Trading slowly to reduce market impact
    
    Key Results:
    - Optimal trajectory is deterministic
    - Trade rate follows sinh/cosh pattern
    - Risk-averse traders execute faster
    
    Usage:
        model = AlmgrenChrissModel()
        slippage = model.estimate_slippage(quantity, price, adv, volatility)
        trajectory = model.optimal_trajectory(quantity, price, adv, volatility, T)
    """
    
    def __init__(self, params: AlmgrenChrissParams = None):
        """
        Initialize Almgren-Chriss model.
        
        Args:
            params: Model parameters
        """
        self.params = params or AlmgrenChrissParams()
        self.temp_impact = TemporaryImpactModel(
            eta=self.params.eta,
            beta=self.params.beta
        )
        self.perm_impact = PermanentImpactModel(
            gamma=self.params.gamma
        )
    
    def estimate_slippage(self,
                          quantity: float,
                          price: float,
                          adv: float,
                          volatility: float,
                          execution_time_minutes: float = None) -> SlippageEstimate:
        """
        Estimate total slippage for an order.
        
        Args:
            quantity: Order size in shares
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
            execution_time_minutes: Execution time (optional, will optimize if None)
            
        Returns:
            SlippageEstimate with detailed breakdown
        """
        participation = quantity / adv
        
        # Optimal execution time if not specified
        if execution_time_minutes is None:
            execution_time_minutes = self._optimal_execution_time(
                quantity, adv, volatility
            )
        
        # Calculate impacts
        temp_impact = self.temp_impact.calculate(
            quantity / execution_time_minutes, adv, volatility, price
        )
        perm_impact = self.perm_impact.calculate(quantity, adv, volatility)
        
        total_slippage = temp_impact + perm_impact
        slippage_cost = quantity * price * total_slippage
        
        # Confidence interval (approximate)
        slippage_std = volatility * safe_sqrt(participation) * 0.5
        ci_low = total_slippage - 1.96 * slippage_std
        ci_high = total_slippage + 1.96 * slippage_std
        
        return SlippageEstimate(
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_slippage=total_slippage,
            slippage_cost=slippage_cost,
            participation_rate=participation,
            execution_time_optimal=execution_time_minutes,
            confidence_interval=(max(0, ci_low), ci_high)
        )
    
    def _optimal_execution_time(self,
                                quantity: float,
                                adv: float,
                                volatility: float) -> float:
        """
        Calculate optimal execution time.
        
        Balances market impact vs timing risk.
        """
        participation = quantity / adv
        
        # Heuristic: execution time proportional to sqrt of participation
        # Constrained by max participation rate
        max_participation = self.params.daily_volume_fraction
        
        if participation <= max_participation:
            # Can execute in one day
            base_time = 30  # Base execution time in minutes
            time_factor = safe_sqrt(participation / 0.01)  # Scale with size
            return min(390, base_time * time_factor)  # Cap at full day
        else:
            # Need multiple days
            days_needed = participation / max_participation
            return days_needed * 390
    
    def optimal_trajectory(self,
                           quantity: float,
                           price: float,
                           adv: float,
                           volatility: float,
                           execution_time: float,
                           n_steps: int = 20) -> OptimalTrajectory:
        """
        Calculate optimal execution trajectory.
        
        The optimal trajectory minimizes E[Cost] + λ * Var[Cost].
        
        Args:
            quantity: Total shares to trade
            price: Current price
            adv: Average daily volume
            volatility: Daily volatility
            execution_time: Total execution time in minutes
            n_steps: Number of time steps
            
        Returns:
            OptimalTrajectory with position and trade rate over time
        """
        T = execution_time
        dt = T / n_steps
        
        # Model parameters
        eta = self.params.eta
        gamma = self.params.gamma
        lambda_risk = self.params.lambda_risk
        sigma = volatility / safe_sqrt(390)  # Per-minute volatility
        
        # Almgren-Chriss kappa parameter
        # κ = sqrt(λ * σ² / η)
        kappa = safe_sqrt(lambda_risk * sigma**2 / (eta + 1e-10))
        
        # Time points
        time_points = np.linspace(0, T, n_steps + 1)
        
        # Optimal position trajectory
        # x(t) = Q * sinh(κ(T-t)) / sinh(κT)
        if kappa * T > 1e-6:
            position = quantity * np.sinh(kappa * (T - time_points)) / np.sinh(kappa * T)
        else:
            # Linear trajectory for small kappa
            position = quantity * (1 - time_points / T)
        
        # Trade rate (negative of position derivative)
        trade_rate = np.zeros(n_steps + 1)
        trade_rate[:-1] = -np.diff(position) / dt
        trade_rate[-1] = trade_rate[-2]
        
        # Cumulative cost
        cumulative_cost = np.zeros(n_steps + 1)
        for i in range(1, n_steps + 1):
            shares_traded = position[i-1] - position[i]
            temp_cost = eta * sigma * (shares_traded / dt / adv * 390) ** self.params.beta
            perm_cost = gamma * sigma * shares_traded / adv
            cumulative_cost[i] = cumulative_cost[i-1] + shares_traded * price * (temp_cost + perm_cost)
        
        # Expected cost and variance
        expected_cost = cumulative_cost[-1]
        cost_variance = (sigma * price * quantity) ** 2 * T / 390
        risk_adjusted = expected_cost + lambda_risk * cost_variance
        
        return OptimalTrajectory(
            time_points=time_points,
            position_trajectory=position,
            trade_rate=trade_rate,
            cumulative_cost=cumulative_cost,
            expected_cost=expected_cost,
            cost_variance=cost_variance,
            risk_adjusted_cost=risk_adjusted
        )
    
    def get_slippage_adjusted_price(self,
                                    price: float,
                                    quantity: float,
                                    adv: float,
                                    volatility: float,
                                    is_buy: bool) -> float:
        """
        Get price adjusted for expected slippage.
        
        Args:
            price: Current market price
            quantity: Order quantity
            adv: Average daily volume
            volatility: Daily volatility
            is_buy: True for buy orders
            
        Returns:
            Slippage-adjusted price
        """
        estimate = self.estimate_slippage(quantity, price, adv, volatility)
        
        if is_buy:
            return price * (1 + estimate.total_slippage)
        else:
            return price * (1 - estimate.total_slippage)
    
    def generate_report(self, estimate: SlippageEstimate, quantity: float, price: float) -> str:
        """Generate slippage analysis report."""
        notional = quantity * price
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ALMGREN-CHRISS SLIPPAGE ANALYSIS                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 ORDER DETAILS                                                            ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  Quantity: {quantity:>15,.0f} shares                                         ║
║  Price: ${price:>18,.2f}                                                     ║
║  Notional: ${notional:>16,.2f}                                               ║
║  Participation Rate: {estimate.participation_rate*100:>6.2f}% of ADV                             ║
║                                                                              ║
║  📉 MARKET IMPACT BREAKDOWN                                                  ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  Temporary Impact: {estimate.temporary_impact*100:>8.4f}%  (${estimate.temporary_impact*notional:>12,.2f})       ║
║  Permanent Impact: {estimate.permanent_impact*100:>8.4f}%  (${estimate.permanent_impact*notional:>12,.2f})       ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  TOTAL SLIPPAGE:   {estimate.total_slippage*100:>8.4f}%  (${estimate.slippage_cost:>12,.2f})       ║
║                                                                              ║
║  ⏱️ OPTIMAL EXECUTION                                                        ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  Optimal Time: {estimate.execution_time_optimal:>6.1f} minutes                                   ║
║  95% CI: [{estimate.confidence_interval[0]*100:.3f}%, {estimate.confidence_interval[1]*100:.3f}%]                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report
