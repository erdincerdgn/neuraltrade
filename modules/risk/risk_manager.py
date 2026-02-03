"""
Risk Manager with Position Sizing and Stop Loss
Author: Erdinc Erdogan
Purpose: Implements Kelly Criterion, Optimal f, ATR-based stops, trailing stops, and portfolio heat management for comprehensive position risk control.
References:
- Kelly Criterion (Kelly, 1956)
- Optimal f (Ralph Vince)
- ATR-Based Stop Loss
- Portfolio Heat Management
Usage:
    manager = RiskManager(portfolio_value=100000)
    size = manager.calculate_position_size('AAPL', method=PositionSizingMethod.KELLY_CRITERION)
    stop = manager.calculate_stop_loss('AAPL', entry_price=150, type=StopLossType.ATR_BASED)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    KELLY_CRITERION = "kelly_criterion"
    FRACTIONAL_KELLY = "fractional_kelly"
    OPTIMAL_F = "optimal_f"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    CORRELATION_ADJUSTED = "correlation_adjusted"


class StopLossType(Enum):
    """Stop loss types"""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    TIME_BASED = "time_based"
    CHANDELIER = "chandelier"


class RiskLimitType(Enum):
    """Risk limit types"""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_PORTFOLIO_VAR = "max_portfolio_var"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_CONCENTRATION = "max_concentration"
    MAX_LEVERAGE = "max_leverage"
    MAX_CORRELATION_EXPOSURE = "max_correlation_exposure"


@dataclass
class PositionSize:
    """Position sizing result"""
    asset_name: str
    quantity: float
    notional_value: float
    portfolio_weight: float
    risk_amount: float
    risk_percent: float
    method: str
    confidence: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class StopLossLevel:
    """Stop loss level"""
    asset_name: str
    entry_price: float
    stop_price: float
    stop_distance: float
    stop_distance_percent: float
    stop_type: str
    atr_multiplier: Optional[float] = None
    trailing_distance: Optional[float] = None


@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_type: str
    limit_value: float
    current_value: float
    utilization: float
    breached: bool
    severity: str


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    total_exposure: float
    total_risk: float
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    current_drawdown: float
    leverage: float
    concentration_hhi: float
    correlation_risk: float
    portfolio_heat: float


@dataclass
class RiskAdjustment:
    """Risk adjustment recommendation"""
    asset_name: str
    current_position: float
    recommended_position: float
    adjustment: float
    adjustment_percent: float
    reason: str
    urgency: str


@dataclass
class KellyCriterionResult:
    """Kelly Criterion calculation result"""
    full_kelly: float
    fractional_kelly: float
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expected_value: float
    recommended_fraction: float


@dataclass
class OptimalFResult:
    """Optimal f calculation result"""
    optimal_f: float
    expected_growth: float
    max_drawdown: float
    trade_history: np.ndarray
    f_values: np.ndarray
    growth_rates: np.ndarray


@dataclass
class DrawdownControl:
    """Drawdown control parameters"""
    max_drawdown_limit: float
    current_drawdown: float
    peak_equity: float
    current_equity: float
    drawdown_percent: float
    risk_reduction_factor: float
    trading_allowed: bool


class RiskManager(BaseModule):
    """
    Institutional-Grade Risk Manager.
    Implements comprehensive risk management including position sizing,
    stop loss management, risk limits, and dynamic risk adjustment.
    Mathematical Framework:
    ----------------------
    
    Kelly Criterion:
        f* = (p × b - q) / b
        where:
        - f* = optimal fraction of capital
        - p = probability of win
        - q = probability of loss (1-p)
        - b = win/loss ratio
    
    Fractional Kelly:
        f_fractional = k × f*
        where k ∈ (0, 1], typically 0.25 to 0.5
    
    Optimal f (Ralph Vince):
        f* = argmax E[ln(1 + f × R_i)]
        
        where R_i are historical returns
    
    Volatility-Based Position Sizing:
        Position = (Risk_Capital × Target_Vol) / (Price × Asset_Vol)
    ATR-Based Stop Loss:
        Stop_Distance = k × ATR_n
        
        where k is multiplier (typically 2-3)
    
    Trailing Stop:
        Stop_Price = max(Stop_Price_{t-1}, Price_t - Trailing_Distance)
    
    Portfolio Heat:
        Heat = Σ (Position_Risk_i / Total_Capital)
        Typically keep Heat < 0.20 (20%)
    
    Correlation-Adjusted Position Size:
        Adjusted_Size = Base_Size / √(1 + ρ × (n-1))
        
        where ρ is average correlation
    
    Maximum Drawdown Control:
        Risk_Reduction = max(0, 1 - Current_DD / Max_DD_Limit)
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.max_position_size: float = self.config.get('max_position_size', 0.10)
        self.max_portfolio_var: float = self.config.get('max_portfolio_var', 0.02)
        self.max_drawdown: float = self.config.get('max_drawdown', 0.20)
        self.max_leverage: float = self.config.get('max_leverage', 2.0)
        self.max_concentration: float = self.config.get('max_concentration', 0.25)
        self.max_portfolio_heat: float = self.config.get('max_portfolio_heat', 0.20)
        self.kelly_fraction: float = self.config.get('kelly_fraction', 0.25)
        self.default_stop_atr_multiplier: float = self.config.get('stop_atr_multiplier', 2.5)
    
    # =========================================================================
    # KELLY CRITERION
    # =========================================================================
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: Optional[float] = None
    ) -> KellyCriterionResult:
        """
        Calculate Kelly Criterion for position sizing.
        
        f* = (p × b - q) / b
        """
        if avg_loss <= 0:
            raise ValueError("Average loss must be positive")
        
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        full_kelly = (p * b - q) / b
        
        fraction = fraction or self.kelly_fraction
        fractional_kelly = fraction * full_kelly
        
        expected_value = p * avg_win - q * avg_loss
        
        recommended = np.clip(fractional_kelly, 0, self.max_position_size)
        
        return KellyCriterionResult(
            full_kelly=float(np.clip(full_kelly, -1, 1)),
            fractional_kelly=float(fractional_kelly),
            win_rate=float(win_rate),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            win_loss_ratio=float(b),
            expected_value=float(expected_value),
            recommended_fraction=float(recommended)
        )
    
    def calculate_kelly_from_trades(
        self,
        trade_returns: np.ndarray,
        fraction: Optional[float] = None
    ) -> KellyCriterionResult:
        """Calculate Kelly Criterion from historical trade returns."""
        trade_returns = np.asarray(trade_returns)
        
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return KellyCriterionResult(
                full_kelly=0.0, fractional_kelly=0.0, win_rate=0.0,
                avg_win=0.0, avg_loss=0.0, win_loss_ratio=0.0,
                expected_value=0.0, recommended_fraction=0.0
            )
        
        win_rate = len(wins) / len(trade_returns)
        avg_win = np.mean(wins)
        avg_loss = np.abs(np.mean(losses))
        
        return self.calculate_kelly_criterion(win_rate, avg_win, avg_loss, fraction)
    
    # =========================================================================
    # OPTIMAL F (RALPH VINCE)
    # =========================================================================
    
    def calculate_optimal_f(
        self,
        trade_returns: np.ndarray,
        n_points: int = 100
    ) -> OptimalFResult:
        """
        Calculate Optimal f using Ralph Vince's method.
        
        Maximize: E[ln(1 + f × R_i)]
        """
        trade_returns = np.asarray(trade_returns)
        
        if len(trade_returns) == 0:
            return OptimalFResult(
                optimal_f=0.0, expected_growth=0.0, max_drawdown=0.0,
                trade_history=trade_returns, f_values=np.array([]),
                growth_rates=np.array([])
            )
        
        largest_loss = np.abs(np.min(trade_returns))
        if largest_loss == 0:
            largest_loss = 1.0
        
        normalized_returns = trade_returns / largest_loss
        
        f_values = np.linspace(0.01, 1.0, n_points)
        growth_rates = np.zeros(n_points)
        
        for i, f in enumerate(f_values):
            hpr = 1 + f * normalized_returns
            if np.any(hpr <= 0):
                growth_rates[i] = -np.inf
            else:
                growth_rates[i] = np.mean(np.log(hpr))
        
        if np.all(np.isinf(growth_rates)):
            optimal_idx = 0
        else:
            optimal_idx = np.argmax(growth_rates)
        
        optimal_f = f_values[optimal_idx]
        expected_growth = growth_rates[optimal_idx]
        
        hpr_optimal = 1 + optimal_f * normalized_returns
        cumulative = np.cumprod(hpr_optimal)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.abs(np.min(drawdown))
        
        return OptimalFResult(
            optimal_f=float(optimal_f),
            expected_growth=float(expected_growth),
            max_drawdown=float(max_dd),
            trade_history=trade_returns,
            f_values=f_values,
            growth_rates=growth_rates
        )
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def calculate_position_size_fixed_fractional(
        self,
        capital: float,
        price: float,
        risk_fraction: float = 0.02
    ) -> PositionSize:
        """Fixed fractional position sizing."""
        risk_amount = capital * risk_fraction
        quantity = risk_amount / price
        notional = quantity * price
        
        return PositionSize(
            asset_name="Asset",
            quantity=float(quantity),
            notional_value=float(notional),
            portfolio_weight=float(risk_fraction),
            risk_amount=float(risk_amount),
            risk_percent=float(risk_fraction * 100),
            method=PositionSizingMethod.FIXED_FRACTIONAL.value,
            confidence=1.0
        )
    
    def calculate_position_size_volatility_based(
        self,
        capital: float,
        price: float,
        asset_volatility: float,
        target_volatility: float = 0.15,
        risk_fraction: float = 0.02
    ) -> PositionSize:
        """Volatility-based position sizing."""
        if asset_volatility <= 0:
            asset_volatility = 0.01
        
        risk_amount = capital * risk_fraction
        vol_adjustment = target_volatility / asset_volatility
        adjusted_risk = risk_amount * vol_adjustment
        
        quantity = adjusted_risk / price
        notional = quantity * price
        portfolio_weight = notional / capital
        
        return PositionSize(
            asset_name="Asset",
            quantity=float(quantity),
            notional_value=float(notional),
            portfolio_weight=float(portfolio_weight),
            risk_amount=float(adjusted_risk),
            risk_percent=float(portfolio_weight * 100),
            method=PositionSizingMethod.VOLATILITY_BASED.value,
            confidence=0.8,
            metadata={'asset_vol': asset_volatility, 'target_vol': target_volatility}
        )
    
    def calculate_position_size_kelly(
        self,
        capital: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: Optional[float] = None
    ) -> PositionSize:
        """Kelly Criterion position sizing."""
        kelly_result = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss, fraction)
        
        risk_amount = capital * kelly_result.recommended_fraction
        quantity = risk_amount / price
        notional = quantity * price
        
        return PositionSize(
            asset_name="Asset",
            quantity=float(quantity),
            notional_value=float(notional),
            portfolio_weight=float(kelly_result.recommended_fraction),
            risk_amount=float(risk_amount),
            risk_percent=float(kelly_result.recommended_fraction * 100),
            method=PositionSizingMethod.KELLY_CRITERION.value,
            confidence=float(kelly_result.win_rate),
            metadata={'kelly_result': kelly_result}
        )
    
    def calculate_position_size_correlation_adjusted(
        self,
        capital: float,
        price: float,
        base_risk_fraction: float,
        correlation_matrix: np.ndarray,
        existing_positions: np.ndarray
    ) -> PositionSize:
        """Correlation-adjusted position sizing."""
        correlation_matrix = np.asarray(correlation_matrix)
        existing_positions = np.asarray(existing_positions)
        
        if len(existing_positions) > 0 and np.sum(existing_positions) > 0:
            weights = existing_positions / np.sum(existing_positions)
            avg_correlation = np.sum(correlation_matrix[-1, :-1] * weights)
            n_positions = len(existing_positions)
            adjustment = 1.0 / np.sqrt(1 + avg_correlation * (n_positions - 1))
        else:
            adjustment = 1.0
            avg_correlation = 0.0
        
        adjusted_risk = base_risk_fraction * adjustment
        risk_amount = capital * adjusted_risk
        quantity = risk_amount / price
        notional = quantity * price
        
        return PositionSize(
            asset_name="Asset",
            quantity=float(quantity),
            notional_value=float(notional),
            portfolio_weight=float(adjusted_risk),
            risk_amount=float(risk_amount),
            risk_percent=float(adjusted_risk * 100),
            method=PositionSizingMethod.CORRELATION_ADJUSTED.value,
            confidence=0.9,
            metadata={'avg_correlation': float(avg_correlation), 'adjustment': float(adjustment)}
        )
    
    def calculate_position_size_risk_parity(
        self,
        capital: float,
        prices: np.ndarray,
        volatilities: np.ndarray,
        target_risk: float = 0.10
    ) -> List[PositionSize]:
        """Risk parity position sizing."""
        prices = np.asarray(prices)
        volatilities = np.asarray(volatilities)
        n_assets = len(prices)
        
        risk_per_asset = target_risk / n_assets
        
        positions = []
        for i in range(n_assets):
            if volatilities[i] > 0:
                notional = (capital * risk_per_asset) / volatilities[i]
                quantity = notional / prices[i]
            else:
                quantity = 0
                notional = 0
            
            positions.append(PositionSize(
                asset_name=f"Asset_{i}",
                quantity=float(quantity),
                notional_value=float(notional),
                portfolio_weight=float(notional / capital),
                risk_amount=float(capital * risk_per_asset),
                risk_percent=float(risk_per_asset * 100),
                method=PositionSizingMethod.RISK_PARITY.value,
                confidence=0.85,
                metadata={'volatility': float(volatilities[i])}
            ))
        
        return positions
    
    # =========================================================================
    # STOP LOSS MANAGEMENT
    # =========================================================================
    
    def calculate_stop_loss_fixed_percent(
        self,
        entry_price: float,
        stop_percent: float = 0.02,
        is_long: bool = True
    ) -> StopLossLevel:
        """Fixed percentage stop loss."""
        if is_long:
            stop_price = entry_price * (1 - stop_percent)
        else:
            stop_price = entry_price * (1 + stop_percent)
        
        stop_distance = abs(entry_price - stop_price)
        
        return StopLossLevel(
            asset_name="Asset",
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            stop_distance=float(stop_distance),
            stop_distance_percent=float(stop_percent * 100),
            stop_type=StopLossType.FIXED_PERCENT.value
        )
    
    def calculate_stop_loss_atr_based(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: Optional[float] = None,
        is_long: bool = True
    ) -> StopLossLevel:
        """ATR-based stop loss."""
        atr_multiplier = atr_multiplier or self.default_stop_atr_multiplier
        
        stop_distance = atr_multiplier * atr
        
        if is_long:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        stop_distance_percent = (stop_distance / entry_price) * 100
        
        return StopLossLevel(
            asset_name="Asset",
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            stop_distance=float(stop_distance),
            stop_distance_percent=float(stop_distance_percent),
            stop_type=StopLossType.ATR_BASED.value,
            atr_multiplier=float(atr_multiplier)
        )
    
    def calculate_stop_loss_volatility_based(
        self,
        entry_price: float,
        volatility: float,
        vol_multiplier: float = 2.0,
        is_long: bool = True
    ) -> StopLossLevel:
        """Volatility-based stop loss."""
        stop_distance = vol_multiplier * volatility * entry_price
        
        if is_long:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        stop_distance_percent = (stop_distance / entry_price) * 100
        
        return StopLossLevel(
            asset_name="Asset",
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            stop_distance=float(stop_distance),
            stop_distance_percent=float(stop_distance_percent),
            stop_type=StopLossType.VOLATILITY_BASED.value
        )
    
    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        trailing_distance: float,
        is_long: bool = True
    ) -> StopLossLevel:
        """Update trailing stop loss."""
        if is_long:
            new_stop = max(current_stop, current_price - trailing_distance)
        else:
            new_stop = min(current_stop, current_price + trailing_distance)
        
        stop_distance = abs(current_price - new_stop)
        stop_distance_percent = (stop_distance / current_price) * 100
        
        return StopLossLevel(
            asset_name="Asset",
            entry_price=float(current_price),
            stop_price=float(new_stop),
            stop_distance=float(stop_distance),
            stop_distance_percent=float(stop_distance_percent),
            stop_type=StopLossType.TRAILING.value,
            trailing_distance=float(trailing_distance)
        )
    
    def calculate_chandelier_stop(
        self,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        atr: float,
        atr_multiplier: float = 3.0,
        lookback: int = 22,
        is_long: bool = True
    ) -> StopLossLevel:
        """Chandelier Exit stop loss."""
        high_prices = np.asarray(high_prices)
        low_prices = np.asarray(low_prices)
        close_prices = np.asarray(close_prices)
        
        if len(high_prices) < lookback:
            lookback = len(high_prices)
        
        if is_long:
            highest_high = np.max(high_prices[-lookback:])
            stop_price = highest_high - atr * atr_multiplier
        else:
            lowest_low = np.min(low_prices[-lookback:])
            stop_price = lowest_low + atr * atr_multiplier
        
        current_price = close_prices[-1]
        stop_distance = abs(current_price - stop_price)
        stop_distance_percent = (stop_distance / current_price) * 100
        
        return StopLossLevel(
            asset_name="Asset",
            entry_price=float(current_price),
            stop_price=float(stop_price),
            stop_distance=float(stop_distance),
            stop_distance_percent=float(stop_distance_percent),
            stop_type=StopLossType.CHANDELIER.value,
            atr_multiplier=float(atr_multiplier)
        )
    
    # =========================================================================
    # RISK LIMITS
    # =========================================================================
    
    def check_position_size_limit(
        self,
        position_value: float,
        portfolio_value: float,
        max_position_size: Optional[float] = None
    ) -> RiskLimit:
        """Check if position size exceeds limit."""
        max_size = max_position_size or self.max_position_size
        
        position_weight = position_value / portfolio_value
        utilization = position_weight / max_size
        breached = position_weight > max_size
        
        if utilization > 1.0:
            severity = "critical"
        elif utilization > 0.9:
            severity = "high"
        elif utilization > 0.75:
            severity = "medium"
        else:
            severity = "low"
        
        return RiskLimit(
            limit_type=RiskLimitType.MAX_POSITION_SIZE.value,
            limit_value=float(max_size),
            current_value=float(position_weight),
            utilization=float(utilization),
            breached=breached,
            severity=severity
        )
    
    def check_portfolio_var_limit(
        self,
        portfolio_var: float,
        max_var: Optional[float] = None
    ) -> RiskLimit:
        """Check if portfolio VaR exceeds limit."""
        max_var = max_var or self.max_portfolio_var
        
        utilization = portfolio_var / max_var
        breached = portfolio_var > max_var
        
        if utilization > 1.0:
            severity = "critical"
        elif utilization > 0.9:
            severity = "high"
        elif utilization > 0.75:
            severity = "medium"
        else:
            severity = "low"
        
        return RiskLimit(
            limit_type=RiskLimitType.MAX_PORTFOLIO_VAR.value,
            limit_value=float(max_var),
            current_value=float(portfolio_var),
            utilization=float(utilization),
            breached=breached,
            severity=severity
        )
    
    def check_drawdown_limit(
        self,
        current_drawdown: float,
        max_drawdown: Optional[float] = None
    ) -> RiskLimit:
        """Check if drawdown exceeds limit."""
        max_dd = max_drawdown or self.max_drawdown
        
        utilization = current_drawdown / max_dd
        breached = current_drawdown > max_dd
        
        if utilization > 1.0:
            severity = "critical"
        elif utilization > 0.9:
            severity = "high"
        elif utilization > 0.75:
            severity = "medium"
        else:
            severity = "low"
        
        return RiskLimit(
            limit_type=RiskLimitType.MAX_DRAWDOWN.value,
            limit_value=float(max_dd),
            current_value=float(current_drawdown),
            utilization=float(utilization),
            breached=breached,
            severity=severity
        )
    
    def check_concentration_limit(
        self,
        position_weights: np.ndarray,
        max_concentration: Optional[float] = None
    ) -> RiskLimit:
        """Check concentration risk using HHI."""
        position_weights = np.asarray(position_weights)
        max_conc = max_concentration or self.max_concentration
        
        hhi = np.sum(position_weights ** 2)
        
        utilization = hhi / max_conc
        breached = hhi > max_conc
        
        if utilization > 1.0:
            severity = "critical"
        elif utilization > 0.9:
            severity = "high"
        elif utilization > 0.75:
            severity = "medium"
        else:
            severity = "low"
        
        return RiskLimit(
            limit_type=RiskLimitType.MAX_CONCENTRATION.value,
            limit_value=float(max_conc),
            current_value=float(hhi),
            utilization=float(utilization),
            breached=breached,
            severity=severity
        )
    
    def check_leverage_limit(
        self,
        total_exposure: float,
        portfolio_value: float,
        max_leverage: Optional[float] = None
    ) -> RiskLimit:
        """Check if leverage exceeds limit."""
        max_lev = max_leverage or self.max_leverage
        
        current_leverage = total_exposure / portfolio_value
        utilization = current_leverage / max_lev
        breached = current_leverage > max_lev
        
        if utilization > 1.0:
            severity = "critical"
        elif utilization > 0.9:
            severity = "high"
        elif utilization > 0.75:
            severity = "medium"
        else:
            severity = "low"
        
        return RiskLimit(
            limit_type=RiskLimitType.MAX_LEVERAGE.value,
            limit_value=float(max_lev),
            current_value=float(current_leverage),
            utilization=float(utilization),
            breached=breached,
            severity=severity
        )
    
    # =========================================================================
    # PORTFOLIO HEAT MANAGEMENT
    # =========================================================================
    
    def calculate_portfolio_heat(
        self,
        position_values: np.ndarray,
        stop_distances: np.ndarray,
        portfolio_value: float
    ) -> float:
        """Calculate portfolio heat."""
        position_values = np.asarray(position_values)
        stop_distances = np.asarray(stop_distances)
        
        position_risks = position_values * stop_distances
        total_risk = np.sum(position_risks)
        
        heat = total_risk / portfolio_value
        
        return float(heat)
    
    def check_portfolio_heat_limit(
        self,
        position_values: np.ndarray,
        stop_distances: np.ndarray,
        portfolio_value: float,
        max_heat: Optional[float] = None
    ) -> RiskLimit:
        """Check if portfolio heat exceeds limit."""
        max_heat = max_heat or self.max_portfolio_heat
        
        current_heat = self.calculate_portfolio_heat(
            position_values, stop_distances, portfolio_value
        )
        
        utilization = current_heat / max_heat
        breached = current_heat > max_heat
        
        if utilization > 1.0:
            severity = "critical"
        elif utilization > 0.9:
            severity = "high"
        elif utilization > 0.75:
            severity = "medium"
        else:
            severity = "low"
        
        return RiskLimit(
            limit_type="max_portfolio_heat",
            limit_value=float(max_heat),
            current_value=float(current_heat),
            utilization=float(utilization),
            breached=breached,
            severity=severity
        )
    
    # =========================================================================
    # DRAWDOWN CONTROL
    # =========================================================================
    
    def calculate_drawdown_control(
        self,
        peak_equity: float,
        current_equity: float,
        max_drawdown_limit: Optional[float] = None
    ) -> DrawdownControl:
        """Calculate drawdown control parameters."""
        max_dd = max_drawdown_limit or self.max_drawdown
        
        current_drawdown = (peak_equity - current_equity) / peak_equity
        drawdown_percent = current_drawdown * 100
        
        if current_drawdown >= max_dd:
            risk_reduction = 0.0
            trading_allowed = False
        else:
            risk_reduction = 1.0 - (current_drawdown / max_dd)
            trading_allowed = True
        
        return DrawdownControl(
            max_drawdown_limit=float(max_dd),
            current_drawdown=float(current_drawdown),
            peak_equity=float(peak_equity),
            current_equity=float(current_equity),
            drawdown_percent=float(drawdown_percent),
            risk_reduction_factor=float(risk_reduction),
            trading_allowed=trading_allowed
        )
    
    def adjust_position_for_drawdown(
        self,
        base_position_size: float,
        drawdown_control: DrawdownControl
    ) -> float:
        """Adjust position size based on drawdown."""
        adjusted_size = base_position_size * drawdown_control.risk_reduction_factor
        return float(adjusted_size)
    
    # =========================================================================
    # PORTFOLIO RISK METRICS
    # =========================================================================
    
    def calculate_portfolio_risk_metrics(
        self,
        position_values: np.ndarray,
        position_weights: np.ndarray,
        returns: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_value: float,
        peak_equity: float,
        stop_distances: Optional[np.ndarray] = None
    ) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        position_values = np.asarray(position_values)
        position_weights = np.asarray(position_weights)
        returns = np.asarray(returns)
        correlation_matrix = np.asarray(correlation_matrix)
        
        total_exposure = np.sum(position_values)
        leverage = total_exposure / portfolio_value
        
        cov_matrix = np.cov(returns, rowvar=False)
        portfolio_variance = position_weights @ cov_matrix @ position_weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        portfolio_returns = returns @ position_weights
        portfolio_var = np.percentile(portfolio_returns, 5)
        portfolio_cvar = np.mean(portfolio_returns[portfolio_returns <= portfolio_var])
        
        current_drawdown = (peak_equity - portfolio_value) / peak_equity
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.abs(np.min(drawdowns))
        
        hhi = np.sum(position_weights ** 2)
        
        n = len(correlation_matrix)
        if n > 1:
            correlation_risk = (np.sum(correlation_matrix) - n) / (n * (n - 1))
        else:
            correlation_risk = 0.0
        
        if stop_distances is not None:
            portfolio_heat = self.calculate_portfolio_heat(
                position_values, stop_distances, portfolio_value
            )
        else:
            portfolio_heat = 0.0
        
        return PortfolioRiskMetrics(
            total_exposure=float(total_exposure),
            total_risk=float(portfolio_std),
            portfolio_var=float(portfolio_var),
            portfolio_cvar=float(portfolio_cvar),
            max_drawdown=float(max_drawdown),
            current_drawdown=float(current_drawdown),
            leverage=float(leverage),
            concentration_hhi=float(hhi),
            correlation_risk=float(correlation_risk),
            portfolio_heat=float(portfolio_heat)
        )
    
    # =========================================================================
    # RISK ADJUSTMENT RECOMMENDATIONS
    # =========================================================================
    
    def generate_risk_adjustments(
        self,
        position_values: np.ndarray,
        position_weights: np.ndarray,
        asset_names: List[str],
        portfolio_value: float,
        risk_limits: List[RiskLimit]
    ) -> List[RiskAdjustment]:
        """Generate risk adjustment recommendations."""
        adjustments = []
        
        critical_limits = [lim for lim in risk_limits if lim.severity == "critical"]
        
        if not critical_limits:
            return adjustments
        
        for i, (value, weight, name) in enumerate(zip(position_values, position_weights, asset_names)):
            if value == 0:
                continue
            
            max_reduction = 0.0
            for limit in critical_limits:
                if limit.breached:
                    reduction = 1.0 - (1.0 / limit.utilization)
                    max_reduction = max(max_reduction, reduction)
            
            if max_reduction > 0:
                recommended_value = value * (1 - max_reduction)
                adjustment_value = value - recommended_value
                adjustment_percent = (adjustment_value / value) * 100
                
                adjustments.append(RiskAdjustment(
                    asset_name=name,
                    current_position=float(value),
                    recommended_position=float(recommended_value),
                    adjustment=float(-adjustment_value),
                    adjustment_percent=float(-adjustment_percent),
                    reason=f"Risk limit breached: {limit.limit_type}",
                    urgency="high"
                ))
        
        return adjustments
    
    # =========================================================================
    # COMPREHENSIVE RISK CHECK
    # =========================================================================
    
    def perform_comprehensive_risk_check(
        self,
        position_values: np.ndarray,
        position_weights: np.ndarray,
        returns: np.ndarray,
        correlation_matrix: np.ndarray,
        portfolio_value: float,
        peak_equity: float,
        stop_distances: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None
    ) -> Dict:
        """Perform comprehensive risk check."""
        
        metrics = self.calculate_portfolio_risk_metrics(
            position_values, position_weights, returns,
            correlation_matrix, portfolio_value, peak_equity, stop_distances
        )
        
        limits = []
        for i, (value, name) in enumerate(zip(position_values, asset_names or [])):
            if value > 0:
                limit = self.check_position_size_limit(value, portfolio_value)
                limits.append(limit)
        
        limits.append(self.check_portfolio_var_limit(abs(metrics.portfolio_var)))
        limits.append(self.check_drawdown_limit(metrics.current_drawdown))
        limits.append(self.check_concentration_limit(position_weights))
        limits.append(self.check_leverage_limit(metrics.total_exposure, portfolio_value))
        
        if stop_distances is not None:
            limits.append(self.check_portfolio_heat_limit(
                position_values, stop_distances, portfolio_value
            ))
        
        drawdown_control = self.calculate_drawdown_control(peak_equity, portfolio_value)
        
        adjustments = self.generate_risk_adjustments(
            position_values, position_weights,
            asset_names or [f"Asset_{i}" for i in range(len(position_values))],
            portfolio_value, limits
        )
        
        critical_breaches = [lim for lim in limits if lim.severity == "critical"]
        high_breaches = [lim for lim in limits if lim.severity == "high"]
        
        if critical_breaches:
            risk_status = "CRITICAL"
        elif high_breaches:
            risk_status = "HIGH"
        elif any(lim.breached for lim in limits):
            risk_status = "ELEVATED"
        else:
            risk_status = "NORMAL"
        
        return {
            'risk_status': risk_status,
            'metrics': metrics,
            'limits': limits,
            'drawdown_control': drawdown_control,
            'adjustments': adjustments,
            'trading_allowed': drawdown_control.trading_allowed
        }