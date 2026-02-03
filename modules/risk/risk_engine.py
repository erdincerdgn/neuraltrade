"""
Institutional Risk Engine
Author: Erdinc Erdogan
Purpose: Calculates VaR, CVaR, EWMA volatility, maximum drawdown, and risk-adjusted performance metrics following Basel III and RiskMetrics standards.
References:
- VaR (Historical, Parametric, Monte Carlo, Cornish-Fisher)
- CVaR / Expected Shortfall
- RiskMetrics EWMA Methodology
Usage:
    engine = RiskEngine(returns)
    var = engine.calculate_var(confidence=0.99, method=VaRMethod.PARAMETRIC)
    cvar = engine.calculate_cvar(confidence=0.99)
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import warnings

# Core imports
from ..core.base import (
    StatisticalConstants, RiskTier, MarketRegime,
    calculate_cvar, classify_risk_tier, calculate_sharpe_ratio
)


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class VaRMethod(Enum):
    """VaR calculation methods."""
    HISTORICAL = auto()
    PARAMETRIC = auto()
    MONTE_CARLO = auto()
    CORNISH_FISHER = auto()  # Adjusted for skewness/kurtosis


class VolatilityModel(Enum):
    """Volatility forecasting models."""
    SIMPLE = auto()          # Simple standard deviation
    EWMA = auto()            # Exponentially Weighted Moving Average
    GARCH = auto()           # GARCH(1,1)
    REALIZED = auto()        # Realized volatility from intraday


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var_value: float
    confidence_level: float
    method: VaRMethod
    holding_period_days: int
    portfolio_value: float
    var_percentage: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "var_value": self.var_value,
            "var_percentage": self.var_percentage,
            "confidence_level": self.confidence_level,
            "method": self.method.name,
            "holding_period_days": self.holding_period_days,
            "portfolio_value": self.portfolio_value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CVaRResult:
    """Conditional Value at Risk (Expected Shortfall) result."""
    cvar_value: float
    var_value: float
    confidence_level: float
    tail_observations: int
    expected_tail_loss: float
    risk_tier: RiskTier
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "cvar_value": self.cvar_value,
            "var_value": self.var_value,
            "expected_shortfall_pct": self.expected_tail_loss * 100,
            "confidence_level": self.confidence_level,
            "tail_observations": self.tail_observations,
            "risk_tier": self.risk_tier.name,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DrawdownResult:
    """Maximum Drawdown analysis result."""
    max_drawdown: float
    max_drawdown_pct: float
    peak_value: float
    trough_value: float
    peak_date: Optional[datetime]
    trough_date: Optional[datetime]
    recovery_date: Optional[datetime]
    drawdown_duration_days: int
    recovery_duration_days: Optional[int]
    current_drawdown: float
    current_drawdown_pct: float
    calmar_ratio: Optional[float]
    
    def to_dict(self) -> Dict:
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct * 100,
            "peak_value": self.peak_value,
            "trough_value": self.trough_value,
            "peak_date": self.peak_date.isoformat() if self.peak_date else None,
            "trough_date": self.trough_date.isoformat() if self.trough_date else None,
            "recovery_date": self.recovery_date.isoformat() if self.recovery_date else None,
            "drawdown_duration_days": self.drawdown_duration_days,
            "recovery_duration_days": self.recovery_duration_days,
            "current_drawdown_pct": self.current_drawdown_pct * 100,
            "calmar_ratio": self.calmar_ratio
        }


@dataclass
class RiskMetricsResult:
    """Comprehensive risk metrics result."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: Optional[float]
    treynor_ratio: Optional[float]
    max_drawdown: float
    volatility_annual: float
    downside_deviation: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    risk_tier: RiskTier
    
    def to_dict(self) -> Dict:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "treynor_ratio": self.treynor_ratio,
            "max_drawdown_pct": self.max_drawdown * 100,
            "volatility_annual_pct": self.volatility_annual * 100,
            "downside_deviation_pct": self.downside_deviation * 100,
            "var_95_pct": self.var_95 * 100,
            "cvar_95_pct": self.cvar_95 * 100,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "risk_tier": self.risk_tier.name
        }


# ============================================================================
# VAR CALCULATOR
# ============================================================================

class VaRCalculator:
    """
    Value at Risk Calculator.
    
    Implements three industry-standard methods:
    1. Historical VaR: Non-parametric, uses actual return distribution
    2. Parametric VaR: Assumes normal distribution
    3. Monte Carlo VaR: Simulation-based
    
    Mathematical Basis:
    - Historical: VaR_α = -Percentile(R, (1-α) × 100)
    - Parametric: VaR_α = -(μ + z_α × σ)
    - Monte Carlo: VaR_α = -Percentile(Simulated_R, (1-α) × 100)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR Calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        """
        self.confidence_level = confidence_level
        self._z_scores = {
            0.90: StatisticalConstants.Z_SCORE_90,
            0.95: StatisticalConstants.Z_SCORE_95,
            0.99: StatisticalConstants.Z_SCORE_99,
            0.999: StatisticalConstants.Z_SCORE_999
        }
    
    def calculate_historical_var(self,
                                  returns: np.ndarray,
                                  portfolio_value: float = 1.0,
                                  holding_period: int = 1) -> VaRResult:
        """
        Calculate Historical VaR.
        
        VaR_α^hist = -Percentile(R, (1-α) × 100)
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            holding_period: Holding period in days
        
        Returns:
            VaRResult with historical VaR
        """
        if len(returns) < StatisticalConstants.MIN_SAMPLES_SIGNIFICANCE:
            warnings.warn(f"Insufficient samples ({len(returns)}) for reliable VaR")
        
        # Calculate percentile
        var_pct = np.percentile(returns, (1 - self.confidence_level) * 100)
        
        # Scale for holding period (square root of time)
        var_pct_scaled = var_pct * np.sqrt(holding_period)
        
        # Convert to dollar value
        var_value = abs(var_pct_scaled) * portfolio_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            method=VaRMethod.HISTORICAL,
            holding_period_days=holding_period,
            portfolio_value=portfolio_value,
            var_percentage=abs(var_pct_scaled)
        )
    
    def calculate_parametric_var(self,
                                  returns: np.ndarray,
                                  portfolio_value: float = 1.0,
                                  holding_period: int = 1) -> VaRResult:
        """
        Calculate Parametric (Variance-Covariance) VaR.
        
        VaR_α^param = -(μ + z_α × σ)
        
        Assumes returns are normally distributed.
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            holding_period: Holding period in days
        
        Returns:
            VaRResult with parametric VaR
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        # Get z-score for confidence level
        z_score = self._z_scores.get(
            self.confidence_level,
            stats.norm.ppf(1 - self.confidence_level)
        )
        
        # Parametric VaR (negative z-score for left tail)
        var_pct = -(mu - z_score * sigma)
        
        # Scale for holding period
        var_pct_scaled = var_pct * np.sqrt(holding_period)
        
        # Convert to dollar value
        var_value = var_pct_scaled * portfolio_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            method=VaRMethod.PARAMETRIC,
            holding_period_days=holding_period,
            portfolio_value=portfolio_value,
            var_percentage=var_pct_scaled
        )
    
    def calculate_monte_carlo_var(self,
                                   returns: np.ndarray,
                                   portfolio_value: float = 1.0,
                                   holding_period: int = 1,
                                   num_simulations: int = 10000) -> VaRResult:
        """
        Calculate Monte Carlo VaR.
        
        Simulates future returns based on historical distribution parameters.
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            holding_period: Holding period in days
            num_simulations: Number of Monte Carlo simulations
        
        Returns:
            VaRResult with Monte Carlo VaR
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        # Simulate returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mu, sigma, num_simulations)
        
        # Scale for holding period
        simulated_returns_scaled = simulated_returns * np.sqrt(holding_period)
        
        # Calculate VaR from simulated distribution
        var_pct = -np.percentile(simulated_returns_scaled, (1 - self.confidence_level) * 100)
        
        # Convert to dollar value
        var_value = var_pct * portfolio_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            method=VaRMethod.MONTE_CARLO,
            holding_period_days=holding_period,
            portfolio_value=portfolio_value,
            var_percentage=var_pct
        )
    
    def calculate_cornish_fisher_var(self,
                                      returns: np.ndarray,
                                      portfolio_value: float = 1.0,
                                      holding_period: int = 1) -> VaRResult:
        """
        Calculate Cornish-Fisher VaR (adjusted for skewness and kurtosis).
        
        Adjusts the z-score for non-normal distributions:
        z_cf = z + (z² - 1)S/6 + (z³ - 3z)(K-3)/24 - (2z³ - 5z)S²/36
        
        Where S = skewness, K = kurtosis
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis
        
        # Standard z-score
        z = stats.norm.ppf(1 - self.confidence_level)
        
        # Cornish-Fisher adjustment
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        # Adjusted VaR
        var_pct = -(mu + z_cf * sigma)
        var_pct_scaled = var_pct * np.sqrt(holding_period)
        var_value = var_pct_scaled * portfolio_value
        
        return VaRResult(
            var_value=var_value,
            confidence_level=self.confidence_level,
            method=VaRMethod.CORNISH_FISHER,
            holding_period_days=holding_period,
            portfolio_value=portfolio_value,
            var_percentage=var_pct_scaled
        )
    
    def calculate_all_methods(self,
                               returns: np.ndarray,
                               portfolio_value: float = 1.0,
                               holding_period: int = 1) -> Dict[str, VaRResult]:
        """Calculate VaR using all methods for comparison."""
        return {
            "historical": self.calculate_historical_var(returns, portfolio_value, holding_period),
            "parametric": self.calculate_parametric_var(returns, portfolio_value, holding_period),
            "monte_carlo": self.calculate_monte_carlo_var(returns, portfolio_value, holding_period),
            "cornish_fisher": self.calculate_cornish_fisher_var(returns, portfolio_value, holding_period)
        }


# ============================================================================
# CVAR CALCULATOR
# ============================================================================

class CVaRCalculator:
    """
    Conditional Value at Risk (Expected Shortfall) Calculator.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    It is a coherent risk measure (subadditive) preferred by Basel III.
    
    Mathematical Basis:
    CVaR_α = E[X | X ≤ VaR_α] = (1/(1-α)) ∫_α^1 VaR_u du
    
    Discrete approximation:
    CVaR_α = -mean(R | R ≤ -VaR_α)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize CVaR Calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)
        """
        self.confidence_level = confidence_level
        self.var_calculator = VaRCalculator(confidence_level)
    
    def calculate_cvar(self,
                       returns: np.ndarray,
                       portfolio_value: float = 1.0,
                       method: VaRMethod = VaRMethod.HISTORICAL) -> CVaRResult:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        CVaR_α = -E[R | R ≤ -VaR_α]
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            method: VaR calculation method to use
        
        Returns:
            CVaRResult with CVaR and related metrics
        """
        # First calculate VaR
        if method == VaRMethod.HISTORICAL:
            var_result = self.var_calculator.calculate_historical_var(returns, portfolio_value)
        elif method == VaRMethod.PARAMETRIC:
            var_result = self.var_calculator.calculate_parametric_var(returns, portfolio_value)
        else:
            var_result = self.var_calculator.calculate_monte_carlo_var(returns, portfolio_value)
        
        var_threshold = -var_result.var_percentage
        
        # Get tail returns (returns worse than VaR)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            # No observations in tail, use VaR as CVaR
            cvar_pct = var_result.var_percentage
            tail_count = 0
        else:
            # CVaR is the mean of tail losses
            cvar_pct = abs(np.mean(tail_returns))
            tail_count = len(tail_returns)
        
        cvar_value = cvar_pct * portfolio_value
        
        # Classify risk tier
        risk_tier = classify_risk_tier(cvar_pct)
        
        return CVaRResult(
            cvar_value=cvar_value,
            var_value=var_result.var_value,
            confidence_level=self.confidence_level,
            tail_observations=tail_count,
            expected_tail_loss=cvar_pct,
            risk_tier=risk_tier
        )
    
    def calculate_component_cvar(self,
                                  returns_matrix: np.ndarray,
                                  weights: np.ndarray,
                                  portfolio_value: float = 1.0) -> Dict[str, float]:
        """
        Calculate Component CVaR for portfolio decomposition.
        
        Shows each asset's contribution to total portfolio CVaR.
        
        Args:
            returns_matrix: Matrix of asset returns (T x N)
            weights: Portfolio weights (N,)
            portfolio_value: Total portfolio value
        
        Returns:
            Dictionary with component CVaR for each asset
        """
        # Portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # Total CVaR
        total_cvar = self.calculate_cvar(portfolio_returns, portfolio_value)
        
        # Component CVaR using marginal contribution
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        tail_mask = portfolio_returns <= var_threshold
        
        component_cvar = {}
        n_assets = returns_matrix.shape[1]
        
        for i in range(n_assets):
            # Marginal contribution
            asset_tail_returns = returns_matrix[tail_mask, i]
            if len(asset_tail_returns) > 0:
                marginal_cvar = abs(np.mean(asset_tail_returns)) * weights[i]
            else:
                marginal_cvar = 0.0
            component_cvar[f"asset_{i}"] = marginal_cvar * portfolio_value
        
        component_cvar["total"] = total_cvar.cvar_value
        
        return component_cvar


# ============================================================================
# VOLATILITY FORECASTER
# ============================================================================

class VolatilityForecaster:
    """
    Volatility Forecasting Engine.
    
    Implements:
    1. Simple Historical Volatility
    2. EWMA (RiskMetrics methodology)
    3. GARCH(1,1)
    
    Mathematical Basis:
    - EWMA: σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}, λ = 0.94
    - GARCH(1,1): σ²_t = ω + αr²_{t-1} + βσ²_{t-1}
    """
    
    def __init__(self, lambda_ewma: float = None):
        """
        Initialize Volatility Forecaster.
        
        Args:
            lambda_ewma: EWMA decay factor (default: RiskMetrics 0.94)
        """
        self.lambda_ewma = lambda_ewma or StatisticalConstants.EWMA_LAMBDA_DAILY
    
    def calculate_simple_volatility(self,
                                     returns: np.ndarray,
                                     annualize: bool = True) -> float:
        """
        Calculate simple historical volatility.
        
        σ = std(R) × √252 (annualized)
        """
        vol = np.std(returns, ddof=1)
        if annualize:
            vol *= np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
        return vol
    
    def calculate_ewma_volatility(self,
                                   returns: np.ndarray,
                                   annualize: bool = True) -> Tuple[float, np.ndarray]:
        """
        Calculate EWMA volatility (RiskMetrics methodology).
        
        σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}
        
        Args:
            returns: Array of returns
            annualize: Whether to annualize the result
        
        Returns:
            Tuple of (current volatility, volatility series)
        """
        n = len(returns)
        variance_series = np.zeros(n)
        
        # Initialize with simple variance
        variance_series[0] = returns[0] ** 2
        
        # EWMA recursion
        for t in range(1, n):
            variance_series[t] = (self.lambda_ewma * variance_series[t-1] +
                                  (1 - self.lambda_ewma) * returns[t-1] ** 2)
        
        # Current volatility (last value)
        current_vol = np.sqrt(variance_series[-1])
        
        if annualize:
            current_vol *= np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
            vol_series = np.sqrt(variance_series) * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
        else:
            vol_series = np.sqrt(variance_series)
        
        return current_vol, vol_series
    
    def calculate_garch_volatility(self,
                                    returns: np.ndarray,
                                    omega: float = 0.00001,
                                    alpha: float = 0.1,
                                    beta: float = 0.85,
                                    annualize: bool = True) -> Tuple[float, np.ndarray]:
        """
        Calculate GARCH(1,1) volatility.
        
        σ²_t = ω + αr²_{t-1} + βσ²_{t-1}
        
        Default parameters approximate typical equity behavior.
        
        Args:
            returns: Array of returns
            omega: Long-run variance weight
            alpha: ARCH coefficient (reaction to shocks)
            beta: GARCH coefficient (persistence)
            annualize: Whether to annualize
        
        Returns:
            Tuple of (current volatility, volatility series)
        """
        n = len(returns)
        variance_series = np.zeros(n)
        
        # Initialize with unconditional variance
        unconditional_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else np.var(returns)
        variance_series[0] = unconditional_var
        
        # GARCH recursion
        for t in range(1, n):
            variance_series[t] = omega + alpha * returns[t-1]**2 + beta * variance_series[t-1]
        
        current_vol = np.sqrt(variance_series[-1])
        
        if annualize:
            current_vol *= np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
            vol_series = np.sqrt(variance_series) * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
        else:
            vol_series = np.sqrt(variance_series)
        
        return current_vol, vol_series
    
    def forecast_volatility(self,
                            returns: np.ndarray,
                            horizon_days: int = 1,
                            model: VolatilityModel = VolatilityModel.EWMA) -> float:
        """
        Forecast volatility for a given horizon.
        
        Args:
            returns: Historical returns
            horizon_days: Forecast horizon in days
            model: Volatility model to use
        
        Returns:
            Forecasted volatility (annualized)
        """
        if model == VolatilityModel.SIMPLE:
            base_vol = self.calculate_simple_volatility(returns, annualize=False)
        elif model == VolatilityModel.EWMA:
            base_vol, _ = self.calculate_ewma_volatility(returns, annualize=False)
        elif model == VolatilityModel.GARCH:
            base_vol, _ = self.calculate_garch_volatility(returns, annualize=False)
        else:
            base_vol = self.calculate_simple_volatility(returns, annualize=False)
        
        # Scale for horizon
        horizon_vol = base_vol * np.sqrt(horizon_days)
        
        # Annualize
        return horizon_vol * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR / horizon_days)


# ============================================================================
# DRAWDOWN ANALYZER
# ============================================================================

class DrawdownAnalyzer:
    """
    Maximum Drawdown Analyzer.
    
    Calculates peak-to-trough drawdowns with recovery analysis.
    
    Mathematical Basis:
    MDD_t = (HWM_t - V_t) / HWM_t
    Where HWM_t = max(V_s) for s ≤ t (High Water Mark)
    """
    
    def analyze_drawdowns(self,
                          values: np.ndarray,
                          dates: List[datetime] = None,
                          annual_return: float = None) -> DrawdownResult:
        """
        Analyze drawdowns from portfolio value series.
        
        Args:
            values: Array of portfolio values
            dates: Optional list of dates corresponding to values
            annual_return: Annual return for Calmar ratio calculation
        
        Returns:
            DrawdownResult with comprehensive drawdown analysis
        """
        n = len(values)
        
        if dates is None:
            dates = [datetime.now() - timedelta(days=n-i) for i in range(n)]
        
        # Calculate running maximum (High Water Mark)
        hwm = np.maximum.accumulate(values)
        
        # Drawdown series
        drawdowns = (hwm - values) / hwm
        
        # Maximum drawdown
        max_dd_idx = np.argmax(drawdowns)
        max_dd = drawdowns[max_dd_idx]
        
        # Find peak (start of max drawdown)
        peak_idx = np.argmax(values[:max_dd_idx + 1])
        peak_value = values[peak_idx]
        peak_date = dates[peak_idx]
        
        # Trough value
        trough_value = values[max_dd_idx]
        trough_date = dates[max_dd_idx]
        
        # Find recovery (if any)
        recovery_idx = None
        recovery_date = None
        recovery_duration = None
        
        for i in range(max_dd_idx + 1, n):
            if values[i] >= peak_value:
                recovery_idx = i
                recovery_date = dates[i]
                recovery_duration = (recovery_date - trough_date).days
                break
        
        # Drawdown duration
        drawdown_duration = (trough_date - peak_date).days
        
        # Current drawdown
        current_dd = drawdowns[-1]
        current_dd_value = hwm[-1] - values[-1]
        
        # Calmar ratio
        calmar = None
        if annual_return is not None and max_dd > 0:
            calmar = annual_return / max_dd
        
        return DrawdownResult(
            max_drawdown=peak_value - trough_value,
            max_drawdown_pct=max_dd,
            peak_value=peak_value,
            trough_value=trough_value,
            peak_date=peak_date,
            trough_date=trough_date,
            recovery_date=recovery_date,
            drawdown_duration_days=drawdown_duration,
            recovery_duration_days=recovery_duration,
            current_drawdown=current_dd_value,
            current_drawdown_pct=current_dd,
            calmar_ratio=calmar
        )
    
    def get_drawdown_series(self, values: np.ndarray) -> np.ndarray:
        """Get the full drawdown series."""
        hwm = np.maximum.accumulate(values)
        return (hwm - values) / hwm
    
    def get_underwater_periods(self,
                                values: np.ndarray,
                                dates: List[datetime] = None) -> List[Dict]:
        """
        Get all underwater periods (drawdown > 0).
        
        Returns list of drawdown periods with start, end, depth.
        """
        drawdowns = self.get_drawdown_series(values)
        n = len(drawdowns)
        
        if dates is None:
            dates = [datetime.now() - timedelta(days=n-i) for i in range(n)]
        
        periods = []
        in_drawdown = False
        start_idx = 0
        
        for i in range(n):
            if drawdowns[i] > 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif drawdowns[i] == 0 and in_drawdown:
                in_drawdown = False
                max_dd = np.max(drawdowns[start_idx:i])
                periods.append({
                    "start_date": dates[start_idx],
                    "end_date": dates[i],
                    "duration_days": (dates[i] - dates[start_idx]).days,
                    "max_drawdown_pct": max_dd * 100
                })
        
        # Handle ongoing drawdown
        if in_drawdown:
            max_dd = np.max(drawdowns[start_idx:])
            periods.append({
                "start_date": dates[start_idx],
                "end_date": None,
                "duration_days": (dates[-1] - dates[start_idx]).days,
                "max_drawdown_pct": max_dd * 100,
                "ongoing": True
            })
        
        return periods


# ============================================================================
# RISK METRICS ENGINE
# ============================================================================

class RiskMetricsEngine:
    """
    Comprehensive Risk Metrics Engine.
    
    Calculates all standard risk-adjusted performance metrics.
    """
    
    def __init__(self, risk_free_rate: float = None):
        """
        Initialize Risk Metrics Engine.
        
        Args:
            risk_free_rate: Annual risk-free rate (default from StatisticalConstants)
        """
        self.risk_free_rate = risk_free_rate or StatisticalConstants.RISK_FREE_RATE
        self.var_calculator = VaRCalculator()
        self.cvar_calculator = CVaRCalculator()
        self.vol_forecaster = VolatilityForecaster()
        self.dd_analyzer = DrawdownAnalyzer()
    
    def calculate_sortino_ratio(self,
                                 returns: np.ndarray,
                                 target_return: float = None) -> float:
        """
        Calculate Sortino Ratio.
        
        SR = (R_p - R_f) / σ_d
        
        Where σ_d is downside deviation (only negative returns).
        """
        target = target_return if target_return is not None else self.risk_free_rate / StatisticalConstants.TRADING_DAYS_YEAR
        
        # Downside returns
        downside_returns = returns[returns < target]
        
        if len(downside_returns) == 0:
            return np.inf  # No downside
        
        # Downside deviation
        downside_dev = np.sqrt(np.mean((downside_returns - target) ** 2))
        downside_dev_annual = downside_dev * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
        
        # Annualized return
        annual_return = np.mean(returns) * StatisticalConstants.TRADING_DAYS_YEAR
        
        if downside_dev_annual == 0:
            return np.inf
        
        return (annual_return - self.risk_free_rate) / downside_dev_annual
    
    def calculate_information_ratio(self,
                                     returns: np.ndarray,
                                     benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        IR = (R_p - R_b) / σ(R_p - R_b)
        
        Measures excess return per unit of tracking error.
        """
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
        
        if tracking_error == 0:
            return 0.0
        
        annual_excess = np.mean(excess_returns) * StatisticalConstants.TRADING_DAYS_YEAR
        
        return annual_excess / tracking_error
    
    def calculate_treynor_ratio(self,
                                 returns: np.ndarray,
                                 benchmark_returns: np.ndarray) -> float:
        """
        Calculate Treynor Ratio.
        
        TR = (R_p - R_f) / β
        
        Measures excess return per unit of systematic risk.
        """
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        
        if benchmark_variance == 0:
            return 0.0
        
        beta = covariance / benchmark_variance
        
        if beta == 0:
            return 0.0
        
        annual_return = np.mean(returns) * StatisticalConstants.TRADING_DAYS_YEAR
        
        return (annual_return - self.risk_free_rate) / beta
    
    def calculate_all_metrics(self,
                               returns: np.ndarray,
                               values: np.ndarray = None,
                               benchmark_returns: np.ndarray = None) -> RiskMetricsResult:
        """
        Calculate all risk metrics.
        
        Args:
            returns: Array of portfolio returns
            values: Array of portfolio values (for drawdown)
            benchmark_returns: Benchmark returns (for IR, Treynor)
        
        Returns:
            RiskMetricsResult with all metrics
        """
        # Basic metrics
        sharpe = calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = self.calculate_sortino_ratio(returns)
        
        # Volatility
        vol_annual = self.vol_forecaster.calculate_simple_volatility(returns)
        
        # Downside deviation
        target = self.risk_free_rate / StatisticalConstants.TRADING_DAYS_YEAR
        downside_returns = returns[returns < target]
        if len(downside_returns) > 0:
            downside_dev = np.sqrt(np.mean((downside_returns - target) ** 2))
            downside_dev_annual = downside_dev * np.sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
        else:
            downside_dev_annual = 0.0
        
        # VaR and CVaR
        var_result = self.var_calculator.calculate_historical_var(returns)
        cvar_result = self.cvar_calculator.calculate_cvar(returns)
        
        # Drawdown
        if values is not None:
            dd_result = self.dd_analyzer.analyze_drawdowns(
                values,
                annual_return=np.mean(returns) * StatisticalConstants.TRADING_DAYS_YEAR
            )
            max_dd = dd_result.max_drawdown_pct
            calmar = dd_result.calmar_ratio or 0.0
        else:
            # Estimate from returns
            cumulative = np.cumprod(1 + returns)
            hwm = np.maximum.accumulate(cumulative)
            drawdowns = (hwm - cumulative) / hwm
            max_dd = np.max(drawdowns)
            annual_return = np.mean(returns) * StatisticalConstants.TRADING_DAYS_YEAR
            calmar = annual_return / max_dd if max_dd > 0 else 0.0
        
        # Benchmark-dependent metrics
        info_ratio = None
        treynor = None
        if benchmark_returns is not None:
            info_ratio = self.calculate_information_ratio(returns, benchmark_returns)
            treynor = self.calculate_treynor_ratio(returns, benchmark_returns)
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return RiskMetricsResult(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            max_drawdown=max_dd,
            volatility_annual=vol_annual,
            downside_deviation=downside_dev_annual,
            var_95=var_result.var_percentage,
            cvar_95=cvar_result.expected_tail_loss,
            skewness=skewness,
            kurtosis=kurtosis,
            risk_tier=cvar_result.risk_tier
        )
    
    def generate_risk_report(self, metrics: RiskMetricsResult) -> str:
        """Generate comprehensive risk report."""
        report = f"""
<risk_metrics_report>
📊 INSTITUTIONAL RISK METRICS REPORT
══════════════════════════════════════════════════════════════

🎯 RISK-ADJUSTED PERFORMANCE:
  • Sharpe Ratio: {metrics.sharpe_ratio:.3f}
  • Sortino Ratio: {metrics.sortino_ratio:.3f}
  • Calmar Ratio: {metrics.calmar_ratio:.3f}
  • Information Ratio: {metrics.information_ratio:.3f if metrics.information_ratio else 'N/A'}
  • Treynor Ratio: {metrics.treynor_ratio:.3f if metrics.treynor_ratio else 'N/A'}

📉 RISK METRICS:
  • Volatility (Annual): {metrics.volatility_annual*100:.2f}%
  • Downside Deviation: {metrics.downside_deviation*100:.2f}%
  • Maximum Drawdown: {metrics.max_drawdown*100:.2f}%
  • VaR (95%): {metrics.var_95*100:.2f}%
  • CVaR (95%): {metrics.cvar_95*100:.2f}%

📈 DISTRIBUTION:
  • Skewness: {metrics.skewness:.3f}
  • Excess Kurtosis: {metrics.kurtosis:.3f}

⚠️ RISK TIER: {metrics.risk_tier.name}

</risk_metrics_report>
"""
        return report
