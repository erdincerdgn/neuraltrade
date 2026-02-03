"""
Institutional Options Greeks Engine
Author: Erdinc Erdogan
Purpose: Calculates first, second, and third-order Greeks using Black-Scholes, binomial trees, and Monte Carlo methods for comprehensive risk sensitivity analysis.
References:
- Black-Scholes-Merton PDE (1973)
- Delta, Gamma, Theta, Vega, Rho (First-Order)
- Vanna, Volga, Charm, Speed, Color, Zomma (Higher-Order)
Usage:
    engine = GreeksEngine(spot=100, strike=105, expiry=0.25, vol=0.2, rate=0.05)
    greeks = engine.calculate_all_greeks(option_type=OptionType.CALL)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, newton
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timedelta


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    """Option exercise style."""
    EUROPEAN = "european"
    AMERICAN = "american"


class GreekMethod(Enum):
    """Greeks calculation method."""
    ANALYTICAL = "analytical"
    FINITE_DIFFERENCE = "finite_difference"
    MONTE_CARLO = "monte_carlo"
    BINOMIAL = "binomial"


@dataclass
class OptionContract:
    """Option contract specification."""
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    style: OptionStyle =OptionStyle.EUROPEAN
    multiplier: float = 100.0
    
    @property
    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years."""
        delta = self.expiry - datetime.now()
        return max(delta.total_seconds() / (365.25 * 24 * 3600), 0.0)


@dataclass
class Greeks:
    """Container for all Greeks values."""
    # First-Order Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Second-Order Greeks
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    vomma: float = 0.0
    # Third-Order Greeks
    speed: float = 0.0
    color: float = 0.0
    zomma: float = 0.0
    ultima: float = 0.0
    
    # Additional Greeks
    lambda_greek: float = 0.0
    epsilon: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert Greeks to dictionary."""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'vanna': self.vanna,
            'volga': self.volga,
            'charm': self.charm,
            'vomma': self.vomma,
            'speed': self.speed,
            'color': self.color,
            'zomma': self.zomma,
            'ultima': self.ultima,
            'lambda': self.lambda_greek,
            'epsilon': self.epsilon,
        }


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks."""
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    total_rho: float = 0.0
    dollar_delta: float = 0.0
    dollar_gamma: float = 0.0
    dollar_theta: float = 0.0
    dollar_vega: float = 0.0
    gamma_pnl_1pct: float = 0.0
    positions: Dict[str, Greeks] = field(default_factory=dict)


@dataclass
class GreeksSurface:
    """Greeks surface across strikes and expiries."""
    strikes: np.ndarray = field(default_factory=lambda: np.array([]))
    expiries: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    vega_surface: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# GREEKS ENGINE
# =============================================================================

class GreeksEngine:
    """
    Institutional-Grade Options Greeks Calculation Engine.
    
    Implements comprehensive Greeks calculation using multiple methods:
    - Analytical (Black-Scholes closed-form)
    - Finite Difference (numerical approximation)
    - Monte Carlo (pathwise and likelihood ratio)
    - Binomial Tree (American options)
    """
    EPSILON = 1e-8
    BUMP_SIZE_SPOT = 0.01
    BUMP_SIZE_VOL = 0.01
    BUMP_SIZE_RATE = 0.0001
    BUMP_SIZE_TIME = 1/365
    
    def __init__(
        self,
        method: GreekMethod = GreekMethod.ANALYTICAL,
        dividend_yield: float = 0.0,
        use_calendar_days: bool = True,
    ):
        """
        Initialize Greeks Engine.
        
        Args:
            method: Greeks calculation method
            dividend_yield: Continuous dividend yield
            use_calendar_days: Use calendar days (True) or trading days (False)
        """
        self.method = method
        self.dividend_yield = dividend_yield
        self.use_calendar_days = use_calendar_days
        self.trading_days_per_year = 252
        self.calendar_days_per_year = 365.25
    
    # =========================================================================
    # BLACK-SCHOLES CORE
    # =========================================================================
    
    def _d1(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculated1 parameter for Black-Scholes.
        d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        numerator = np.log(S / K) + (r - q +0.5 * sigma**2) * T
        denominator = sigma * np.sqrt(T)
        
        return numerator / denominator
    
    def _d2(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate d2 parameter for Black-Scholes.
        
        d2 =d1 - σ√T
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        return self._d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Black-Scholes option price.
        
        Call: C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        Put:  P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        discount_factor = np.exp(-r * T)
        forward_factor = np.exp(-q * T)
        
        if option_type == OptionType.CALL:
            price = S * forward_factor * stats.norm.cdf(d1) - K * discount_factor * stats.norm.cdf(d2)
        else:
            price = K * discount_factor * stats.norm.cdf(-d2) - S * forward_factor * stats.norm.cdf(-d1)
        
        return max(price, 0.0)
    
    # =========================================================================
    # FIRST-ORDER GREEKS (ANALYTICAL)
    # =========================================================================
    
    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Delta - sensitivity to underlying price.
        
        Call Delta = e^(-qT) * N(d1)
        Put Delta  = e^(-qT) * (N(d1) - 1)
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        forward_factor = np.exp(-q * T)
        
        if option_type == OptionType.CALL:
            return forward_factor * stats.norm.cdf(d1)
        else:
            return forward_factor * (stats.norm.cdf(d1) - 1)
    
    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Gamma - rate of change of Delta.
        Gamma = e^(-qT) * N'(d1) / (S * σ * √T)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        forward_factor = np.exp(-q * T)
        
        return forward_factor * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Theta - time decay (per day).
        
        Returns daily theta (divide annual by 365)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        
        forward_factor = np.exp(-q * T)
        discount_factor = np.exp(-r * T)
        
        common = -S * sigma * forward_factor * stats.norm.pdf(d1) / (2 * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            theta_annual = (
                common
                - r * K * discount_factor * stats.norm.cdf(d2)
                + q * S * forward_factor * stats.norm.cdf(d1)
            )
        else:
            theta_annual = (
                common
                + r * K * discount_factor * stats.norm.cdf(-d2)
                - q * S * forward_factor * stats.norm.cdf(-d1)
            )
        
        days_per_year = self.calendar_days_per_year if self.use_calendar_days else self.trading_days_per_year
        return theta_annual / days_per_year
    
    def vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Vega - sensitivity to volatility.
        
        Vega = S * e^(-qT) * √T * N'(d1)
        Returns vega per 1% move in volatility (divide by 100)
        """
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        forward_factor = np.exp(-q * T)
        
        return S * forward_factor * np.sqrt(T) * stats.norm.pdf(d1) / 100
    
    def rho(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Rho - sensitivity to interest rate.
        
        Call Rho = K * T * e^(-rT) * N(d2)
        Put Rho  = -K * T * e^(-rT) * N(-d2)
        Returns rho per 1% move in rates (divide by 100)
        """
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, r, sigma, q)
        discount_factor = np.exp(-r * T)
        
        if option_type == OptionType.CALL:
            return K * T * discount_factor * stats.norm.cdf(d2) / 100
        else:
            return -K * T * discount_factor * stats.norm.cdf(-d2) / 100
    
    # =========================================================================
    # SECOND-ORDER GREEKS
    # =========================================================================
    
    def vanna(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Vanna - cross-gamma between spot and vol.
        
        Vanna = ∂Delta/∂σ =∂Vega/∂S = -e^(-qT) * N'(d1) * d2/ σ
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        forward_factor = np.exp(-q * T)
        
        return -forward_factor * stats.norm.pdf(d1) * d2 / sigma
    
    def volga(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Volga (Vomma) - vega convexity.
        
        Volga = ∂Vega/∂σ = S * e^(-qT) * √T * N'(d1) * d1 * d2 / σ
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        forward_factor = np.exp(-q * T)
        
        vega_value = S * forward_factor * np.sqrt(T) * stats.norm.pdf(d1)
        return vega_value * d1 * d2 / sigma
    
    def charm(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Charm - delta decay.
        
        Charm =∂Delta/∂T
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        forward_factor = np.exp(-q * T)
        sqrt_T = np.sqrt(T)
        
        if option_type == OptionType.CALL:
            term1 = q * forward_factor * stats.norm.cdf(d1)
        else:
            term1 = q * forward_factor * stats.norm.cdf(-d1)
        
        term2 = forward_factor * stats.norm.pdf(d1) * (2 * (r - q) * T -d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        
        if option_type == OptionType.CALL:
            return -term1 - term2
        else:
            return term1 - term2
    
    # =========================================================================
    # THIRD-ORDER GREEKS
    # =========================================================================
    
    def speed(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Speed - rate of change of Gamma.
        
        Speed = ∂Gamma/∂S = -Gamma * (1 + d1/(σ*√T)) / S
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        gamma_value = self.gamma(S, K, T, r, sigma, q)
        sqrt_T = np.sqrt(T)
        
        return -gamma_value * (1 + d1 / (sigma * sqrt_T)) / S
    
    def color(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Color - gamma decay.
        
        Color = ∂Gamma/∂T
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        forward_factor = np.exp(-q * T)
        sqrt_T = np.sqrt(T)
        
        term1 = 2 * q * T +1
        term2 = d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
        
        return -forward_factor * stats.norm.pdf(d1) * (term1 + term2) / (2 * S * T * sigma * sqrt_T)
    
    def zomma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """
        Calculate Zomma - gamma sensitivity to volatility.
        
        Zomma = ∂Gamma/∂σ = Gamma * (d1*d2 - 1) / σ
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = self._d2(S, K, T, r, sigma, q)
        gamma_value = self.gamma(S, K, T, r, sigma, q)
        
        return gamma_value * (d1 * d2 - 1) / sigma
    
    # =========================================================================
    # COMPREHENSIVE GREEKS CALCULATION
    # =========================================================================
    
    def calculate_all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: Optional[float] = None,
    ) -> Greeks:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: CALL or PUT
            q: Dividend yield (uses instance default if None)
            
        Returns:
            Greeks object with all values
        """
        if q is None:
            q = self.dividend_yield
        
        price = self.black_scholes_price(S, K, T, r, sigma, option_type, q)
        
        delta_val = self.delta(S, K, T, r, sigma, option_type, q)
        gamma_val = self.gamma(S, K, T, r, sigma, q)
        theta_val = self.theta(S, K, T, r, sigma, option_type, q)
        vega_val = self.vega(S, K, T, r, sigma, q)
        rho_val = self.rho(S, K, T, r, sigma, option_type, q)
        
        vanna_val = self.vanna(S, K, T, r, sigma, q)
        volga_val = self.volga(S, K, T, r, sigma, q)
        charm_val = self.charm(S, K, T, r, sigma, option_type, q)
        speed_val = self.speed(S, K, T, r, sigma, q)
        color_val = self.color(S, K, T, r, sigma, q)
        zomma_val = self.zomma(S, K, T, r, sigma, q)
        
        lambda_val = delta_val * S / price if price > self.EPSILON else 0.0
        
        return Greeks(
            delta=delta_val,
            gamma=gamma_val,
            theta=theta_val,
            vega=vega_val,
            rho=rho_val,
            vanna=vanna_val,
            volga=volga_val,
            charm=charm_val,
            vomma=volga_val,
            speed=speed_val,
            color=color_val,
            zomma=zomma_val,
            lambda_greek=lambda_val,
        )
    
    # =========================================================================
    # IMPLIED VOLATILITY
    # =========================================================================
    
    def implied_volatility(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        q: float = 0.0,
        initial_guess: float = 0.2,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        """
        if T <= 0:
            warnings.warn("Option has expired, cannot calculate IV")
            return 0.0
        
        if option_type == OptionType.CALL:
            intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        else:
            intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        
        if price< intrinsic - self.EPSILON:
            warnings.warn("Price below intrinsic value")
            return 0.0
        
        sigma = initial_guess
        
        for _ in range(max_iterations):
            bs_price = self.black_scholes_price(S, K, T, r, sigma, option_type, q)
            vega_val = self.vega(S, K, T, r, sigma, q) *100
            
            if abs(vega_val) < self.EPSILON:
                break
            
            diff = bs_price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            sigma = sigma - diff / vega_val
            sigma = max(0.001, min(sigma, 5.0))
        
        return sigma
    
    # =========================================================================
    # PORTFOLIO GREEKS
    # =========================================================================
    
    def portfolio_greeks(
        self,
        positions: List[Dict],
        spot_prices: Dict[str, float],
        risk_free_rate: float,) -> PortfolioGreeks:
        """
        Calculate aggregated portfolio Greeks.
        
        Args:
            positions: List of position dictionaries with keys:
                - contract:OptionContract
                - quantity: int (positive for long, negative for short)
                - volatility: float
            spot_prices: Dict mapping underlying to spot price
            risk_free_rate: Risk-free rate
            
        Returns:
            PortfolioGreeks with aggregated values
        """
        result = PortfolioGreeks()
        
        for pos in positions:
            contract = pos['contract']
            quantity = pos['quantity']
            sigma = pos['volatility']
            
            S = spot_prices.get(contract.underlying, 0)
            if S <= 0:
                continue
            
            T = contract.time_to_expiry
            K = contract.strike
            multiplier = contract.multiplier
            
            greeks = self.calculate_all_greeks(
                S=S,
                K=K,
                T=T,
                r=risk_free_rate,
                sigma=sigma,
                option_type=contract.option_type,
            )
            
            position_multiplier = quantity * multiplier
            
            result.total_delta += greeks.delta * position_multiplier
            result.total_gamma += greeks.gamma * position_multiplier
            result.total_theta += greeks.theta * position_multiplier
            result.total_vega += greeks.vega * position_multiplier
            result.total_rho += greeks.rho * position_multiplier
            
            result.dollar_delta += greeks.delta * position_multiplier * S
            result.dollar_gamma += greeks.gamma * position_multiplier * S * S / 100
            result.dollar_theta += greeks.theta * position_multiplier
            result.dollar_vega += greeks.vega * position_multiplier
            
            key = f"{contract.underlying}_{contract.strike}_{contract.option_type.value}"
            result.positions[key] = greeks
        
        result.gamma_pnl_1pct = result.dollar_gamma
        return result
    
    # =========================================================================
    # GREEKS SURFACE
    # =========================================================================
    
    def generate_greeks_surface(
        self,
        S: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        strike_range: Tuple[float, float] = (0.8, 1.2),
        expiry_range: Tuple[float, float] = (0.01, 1.0),
        num_strikes: int = 21,
        num_expiries: int = 21,
        q: float = 0.0,
    ) -> GreeksSurface:
        """
        Generate Greeks surface across strikes and expiries.
        """
        strikes = np.linspace(S * strike_range[0], S * strike_range[1], num_strikes)
        expiries = np.linspace(expiry_range[0], expiry_range[1], num_expiries)
        
        delta_surface = np.zeros((num_expiries, num_strikes))
        gamma_surface = np.zeros((num_expiries, num_strikes))
        theta_surface = np.zeros((num_expiries, num_strikes))
        vega_surface = np.zeros((num_expiries, num_strikes))
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                delta_surface[i, j] = self.delta(S, K, T, r, sigma, option_type, q)
                gamma_surface[i, j] = self.gamma(S, K, T, r, sigma, q)
                theta_surface[i, j] = self.theta(S, K, T, r, sigma, option_type, q)
                vega_surface[i, j] = self.vega(S, K, T, r, sigma, q)
        
        return GreeksSurface(
            strikes=strikes,
            expiries=expiries,
            delta_surface=delta_surface,
            gamma_surface=gamma_surface,
            theta_surface=theta_surface,
            vega_surface=vega_surface,
        )


# =============================================================================
# FINITE DIFFERENCE GREEKS (NUMERICAL)
# =============================================================================

class FiniteDifferenceGreeks:
    """
    Finite Difference Greeks Calculator.
    
    Uses central difference approximation for numerical Greeks.
    Useful for exotic options without closed-form solutions.
    """
    
    def __init__(
        self,
        pricing_function,
        bump_spot: float = 0.01,
        bump_vol: float = 0.01,
        bump_rate: float = 0.0001,
        bump_time: float = 1/365,):
        """
        Initialize Finite Difference Greeks calculator.
        
        Args:
            pricing_function: Function that takes (S, K, T, r, sigma) and returns price
            bump_spot: Spot price bump size (as fraction)
            bump_vol: Volatility bump size (absolute)
            bump_rate: Rate bump size (absolute)
            bump_time: Time bump size (in years)
        """
        self.pricing_function = pricing_function
        self.bump_spot = bump_spot
        self.bump_vol = bump_vol
        self.bump_rate = bump_rate
        self.bump_time = bump_time
    
    def delta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Delta using central difference."""
        dS = S * self.bump_spot
        price_up = self.pricing_function(S +dS, K, T, r, sigma)
        price_down = self.pricing_function(S - dS, K, T, r, sigma)
        return (price_up - price_down) / (2 * dS)
    
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Gamma using central difference."""
        dS = S * self.bump_spot
        price_up = self.pricing_function(S + dS, K, T, r, sigma)
        price_mid = self.pricing_function(S, K, T, r, sigma)
        price_down = self.pricing_function(S - dS, K, T, r, sigma)
        return (price_up - 2 * price_mid + price_down) / (dS ** 2)
    
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Vega using central difference."""
        price_up = self.pricing_function(S, K, T, r, sigma + self.bump_vol)
        price_down = self.pricing_function(S, K, T, r, sigma - self.bump_vol)
        return (price_up - price_down) / (2 * self.bump_vol) /100
    
    def theta(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Theta using forward difference."""
        price_now = self.pricing_function(S, K, T, r, sigma)
        price_later = self.pricing_function(S, K, T - self.bump_time, r, sigma)
        return (price_later - price_now) / self.bump_time / 365
    
    def rho(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Rho using central difference."""
        price_up = self.pricing_function(S, K, T, r + self.bump_rate, sigma)
        price_down = self.pricing_function(S, K, T, r - self.bump_rate, sigma)
        return (price_up - price_down) / (2 * self.bump_rate) / 100


# =============================================================================
# BINOMIAL TREE GREEKS (AMERICAN OPTIONS)
# =============================================================================

class BinomialTreeGreeks:
    """
    Binomial Tree Greeks Calculator for American Options.
    
    Uses Cox-Ross-Rubinstein (CRR) binomial tree model.
    """
    
    def __init__(self, num_steps: int = 100):
        """
        Initialize Binomial Tree Greeks calculator.
        
        Args:
            num_steps: Number of time steps in the tree
        """
        self.num_steps = num_steps
    
    def _build_tree(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """Build binomial tree and return option price and tree."""
        dt = T / self.num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Build price tree
        price_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for i in range(self.num_steps + 1):
            for j in range(i + 1):
                price_tree[j, i] = S * (u ** (i - j)) * (d ** j)
        
        # Build option value tree (backward induction)
        option_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        # Terminal payoffs
        for j in range(self.num_steps + 1):
            if option_type == OptionType.CALL:
                option_tree[j, self.num_steps] = max(price_tree[j, self.num_steps] - K, 0)
            else:
                option_tree[j, self.num_steps] = max(K - price_tree[j, self.num_steps], 0)
        
        # Backward induction with early exercise
        discount = np.exp(-r * dt)
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                hold_value = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
                if option_type == OptionType.CALL:
                    exercise_value = max(price_tree[j, i] - K, 0)
                else:
                    exercise_value = max(K - price_tree[j, i], 0)
                
                option_tree[j, i] = max(hold_value, exercise_value)
        
        return option_tree[0, 0], option_tree
    
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """Calculate American option price."""
        price, _ = self._build_tree(S, K, T, r, sigma, option_type, q)
        return price
    
    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """Calculate Delta for American option."""
        dS = S * 0.01
        price_up = self.price(S + dS, K, T, r, sigma, option_type, q)
        price_down = self.price(S - dS, K, T, r, sigma, option_type, q)
        return (price_up - price_down) / (2 * dS)
    
    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> float:
        """Calculate Gamma for American option."""
        dS = S * 0.01
        price_up = self.price(S + dS, K, T, r, sigma, option_type, q)
        price_mid = self.price(S, K, T, r, sigma, option_type, q)
        price_down = self.price(S - dS, K, T, r, sigma, option_type, q)
        return (price_up - 2 * price_mid + price_down) / (dS ** 2)


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    # Enums
    'OptionType',
    'OptionStyle',
    'GreekMethod',
    # Data Classes
    'OptionContract',
    'Greeks',
    'PortfolioGreeks',
    'GreeksSurface',
    # Main Engine
    'GreeksEngine',
    # Alternative Calculators
    'FiniteDifferenceGreeks',
    'BinomialTreeGreeks',
]