"""
Volatility Surface Modeling Engine
Author: Erdinc Erdogan
Purpose: Constructs and calibrates implied volatility surfaces using SABR, SVI, and Dupire local volatility models with arbitrage-free validation.
References:
- SABR Model (Hagan et al., 2002)
- SVI Parameterization (Gatheral, 2004)
- Dupire Local Volatility (1994)
- Volatility Smile/Skew Analysis
Usage:
    surface = VolatilitySurface()
    surface.add_quotes(strikes, expiries, implied_vols)
    calibrated = surface.calibrate_sabr()
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate
from scipy import optimize
from scipy.optimize import minimize, least_squares, differential_evolution
from scipy.interpolate import RectBivariateSpline, CubicSpline, interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timedelta
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class VolatilitySurfaceType(Enum):
    """Type of volatility surface."""
    IMPLIED = "implied"
    LOCAL = "local"
    FORWARD = "forward"


class InterpolationMethod(Enum):
    """Surface interpolation method."""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    SABR = "sabr"
    SVI = "svi"
    BIVARIATE_SPLINE = "bivariate_spline"


class SmileModel(Enum):
    """Volatility smile model."""
    SABR = "sabr"
    SVI = "svi"
    POLYNOMIAL = "polynomial"
    VANNA_VOLGA = "vanna_volga"


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SurfacePoint:
    """Single point on volatility surface."""
    strike: float
    expiry: float
    volatility: float
    forward: Optional[float] = None
    option_type: Optional[OptionType] = None
    bid_vol: Optional[float] = None
    ask_vol: Optional[float] = None
    
    @property
    def moneyness(self) -> float:
        """Calculate moneyness (K/F)."""
        if self.forward and self.forward > 0:
            return self.strike / self.forward
        return 1.0
    
    @property
    def log_moneyness(self) -> float:
        """Calculate log-moneyness ln(K/F)."""
        if self.forward and self.forward > 0:
            return np.log(self.strike / self.forward)
        return 0.0


@dataclass
class SABRParameters:
    """SABR model parameters."""
    alpha: float = 0.2
    beta: float = 0.5
    rho: float = -0.3
    nu: float = 0.4
    forward: float = 100.0
    expiry: float = 1.0
    
    def validate(self) -> bool:
        """Validate SABR parameters."""
        if self.alpha <= 0:
            return False
        if not (0 <= self.beta <= 1):
            return False
        if not (-1 <= self.rho <= 1):
            return False
        if self.nu < 0:
            return False
        return True


@dataclass
class SVIParameters:
    """SVI (Stochastic Volatility Inspired) parameters."""
    a: float = 0.04
    b: float = 0.1
    rho: float = -0.3
    m: float = 0.0
    sigma: float = 0.1
    
    def validate(self) -> bool:
        """Validate SVI parameters for no-arbitrage."""
        if self.b < 0:
            return False
        if not (-1 < self.rho < 1):
            return False
        if self.sigma <= 0:
            return False
        if self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) < 0:
            return False
        return True


@dataclass
class VolatilitySmile:
    """Volatility smile for a single expiry."""
    expiry: float
    strikes: np.ndarray
    volatilities: np.ndarray
    forward: float
    atm_vol: float
    skew: float = 0.0
    convexity: float = 0.0
    sabr_params: Optional[SABRParameters] = None
    svi_params: Optional[SVIParameters] = None


@dataclass
class VolatilityTerm:
    """Volatility term structure."""
    expiries: np.ndarray
    atm_vols: np.ndarray
    forward_vols: Optional[np.ndarray] = None


@dataclass
class SurfaceCalibration:
    """Calibration results for volatility surface."""
    model: SmileModel
    parameters: Dict[float, Union[SABRParameters, SVIParameters]]
    rmse: float
    max_error: float
    calibration_time: float
    converged: bool


@dataclass
class LocalVolSurface:
    """Local volatility surface (Dupire)."""
    strikes: np.ndarray
    expiries: np.ndarray
    local_vols: np.ndarray
    implied_vols: np.ndarray


@dataclass
class ImpliedVolSurface:
    """Implied volatility surface."""
    strikes: np.ndarray
    expiries: np.ndarray
    volatilities: np.ndarray
    forwards: np.ndarray
    interpolator: Optional[Callable] = None


@dataclass
class VolatilityCone:
    """Historical volatility cone."""
    windows: np.ndarray
    percentiles: Dict[int, np.ndarray]
    current_vol: float
    percentile_rank: float


@dataclass
class SkewMetrics:
    """Volatility skew metrics."""
    expiry: float
    atm_vol: float
    skew_25d: float
    skew_10d: float
    risk_reversal_25d: float
    butterfly_25d: float
    smile_curvature: float


@dataclass
class TermStructureMetrics:
    """Term structure metrics."""
    spot_vol: float
    forward_vol_1m: float
    forward_vol_3m: float
    vol_of_vol: float
    term_slope: float
    term_curvature: float


@dataclass
class SurfaceArbitrage:
    """Arbitrage detection results."""
    has_calendar_arbitrage: bool
    has_butterfly_arbitrage: bool
    arbitrage_points: List[Tuple[float, float]]
    max_arbitrage_amount: float


# =============================================================================
# SABR MODEL
# =============================================================================

class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) Model Implementation.
    
    The SABR model is defined by:
    dF = alpha * F^beta * dW1
    dα = nu * alpha * dW2
    <dW1, dW2> = rho * dt
    """
    
    EPSILON = 1e-10
    
    @staticmethod
    def implied_volatility(
        strike: float,
        forward: float,
        expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Calculate SABR implied volatility using Hagan's approximation.
        """
        if expiry <= 0:
            return alpha
        
        F = forward
        K = strike
        T = expiry
        
        # Handle ATM case
        if abs(F - K) < SABRModel.EPSILON:
            FK_beta = F ** (1 - beta)
            term1 = alpha / FK_beta
            term2 = 1 + T * (
                ((1 - beta)**2 / 24) * (alpha**2 / FK_beta**2) +
                (rho * beta * nu * alpha) / (4 * FK_beta) +
                ((2 - 3 * rho**2) / 24) * nu**2
            )
            return term1 * term2
        
        # General case
        FK = F * K
        FK_beta = FK ** ((1 - beta) / 2)
        log_FK = np.log(F / K)
        
        # z parameter
        z = (nu / alpha) * FK_beta * log_FK
        
        # x(z) function
        sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
        x_z = np.log((sqrt_term + z - rho) / (1 - rho))
        
        if abs(x_z) < SABRModel.EPSILON:
            x_z = z * (1 + z * (rho / 2 + z * (1 - rho**2) / 6))
        
        # Numerator
        numerator = alpha
        
        # Denominator
        denom = FK_beta * (
            1 + ((1 - beta)**2 / 24) * log_FK**2 +
            ((1 - beta)**4 / 1920) * log_FK**4
        )
        
        # Correction term
        correction = 1 + T * (
            ((1 - beta)**2 / 24) * (alpha**2 / (FK_beta**2)) +
            (rho * beta * nu * alpha) / (4 * FK_beta) +
            ((2 - 3 * rho**2) / 24) * nu**2
        )
        
        sigma = (numerator / denom) * (z / x_z) * correction
        
        return max(sigma, SABRModel.EPSILON)
    
    @staticmethod
    def calibrate(
        strikes: np.ndarray,
        market_vols: np.ndarray,
        forward: float,
        expiry: float,
        beta: float = 0.5,
        initial_guess: Optional[Tuple[float, float, float]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> SABRParameters:
        """
        Calibrate SABR parameters to market volatilities.
        """
        if initial_guess is None:
            atm_idx = np.argmin(np.abs(strikes - forward))
            initial_alpha = market_vols[atm_idx] * (forward ** (1 - beta))
            initial_guess = (initial_alpha, -0.2, 0.3)
        
        if weights is None:
            weights = np.ones(len(strikes))
        
        def objective(params):
            alpha_val, rho_val, nu_val = params
            
            if alpha_val <= 0 or nu_val < 0 or abs(rho_val) >= 1:
                return 1e10
            
            model_vols = np.array([
                SABRModel.implied_volatility(K, forward, expiry, alpha_val, beta, rho_val, nu_val)
                for K in strikes
            ])
            
            errors = (model_vols - market_vols) * weights
            return np.sum(errors**2)
        
        bounds = [
            (0.001, 2.0),
            (-0.999, 0.999),
            (0.001, 2.0),
        ]
        
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        alpha_opt, rho_opt, nu_opt = result.x
        
        return SABRParameters(
            alpha=alpha_opt,
            beta=beta,
            rho=rho_opt,
            nu=nu_opt,
            forward=forward,
            expiry=expiry,
        )


# =============================================================================
# SVI MODEL
# =============================================================================

class SVIModel:
    """
    SVI (Stochastic Volatility Inspired) Model Implementation.
    
    The SVI parameterization for total implied variance:
    w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))
    """
    
    @staticmethod
    def total_variance(
        log_moneyness: float,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ) -> float:
        """Calculate total implied variance w(k) = sigma^2 * T."""
        k = log_moneyness
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    @staticmethod
    def implied_volatility(
        log_moneyness: float,
        expiry: float,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ) -> float:
        """Calculate implied volatility from SVI parameters."""
        w = SVIModel.total_variance(log_moneyness, a, b, rho, m, sigma)
        if w <= 0 or expiry <= 0:
            return 0.0
        return np.sqrt(w / expiry)
    
    @staticmethod
    def calibrate(
        log_moneyness: np.ndarray,
        market_vols: np.ndarray,
        expiry: float,
        initial_guess: Optional[Tuple[float, float, float, float, float]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> SVIParameters:
        """Calibrate SVI parameters to market volatilities."""
        market_variance = market_vols**2 * expiry
        
        if initial_guess is None:
            atm_var = np.interp(0, log_moneyness, market_variance)
            initial_guess = (atm_var, 0.1, -0.2, 0.0, 0.1)
        
        if weights is None:
            weights = np.ones(len(log_moneyness))
        
        def objective(params):
            a_val, b_val, rho_val, m_val, sigma_val = params
            
            if b_val < 0 or sigma_val <= 0 or abs(rho_val) >= 1:
                return 1e10
            
            if a_val + b_val * sigma_val * np.sqrt(1 - rho_val**2) < 0:
                return 1e10
            
            model_variance = np.array([
                SVIModel.total_variance(k, a_val, b_val, rho_val, m_val, sigma_val)
                for k in log_moneyness
            ])
            
            errors = (model_variance - market_variance) * weights
            return np.sum(errors**2)
        
        bounds = [
            (-0.5, 0.5),
            (0.001, 1.0),
            (-0.999, 0.999),
            (-0.5, 0.5),
            (0.001, 1.0),
        ]
        
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        a_opt, b_opt, rho_opt, m_opt, sigma_opt = result.x
        
        return SVIParameters(a=a_opt, b=b_opt, rho=rho_opt, m=m_opt, sigma=sigma_opt)


# =============================================================================
# VOLATILITY SURFACE ENGINE
# =============================================================================

class VolatilitySurface:
    """
    Institutional-Grade Volatility Surface Engine.
    
    Features:
    - Multi-model calibration (SABR, SVI)
    - Arbitrage-free surface construction
    - Local volatility extraction (Dupire)
    - Surface interpolation and extrapolation
    - Skew and term structure analysis
    - Real-time surface updates
    """
    
    def __init__(
        self,
        spot: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        interpolation_method: InterpolationMethod = InterpolationMethod.SABR,
    ):
        """Initialize Volatility Surface Engine."""
        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.interpolation_method = interpolation_method
        
        self.market_data: List[SurfacePoint] = []
        self.smiles: Dict[float, VolatilitySmile] = {}
        self.calibration: Optional[SurfaceCalibration] = None
        self.implied_surface: Optional[ImpliedVolSurface] = None
        self.local_surface: Optional[LocalVolSurface] = None
    
    def forward_price(self, expiry: float) -> float:
        """Calculate forward price for given expiry."""
        return self.spot * np.exp((self.risk_free_rate - self.dividend_yield) * expiry)
    
    def add_market_data(self, points: List[SurfacePoint]) -> None:
        """Add market data points to the surface."""
        self.market_data.extend(points)
        self._organize_by_expiry()
    
    def _organize_by_expiry(self) -> None:
        """Organize market data by expiry into smiles."""
        expiry_data: Dict[float, List[SurfacePoint]] = {}
        
        for point in self.market_data:
            if point.expiry not in expiry_data:
                expiry_data[point.expiry] = []
            expiry_data[point.expiry].append(point)
        
        for expiry, points in expiry_data.items():
            points.sort(key=lambda x: x.strike)
            strikes = np.array([p.strike for p in points])
            vols = np.array([p.volatility for p in points])
            forward = self.forward_price(expiry)
            
            atm_idx = np.argmin(np.abs(strikes - forward))
            atm_vol = vols[atm_idx]
            
            if len(strikes) >= 3:
                skew = (vols[0] - vols[-1]) / (strikes[-1] - strikes[0]) * forward
            else:
                skew = 0.0
            
            self.smiles[expiry] = VolatilitySmile(
                expiry=expiry,
                strikes=strikes,
                volatilities=vols,
                forward=forward,
                atm_vol=atm_vol,
                skew=skew,
            )
    
    def calibrate_sabr(
        self,
        beta: float = 0.5,
        expiries: Optional[List[float]] = None,
    ) -> SurfaceCalibration:
        """Calibrate SABR model to all expiries."""
        start_time = time.time()
        
        if expiries is None:
            expiries = list(self.smiles.keys())
        
        parameters: Dict[float, SABRParameters] = {}
        total_error = 0.0
        max_error = 0.0
        
        for expiry in expiries:
            if expiry not in self.smiles:
                continue
            
            smile = self.smiles[expiry]
            
            sabr_params = SABRModel.calibrate(
                strikes=smile.strikes,
                market_vols=smile.volatilities,
                forward=smile.forward,
                expiry=expiry,
                beta=beta,
            )
            
            parameters[expiry] = sabr_params
            smile.sabr_params = sabr_params
            
            model_vols = np.array([
                SABRModel.implied_volatility(
                    K, smile.forward, expiry,
                    sabr_params.alpha, sabr_params.beta,
                    sabr_params.rho, sabr_params.nu
                )
                for K in smile.strikes
            ])
            
            errors = np.abs(model_vols - smile.volatilities)
            total_error += np.sum(errors**2)
            max_error = max(max_error, np.max(errors))
        
        num_points = sum(len(self.smiles[e].strikes) for e in expiries if e in self.smiles)
        rmse = np.sqrt(total_error / num_points) if num_points > 0 else 0.0
        
        self.calibration = SurfaceCalibration(
            model=SmileModel.SABR,
            parameters=parameters,
            rmse=rmse,
            max_error=max_error,
            calibration_time=time.time() - start_time,
            converged=True,
        )
        
        return self.calibration
    
    def calibrate_svi(
        self,
        expiries: Optional[List[float]] = None,
    ) -> SurfaceCalibration:
        """Calibrate SVI model to all expiries."""
        start_time = time.time()
        
        if expiries is None:
            expiries = list(self.smiles.keys())
        
        parameters: Dict[float, SVIParameters] = {}
        total_error = 0.0
        max_error = 0.0
        
        for expiry in expiries:
            if expiry not in self.smiles:
                continue
            
            smile = self.smiles[expiry]
            log_money = np.log(smile.strikes / smile.forward)
            
            svi_params = SVIModel.calibrate(
                log_moneyness=log_money,
                market_vols=smile.volatilities,
                expiry=expiry,
            )
            
            parameters[expiry] = svi_params
            smile.svi_params = svi_params
            
            model_vols = np.array([
                SVIModel.implied_volatility(
                    k, expiry, svi_params.a, svi_params.b,
                    svi_params.rho, svi_params.m, svi_params.sigma
                )
                for k in log_money
            ])
            
            errors = np.abs(model_vols - smile.volatilities)
            total_error += np.sum(errors**2)
            max_error = max(max_error, np.max(errors))
        
        num_points = sum(len(self.smiles[e].strikes) for e in expiries if e in self.smiles)
        rmse = np.sqrt(total_error / num_points) if num_points > 0 else 0.0
        
        self.calibration = SurfaceCalibration(
            model=SmileModel.SVI,
            parameters=parameters,
            rmse=rmse,
            max_error=max_error,
            calibration_time=time.time() - start_time,
            converged=True,
        )
        
        return self.calibration
    
    def get_volatility(
        self,
        strike: float,
        expiry: float,
    ) -> float:
        """Get interpolated volatility for any strike/expiry."""
        if not self.calibration:
            raise ValueError("Surface not calibrated. Call calibrate_sabr() or calibrate_svi() first.")
        
        forward = self.forward_price(expiry)
        expiries_list = sorted(self.calibration.parameters.keys())
        
        if expiry <= expiries_list[0]:
            params = self.calibration.parameters[expiries_list[0]]
            return self._get_vol_from_params(strike, forward, expiry, params)
        
        if expiry >= expiries_list[-1]:
            params = self.calibration.parameters[expiries_list[-1]]
            return self._get_vol_from_params(strike, forward, expiry, params)
        
        for i in range(len(expiries_list) - 1):
            if expiries_list[i] <= expiry <= expiries_list[i + 1]:
                t1, t2 = expiries_list[i], expiries_list[i + 1]
                params1 = self.calibration.parameters[t1]
                params2 = self.calibration.parameters[t2]
                
                vol1 = self._get_vol_from_params(strike, forward, t1, params1)
                vol2 = self._get_vol_from_params(strike, forward, t2, params2)
                
                var1 = vol1**2 * t1
                var2 = vol2**2 * t2
                
                weight = (expiry - t1) / (t2 - t1)
                var_interp = var1 * (1 - weight) + var2 * weight
                
                return np.sqrt(var_interp / expiry)
        
        return 0.0
    
    def _get_vol_from_params(
        self,
        strike: float,
        forward: float,
        expiry: float,
        params: Union[SABRParameters, SVIParameters],
    ) -> float:
        """Get volatility from calibrated parameters."""
        if isinstance(params, SABRParameters):
            return SABRModel.implied_volatility(
                strike, forward, expiry,
                params.alpha, params.beta, params.rho, params.nu
            )
        elif isinstance(params, SVIParameters):
            log_money = np.log(strike / forward)
            return SVIModel.implied_volatility(
                log_money, expiry,
                params.a, params.b, params.rho, params.m, params.sigma
            )
        return 0.0
    
    def build_surface_grid(
        self,
        strike_range: Tuple[float, float] = (0.7, 1.3),
        expiry_range: Tuple[float, float] = (0.01, 2.0),
        num_strikes: int = 51,
        num_expiries: int = 25,
    ) -> ImpliedVolSurface:
        """Build full volatility surface on a grid."""
        strikes = np.linspace(
            self.spot * strike_range[0],
            self.spot * strike_range[1],
            num_strikes
        )
        expiries_arr = np.linspace(expiry_range[0], expiry_range[1], num_expiries)
        
        volatilities = np.zeros((num_expiries, num_strikes))
        forwards = np.array([self.forward_price(t) for t in expiries_arr])
        
        for i, t in enumerate(expiries_arr):
            for j, k in enumerate(strikes):
                try:
                    volatilities[i, j] = self.get_volatility(k, t)
                except Exception:
                    volatilities[i, j] = np.nan
        
        valid_mask = ~np.isnan(volatilities)
        if np.any(valid_mask):
            interp_func = RectBivariateSpline(expiries_arr, strikes, volatilities)
        else:
            interp_func = None
        
        self.implied_surface = ImpliedVolSurface(
            strikes=strikes,
            expiries=expiries_arr,
            volatilities=volatilities,
            forwards=forwards,
            interpolator=interp_func,
        )
        
        return self.implied_surface
    
    def calculate_local_volatility(
        self,
        strike_range: Tuple[float, float] = (0.7, 1.3),
        expiry_range: Tuple[float, float] = (0.01, 2.0),
        num_strikes: int = 51,
        num_expiries: int = 25,
    ) -> LocalVolSurface:
        """
        Calculate Dupire local volatility surface.
        """
        if self.implied_surface is None:
            self.build_surface_grid(strike_range, expiry_range, num_strikes, num_expiries)
        
        strikes = self.implied_surface.strikes
        expiries_arr = self.implied_surface.expiries
        impl_vols = self.implied_surface.volatilities
        
        local_vols = np.zeros_like(impl_vols)
        
        dK = strikes[1] - strikes[0] if len(strikes) > 1 else 1.0
        dT = expiries_arr[1] - expiries_arr[0] if len(expiries_arr) > 1 else 0.01
        
        for i in range(1, len(expiries_arr) - 1):
            for j in range(1, len(strikes) - 1):
                T = expiries_arr[i]
                K = strikes[j]
                sigma = impl_vols[i, j]
                
                if np.isnan(sigma) or sigma <= 0:
                    local_vols[i, j] = np.nan
                    continue
                
                # Numerical derivatives of implied vol
                dsigma_dT = (impl_vols[i + 1, j] - impl_vols[i - 1, j]) / (2 * dT)
                dsigma_dK = (impl_vols[i, j + 1] - impl_vols[i, j - 1]) / (2 * dK)
                d2sigma_dK2 = (impl_vols[i, j + 1] - 2 * impl_vols[i, j] + impl_vols[i, j - 1]) / (dK**2)
                
                # Total variance w = sigma^2 * T
                w = sigma**2 * T
                
                # dw/dT = 2*sigma*T*(dsigma/dT) + sigma^2
                dw_dT = 2 * sigma * T * dsigma_dT + sigma**2
                
                # dw/dK = 2*sigma*T*(dsigma/dK)
                dw_dK = 2 * sigma * T * dsigma_dK
                
                # d2w/dK2 = 2*T*[(dsigma/dK)^2 + sigma*(d2sigma/dK2)]
                d2w_dK2 = 2 * T * (dsigma_dK**2 + sigma * d2sigma_dK2)
                
                # Dupire formula
                numerator = dw_dT
                
                term1 = 1.0
                if w > 0:
                    term2 = -(K / w) * dw_dK
                    term3 = 0.25 * (-0.25 - 1/w + K**2/w**2) * dw_dK**2
                else:
                    term2 = 0.0
                    term3 = 0.0
                term4 = 0.5 * d2w_dK2
                
                denominator = term1 + term2 + term3 + term4
                
                if denominator > 0 and numerator > 0:
                    local_vols[i, j] = np.sqrt(numerator / denominator)
                else:
                    local_vols[i, j] = sigma
        
        # Fill boundaries
        local_vols[0, :] = local_vols[1, :]
        local_vols[-1, :] = local_vols[-2, :]
        local_vols[:, 0] = local_vols[:, 1]
        local_vols[:, -1] = local_vols[:, -2]
        
        self.local_surface = LocalVolSurface(
            strikes=strikes,
            expiries=expiries_arr,
            local_vols=local_vols,
            implied_vols=impl_vols,
        )
        
        return self.local_surface
    
    def calculate_skew_metrics(self, expiry: float) -> SkewMetrics:
        """Calculate volatility skew metrics for a given expiry."""
        if expiry not in self.smiles:
            raise ValueError(f"No smile data for expiry {expiry}")
        
        smile = self.smiles[expiry]
        forward = smile.forward
        atm_vol = smile.atm_vol
        
        # Calculate delta-based strikes (approximate)
        sqrt_t = np.sqrt(expiry)
        
        # 25-delta strikes
        strike_25d_call = forward * np.exp(0.5 * atm_vol**2 * expiry + atm_vol * sqrt_t * stats.norm.ppf(0.75))
        strike_25d_put = forward * np.exp(0.5 * atm_vol**2 * expiry + atm_vol * sqrt_t * stats.norm.ppf(0.25))
        
        # 10-delta strikes
        strike_10d_call = forward * np.exp(0.5 * atm_vol**2 * expiry + atm_vol * sqrt_t * stats.norm.ppf(0.90))
        strike_10d_put = forward * np.exp(0.5 * atm_vol**2 * expiry + atm_vol * sqrt_t * stats.norm.ppf(0.10))
        
        # Interpolate vols at delta strikes
        vol_25d_call = np.interp(strike_25d_call, smile.strikes, smile.volatilities)
        vol_25d_put = np.interp(strike_25d_put, smile.strikes, smile.volatilities)
        vol_10d_call = np.interp(strike_10d_call, smile.strikes, smile.volatilities)
        vol_10d_put = np.interp(strike_10d_put, smile.strikes, smile.volatilities)
        
        # Skew metrics
        skew_25d = vol_25d_put - vol_25d_call
        skew_10d = vol_10d_put - vol_10d_call
        risk_reversal_25d = vol_25d_call - vol_25d_put
        butterfly_25d = 0.5 * (vol_25d_call + vol_25d_put) - atm_vol
        
        # Smile curvature (second derivative at ATM)
        atm_idx = np.argmin(np.abs(smile.strikes - forward))
        if 0 < atm_idx < len(smile.strikes) - 1:
            dK = smile.strikes[atm_idx + 1] - smile.strikes[atm_idx]
            curvature = (smile.volatilities[atm_idx + 1] - 2 * smile.volatilities[atm_idx] + 
                        smile.volatilities[atm_idx - 1]) / (dK**2)
        else:
            curvature = 0.0
        
        return SkewMetrics(
            expiry=expiry,
            atm_vol=atm_vol,
            skew_25d=skew_25d,
            skew_10d=skew_10d,
            risk_reversal_25d=risk_reversal_25d,
            butterfly_25d=butterfly_25d,
            smile_curvature=curvature,
        )
    
    def calculate_term_structure(self) -> TermStructureMetrics:
        """Calculate term structure metrics."""
        expiries_list = sorted(self.smiles.keys())
        atm_vols = [self.smiles[t].atm_vol for t in expiries_list]
        
        if len(expiries_list) < 2:
            raise ValueError("Need at least 2 expiries for term structure analysis")
        
        spot_vol = atm_vols[0]
        
        def calc_forward_vol(t1: float, t2: float, v1: float, v2: float) -> float:
            """Calculate forward volatility between t1 and t2."""
            if t2 <= t1:
                return v2
            var1 = v1**2 * t1
            var2 = v2**2 * t2
            return np.sqrt((var2 - var1) / (t2 - t1))
        
        forward_vol_1m = 0.0
        forward_vol_3m = 0.0
        
        one_month = 1.0 / 12.0
        three_months = 0.25
        
        for i in range(len(expiries_list) - 1):
            if expiries_list[i] <= one_month < expiries_list[i + 1]:
                forward_vol_1m = calc_forward_vol(expiries_list[i], expiries_list[i + 1], atm_vols[i], atm_vols[i + 1])
            if expiries_list[i] <= three_months < expiries_list[i + 1]:
                forward_vol_3m = calc_forward_vol(expiries_list[i], expiries_list[i + 1], atm_vols[i], atm_vols[i + 1])
        
        vol_of_vol = np.std(atm_vols)
        
        if len(expiries_list) >= 2:
            slope, intercept = np.polyfit(expiries_list, atm_vols, 1)
        else:
            slope = 0.0
        
        if len(expiries_list) >= 3:
            coeffs = np.polyfit(expiries_list, atm_vols, 2)
            curvature = coeffs[0]
        else:
            curvature = 0.0
        
        return TermStructureMetrics(
            spot_vol=spot_vol,
            forward_vol_1m=forward_vol_1m,
            forward_vol_3m=forward_vol_3m,
            vol_of_vol=vol_of_vol,
            term_slope=slope,
            term_curvature=curvature,
        )
    
    def check_arbitrage(self) -> SurfaceArbitrage:
        """
        Check for arbitrage violations in the surface.
        """
        arbitrage_points: List[Tuple[float, float]] = []
        has_calendar = False
        has_butterfly = False
        max_arbitrage = 0.0
        
        expiries_list = sorted(self.smiles.keys())
        
        # Calendar arbitrage check
        for i in range(len(expiries_list) - 1):
            t1, t2 = expiries_list[i], expiries_list[i + 1]
            smile1, smile2 = self.smiles[t1], self.smiles[t2]
            
            common_strikes = np.intersect1d(
                smile1.strikes.astype(int),
                smile2.strikes.astype(int)
            ).astype(float)
            
            for K in common_strikes:
                idx1 = np.argmin(np.abs(smile1.strikes - K))
                idx2 = np.argmin(np.abs(smile2.strikes - K))
                
                var1 = smile1.volatilities[idx1]**2 * t1
                var2 = smile2.volatilities[idx2]**2 * t2
                
                if var2 < var1 - 1e-6:
                    has_calendar = True
                    arbitrage_points.append((K, t2))
                    max_arbitrage = max(max_arbitrage, var1 - var2)
        
        # Butterfly arbitrage check
        for expiry, smile in self.smiles.items():
            for i in range(1, len(smile.strikes) - 1):
                v_low = smile.volatilities[i - 1]
                v_mid = smile.volatilities[i]
                v_high = smile.volatilities[i + 1]
                
                threshold = 0.5 * (v_low + v_high) + 0.01
                if v_mid > threshold:
                    has_butterfly = True
                    arbitrage_points.append((smile.strikes[i], expiry))
                    max_arbitrage = max(max_arbitrage, v_mid - 0.5 * (v_low + v_high))
        
        return SurfaceArbitrage(
            has_calendar_arbitrage=has_calendar,
            has_butterfly_arbitrage=has_butterfly,
            arbitrage_points=arbitrage_points,
            max_arbitrage_amount=max_arbitrage,
        )


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'VolatilitySurfaceType',
    'InterpolationMethod',
    'SmileModel',
    'OptionType',
    'SurfacePoint',
    'SABRParameters',
    'SVIParameters',
    'VolatilitySmile',
    'VolatilityTerm',
    'SurfaceCalibration',
    'LocalVolSurface',
    'ImpliedVolSurface',
    'VolatilityCone',
    'SkewMetrics',
    'TermStructureMetrics',
    'SurfaceArbitrage',
    'SABRModel',
    'SVIModel',
    'VolatilitySurface',
]
