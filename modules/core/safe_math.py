"""
Safe Math Utilities - Numerical Hardening for Financial Calculations
Author: Erdinc Erdogan
Purpose: Provides centralized numerical safety guards including safe_log, safe_sqrt,
safe_divide, and array validation to prevent NaN/Inf in production calculations.
References:
- IEEE 754 Floating-Point Standard
- Numerical Stability in Scientific Computing
- Defensive Programming for Financial Systems
Usage:
    from modules.core.safe_math import safe_log, safe_divide, validate_array
    result = safe_divide(numerator, denominator, default=0.0)
    clean_array = validate_array(raw_array)
"""

import numpy as np
from typing import Union, Optional, Tuple
from functools import wraps
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS
# ============================================================================

EPS = 1e-10          # Default epsilon for numerical stability
LOG_EPS = 1e-10      # Epsilon for log operations
SQRT_EPS = 0.0       # Minimum value for sqrt (0 is safe)
EXP_MAX = 700        # Maximum exponent before overflow (e^700 ≈ 1e304)
EXP_MIN = -700       # Minimum exponent before underflow
DIV_EPS = 1e-10      # Epsilon for division operations


# ============================================================================
# CORE SAFE OPERATIONS
# ============================================================================

def safe_log(x: Union[np.ndarray, float], 
             eps: float = LOG_EPS,
             replace_neg_inf: float = -23.0) -> Union[np.ndarray, float]:
    """
    Safe logarithm that prevents -Inf and NaN.
    
    log(x) is undefined for x <= 0:
    - log(0) = -Inf
    - log(negative) = NaN
    
    Solution: Clip input to [eps, inf) before taking log.
    
    Args:
        x: Input value(s)
        eps: Minimum value to clip to (default: 1e-10)
        replace_neg_inf: Value to replace any remaining -Inf (default: -23 ≈ log(1e-10))
        
    Returns:
        Safe logarithm result
        
    Example:
        >>> safe_log(0)  # Returns -23.0 instead of -Inf
        >>> safe_log(-1)  # Returns -23.0 instead of NaN
    """
    x = np.asarray(x)
    x_clipped = np.clip(x, eps, None)
    result = np.log(x_clipped)
    
    # Handle any remaining -Inf (shouldn't happen but safety first)
    if np.isscalar(result):
        if np.isinf(result) and result < 0:
            return replace_neg_inf
    else:
        result = np.where(np.isneginf(result), replace_neg_inf, result)
    
    return result


def safe_log1p(x: Union[np.ndarray, float],
               eps: float = LOG_EPS) -> Union[np.ndarray, float]:
    """
    Safe log(1 + x) for small x values.
    
    More numerically stable than log(1 + x) for small x.
    """
    x = np.asarray(x)
    # Ensure 1 + x > 0, so x > -1
    x_safe = np.maximum(x, -1 + eps)
    return np.log1p(x_safe)


def safe_sqrt(x: Union[np.ndarray, float],
              eps: float = SQRT_EPS) -> Union[np.ndarray, float]:
    """
    Safe square root that prevents NaN on negative inputs.
    
    sqrt(x) is undefined for x < 0 (returns NaN).
    
    Solution: Clip input to [eps, inf) before taking sqrt.
    
    Args:
        x: Input value(s)
        eps: Minimum value to clip to (default: 0.0)
        
    Returns:
        Safe square root result
        
    Example:
        >>> safe_sqrt(-1)  # Returns 0.0 instead of NaN
        >>> safe_sqrt(-0.001)  # Returns 0.0 instead of NaN
    """
    x = np.asarray(x)
    x_clipped = np.maximum(x, eps)
    return np.sqrt(x_clipped)


def safe_exp(x: Union[np.ndarray, float],
             max_exp: float = EXP_MAX,
             min_exp: float = EXP_MIN) -> Union[np.ndarray, float]:
    """
    Safe exponential that prevents overflow and underflow.
    
    exp(x) overflows for x > ~709 (returns Inf).
    exp(x) underflows for x < ~-745 (returns 0, which is fine).
    
    Solution: Clip input to [min_exp, max_exp] before taking exp.
    
    Args:
        x: Input value(s)
        max_exp: Maximum exponent (default: 700)
        min_exp: Minimum exponent (default: -700)
        
    Returns:
        Safe exponential result
        
    Example:
        >>> safe_exp(1000)  # Returns ~1e304 instead of Inf
    """
    x = np.asarray(x)
    x_clipped = np.clip(x, min_exp, max_exp)
    return np.exp(x_clipped)


def safe_divide(numerator: Union[np.ndarray, float],
                denominator: Union[np.ndarray, float],
                eps: float = DIV_EPS,
                default: float = 0.0) -> Union[np.ndarray, float]:
    """
    Safe division that prevents division by zero and Inf results.
    
    a / b is undefined when b = 0 (returns Inf or NaN).
    
    Solution: Add epsilon to denominator OR return default when |b| < eps.
    
    Args:
        numerator: Dividend
        denominator: Divisor
        eps: Threshold below which denominator is considered zero
        default: Value to return when denominator is ~zero
        
    Returns:
        Safe division result
        
    Example:
        >>> safe_divide(1, 0)  # Returns 0.0 instead of Inf
        >>> safe_divide(1, 1e-15)  # Returns 0.0 instead of 1e15
    """
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    
    # Use np.where for vectorized safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.abs(denominator) > eps,
            numerator / denominator,
            default
        )
    
    # Clean up any remaining NaN/Inf
    result = np.nan_to_num(result, nan=default, posinf=default, neginf=default)
    
    return result


def safe_power(base: Union[np.ndarray, float],
               exponent: Union[np.ndarray, float],
               eps: float = EPS) -> Union[np.ndarray, float]:
    """
    Safe power operation that handles edge cases.
    
    Edge cases:
    - 0^negative = Inf
    - negative^fractional = NaN (complex)
    - large^large = Inf
    
    Args:
        base: Base value(s)
        exponent: Exponent value(s)
        eps: Minimum base value for negative exponents
        
    Returns:
        Safe power result
    """
    base = np.asarray(base)
    exponent = np.asarray(exponent)
    
    # For negative exponents, ensure base is not too small
    if np.any(exponent < 0):
        base = np.where(exponent < 0, np.maximum(np.abs(base), eps), base)
    
    # For fractional exponents, ensure base is non-negative
    if np.any((exponent % 1) != 0):
        base = np.maximum(base, 0)
    
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.power(base, exponent)
    
    # Clean up
    result = np.nan_to_num(result, nan=0.0, posinf=1e300, neginf=-1e300)
    
    return result


# ============================================================================
# ARRAY VALIDATION AND REPAIR
# ============================================================================

def validate_array(x: np.ndarray,
                   name: str = "array",
                   repair: bool = True,
                   nan_value: float = 0.0,
                   inf_value: float = 0.0,
                   raise_on_invalid: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Comprehensive array validation with optional repair.
    
    Checks for:
    - NaN values
    - Positive infinity
    - Negative infinity
    - All-zero arrays (potential upstream failure)
    
    Args:
        x: Input array
        name: Name for error messages
        repair: Whether to repair invalid values
        nan_value: Replacement for NaN
        inf_value: Replacement for Inf
        raise_on_invalid: Raise exception instead of repairing
        
    Returns:
        Tuple of (validated_array, diagnostics_dict)
        
    Example:
        >>> arr, diag = validate_array(my_data, "returns", repair=True)
        >>> if diag['had_issues']:
        ...     logger.warning(f"Repaired {diag['nan_count']} NaN values")
    """
    x = np.asarray(x)
    
    diagnostics = {
        'name': name,
        'shape': x.shape,
        'nan_count': int(np.sum(np.isnan(x))),
        'posinf_count': int(np.sum(np.isposinf(x))),
        'neginf_count': int(np.sum(np.isneginf(x))),
        'all_zero': bool(np.all(x == 0)),
        'had_issues': False
    }
    
    diagnostics['had_issues'] = (
        diagnostics['nan_count'] > 0 or
        diagnostics['posinf_count'] > 0 or
        diagnostics['neginf_count'] > 0
    )
    
    if diagnostics['had_issues']:
        if raise_on_invalid:
            raise ValueError(
                f"Invalid values in {name}: "
                f"{diagnostics['nan_count']} NaN, "
                f"{diagnostics['posinf_count']} +Inf, "
                f"{diagnostics['neginf_count']} -Inf"
            )
        
        if repair:
            x = np.nan_to_num(x, nan=nan_value, posinf=inf_value, neginf=-inf_value)
    
    return x, diagnostics


def validate_probability(p: np.ndarray,
                        eps: float = EPS,
                        name: str = "probability") -> np.ndarray:
    """
    Validate and normalize probability distribution.
    
    Ensures:
    - All values in [0, 1]
    - Sum equals 1
    - No NaN/Inf
    
    Args:
        p: Probability array
        eps: Minimum probability value
        name: Name for error messages
        
    Returns:
        Valid probability distribution
    """
    p = np.asarray(p)
    
    # Handle NaN/Inf
    p = np.nan_to_num(p, nan=eps, posinf=1.0, neginf=0.0)
    
    # Ensure non-negative
    p = np.maximum(p, eps)
    
    # Normalize to sum to 1
    total = p.sum()
    if total < eps:
        # All zeros - return uniform
        return np.ones_like(p) / len(p)
    
    return p / total


def validate_covariance(cov: np.ndarray,
                       min_eigenvalue: float = 1e-8,
                       regularization: float = 1e-6) -> Tuple[np.ndarray, bool]:
    """
    Validate and repair covariance matrix.
    
    Ensures:
    - Symmetric
    - Positive semi-definite
    - No NaN/Inf
    - Invertible (for optimization)
    
    Args:
        cov: Covariance matrix
        min_eigenvalue: Minimum acceptable eigenvalue
        regularization: Ridge regularization to add if singular
        
    Returns:
        Tuple of (valid_covariance, was_repaired)
    """
    cov = np.asarray(cov)
    was_repaired = False
    
    # Handle NaN/Inf
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        cov = np.nan_to_num(cov, nan=0.0, posinf=1e10, neginf=-1e10)
        was_repaired = True
    
    # Ensure symmetric
    cov = (cov + cov.T) / 2
    
    # Check eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        if np.min(eigenvalues) < min_eigenvalue:
            # Add ridge regularization
            cov = cov + regularization * np.eye(cov.shape[0])
            was_repaired = True
    except np.linalg.LinAlgError:
        # Matrix is severely ill-conditioned
        cov = cov + regularization * np.eye(cov.shape[0])
        was_repaired = True
    
    return cov, was_repaired


# ============================================================================
# DECORATORS FOR AUTOMATIC VALIDATION
# ============================================================================

def validate_inputs(*arg_names):
    """
    Decorator to automatically validate function inputs.
    
    Usage:
        @validate_inputs('returns', 'weights')
        def calculate_portfolio_return(returns, weights):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function argument names
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate specified arguments
            for name in arg_names:
                if name in bound.arguments:
                    val = bound.arguments[name]
                    if isinstance(val, np.ndarray):
                        val, diag = validate_array(val, name, repair=True)
                        bound.arguments[name] = val
            
            return func(*bound.args, **bound.kwargs)
        return wrapper
    return decorator


def safe_output(default_value=0.0):
    """
    Decorator to ensure function output is NaN/Inf free.
    
    Usage:
        @safe_output(default_value=0.0)
        def risky_calculation(x):
            return np.log(x) / x  # Could produce NaN/Inf
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                result = np.nan_to_num(result, nan=default_value, 
                                       posinf=default_value, neginf=default_value)
            elif isinstance(result, (int, float)):
                if np.isnan(result) or np.isinf(result):
                    result = default_value
            return result
        return wrapper
    return decorator


# ============================================================================
# SPECIALIZED FINANCIAL CALCULATIONS
# ============================================================================

def safe_returns(prices: np.ndarray,
                 method: str = 'log') -> np.ndarray:
    """
    Calculate returns with safety guards.
    
    Args:
        prices: Price series
        method: 'log' for log returns, 'simple' for simple returns
        
    Returns:
        Returns array (one element shorter than prices)
    """
    prices = np.asarray(prices)
    
    if method == 'log':
        # log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
        log_prices = safe_log(prices)
        returns = np.diff(log_prices)
    else:
        # (P_t - P_{t-1}) / P_{t-1}
        returns = safe_divide(np.diff(prices), prices[:-1])
    
    return returns


def safe_volatility(returns: np.ndarray,
                    annualization: float = 252) -> float:
    """
    Calculate annualized volatility with safety guards.
    """
    returns, _ = validate_array(returns, "returns", repair=True)
    
    if len(returns) < 2:
        return 0.0
    
    variance = np.var(returns, ddof=1)
    vol = safe_sqrt(variance * annualization)
    
    return float(vol)


def safe_sharpe(returns: np.ndarray,
                risk_free: float = 0.0,
                annualization: float = 252) -> float:
    """
    Calculate Sharpe ratio with safety guards.
    """
    returns, _ = validate_array(returns, "returns", repair=True)
    
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free / annualization
    mean_excess = np.mean(excess_returns)
    vol = np.std(returns, ddof=1)
    
    sharpe = safe_divide(mean_excess, vol) * safe_sqrt(annualization)
    
    return float(sharpe)
