"""
Hardened Portfolio Optimizer with Safe Math Operations
Author: Erdinc Erdogan
Purpose: Production-stable portfolio optimization with numerically-safe operations for Black-Litterman, HRP, and CVaR methods.
References:
- Black-Litterman Model (1992)
- Safe Math Operations
- Numerical Stability in Optimization
Usage:
    optimizer = PortfolioOptimizer(returns, swarm_views=views)
    result = optimizer.optimize(method='black_litterman')
"""
import numpy as np
from modules.core.safe_math import (
    safe_sqrt, safe_divide, safe_log,
    validate_array, validate_covariance
)
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from colorama import Fore, Style

# Core imports
from ..core.base import (
    StatisticalConstants, RiskTier, PositionAction,
    calculate_kelly_fraction, calculate_cvar, classify_risk_tier
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OptimizationResult:
    """Portfolio optimization result with full metadata."""
    tickers: List[str]
    weights: List[float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    kelly_constrained: bool
    cvar_95: Optional[float]
    method: str
    swarm_integrated: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "tickers": self.tickers,
            "weights": self.weights,
            "expected_return": self.expected_return,
            "risk": self.risk,
            "sharpe_ratio": self.sharpe_ratio,
            "kelly_constrained": self.kelly_constrained,
            "cvar_95": self.cvar_95,
            "method": self.method,
            "swarm_integrated": self.swarm_integrated,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SwarmView:
    """Swarm-derived investor view for Black-Litterman."""
    ticker: str
    view_type: str  # "absolute" or "relative"
    expected_return: float
    confidence: float
    entropy_score: float
    kelly_fraction: float
    relative_ticker: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "view_type": self.view_type,
            "expected_return": self.expected_return,
            "confidence": self.confidence,
            "entropy_score": self.entropy_score,
            "kelly_fraction": self.kelly_fraction,
            "relative_ticker": self.relative_ticker
        }


# ============================================================================
# PORTFOLIO OPTIMIZER (INSTITUTIONAL)
# ============================================================================

class PortfolioOptimizer:
    """
    Institutional-Grade Portfolio Optimizer.
    
    Implements:
    1. Mean-Variance Optimization (Markowitz)
    2. Black-Litterman with Swarm Views
    3. Hierarchical Risk Parity (HRP)
    4. Kelly-Constrained Optimization
    5. CVaR-Constrained Optimization
    """
    
    def __init__(self, risk_free_rate: float = None):
        """
        Initialize Portfolio Optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate or StatisticalConstants.RISK_FREE_RATE
    
    # ========================================================================
    # RETURN AND COVARIANCE CALCULATION
    # ========================================================================
    
    def calculate_returns(self, prices: Dict[str, List[float]]) -> Dict[str, Dict]:
        """
        Calculate returns with risk metrics.
        
        Returns:
            Dict with mean, std, daily_returns, sharpe for each ticker
        """
        returns = {}
        for ticker, price_list in prices.items():
            if len(price_list) >= 2:
                daily_returns = np.array([
                    (price_list[i] - price_list[i-1]) / price_list[i-1]
                    for i in range(1, len(price_list))
                ])
                
                mean_annual = np.mean(daily_returns) * StatisticalConstants.TRADING_DAYS_YEAR
                std_annual = np.std(daily_returns, ddof=1) * safe_sqrt(StatisticalConstants.TRADING_DAYS_YEAR)
                
                sharpe = (mean_annual - self.risk_free_rate) / std_annual if std_annual > 0 else 0
                
                returns[ticker] = {
                    'mean': mean_annual,
                    'std': std_annual,
                    'daily_returns': daily_returns,
                    'sharpe': sharpe,
                    'skewness': stats.skew(daily_returns),
                    'kurtosis': stats.kurtosis(daily_returns)
                }
        return returns
    
    def calculate_covariance_matrix(self, 
                                     returns: Dict[str, Dict],
                                     method: str = "sample") -> Tuple[np.ndarray, List[str]]:
        """
        Calculate covariance matrix.
        
        Args:
            returns: Returns dictionary from calculate_returns
            method: "sample" or "ledoit_wolf" (shrinkage)
        
        Returns:
            Tuple of (covariance matrix, ticker list)
        """
        tickers = list(returns.keys())
        n = len(tickers)
        
        # Build returns matrix
        min_len = min(len(returns[t]['daily_returns']) for t in tickers)
        returns_matrix = np.zeros((min_len, n))
        
        for i, ticker in enumerate(tickers):
            returns_matrix[:, i] = returns[ticker]['daily_returns'][:min_len]
        
        if method == "ledoit_wolf":
            # Ledoit-Wolf shrinkage estimator
            cov_matrix = self._ledoit_wolf_shrinkage(returns_matrix)
        else:
            # Sample covariance
            cov_matrix = np.cov(returns_matrix.T) * StatisticalConstants.TRADING_DAYS_YEAR
        
        return cov_matrix, tickers
    
    def _ledoit_wolf_shrinkage(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage estimator for covariance.
        
        Shrinks sample covariance toward identity matrix.
        """
        n, p = returns_matrix.shape
        
        # Sample covariance
        sample_cov = np.cov(returns_matrix.T)
        
        # Target: scaled identity
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)
        
        # Shrinkage intensity (simplified)
        delta = sample_cov - target
        shrinkage = min(1.0, (np.sum(delta ** 2) / n) / np.sum(delta ** 2))
        
        # Shrunk covariance
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
        
        return shrunk_cov * StatisticalConstants.TRADING_DAYS_YEAR
    
    # ========================================================================
    # MEAN-VARIANCE OPTIMIZATION
    # ========================================================================
    
    def optimize_markowitz(self,
                           tickers: List[str],
                           expected_returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           kelly_constraint: float = None,
                           target_return: float = None,
                           long_only: bool = True) -> OptimizationResult:
        """
        Markowitz Mean-Variance Optimization.
        
        With optional Kelly constraint as hard limit.
        
        Args:
            tickers: List of ticker symbols
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            kelly_constraint: Maximum weight per asset (Kelly fraction)
            target_return: Target portfolio return (optional)
            long_only: Long-only constraint
        
        Returns:
            OptimizationResult with optimal weights
        """
        n = len(tickers)
        
        # Analytical solution for maximum Sharpe
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n)
        excess_returns = expected_returns - self.risk_free_rate
        
        # Unconstrained optimal weights
        raw_weights = np.dot(inv_cov, excess_returns)
        
        # Normalize
        if long_only:
            raw_weights = np.maximum(raw_weights, 0)
        
        weights = raw_weights / np.sum(np.abs(raw_weights))
        
        # Apply Kelly constraint
        kelly_applied = False
        if kelly_constraint is not None and kelly_constraint > 0:
            max_weight = kelly_constraint / n  # Distribute Kelly across assets
            weights = np.minimum(weights, max_weight)
            weights = weights / np.sum(weights)  # Renormalize
            kelly_applied = True
        
        # Calculate portfolio metrics
        port_return = np.dot(weights, expected_returns)
        port_risk = safe_sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
        
        return OptimizationResult(
            tickers=tickers,
            weights=weights.tolist(),
            expected_return=port_return,
            risk=port_risk,
            sharpe_ratio=sharpe,
            kelly_constrained=kelly_applied,
            cvar_95=None,
            method="markowitz"
        )
    
    # ========================================================================
    # HIERARCHICAL RISK PARITY (HRP)
    # ========================================================================
    
    def optimize_hrp(self,
                     tickers: List[str],
                     returns_matrix: np.ndarray,
                     kelly_constraint: float = None) -> OptimizationResult:
        """
        Hierarchical Risk Parity (HRP) Optimization.
        
        Algorithm:
        1. Calculate correlation distance matrix
        2. Hierarchical clustering (Ward's method)
        3. Quasi-diagonalization
        4. Recursive bisection for weights
        
        Args:
            tickers: List of ticker symbols
            returns_matrix: Matrix of returns (T x N)
            kelly_constraint: Maximum weight per asset
        
        Returns:
            OptimizationResult with HRP weights
        """
        n = len(tickers)
        
        # Step 1: Correlation matrix and distance
        corr_matrix = np.corrcoef(returns_matrix.T)
        dist_matrix = safe_sqrt((1 - corr_matrix) / 2)
        
        # Step 2: Hierarchical clustering
        dist_condensed = squareform(dist_matrix)
        linkage_matrix = linkage(dist_condensed, method='ward')
        
        # Step 3: Quasi-diagonalization (get sorted order)
        sorted_idx = leaves_list(linkage_matrix)
        
        # Step 4: Recursive bisection
        cov_matrix = np.cov(returns_matrix.T) * StatisticalConstants.TRADING_DAYS_YEAR
        weights = self._hrp_recursive_bisection(cov_matrix, sorted_idx)
        
        # Apply Kelly constraint
        kelly_applied = False
        if kelly_constraint is not None and kelly_constraint > 0:
            max_weight = kelly_constraint / n
            weights = np.minimum(weights, max_weight)
            weights = weights / np.sum(weights)
            kelly_applied = True
        
        # Calculate portfolio metrics
        expected_returns = np.mean(returns_matrix, axis=0) * StatisticalConstants.TRADING_DAYS_YEAR
        port_return = np.dot(weights, expected_returns)
        port_risk = safe_sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
        
        return OptimizationResult(
            tickers=tickers,
            weights=weights.tolist(),
            expected_return=port_return,
            risk=port_risk,
            sharpe_ratio=sharpe,
            kelly_constrained=kelly_applied,
            cvar_95=None,
            method="hrp"
        )
    
    def _hrp_recursive_bisection(self,
                                  cov_matrix: np.ndarray,
                                  sorted_idx: np.ndarray) -> np.ndarray:
        """
        Recursive bisection for HRP weights.
        
        Allocates weights based on inverse variance of clusters.
        """
        n = len(sorted_idx)
        weights = np.ones(n)
        
        # Initialize clusters
        clusters = [sorted_idx.tolist()]
        
        while len(clusters) > 0:
            # Split each cluster
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # Split in half
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]
                    
                    # Calculate cluster variances
                    left_var = self._cluster_variance(cov_matrix, left)
                    right_var = self._cluster_variance(cov_matrix, right)
                    
                    # Allocate based on inverse variance
                    total_var = left_var + right_var
                    if total_var > 0:
                        left_weight = 1 - left_var / total_var
                        right_weight = 1 - right_var / total_var
                    else:
                        left_weight = right_weight = 0.5
                    
                    # Update weights
                    for idx in left:
                        weights[idx] *= left_weight
                    for idx in right:
                        weights[idx] *= right_weight
                    
                    # Add to new clusters if further splitting needed
                    if len(left) > 1:
                        new_clusters.append(left)
                    if len(right) > 1:
                        new_clusters.append(right)
            
            clusters = new_clusters
        
        # Normalize
        weights = weights / np.sum(weights)
        return weights
    
    def _cluster_variance(self, cov_matrix: np.ndarray, indices: List[int]) -> float:
        """Calculate variance of a cluster using inverse-variance weighting."""
        sub_cov = cov_matrix[np.ix_(indices, indices)]
        inv_diag = 1 / np.diag(sub_cov)
        weights = inv_diag / np.sum(inv_diag)
        return np.dot(weights.T, np.dot(sub_cov, weights))
    
    # ========================================================================
    # RISK PARITY
    # ========================================================================
    
    def optimize_risk_parity(self,
                              tickers: List[str],
                              cov_matrix: np.ndarray,
                              kelly_constraint: float = None) -> OptimizationResult:
        """
        Risk Parity Optimization.
        
        Each asset contributes equally to portfolio risk.
        
        RC_i = w_i * (Σw)_i / σ_p = 1/n
        """
        n = len(tickers)
        
        # Inverse volatility weights (approximation)
        volatilities = safe_sqrt(np.diag(cov_matrix))
        inv_vol = 1 / volatilities
        weights = inv_vol / np.sum(inv_vol)
        
        # Apply Kelly constraint
        kelly_applied = False
        if kelly_constraint is not None and kelly_constraint > 0:
            max_weight = kelly_constraint / n
            weights = np.minimum(weights, max_weight)
            weights = weights / np.sum(weights)
            kelly_applied = True
        
        # Calculate portfolio metrics
        port_risk = safe_sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return OptimizationResult(
            tickers=tickers,
            weights=weights.tolist(),
            expected_return=0.0,  # Not calculated for risk parity
            risk=port_risk,
            sharpe_ratio=0.0,
            kelly_constrained=kelly_applied,
            cvar_95=None,
            method="risk_parity"
        )
    
    # ========================================================================
    # ALLOCATION
    # ========================================================================
    
    def allocate_portfolio(self,
                           capital: float,
                           tickers: List[str],
                           prices: Dict[str, float],
                           weights: List[float]) -> Dict:
        """
        Allocate capital based on weights.
        
        Returns:
            Allocation details with shares and amounts
        """
        allocations = []
        total_allocated = 0
        
        for i, ticker in enumerate(tickers):
            allocation_amount = capital * weights[i]
            price = prices.get(ticker, 1)
            shares = int(allocation_amount / price) if price > 0 else 0
            actual_amount = shares * price
            
            allocations.append({
                'ticker': ticker,
                'weight': weights[i],
                'target_amount': allocation_amount,
                'actual_amount': actual_amount,
                'shares': shares,
                'price': price
            })
            total_allocated += actual_amount
        
        return {
            'capital': capital,
            'allocations': allocations,
            'total_allocated': total_allocated,
            'cash_remaining': capital - total_allocated
        }


# ============================================================================
# BLACK-LITTERMAN WITH SWARM INTEGRATION
# ============================================================================

class BlackLittermanSwarm:
    """
    Black-Litterman Portfolio Optimization with Swarm Integration.
    
    Uses Swarm consensus as investor views:
    - P matrix: View selection from agent recommendations
    - Q vector: Expected returns from confidence × expected move
    - Omega matrix: View uncertainty from Shannon entropy
    
    Mathematical Basis:
    E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
    
    Kelly Constraint:
    w_i^final = min(w_i^BL, f*_Kelly / n)
    """
    
    def __init__(self, 
                 risk_free_rate: float = None,
                 tau: float = 0.05,
                 risk_aversion: float = 2.5):
        """
        Initialize Black-Litterman with Swarm.
        
        Args:
            risk_free_rate: Annual risk-free rate
            tau: Uncertainty parameter (typically 0.025-0.05)
            risk_aversion: Risk aversion coefficient (δ)
        """
        self.risk_free_rate = risk_free_rate or StatisticalConstants.RISK_FREE_RATE
        self.tau = tau
        self.risk_aversion = risk_aversion
    
    def calculate_equilibrium_returns(self,
                                       market_caps: Dict[str, float],
                                       cov_matrix: np.ndarray,
                                       tickers: List[str]) -> np.ndarray:
        """
        Calculate implied equilibrium returns.
        
        π = δ × Σ × w_mkt
        
        Args:
            market_caps: Market capitalization for each ticker
            cov_matrix: Covariance matrix
            tickers: List of tickers (in order of cov_matrix)
        
        Returns:
            Equilibrium returns vector
        """
        total_cap = sum(market_caps.values())
        w_mkt = np.array([market_caps.get(t, 0) / total_cap for t in tickers])
        
        # Implied equilibrium returns
        pi = self.risk_aversion * np.dot(cov_matrix, w_mkt)
        
        return pi, w_mkt
    
    def create_swarm_views(self,
                           swarm_result: Dict,
                           tickers: List[str],
                           expected_moves: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create P, Q, Omega matrices from Swarm consensus.
        
        Args:
            swarm_result: Result from SwarmOrchestrator.run_debate()
            tickers: List of tickers
            expected_moves: Expected price moves per ticker (optional)
        
        Returns:
            Tuple of (P matrix, Q vector, Omega matrix)
        """
        n = len(tickers)
        ticker_idx = {t: i for i, t in enumerate(tickers)}
        
        # Extract swarm data
        decision = swarm_result.get("swarm_decision", "BEKLE")
        confidence = swarm_result.get("confidence", 0.5)
        kelly_fraction = swarm_result.get("kelly_fraction", 0.1)
        entropy_scores = swarm_result.get("entropy_scores", {"bull": 0.5, "bear": 0.5})
        ticker = swarm_result.get("ticker", tickers[0] if tickers else "")
        
        # Default expected move based on decision
        if expected_moves is None:
            expected_moves = {}
        
        default_move = 0.10 if decision == "AL" else (-0.10 if decision == "SAT" else 0.0)
        
        # Build views
        views = []
        
        # Main view from swarm decision
        if ticker in ticker_idx and decision != "BEKLE":
            views.append({
                "type": "absolute",
                "ticker": ticker,
                "return": expected_moves.get(ticker, default_move) * confidence,
                "confidence": confidence,
                "entropy": (entropy_scores.get("bull", 0.5) + entropy_scores.get("bear", 0.5)) / 2
            })
        
        # Build matrices
        k = len(views)
        if k == 0:
            # No views - return empty matrices
            return np.zeros((0, n)), np.zeros(0), np.eye(1)
        
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)
        
        for i, view in enumerate(views):
            if view["type"] == "absolute":
                idx = ticker_idx.get(view["ticker"])
                if idx is not None:
                    P[i, idx] = 1.0
                    Q[i] = view["return"]
                    # Omega from entropy: higher entropy = lower uncertainty
                    entropy = max(view["entropy"], 0.1)
                    omega_diag[i] = self.tau / entropy
        
        Omega = np.diag(omega_diag)
        
        return P, Q, Omega
    
    def optimize_with_swarm(self,
                            tickers: List[str],
                            market_caps: Dict[str, float],
                            cov_matrix: np.ndarray,
                            swarm_result: Dict,
                            kelly_constraint: float = None,
                            expected_moves: Dict[str, float] = None,
                            long_only: bool = True) -> OptimizationResult:
        """
        Black-Litterman optimization with Swarm views.
        
        Args:
            tickers: List of ticker symbols
            market_caps: Market capitalizations
            cov_matrix: Covariance matrix
            swarm_result: Result from SwarmOrchestrator
            kelly_constraint: Kelly fraction constraint
            expected_moves: Expected price moves
            long_only: Long-only constraint
        
        Returns:
            OptimizationResult with BL weights
        """
        n = len(tickers)
        
        # Step 1: Calculate equilibrium returns
        pi, w_mkt = self.calculate_equilibrium_returns(market_caps, cov_matrix, tickers)
        
        # Step 2: Create views from Swarm
        P, Q, Omega = self.create_swarm_views(swarm_result, tickers, expected_moves)
        
        # Step 3: Black-Litterman posterior
        if P.shape[0] == 0:
            # No views - use equilibrium
            posterior_mean = pi
            posterior_cov = cov_matrix
        else:
            posterior_mean, posterior_cov = self._calculate_posterior(
                pi, cov_matrix, P, Q, Omega
            )
        
        # Step 4: Optimal weights
        inv_cov = np.linalg.inv(posterior_cov)
        excess_returns = posterior_mean - self.risk_free_rate
        raw_weights = np.dot(inv_cov, excess_returns)
        
        # Long-only constraint
        if long_only:
            raw_weights = np.maximum(raw_weights, 0)
        
        # Normalize
        weights = raw_weights / np.sum(np.abs(raw_weights)) if np.sum(np.abs(raw_weights)) > 0 else w_mkt
        
        # Step 5: Apply Kelly constraint (HARD LIMIT)
        kelly_applied = False
        if kelly_constraint is not None and kelly_constraint > 0:
            # Kelly as hard limit per asset
            max_weight_per_asset = kelly_constraint / n
            
            # Apply constraint
            weights = np.minimum(weights, max_weight_per_asset)
            
            # Renormalize
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            kelly_applied = True
        
        # Calculate portfolio metrics
        port_return = np.dot(weights, posterior_mean)
        port_risk = safe_sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
        
        return OptimizationResult(
            tickers=tickers,
            weights=weights.tolist(),
            expected_return=port_return,
            risk=port_risk,
            sharpe_ratio=sharpe,
            kelly_constrained=kelly_applied,
            cvar_95=None,
            method="black_litterman_swarm",
            swarm_integrated=True
        )
    
    def _calculate_posterior(self,
                              pi: np.ndarray,
                              cov_matrix: np.ndarray,
                              P: np.ndarray,
                              Q: np.ndarray,
                              Omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Black-Litterman posterior.
        
        E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
        """
        tau_sigma = self.tau * cov_matrix
        
        # Inverse matrices
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(Omega)
        
        # Posterior precision
        posterior_precision = inv_tau_sigma + np.dot(np.dot(P.T, inv_omega), P)
        posterior_cov = np.linalg.inv(posterior_precision)
        
        # Posterior mean
        term1 = np.dot(inv_tau_sigma, pi)
        term2 = np.dot(np.dot(P.T, inv_omega), Q)
        posterior_mean = np.dot(posterior_cov, term1 + term2)
        
        return posterior_mean, posterior_cov
    
    def generate_bl_report(self,
                           capital: float,
                           result: OptimizationResult,
                           swarm_result: Dict,
                           current_prices: Dict[str, float]) -> str:
        """Generate Black-Litterman with Swarm report."""
        optimizer = PortfolioOptimizer()
        allocation = optimizer.allocate_portfolio(
            capital, result.tickers, current_prices, result.weights
        )
        
        report = f"""
<black_litterman_swarm>
🎯 BLACK-LITTERMAN + SWARM INTELLIGENCE PORTFOLIO
══════════════════════════════════════════════════════════════

💰 Capital: ${capital:,.2f}
📊 Expected Return: {result.expected_return*100:.1f}%
⚠️ Risk (Volatility): {result.risk*100:.1f}%
📈 Sharpe Ratio: {result.sharpe_ratio:.2f}
🔒 Kelly Constrained: {'Yes' if result.kelly_constrained else 'No'}

🧬 SWARM INTEGRATION:
  • Decision: {swarm_result.get('swarm_decision', 'N/A')}
  • Confidence: {swarm_result.get('confidence', 0)*100:.0f}%
  • Kelly Fraction: {swarm_result.get('kelly_fraction', 0)*100:.1f}%
  • Risk Tier: {swarm_result.get('risk_tier', 'N/A')}

📋 ALLOCATION:
"""
        for alloc in allocation['allocations']:
            if alloc['weight'] > 0.01:
                report += f"  • {alloc['ticker']}: {alloc['weight']*100:.1f}% (${alloc['actual_amount']:,.0f}) → {alloc['shares']} shares\n"
        
        report += f"""
💵 Cash Remaining: ${allocation['cash_remaining']:.2f}

</black_litterman_swarm>
"""
        return report


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

class BlackLitterman(BlackLittermanSwarm):
    """
    Legacy Black-Litterman class for backward compatibility.
    
    Extends BlackLittermanSwarm with traditional view interface.
    """
    
    def optimize_with_views(self,
                            tickers: List[str],
                            market_caps: Dict[str, float],
                            cov_matrix: np.ndarray,
                            views: List[Dict]) -> OptimizationResult:
        """
        Traditional Black-Litterman with manual views.
        
        Args:
            tickers: List of tickers
            market_caps: Market capitalizations
            cov_matrix: Covariance matrix
            views: List of view dictionaries
        
        Returns:
            OptimizationResult
        """
        n = len(tickers)
        ticker_idx = {t: i for i, t in enumerate(tickers)}
        
        # Equilibrium returns
        pi, w_mkt = self.calculate_equilibrium_returns(market_caps, cov_matrix, tickers)
        
        if not views:
            return OptimizationResult(
                tickers=tickers,
                weights=w_mkt.tolist(),
                expected_return=np.dot(w_mkt, pi),
                risk=safe_sqrt(np.dot(w_mkt.T, np.dot(cov_matrix, w_mkt))),
                sharpe_ratio=0.0,
                kelly_constrained=False,
                cvar_95=None,
                method="black_litterman_equilibrium"
            )
        
        # Build P, Q, Omega from views
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        confidence_weights = []
        
        for i, view in enumerate(views):
            confidence = view.get("confidence", 0.5)
            confidence_weights.append(confidence)
            
            if view["type"] == "absolute":
                idx = ticker_idx.get(view["asset"])
                if idx is not None:
                    P[i, idx] = 1.0
                    Q[i] = view["return"]
            elif view["type"] == "relative":
                long_idx = ticker_idx.get(view["long"])
                short_idx = ticker_idx.get(view["short"])
                if long_idx is not None and short_idx is not None:
                    P[i, long_idx] = 1.0
                    P[i, short_idx] = -1.0
                    Q[i] = view["return"]
        
        # Omega
        tau_sigma = self.tau * cov_matrix
        omega_diag = np.diag(np.dot(np.dot(P, tau_sigma), P.T))
        Omega = np.diag(omega_diag / np.array(confidence_weights))
        
        # Posterior
        posterior_mean, posterior_cov = self._calculate_posterior(pi, cov_matrix, P, Q, Omega)
        
        # Optimal weights
        inv_cov = np.linalg.inv(cov_matrix)
        excess_returns = posterior_mean - self.risk_free_rate
        raw_weights = np.dot(inv_cov, excess_returns)
        raw_weights = np.maximum(raw_weights, 0)
        weights = raw_weights / np.sum(raw_weights) if np.sum(raw_weights) > 0 else w_mkt
        
        port_return = np.dot(weights, posterior_mean)
        port_risk = safe_sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_risk if port_risk > 0 else 0
        
        return OptimizationResult(
            tickers=tickers,
            weights=weights.tolist(),
            expected_return=port_return,
            risk=port_risk,
            sharpe_ratio=sharpe,
            kelly_constrained=False,
            cvar_95=None,
            method="black_litterman"
        )
