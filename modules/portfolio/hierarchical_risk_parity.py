"""
Hierarchical Risk Parity (HRP) Portfolio Allocation
Author: Erdinc Erdogan
Purpose: Implements López de Prado's HRP algorithm using hierarchical clustering, quasi-diagonalization, and recursive bisection for robust diversification.
References:
- López de Prado (2016) "Building Diversified Portfolios"
- Agglomerative Hierarchical Clustering
- Inverse-Variance Recursive Bisection
Usage:
    hrp = HierarchicalRiskParityEngine()
    result = hrp.optimize(returns, asset_names=['AAPL', 'MSFT', 'GOOGL'])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
import warnings

try:
    from core.base import BaseModule
except ImportError:
    class BaseModule:
        """Fallback base class for standalone execution"""
        def __init__(self, config: dict = None):
            self.config = config or {}


class ClusteringMethod:
    """Supported hierarchical clustering linkage methods"""
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"



class DistanceMetric:
    """Distance metrics for correlation-based clustering"""
    ANGULAR = "angular"           # d = √(0.5 × (1 - ρ))
    ABSOLUTE_ANGULAR = "abs_angular"  # d = √(0.5 × (1 - |ρ|))
    SQUARED = "squared"           # d = √(1 - ρ²)


@dataclass
class HRPResult:
    """Container for HRP optimization results"""
    weights: np.ndarray
    cluster_weights: Dict[int, float]
    sorted_indices: List[int]
    linkage_matrix: np.ndarray
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    quasi_diag_covariance: np.ndarray
    cluster_variance: Dict[int, float]
    diversification_ratio: float
    effective_n_assets: float
    asset_names: Optional[List[str]] = None


@dataclass
class ClusterNode:
    """Represents a node in the hierarchical cluster tree"""
    node_id: int
    left_child: Optional['ClusterNode'] = None
    right_child: Optional['ClusterNode'] = None
    assets: List[int] = field(default_factory=list)
    weight: float = 1.0
    variance: float = 0.0


class HierarchicalRiskParityEngine(BaseModule):
    """
    Institutional-Grade Hierarchical Risk Parity (HRP) Engine.
    
    Implements López de Prado's HRP algorithm with enhancements:
    - Multiple distance metrics and linkage methods
    - Cluster-level risk contribution analysis
    - Constraint handling (min/max weights, sector limits)
    - Robust covariance estimation options
    Algorithm Overview:
    ------------------
    1. TREE CLUSTERING
       - Compute correlation matrix from returns
       - Convert to distance matrix: d(i,j) = √(0.5 × (1 - ρᵢⱼ))
       - Apply hierarchical clustering (default: single linkage)
       
    2. QUASI-DIAGONALIZATION
       - Extract leaf ordering from dendrogram
       - Reorder covariance matrix to place similar assets adjacent
    3. RECURSIVE BISECTION
       - Start with full portfolio (weight = 1)
       - Recursively split into left/right clusters
       - Allocate weights inversely proportional to cluster variance
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.linkage_method: str = self.config.get('linkage_method', ClusteringMethod.SINGLE)
        self.distance_metric: str = self.config.get('distance_metric', DistanceMetric.ANGULAR)
        self.min_weight: float = self.config.get('min_weight', 0.0)
        self.max_weight: float = self.config.get('max_weight', 1.0)
        self._linkage_matrix: Optional[np.ndarray] = None
        self._sorted_indices: Optional[List[int]] = None
    
    def compute_correlation_matrix(
        self,
        returns: np.ndarray,
        method: str = "pearson"
    ) -> np.ndarray:
        """Compute correlation matrix from returns."""
        if method == "spearman":
            corr_matrix, _ = spearmanr(returns)
            if isinstance(corr_matrix, float):
                corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
        else:
            corr_matrix = np.corrcoef(returns, rowvar=False)
        
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        return corr_matrix
    
    def correlation_to_distance(
        self,
        correlation_matrix: np.ndarray,
        metric: Optional[str] = None
    ) -> np.ndarray:
        """Convert correlation matrix to distance matrix."""
        metric = metric or self.distance_metric
        
        if metric == DistanceMetric.ANGULAR:
            distance = np.sqrt(0.5 * (1.0 - correlation_matrix))
        elif metric == DistanceMetric.ABSOLUTE_ANGULAR:
            distance = np.sqrt(0.5 * (1.0 - np.abs(correlation_matrix)))
        elif metric == DistanceMetric.SQUARED:
            distance = np.sqrt(1.0 - correlation_matrix ** 2)
        else:
            distance = np.sqrt(0.5 * (1.0 - correlation_matrix))
        
        np.fill_diagonal(distance, 0.0)
        distance = np.clip(distance, 0.0, 1.0)
        return distance
    
    def perform_clustering(
        self,
        distance_matrix: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """Perform hierarchical agglomerative clustering."""
        method = method or self.linkage_method
        condensed_dist = squareform(distance_matrix, checks=False)
        condensed_dist = np.nan_to_num(condensed_dist, nan=0.0, posinf=1.0, neginf=0.0)
        linkage_matrix = linkage(condensed_dist, method=method)
        self._linkage_matrix = linkage_matrix
        return linkage_matrix
    
    def quasi_diagonalize(self, linkage_matrix: np.ndarray) -> List[int]:
        """Quasi-diagonalization via seriation."""
        sorted_indices = list(leaves_list(linkage_matrix))
        self._sorted_indices = sorted_indices
        return sorted_indices
    
    def compute_cluster_variance(
        self,
        covariance_matrix: np.ndarray,
        cluster_assets: List[int]
    ) -> float:
        """Compute variance of an inverse-variance weighted cluster."""
        if len(cluster_assets) == 1:
            idx = cluster_assets[0]
            return covariance_matrix[idx, idx]
        
        cluster_cov = covariance_matrix[np.ix_(cluster_assets, cluster_assets)]
        variances = np.diag(cluster_cov)
        variances = np.maximum(variances, 1e-10)
        inv_var = 1.0 / variances
        weights = inv_var / np.sum(inv_var)
        cluster_variance = weights @ cluster_cov @ weights
        return float(cluster_variance)
    
    def recursive_bisection(
        self,
        covariance_matrix: np.ndarray,
        sorted_indices: List[int]
    ) -> np.ndarray:
        """Recursive bisection for weight allocation."""
        n_assets = covariance_matrix.shape[0]
        weights = np.zeros(n_assets)
        cluster_weights = {tuple(sorted_indices):1.0}
        
        while cluster_weights:
            new_cluster_weights = {}
            
            for cluster, weight in cluster_weights.items():
                cluster = list(cluster)
                
                if len(cluster) == 1:
                    weights[cluster[0]] = weight
                    continue
                
                split_point = len(cluster) // 2
                left_cluster = cluster[:split_point]
                right_cluster = cluster[split_point:]
                
                left_var = self.compute_cluster_variance(covariance_matrix, left_cluster)
                right_var = self.compute_cluster_variance(covariance_matrix, right_cluster)
                
                total_var = left_var + right_var
                if total_var < 1e-10:
                    alpha = 0.5
                else:
                    alpha = 1.0 - left_var / total_var
                
                alpha = np.clip(alpha, 0.0, 1.0)
                
                if len(left_cluster) == 1:
                    weights[left_cluster[0]] = weight * alpha
                else:
                    new_cluster_weights[tuple(left_cluster)] = weight * alpha
                
                if len(right_cluster) == 1:
                    weights[right_cluster[0]] = weight * (1.0 - alpha)
                else:
                    new_cluster_weights[tuple(right_cluster)] = weight * (1.0 - alpha)
            
            cluster_weights = new_cluster_weights
        
        return weights
    
    def apply_constraints(
        self,
        weights: np.ndarray,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None
    ) -> np.ndarray:
        """Apply weight constraints with renormalization."""
        min_w = min_weight if min_weight is not None else self.min_weight
        max_w = max_weight if max_weight is not None else self.max_weight
        
        constrained = np.clip(weights, min_w, max_w)
        
        for _ in range(100):
            total = np.sum(constrained)
            if abs(total - 1.0) < 1e-10:
                break
            constrained = constrained / total
            constrained = np.clip(constrained, min_w, max_w)
        
        constrained = constrained / np.sum(constrained)
        return constrained
    
    def compute_diversification_ratio(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """Compute diversification ratio: DR = (w' × σ) / √(w' × Σ × w)"""
        volatilities = np.sqrt(np.diag(covariance_matrix))
        weighted_vol_sum = weights @ volatilities
        portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
        
        if portfolio_vol < 1e-10:
            return 1.0
        return weighted_vol_sum / portfolio_vol
    
    def compute_effective_n(self, weights: np.ndarray) -> float:
        """Compute effective number of assets: N_eff = 1 / Σ(w²)"""
        return 1.0 / np.sum(weights ** 2)
    
    def fit(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        asset_names: Optional[List[str]] = None,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None
    ) -> HRPResult:
        """Full HRP optimization pipeline."""
        if isinstance(returns, pd.DataFrame):
            if asset_names is None:
                asset_names = list(returns.columns)
            returns = returns.values
        
        n_assets = returns.shape[1]
        
        correlation_matrix = self.compute_correlation_matrix(returns)
        covariance_matrix = np.cov(returns, rowvar=False)
        distance_matrix = self.correlation_to_distance(correlation_matrix)
        linkage_matrix = self.perform_clustering(distance_matrix)
        sorted_indices = self.quasi_diagonalize(linkage_matrix)
        
        quasi_diag_cov = covariance_matrix[np.ix_(sorted_indices, sorted_indices)]
        weights = self.recursive_bisection(covariance_matrix, sorted_indices)
        
        if min_weight is not None or max_weight is not None:
            weights = self.apply_constraints(weights, min_weight, max_weight)
        
        cluster_weights = {}
        cluster_variance = {}
        for i, idx in enumerate(sorted_indices):
            cluster_weights[idx] = weights[idx]
            cluster_variance[idx] = covariance_matrix[idx, idx]
        
        div_ratio = self.compute_diversification_ratio(weights, covariance_matrix)
        eff_n = self.compute_effective_n(weights)
        
        return HRPResult(
            weights=weights,
            cluster_weights=cluster_weights,
            sorted_indices=sorted_indices,
            linkage_matrix=linkage_matrix,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            quasi_diag_covariance=quasi_diag_cov,
            cluster_variance=cluster_variance,
            diversification_ratio=div_ratio,
            effective_n_assets=eff_n,
            asset_names=asset_names
        )
    
    def get_cluster_composition(
        self,
        linkage_matrix: np.ndarray,
        n_clusters: int,
        asset_names: Optional[List[str]] = None
    ) -> Dict[int, List]:
        """Extract cluster composition at a given level."""
        n_assets = linkage_matrix.shape[0] + 1
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            name = asset_names[i] if asset_names else i
            clusters[label].append(name)
        
        return clusters