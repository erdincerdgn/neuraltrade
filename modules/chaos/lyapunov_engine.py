"""
Lyapunov Exponent Engine - Rosenstein's Method Implementation
Author: Erdinc Erdogan
Purpose: Computes Lyapunov exponents using Rosenstein's method to detect chaos versus
randomness in financial time series for regime-aware trading.
References:
- Rosenstein et al. (1993) "A Practical Method for Calculating Largest Lyapunov Exponents"
- Wolf et al. (1985) "Determining Lyapunov Exponents from a Time Series"
- Chaos Detection in Financial Markets
Usage:
    lyapunov = LyapunovEngine(embedding_dim=3, tau=1)
    result = lyapunov.compute(price_series)
    if result.is_chaotic: reduce_leverage()
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class LyapunovResult:
    lyapunov: float
    is_chaotic: bool
    divergence_rate: float
    embedding_dim: int
    is_valid: bool

class LyapunovEngine:
    EPSILON = 1e-10
    
    def __init__(self, embedding_dim: int = 10, tau: int = 1, min_neighbors: int = 5, theiler_window: int = 10):
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.min_neighbors = min_neighbors
        self.theiler_window = theiler_window
    
    def _embed(self, data: np.ndarray) -> np.ndarray:
        n = len(data) - (self.embedding_dim - 1) * self.tau
        if n <= 0:
            return np.array([])
        embedded = np.zeros((n, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = data[i * self.tau:i * self.tau + n]
        return embedded
    
    def _find_nearest_neighbors(self, embedded: np.ndarray) -> np.ndarray:
        n = len(embedded)
        neighbors = np.zeros(n, dtype=int)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if abs(i - j) > self.theiler_window:
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    if dist < min_dist and dist > self.EPSILON:
                        min_dist = dist
                        neighbors[i] = j
        return neighbors
    
    def _compute_divergence(self, embedded: np.ndarray, neighbors: np.ndarray, max_iter: int) -> np.ndarray:
        n = len(embedded)
        divergence = np.zeros(max_iter)
        counts = np.zeros(max_iter)
        
        for i in range(n - max_iter):
            j = neighbors[i]
            if j < n - max_iter:
                for k in range(max_iter):
                    d0 = np.linalg.norm(embedded[i] - embedded[j]) + self.EPSILON
                    dk = np.linalg.norm(embedded[i + k] - embedded[j + k]) + self.EPSILON
                    divergence[k] += np.log(dk / d0)
                    counts[k] += 1
        
        valid = counts > 0
        divergence[valid] /= counts[valid]
        return divergence
    
    def compute(self, data: np.ndarray, max_iter: int = 50) -> LyapunovResult:
        if len(data) < self.embedding_dim * 10:
            return LyapunovResult(0.0, False, 0.0, self.embedding_dim, False)
        
        embedded = self._embed(data)
        if len(embedded) < max_iter * 2:
            return LyapunovResult(0.0, False, 0.0, self.embedding_dim, False)
        
        neighbors = self._find_nearest_neighbors(embedded)
        divergence = self._compute_divergence(embedded, neighbors, max_iter)
        
        # Linear fit to divergence curve
        valid_range = min(max_iter // 2, 20)
        x = np.arange(valid_range)
        y = divergence[:valid_range]
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return LyapunovResult(0.0, False, 0.0, self.embedding_dim, False)
        
        coeffs = np.polyfit(x, y, 1)
        lyapunov = coeffs[0]
        
        is_chaotic = lyapunov > 0.05
        return LyapunovResult(lyapunov, is_chaotic, lyapunov, self.embedding_dim, True)
