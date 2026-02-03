"""
PC-Stable Algorithm with Parallelization - Multi-threaded Causal Discovery
Author: Erdinc Erdogan
Purpose: Implements the PC-Stable algorithm with multi-threaded execution and robust conditional
independence testing for scalable causal structure learning.
References:
- Colombo & Maathuis (2014) "Order-independent Constraint-based Causal Structure Learning"
- Parallel Algorithms for Constraint-based Causal Discovery
- Thread Pool Executor Patterns
Usage:
    pc = PCStableParallel(n_workers=4, use_robust_ci=True)
    dag = pc.fit(data, variable_names)
"""

import numpy as np
from scipy import stats
from scipy.linalg import inv
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations


@dataclass
class CITestResult:
    is_independent: bool
    statistic: float
    p_value: float
    conditioning_set: Tuple[int, ...]


class CITestCache:
    def __init__(self):
        self._cache: Dict[Tuple, CITestResult] = {}
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, x: int, y: int, cond_set: List[int]) -> Tuple:
        return (min(x, y), max(x, y), tuple(sorted(cond_set)))
    
    def get(self, x: int, y: int, cond_set: List[int]) -> Optional[CITestResult]:
        key = self._make_key(x, y, cond_set)
        result = self._cache.get(key)
        self._hits += 1 if result else 0
        self._misses += 0 if result else 1
        return result
    
    def put(self, x: int, y: int, cond_set: List[int], result: CITestResult):
        self._cache[self._make_key(x, y, cond_set)] = result
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def clear(self):
        self._cache.clear()
        self._hits = self._misses = 0


class PCStableParallel:
    def __init__(self, alpha: float = 0.05, max_cond_size: int = 4,
                 n_workers: int = 4, use_robust_ci: bool = True):
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.n_workers = n_workers
        self.use_robust_ci = use_robust_ci
        self.cache = CITestCache()
        self._sep_sets: Dict[Tuple[int, int], Set[int]] = {}
        
    def _ci_test(self, data: np.ndarray, x: int, y: int, 
                 cond_set: List[int]) -> CITestResult:
        cached = self.cache.get(x, y, cond_set)
        if cached:
            return cached
        
        n = data.shape[0]
        if len(cond_set) == 0:
            if self.use_robust_ci:
                corr, _ = stats.spearmanr(data[:, x], data[:, y])
            else:
                corr = np.corrcoef(data[:, x], data[:, y])[0, 1]
            partial_corr = corr
        else:
            all_vars = [x, y] + list(cond_set)
            sub_data = data[:, all_vars]
            if self.use_robust_ci:
                ranked = np.apply_along_axis(stats.rankdata, 0, sub_data)
                corr_matrix = np.corrcoef(ranked.T)
            else:
                corr_matrix = np.corrcoef(sub_data.T)
            
            cond_num = np.linalg.cond(corr_matrix)
            reg = max(1e-6, 0.01 / cond_num) if not np.isinf(cond_num) else 0.1
            try:
                precision = inv(corr_matrix + reg * np.eye(len(all_vars)))
                partial_corr = -precision[0, 1] / np.sqrt(
                    abs(precision[0, 0] * precision[1, 1]) + 1e-10)
            except:
                partial_corr = 0.0
        
        partial_corr = np.clip(partial_corr, -0.9999, 0.9999)
        if np.isnan(partial_corr):
            result = CITestResult(True, 0.0, 1.0, tuple(cond_set))
        else:
            z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr + 1e-10))
            se = 1.0 / np.sqrt(max(1, n - len(cond_set) - 3))
            z_stat = abs(z) / se
            p_value = 2 * (1 - stats.norm.cdf(z_stat))
            result = CITestResult(p_value > self.alpha, z_stat, p_value, tuple(cond_set))
        
        self.cache.put(x, y, cond_set, result)
        return result
    
    def _test_edge(self, data: np.ndarray, edge: Tuple[int, int],
                   adj_matrix: np.ndarray, cond_size: int) -> Optional[Tuple]:
        x, y = edge
        neighbors = (set(np.where(adj_matrix[x, :] == 1)[0]) | 
                    set(np.where(adj_matrix[y, :] == 1)[0])) - {x, y}
        
        if len(neighbors) < cond_size:
            return None
        
        for cond_set in combinations(neighbors, cond_size):
            if self._ci_test(data, x, y, list(cond_set)).is_independent:
                return (x, y, set(cond_set))
        return None
    
    def learn_skeleton(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        n_vars = data.shape[1]
        adj_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        self._sep_sets = {}
        self.cache.clear()
        
        for cond_size in range(self.max_cond_size + 1):
            edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)
                    if adj_matrix[i, j] == 1]
            if not edges:
                break
            
            removals = []
            if self.n_workers > 1 and len(edges) > 10:
                with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                    futures = {ex.submit(self._test_edge, data, e, adj_matrix.copy(), 
                              cond_size): e for e in edges}
                    for f in as_completed(futures):
                        if f.result():
                            removals.append(f.result())
            else:
                for e in edges:
                    r = self._test_edge(data, e, adj_matrix, cond_size)
                    if r:
                        removals.append(r)
            
            for x, y, sep in removals:
                adj_matrix[x, y] = adj_matrix[y, x] = 0
                self._sep_sets[(min(x, y), max(x, y))] = sep
        
        return adj_matrix, self._sep_sets
    
    def orient_edges(self, skeleton: np.ndarray, sep_sets: Dict) -> np.ndarray:
        n_vars = skeleton.shape[0]
        dag = skeleton.copy()
        
        for z in range(n_vars):
            neighbors = list(np.where(skeleton[z, :] == 1)[0])
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    if skeleton[x, y] == 0:
                        key = (min(x, y), max(x, y))
                        if z not in sep_sets.get(key, set()):
                            dag[x, z], dag[z, x] = 1, 0
                            dag[y, z], dag[z, y] = 1, 0
        
        changed = True
        while changed:
            changed = False
            for i in range(n_vars):
                for j in range(n_vars):
                    if dag[i, j] == 1 and dag[j, i] == 1:
                        for k in range(n_vars):
                            if dag[k, i] == 1 and dag[i, k] == 0:
                                if dag[k, j] == 0 and dag[j, k] == 0:
                                    dag[i, j], dag[j, i] = 1, 0
                                    changed = True
                                    break
        return dag
    
    def fit(self, data: np.ndarray, variable_names: List[str] = None) -> np.ndarray:
        skeleton, sep_sets = self.learn_skeleton(data)
        return self.orient_edges(skeleton, sep_sets)
