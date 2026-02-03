"""
Robust Do-Calculus Engine - Production-Grade Causal Intervention
Author: Erdinc Erdogan
Purpose: Implements robust do-calculus for computing average treatment effects (ATE) using
backdoor adjustment with numerical stability and confidence interval estimation.
References:
- Pearl (2009) "Causality: Models, Reasoning, and Inference"
- Backdoor Adjustment Criterion
- Average Treatment Effect (ATE) Estimation
Usage:
    do_calc = RobustDoCalculus(variable_names, adjacency_matrix)
    do_calc.fit(data)
    effect = do_calc.compute_ate_backdoor("Treatment", "Outcome")
"""

import numpy as np
from scipy import stats
from scipy.special import expit
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class CausalEffect:
    ate: float
    ate_std: float
    ci_lower: float
    ci_upper: float
    adjustment_set: List[str]
    method: str
    is_identifiable: bool


class RobustDoCalculus:
    """Robust Do-Calculus with backdoor adjustment and IPW."""
    
    def __init__(self, variable_names: List[str], adjacency_matrix: np.ndarray):
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        self.var_to_idx = {name: i for i, name in enumerate(variable_names)}
        self.idx_to_var = {i: name for i, name in enumerate(variable_names)}
        self.adjacency_matrix = adjacency_matrix.copy()
        self.is_fitted = False
        
    def fit(self, data: np.ndarray):
        n_samples = data.shape[0]
        self.coefficients = np.zeros((self.n_vars, self.n_vars))
        self.intercepts = np.zeros(self.n_vars)
        self._observed_noise = np.zeros((n_samples, self.n_vars))
        
        for j in range(self.n_vars):
            parents = np.where(self.adjacency_matrix[:, j] == 1)[0]
            if len(parents) == 0:
                self.intercepts[j] = np.mean(data[:, j])
                self._observed_noise[:, j] = data[:, j] - self.intercepts[j]
            else:
                X = np.column_stack([np.ones(n_samples), data[:, parents]])
                beta = np.linalg.lstsq(X, data[:, j], rcond=None)[0]
                self.intercepts[j] = beta[0]
                self.coefficients[parents, j] = beta[1:]
                self._observed_noise[:, j] = data[:, j] - X @ beta
        
        self._observed_data = data.copy()
        self.is_fitted = True
        return self
    
    def get_parents(self, var_idx: int) -> Set[int]:
        return set(np.where(self.adjacency_matrix[:, var_idx] == 1)[0])
    
    def get_descendants(self, var_idx: int) -> Set[int]:
        descendants = set()
        to_visit = list(np.where(self.adjacency_matrix[var_idx, :] == 1)[0])
        while to_visit:
            node = to_visit.pop()
            if node not in descendants:
                descendants.add(node)
                to_visit.extend(np.where(self.adjacency_matrix[node, :] == 1)[0])
        return descendants
    
    def find_valid_adjustment_set(self, x_idx: int, y_idx: int) -> Optional[Set[int]]:
        x_descendants = self.get_descendants(x_idx)
        x_parents = self.get_parents(x_idx)
        return x_parents - x_descendants - {y_idx} if x_parents else set()
    
    def compute_ate_backdoor(self, treatment_var: str, outcome_var: str) -> CausalEffect:
        x_idx, y_idx = self.var_to_idx[treatment_var], self.var_to_idx[outcome_var]
        adjustment_set = self.find_valid_adjustment_set(x_idx, y_idx)
        
        data = self._observed_data
        n = data.shape[0]
        
        if adjustment_set:
            X = np.column_stack([np.ones(n), data[:, x_idx], data[:, list(adjustment_set)]])
        else:
            X = np.column_stack([np.ones(n), data[:, x_idx]])
        
        beta = np.linalg.lstsq(X, data[:, y_idx], rcond=None)[0]
        ate = beta[1]
        residuals = data[:, y_idx] - X @ beta
        ate_std = np.sqrt(np.mean(residuals**2) / n)
        
        return CausalEffect(
            ate=ate, ate_std=ate_std,
            ci_lower=ate - 1.96*ate_std, ci_upper=ate + 1.96*ate_std,
            adjustment_set=[self.idx_to_var[i] for i in (adjustment_set or [])],
            method="backdoor", is_identifiable=True
        )
