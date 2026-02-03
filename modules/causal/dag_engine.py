"""
Causal Discovery Engine - PC Algorithm Implementation
Author: Erdinc Erdogan
Purpose: Infers causal structures from financial time-series data using the PC algorithm
for directed acyclic graph (DAG) structure learning.
References:
- Spirtes, Glymour, Scheines (2000) "Causation, Prediction, and Search"
- Pearl (2009) "Causality: Models, Reasoning, and Inference"
- PC Algorithm for Constraint-Based Causal Discovery
Usage:
    engine = CausalDiscoveryEngine(alpha=0.05)
    engine.fit(market_data)
    edges = engine.get_dag_edges()
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import List, Tuple, Set, Dict, Optional


class CausalDiscoveryEngine:
    """PC Algorithm implementation for DAG structure learning from market data."""
    
    def __init__(self, alpha: float = 0.05, max_cond_set_size: int = 3):
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.skeleton = None
        self.separating_sets = {}
        self.dag = None
        self.variables = None
        
    def fit(self, data: pd.DataFrame) -> 'CausalDiscoveryEngine':
        self.variables = list(data.columns)
        self.skeleton = self._learn_skeleton(data)
        self.dag = self._orient_edges()
        return self
    
    def _learn_skeleton(self, data: pd.DataFrame) -> Dict[str, Set[str]]:
        skeleton = {var: set(self.variables) - {var} for var in self.variables}
        
        for cond_size in range(self.max_cond_set_size + 1):
            edges_to_test = []
            for i, x in enumerate(self.variables):
                for y in list(skeleton[x]):
                    if self.variables.index(y) > i:
                        neighbors = (skeleton[x] | skeleton[y]) - {x, y}
                        if len(neighbors) >= cond_size:
                            edges_to_test.append((x, y, neighbors))
            
            for x, y, neighbors in edges_to_test:
                if y not in skeleton[x]:
                    continue
                for cond_set in combinations(neighbors, cond_size):
                    cond_set = set(cond_set)
                    independent, _ = self._conditional_independence_test(data, x, y, cond_set)
                    if independent:
                        skeleton[x].discard(y)
                        skeleton[y].discard(x)
                        self.separating_sets[(x, y)] = cond_set
                        self.separating_sets[(y, x)] = cond_set
                        break
        return skeleton
    
    def _conditional_independence_test(self, data: pd.DataFrame, x: str, y: str, 
                                        cond_set: Set[str]) -> Tuple[bool, float]:
        n = len(data)
        if len(cond_set) == 0:
            corr, p_value = stats.pearsonr(data[x], data[y])
        else:
            cond_vars = list(cond_set)
            X_design = np.column_stack([np.ones(n), data[cond_vars].values])
            try:
                beta_x = np.linalg.lstsq(X_design, data[x].values, rcond=None)[0]
                resid_x = data[x].values - X_design @ beta_x
                beta_y = np.linalg.lstsq(X_design, data[y].values, rcond=None)[0]
                resid_y = data[y].values - X_design @ beta_y
                corr, _ = stats.pearsonr(resid_x, resid_y)
            except np.linalg.LinAlgError:
                return False, 0.0
        
        if abs(corr) >= 1.0:
            corr = np.sign(corr) * 0.9999
        z = 0.5 * np.log((1 + corr) / (1 - corr))
        se = 1.0 / np.sqrt(n - len(cond_set) - 3)
        p_value = 2 * (1 - stats.norm.cdf(abs(z) / se))
        return p_value > self.alpha, p_value
    
    def _orient_edges(self) -> Dict[str, Set[str]]:
        dag = {var: set() for var in self.variables}
        undirected = {var: self.skeleton[var].copy() for var in self.variables}
        
        for z in self.variables:
            neighbors = list(undirected[z])
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    if y not in undirected[x]:
                        sep_set = self.separating_sets.get((x, y), set())
                        if z not in sep_set:
                            dag[x].add(z)
                            dag[y].add(z)
                            undirected[x].discard(z)
                            undirected[z].discard(x)
                            undirected[y].discard(z)
                            undirected[z].discard(y)
        
        changed = True
        while changed:
            changed = False
            for x in self.variables:
                for y in list(undirected[x]):
                    for z in dag[x]:
                        if y in undirected[z] and y not in dag[x] and x not in dag[y]:
                            dag[z].add(y)
                            undirected[z].discard(y)
                            undirected[y].discard(z)
                            changed = True
        
        for x in self.variables:
            for y in list(undirected[x]):
                if not self._creates_cycle(dag, x, y):
                    dag[x].add(y)
                undirected[x].discard(y)
                undirected[y].discard(x)
        return dag
    
    def _creates_cycle(self, dag: Dict[str, Set[str]], source: str, target: str) -> bool:
        visited = set()
        stack = [target]
        while stack:
            node = stack.pop()
            if node == source:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(dag[node])
        return False
    
    def get_dag_edges(self) -> List[Tuple[str, str]]:
        if self.dag is None:
            raise ValueError("Must call fit() first")
        return [(p, c) for p, children in self.dag.items() for c in children]
    
    def get_parents(self, variable: str) -> Set[str]:
        if self.dag is None:
            raise ValueError("Must call fit() first")
        return {v for v, children in self.dag.items() if variable in children}
    
    def get_children(self, variable: str) -> Set[str]:
        if self.dag is None:
            raise ValueError("Must call fit() first")
        return self.dag.get(variable, set()).copy()
    
    def summary(self) -> str:
        if self.dag is None:
            return "No DAG learned. Call fit() first."
        edges = self.get_dag_edges()
        lines = ["CAUSAL DISCOVERY RESULTS", f"Variables: {self.variables}", 
                 f"Edges: {len(edges)}", "Relationships:"]
        for p, c in sorted(edges):
            lines.append(f"  {p} -> {c}")
        return "\n".join(lines)
