"""
PC-Stable Engine - Parallel Causal Discovery Algorithm
Author: Erdinc Erdogan
Purpose: Implements the PC-Stable algorithm with parallel conditional independence testing
for efficient and order-independent causal structure learning from data.
References:
- Colombo & Maathuis (2014) "Order-independent Constraint-based Causal Structure Learning"
- Spirtes, Glymour, Scheines (2000) "Causation, Prediction, and Search"
- Parallel Computing for Statistical Inference
Usage:
    ci_test = RobustConditionalIndependenceTest(method="adaptive")
    engine = PCStableEngine(ci_test, n_jobs=-1)
    dag = engine.learn_structure(data, variable_names)
"""

import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set
from itertools import combinations
import multiprocessing as mp


class PCStableEngine:
    """PC-Stable algorithm with parallel CI testing."""
    
    def __init__(self, ci_test, alpha=0.05, max_cond_set_size=3, n_jobs=-1):
        self.ci_test = ci_test
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.stats = {'ci_tests_performed': 0, 'cache_hits': 0, 'total_runtime': 0.0}
        
    def learn_structure(self, data: np.ndarray, variable_names: List[str]) -> Dict[str, Set[str]]:
        start_time = time.time()
        skeleton, separating_sets = self._learn_skeleton_parallel(data, variable_names)
        dag = self._orient_edges_stable(skeleton, separating_sets, variable_names)
        self.stats['total_runtime'] = time.time() - start_time
        return dag
    
    def _learn_skeleton_parallel(self, data, variable_names):
        n_vars = len(variable_names)
        skeleton = {var: set(variable_names) - {var} for var in variable_names}
        separating_sets = {}
        
        for cond_size in range(self.max_cond_set_size + 1):
            test_jobs = []
            for i, x in enumerate(variable_names):
                for y in list(skeleton[x]):
                    if variable_names.index(y) > i:
                        neighbors = (skeleton[x] | skeleton[y]) - {x, y}
                        if len(neighbors) >= cond_size:
                            for cond_set in combinations(neighbors, cond_size):
                                test_jobs.append((x, y, set(cond_set)))
            
            if not test_jobs:
                continue
                
            results = self._parallel_ci_tests(data, test_jobs, variable_names)
            
            edges_to_remove = []
            for (x, y, cond_set), (independent, p_value) in results.items():
                if independent and y in skeleton[x]:
                    edges_to_remove.append((x, y, cond_set))
            
            for x, y, cond_set in edges_to_remove:
                if y in skeleton[x]:
                    skeleton[x].discard(y)
                    skeleton[y].discard(x)
                    separating_sets[(x, y)] = cond_set
                    separating_sets[(y, x)] = cond_set
        
        return skeleton, separating_sets
    
    def _parallel_ci_tests(self, data, test_jobs, variable_names):
        var_to_idx = {name: i for i, name in enumerate(variable_names)}
        results = {}
        
        index_jobs = []
        for x, y, cond_set in test_jobs:
            x_idx, y_idx = var_to_idx[x], var_to_idx[y]
            cond_indices = [var_to_idx[var] for var in cond_set]
            index_jobs.append((x_idx, y_idx, cond_indices, x, y, frozenset(cond_set)))
        
        batch_size = max(1, len(index_jobs) // (self.n_jobs * 4))
        batches = [index_jobs[i:i + batch_size] for i in range(0, len(index_jobs), batch_size)]
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_batch = {executor.submit(self._ci_test_batch, data, batch): batch for batch in batches}
            for future in as_completed(future_to_batch):
                results.update(future.result())
        
        return results
    
    def _ci_test_batch(self, data, batch):
        results = {}
        for x_idx, y_idx, cond_indices, x_name, y_name, cond_set in batch:
            independent, test_stat, p_value = self.ci_test.test(data, x_idx, y_idx, cond_indices)
            results[(x_name, y_name, cond_set)] = (independent, p_value)
            self.stats['ci_tests_performed'] += 1
        return results
    
    def _orient_edges_stable(self, skeleton, separating_sets, variable_names):
        dag = {var: set() for var in variable_names}
        undirected = {var: skeleton[var].copy() for var in variable_names}
        
        # Orient v-structures
        for z in variable_names:
            neighbors = list(undirected[z])
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    if y not in undirected[x]:
                        sep_set = separating_sets.get((x, y), set())
                        if z not in sep_set:
                            if z in undirected[x] and z in undirected[y]:
                                dag[x].add(z)
                                dag[y].add(z)
                                undirected[x].discard(z)
                                undirected[z].discard(x)
                                undirected[y].discard(z)
                                undirected[z].discard(y)
        
        # Apply Meek's rules
        changed = True
        while changed:
            changed = False
            for x in variable_names:
                for z in dag[x]:
                    for y in list(undirected[z]):
                        if y not in dag[x] and x not in dag[y]:
                            dag[z].add(y)
                            undirected[z].discard(y)
                            undirected[y].discard(z)
                            changed = True
        
        # Orient remaining edges
        for x in variable_names:
            for y in list(undirected[x]):
                if not self._creates_cycle(dag, x, y):
                    dag[x].add(y)
                undirected[x].discard(y)
                undirected[y].discard(x)
        
        return dag
    
    def _creates_cycle(self, dag, source, target):
        visited, stack = set(), [target]
        while stack:
            node = stack.pop()
            if node == source:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(dag[node])
        return False
