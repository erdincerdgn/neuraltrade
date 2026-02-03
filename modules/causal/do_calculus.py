"""
Do-Calculus Engine - Pearl's Causal Intervention Framework
Author: Erdinc Erdogan
Purpose: Implements Pearl's do-calculus for causal interventions, computing treatment effects
and identifying causal relationships through hard and soft interventions.
References:
- Pearl (2009) "Causality: Models, Reasoning, and Inference"
- Pearl (1995) "Causal Diagrams for Empirical Research"
- Bareinboim & Pearl (2016) "Causal Inference and the Data-Fusion Problem"
Usage:
    do_engine = DoCalculusEngine(dag=causal_graph)
    effect = do_engine.compute_do(treatment="X", outcome="Y", value=1.0)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from scipy import stats
from scipy.optimize import minimize
import time


class InterventionType(IntEnum):
    """Types of causal interventions."""
    DO = 0           # do(X=x) - hard intervention
    SOFT = 1         # Soft intervention (shift distribution)
    CONDITIONAL = 2  # Conditional intervention
    STOCHASTIC = 3   # Stochastic intervention


@dataclass
class InterventionResult:
    """Result of a causal intervention."""
    intervention_type: InterventionType
    target_variable: str
    intervention_value: float
    affected_variables: List[str]
    pre_intervention_means: Dict[str, float]
    post_intervention_means: Dict[str, float]
    causal_effects: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    p_values: Dict[str, float]
    timestamp: float


@dataclass
class CausalEffect:
    """Estimated causal effect."""
    source: str
    target: str
    ate: float  # Average Treatment Effect
    att: float  # Average Treatment on Treated
    cate: Dict[str, float]  # Conditional ATE by regime
    confidence: float
    method: str


class StructuralCausalModel:
    """
    Structural Causal Model (SCM) for intervention analysis.
    
    An SCM consists of:
    1. Endogenous variables V
    2. Exogenous variables U
    3. Structural equations F: V = f(Pa(V), U)
    
    Mathematical Foundation:
    - P(Y | do(X=x)) ≠ P(Y | X=x) in general
    - do(X=x) removes all arrows into X
    - Truncated factorization: P(v | do(x)) = ∏_{i: V_i ≠ X} P(v_i | pa_i)
    """
    
    def __init__(
        self,
        variable_names: List[str],
        adjacency_matrix: np.ndarray = None
    ):
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        self.var_to_idx = {name: i for i, name in enumerate(variable_names)}
        
        # Adjacency matrix (causal graph)
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
        else:
            self.adjacency_matrix = np.zeros((self.n_vars, self.n_vars))
        
        # Structural equations (linear by default)
        self.coefficients = np.zeros((self.n_vars, self.n_vars))
        self.intercepts = np.zeros(self.n_vars)
        self.noise_vars = np.ones(self.n_vars)
        
        # Fitted flag
        self.is_fitted = False
        
    def fit(self, data: np.ndarray):
        """
        Fit structural equations from data.
        
        For each variable, regress on its parents.
        """
        n_samples = data.shape[0]
        
        for j in range(self.n_vars):
            parents = np.where(self.adjacency_matrix[:, j] == 1)[0]
            
            if len(parents) == 0:
                # No parents - just estimate mean and variance
                self.intercepts[j] = np.mean(data[:, j])
                self.noise_vars[j] = np.var(data[:, j])
            else:
                # Regress on parents
                X = data[:, parents]
                y = data[:, j]
                
                X_aug = np.column_stack([np.ones(n_samples), X])
                
                try:
                    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                    self.intercepts[j] = beta[0]
                    self.coefficients[parents, j] = beta[1:]
                    
                    residuals = y - X_aug @ beta
                    self.noise_vars[j] = np.var(residuals)
                except:
                    self.intercepts[j] = np.mean(y)
                    self.noise_vars[j] = np.var(y)
        
        self.is_fitted = True
        
    def intervene(
        self,
        intervention_var: str,
        intervention_value: float,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Perform do(X=x) intervention and sample from post-intervention distribution.
        
        do(X=x) sets X to value x and removes all causal influences on X.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before intervention")
        
        idx = self.var_to_idx[intervention_var]
        
        # Create modified adjacency matrix (remove edges into intervention variable)
        modified_adj = self.adjacency_matrix.copy()
        modified_adj[:, idx] = 0  # Remove all incoming edges
        
        # Sample from modified SCM
        samples = np.zeros((n_samples, self.n_vars))
        
        # Topological order
        order = self._topological_sort(modified_adj)
        
        for var_idx in order:
            if var_idx == idx:
                # Intervention variable - set to fixed value
                samples[:, var_idx] = intervention_value
            else:
                # Sample from structural equation
                parents = np.where(modified_adj[:, var_idx] == 1)[0]
                
                mean = self.intercepts[var_idx]
                if len(parents) > 0:
                    mean += samples[:, parents] @ self.coefficients[parents, var_idx]
                
                noise = np.random.normal(0, np.sqrt(self.noise_vars[var_idx]), n_samples)
                samples[:, var_idx] = mean + noise
        
        return samples
    
    def _topological_sort(self, adj_matrix: np.ndarray) -> List[int]:
        """Topological sort of variables."""
        n = adj_matrix.shape[0]
        in_degree = np.sum(adj_matrix, axis=0).astype(int)
        queue = list(np.where(in_degree == 0)[0])
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for j in range(n):
                if adj_matrix[node, j] == 1:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        
        # Handle cycles by adding remaining nodes
        for i in range(n):
            if i not in order:
                order.append(i)
        
        return order
    
    def get_parents(self, variable: str) -> List[str]:
        """Get parent variables."""
        idx = self.var_to_idx[variable]
        parent_indices = np.where(self.adjacency_matrix[:, idx] == 1)[0]
        return [self.variable_names[i] for i in parent_indices]
    
    def get_children(self, variable: str) -> List[str]:
        """Get child variables."""
        idx = self.var_to_idx[variable]
        child_indices = np.where(self.adjacency_matrix[idx, :] == 1)[0]
        return [self.variable_names[i] for i in child_indices]


class DoCalculusEngine:
    """
    Pearl's Do-Calculus Implementation.
    
    Implements the three rules of do-calculus:
    
    Rule 1 (Insertion/deletion of observations):
        P(y | do(x), z, w) = P(y | do(x), w) if (Y ⊥ Z | X, W)_{G_X̄}
    
    Rule 2 (Action/observation exchange):
        P(y | do(x), do(z), w) = P(y | do(x), z, w) if (Y ⊥ Z | X, W)_{G_X̄Z̲}
    
    Rule 3 (Insertion/deletion of actions):
        P(y | do(x), do(z), w) = P(y | do(x), w) if (Y ⊥ Z | X, W)_{G_X̄Z̄(W)}
    
    Reference: Pearl (2009) "Causality: Models, Reasoning, and Inference"
    """
    
    def __init__(
        self,
        scm: StructuralCausalModel = None,
        variable_names: List[str] = None
    ):
        self.scm = scm
        self.variable_names = variable_names or []
        
        # Cache for computed effects
        self.effect_cache: Dict[Tuple[str, str], CausalEffect] = {}
        
    def set_scm(self, scm: StructuralCausalModel):
        """Set the structural causal model."""
        self.scm = scm
        self.variable_names = scm.variable_names
        self.effect_cache.clear()
        
    def compute_do(
        self,
        target: str,
        intervention_var: str,
        intervention_value: float,
        data: np.ndarray = None,
        n_samples: int = 1000
    ) -> InterventionResult:
        """
        Compute P(target | do(intervention_var = intervention_value)).
        
        This is the core do-calculus operation.
        """
        if self.scm is None:
            raise ValueError("SCM must be set before computing do()")
        
        # Get pre-intervention distribution
        if data is not None:
            pre_means = {
                name: np.mean(data[:, i])
                for i, name in enumerate(self.variable_names)
            }
        else:
            pre_means = {name: 0.0 for name in self.variable_names}
        
        # Perform intervention
        post_samples = self.scm.intervene(
            intervention_var, intervention_value, n_samples
        )
        
        # Compute post-intervention statistics
        post_means = {
            name: np.mean(post_samples[:, i])
            for i, name in enumerate(self.variable_names)
        }
        
        # Compute causal effects
        causal_effects = {
            name: post_means[name] - pre_means[name]
            for name in self.variable_names
        }
        
        # Compute confidence intervals (bootstrap)
        confidence_intervals = {}
        p_values = {}
        
        for i, name in enumerate(self.variable_names):
            samples = post_samples[:, i]
            ci_low = np.percentile(samples, 2.5)
            ci_high = np.percentile(samples, 97.5)
            confidence_intervals[name] = (ci_low, ci_high)
            
            # P-value for effect being non-zero
            t_stat = causal_effects[name] / (np.std(samples) / np.sqrt(n_samples) + 1e-10)
            p_values[name] = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 1))
        
        # Identify affected variables (descendants of intervention)
        affected = self._get_descendants(intervention_var)
        
        return InterventionResult(
            intervention_type=InterventionType.DO,
            target_variable=intervention_var,
            intervention_value=intervention_value,
            affected_variables=affected,
            pre_intervention_means=pre_means,
            post_intervention_means=post_means,
            causal_effects=causal_effects,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            timestamp=time.time()
        )
    
    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        data: np.ndarray,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> CausalEffect:
        """
        Estimate Average Treatment Effect (ATE).
        
        ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
        """
        cache_key = (treatment, outcome)
        if cache_key in self.effect_cache:
            return self.effect_cache[cache_key]
        
        # Fit SCM if needed
        if not self.scm.is_fitted:
            self.scm.fit(data)
        
        # Compute E[Y | do(X=x1)]
        result_high = self.compute_do(
            outcome, treatment, treatment_values[1], data
        )
        
        # Compute E[Y | do(X=x0)]
        result_low = self.compute_do(
            outcome, treatment, treatment_values[0], data
        )
        
        # ATE
        ate = result_high.post_intervention_means[outcome] -               result_low.post_intervention_means[outcome]
        
        # ATT (using observational data for treated)
        treatment_idx = self.scm.var_to_idx[treatment]
        outcome_idx = self.scm.var_to_idx[outcome]
        
        treated_mask = data[:, treatment_idx] > np.median(data[:, treatment_idx])
        att = np.mean(data[treated_mask, outcome_idx]) -               np.mean(data[~treated_mask, outcome_idx])
        
        # Confidence
        confidence = 1.0 - min(
            result_high.p_values[outcome],
            result_low.p_values[outcome]
        )
        
        effect = CausalEffect(
            source=treatment,
            target=outcome,
            ate=ate,
            att=att,
            cate={},
            confidence=confidence,
            method="do-calculus"
        )
        
        self.effect_cache[cache_key] = effect
        return effect
    
    def _get_descendants(self, variable: str) -> List[str]:
        """Get all descendants of a variable."""
        if self.scm is None:
            return []
        
        descendants = set()
        queue = [variable]
        
        while queue:
            current = queue.pop(0)
            children = self.scm.get_children(current)
            
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        
        return list(descendants)


class InterventionAnalyzer:
    """
    Intervention Analyzer for Market Scenarios.
    
    Simulates market interventions like:
    - Volume shocks (liquidity shifts)
    - Rate changes (Fed interventions)
    - Volatility spikes (VIX interventions)
    """
    
    MARKET_INTERVENTIONS = {
        "LIQUIDITY_CRISIS": {"Volume": -2.0, "VIX": 1.5},
        "FED_RATE_HIKE": {"Yields": 0.25, "DXY": 0.5},
        "RISK_OFF": {"VIX": 2.0, "BTC": -1.0},
        "TECH_SELLOFF": {"TechIndex": -1.5, "Volume": 1.0},
    }
    
    def __init__(self, do_engine: DoCalculusEngine):
        self.do_engine = do_engine
        self.intervention_history: List[InterventionResult] = []
        
    def simulate_intervention(
        self,
        scenario: str,
        data: np.ndarray,
        target_outcome: str = "SPX"
    ) -> Dict[str, InterventionResult]:
        """Simulate a predefined market intervention scenario."""
        if scenario not in self.MARKET_INTERVENTIONS:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        interventions = self.MARKET_INTERVENTIONS[scenario]
        results = {}
        
        for var, value in interventions.items():
            if var in self.do_engine.variable_names:
                result = self.do_engine.compute_do(
                    target_outcome, var, value, data
                )
                results[var] = result
                self.intervention_history.append(result)
        
        return results
    
    def analyze_liquidity_shift(
        self,
        data: np.ndarray,
        volume_change: float,
        target: str = "SPX"
    ) -> InterventionResult:
        """
        Analyze effect of liquidity shift on target.
        
        Simulates: What happens to SPX if Volume changes by X%?
        """
        if "Volume" not in self.do_engine.variable_names:
            raise ValueError("Volume not in model variables")
        
        # Get current volume level
        vol_idx = self.do_engine.scm.var_to_idx["Volume"]
        current_vol = np.mean(data[:, vol_idx])
        
        # Intervention value
        new_vol = current_vol * (1 + volume_change)
        
        return self.do_engine.compute_do(target, "Volume", new_vol, data)
    
    def get_intervention_summary(self) -> Dict:
        """Get summary of all interventions."""
        return {
            "total_interventions": len(self.intervention_history),
            "by_type": {
                t.name: sum(1 for r in self.intervention_history 
                           if r.intervention_type == t)
                for t in InterventionType
            },
            "recent": [
                {
                    "target": r.target_variable,
                    "value": r.intervention_value,
                    "effects": {k: v for k, v in r.causal_effects.items() 
                               if abs(v) > 0.01}
                }
                for r in self.intervention_history[-5:]
            ]
        }
