"""
Counterfactual Engine - What-If Analysis for Trading
Author: Erdinc Erdogan
Purpose: Implements counterfactual reasoning for trading strategies, enabling what-if analysis
and counterfactual backtesting to evaluate alternative policy decisions.
References:
- Pearl (2009) "Causality: Models, Reasoning, and Inference" - Chapter 7
- Counterfactual Analysis in Economics
- Structural Causal Models (SCM)
Usage:
    engine = CounterfactualEngine(causal_model=scm)
    result = engine.query(scenario=CounterfactualScenario(type=CounterfactualType.POLICY, ...))
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from enum import IntEnum
import time


class CounterfactualType(IntEnum):
    """Types of counterfactual queries."""
    POLICY = 0        # What if policy X was different?
    REGIME = 1        # What if regime was different?
    EVENT = 2         # What if event X didn't happen?
    PARAMETER = 3     # What if parameter X was different?


@dataclass
class CounterfactualScenario:
    """Definition of a counterfactual scenario."""
    name: str
    scenario_type: CounterfactualType
    interventions: Dict[str, float]
    description: str
    regime: str = "NORMAL"
    timestamp_range: Tuple[float, float] = None


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    scenario: CounterfactualScenario
    factual_outcome: float
    counterfactual_outcome: float
    effect: float
    effect_pct: float
    confidence: float
    pnl_impact: float
    sharpe_impact: float
    max_drawdown_impact: float
    regime_breakdown: Dict[str, float]
    timestamp: float


class CounterfactualEngine:
    """
    Counterfactual Reasoning Engine.
    
    Implements Pearl's three-step counterfactual algorithm:
    1. Abduction: Infer exogenous noise from observed data
    2. Action: Modify structural equations for intervention
    3. Prediction: Compute counterfactual outcome
    
    Mathematical Foundation:
    Y_{X=x}(u) = counterfactual outcome for unit u if X had been x
    
    Reference: Pearl (2009) Chapter 7 "The Logic of Counterfactuals"
    """
    
    # Predefined policy scenarios
    POLICY_SCENARIOS = {
        "FED_NO_HIKE": CounterfactualScenario(
            name="Fed No Rate Hike",
            scenario_type=CounterfactualType.POLICY,
            interventions={"Yields": 0.0, "DXY": 0.0},
            description="What if the Fed had NOT raised rates?"
        ),
        "FED_DOUBLE_HIKE": CounterfactualScenario(
            name="Fed Double Rate Hike",
            scenario_type=CounterfactualType.POLICY,
            interventions={"Yields": 0.50, "DXY": 1.0},
            description="What if the Fed had raised rates by 50bps?"
        ),
        "NO_QE": CounterfactualScenario(
            name="No Quantitative Easing",
            scenario_type=CounterfactualType.POLICY,
            interventions={"Volume": -0.5, "VIX": 0.3},
            description="What if there was no QE?"
        ),
        "COVID_NO_CRASH": CounterfactualScenario(
            name="No COVID Crash",
            scenario_type=CounterfactualType.EVENT,
            interventions={"VIX": -1.5, "Volume": 0.0},
            description="What if COVID crash didn't happen?"
        ),
        "BTC_MAINSTREAM": CounterfactualScenario(
            name="BTC Mainstream Adoption",
            scenario_type=CounterfactualType.EVENT,
            interventions={"BTC": 2.0, "Volume": 0.5},
            description="What if BTC had mainstream adoption earlier?"
        ),
    }
    
    def __init__(
        self,
        scm=None,
        variable_names: List[str] = None
    ):
        self.scm = scm
        self.variable_names = variable_names or []
        self.counterfactual_history: List[CounterfactualResult] = []
        
    def set_scm(self, scm):
        """Set the structural causal model."""
        self.scm = scm
        self.variable_names = scm.variable_names
        
    def compute_counterfactual(
        self,
        scenario: CounterfactualScenario,
        observed_data: np.ndarray,
        outcome_variable: str,
        returns: np.ndarray = None
    ) -> CounterfactualResult:
        """
        Compute counterfactual outcome using Pearl's three-step algorithm.
        
        Args:
            scenario: Counterfactual scenario definition
            observed_data: Observed factual data
            outcome_variable: Variable to compute counterfactual for
            returns: Optional returns data for PnL analysis
        """
        if self.scm is None:
            raise ValueError("SCM must be set before computing counterfactuals")
        
        n_samples = observed_data.shape[0]
        outcome_idx = self.scm.var_to_idx[outcome_variable]
        
        # Step 1: ABDUCTION - Infer exogenous noise
        noise = self._abduction(observed_data)
        
        # Step 2: ACTION - Modify structural equations
        modified_scm = self._action(scenario.interventions)
        
        # Step 3: PREDICTION - Compute counterfactual
        counterfactual_data = self._prediction(modified_scm, noise, scenario.interventions)
        
        # Compute outcomes
        factual_outcome = np.mean(observed_data[:, outcome_idx])
        counterfactual_outcome = np.mean(counterfactual_data[:, outcome_idx])
        effect = counterfactual_outcome - factual_outcome
        effect_pct = effect / (abs(factual_outcome) + 1e-10) * 100
        
        # PnL analysis
        pnl_impact = 0.0
        sharpe_impact = 0.0
        max_dd_impact = 0.0
        
        if returns is not None:
            # Factual PnL
            factual_pnl = np.sum(returns)
            factual_sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            factual_dd = self._max_drawdown(np.cumsum(returns))
            
            # Counterfactual returns (scaled by effect)
            cf_returns = returns * (1 + effect / (abs(factual_outcome) + 1e-10))
            cf_pnl = np.sum(cf_returns)
            cf_sharpe = np.mean(cf_returns) / (np.std(cf_returns) + 1e-10) * np.sqrt(252)
            cf_dd = self._max_drawdown(np.cumsum(cf_returns))
            
            pnl_impact = cf_pnl - factual_pnl
            sharpe_impact = cf_sharpe - factual_sharpe
            max_dd_impact = cf_dd - factual_dd
        
        # Regime breakdown (if regime info available)
        regime_breakdown = {}
        
        # Confidence based on sample size and effect magnitude
        confidence = min(0.95, 1.0 - 1.0 / np.sqrt(n_samples))
        
        result = CounterfactualResult(
            scenario=scenario,
            factual_outcome=factual_outcome,
            counterfactual_outcome=counterfactual_outcome,
            effect=effect,
            effect_pct=effect_pct,
            confidence=confidence,
            pnl_impact=pnl_impact,
            sharpe_impact=sharpe_impact,
            max_drawdown_impact=max_dd_impact,
            regime_breakdown=regime_breakdown,
            timestamp=time.time()
        )
        
        self.counterfactual_history.append(result)
        return result
    
    def _abduction(self, observed_data: np.ndarray) -> np.ndarray:
        """
        Step 1: Abduction - Infer exogenous noise from observations.
        
        For linear SCM: U = Y - f(Pa(Y))
        """
        n_samples = observed_data.shape[0]
        noise = np.zeros_like(observed_data)
        
        for j in range(self.scm.n_vars):
            parents = np.where(self.scm.adjacency_matrix[:, j] == 1)[0]
            
            if len(parents) == 0:
                # No parents - noise is deviation from mean
                noise[:, j] = observed_data[:, j] - self.scm.intercepts[j]
            else:
                # Compute residuals
                predicted = self.scm.intercepts[j] +                            observed_data[:, parents] @ self.scm.coefficients[parents, j]
                noise[:, j] = observed_data[:, j] - predicted
        
        return noise
    
    def _action(self, interventions: Dict[str, float]):
        """
        Step 2: Action - Create modified SCM with interventions.
        
        For do(X=x): Remove all edges into X, set X = x
        """
        # Create copy of SCM
        modified_adj = self.scm.adjacency_matrix.copy()
        modified_intercepts = self.scm.intercepts.copy()
        
        for var, value in interventions.items():
            if var in self.scm.var_to_idx:
                idx = self.scm.var_to_idx[var]
                # Remove incoming edges
                modified_adj[:, idx] = 0
                # Set intercept to intervention value
                modified_intercepts[idx] = value
        
        return {
            'adjacency_matrix': modified_adj,
            'intercepts': modified_intercepts,
            'coefficients': self.scm.coefficients,
            'noise_vars': self.scm.noise_vars
        }
    
    def _prediction(
        self,
        modified_scm: Dict,
        noise: np.ndarray,
        interventions: Dict[str, float]
    ) -> np.ndarray:
        """
        Step 3: Prediction - Compute counterfactual outcomes.
        
        Use inferred noise with modified structural equations.
        """
        n_samples = noise.shape[0]
        counterfactual = np.zeros_like(noise)
        
        # Topological order
        order = self._topological_sort(modified_scm['adjacency_matrix'])
        
        for var_idx in order:
            var_name = self.variable_names[var_idx]
            
            if var_name in interventions:
                # Intervention variable - set to intervention value
                counterfactual[:, var_idx] = interventions[var_name]
            else:
                # Compute from structural equation with original noise
                parents = np.where(modified_scm['adjacency_matrix'][:, var_idx] == 1)[0]
                
                mean = modified_scm['intercepts'][var_idx]
                if len(parents) > 0:
                    mean += counterfactual[:, parents] @ modified_scm['coefficients'][parents, var_idx]
                
                counterfactual[:, var_idx] = mean + noise[:, var_idx]
        
        return counterfactual
    
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
        
        for i in range(n):
            if i not in order:
                order.append(i)
        
        return order
    
    def _max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-10)
        return np.min(drawdown)
    
    def run_policy_scenario(
        self,
        scenario_name: str,
        observed_data: np.ndarray,
        outcome_variable: str = "SPX",
        returns: np.ndarray = None
    ) -> CounterfactualResult:
        """Run a predefined policy scenario."""
        if scenario_name not in self.POLICY_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.POLICY_SCENARIOS[scenario_name]
        return self.compute_counterfactual(scenario, observed_data, outcome_variable, returns)
    
    def get_summary(self) -> Dict:
        """Get summary of counterfactual analyses."""
        if not self.counterfactual_history:
            return {"status": "no_analyses"}
        
        return {
            "total_analyses": len(self.counterfactual_history),
            "avg_effect": np.mean([r.effect for r in self.counterfactual_history]),
            "avg_pnl_impact": np.mean([r.pnl_impact for r in self.counterfactual_history]),
            "scenarios_analyzed": list(set(r.scenario.name for r in self.counterfactual_history)),
            "recent": [
                {
                    "scenario": r.scenario.name,
                    "effect": r.effect,
                    "effect_pct": r.effect_pct,
                    "pnl_impact": r.pnl_impact
                }
                for r in self.counterfactual_history[-5:]
            ]
        }


class CounterfactualBacktester:
    """
    Counterfactual Backtesting Engine.
    
    Extends traditional backtesting with counterfactual analysis:
    - What would PnL have been under different policies?
    - How sensitive is strategy to regime changes?
    - What-if analysis for risk management
    """
    
    def __init__(self, cf_engine: CounterfactualEngine):
        self.cf_engine = cf_engine
        self.backtest_results: List[Dict] = []
        
    def backtest_with_counterfactuals(
        self,
        strategy_returns: np.ndarray,
        market_data: np.ndarray,
        scenarios: List[str] = None,
        outcome_variable: str = "SPX"
    ) -> Dict:
        """
        Run backtest with counterfactual scenarios.
        
        Args:
            strategy_returns: Actual strategy returns
            market_data: Market data for causal analysis
            scenarios: List of scenario names to test
            outcome_variable: Target variable
        """
        if scenarios is None:
            scenarios = list(CounterfactualEngine.POLICY_SCENARIOS.keys())
        
        # Factual performance
        factual_pnl = np.sum(strategy_returns)
        factual_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
        factual_dd = self._max_drawdown(np.cumsum(strategy_returns))
        
        # Counterfactual performance
        cf_results = {}
        for scenario_name in scenarios:
            try:
                result = self.cf_engine.run_policy_scenario(
                    scenario_name, market_data, outcome_variable, strategy_returns
                )
                cf_results[scenario_name] = {
                    "effect": result.effect,
                    "effect_pct": result.effect_pct,
                    "pnl_impact": result.pnl_impact,
                    "sharpe_impact": result.sharpe_impact,
                    "confidence": result.confidence
                }
            except Exception as e:
                cf_results[scenario_name] = {"error": str(e)}
        
        backtest_result = {
            "factual": {
                "pnl": factual_pnl,
                "sharpe": factual_sharpe,
                "max_drawdown": factual_dd
            },
            "counterfactual": cf_results,
            "sensitivity": self._compute_sensitivity(cf_results),
            "timestamp": time.time()
        }
        
        self.backtest_results.append(backtest_result)
        return backtest_result
    
    def _compute_sensitivity(self, cf_results: Dict) -> Dict:
        """Compute strategy sensitivity to different scenarios."""
        valid_results = {k: v for k, v in cf_results.items() if "error" not in v}
        
        if not valid_results:
            return {"status": "no_valid_results"}
        
        effects = [v["effect_pct"] for v in valid_results.values()]
        pnl_impacts = [v["pnl_impact"] for v in valid_results.values()]
        
        return {
            "avg_effect_pct": np.mean(effects),
            "max_effect_pct": np.max(np.abs(effects)),
            "avg_pnl_impact": np.mean(pnl_impacts),
            "worst_case_pnl": np.min(pnl_impacts),
            "best_case_pnl": np.max(pnl_impacts),
            "most_sensitive_scenario": max(valid_results.keys(), 
                                          key=lambda k: abs(valid_results[k]["effect_pct"]))
        }
    
    def _max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-10)
        return np.min(drawdown)
