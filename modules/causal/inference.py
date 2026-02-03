"""
Causal Inference AI Engine - DAG Builder and Do-Calculus
Author: Erdinc Erdogan
Purpose: Implements Judea Pearl's causality theory for building directed acyclic graphs (DAGs),
performing intervention analysis with do-calculus, and querying counterfactuals.
References:
- Pearl (2009) "Causality: Models, Reasoning, and Inference"
- Pearl (2018) "The Book of Why"
- Structural Causal Models (SCM)
Usage:
    graph = CausalGraph()
    graph.add_edge("X", "Y", strength=0.8)
    effect = graph.do_intervention("X", target="Y")
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from colorama import Fore, Style


class CausalGraph:
    """
    Causal Graph Builder.
    
    Judea Pearl's causality theory:
    - Building DAGs (Directed Acyclic Graphs)
    - Intervention analysis with Do-calculus
    - Counterfactual querying
    """
    
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # {from: [(to, strength), ...]}
        self.node_values = {}
        self.causal_chains = []
    
    def add_node(self, name: str, initial_value: float = 0):
        """Add a node."""
        self.nodes.add(name)
        self.node_values[name] = initial_value
        if name not in self.edges:
            self.edges[name] = []
    
    def add_edge(self, cause: str, effect: str, strength: float = 1.0, lag_days: int = 0):
        """
        Add a causal edge.
        
        Args:
            cause: Cause variable
            effect: Effect variable
            strength: Strength of the link (between -1 and 1)
            lag_days: Lag (days)
        """
        if cause not in self.nodes:
            self.add_node(cause)
        if effect not in self.nodes:
            self.add_node(effect)
        
        self.edges[cause].append({
            "target": effect,
            "strength": strength,
            "lag": lag_days
        })
    
    def build_financial_graph(self):
        """Basic financial causality graph."""
        # Macroeconomics
        self.add_node("FED_RATE")
        self.add_node("INFLATION")
        self.add_node("GDP_GROWTH")
        self.add_node("UNEMPLOYMENT")
        
        # Market
        self.add_node("CREDIT_COST")
        self.add_node("CORP_EARNINGS")
        self.add_node("STOCK_PRICES")
        self.add_node("BOND_YIELDS")
        
        # Connections
        self.add_edge("FED_RATE", "CREDIT_COST", 0.9, lag_days=0)
        self.add_edge("FED_RATE", "BOND_YIELDS", 0.8, lag_days=1)
        self.add_edge("CREDIT_COST", "CORP_EARNINGS", -0.6, lag_days=90)
        self.add_edge("CORP_EARNINGS", "STOCK_PRICES", 0.7, lag_days=30)
        self.add_edge("INFLATION", "FED_RATE", 0.5, lag_days=60)
        self.add_edge("GDP_GROWTH", "CORP_EARNINGS", 0.6, lag_days=30)
        self.add_edge("UNEMPLOYMENT", "GDP_GROWTH", -0.4, lag_days=30)
        
        print(f"{Fore.CYAN}🔗 Financial causality graph built{Style.RESET_ALL}", flush=True)
    
    def do_intervention(self, node: str, new_value: float) -> Dict:
        """
        Do-calculus: Intervention analysis.
        
        "If we forcibly set X to Y, what would happen to Z?"
        """
        # Save original values
        original_values = dict(self.node_values)
        
        # Intervention
        self.node_values[node] = new_value
        
        # Propagate changes
        affected = self._propagate_effects(node, new_value - original_values.get(node, 0))
        
        return {
            "intervention": {"node": node, "value": new_value},
            "affected_nodes": affected,
            "causal_chain": self._trace_chain(node)
        }
    
    def _propagate_effects(self, source: str, delta: float) -> Dict:
        """Propagate effects."""
        affected = {}
        queue = [(source, delta)]
        visited = set()
        
        while queue:
            node, d = queue.pop(0)
            
            if node in visited:
                continue
            visited.add(node)
            
            for edge in self.edges.get(node, []):
                target = edge["target"]
                effect = d * edge["strength"]
                
                if abs(effect) > 0.01:  # Minimum threshold
                    old_val = self.node_values.get(target, 0)
                    new_val = old_val + effect
                    self.node_values[target] = new_val
                    
                    affected[target] = {
                        "from": old_val,
                        "to": new_val,
                        "change": effect,
                        "lag_days": edge["lag"]
                    }
                    
                    queue.append((target, effect))
        
        return affected
    
    def _trace_chain(self, source: str, max_depth: int = 5) -> List[str]:
        """Trace causality chain."""
        chain = [source]
        current = source
        depth = 0
        
        while depth < max_depth:
            edges = self.edges.get(current, [])
            if not edges:
                break
            
            # Follow strongest link
            strongest = max(edges, key=lambda e: abs(e["strength"]))
            chain.append(f"--({strongest['strength']:+.2f})--> {strongest['target']}")
            current = strongest["target"]
            depth += 1
        
        return chain
    
    def counterfactual_query(self, 
                            observed: Dict[str, float],
                            intervention: Dict[str, float]) -> Dict:
        """
        Counterfactual query.
        
        "If X is observed, what would happen to Z if Y were different?"
        """
        # Apply observations
        for node, value in observed.items():
            self.node_values[node] = value
        
        # Intervention result
        result = self.do_intervention(
            list(intervention.keys())[0],
            list(intervention.values())[0]
        )
        
        return {
            "observed": observed,
            "intervention": intervention,
            "counterfactual_result": result
        }
    
    def analyze_scenario(self, scenario: str) -> Dict:
        """
        Scenario analysis.
        
        Example: "What happens to stocks if the FED hikes rates?"
        """
        scenarios = {
            "RATE_HIKE": {"FED_RATE": 0.5},
            "RATE_CUT": {"FED_RATE": -0.5},
            "RECESSION": {"GDP_GROWTH": -0.3, "UNEMPLOYMENT": 0.2},
            "INFLATION_SPIKE": {"INFLATION": 0.3},
            "EARNINGS_MISS": {"CORP_EARNINGS": -0.2}
        }
        
        if scenario not in scenarios:
            return {"error": f"Unknown scenario: {scenario}"}
        
        print(f"{Fore.CYAN}🔮 Scenario analysis: {scenario}{Style.RESET_ALL}", flush=True)
        
        # Apply each intervention
        total_affected = {}
        chains = []
        
        for node, delta in scenarios[scenario].items():
            original = self.node_values.get(node, 0)
            result = self.do_intervention(node, original + delta)
            total_affected.update(result["affected_nodes"])
            chains.extend(result["causal_chain"])
        
        # Stock effect
        stock_effect = total_affected.get("STOCK_PRICES", {}).get("change", 0)
        
        return {
            "scenario": scenario,
            "affected": total_affected,
            "causal_chains": chains,
            "stock_impact_pct": stock_effect * 100,
            "recommendation": "SELL" if stock_effect < -0.05 else "BUY" if stock_effect > 0.05 else "HOLD"
        }
    
    def generate_causal_report(self) -> str:
        """Causal report."""
        report = f"""
<causal_inference>
🔗 CAUSAL ANALYSIS
════════════════════════════════════════

📊 GRAPH STRUCTURE:
  • Nodes: {len(self.nodes)}
  • Edges: {sum(len(e) for e in self.edges.values())}

🔢 CURRENT VALUES:
"""
        for node, value in self.node_values.items():
            report += f"  • {node}: {value:.3f}\n"
        
        report += """
💡 Correlation ≠ Causation
   Do-calculus is used for intervention analysis.
</causal_inference>
"""
        return report
