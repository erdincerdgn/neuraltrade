"""
Dynamic Rule Engine - Hot-Reloadable Compliance Rules
Author: Erdinc Erdogan
Purpose: Enables hot-reloadable compliance rules without system restart, allowing real-time
rule updates and version-controlled rule management.
References:
- Rule Engine Design Patterns
- Hot-Reload Patterns in Production Systems
- Dynamic Policy Enforcement
Usage:
    engine = DynamicRuleEngine()
    engine.add_rule(Rule(rule_id="R001", condition="size > 10000", action="BLOCK"))
    result = engine.evaluate(order_context)
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime

@dataclass
class Rule:
    rule_id: str
    name: str
    condition: str  # Python expression
    action: str     # "BLOCK", "WARN", "LOG"
    enabled: bool = True
    version: int = 1

class DynamicRuleEngine:
    """Hot-reloadable compliance rule engine."""
    
    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._rule_hash: str = ""
        self._load_default_rules()
    
    def _load_default_rules(self):
        defaults = [
            Rule("R001", "MaxOrderSize", "order.quantity > 100", "BLOCK"),
            Rule("R002", "MinOrderSize", "order.quantity < 0.001", "BLOCK"),
            Rule("R003", "MaxDailyTrades", "daily_count > 1000", "WARN"),
            Rule("R004", "TradingHours", "not (9 <= hour <= 16)", "BLOCK"),
            Rule("R005", "MaxLeverage", "leverage > 10", "BLOCK"),
        ]
        for r in defaults:
            self._rules[r.rule_id] = r
        self._update_hash()
    
    def _update_hash(self):
        data = json.dumps({k: v.__dict__ for k, v in self._rules.items()}, sort_keys=True)
        self._rule_hash = hashlib.md5(data.encode()).hexdigest()[:8]
    
    def add_rule(self, rule: Rule) -> bool:
        self._rules[rule.rule_id] = rule
        self._update_hash()
        return True
    
    def update_rule(self, rule_id: str, **kwargs) -> bool:
        if rule_id not in self._rules:
            return False
        for k, v in kwargs.items():
            if hasattr(self._rules[rule_id], k):
                setattr(self._rules[rule_id], k, v)
        self._rules[rule_id].version += 1
        self._update_hash()
        return True
    
    def evaluate(self, context: Dict[str, Any]) -> List[tuple]:
        """Evaluate all rules against context. Returns list of (rule_id, action)."""
        violations = []
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            try:
                if eval(rule.condition, {"__builtins__": {}}, context):
                    violations.append((rule.rule_id, rule.action, rule.name))
            except:
                pass
        return violations
    
    def get_hash(self) -> str:
        return self._rule_hash
