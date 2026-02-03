"""
Self-Matching Prevention Engine - Exchange Ban Protection
Author: Erdinc Erdogan
Purpose: Prevents trading against own sub-agents using O(1) hash-based lookup to avoid
self-matching violations that could result in exchange bans.
References:
- Exchange Self-Trade Prevention Rules
- Agent Hierarchy Management
- Hash-Based O(1) Lookup Patterns
Usage:
    guard = SelfMatchingGuard()
    guard.register_agent(AgentIdentity(agent_id="agent_01", parent_id="master"))
    is_self_match = guard.check_match(order_a, order_b)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, FrozenSet
from collections import defaultdict
import hashlib

@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    parent_id: Optional[str] = None
    hierarchy_hash: str = ""
    
    def __post_init__(self):
        if not self.hierarchy_hash:
            h = hashlib.md5(f"{self.agent_id}:{self.parent_id}".encode()).hexdigest()[:8]
            object.__setattr__(self, 'hierarchy_hash', h)

class SelfMatchingGuard:
    """O(1) self-matching prevention with agent hierarchy tracking."""
    
    def __init__(self):
        self._agent_registry: Dict[str, AgentIdentity] = {}
        self._hierarchy_groups: Dict[str, Set[str]] = defaultdict(set)
        self._active_orders: Dict[str, Dict] = {}  # order_id -> {agent_id, symbol, side}
        self._symbol_sides: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: {"BUY": set(), "SELL": set()})
    
    def register_agent(self, agent_id: str, parent_id: Optional[str] = None) -> AgentIdentity:
        """Register agent with O(1) hierarchy grouping."""
        identity = AgentIdentity(agent_id, parent_id)
        self._agent_registry[agent_id] = identity
        
        root = self._find_root(agent_id)
        self._hierarchy_groups[root].add(agent_id)
        return identity
    
    def _find_root(self, agent_id: str) -> str:
        """Find root parent for hierarchy grouping."""
        visited = set()
        current = agent_id
        while current in self._agent_registry:
            if current in visited:
                return current
            visited.add(current)
            parent = self._agent_registry[current].parent_id
            if parent is None:
                return current
            current = parent
        return agent_id
    
    def check_self_match(self, agent_id: str, symbol: str, side: str) -> tuple:
        """O(1) self-matching check against active orders."""
        opposite = "SELL" if side == "BUY" else "BUY"
        
        root = self._find_root(agent_id)
        sibling_agents = self._hierarchy_groups.get(root, {agent_id})
        
        active_opposite = self._symbol_sides[symbol][opposite]
        
        for sibling in sibling_agents:
            if sibling in active_opposite:
                return False, "SM-001", f"Self-match: {agent_id} vs {sibling} on {symbol}"
        
        return True, None, "OK"
    
    def register_order(self, order_id: str, agent_id: str, symbol: str, side: str):
        """Register active order for self-match tracking."""
        self._active_orders[order_id] = {"agent_id": agent_id, "symbol": symbol, "side": side}
        self._symbol_sides[symbol][side].add(agent_id)
    
    def unregister_order(self, order_id: str):
        """Remove filled/cancelled order from tracking."""
        if order_id in self._active_orders:
            order = self._active_orders.pop(order_id)
            self._symbol_sides[order["symbol"]][order["side"]].discard(order["agent_id"])
    
    def get_hierarchy_group(self, agent_id: str) -> Set[str]:
        """Get all agents in same hierarchy group."""
        root = self._find_root(agent_id)
        return self._hierarchy_groups.get(root, {agent_id})
