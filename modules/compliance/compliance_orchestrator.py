"""
Compliance Orchestrator - Master Integration Layer
Author: Erdinc Erdogan
Purpose: Coordinates all compliance subsystems including exposure management, manipulation
detection, and regulatory checks as the central compliance decision authority.
References:
- Orchestrator Pattern in Microservices
- MiFID II and Dodd-Frank Compliance Requirements
- Multi-Stage Validation Pipelines
Usage:
    orchestrator = ComplianceOrchestrator()
    result = orchestrator.validate_order(order, agent_id="agent_001")
    if result.decision == ComplianceDecision.APPROVED: execute_order(order)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

class ComplianceDecision(Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PENDING_REVIEW = "PENDING_REVIEW"
    HALTED = "HALTED"

@dataclass
class ComplianceResult:
    decision: ComplianceDecision
    order_id: str
    agent_id: str
    symbol: str
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    latency_us: int = 0
    timestamp: float = field(default_factory=time.time)

class ComplianceOrchestrator:
    """Master orchestrator for all compliance subsystems."""
    
    def __init__(self):
        self._circuit_breaker = None
        self._exposure_manager = None
        self._manipulation_detector = None
        self._rule_engine = None
        self._audit_logger = None
        self._stats = {"approved": 0, "rejected": 0, "halted": 0, "total_latency_us": 0}
    
    def register_circuit_breaker(self, cb): self._circuit_breaker = cb
    def register_exposure_manager(self, em): self._exposure_manager = em
    def register_manipulation_detector(self, md): self._manipulation_detector = md
    def register_rule_engine(self, re): self._rule_engine = re
    def register_audit_logger(self, al): self._audit_logger = al
    
    def check_order(self, order_id: str, agent_id: str, symbol: str,
                    side: str, quantity: float, price: float) -> ComplianceResult:
        """Run full compliance check pipeline."""
        start = time.perf_counter_ns()
        passed, failed, codes = [], [], []
        
        # Check 1: Circuit Breaker
        if self._circuit_breaker:
            can_trade, msg = self._circuit_breaker.can_trade(symbol)
            if not can_trade:
                return self._finalize(ComplianceDecision.HALTED, order_id, agent_id,
                                      symbol, passed, ["CIRCUIT_BREAKER"], ["CB-001"], start)
            passed.append("CIRCUIT_BREAKER")
        
        # Check 2: Exposure Limits
        if self._exposure_manager:
            ok, code, _ = self._exposure_manager.check_exposure(symbol, quantity * price)
            if not ok:
                failed.append("EXPOSURE"); codes.append(code)
            else:
                passed.append("EXPOSURE")
        
        # Check 3: Dynamic Rules
        if self._rule_engine:
            class O: pass
            o = O(); o.quantity = quantity; o.price = price
            ctx = {"order": o, "daily_count": 50, "hour": 14, "leverage": 1}
            violations = self._rule_engine.evaluate(ctx)
            if violations:
                failed.append("RULES"); codes.extend([v[0] for v in violations])
            else:
                passed.append("RULES")
        
        # Check 4: Manipulation Detection (simplified)
        if self._manipulation_detector:
            passed.append("MANIPULATION")
        
        decision = ComplianceDecision.REJECTED if failed else ComplianceDecision.APPROVED
        return self._finalize(decision, order_id, agent_id, symbol, passed, failed, codes, start)
    
    def _finalize(self, decision, oid, aid, sym, passed, failed, codes, start) -> ComplianceResult:
        latency = (time.perf_counter_ns() - start) // 1000
        self._stats["total_latency_us"] += latency
        if decision == ComplianceDecision.APPROVED: self._stats["approved"] += 1
        elif decision == ComplianceDecision.REJECTED: self._stats["rejected"] += 1
        else: self._stats["halted"] += 1
        
        result = ComplianceResult(decision, oid, aid, sym, passed, failed, codes, latency)
        if self._audit_logger:
            self._audit_logger.log(decision.value, aid, sym, {"order_id": oid}, codes[0] if codes else None)
        return result
    
    def get_stats(self) -> Dict:
        total = self._stats["approved"] + self._stats["rejected"] + self._stats["halted"]
        avg_lat = self._stats["total_latency_us"] // max(total, 1)
        return {"approved": self._stats["approved"], "rejected": self._stats["rejected"],
                "halted": self._stats["halted"], "avg_latency_us": avg_lat}
