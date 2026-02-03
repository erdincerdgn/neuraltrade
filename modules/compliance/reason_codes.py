"""
Compliance Reason Codes - Standardized Rejection Taxonomy
Author: Erdinc Erdogan
Purpose: Provides standardized rejection codes for compliance failures, enabling
forensic audit trails and consistent rejection handling across the system.
References:
- FIX Protocol Rejection Reason Codes
- Regulatory Reporting Requirements
- Audit Trail Best Practices
Usage:
    code = ReasonCodes.get_code("EXP-001")
    log_rejection(order_id, code.code, code.description)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ReasonCategory(Enum):
    EXPOSURE = "EXP"
    MANIPULATION = "MAN"
    LATENCY = "LAT"
    RULE = "RUL"
    EXCHANGE = "EXC"
    SYSTEM = "SYS"

@dataclass
class ReasonCode:
    code: str
    category: ReasonCategory
    description: str
    severity: str  # "BLOCK", "WARN", "INFO"

class ComplianceReasonCodes:
    """Registry of all compliance rejection reason codes."""
    
    # Exposure violations
    EXP_001 = ReasonCode("EXP-001", ReasonCategory.EXPOSURE, "Max position size exceeded", "BLOCK")
    EXP_002 = ReasonCode("EXP-002", ReasonCategory.EXPOSURE, "Portfolio exposure limit exceeded", "BLOCK")
    EXP_003 = ReasonCode("EXP-003", ReasonCategory.EXPOSURE, "Correlated asset exposure exceeded", "BLOCK")
    EXP_004 = ReasonCode("EXP-004", ReasonCategory.EXPOSURE, "Leverage limit exceeded", "BLOCK")
    
    # Manipulation violations
    MAN_001 = ReasonCode("MAN-001", ReasonCategory.MANIPULATION, "Wash trade detected", "BLOCK")
    MAN_002 = ReasonCode("MAN-002", ReasonCategory.MANIPULATION, "Spoofing pattern detected", "BLOCK")
    MAN_003 = ReasonCode("MAN-003", ReasonCategory.MANIPULATION, "Layering pattern detected", "BLOCK")
    
    # Latency violations
    LAT_001 = ReasonCode("LAT-001", ReasonCategory.LATENCY, "Compliance check timeout", "BLOCK")
    
    # Rule violations
    RUL_001 = ReasonCode("RUL-001", ReasonCategory.RULE, "Trading hours violation", "BLOCK")
    RUL_002 = ReasonCode("RUL-002", ReasonCategory.RULE, "Min order size violation", "BLOCK")
    RUL_003 = ReasonCode("RUL-003", ReasonCategory.RULE, "Max daily trades exceeded", "BLOCK")
    
    # System
    SYS_001 = ReasonCode("SYS-001", ReasonCategory.SYSTEM, "Compliance engine unavailable", "BLOCK")
    
    @classmethod
    def get_all_codes(cls) -> dict:
        return {k: v for k, v in cls.__dict__.items() if isinstance(v, ReasonCode)}
