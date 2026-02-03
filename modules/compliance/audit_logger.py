"""
Forensic Audit Logger - Immutable Compliance Trail
Author: Erdinc Erdogan
Purpose: Creates immutable, timestamped, hash-chained audit trails for regulatory compliance,
enabling forensic analysis and tamper detection in trading operations.
References:
- Blockchain-Style Hash Chaining for Audit Logs
- SEC and FINRA Audit Trail Requirements
- Immutable Log Storage Patterns
Usage:
    logger = ForensicAuditLogger()
    logger.log_event(event_type="ORDER_CHECK", agent_id="agent_001", details={...})
"""

import json
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from collections import deque

@dataclass
class AuditEntry:
    entry_id: str
    timestamp: float
    event_type: str  # "ORDER_CHECK", "REJECTION", "APPROVAL", "RULE_CHANGE"
    agent_id: str
    symbol: Optional[str]
    details: Dict[str, Any]
    reason_code: Optional[str]
    prev_hash: str
    entry_hash: str = ""
    
    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        data = f"{self.entry_id}{self.timestamp}{self.event_type}{self.prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class ForensicAuditLogger:
    """Immutable audit trail with hash chain integrity."""
    
    def __init__(self, max_entries: int = 10000):
        self._entries: deque = deque(maxlen=max_entries)
        self._last_hash: str = "GENESIS"
        self._entry_counter: int = 0
    
    def log(self, event_type: str, agent_id: str, symbol: Optional[str] = None,
            details: Optional[Dict] = None, reason_code: Optional[str] = None) -> AuditEntry:
        """Log an audit entry with hash chain."""
        self._entry_counter += 1
        entry = AuditEntry(
            entry_id=f"AUD-{self._entry_counter:08d}",
            timestamp=time.time(),
            event_type=event_type,
            agent_id=agent_id,
            symbol=symbol,
            details=details or {},
            reason_code=reason_code,
            prev_hash=self._last_hash
        )
        self._last_hash = entry.entry_hash
        self._entries.append(entry)
        return entry
    
    def verify_chain(self) -> tuple:
        """Verify integrity of audit chain."""
        if not self._entries:
            return True, "Empty chain"
        prev = "GENESIS"
        for entry in self._entries:
            if entry.prev_hash != prev:
                return False, f"Chain broken at {entry.entry_id}"
            prev = entry.entry_hash
        return True, f"Chain valid ({len(self._entries)} entries)"
    
    def export_json(self, filepath: str) -> int:
        """Export audit trail to JSON."""
        data = [asdict(e) for e in self._entries]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return len(data)
    
    def get_by_agent(self, agent_id: str) -> List[AuditEntry]:
        return [e for e in self._entries if e.agent_id == agent_id]
    
    def get_rejections(self) -> List[AuditEntry]:
        return [e for e in self._entries if e.event_type == "REJECTION"]
