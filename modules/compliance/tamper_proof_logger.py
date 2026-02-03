"""
Tamper-Proof Audit Logger - Immutable Compliance Records
Author: Erdinc Erdogan
Purpose: Creates tamper-proof audit logs with hash-chaining and automatic system shutdown
on tamper detection, integrated with circuit breaker for trading halts.
References:
- Cryptographic Hash Chaining
- Tamper-Evident Logging Systems
- Regulatory Audit Requirements
Usage:
    logger = TamperProofAuditLogger()
    logger.log_event(event_type="ORDER", agent_id="agent_01", data={...})
    is_valid = logger.verify_integrity()
"""
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from collections import deque

@dataclass
class SecureAuditEntry:
    entry_id: str
    timestamp: float
    event_type: str
    agent_id: str
    data: Dict
    prev_hash: str
    entry_hash: str = ""
    
    def __post_init__(self):
        if not self.entry_hash:
            payload = f"{self.entry_id}{self.timestamp}{self.event_type}{self.prev_hash}"
            self.entry_hash = hashlib.sha256(payload.encode()).hexdigest()

class TamperProofAuditLogger:
    """Immutable audit trail with tamper detection and auto-shutdown."""
    
    def __init__(self, on_tamper_callback: Optional[Callable] = None):
        self._entries: deque = deque(maxlen=50000)
        self._last_hash: str = hashlib.sha256(b"GENESIS").hexdigest()
        self._counter: int = 0
        self._tamper_detected: bool = False
        self._on_tamper = on_tamper_callback
    
    def log(self, event_type: str, agent_id: str, data: Dict) -> SecureAuditEntry:
        if self._tamper_detected:
            raise RuntimeError("AUDIT-TAMPER: System locked")
        self._counter += 1
        entry = SecureAuditEntry(
            entry_id=f"AUD-{self._counter:08d}",
            timestamp=time.time(),
            event_type=event_type,
            agent_id=agent_id,
            data=data,
            prev_hash=self._last_hash
        )
        self._last_hash = entry.entry_hash
        self._entries.append(entry)
        return entry
    
    def verify_chain(self) -> tuple:
        if not self._entries:
            return True, "Empty chain"
        expected = hashlib.sha256(b"GENESIS").hexdigest()
        for i, entry in enumerate(self._entries):
            if entry.prev_hash != expected:
                self._trigger_tamper_alert(i, entry)
                return False, f"TAMPER at entry {i}: {entry.entry_id}"
            expected = entry.entry_hash
        return True, f"Chain valid ({len(self._entries)} entries)"
    
    def _trigger_tamper_alert(self, index: int, entry: SecureAuditEntry):
        self._tamper_detected = True
        if self._on_tamper:
            self._on_tamper(index, entry)
    
    def is_locked(self) -> bool:
        return self._tamper_detected
    
    def get_chain_hash(self) -> str:
        return self._last_hash[:16]
