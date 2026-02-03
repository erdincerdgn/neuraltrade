"""
Resilient Connection Manager with Auto-Reconnect
Author: Erdinc Erdogan
Purpose: Manages cloud connections with heartbeat monitoring, exponential backoff reconnection, and connection state tracking.
References:
- Exponential Backoff Algorithm
- Connection Heartbeat Patterns
- Circuit Breaker Pattern
Usage:
    manager = ResilientConnectionManager(config=ReconnectConfig(max_attempts=10))
    manager.connect()
"""

import time
import random
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum
from datetime import datetime, timezone

class ConnectionState(Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    DEGRADED = "DEGRADED"

@dataclass
class ReconnectConfig:
    """Configuration for reconnection behavior."""
    initial_delay: float = 1.0
    max_delay: float = 300.0  # 5 minutes max
    multiplier: float = 2.0
    jitter: float = 0.1
    max_attempts: int = 10

class ResilientConnectionManager:
    """Manages cloud connection with heartbeat and auto-reconnect."""
    
    def __init__(self, config: Optional[ReconnectConfig] = None):
        self.config = config or ReconnectConfig()
        self._state = ConnectionState.DISCONNECTED
        self._last_heartbeat: Optional[float] = None
        self._reconnect_attempts = 0
        self._current_delay = self.config.initial_delay
    
    def get_state(self) -> ConnectionState:
        return self._state
    
    def _calculate_backoff(self) -> float:
        """Calculate next backoff delay with jitter."""
        delay = min(self._current_delay, self.config.max_delay)
        jitter = delay * self.config.jitter * (2 * random.random() - 1)
        return delay + jitter
    
    def _reset_backoff(self):
        """Reset backoff after successful connection."""
        self._current_delay = self.config.initial_delay
        self._reconnect_attempts = 0
    
    def _increase_backoff(self):
        """Increase backoff delay for next attempt."""
        self._current_delay = min(
            self._current_delay * self.config.multiplier,
            self.config.max_delay
        )
        self._reconnect_attempts += 1
    
    def connect(self) -> bool:
        """Attempt to connect to cloud."""
        self._state = ConnectionState.RECONNECTING
        # Simulated connection (in real impl, this would be actual connection)
        success = True  # Placeholder
        
        if success:
            self._state = ConnectionState.CONNECTED
            self._last_heartbeat = time.time()
            self._reset_backoff()
            return True
        else:
            self._increase_backoff()
            self._state = ConnectionState.DISCONNECTED
            return False
    
    def should_reconnect(self) -> bool:
        """Check if reconnection should be attempted."""
        if self._state == ConnectionState.CONNECTED:
            return False
        return self._reconnect_attempts < self.config.max_attempts
    
    def get_next_retry_delay(self) -> float:
        """Get delay before next reconnection attempt."""
        return self._calculate_backoff()
    
    def heartbeat(self) -> bool:
        """Send heartbeat and check connection health."""
        if self._state != ConnectionState.CONNECTED:
            return False
        
        self._last_heartbeat = time.time()
        return True
    
    def check_health(self, timeout: float = 30.0) -> bool:
        """Check if connection is healthy based on last heartbeat."""
        if not self._last_heartbeat:
            return False
        
        elapsed = time.time() - self._last_heartbeat
        if elapsed > timeout:
            self._state = ConnectionState.DEGRADED
            return False
        return True
    
    def get_status(self) -> dict:
        """Get connection status summary."""
        return {
            "state": self._state.value,
            "reconnect_attempts": self._reconnect_attempts,
            "next_retry_delay": self._calculate_backoff(),
            "last_heartbeat": self._last_heartbeat
        }
