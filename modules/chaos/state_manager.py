"""
Chaos State Manager - Checkpoint and Recovery System
Author: Erdinc Erdogan
Purpose: Provides state persistence and recovery for chaos calculations, enabling incremental
computation without full history recalculation after cloud connection drops.
References:
- Checkpoint/Restart Patterns in Distributed Systems
- State Persistence for Real-Time Systems
- Incremental Computation Algorithms
Usage:
    manager = ChaosStateManager(checkpoint_dir="./checkpoints")
    manager.save_checkpoint(chaos_state)
    recovered_state = manager.load_latest_checkpoint()
"""

import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

@dataclass
class ChaosCheckpoint:
    """Checkpoint for chaos state recovery."""
    checkpoint_id: str
    timestamp: str
    last_hurst: float
    last_lyapunov: float
    last_complexity: float
    last_regime: str
    data_window_start: int
    data_window_end: int
    cumulative_profile: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChaosCheckpoint":
        return cls(**data)

class ChaosStateManager:
    """Manages chaos state persistence and recovery."""
    
    def __init__(self, checkpoint_dir: str = "./chaos_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self._current_state: Optional[ChaosCheckpoint] = None
        self._data_buffer: List[float] = []
        self._max_buffer_size = 10000
    
    def _ensure_dir(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, hurst: float, lyapunov: float, complexity: float,
                        regime: str, data_start: int, data_end: int,
                        profile: np.ndarray) -> str:
        """Save current state to checkpoint."""
        self._ensure_dir()
        
        checkpoint_id = f"chaos_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint = ChaosCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            last_hurst=float(hurst),
            last_lyapunov=float(lyapunov),
            last_complexity=float(complexity),
            last_regime=str(regime),
            data_window_start=int(data_start),
            data_window_end=int(data_end),
            cumulative_profile=profile.tolist() if isinstance(profile, np.ndarray) else list(profile)
        )
        
        filepath = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(filepath, "w") as f:
            json.dump(checkpoint.to_dict(), f)
        
        self._current_state = checkpoint
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ChaosCheckpoint]:
        """Load checkpoint from disk."""
        filepath = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self._current_state = ChaosCheckpoint.from_dict(data)
        return self._current_state
    
    def get_latest_checkpoint(self) -> Optional[ChaosCheckpoint]:
        """Get most recent checkpoint."""
        self._ensure_dir()
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".json")]
        if not files:
            return None
        
        latest = sorted(files)[-1]
        return self.load_checkpoint(latest.replace(".json", ""))
    
    def can_resume(self, new_data_start: int) -> bool:
        """Check if we can resume from checkpoint."""
        if self._current_state is None:
            return False
        return new_data_start <= self._current_state.data_window_end
    
    def get_resume_offset(self) -> int:
        """Get offset to resume calculations."""
        if self._current_state is None:
            return 0
        return self._current_state.data_window_end
    
    def clear_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only recent ones."""
        self._ensure_dir()
        files = sorted([f for f in os.listdir(self.checkpoint_dir) if f.endswith(".json")])
        for f in files[:-keep_last]:
            os.remove(os.path.join(self.checkpoint_dir, f))
