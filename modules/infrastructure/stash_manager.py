"""
Camber Stash Data Persistence Manager
Author: Erdinc Erdogan
Purpose: Manages checkpoints, results, and model artifacts on Camber Stash cloud storage for team sharing and experiment reproducibility.
References:
- Camber Stash Cloud Storage
- ML Checkpoint Patterns
- Team Data Sharing
Usage:
    stash = CamberStashManager(base_path="neuraltrade")
    stash.save_checkpoint(model_state, name="epoch_100")
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

@dataclass
class StashPath:
    """Represents a path in Camber Stash."""
    stash_type: str  # "private", "public", "team"
    path: str
    team_name: Optional[str] = None
    
    def to_string(self) -> str:
        if self.stash_type == "team" and self.team_name:
            return f"stash://{self.team_name}/{self.path}"
        return f"stash://{self.stash_type}/{self.path}"

@dataclass
class StashFile:
    """Metadata for a file in Stash."""
    name: str
    path: str
    size: int
    modified: str
    is_directory: bool

class CamberStashManager:
    """Manager for Camber Stash operations."""
    
    NEURALTRADE_ROOT = "neuraltrade"
    CHECKPOINTS_DIR = "checkpoints"
    RESULTS_DIR = "results"
    MODELS_DIR = "models"
    
    def __init__(self, base_path: str = "neuraltrade"):
        self.base_path = base_path
        self._ensure_structure()
    
    def _ensure_structure(self):
        """Ensure NeuralTrade directory structure exists."""
        self.paths = {
            "root": f"~/{self.base_path}",
            "checkpoints": f"~/{self.base_path}/{self.CHECKPOINTS_DIR}",
            "results": f"~/{self.base_path}/{self.RESULTS_DIR}",
            "models": f"~/{self.base_path}/{self.MODELS_DIR}",
            "chaos": f"~/{self.base_path}/chaos",
        }
    
    def get_checkpoint_path(self, checkpoint_id: str) -> str:
        """Get full path for a checkpoint file."""
        return f"{self.paths['checkpoints']}/{checkpoint_id}.json"
    
    def get_result_path(self, job_id: str) -> str:
        """Get full path for a job result file."""
        return f"{self.paths['results']}/{job_id}_result.json"
    
    def serialize_for_stash(self, data: Any) -> str:
        """Serialize data for Stash storage."""
        def convert(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        return json.dumps(data, default=convert, indent=2)
    
    def create_chaos_checkpoint(self, metrics: Dict[str, Any], 
                                 data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chaos checkpoint for Stash storage."""
        return {
            "checkpoint_type": "chaos_metrics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "data_info": data_info,
            "version": "1.0.0"
        }
    
    def create_job_result(self, job_id: str, status: str, 
                          output: Dict[str, Any]) -> Dict[str, Any]:
        """Create a job result for Stash storage."""
        return {
            "job_id": job_id,
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output": output,
            "version": "1.0.0"
        }
    
    def generate_stash_commands(self, operation: str, **kwargs) -> List[str]:
        """Generate Camber Stash CLI commands."""
        commands = []
        
        if operation == "upload_checkpoint":
            local_path = kwargs.get("local_path")
            checkpoint_id = kwargs.get("checkpoint_id")
            remote_path = self.get_checkpoint_path(checkpoint_id)
            commands.append(f"# Upload checkpoint to Stash")
            commands.append(f"prv_stash.cp(src_path='{local_path}', dest_path='{remote_path}')")
        
        elif operation == "download_checkpoint":
            checkpoint_id = kwargs.get("checkpoint_id")
            local_path = kwargs.get("local_path")
            remote_path = self.get_checkpoint_path(checkpoint_id)
            commands.append(f"# Download checkpoint from Stash")
            commands.append(f"prv_stash.cp(src_path='{remote_path}', dest_path='{local_path}')")
        
        elif operation == "list_checkpoints":
            commands.append(f"# List all checkpoints")
            commands.append(f"prv_stash.ls('{self.paths['checkpoints']}')")
        
        elif operation == "share_to_team":
            team_name = kwargs.get("team_name")
            file_path = kwargs.get("file_path")
            commands.append(f"# Share to team stash")
            commands.append(f"prv_stash.cp(dest_stash=team_stash['{team_name}'], "
                          f"src_path='{file_path}', dest_path='{file_path}')")
        
        return commands
    
    def get_python_stash_code(self) -> str:
        """Generate Python code for Stash operations."""
        return f