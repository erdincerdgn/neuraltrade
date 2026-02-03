"""
Camber Cloud Connector
Author: Erdinc Erdogan
Purpose: Interface to Camber Cloud for distributed chaos computation with MPI job submission, monitoring, and result retrieval.
References:
- Camber Cloud HPC Platform
- MPI (Message Passing Interface)
- Cloud Job Orchestration Patterns
Usage:
    connector = CamberCloudConnector()
    job_id = connector.submit_job(CloudJobConfig(command="python chaos.py", node_size="MEDIUM"))
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum

class NodeSize(Enum):
    """Available Camber Cloud node sizes."""
    XXSMALL = "XXSMALL"  # 4 CPU / 16GB RAM
    XSMALL = "XSMALL"    # 8 CPU / 32GB RAM / 1 GPU
    SMALL = "SMALL"      # 16 CPU / 64GB RAM
    MEDIUM = "MEDIUM"    # 48 CPU / 192GB RAM / 4 GPU
    LARGE = "LARGE"      # 64 CPU / 256GB RAM

class JobStatus(Enum):
    """Job status states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class CloudJobConfig:
    """Configuration for a Camber Cloud job."""
    command: str
    node_size: str = "XXSMALL"
    extra_env_vars: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class CloudJobResult:
    """Result from a Camber Cloud job."""
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str]
    output: Optional[str]
    error: Optional[str]

class CamberCloudConnector:
    """Main interface to Camber Cloud for NeuralTrade."""
    
    VERSION = "1.0.0"
    
    def __init__(self, default_node_size: str = "XXSMALL"):
        self.default_node_size = default_node_size
        self._active_jobs: Dict[str, Any] = {}
    
    def create_chaos_job(self, script_content: str, node_size: Optional[str] = None,
                         env_vars: Optional[Dict[str, str]] = None) -> CloudJobConfig:
        """Create a chaos computation job configuration."""
        size = node_size or self.default_node_size
        
        # Validate node size
        valid_sizes = [s.value for s in NodeSize]
        if size not in valid_sizes:
            raise ValueError(f"Invalid node_size: {size}. Must be one of {valid_sizes}")
        
        return CloudJobConfig(
            command=f"python chaos_compute.py",
            node_size=size,
            extra_env_vars=env_vars
        )
    
    def select_optimal_node(self, data_size: int, compute_type: str) -> str:
        """Select optimal node size based on workload."""
        if compute_type == "dfa_only":
            return "XXSMALL" if data_size < 10000 else "SMALL"
        elif compute_type == "lyapunov":
            return "SMALL" if data_size < 50000 else "LARGE"
        elif compute_type == "mse":
            return "SMALL" if data_size < 5000 else "LARGE"
        elif compute_type == "full_chaos":
            return "LARGE"
        elif compute_type == "gpu_accelerated":
            return "XSMALL"
        return self.default_node_size
    
    def generate_mpi_command(self, script_path: str, num_processes: int = 4) -> str:
        """Generate MPI command for distributed computation."""
        return f"mpirun -np {num_processes} python {script_path}"
    
    def estimate_cost(self, node_size: str, duration_hours: float) -> Dict[str, float]:
        """Estimate job cost based on node size and duration."""
        # Approximate cost rates (example values)
        rates = {
            "XXSMALL": 0.10,
            "XSMALL": 0.25,
            "SMALL": 0.50,
            "MEDIUM": 2.00,
            "LARGE": 1.50
        }
        rate = rates.get(node_size, 0.10)
        return {
            "hourly_rate": rate,
            "estimated_cost": rate * duration_hours,
            "node_size": node_size
        }
