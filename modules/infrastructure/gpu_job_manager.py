"""
GPU Job Manager for Cloud HPC
Author: Erdinc Erdogan
Purpose: Manages GPU-accelerated jobs for heavy MSE and Lyapunov calculations with automatic node sizing based on data volume.
References:
- CUDA GPU Computing
- Cloud GPU Node Selection
- GPU Speedup Estimation
Usage:
    manager = GPUJobManager()
    config = manager.create_gpu_job_config("mse_compute.py", compute_type="mse", data_size=100000)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class GPUNodeSize(Enum):
    XSMALL = "XSMALL"   # 8 CPU / 32GB RAM / 1 GPU
    MEDIUM = "MEDIUM"   # 48 CPU / 192GB RAM / 4 GPU

@dataclass
class GPUJobConfig:
    command: str
    node_size: str
    num_gpus: int
    cuda_version: str = "12.0"
    extra_env_vars: Optional[Dict[str, str]] = None

class GPUJobManager:
    GPU_NODES = {
        "XSMALL": {"cpus": 8, "ram_gb": 32, "gpus": 1},
        "MEDIUM": {"cpus": 48, "ram_gb": 192, "gpus": 4}
    }
    
    def __init__(self):
        self._active_gpu_jobs: Dict[str, Any] = {}
    
    def select_gpu_node(self, compute_type: str, data_size: int) -> str:
        if compute_type == "mse" and data_size > 50000:
            return "MEDIUM"
        elif compute_type == "lyapunov_gpu" and data_size > 100000:
            return "MEDIUM"
        return "XSMALL"
    
    def create_gpu_job_config(self, script_path: str, compute_type: str, data_size: int) -> GPUJobConfig:
        node_size = self.select_gpu_node(compute_type, data_size)
        num_gpus = self.GPU_NODES[node_size]["gpus"]
        return GPUJobConfig(
            command=f"python {script_path}",
            node_size=node_size,
            num_gpus=num_gpus,
            extra_env_vars={"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(num_gpus))}
        )
    
    def estimate_gpu_speedup(self, data_size: int, num_gpus: int) -> Dict[str, float]:
        base_speedup = 10.0
        multi_gpu_efficiency = 0.8
        speedup = base_speedup * (1 + (num_gpus - 1) * multi_gpu_efficiency) if num_gpus > 1 else base_speedup
        cpu_time = data_size * 0.01
        return {"cpu_time_sec": cpu_time, "gpu_time_sec": cpu_time / speedup, "speedup": speedup, "num_gpus": num_gpus}
