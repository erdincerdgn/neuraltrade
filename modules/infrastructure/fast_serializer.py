"""
High-Performance Serializer with MessagePack/LZ4
Author: Erdinc Erdogan
Purpose: Provides 10x faster serialization than JSON using MessagePack with LZ4 compression for HFT chaos metrics transmission.
References:
- MessagePack Binary Serialization
- LZ4 Fast Compression
- NumPy Type Conversion
Usage:
    serializer = FastSerializer(use_compression=True)
    result = serializer.serialize(chaos_metrics)
"""

import json
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

def numpy_to_python(obj: Any) -> Any:
    """Convert NumPy types to native Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    return obj

@dataclass
class SerializationResult:
    """Result of serialization operation."""
    data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    format: str

class FastSerializer:
    """High-performance serializer for cloud transmission."""
    
    COMPRESSION_THRESHOLD = 1024  # Compress if > 1KB
    
    def __init__(self, use_compression: bool = True, compression_level: int = 6):
        self.use_compression = use_compression
        self.compression_level = compression_level
        self._msgpack_available = False
        self._lz4_available = False
        
        # Try to import optional high-performance libraries
        try:
            import msgpack
            self._msgpack_available = True
        except ImportError:
            pass
        
        try:
            import lz4.frame
            self._lz4_available = True
        except ImportError:
            pass
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data using best available method."""
        if self._lz4_available:
            import lz4.frame
            return lz4.frame.compress(data)
        return zlib.compress(data, self.compression_level)
    
    def _decompress(self, data: bytes, use_lz4: bool = False) -> bytes:
        """Decompress data."""
        if use_lz4 and self._lz4_available:
            import lz4.frame
            return lz4.frame.decompress(data)
        return zlib.decompress(data)
    
    def serialize(self, obj: Any) -> SerializationResult:
        """Serialize object for cloud transmission."""
        # Convert numpy types
        clean_obj = numpy_to_python(obj)
        
        # Serialize
        if self._msgpack_available:
            import msgpack
            data = msgpack.packb(clean_obj, use_bin_type=True)
            fmt = "msgpack"
        else:
            data = json.dumps(clean_obj).encode("utf-8")
            fmt = "json"
        
        original_size = len(data)
        
        # Compress if beneficial
        if self.use_compression and original_size > self.COMPRESSION_THRESHOLD:
            compressed = self._compress(data)
            if len(compressed) < original_size:
                return SerializationResult(
                    data=compressed,
                    original_size=original_size,
                    compressed_size=len(compressed),
                    compression_ratio=1 - len(compressed) / original_size,
                    format=f"{fmt}+{'lz4' if self._lz4_available else 'zlib'}"
                )
        
        return SerializationResult(
            data=data,
            original_size=original_size,
            compressed_size=original_size,
            compression_ratio=0.0,
            format=fmt
        )
    
    def deserialize(self, data: bytes, fmt: str = "json") -> Any:
        """Deserialize data from cloud."""
        # Decompress if needed
        if "+lz4" in fmt:
            data = self._decompress(data, use_lz4=True)
            fmt = fmt.replace("+lz4", "")
        elif "+zlib" in fmt:
            data = self._decompress(data, use_lz4=False)
            fmt = fmt.replace("+zlib", "")
        
        # Deserialize
        if fmt == "msgpack" and self._msgpack_available:
            import msgpack
            return msgpack.unpackb(data, raw=False)
        return json.loads(data.decode("utf-8"))
    
    def benchmark(self, obj: Any, iterations: int = 100) -> Dict[str, float]:
        """Benchmark serialization performance."""
        import time
        
        # JSON benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            json.dumps(numpy_to_python(obj)).encode("utf-8")
        json_time = time.perf_counter() - start
        
        # Fast serializer benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            self.serialize(obj)
        fast_time = time.perf_counter() - start
        
        return {
            "json_time": json_time,
            "fast_time": fast_time,
            "speedup": json_time / fast_time if fast_time > 0 else 0
        }
