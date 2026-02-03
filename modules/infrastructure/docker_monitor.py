"""
Docker Container Log Monitor
Author: Erdinc Erdogan
Purpose: Provides real-time container log monitoring with pattern detection, error classification, and Prometheus metrics integration.
References:
- Docker Container API
- Prometheus Monitoring
- Log Pattern Detection
Usage:
    monitor = DockerLogMonitor(container_names=["neuraltrade-bot"])
    monitor.start_monitoring()
"""

import docker
import re
import time
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram

# ============================================
# PROMETHEUS METRICS
# ============================================
docker_errors_total = Counter(
    'docker_errors_total',
    'Total Docker container errors detected',
    ['container', 'pattern_type']
)

docker_warnings_total = Counter(
    'docker_warnings_total',
    'Total Docker container warnings detected',
    ['container', 'pattern_type']
)

container_health = Gauge(
    'container_health_status',
    'Container health status (1=healthy, 0=unhealthy)',
    ['container']
)

log_processing_duration = Histogram(
    'log_processing_seconds',
    'Time spent processing logs',
    ['container']
)

# ============================================
# ERROR/WARNING PATTERNS
# ============================================
CRITICAL_PATTERNS = {
    'error': re.compile(r'(Error|ERROR|Exception|EXCEPTION|CRITICAL|FATAL)', re.IGNORECASE),
    'connection': re.compile(r'Connection (refused|reset|timeout|failed)', re.IGNORECASE),
    'memory': re.compile(r'(Out of memory|OOM|Memory exhausted)', re.IGNORECASE),
    'segfault': re.compile(r'Segmentation fault|SIGSEGV', re.IGNORECASE),
    'failed': re.compile(r'Failed to|Cannot|Unable to', re.IGNORECASE),
    'timeout': re.compile(r'Timeout|timed out', re.IGNORECASE),
}

WARNING_PATTERNS = {
    'warning': re.compile(r'Warning|WARN|deprecated', re.IGNORECASE),
    'slow': re.compile(r'Slow (query|request|response)', re.IGNORECASE),
    'retry': re.compile(r'Retry|Retrying|attempt', re.IGNORECASE),
    'unhealthy': re.compile(r'unhealthy|degraded', re.IGNORECASE),
}

# Patterns to IGNORE (false positives)
IGNORE_PATTERNS = [
    re.compile(r'ℹ️.*CrossEncoder.*optional'),  # Our info message
    re.compile(r'✅|🎉|📊|🔧'),  # Success emojis
    re.compile(r'pipeline.*tamamlandı', re.IGNORECASE),  # Success messages
]

# ============================================
# DOCKER LOG MONITOR
# ============================================
class DockerLogMonitor:
    """Monitor Docker container logs for errors and warnings"""
    
    def __init__(self, containers: List[str], verbose: bool = False):
        """
        Initialize Docker log monitor
        
        Args:
            containers: List of container names to monitor
            verbose: Print all log messages (not just errors/warnings)
        """
        self.containers = containers
        self.verbose = verbose
        self.client = None
        self.monitoring = False
        self.threads = []
        self.error_counts = defaultdict(int)
        self.warning_counts = defaultdict(int)
        
    def connect(self) -> bool:
        """Connect to Docker daemon"""
        try:
            self.client = docker.from_env()
            self.client.ping()
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Docker: {e}")
            return False
    
    def _should_ignore(self, log_line: str) -> bool:
        """Check if log line should be ignored"""
        for pattern in IGNORE_PATTERNS:
            if pattern.search(log_line):
                return True
        return False
    
    def _detect_patterns(self, log_line: str, container_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect error/warning patterns in log line
        
        Returns:
            (severity, pattern_type) or (None, None) if no match
        """
        if self._should_ignore(log_line):
            return None, None
        
        # Check critical patterns
        for pattern_name, pattern in CRITICAL_PATTERNS.items():
            if pattern.search(log_line):
                docker_errors_total.labels(
                    container=container_name,
                    pattern_type=pattern_name
                ).inc()
                return 'ERROR', pattern_name
        
        # Check warning patterns
        for pattern_name, pattern in WARNING_PATTERNS.items():
            if pattern.search(log_line):
                docker_warnings_total.labels(
                    container=container_name,
                    pattern_type=pattern_name
                ).inc()
                return 'WARNING', pattern_name
        
        return None, None
    
    def _monitor_container(self, container_name: str):
        """Monitor a single container's logs"""
        try:
            container = self.client.containers.get(container_name)
            
            print(f"📡 Monitoring {container_name}...")
            
            # Stream logs
            for log in container.logs(stream=True, follow=True):
                if not self.monitoring:
                    break
                
                log_line = log.decode('utf-8', errors='ignore').strip()
                
                with log_processing_duration.labels(container=container_name).time():
                    severity, pattern_type = self._detect_patterns(log_line, container_name)
                
                if severity == 'ERROR':
                    self.error_counts[container_name] += 1
                    print(f"🚨 [{container_name}] ERROR ({pattern_type}): {log_line[:200]}")
                    container_health.labels(container=container_name).set(0)
                    
                elif severity == 'WARNING':
                    self.warning_counts[container_name] += 1
                    print(f"⚠️  [{container_name}] WARNING ({pattern_type}): {log_line[:200]}")
                    
                elif self.verbose:
                    print(f"   [{container_name}] {log_line[:150]}")
                
        except docker.errors.NotFound:
            print(f"❌ Container {container_name} not found")
        except Exception as e:
            print(f"❌ Error monitoring {container_name}: {e}")
    
    def start_monitoring(self):
        """Start monitoring all containers in background threads"""
        if not self.connect():
            return False
        
        self.monitoring = True
        
        for container_name in self.containers:
            thread = threading.Thread(
                target=self._monitor_container,
                args=(container_name,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        print(f"✅ Monitoring {len(self.containers)} containers")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring"""
        print("🛑 Stopping Docker log monitor...")
        self.monitoring = False
        
        for thread in self.threads:
            thread.join(timeout=2)
        
        self.print_summary()
    
    def print_summary(self):
        """Print error/warning summary"""
        print("\n" + "="*60)
        print("📊 DOCKER MONITOR SUMMARY")
        print("="*60)
        
        if self.error_counts:
            print("\n🚨 ERRORS:")
            for container, count in sorted(self.error_counts.items()):
                print(f"   {container}: {count} errors")
        
        if self.warning_counts:
            print("\n⚠️  WARNINGS:")
            for container, count in sorted(self.warning_counts.items()):
                print(f"   {container}: {count} warnings")
        
        if not self.error_counts and not self.warning_counts:
            print("\n✅ No errors or warnings detected!")

# ============================================
# CLI FOR TESTING
# ============================================
if __name__ == "__main__":
    import sys
    
    # Default containers to monitor
    containers = [
        "neuraltrade-bot",
        "neuraltrade-redis",
        "neuraltrade-qdrant",
        "neuraltrade-prometheus",
        "neuraltrade-grafana",
        "neuraltrade-alertmanager"
    ]
    
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    monitor = DockerLogMonitor(containers, verbose=verbose)
    
    if monitor.start_monitoring():
        try:
            print("\n⏸️  Press Ctrl+C to stop monitoring...\n")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    else:
        print("❌ Failed to start monitoring")
        sys.exit(1)
