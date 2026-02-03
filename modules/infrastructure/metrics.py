"""
Prometheus Metrics Wrapper
Author: Erdinc Erdogan
Purpose: Exposes trading system metrics (latency, trade counts, portfolio value, memory) via Prometheus endpoint for monitoring and alerting.
References:
- Prometheus Client Library
- Trading System Observability
- RED Metrics (Rate, Errors, Duration)
Usage:
    metrics = NeuralTradeMetrics(port=8000)
    metrics.record_trade(ticker="AAPL", side="BUY", quantity=100)
"""
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from colorama import Fore, Style

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class NeuralTradeMetrics:
    """
    Prometheus metrikleri için wrapper.
    
    Metrikler:
    - API gecikmesi (latency)
    - Trade sayıları
    - Portföy değeri
    - Bellek kullanımı
    - Model inference süresi
    """
    
    def __init__(self, port: int = 8000):
        """
        Args:
            port: Prometheus metrics endpoint portu
        """
        self.port = port
        self.enabled = PROMETHEUS_AVAILABLE
        
        if self.enabled:
            self._setup_metrics()
        else:
            print(f"{Fore.YELLOW}⚠️ prometheus_client yüklü değil, metrikler demo modda{Style.RESET_ALL}", flush=True)
            self._setup_demo_metrics()
    
    def _setup_metrics(self):
        """Prometheus metriklerini oluştur."""
        # Counters
        self.trades_total = Counter(
            'neuraltrade_trades_total',
            'Total number of trades',
            ['side', 'ticker', 'status']
        )
        
        self.api_requests_total = Counter(
            'neuraltrade_api_requests_total',
            'Total API requests',
            ['endpoint', 'status']
        )
        
        self.signals_generated = Counter(
            'neuraltrade_signals_total',
            'Total trading signals generated',
            ['signal_type', 'ticker']
        )
        
        # Gauges
        self.portfolio_value = Gauge(
            'neuraltrade_portfolio_value_usd',
            'Current portfolio value in USD'
        )
        
        self.cash_balance = Gauge(
            'neuraltrade_cash_balance_usd',
            'Current cash balance in USD'
        )
        
        self.positions_count = Gauge(
            'neuraltrade_positions_count',
            'Number of open positions'
        )
        
        self.daily_pnl = Gauge(
            'neuraltrade_daily_pnl_usd',
            'Daily profit and loss in USD'
        )
        
        self.drawdown_pct = Gauge(
            'neuraltrade_drawdown_percent',
            'Current drawdown percentage'
        )
        
        self.circuit_breaker_status = Gauge(
            'neuraltrade_circuit_breaker_status',
            'Circuit breaker status (0=active, 1=triggered)'
        )
        
        # Histograms
        self.api_latency = Histogram(
            'neuraltrade_api_latency_seconds',
            'API request latency',
            ['endpoint'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.model_inference_time = Histogram(
            'neuraltrade_model_inference_seconds',
            'Model inference time',
            ['model_type'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.trade_execution_time = Histogram(
            'neuraltrade_trade_execution_seconds',
            'Trade execution time',
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
        )
        
        # Summaries
        self.rag_retrieval_time = Summary(
            'neuraltrade_rag_retrieval_seconds',
            'RAG retrieval time'
        )
    
    def _setup_demo_metrics(self):
        """Demo metrikler (prometheus_client olmadan)."""
        self._demo_data = {
            "trades_total": 0,
            "api_requests": 0,
            "portfolio_value": 100000,
            "daily_pnl": 0,
            "latencies": []
        }
    
    def start_server(self):
        """Prometheus HTTP server'ı başlat."""
        if self.enabled:
            start_http_server(self.port)
            print(f"{Fore.GREEN}📊 Prometheus metrics: http://localhost:{self.port}/metrics{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.YELLOW}📊 Demo metrics mode (port {self.port}){Style.RESET_ALL}", flush=True)
    
    # ========== Trade Metrics ==========
    def record_trade(self, side: str, ticker: str, status: str = "filled"):
        """Trade kaydı."""
        if self.enabled:
            self.trades_total.labels(side=side, ticker=ticker, status=status).inc()
        else:
            self._demo_data["trades_total"] += 1
    
    def record_signal(self, signal_type: str, ticker: str):
        """Sinyal kaydı."""
        if self.enabled:
            self.signals_generated.labels(signal_type=signal_type, ticker=ticker).inc()
    
    # ========== Portfolio Metrics ==========
    def update_portfolio(self, value: float, cash: float, positions: int, pnl: float):
        """Portföy metriklerini güncelle."""
        if self.enabled:
            self.portfolio_value.set(value)
            self.cash_balance.set(cash)
            self.positions_count.set(positions)
            self.daily_pnl.set(pnl)
        else:
            self._demo_data["portfolio_value"] = value
            self._demo_data["daily_pnl"] = pnl
    
    def update_drawdown(self, drawdown: float):
        """Drawdown güncelle."""
        if self.enabled:
            self.drawdown_pct.set(drawdown)
    
    def update_circuit_breaker(self, is_triggered: bool):
        """Circuit breaker durumunu güncelle."""
        if self.enabled:
            self.circuit_breaker_status.set(1 if is_triggered else 0)
    
    # ========== Latency Metrics ==========
    def record_api_latency(self, endpoint: str, latency: float):
        """API gecikmesi kaydet."""
        if self.enabled:
            self.api_latency.labels(endpoint=endpoint).observe(latency)
            self.api_requests_total.labels(endpoint=endpoint, status="success").inc()
        else:
            self._demo_data["api_requests"] += 1
            self._demo_data["latencies"].append(latency)
    
    def record_model_inference(self, model_type: str, duration: float):
        """Model inference süresi kaydet."""
        if self.enabled:
            self.model_inference_time.labels(model_type=model_type).observe(duration)
    
    def record_rag_retrieval(self, duration: float):
        """RAG retrieval süresi kaydet."""
        if self.enabled:
            self.rag_retrieval_time.observe(duration)
    
    # ========== Context Manager ==========
    def measure_time(self, metric_name: str):
        """Zaman ölçümü için context manager."""
        return TimeMeasurement(self, metric_name)
    
    def get_demo_summary(self) -> Dict:
        """Demo özeti (prometheus olmadan)."""
        return {
            "trades": self._demo_data.get("trades_total", 0),
            "api_requests": self._demo_data.get("api_requests", 0),
            "portfolio_value": self._demo_data.get("portfolio_value", 0),
            "avg_latency_ms": sum(self._demo_data.get("latencies", [])) / max(len(self._demo_data.get("latencies", [])), 1) * 1000
        }


class TimeMeasurement:
    """Zaman ölçümü context manager."""
    
    def __init__(self, metrics: NeuralTradeMetrics, metric_name: str):
        self.metrics = metrics
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        if "api" in self.metric_name.lower():
            self.metrics.record_api_latency(self.metric_name, duration)
        elif "model" in self.metric_name.lower():
            self.metrics.record_model_inference(self.metric_name, duration)
        elif "rag" in self.metric_name.lower():
            self.metrics.record_rag_retrieval(duration)


# Grafana Dashboard JSON
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "NeuralTrade Monitor",
        "tags": ["trading", "ai"],
        "panels": [
            {
                "title": "Portfolio Value",
                "type": "stat",
                "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
                "targets": [{"expr": "neuraltrade_portfolio_value_usd"}]
            },
            {
                "title": "Daily P&L",
                "type": "stat",
                "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                "targets": [{"expr": "neuraltrade_daily_pnl_usd"}]
            },
            {
                "title": "Trades per Minute",
                "type": "graph",
                "gridPos": {"x": 0, "y": 4, "w": 12, "h": 6},
                "targets": [{"expr": "rate(neuraltrade_trades_total[1m])"}]
            },
            {
                "title": "API Latency (p95)",
                "type": "graph",
                "gridPos": {"x": 0, "y": 10, "w": 12, "h": 6},
                "targets": [{"expr": "histogram_quantile(0.95, neuraltrade_api_latency_seconds_bucket)"}]
            },
            {
                "title": "Circuit Breaker Status",
                "type": "stat",
                "gridPos": {"x": 12, "y": 0, "w": 4, "h": 4},
                "targets": [{"expr": "neuraltrade_circuit_breaker_status"}]
            },
            {
                "title": "Model Inference Time",
                "type": "heatmap",
                "gridPos": {"x": 12, "y": 4, "w": 12, "h": 6},
                "targets": [{"expr": "neuraltrade_model_inference_seconds_bucket"}]
            }
        ]
    }
}


def generate_prometheus_config() -> str:
    """Prometheus config dosyası oluştur."""
    return """
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'neuraltrade'
    static_configs:
      - targets: ['neuraltrade:8000']
    metrics_path: /metrics
"""


def generate_grafana_datasource() -> str:
    """Grafana datasource config."""
    return """
# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
"""
