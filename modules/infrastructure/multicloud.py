"""
Multi-Cloud Geo-Arbitrage Router
Author: Erdinc Erdogan
Purpose: Routes requests across AWS, GCP, Azure data centers based on geographic distance and latency for optimal execution speed.
References:
- Geo-Arbitrage (Light Speed Trading)
- Multi-Cloud Failover Patterns
- Haversine Distance Formula
Usage:
    router = GeoArbitrageRouter()
    best_dc = router.select_optimal_datacenter(target_exchange="NYSE", user_location=(40.7, -74.0))
"""
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from colorama import Fore, Style


class CloudProvider(Enum):
    """Bulut sağlayıcıları."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ALIBABA = "alibaba"


class DataCenter(Enum):
    """Veri merkezleri."""
    # AWS
    AWS_TOKYO = ("aws", "ap-northeast-1", 35.6762, 139.6503)
    AWS_SINGAPORE = ("aws", "ap-southeast-1", 1.3521, 103.8198)
    AWS_VIRGINIA = ("aws", "us-east-1", 37.4316, -78.6569)
    AWS_FRANKFURT = ("aws", "eu-central-1", 50.1109, 8.6821)
    
    # GCP
    GCP_TOKYO = ("gcp", "asia-northeast1", 35.6762, 139.6503)
    GCP_SINGAPORE = ("gcp", "asia-southeast1", 1.3521, 103.8198)
    
    # Azure
    AZURE_TOKYO = ("azure", "japaneast", 35.6762, 139.6503)
    
    def __init__(self, provider: str, region: str, lat: float, lon: float):
        self.provider = provider
        self.region = region
        self.lat = lat
        self.lon = lon


class GeoArbitrageRouter:
    """
    Geo-Arbitrage Router.
    
    Işık hızı sınırını lehine kullanarak
    en düşük latency'li veri merkezini seçer.
    """
    
    # Exchange lokasyonları
    EXCHANGE_LOCATIONS = {
        "BINANCE": (35.6762, 139.6503),      # Tokyo
        "NYSE": (40.7128, -74.0060),          # New York
        "NASDAQ": (40.7128, -74.0060),        # New York
        "LSE": (51.5074, -0.1278),            # London
        "CME": (41.8781, -87.6298),           # Chicago
        "COINBASE": (37.7749, -122.4194),     # San Francisco
        "KRAKEN": (37.7749, -122.4194),       # San Francisco
    }
    
    # Işık hızı (fiber optik = ~200,000 km/s)
    SPEED_OF_LIGHT_KM_MS = 200  # km per ms
    
    def __init__(self):
        self.datacenter_health = {dc: True for dc in DataCenter}
        self.latency_cache = {}
        self.routing_history = []
    
    def calculate_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formülü ile mesafe hesapla."""
        import math
        
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def estimate_latency_ms(self, datacenter: DataCenter, exchange: str) -> float:
        """
        Latency tahmini.
        
        Işık hızı + network overhead.
        """
        if exchange not in self.EXCHANGE_LOCATIONS:
            return 100  # Default
        
        exchange_loc = self.EXCHANGE_LOCATIONS[exchange]
        distance = self.calculate_distance_km(
            datacenter.lat, datacenter.lon,
            exchange_loc[0], exchange_loc[1]
        )
        
        # Işık hızı latency (tek yön, x2 for round trip)
        light_latency = distance / self.SPEED_OF_LIGHT_KM_MS * 2
        
        # Network overhead (routing, processing)
        overhead = random.uniform(1, 5)
        
        return light_latency + overhead
    
    def select_best_datacenter(self, exchange: str) -> Tuple[DataCenter, float]:
        """En iyi veri merkezini seç."""
        best_dc = None
        best_latency = float('inf')
        
        for dc in DataCenter:
            if not self.datacenter_health[dc]:
                continue
            
            latency = self.estimate_latency_ms(dc, exchange)
            
            if latency < best_latency:
                best_latency = latency
                best_dc = dc
        
        return best_dc, best_latency
    
    def route_order(self, exchange: str, order: Dict) -> Dict:
        """Emri optimal veri merkezine yönlendir."""
        best_dc, latency = self.select_best_datacenter(exchange)
        
        routing = {
            "exchange": exchange,
            "datacenter": best_dc.name if best_dc else "FALLBACK",
            "provider": best_dc.provider if best_dc else "aws",
            "region": best_dc.region if best_dc else "us-east-1",
            "estimated_latency_ms": latency,
            "order": order,
            "timestamp": datetime.now().isoformat()
        }
        
        self.routing_history.append(routing)
        
        print(f"{Fore.CYAN}🌐 Route: {exchange} → {best_dc.name if best_dc else 'FALLBACK'} ({latency:.1f}ms){Style.RESET_ALL}", flush=True)
        
        return routing


class FailoverManager:
    """
    Multi-Cloud Failover Manager.
    
    Bir bulut çökerse trafiği diğerine yönlendirir.
    """
    
    def __init__(self):
        self.primary = CloudProvider.AWS
        self.secondary = CloudProvider.GCP
        self.tertiary = CloudProvider.AZURE
        
        self.health_status = {
            CloudProvider.AWS: {"healthy": True, "last_check": None},
            CloudProvider.GCP: {"healthy": True, "last_check": None},
            CloudProvider.AZURE: {"healthy": True, "last_check": None},
            CloudProvider.ALIBABA: {"healthy": True, "last_check": None},
        }
        
        self.failover_history = []
    
    def health_check(self, provider: CloudProvider) -> bool:
        """Sağlık kontrolü."""
        # Simülasyon
        is_healthy = random.random() > 0.05  # %95 uptime
        
        self.health_status[provider]["healthy"] = is_healthy
        self.health_status[provider]["last_check"] = datetime.now()
        
        return is_healthy
    
    def check_all_providers(self) -> Dict:
        """Tüm sağlayıcıları kontrol et."""
        results = {}
        for provider in CloudProvider:
            results[provider.value] = self.health_check(provider)
        
        return results
    
    def get_active_provider(self) -> CloudProvider:
        """Aktif sağlayıcıyı getir."""
        if self.health_status[self.primary]["healthy"]:
            return self.primary
        
        # Failover
        if self.health_status[self.secondary]["healthy"]:
            self._log_failover(self.primary, self.secondary)
            return self.secondary
        
        if self.health_status[self.tertiary]["healthy"]:
            self._log_failover(self.primary, self.tertiary)
            return self.tertiary
        
        # Tümü çökmüş - yine de primary dene
        return self.primary
    
    def _log_failover(self, from_provider: CloudProvider, to_provider: CloudProvider):
        """Failover kaydet."""
        event = {
            "from": from_provider.value,
            "to": to_provider.value,
            "timestamp": datetime.now().isoformat()
        }
        self.failover_history.append(event)
        
        print(f"{Fore.YELLOW}🔄 FAILOVER: {from_provider.value} → {to_provider.value}{Style.RESET_ALL}", flush=True)
    
    def generate_cloud_report(self) -> str:
        """Multi-cloud raporu."""
        report = f"""
<multi_cloud>
🌐 MULTI-CLOUD RAPORU
════════════════════════════════════════

📊 SAĞLAYICI DURUMU:
"""
        for provider, status in self.health_status.items():
            emoji = "🟢" if status["healthy"] else "🔴"
            report += f"  {emoji} {provider.value.upper()}: {'HEALTHY' if status['healthy'] else 'DOWN'}\n"
        
        report += f"""
🔄 AKTİF: {self.get_active_provider().value.upper()}

📈 FAILOVER GEÇMİŞİ: {len(self.failover_history)} olay
"""
        for event in self.failover_history[-3:]:
            report += f"  • {event['from']} → {event['to']}\n"
        
        report += "</multi_cloud>\n"
        return report
