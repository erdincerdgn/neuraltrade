"""
Colocation Latency Simulator
Author: Erdinc Erdogan
Purpose: Simulates network latency between data centers and exchanges to model colocation advantages and latency arbitrage opportunities.
References:
- Exchange Colocation (NYSE Mahwah, NASDAQ Carteret)
- Network Latency Models
- Latency Arbitrage Theory
Usage:
    simulator = ColocationSimulator()
    latency = simulator.get_latency("NYSE_MAHWAH", "TRADER_NYC")
"""
import os
import time
import random
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from colorama import Fore, Style


@dataclass
class DataCenter:
    """Veri merkezi bilgisi."""
    name: str
    location: str
    exchange: str
    base_latency_ms: float  # Ortalama ping
    jitter_ms: float        # Değişkenlik
    packet_loss_pct: float  # Paket kaybı oranı


class ColocationSimulator:
    """
    Colocation (Sunucu Yakınlık) Simülatörü.
    
    Özellikler:
    - Farklı veri merkezleri arasındaki gecikme simülasyonu
    - Network jitter (değişkenlik)
    - Packet loss simülasyonu
    - Latency arbitrage hesaplama
    """
    
    # Bilinen veri merkezleri
    DATA_CENTERS = {
        # NYSE/NASDAQ - Mahwah, NJ
        "NYSE_MAHWAH": DataCenter(
            name="NYSE Mahwah",
            location="Mahwah, NJ, USA",
            exchange="NYSE",
            base_latency_ms=0.1,  # Colocated
            jitter_ms=0.02,
            packet_loss_pct=0.001
        ),
        "NASDAQ_CARTERET": DataCenter(
            name="NASDAQ Carteret",
            location="Carteret, NJ, USA",
            exchange="NASDAQ",
            base_latency_ms=0.1,
            jitter_ms=0.02,
            packet_loss_pct=0.001
        ),
        # Chicago - CME
        "CME_AURORA": DataCenter(
            name="CME Aurora",
            location="Aurora, IL, USA",
            exchange="CME",
            base_latency_ms=4.0,  # NY'den Chicago'ya fiber
            jitter_ms=0.5,
            packet_loss_pct=0.01
        ),
        # London
        "LSE_LONDON": DataCenter(
            name="LSE London",
            location="London, UK",
            exchange="LSE",
            base_latency_ms=35.0,  # Transatlantik
            jitter_ms=5.0,
            packet_loss_pct=0.05
        ),
        # Tokyo
        "TSE_TOKYO": DataCenter(
            name="TSE Tokyo",
            location="Tokyo, Japan",
            exchange="TSE",
            base_latency_ms=85.0,  # Transpasifik
            jitter_ms=10.0,
            packet_loss_pct=0.1
        ),
        # Frankfurt
        "EUREX_FRANKFURT": DataCenter(
            name="EUREX Frankfurt",
            location="Frankfurt, Germany",
            exchange="EUREX",
            base_latency_ms=40.0,
            jitter_ms=5.0,
            packet_loss_pct=0.05
        ),
        # Retail (Home)
        "RETAIL_HOME": DataCenter(
            name="Retail Home",
            location="Istanbul, Turkey",
            exchange="REMOTE",
            base_latency_ms=150.0,
            jitter_ms=30.0,
            packet_loss_pct=0.5
        ),
    }
    
    def __init__(self, primary_datacenter: str = "RETAIL_HOME"):
        """
        Args:
            primary_datacenter: Birincil veri merkezi
        """
        self.primary = self.DATA_CENTERS.get(primary_datacenter, self.DATA_CENTERS["RETAIL_HOME"])
        self.latency_history = []
        self.packet_losses = 0
        self.total_requests = 0
    
    def simulate_latency(self, target_dc: str = "NYSE_MAHWAH") -> Dict:
        """
        Hedef veri merkezine gecikmeyi simüle et.
        
        Args:
            target_dc: Hedef veri merkezi
        """
        target = self.DATA_CENTERS.get(target_dc, self.DATA_CENTERS["NYSE_MAHWAH"])
        
        # Toplam base latency
        if self.primary.name == target.name:
            base_latency = target.base_latency_ms
        else:
            base_latency = self.primary.base_latency_ms + target.base_latency_ms
        
        # Jitter ekle
        jitter = random.gauss(0, (self.primary.jitter_ms + target.jitter_ms) / 2)
        
        # Network congestion (yoğun saatlerde)
        hour = datetime.now().hour
        if 9 <= hour <= 16:  # Piyasa saatleri
            congestion = random.uniform(0, base_latency * 0.3)
        else:
            congestion = 0
        
        total_latency = max(0.01, base_latency + jitter + congestion)
        
        # Packet loss simülasyonu
        combined_loss = self.primary.packet_loss_pct + target.packet_loss_pct
        packet_lost = random.random() < combined_loss
        
        self.total_requests += 1
        if packet_lost:
            self.packet_losses += 1
            total_latency *= 2  # Retransmit
        
        self.latency_history.append({
            "timestamp": datetime.now(),
            "latency_ms": total_latency,
            "packet_lost": packet_lost
        })
        
        return {
            "from": self.primary.name,
            "to": target.name,
            "base_latency_ms": base_latency,
            "jitter_ms": jitter,
            "congestion_ms": congestion,
            "total_latency_ms": total_latency,
            "round_trip_ms": total_latency * 2,
            "packet_lost": packet_lost
        }
    
    def execute_with_latency(self, func: Callable, target_dc: str = "NYSE_MAHWAH") -> Dict:
        """
        Fonksiyonu gecikme simülasyonu ile çalıştır.
        """
        # Pre-execution latency
        latency_info = self.simulate_latency(target_dc)
        time.sleep(latency_info["total_latency_ms"] / 1000)
        
        # Execute
        start = time.time()
        result = func()
        execution_time = (time.time() - start) * 1000
        
        # Post-execution latency (response)
        time.sleep(latency_info["total_latency_ms"] / 1000)
        
        return {
            "result": result,
            "latency": latency_info,
            "execution_ms": execution_time,
            "total_time_ms": latency_info["round_trip_ms"] + execution_time
        }
    
    def calculate_latency_arbitrage(self, price_a: float, price_b: float,
                                    dc_a: str, dc_b: str) -> Dict:
        """
        Latency arbitrage fırsatını hesapla.
        
        İki borsa arasındaki fiyat farkı latency'den önce
        kapanırsa arbitraj kaybedilir.
        """
        lat_a = self.simulate_latency(dc_a)
        lat_b = self.simulate_latency(dc_b)
        
        price_diff = abs(price_a - price_b)
        price_diff_pct = (price_diff / min(price_a, price_b)) * 100
        
        # Toplam round-trip time
        total_latency = lat_a["round_trip_ms"] + lat_b["round_trip_ms"]
        
        # Fiyat farkının kapanma süresi tahmini (volatilite bazlı)
        estimated_close_time_ms = 50 + random.uniform(0, 100)
        
        # Arbitraj mümkün mü?
        is_profitable = total_latency < estimated_close_time_ms and price_diff_pct > 0.01
        
        return {
            "price_a": price_a,
            "price_b": price_b,
            "price_diff": price_diff,
            "price_diff_pct": price_diff_pct,
            "latency_a_ms": lat_a["round_trip_ms"],
            "latency_b_ms": lat_b["round_trip_ms"],
            "total_latency_ms": total_latency,
            "estimated_close_time_ms": estimated_close_time_ms,
            "is_profitable": is_profitable,
            "potential_profit_pct": price_diff_pct if is_profitable else 0
        }
    
    def get_statistics(self) -> Dict:
        """Latency istatistikleri."""
        if not self.latency_history:
            return {"error": "No data"}
        
        latencies = [h["latency_ms"] for h in self.latency_history]
        
        return {
            "total_requests": self.total_requests,
            "packet_losses": self.packet_losses,
            "packet_loss_rate": (self.packet_losses / self.total_requests * 100) if self.total_requests > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
            "primary_dc": self.primary.name
        }
    
    def generate_latency_report(self) -> str:
        """Latency raporu oluştur."""
        stats = self.get_statistics()
        
        report = f"""
<colocation_simulator>
⚡ COLOCATION LATENCY RAPORU
════════════════════════════════════════

📍 Birincil Konum: {self.primary.name}
   ({self.primary.location})

📊 İSTATİSTİKLER:
  • Toplam İstek: {stats.get('total_requests', 0)}
  • Paket Kaybı: {stats.get('packet_loss_rate', 0):.2f}%
  • Ort. Latency: {stats.get('avg_latency_ms', 0):.2f}ms
  • Min Latency: {stats.get('min_latency_ms', 0):.2f}ms
  • Max Latency: {stats.get('max_latency_ms', 0):.2f}ms
  • P95 Latency: {stats.get('p95_latency_ms', 0):.2f}ms

🏢 VERİ MERKEZLERİ:
"""
        for dc_name, dc in self.DATA_CENTERS.items():
            emoji = "🟢" if dc.base_latency_ms < 10 else "🟡" if dc.base_latency_ms < 50 else "🔴"
            report += f"  {emoji} {dc.name}: {dc.base_latency_ms}ms ({dc.location})\n"
        
        report += "</colocation_simulator>\n"
        return report


class HighFrequencySimulator:
    """
    Yüksek Frekanslı İşlem (HFT) Simülatörü.
    
    Mikrosaniye seviyesinde işlem simülasyonu.
    """
    
    def __init__(self, colocation: ColocationSimulator):
        self.colocation = colocation
        self.orders_per_second = 0
        self.order_history = []
    
    def simulate_order_burst(self, num_orders: int = 100, target_dc: str = "NYSE_MAHWAH") -> Dict:
        """Emir patlaması simüle et."""
        start = time.time()
        latencies = []
        
        for i in range(num_orders):
            lat = self.colocation.simulate_latency(target_dc)
            latencies.append(lat["total_latency_ms"])
        
        duration = time.time() - start
        
        return {
            "num_orders": num_orders,
            "duration_seconds": duration,
            "orders_per_second": num_orders / duration if duration > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies)
        }
    
    def estimate_hft_advantage(self, retail_dc: str = "RETAIL_HOME",
                               hft_dc: str = "NYSE_MAHWAH") -> Dict:
        """HFT avantajını hesapla."""
        retail_lat = self.colocation.simulate_latency(retail_dc)
        
        # HFT için ayrı simülatör
        hft_sim = ColocationSimulator(hft_dc)
        hft_lat = hft_sim.simulate_latency(hft_dc)
        
        advantage_ms = retail_lat["total_latency_ms"] - hft_lat["total_latency_ms"]
        advantage_factor = retail_lat["total_latency_ms"] / hft_lat["total_latency_ms"] if hft_lat["total_latency_ms"] > 0 else 0
        
        return {
            "retail_latency_ms": retail_lat["total_latency_ms"],
            "hft_latency_ms": hft_lat["total_latency_ms"],
            "advantage_ms": advantage_ms,
            "advantage_factor": f"{advantage_factor:.0f}x",
            "warning": f"HFT botları sizden {advantage_ms:.0f}ms önce işlem yapabilir!"
        }
